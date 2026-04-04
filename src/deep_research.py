"""Deep (recursive) research searcher.

This module implements a bounded multi-round search strategy:
- Execute planned queries
- Extract content for top results
- Filter by credibility
- Generate follow-up queries (LLM) and repeat until depth/budget limits

It is designed to be used as a LangGraph node returning a dict of state updates.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.language_models import BaseChatModel

from src.config import config
from src.state import ResearchState, SearchResult, SearchQuery
from src.utils.cache import ToolCache
from src.utils.web_utils import WebSearchTool, ContentExtractor, DuckDuckGoProvider, TavilyProvider
from src.utils.credibility import CredibilityScorer
from src.callbacks import (
    emit_search_start,
    emit_extraction_start,
    emit_extraction_complete,
    emit_error,
)
from src.observability import runnable_config
from src.prompts.deep_research import (
    DEEP_QUERY_EXPANDER_SYSTEM_PROMPT,
    DEEP_QUERY_EXPANDER_USER_TEMPLATE,
)
from src.local_docs import get_default_index

logger = logging.getLogger(__name__)


@dataclass
class DeepResearchRunStats:
    depth: int = 0
    queries_executed: int = 0
    results_collected: int = 0


class DeepResearchSearcher:
    """Planned-query searcher with optional recursive query expansion."""

    def __init__(
        self,
        llm: BaseChatModel,
        credibility_scorer: CredibilityScorer | None = None,
    ):
        self.llm = llm
        self.credibility_scorer = credibility_scorer or CredibilityScorer()
        self.cache = ToolCache()
        self.searcher = WebSearchTool(
            max_results=config.max_search_results_per_query,
            providers=self._build_search_providers(),
        )
        self.extractor = ContentExtractor(timeout=10)
        self.query_sem = asyncio.Semaphore(max(1, config.max_parallel_searches))
        self.extract_sem = asyncio.Semaphore(max(1, config.max_parallel_searches))

    def _build_search_providers(self):
        if config.search_provider == "tavily":
            return [TavilyProvider(api_key=config.tavily_api_key or None, max_results=config.max_search_results_per_query)]
        return [DuckDuckGoProvider(config.max_search_results_per_query)]

    async def deep_search(self, state: ResearchState) -> dict[str, Any]:
        if not state.plan:
            await emit_error("No research plan available")
            return {"error": "No research plan available"}

        stats = DeepResearchRunStats()

        # Start from planned queries, but respect deep caps.
        planned_queries = list(state.plan.search_queries)
        planned_queries = planned_queries[: config.deep_max_total_queries]

        all_results: list[SearchResult] = []
        all_cred_scores: list[dict[str, Any]] = []

        seen_urls: set[str] = set()
        seen_queries: set[str] = {q.query.strip().lower() for q in planned_queries}

        # Seed with relevant local-doc chunks (if enabled). This ensures deep mode
        # can incorporate a local corpus even before web search begins.
        local_seed = self._local_seed_results(state)
        if local_seed:
            for r in local_seed:
                if r.url:
                    seen_urls.add(r.url.strip())

            scored_local = self.credibility_scorer.score_search_results(local_seed)
            filtered_local = [
                item for item in scored_local if item["credibility"]["score"] >= config.min_credibility_score
            ]
            all_results.extend([item["result"] for item in filtered_local])
            all_cred_scores.extend([item["credibility"] for item in filtered_local])

        for depth in range(1, config.deep_max_depth + 1):
            stats.depth = depth

            remaining_budget = config.deep_max_total_queries - stats.queries_executed
            if remaining_budget <= 0:
                break

            # Determine this round's queries (not yet executed)
            round_queries = [q for q in planned_queries if not q.completed]
            if not round_queries:
                break

            # Cap by remaining budget
            round_queries = round_queries[:remaining_budget]

            # Emit progress
            total_round = len(round_queries)
            for i, q in enumerate(round_queries, 1):
                await emit_search_start(q.query, i, total_round)

            # Execute searches in parallel (bounded)
            round_results = await self._execute_queries(round_queries)
            stats.queries_executed += len(round_queries)

            # mark completed
            for q in round_queries:
                q.completed = True

            # Dedupe URLs and optionally extract content for new URLs
            new_results = []
            for r in round_results:
                if not r.url:
                    continue
                url_key = r.url.strip()
                if url_key in seen_urls:
                    continue
                seen_urls.add(url_key)
                new_results.append(r)

            # Extract content (bounded) for at most N results per round
            # Keep extraction cost reasonable: extract for up to 2*max_results_per_query per round.
            max_extract = min(len(new_results), max(1, config.max_search_results_per_query * 2))
            to_extract = new_results[:max_extract]
            extracted = await self._extract_contents(to_extract)

            # Merge extracted content back
            extracted_count = sum(1 for r in extracted if r.content)
            extracted_chars = sum(len(r.content or "") for r in extracted)
            await emit_extraction_complete(extracted_count, extracted_chars)

            # Apply credibility scoring + filter
            scored = self.credibility_scorer.score_search_results(new_results)
            filtered_scored = [
                item for item in scored if item["credibility"]["score"] >= config.min_credibility_score
            ]
            round_kept = [item["result"] for item in filtered_scored]
            round_scores = [item["credibility"] for item in filtered_scored]

            all_results.extend(round_kept)
            all_cred_scores.extend(round_scores)

            # Hard cap total kept results
            if len(all_results) >= config.deep_max_total_results:
                all_results = all_results[: config.deep_max_total_results]
                all_cred_scores = all_cred_scores[: config.deep_max_total_results]
                break

            # If more depth remains, expand with follow-up queries
            if depth < config.deep_max_depth:
                followups = await self._expand_queries(
                    topic=state.research_topic,
                    objectives=state.plan.objectives,
                    previous_queries=[q.query for q in planned_queries],
                    snippets=[r.snippet for r in all_results[:15] if r.snippet],
                    breadth=config.deep_breadth,
                )

                for fq in followups:
                    qtxt = fq.query.strip()
                    if not qtxt:
                        continue
                    qkey = qtxt.lower()
                    if qkey in seen_queries:
                        continue
                    if len(planned_queries) >= config.deep_max_total_queries:
                        break
                    seen_queries.add(qkey)
                    planned_queries.append(fq)

        # Update plan in state (so writer can show what was searched)
        state.plan.search_queries = planned_queries

        if not all_results:
            return {"error": "Deep search produced no results"}

        return {
            "plan": state.plan,
            "search_results": all_results,
            "credibility_scores": all_cred_scores,
            "current_stage": "synthesizing",
            "iterations": state.iterations + 1,
        }

    def _local_seed_results(self, state: ResearchState) -> list[SearchResult]:
        index = get_default_index()
        if index is None:
            return []

        try:
            index.load_or_build()
        except Exception as e:
            logger.warning(f"Local-doc index unavailable: {e}")
            return []

        queries: list[str] = [state.research_topic]
        if state.plan and state.plan.objectives:
            queries.extend(state.plan.objectives[:3])

        max_k = max(1, min(8, int(config.max_search_results_per_query or 3)))

        results: list[SearchResult] = []
        seen: set[str] = set()
        for q in queries:
            try:
                matches = index.search(q, k=max_k)
            except Exception:
                matches = []
            for m in matches:
                if not isinstance(m, dict):
                    continue
                path = str(m.get("path") or "").strip()
                if not path:
                    continue

                url = f"local://{path}"
                dedupe_key = f"{url}::{m.get('snippet','')[:80]}"
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)

                title = str(m.get("title") or path)
                snippet = str(m.get("snippet") or "")
                content = m.get("content")
                content_text = str(content) if content is not None else None

                results.append(
                    SearchResult(
                        query=q,
                        title=title,
                        url=url,
                        snippet=snippet,
                        content=content_text,
                    )
                )

        return results

    async def _execute_queries(self, queries: list[SearchQuery]) -> list[SearchResult]:
        async def run_one(q: SearchQuery) -> list[SearchResult]:
            async with self.query_sem:
                max_results = config.max_search_results_per_query
                cached = self.cache.get_search(config.search_provider, q.query, int(max_results))
                if isinstance(cached, list) and cached:
                    return [
                        SearchResult(
                            query=item.get("query", q.query),
                            title=item.get("title", ""),
                            url=item.get("url", ""),
                            snippet=item.get("snippet", ""),
                            content=None,
                        )
                        for item in cached
                        if isinstance(item, dict)
                    ]

                results = await self.searcher.search_async(q.query)
                payload = [
                    {"query": r.query, "title": r.title, "url": r.url, "snippet": r.snippet}
                    for r in results
                ]
                if payload:
                    self.cache.set_search(config.search_provider, q.query, int(max_results), payload)

                return results

        tasks = [run_one(q) for q in queries]
        grouped = await asyncio.gather(*tasks)
        flat: list[SearchResult] = []
        for group in grouped:
            flat.extend(group)
        return flat

    async def _extract_contents(self, results: list[SearchResult]) -> list[SearchResult]:
        async def run_one(r: SearchResult, idx: int, total: int) -> SearchResult:
            await emit_extraction_start(r.url, idx, total)
            async with self.extract_sem:
                cached = self.cache.get_content(r.url)
                if cached is not None:
                    r.content = cached
                    return r

                try:
                    content = await self.extractor.extract_content_async(r.url)
                except Exception:
                    content = None

                self.cache.set_content(r.url, content)
                r.content = content
                return r

        total = len(results)
        tasks = [run_one(r, i + 1, total) for i, r in enumerate(results)]
        return list(await asyncio.gather(*tasks))

    async def _expand_queries(
        self,
        *,
        topic: str,
        objectives: list[str],
        previous_queries: list[str],
        snippets: list[str],
        breadth: int,
    ) -> list[SearchQuery]:
        system = DEEP_QUERY_EXPANDER_SYSTEM_PROMPT
        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", DEEP_QUERY_EXPANDER_USER_TEMPLATE),
        ])

        objectives_text = "\n".join(f"- {o}" for o in objectives[:8])
        prev_text = "\n".join(f"- {q}" for q in previous_queries[:30]) or "(none)"
        snippet_text = "\n".join(f"- {s[:200]}" for s in snippets[:12]) or "(none)"

        input_message = {
            "topic": topic,
            "objectives": objectives_text,
            "previous_queries": prev_text,
            "snippets": snippet_text,
            "breadth": breadth,
        }

        invoke_cfg = runnable_config(tags=["deep", "query-expansion"], metadata={"topic": topic})

        start = time.time()
        chain = prompt | self.llm | JsonOutputParser()
        try:
            result = await chain.ainvoke(input_message, config=invoke_cfg)
        except Exception as e:
            logger.warning(f"Deep query expansion failed: {e}")
            return []

        _ = time.time() - start

        queries_out: list[SearchQuery] = []
        try:
            items = result.get("queries", []) if isinstance(result, dict) else []
            for item in items:
                if not isinstance(item, dict):
                    continue
                q = (item.get("query") or "").strip()
                purpose = (item.get("purpose") or "").strip() or "Follow-up query"
                if q:
                    queries_out.append(SearchQuery(query=q, purpose=purpose))
        except Exception:
            return []

        return queries_out[:breadth]
