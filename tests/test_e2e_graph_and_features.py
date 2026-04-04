from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config import config
from src.deep_research import DeepResearchSearcher
from src.state import ResearchPlan, SearchQuery, SearchResult


@pytest.mark.asyncio
async def test_graph_deep_routing_and_research_cache(monkeypatch, tmp_path: Path):
    import src.graph as graph

    calls = {"plan": 0, "search": 0, "deep": 0, "synth": 0, "write": 0}

    async def fake_plan(self, state):
        calls["plan"] += 1
        plan = ResearchPlan(
            topic=state.research_topic,
            objectives=["obj"],
            search_queries=[
                SearchQuery(query="q1", purpose="p"),
                SearchQuery(query="q2", purpose="p"),
            ],
            report_outline=["Intro"],
        )
        return {"plan": plan}

    async def fake_search(self, state):
        calls["search"] += 1
        raise AssertionError("search node should not run in deep mode")

    async def fake_deep_search(self, state):
        calls["deep"] += 1
        return {
            "search_results": [
                SearchResult(query="q1", title="t1", url="https://a", snippet="s1", content="c"),
                SearchResult(query="q2", title="t2", url="https://b", snippet="s2", content="c"),
            ],
            "current_stage": "synthesizing",
        }

    async def fake_synthesize(self, state):
        calls["synth"] += 1
        return {"key_findings": ["finding"]}

    async def fake_write(self, state):
        calls["write"] += 1
        return {"final_report": "# Report\n\nDone."}

    # Route into deep_search.
    monkeypatch.setattr(config, "deep_research", True)

    # Make caching write into tmp_path rather than the repo.
    real_cache_cls = graph.ResearchCache

    class TmpResearchCache(real_cache_cls):
        def __init__(self):
            super().__init__(cache_dir=tmp_path / "research_cache")

    monkeypatch.setattr(graph, "ResearchCache", TmpResearchCache)

    # Patch node methods to deterministic behavior.
    monkeypatch.setattr(graph.ResearchPlanner, "plan", fake_plan)
    monkeypatch.setattr(graph.ResearchSearcher, "search", fake_search)
    monkeypatch.setattr(graph.DeepResearchSearcher, "deep_search", fake_deep_search)
    monkeypatch.setattr(graph.ResearchSynthesizer, "synthesize", fake_synthesize)
    monkeypatch.setattr(graph.ReportWriter, "write_report", fake_write)

    out1 = await graph.run_research(
        topic="Topic",
        verbose=False,
        use_cache=True,
        use_checkpoints=False,
        thread_id="t1",
    )
    assert out1.get("final_report")
    assert calls["deep"] == 1

    # Second run should return from cache without re-executing nodes.
    out2 = await graph.run_research(
        topic="Topic",
        verbose=False,
        use_cache=True,
        use_checkpoints=False,
        thread_id="t2",
    )
    assert out2.get("final_report") == out1.get("final_report")
    assert calls["deep"] == 1
    assert calls["plan"] == 1


@pytest.mark.asyncio
async def test_deep_search_local_seed_without_web(monkeypatch, tmp_path: Path):
    from src.local_docs import LocalDocIndex
    import src.deep_research as deep

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("Local-only content about LangGraph and agents. " * 20, encoding="utf-8")

    index = LocalDocIndex(docs, cache_dir=tmp_path / "cache")
    index.build()

    monkeypatch.setattr(deep, "get_default_index", lambda: index)
    monkeypatch.setattr(config, "min_credibility_score", 0)
    monkeypatch.setattr(config, "deep_max_depth", 1)
    monkeypatch.setattr(config, "deep_max_total_queries", 2)

    # LLM won't be used because we cap depth to 1 and stub out web execution.
    searcher = DeepResearchSearcher(llm=object())

    async def no_web(_queries):
        return []

    async def no_extract(_results):
        return []

    monkeypatch.setattr(searcher, "_execute_queries", no_web)
    monkeypatch.setattr(searcher, "_extract_contents", no_extract)

    plan = ResearchPlan(
        topic="T",
        objectives=["LangGraph"],
        search_queries=[SearchQuery(query="q", purpose="p")],
        report_outline=["Intro"],
    )

    state = SimpleNamespace(research_topic="T", plan=plan, iterations=0, error=None)

    result = await searcher.deep_search(state)
    sr = result.get("search_results")
    assert sr
    assert any(isinstance(r, SearchResult) and str(r.url).startswith("local://") for r in sr)


def test_provider_and_observability_smoke(monkeypatch):
    from langchain_openai import ChatOpenAI
    from src.agents import get_llm
    from src.observability.langfuse import get_langfuse_callbacks

    # OpenRouter provider wiring
    monkeypatch.setattr(config, "model_provider", "openrouter")
    monkeypatch.setattr(config, "openrouter_api_key", "test")
    llm = get_llm(model_override="openai/gpt-4o-mini")
    assert isinstance(llm, ChatOpenAI)

    # LiteLLM provider wiring
    monkeypatch.setattr(config, "model_provider", "litellm")
    monkeypatch.setattr(config, "litellm_base_url", "http://localhost:4000")
    monkeypatch.setattr(config, "litellm_api_key", "test")
    llm2 = get_llm(model_override="gpt-4o-mini")
    assert isinstance(llm2, ChatOpenAI)

    # Langfuse integration should be a no-op unless enabled, and must never raise.
    monkeypatch.setattr(config, "langfuse_enabled", True)
    monkeypatch.setattr(config, "langfuse_public_key", "")
    monkeypatch.setattr(config, "langfuse_secret_key", "")
    callbacks = get_langfuse_callbacks()
    assert isinstance(callbacks, list)
