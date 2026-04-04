import json
from typing import Any

import httpx
import pytest

import api.server as server
from src.callbacks import emit_progress, ResearchStage


@pytest.mark.asyncio
async def test_api_research_http_and_sse_end_to_end(monkeypatch):
    # Keep the in-memory job store isolated.
    server._JOBS.clear()

    async def fake_run_research(*, topic: str, **kwargs: Any):
        # These progress events should be routed into the job SSE queue via CURRENT_JOB_ID.
        await emit_progress(ResearchStage.PLANNING, "planning", progress_pct=10)
        await emit_progress(ResearchStage.SEARCHING, "searching", progress_pct=50)
        await emit_progress(ResearchStage.COMPLETE, "complete", progress_pct=100)
        return {
            "final_report": f"Report for {topic}",
            "error": None,
            "search_results": [],
            "key_findings": ["ok"],
        }

    monkeypatch.setattr(server, "run_research", fake_run_research)

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        start = await client.post(
            "/research",
            json={"topic": "Test topic", "use_cache": False, "use_checkpoints": False},
        )
        assert start.status_code == 200
        payload = start.json()
        job_id = payload["job_id"]
        events_url = payload["events_url"]

        seen_hello = False
        seen_progress = False
        seen_completed_job = False
        progress_job_ids: set[str] = set()

        # Consume SSE events until completion.
        async with client.stream("GET", events_url) as resp:
            assert resp.status_code == 200

            cur_event = None
            cur_data = None

            async for line in resp.aiter_lines():
                line = (line or "").strip()
                if not line:
                    if cur_event and cur_data:
                        data = json.loads(cur_data)
                        if cur_event == "hello":
                            seen_hello = True
                            assert data.get("job_id") == job_id
                        if cur_event == "progress":
                            seen_progress = True
                            md = data.get("metadata") or {}
                            if isinstance(md, dict) and md.get("job_id"):
                                progress_job_ids.add(str(md.get("job_id")))
                        if cur_event == "job" and data.get("status") == "completed":
                            seen_completed_job = True
                            break
                    cur_event = None
                    cur_data = None
                    continue

                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    cur_event = line.split(":", 1)[1].strip()
                if line.startswith("data:"):
                    cur_data = line.split(":", 1)[1].strip()

        assert seen_hello
        assert seen_progress
        assert progress_job_ids == {job_id}
        assert seen_completed_job

        status = await client.get(f"/research/{job_id}")
        assert status.status_code == 200
        s = status.json()
        assert s["status"] == "completed"
        assert s["result"]["final_report"] == "Report for Test topic"
