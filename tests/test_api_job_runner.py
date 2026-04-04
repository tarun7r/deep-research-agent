import asyncio

import pytest

import api.server as server
from src.callbacks import emit_progress, ResearchStage


@pytest.mark.asyncio
async def test_run_job_emits_events_and_completes(monkeypatch):
    async def fake_run_research(*, topic: str, **kwargs):
        await emit_progress(ResearchStage.PLANNING, "planning", progress_pct=5)
        await emit_progress(ResearchStage.COMPLETE, "done", progress_pct=100)
        return {"final_report": f"Report for {topic}"}

    monkeypatch.setattr(server, "run_research", fake_run_research)

    q: asyncio.Queue = asyncio.Queue(maxsize=50)
    job = server.JobRecord(job_id="job123", topic="Test topic", queue=q)

    await server._run_job(job, use_cache=False, use_checkpoints=False)

    assert job.status == "completed"
    assert job.result and "final_report" in job.result

    # Ensure at least one progress update made it into the queue.
    items = []
    while not q.empty():
        items.append(q.get_nowait())

    assert any(getattr(i, "stage", None) is not None for i in items)  # ProgressUpdate
    assert any(isinstance(i, dict) and i.get("type") == "job" for i in items)
