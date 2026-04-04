"""FastAPI server with SSE progress streaming.

Endpoints:
- GET  /health
- POST /research             -> start a background research job
- GET  /research/{job_id}    -> job status (+ final_report when completed)
- GET  /research/{job_id}/events -> Server-Sent Events stream of ProgressUpdate

This server is intentionally minimal and uses in-memory job storage.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.callbacks import ProgressUpdate, progress_callback, CURRENT_JOB_ID
from src.graph import run_research

logger = logging.getLogger(__name__)

app = FastAPI(title="Deep Research Agent API", version="0.1")


class ResearchRequest(BaseModel):
    topic: str = Field(min_length=1, description="Research topic")
    use_cache: bool = Field(default=True, description="Use cached results when available")
    use_checkpoints: bool = Field(default=False, description="Enable LangGraph checkpoint thread_id")


class StartResponse(BaseModel):
    job_id: str
    status_url: str
    events_url: str


JobStatus = Literal["queued", "running", "completed", "failed"]


@dataclass
class JobRecord:
    job_id: str
    topic: str
    status: JobStatus = "queued"
    created_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    queue: asyncio.Queue[ProgressUpdate | dict[str, Any]] | None = None


_JOBS: dict[str, JobRecord] = {}


def _now() -> float:
    return time.time()


def _serialize_update(update: ProgressUpdate) -> dict[str, Any]:
    return {
        "type": "progress",
        "stage": update.stage.value,
        "message": update.message,
        "details": update.details,
        "progress_pct": update.progress_pct,
        "metadata": update.metadata,
        "timestamp": update.timestamp.isoformat(),
    }


def _sse(event: str, data: dict[str, Any], event_id: str | None = None) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    lines.append(f"data: {payload}")
    return "\n".join(lines) + "\n\n"


async def _run_job(job: JobRecord, *, use_cache: bool, use_checkpoints: bool) -> None:
    job.status = "running"
    job.started_at = _now()

    assert job.queue is not None

    token = CURRENT_JOB_ID.set(job.job_id)

    async def on_update(update: ProgressUpdate) -> None:
        try:
            if update.metadata.get("job_id") != job.job_id:
                return
            job.queue.put_nowait(update)
        except asyncio.QueueFull:
            # Drop if the consumer is too slow.
            return
        except Exception:
            return

    progress_callback.register_async(on_update)

    try:
        job.queue.put_nowait(
            {
                "type": "job",
                "status": "running",
                "job_id": job.job_id,
                "topic": job.topic,
                "timestamp": datetime.now().isoformat(),
            }
        )

        result = await run_research(
            topic=job.topic,
            verbose=False,
            use_cache=use_cache,
            use_checkpoints=use_checkpoints,
            thread_id=f"api-{job.job_id}",
        )

        job.result = result
        if result.get("error"):
            job.status = "failed"
            job.error = str(result.get("error"))
        else:
            job.status = "completed"

        job.finished_at = _now()

        try:
            job.queue.put_nowait(
                {
                    "type": "job",
                    "status": job.status,
                    "job_id": job.job_id,
                    "topic": job.topic,
                    "error": job.error,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        except Exception:
            pass

    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        job.finished_at = _now()
        try:
            job.queue.put_nowait(
                {
                    "type": "job",
                    "status": "failed",
                    "job_id": job.job_id,
                    "topic": job.topic,
                    "error": job.error,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        except Exception:
            pass
        logger.exception("API research job failed")
    finally:
        progress_callback.unregister(on_update)
        CURRENT_JOB_ID.reset(token)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/research", response_model=StartResponse)
async def start_research(req: ResearchRequest) -> StartResponse:
    topic = (req.topic or "").strip()
    if not topic:
        raise HTTPException(status_code=400, detail="topic is required")

    job_id = uuid.uuid4().hex[:12]
    q: asyncio.Queue[ProgressUpdate | dict[str, Any]] = asyncio.Queue(maxsize=500)

    job = JobRecord(
        job_id=job_id,
        topic=topic,
        status="queued",
        created_at=_now(),
        queue=q,
    )
    _JOBS[job_id] = job

    # Kick off background job
    asyncio.create_task(_run_job(job, use_cache=req.use_cache, use_checkpoints=req.use_checkpoints))

    return StartResponse(
        job_id=job_id,
        status_url=f"/research/{job_id}",
        events_url=f"/research/{job_id}/events",
    )


@app.get("/research/{job_id}")
async def get_research(job_id: str, include_report: bool = True) -> dict[str, Any]:
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    out: dict[str, Any] = {
        "job_id": job.job_id,
        "topic": job.topic,
        "status": job.status,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "error": job.error,
    }

    if job.status in ("completed", "failed") and job.result is not None:
        out["result"] = job.result if include_report else {k: v for k, v in job.result.items() if k != "final_report"}

    return out


@app.get("/research/{job_id}/events")
async def research_events(job_id: str, request: Request) -> StreamingResponse:
    job = _JOBS.get(job_id)
    if job is None or job.queue is None:
        raise HTTPException(status_code=404, detail="job not found")

    async def event_stream():
        seq = 0
        # Emit a hello event
        yield _sse("hello", {"job_id": job_id, "status": job.status}, event_id=str(seq))
        seq += 1

        while True:
            if await request.is_disconnected():
                break

            try:
                item = await asyncio.wait_for(job.queue.get(), timeout=15)
            except asyncio.TimeoutError:
                # keepalive ping
                yield ": ping\n\n"
                continue

            if isinstance(item, ProgressUpdate):
                payload = _serialize_update(item)
                yield _sse("progress", payload, event_id=str(seq))
            else:
                yield _sse("job", item, event_id=str(seq))
            seq += 1

            # Stop after job completion and queue drains.
            if job.status in ("completed", "failed") and job.queue.empty():
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")
