"""Durable single-consumer queue for Image Studio Qwen generation."""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select, text

from app.db import SessionLocal, engine
from app.platform.models import ImageStudioRun

QUEUED = "QUEUED"
RUNNING = "RUNNING"
SUCCESS = "SUCCESS"
FAILED = "FAILED"

# One local consumer per Cloud Run process. PostgreSQL advisory locking below
# provides the cross-instance singleton for the shared L4/ComfyUI worker.
_WORKER_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="image-studio-queue")
_LOCAL_QUEUE_LOCK = threading.Lock()
_RETRY_TIMER_LOCK = threading.Lock()
_RETRY_TIMER: threading.Timer | None = None
_QUEUE_LOCK_KEY = 493_867_251_031


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _request_state(run: ImageStudioRun) -> dict[str, Any]:
    try:
        payload = json.loads(run.request_json or "{}")
    except json.JSONDecodeError:
        payload = {}
    return payload if isinstance(payload, dict) else {}


def _set_generation_stage(run: ImageStudioRun, status: str) -> None:
    state = _request_state(run)
    stages = state.get("stages")
    if not isinstance(stages, list):
        stages = []
        state["stages"] = stages
    stage = next(
        (
            item
            for item in stages
            if isinstance(item, dict) and item.get("name") == "qwen_generation"
        ),
        None,
    )
    if stage is None:
        stage = {"name": "qwen_generation", "status": status}
        stages.append(stage)
    else:
        stage["status"] = status
    run.request_json = json.dumps(state, ensure_ascii=False)


def _requeue_orphaned_running_jobs() -> int:
    """Recover RUNNING rows only after this process owns the global queue lock.

    The advisory lock is held for the entire lifetime of a live queue consumer.
    Therefore, once another process successfully acquires it, any RUNNING row
    left in the database cannot still be owned by a live Image Studio consumer.
    This makes restart/deploy recovery immediate instead of waiting for a fixed
    lease timeout.
    """

    db = SessionLocal()
    try:
        rows = list(
            db.scalars(
                select(ImageStudioRun)
                .where(ImageStudioRun.status == RUNNING)
                .order_by(ImageStudioRun.created_at.asc(), ImageStudioRun.run_id.asc())
                .with_for_update(skip_locked=True)
            )
        )
        for run in rows:
            run.status = QUEUED
            run.started_at = None
            run.finished_at = None
            run.error_code = None
            run.error_message = None
            _set_generation_stage(run, QUEUED)
        if rows:
            db.commit()
        return len(rows)
    finally:
        db.close()


def _claim_next_job() -> str | None:
    db = SessionLocal()
    try:
        run = db.scalar(
            select(ImageStudioRun)
            .where(ImageStudioRun.status == QUEUED)
            .order_by(ImageStudioRun.created_at.asc(), ImageStudioRun.run_id.asc())
            .with_for_update(skip_locked=True)
        )
        if run is None:
            return None
        run.status = RUNNING
        run.started_at = _utcnow()
        run.finished_at = None
        run.error_code = None
        run.error_message = None
        _set_generation_stage(run, RUNNING)
        run_id = str(run.run_id)
        db.commit()
        return run_id
    finally:
        db.close()


def requeue_image_studio_job(run_id: str) -> None:
    """Return a transient worker-not-ready job to the front of the durable queue."""

    db = SessionLocal()
    try:
        run = db.get(ImageStudioRun, str(run_id))
        if run is None:
            return
        run.status = QUEUED
        run.started_at = None
        run.finished_at = None
        run.error_code = None
        run.error_message = None
        _set_generation_stage(run, QUEUED)
        db.commit()
    finally:
        db.close()


def _acquire_cross_instance_lock():
    """Return a held lock token, or None when another consumer owns the queue."""

    if engine.dialect.name == "postgresql":
        connection = engine.connect()
        acquired = bool(
            connection.execute(
                text("SELECT pg_try_advisory_lock(:lock_key)"),
                {"lock_key": _QUEUE_LOCK_KEY},
            ).scalar()
        )
        if not acquired:
            connection.close()
            return None
        return connection

    if not _LOCAL_QUEUE_LOCK.acquire(blocking=False):
        return None
    return _LOCAL_QUEUE_LOCK


def _release_cross_instance_lock(token) -> None:
    if token is None:
        return
    if engine.dialect.name == "postgresql":
        try:
            token.execute(
                text("SELECT pg_advisory_unlock(:lock_key)"),
                {"lock_key": _QUEUE_LOCK_KEY},
            )
        finally:
            token.close()
        return
    token.release()


def _schedule_drain_retry(delay_seconds: float = 1.0) -> None:
    """Ensure a queued job is retried when another instance owns the advisory lock."""

    global _RETRY_TIMER
    with _RETRY_TIMER_LOCK:
        if _RETRY_TIMER is not None and _RETRY_TIMER.is_alive():
            return
        timer = threading.Timer(max(0.2, delay_seconds), enqueue_image_studio_queue)
        timer.daemon = True
        _RETRY_TIMER = timer
        timer.start()


def _drain_queue() -> None:
    token = _acquire_cross_instance_lock()
    if token is None:
        _schedule_drain_retry()
        return
    try:
        # Owning the global advisory lock proves no live Image Studio consumer
        # is currently responsible for RUNNING rows. Recover them immediately
        # before claiming the next FIFO job.
        _requeue_orphaned_running_jobs()
        while True:
            run_id = _claim_next_job()
            if run_id is None:
                return

            # Lazy import avoids a route/service import cycle while keeping the
            # generation implementation and storage contract in one place.
            from app.platform.routes.image_studio import _execute_queued_run

            outcome = _execute_queued_run(run_id)
            if outcome == "REQUEUED":
                # Qwen can briefly report loading during a worker restart.
                # Keep the job durable and retry without exposing a false FAIL.
                time.sleep(max(1.0, float(os.getenv("IMAGE_STUDIO_QUEUE_RETRY_SECONDS", "5"))))
    finally:
        _release_cross_instance_lock(token)


def enqueue_image_studio_queue() -> None:
    """Wake the durable queue consumer without blocking the HTTP request."""

    _WORKER_POOL.submit(_drain_queue)


def recover_pending_image_studio_jobs(*, limit: int = 100) -> int:
    """Resume durable work after application startup.

    Recovery itself runs inside the same advisory-lock protected drain path.
    If an older Cloud Run revision is still consuming a job, lock acquisition
    waits/retries. Once that owner disappears, orphaned RUNNING rows are
    immediately returned to FIFO order and processing resumes.
    """

    db = SessionLocal()
    try:
        pending = int(
            db.scalar(
                select(func.count())
                .select_from(ImageStudioRun)
                .where(ImageStudioRun.status.in_([QUEUED, RUNNING]))
            )
            or 0
        )
    finally:
        db.close()

    if pending:
        enqueue_image_studio_queue()
    return pending


__all__ = [
    "FAILED",
    "QUEUED",
    "RUNNING",
    "SUCCESS",
    "enqueue_image_studio_queue",
    "recover_pending_image_studio_jobs",
    "requeue_image_studio_job",
]
