from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import SessionLocal, get_db
from app.feedback_pipeline import materialize_feedback_batch
from app.freeze_policy import select_freeze_candidates
from app.models import FeedbackEvent
from app.species_alias import alias_resolution

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/automation", tags=["p0-automation"])

DEFAULT_FEEDBACK_THRESHOLD = 20
DEFAULT_FEEDBACK_BATCH_SIZE = 20


def feedback_threshold() -> int:
    return max(1, int(os.getenv("FEEDBACK_AUTO_BATCH_THRESHOLD", str(DEFAULT_FEEDBACK_THRESHOLD))))


def feedback_batch_size() -> int:
    return max(1, min(500, int(os.getenv("FEEDBACK_AUTO_BATCH_SIZE", str(DEFAULT_FEEDBACK_BATCH_SIZE)))))


def normalize_feedback_payload(payload: dict) -> tuple[dict, list[dict]]:
    """Normalize safe aliases while recording the raw user/model label in notes."""
    doc = dict(payload)
    changes: list[dict] = []
    for field in ("predicted_species", "corrected_species"):
        raw = doc.get(field)
        resolved = alias_resolution(raw)
        if resolved["normalized"]:
            doc[field] = resolved["canonical"]
            changes.append({"field": field, **resolved})

    if changes:
        existing = str(doc.get("user_note") or "").strip()
        audit = "; ".join(
            f"{item['field']}_raw={item['raw']}->{item['canonical']}" for item in changes
        )
        doc["user_note"] = "\n".join(x for x in (existing, f"[alias_normalized] {audit}") if x)
    return doc, changes


def _eligible_feedback_count(db: Session) -> int:
    return int(
        db.scalar(
            select(func.count())
            .select_from(FeedbackEvent)
            .where(FeedbackEvent.pipeline_status == "NEW", FeedbackEvent.image_gcs_uri.is_not(None))
        )
        or 0
    )


def feedback_auto_batch_id(last_event_id: int, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return f"BATCH_{now:%Y%m%d}_FEEDBACK_{last_event_id:08d}"


def maybe_auto_materialize_feedback(db: Session) -> dict:
    """Materialize the oldest feedback slice once the configured threshold is met."""
    threshold = feedback_threshold()
    size = max(threshold, feedback_batch_size())
    eligible = _eligible_feedback_count(db)
    if eligible < threshold:
        return {
            "triggered": False,
            "eligible": eligible,
            "threshold": threshold,
            "batch_size": size,
        }

    rows = db.scalars(
        select(FeedbackEvent)
        .where(FeedbackEvent.pipeline_status == "NEW", FeedbackEvent.image_gcs_uri.is_not(None))
        .order_by(FeedbackEvent.created_at, FeedbackEvent.id)
        .limit(size)
    ).all()
    if not rows:
        return {"triggered": False, "eligible": 0, "threshold": threshold, "batch_size": size}

    batch_id = feedback_auto_batch_id(int(rows[-1].id))
    result = materialize_feedback_batch(db, batch_id=batch_id, limit=len(rows))
    return {
        "triggered": True,
        "eligible_before": eligible,
        "threshold": threshold,
        "batch_size": len(rows),
        **result,
    }


def build_dataset_readiness(db: Session) -> dict:
    """Continuously derive trainability from the current reviewed Master Pool."""
    policy = select_freeze_candidates(
        db,
        seed=20260826,
        train=0.70,
        val=0.15,
        allow_split_blockers=True,
    )
    enabled = list(policy.get("training_enabled_species") or [])
    disabled = list(policy.get("training_disabled_species") or [])
    blockers = list(policy.get("split_blockers") or [])
    return {
        "approved_master_pool_count": int(policy.get("approved_master_pool_count", 0) or 0),
        "trainable_image_count": len(policy.get("selected") or []),
        "training_ready_species_count": len(enabled),
        "training_disabled_species_count": len(disabled),
        "training_thresholds": policy.get("training_thresholds") or {},
        "training_enabled_species": enabled,
        "training_disabled_species": disabled,
        "split_blockers": blockers,
        "freeze_ready": not bool(blockers),
        "split_strategy": policy.get("split_strategy"),
        "split_group_count": int(policy.get("split_group_count", 0) or 0),
    }


@router.get("/dataset-readiness")
def dataset_readiness(db: Session = Depends(get_db)):
    return build_dataset_readiness(db)


@router.get("/feedback-status")
def feedback_status(db: Session = Depends(get_db)):
    eligible = _eligible_feedback_count(db)
    return {
        "eligible_new_feedback": eligible,
        "auto_batch_threshold": feedback_threshold(),
        "auto_batch_size": max(feedback_threshold(), feedback_batch_size()),
        "will_materialize_on_next_feedback": eligible >= feedback_threshold(),
    }


def install_feedback_automation(app) -> None:
    """Install safe alias normalization + threshold-triggered feedback batching.

    Feedback persistence remains authoritative: auto materialization is best-effort
    and never turns an otherwise valid feedback POST into a failure.
    """

    @app.middleware("http")
    async def feedback_automation(request: Request, call_next):
        is_feedback_post = request.method == "POST" and request.url.path == "/api/feedback"
        alias_changes: list[dict] = []

        if is_feedback_post:
            raw_body = await request.body()
            body_for_downstream = raw_body
            try:
                payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
                normalized, alias_changes = normalize_feedback_payload(payload)
                if alias_changes:
                    body_for_downstream = json.dumps(normalized, ensure_ascii=False).encode("utf-8")
            except Exception:
                logger.exception("feedback alias normalization failed; forwarding original request")

            async def receive():
                return {"type": "http.request", "body": body_for_downstream, "more_body": False}

            request._receive = receive  # Replay consumed body for downstream Pydantic parsing.

        response = await call_next(request)

        if is_feedback_post and 200 <= response.status_code < 300:
            if alias_changes:
                response.headers["X-YuJian-Alias-Normalized"] = str(len(alias_changes))
            db = SessionLocal()
            try:
                result = maybe_auto_materialize_feedback(db)
                response.headers["X-YuJian-Feedback-Auto-Batch"] = (
                    str(result.get("batch_id")) if result.get("triggered") else "not-ready"
                )
            except Exception:
                db.rollback()
                logger.exception("feedback auto materialization failed; feedback remains persisted")
                response.headers["X-YuJian-Feedback-Auto-Batch"] = "error"
            finally:
                db.close()
        return response
