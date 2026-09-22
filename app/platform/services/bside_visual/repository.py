from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.platform.models import BsideVisualSession, BsideVisualStep

from .schemas import STEP_ORDER


LEGACY_STATUS_ALIASES = {
    "RUNNING": "PROCESSING",
    "COMPLETE": "SUCCESS",
    "FAILED": "ERROR",
}


def get_session(db: Session, session_id: str) -> BsideVisualSession | None:
    return db.get(BsideVisualSession, str(session_id))


def get_session_for_qwen_run(db: Session, run_id: str) -> BsideVisualSession | None:
    return db.scalar(
        select(BsideVisualSession).where(BsideVisualSession.source_qwen_run_id == str(run_id))
    )


def get_steps(db: Session, session_id: str) -> dict[str, BsideVisualStep]:
    rows = db.scalars(
        select(BsideVisualStep).where(BsideVisualStep.session_id == str(session_id))
    ).all()
    result = {row.step_key: row for row in rows}
    # Existing B-side sessions were created with three steps. Materialize the
    # missing transparent-extraction row lazily so their old assets remain
    # viewable while new sessions use the four-step contract.
    for key in STEP_ORDER:
        if key not in result:
            row = BsideVisualStep(session_id=session_id, step_key=key, status="NOT_STARTED")
            db.add(row)
            result[key] = row
    for row in result.values():
        row.status = LEGACY_STATUS_ALIASES.get(str(row.status or "").upper(), row.status)
    db.flush()
    return result


def create_steps(db: Session, session_id: str) -> dict[str, BsideVisualStep]:
    rows = {
        key: BsideVisualStep(session_id=session_id, step_key=key, status="NOT_STARTED")
        for key in STEP_ORDER
    }
    db.add_all(rows.values())
    return rows
