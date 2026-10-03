"""Controlled write fence and authority fingerprint for historical cleanup."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import ImageAsset, SpeciesCatalog

WRITE_FENCE_ENV = "HISTORICAL_DUPLICATE_CLOSURE_WRITE_FENCE"
WRITE_FENCE_CODE = "HISTORICAL_DUPLICATE_CLOSURE_IN_PROGRESS"
WRITE_FENCE_MESSAGE = "Historical duplicate cleanup maintenance window is active."

# This is the review/truth/ingestion surface deliberately covered by the
# closure window.  Read-only paths and Phase A/B operators are not included.
PROTECTED_ROUTE_PATHS = (
    "/api/bulk-review/apply",
    "/api/review/{batch_id}/{image_id}",
    "/api/platform/review/batch-confirm",
    "/api/platform/review/species",
    "/api/platform/review/bbox",
    "/api/platform/review/batch-delete",
    "/api/crop-review/{batch_id}/{image_id}",
    "/api/dataset-accepted-bbox/{batch_id}/{image_id}",
    "/api/dataset-accepted-bbox/bulk",
    "/api/presence/reject-no-fish",
    "/api/dedupe/reject-duplicates",
    "/api/inspect/presence/{batch_id}/{image_id}",
    "/api/batches/promote",
    "/api/batches/sync",
    "/api/batches/upload-file",
    "/api/batches/upload-finalize",
    "/api/batches/upload",
    "/api/species",
    "/api/species/{species_key}/status",
    "/api/feedback/materialize",
    "/api/feedback/ingest",
    "/api/v1/inference/upload",
    "/api/v1/inference/{image_id}/review",
    "/api/dataset-freeze/{dataset_version}/finalize",
    "/api/datasets/freeze",
)


class HistoricalDuplicateClosureWriteFenceLocked(RuntimeError):
    """Raised when a protected mutation is attempted during closure."""

    code = WRITE_FENCE_CODE
    message = WRITE_FENCE_MESSAGE

    def as_payload(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def write_fence_active() -> bool:
    return _truthy(os.getenv(WRITE_FENCE_ENV, "false"))


def assert_training_authority_writable() -> None:
    if write_fence_active():
        raise HistoricalDuplicateClosureWriteFenceLocked()


def assert_feedback_species_allowed(db: Session, corrected_species: str | None) -> None:
    """Allow feedback labels only when closure cannot create a new candidate."""

    corrected = (corrected_species or "").strip()
    if not write_fence_active() or not corrected:
        return
    known = db.scalar(select(SpeciesCatalog).where(SpeciesCatalog.common_name_zh == corrected))
    if known is None:
        raise HistoricalDuplicateClosureWriteFenceLocked()


def authority_fingerprint(db: Session) -> dict[str, Any]:
    """Return the closure proof over the fields that define review authority."""

    rows = db.execute(
        select(
            ImageAsset.id,
            ImageAsset.truth_species,
            ImageAsset.truth_status,
            ImageAsset.review_status,
        ).order_by(ImageAsset.id)
    ).all()
    ordered = [
        [int(row.id), row.truth_species, row.truth_status, row.review_status]
        for row in rows
    ]
    encoded = json.dumps(ordered, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    latest = db.scalar(select(func.max(ImageAsset.updated_at)))
    return {
        "image_asset_count": len(ordered),
        "latest_updated_at": latest.isoformat() if latest is not None else None,
        "truth_fingerprint": hashlib.sha256(encoded).hexdigest(),
    }
