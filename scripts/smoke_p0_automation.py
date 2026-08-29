#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("REGISTRY_DB_URL", "sqlite:///:memory:")
os.environ.setdefault("FEEDBACK_AUTO_BATCH_THRESHOLD", "20")
os.environ.setdefault("FEEDBACK_AUTO_BATCH_SIZE", "20")

from sqlalchemy import select  # noqa: E402

from app.db import SessionLocal, init_db  # noqa: E402
from app.entry import app  # noqa: E402
from app.flywheel import record_feedback  # noqa: E402
from app.models import FeedbackEvent, SpeciesCatalog  # noqa: E402
from app.p0_automation import (  # noqa: E402
    build_dataset_readiness,
    feedback_auto_batch_id,
    maybe_auto_materialize_feedback,
    normalize_feedback_payload,
)
from app.species_alias import alias_resolution, normalize_species_name  # noqa: E402
from app.species_policy import ensure_target_species  # noqa: E402


def main() -> None:
    init_db()

    assert normalize_species_name("桂鱼") == "鳜鱼"
    assert normalize_species_name("花鲢") == "鳙鱼"
    assert normalize_species_name("黄辣丁") == "黄骨鱼"
    assert normalize_species_name("鲢鱼") == "鲢鱼"  # ambiguous search-only alias
    assert alias_resolution("桂花鱼")["known"] is True
    assert alias_resolution("鳊鱼")["search_only"] is True

    normalized, changes = normalize_feedback_payload(
        {
            "predicted_species": "花鲢",
            "corrected_species": "桂鱼",
            "user_note": "user correction",
        }
    )
    assert normalized["predicted_species"] == "鳙鱼"
    assert normalized["corrected_species"] == "鳜鱼"
    assert len(changes) == 2
    assert "corrected_species_raw=桂鱼->鳜鱼" in normalized["user_note"]

    batch_id = feedback_auto_batch_id(23, datetime(2026, 8, 29, tzinfo=timezone.utc))
    assert batch_id == "BATCH_20260829_FEEDBACK_00000023"

    db = SessionLocal()
    try:
        ensure_target_species(db)

        # The normalized correction must resolve to the existing canonical species,
        # not create a duplicate candidate named 桂鱼.
        feedback = record_feedback(
            db,
            source_event_id="P0_ALIAS_SMOKE_001",
            feedback_type="corrected",
            source="smoke",
            predicted_species=normalized["predicted_species"],
            corrected_species=normalized["corrected_species"],
            user_note=normalized["user_note"],
        )
        assert feedback["feedback_type"] == "corrected"
        assert feedback["corrected_species"] == "鳜鱼"
        assert db.scalar(select(SpeciesCatalog).where(SpeciesCatalog.common_name_zh == "桂鱼")) is None
        stored = db.scalar(select(FeedbackEvent).where(FeedbackEvent.source_event_id == "P0_ALIAS_SMOKE_001"))
        assert stored is not None and "corrected_species_raw=桂鱼->鳜鱼" in (stored.user_note or "")

        status = maybe_auto_materialize_feedback(db)
        assert status["triggered"] is False
        assert status["eligible"] == 0
        assert status["threshold"] == 20

        readiness = build_dataset_readiness(db)
        assert readiness["training_ready_species_count"] == 0
        assert readiness["training_disabled_species_count"] >= 20
        assert readiness["freeze_ready"] is False
        assert readiness["training_thresholds"]["total"] == 20
        assert readiness["training_thresholds"]["test"] == 3
    finally:
        db.close()

    paths = app.openapi()["paths"]
    assert "/api/automation/dataset-readiness" in paths
    assert "/api/automation/feedback-status" in paths

    batches_html = (ROOT / "app/templates/batches.html").read_text(encoding="utf-8")
    for token in (
        "1/5 自动检查",
        "4/5 自动检测近重复",
        "5/5 自动检测鱼体",
        "/api/dedupe/scan",
        "/api/presence/scan",
        "直接进入“快速审核”",
    ):
        assert token in batches_html, token

    middleware_text = (ROOT / "app/p0_automation.py").read_text(encoding="utf-8")
    assert "Replay consumed body for downstream Pydantic parsing" in middleware_text

    print("P0 automation smoke OK", {"readiness_disabled": readiness["training_disabled_species_count"]})


if __name__ == "__main__":
    main()
