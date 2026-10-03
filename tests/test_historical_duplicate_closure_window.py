from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.bulk_review import BulkReviewApply, api_bulk_apply
from app.db import Base
from app.historical_duplicate_closure import (
    PROTECTED_ROUTE_PATHS,
    HistoricalDuplicateClosureWriteFenceLocked,
    assert_training_authority_writable,
    authority_fingerprint,
    write_fence_active,
)
from app.main import ReviewUpdate
from app.entry import app
from app.models import Batch, ImageAsset
from app.presence import reject_no_fish
from app.dedupe import reject_duplicates
from app.factory import sync_batch_registry


def test_fence_defaults_false_and_disabled_fence_restores_writes(monkeypatch):
    monkeypatch.delenv("HISTORICAL_DUPLICATE_CLOSURE_WRITE_FENCE", raising=False)
    assert write_fence_active() is False
    assert_training_authority_writable()

    monkeypatch.setenv("HISTORICAL_DUPLICATE_CLOSURE_WRITE_FENCE", "true")
    assert write_fence_active() is True
    with pytest.raises(HistoricalDuplicateClosureWriteFenceLocked) as raised:
        assert_training_authority_writable()
    assert raised.value.as_payload() == {
        "code": "HISTORICAL_DUPLICATE_CLOSURE_IN_PROGRESS",
        "message": "Historical duplicate cleanup maintenance window is active.",
    }

    monkeypatch.setenv("HISTORICAL_DUPLICATE_CLOSURE_WRITE_FENCE", "false")
    assert_training_authority_writable()


def test_fence_http_contract_is_423():
    handler = app.exception_handlers[HistoricalDuplicateClosureWriteFenceLocked]
    response = __import__("asyncio").run(
        handler(None, HistoricalDuplicateClosureWriteFenceLocked())
    )
    assert response.status_code == 423
    assert json.loads(response.body) == {
        "code": "HISTORICAL_DUPLICATE_CLOSURE_IN_PROGRESS",
        "message": "Historical duplicate cleanup maintenance window is active.",
    }


def test_protected_route_inventory_is_registered_and_read_only_routes_are_not_listed():
    def paths(routes):
        result = set()
        for route in routes:
            if hasattr(route, "path"):
                result.add(route.path)
            original = getattr(route, "original_router", None)
            if original is not None:
                prefix = getattr(original, "prefix", "")
                for child in original.routes:
                    child_path = getattr(child, "path", None)
                    if child_path:
                        result.add(
                            child_path
                            if child_path.startswith(prefix)
                            else prefix + child_path
                        )
            if hasattr(route, "routes"):
                result.update(paths(route.routes))
        return result

    registered = paths(app.routes)
    assert set(PROTECTED_ROUTE_PATHS) <= registered
    assert "/api/review" not in PROTECTED_ROUTE_PATHS
    assert "/api/review/stats" not in PROTECTED_ROUTE_PATHS
    assert "/api/species" in registered
    assert "/health/deploy" not in PROTECTED_ROUTE_PATHS


def test_bulk_and_single_review_are_blocked_before_database_access(monkeypatch):
    monkeypatch.setenv("HISTORICAL_DUPLICATE_CLOSURE_WRITE_FENCE", "true")
    with pytest.raises(HistoricalDuplicateClosureWriteFenceLocked):
        api_bulk_apply(
            BulkReviewApply(batch_id="BATCH_TEST", items=[{"image_id": "image-1", "review_status": "pending"}]),
            None,
        )
    with pytest.raises(HistoricalDuplicateClosureWriteFenceLocked):
        from app.main import update_review

        update_review("BATCH_TEST", "image-1", ReviewUpdate(), None)


def test_automated_review_and_ingestion_writes_are_blocked(monkeypatch):
    monkeypatch.setenv("HISTORICAL_DUPLICATE_CLOSURE_WRITE_FENCE", "true")
    with pytest.raises(HistoricalDuplicateClosureWriteFenceLocked):
        reject_no_fish(None, "BATCH_TEST")
    with pytest.raises(HistoricalDuplicateClosureWriteFenceLocked):
        reject_duplicates(None, "BATCH_TEST")
    with pytest.raises(HistoricalDuplicateClosureWriteFenceLocked):
        sync_batch_registry(None, "BATCH_TEST")


def test_authority_fingerprint_is_ordered_over_truth_and_review_fields():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[Batch.__table__, ImageAsset.__table__])
    db = sessionmaker(bind=engine)()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    db.add(Batch(batch_id="BATCH_TEST", source="test", manifest_uri="m", raw_uri="r"))
    db.flush()
    db.add_all(
        [
            ImageAsset(
                batch_id="BATCH_TEST", image_id="b", file_name="b.jpg", object_name="b.jpg", gcs_uri="gs://b",
                truth_species="草鱼", truth_status="LIKELY_CORRECT", review_status="approved", created_at=now,
            ),
            ImageAsset(
                batch_id="BATCH_TEST", image_id="a", file_name="a.jpg", object_name="a.jpg", gcs_uri="gs://a",
                truth_species=None, truth_status="UNCERTAIN", review_status="pending", created_at=now,
            ),
        ]
    )
    db.commit()
    before = authority_fingerprint(db)
    assert before["image_asset_count"] == 2
    assert len(before["truth_fingerprint"]) == 64
    assert before["latest_updated_at"] is not None

    row = db.query(ImageAsset).filter_by(image_id="a").one()
    row.truth_species = "鲤鱼"
    db.commit()
    after = authority_fingerprint(db)
    assert after["image_asset_count"] == before["image_asset_count"]
    assert after["truth_fingerprint"] != before["truth_fingerprint"]
    expected = hashlib.sha256(
        json.dumps(
            [[row.id, row.truth_species, row.truth_status, row.review_status] for row in db.query(ImageAsset).order_by(ImageAsset.id)],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert after["truth_fingerprint"] == expected
    db.close()
