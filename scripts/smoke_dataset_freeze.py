#!/usr/bin/env python3
"""Offline Dataset Freeze V0.1 smoke test.

Runs against SQLite and never calls GCS/Vision. It validates preview selection,
stable group splitting, DatasetItem schema registration, and FastAPI routes.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("REGISTRY_DB_URL", "sqlite:///:memory:")

from app.entry import app  # noqa: E402
from app.dataset_api import DatasetFreezePreviewRequest, build_preview  # noqa: E402
from app.dataset_models import DatasetItem  # noqa: E402
from app.db import Base, SessionLocal, init_db  # noqa: E402
from app.dedupe import ImageFingerprint  # noqa: E402
from app.flywheel import ensure_species_catalog  # noqa: E402
from app.models import Batch, ImageAsset  # noqa: E402
from app.presence import FishPresenceResult  # noqa: E402


def add_image(db, batch_id: str, image_id: str, species: str, status: str = "approved") -> ImageAsset:
    row = ImageAsset(
        batch_id=batch_id,
        image_id=image_id,
        file_name=f"{image_id}.jpg",
        object_name=f"raw/{batch_id}/{image_id}.jpg",
        gcs_uri=f"gs://test-bucket/raw/{batch_id}/{image_id}.jpg",
        claimed_species=species,
        truth_species=species,
        review_status=status,
    )
    db.add(row)
    db.flush()
    return row


def main() -> None:
    init_db()
    assert "dataset_items" in Base.metadata.tables
    assert DatasetItem.__tablename__ == "dataset_items"

    db = SessionLocal()
    try:
        ensure_species_catalog(db)
        batch_id = "BATCH_DATASET_FREEZE_SMOKE"
        db.add(
            Batch(
                batch_id=batch_id,
                source="smoke",
                image_count=6,
                manifest_uri="gs://test-bucket/smoke/manifest.csv",
                raw_uri="gs://test-bucket/smoke/",
                status="INGESTED",
            )
        )
        db.flush()

        single = add_image(db, batch_id, "IMG_SINGLE", "鲫鱼")
        add_image(db, batch_id, "IMG_UNSCANNED", "鲤鱼")
        multi = add_image(db, batch_id, "IMG_MULTI", "草鱼")
        duplicate = add_image(db, batch_id, "IMG_DUP", "黑鱼")
        no_fish = add_image(db, batch_id, "IMG_NO_FISH", "黄骨鱼")
        pending = add_image(db, batch_id, "IMG_PENDING", "加州鲈", status="pending")

        db.add_all(
            [
                FishPresenceResult(image_asset_id=single.id, batch_id=batch_id, status="single_fish", fish_count=1),
                FishPresenceResult(image_asset_id=multi.id, batch_id=batch_id, status="multi_fish", fish_count=2),
                FishPresenceResult(image_asset_id=no_fish.id, batch_id=batch_id, status="no_fish", fish_count=0),
                FishPresenceResult(image_asset_id=pending.id, batch_id=batch_id, status="single_fish", fish_count=1),
                ImageFingerprint(
                    image_asset_id=duplicate.id,
                    batch_id=batch_id,
                    sha256="d" * 64,
                    phash_json="[]",
                    dhash="0" * 16,
                    crop_hash="",
                    histogram_json="[]",
                    width=100,
                    height=100,
                    duplicate_group="DUP_SMOKE",
                    is_representative=False,
                    duplicate_kind="near",
                ),
            ]
        )
        db.commit()

        payload = DatasetFreezePreviewRequest(dataset_version="DS_M1_smoke", seed=7, train=0.70, val=0.15)
        first = build_preview(db, payload)
        second = build_preview(db, payload)
        assert first == second, (first, second)
        assert first["approved_master_pool_count"] == 5, first
        assert first["image_count"] == 2, first
        assert first["species_count"] == 2, first
        assert first["species_counts"] == {"鲫鱼": 1, "鲤鱼": 1}, first
        assert first["excluded_quality_counts"].get("multi_fish") == 1, first
        assert first["excluded_quality_counts"].get("no_fish") == 1, first
        assert first["excluded_quality_counts"].get("near_duplicate") == 1, first
        assert sum(first["split_counts"].values()) == 2, first

        paths = app.openapi()["paths"]
        assert "/api/datasets/freeze" in paths
        assert "/api/dataset-freeze/preview" in paths
        assert "/api/dataset-freeze/{dataset_version}/finalize" in paths
        assert "/api/dataset-freeze/{dataset_version}/items" in paths
        print("Dataset Freeze smoke OK", first)
    finally:
        db.close()


if __name__ == "__main__":
    main()
