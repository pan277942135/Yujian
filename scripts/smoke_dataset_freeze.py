#!/usr/bin/env python3
"""Offline Dataset Freeze smoke test.

Runs against SQLite and never calls GCS/Vision. It validates canonical selection,
quality gates, deterministic stratified group splitting, split blockers, DatasetItem
schema registration, and FastAPI routes.
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
from app.freeze_policy import SPLIT_STRATEGY, select_freeze_candidates  # noqa: E402
from app.models import Batch, ImageAsset  # noqa: E402
from app.presence import FishPresenceResult  # noqa: E402


def add_image(
    db,
    batch_id: str,
    image_id: str,
    species: str,
    status: str = "approved",
    group_id: str | None = None,
) -> ImageAsset:
    row = ImageAsset(
        batch_id=batch_id,
        image_id=image_id,
        file_name=f"{image_id}.jpg",
        object_name=f"raw/{batch_id}/{image_id}.jpg",
        gcs_uri=f"gs://test-bucket/raw/{batch_id}/{image_id}.jpg",
        claimed_species=species,
        truth_species=species,
        review_status=status,
        group_id=group_id,
    )
    db.add(row)
    db.flush()
    return row


def add_qa(db, image: ImageAsset, *, sha_char: str, presence: str = "single_fish") -> None:
    db.add(
        FishPresenceResult(
            image_asset_id=image.id,
            batch_id=image.batch_id,
            status=presence,
            fish_count=1 if presence == "single_fish" else (2 if presence == "multi_fish" else 0),
        )
    )
    db.add(
        ImageFingerprint(
            image_asset_id=image.id,
            batch_id=image.batch_id,
            sha256=(sha_char * 64)[:64],
            phash_json="[]",
            dhash="0" * 16,
            crop_hash="",
            histogram_json="[]",
            width=100,
            height=100,
        )
    )


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
                image_count=20,
                manifest_uri="gs://test-bucket/smoke/manifest.csv",
                raw_uri="gs://test-bucket/smoke/",
                status="INGESTED",
            )
        )
        db.flush()

        single = add_image(db, batch_id, "IMG_SINGLE", "鲫鱼", group_id="CRUCIAN_G0")
        unscanned = add_image(db, batch_id, "IMG_UNSCANNED", "鲤鱼", group_id="CARP_G0")
        multi = add_image(db, batch_id, "IMG_MULTI", "草鱼")
        duplicate = add_image(db, batch_id, "IMG_DUP", "黑鱼")
        no_fish = add_image(db, batch_id, "IMG_NO_FISH", "黄骨鱼")
        pending = add_image(db, batch_id, "IMG_PENDING", "加州鲈", status="pending")

        db.add_all(
            [
                FishPresenceResult(image_asset_id=single.id, batch_id=batch_id, status="single_fish", fish_count=1),
                FishPresenceResult(image_asset_id=unscanned.id, batch_id=batch_id, status="single_fish", fish_count=1),
                FishPresenceResult(image_asset_id=multi.id, batch_id=batch_id, status="multi_fish", fish_count=2),
                FishPresenceResult(image_asset_id=no_fish.id, batch_id=batch_id, status="no_fish", fish_count=0),
                FishPresenceResult(image_asset_id=duplicate.id, batch_id=batch_id, status="single_fish", fish_count=1),
                FishPresenceResult(image_asset_id=pending.id, batch_id=batch_id, status="single_fish", fish_count=1),
                ImageFingerprint(
                    image_asset_id=single.id,
                    batch_id=batch_id,
                    sha256="a" * 64,
                    phash_json="[]",
                    dhash="0" * 16,
                    crop_hash="",
                    histogram_json="[]",
                    width=100,
                    height=100,
                ),
                ImageFingerprint(
                    image_asset_id=unscanned.id,
                    batch_id=batch_id,
                    sha256="b" * 64,
                    phash_json="[]",
                    dhash="0" * 16,
                    crop_hash="",
                    histogram_json="[]",
                    width=100,
                    height=100,
                ),
                ImageFingerprint(
                    image_asset_id=multi.id,
                    batch_id=batch_id,
                    sha256="c" * 64,
                    phash_json="[]",
                    dhash="0" * 16,
                    crop_hash="",
                    histogram_json="[]",
                    width=100,
                    height=100,
                ),
                ImageFingerprint(
                    image_asset_id=no_fish.id,
                    batch_id=batch_id,
                    sha256="e" * 64,
                    phash_json="[]",
                    dhash="0" * 16,
                    crop_hash="",
                    histogram_json="[]",
                    width=100,
                    height=100,
                ),
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

        # Preview remains diagnostic even when represented species cannot cover all splits.
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
        assert first["split_strategy"] == SPLIT_STRATEGY, first
        assert first["freeze_ready"] is False, first
        assert first["split_blockers"], first

        # Formal selection is strict: the same small pool cannot be frozen.
        try:
            select_freeze_candidates(db, seed=7, train=0.70, val=0.15)
        except ValueError as exc:
            assert "Dataset Split Gate" in str(exc), exc
        else:
            raise AssertionError("formal split gate should reject zero-coverage species")

        # Add four independent groups to each represented species. With five groups per
        # species the v0.3 stratified splitter must produce non-zero train/val/test.
        chars = iter("fghijklmnopqrstuvwxyz0123456789")
        for species, prefix in (("鲫鱼", "CRUCIAN"), ("鲤鱼", "CARP")):
            for idx in range(1, 5):
                image = add_image(
                    db,
                    batch_id,
                    f"IMG_{prefix}_{idx}",
                    species,
                    group_id=f"{prefix}_G{idx}",
                )
                add_qa(db, image, sha_char=next(chars))
        db.commit()

        ready_payload = DatasetFreezePreviewRequest(dataset_version="DS_M1_smoke_v03", seed=7, train=0.70, val=0.15)
        ready_a = build_preview(db, ready_payload)
        ready_b = build_preview(db, ready_payload)
        assert ready_a == ready_b, (ready_a, ready_b)
        assert ready_a["freeze_ready"] is True, ready_a
        assert not ready_a["split_blockers"], ready_a
        assert ready_a["split_strategy"] == SPLIT_STRATEGY, ready_a
        assert ready_a["species_counts"] == {"鲫鱼": 5, "鲤鱼": 5}, ready_a
        for species in ("鲫鱼", "鲤鱼"):
            split = ready_a["per_species_split_counts"][species]
            assert split["train"] > 0 and split["val"] > 0 and split["test"] > 0, split
            assert split["group_count"] == 5, split

        strict = select_freeze_candidates(db, seed=7, train=0.70, val=0.15)
        group_to_split = {}
        for item in strict["selected"]:
            group = item["group_key"]
            previous = group_to_split.setdefault(group, item["split"])
            assert previous == item["split"], (group, previous, item["split"])

        paths = app.openapi()["paths"]
        assert "/api/datasets/freeze" in paths
        assert "/api/dataset-freeze/preview" in paths
        assert "/api/dataset-freeze/{dataset_version}/finalize" in paths
        assert "/api/dataset-freeze/{dataset_version}/items" in paths
        print("Dataset Freeze smoke OK", ready_a)
    finally:
        db.close()


if __name__ == "__main__":
    main()
