#!/usr/bin/env python3
"""Offline Dataset Freeze smoke test.

Runs against SQLite and never calls GCS/Vision. It validates canonical selection,
quality gates, deterministic stratified group splitting, the default training
eligibility gate, DatasetItem schema registration, and FastAPI routes.
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
from app.models import Batch, ImageAsset, SpeciesCatalog  # noqa: E402
from app.presence import FishPresenceResult  # noqa: E402
from app.species_policy import ensure_target_species, training_thresholds  # noqa: E402


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


def add_qa(db, image: ImageAsset, *, presence: str = "single_fish", representative: bool = True) -> None:
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
            sha256=f"{image.id:064x}",
            phash_json="[]",
            dhash="0" * 16,
            crop_hash="",
            histogram_json="[]",
            width=100,
            height=100,
            duplicate_group=None if representative else "DUP_SMOKE",
            is_representative=representative,
            duplicate_kind=None if representative else "near",
        )
    )


def main() -> None:
    init_db()
    assert "dataset_items" in Base.metadata.tables
    assert DatasetItem.__tablename__ == "dataset_items"

    db = SessionLocal()
    try:
        ensure_species_catalog(db)
        ensure_target_species(db)

        target_rows = db.query(SpeciesCatalog).filter(SpeciesCatalog.species_key.in_([
            "grass_carp",
            "tilapia",
            "mandarin_fish",
            "sharpbelly",
            "redfin_culter",
        ])).all()
        assert len(target_rows) == 5, target_rows
        assert training_thresholds() == {"total": 20, "train": 10, "val": 3, "test": 3, "group_count": 3}

        batch_id = "BATCH_DATASET_FREEZE_SMOKE"
        db.add(
            Batch(
                batch_id=batch_id,
                source="smoke",
                image_count=40,
                manifest_uri="gs://test-bucket/smoke/manifest.csv",
                raw_uri="gs://test-bucket/smoke/",
                status="INGESTED",
            )
        )
        db.flush()

        # Low-data 鲫鱼: five valid independent samples. It must remain available in
        # Species Catalog but be default-disabled for training.
        for idx in range(5):
            image = add_image(db, batch_id, f"IMG_CRUCIAN_{idx}", "鲫鱼", group_id=f"CRUCIAN_G{idx}")
            add_qa(db, image)

        # Mature 鲤鱼: twenty valid independent samples. This class should be enabled.
        for idx in range(20):
            image = add_image(db, batch_id, f"IMG_CARP_{idx}", "鲤鱼", group_id=f"CARP_G{idx}")
            add_qa(db, image)

        # Quality exclusions still apply before the training eligibility gate.
        multi = add_image(db, batch_id, "IMG_MULTI", "草鱼")
        add_qa(db, multi, presence="multi_fish")
        no_fish = add_image(db, batch_id, "IMG_NO_FISH", "黄骨鱼")
        add_qa(db, no_fish, presence="no_fish")
        duplicate = add_image(db, batch_id, "IMG_DUP", "黑鱼")
        add_qa(db, duplicate, representative=False)
        add_image(db, batch_id, "IMG_PENDING", "加州鲈", status="pending")
        db.commit()

        payload = DatasetFreezePreviewRequest(dataset_version="DS_M1_smoke_v03", seed=7, train=0.70, val=0.15)
        first = build_preview(db, payload)
        second = build_preview(db, payload)
        assert first == second, (first, second)
        assert first["approved_master_pool_count"] == 28, first
        assert first["image_count"] == 20, first
        assert first["species_count"] == 1, first
        assert first["species_counts"] == {"鲤鱼": 20}, first
        assert first["excluded_quality_counts"].get("multi_fish") == 1, first
        assert first["excluded_quality_counts"].get("no_fish") == 1, first
        assert first["excluded_quality_counts"].get("near_duplicate") == 1, first
        assert first["split_strategy"] == SPLIT_STRATEGY, first
        assert first["freeze_ready"] is True, first
        assert not first["split_blockers"], first

        split = first["per_species_split_counts"]["鲤鱼"]
        assert split["total"] == 20, split
        assert split["train"] >= 10, split
        assert split["val"] >= 3, split
        assert split["test"] >= 3, split
        assert split["group_count"] == 20, split

        strict = select_freeze_candidates(db, seed=7, train=0.70, val=0.15)
        assert {item["catalog"].common_name_zh for item in strict["selected"]} == {"鲤鱼"}, strict
        disabled = {row["species"]: row for row in strict["training_disabled_species"]}
        assert "鲫鱼" in disabled, disabled
        assert disabled["鲫鱼"]["total"] == 5, disabled["鲫鱼"]
        assert any(reason.startswith("总数 5 < 20") for reason in disabled["鲫鱼"]["reasons"]), disabled["鲫鱼"]
        enabled = {row["species"] for row in strict["training_enabled_species"]}
        assert "鲤鱼" in enabled, enabled

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
        print("Dataset Freeze + training eligibility smoke OK", first)
    finally:
        db.close()


if __name__ == "__main__":
    main()
