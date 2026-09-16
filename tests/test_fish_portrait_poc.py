from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.dataset_models import DatasetItem
from app.db import Base
from app.models import Batch, DatasetVersion, ImageAsset
from app.platform.models import FishAsset, PipelineRun
from app.platform.routes.portrait import (
    PortraitJobCreate,
    fish_reference_assets,
    portrait_dataset_items,
    portrait_reference,
    create_portrait_job,
)
from fastapi import BackgroundTasks


def _session(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'portrait.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)()


def _seed(db):
    db.add(
        Batch(
            batch_id="BATCH_PORTRAIT",
            source="test",
            manifest_uri="/tmp/manifest.json",
            raw_uri="/tmp/raw",
            image_count=1,
            status="INGESTED",
        )
    )
    db.add(
        DatasetVersion(
            dataset_version="DS_PORTRAIT",
            manifest_uri="/tmp/dataset.json",
            git_commit="test",
            train_count=1,
            val_count=0,
            test_count=0,
            species_count=1,
            status="FROZEN",
        )
    )
    db.flush()
    image = ImageAsset(
        batch_id="BATCH_PORTRAIT",
        image_id="IMG0001",
        file_name="fish.jpg",
        object_name="fish.jpg",
        gcs_uri="gs://private/fish.jpg",
        claimed_species="鲫鱼",
        review_status="approved",
        source_platform="MANUAL_V2",
    )
    db.add(image)
    db.flush()
    db.add(
        DatasetItem(
            dataset_version="DS_PORTRAIT",
            image_asset_id=image.id,
            batch_id="BATCH_PORTRAIT",
            image_id="IMG0001",
            gcs_uri="gs://private/fish.jpg",
            species_key="crucian_carp",
            species_name="鲫鱼",
            class_index=4,
            split="train",
        )
    )
    db.add(
        FishAsset(
            asset_id="REF_CRUCIAN",
            species="crucian_carp",
            status="ACTIVE",
            transparent_uri="/tmp/crucian.png",
            version="v1",
        )
    )
    db.commit()


def test_portrait_reuses_dataset_items_and_reference_assets(tmp_path):
    db = _session(tmp_path)
    try:
        _seed(db)
        payload = portrait_dataset_items("DS_PORTRAIT", page=1, size=60, db=db)
        assert payload["total"] == 1
        assert payload["items"][0]["image_url"] == "/media/BATCH_PORTRAIT/IMG0001"

        references = fish_reference_assets("crucian_carp", "transparent", db)
        assert references["assets"][0]["asset_id"] == "REF_CRUCIAN"
        assert references["assets"][0]["url"].endswith("/REF_CRUCIAN/media/transparent")

        matched = portrait_reference("crucian_carp", db)
        assert matched["reference_asset"]["asset_id"] == "REF_CRUCIAN"
    finally:
        db.close()


def test_portrait_job_creation_is_idempotent_for_active_request(tmp_path):
    db = _session(tmp_path)
    try:
        _seed(db)
        request = PortraitJobCreate(
            dataset_id="DS_PORTRAIT",
            source_item_id=1,
            reference_asset_id="REF_CRUCIAN",
        )
        first = create_portrait_job(request, BackgroundTasks(), db)
        second = create_portrait_job(request, BackgroundTasks(), db)
        assert first["run_id"] == second["run_id"]
        assert second["already_running"] is True
        assert db.query(PipelineRun).filter(PipelineRun.pipeline_type == "FISH_PORTRAIT_POC").count() == 1
    finally:
        db.close()
