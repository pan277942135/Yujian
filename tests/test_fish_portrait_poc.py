from __future__ import annotations

import json

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.dataset_models import DatasetItem
from app.db import Base
from app.fish_knowledge.import_batch import FishKnowledgeAssetVersion
from app.fish_knowledge.species import FishSpecies
from app.models import Batch, DatasetVersion, ImageAsset, SpeciesCatalog
from app.platform.models import FishAsset, PipelineRun
from app.platform.routes.portrait import (
    INPAINT_MODE,
    PortraitJobCreate,
    PortraitInpaintParams,
    PortraitParams,
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


def test_portrait_uses_cover_package_when_transparent_asset_is_missing(tmp_path):
    db = _session(tmp_path)
    try:
        catalog = SpeciesCatalog(
            species_key="crucian_carp",
            catalog_order=1,
            common_name_zh="鲫鱼",
            status="active",
            is_other=False,
        )
        db.add(catalog)
        db.flush()
        db.add(
            FishSpecies(
                id="crucian_carp",
                name_cn="鲫鱼",
                alias=[],
                category="淡水鱼",
                summary="",
                status="ACTIVE",
            )
        )
        db.add(
            FishKnowledgeAssetVersion(
                species_id="crucian_carp",
                asset_type="COVER",
                version=7,
                object_name="fish-assets/fish-knowledge/crucian_carp/cover/v7.webp",
                image_url="/api/v1/fish/knowledge-media/crucian_carp/cover/v7.webp",
                status="ACTIVE",
                sha256="a" * 64,
                metadata_json=json.dumps(
                    {
                        "source_filename": "02_transparent_alt.png",
                        "cover_variant": "COVER_CARD_TRANSPARENT_RIGHT",
                    }
                ),
            )
        )
        db.commit()

        references = fish_reference_assets("crucian_carp", "transparent", db)
        assert references["assets"][0]["type"] == "COVER_CARD_TRANSPARENT_RIGHT"
        assert references["assets"][0]["source_kind"] == "knowledge_cover"
        assert references["assets"][0]["url"].endswith("/crucian_carp/cover/v7.webp")

        matched = portrait_reference("crucian_carp", db)
        assert matched["reference_asset"]["asset_id"].startswith("KNOWLEDGE_COVER_crucian_carp_")
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
            params=PortraitParams(
                source_scale=0.9,
                reference_scale=0.15,
                steps=25,
                width=768,
                height=768,
            ),
        )
        first = create_portrait_job(request, BackgroundTasks(), db)
        second = create_portrait_job(request, BackgroundTasks(), db)
        assert first["run_id"] == second["run_id"]
        assert second["already_running"] is True
        assert db.query(PipelineRun).filter(PipelineRun.pipeline_type == "FISH_PORTRAIT_POC").count() == 1
        state = json.loads(db.get(PipelineRun, first["run_id"]).stage_json)
        assert state["request"]["params"] == {
            "source_scale": 0.9,
            "reference_scale": 0.15,
            "steps": 25,
            "width": 768,
            "height": 768,
        }
        assert state["experiment"] == {
            "mode": "dual_ip_adapter_v1",
            "adapter_config": {"source_scale": 0.9, "reference_scale": 0.15},
            "generation": {"steps": 25, "width": 768, "height": 768},
        }
    finally:
        db.close()


def test_preserve_inpaint_job_persists_mode_masks_and_experiment_params(tmp_path):
    db = _session(tmp_path)
    try:
        request = PortraitJobCreate(
            mode=INPAINT_MODE,
            original_image_uri="gs://private/original.png",
            fish_mask_uri="gs://private/fish-mask.png",
            completion_mask_uri="gs://private/completion-mask.png",
            species="鲫鱼",
            prompt="stable fish portrait",
            negative_prompt="changed fish",
            inpaint=PortraitInpaintParams(strength=0.35, steps=25, width=768, height=768, seed=12345),
        )
        first = create_portrait_job(request, BackgroundTasks(), db)
        assert first["mode"] == INPAINT_MODE
        assert first["steps"][1]["name"] == "load_masks"
        state = json.loads(db.get(PipelineRun, first["run_id"]).stage_json)
        assert state["request"]["mode"] == INPAINT_MODE
        assert state["request"]["original_image_uri"] == "gs://private/original.png"
        assert state["request"]["fish_mask_uri"] == "gs://private/fish-mask.png"
        assert state["request"]["completion_mask_uri"] == "gs://private/completion-mask.png"
        assert state["request"]["inpaint"] == {
            "strength": 0.35,
            "steps": 25,
            "width": 768,
            "height": 768,
            "seed": 12345,
        }
        assert state["experiment"]["mode"] == INPAINT_MODE
        assert state["experiment"]["mask_type"] == "completion_mask"
        assert state["experiment"]["strength"] == 0.35
        assert state["experiment"]["seed"] == 12345
    finally:
        db.close()


def test_preserve_inpaint_accepts_flat_worker_contract_fields(tmp_path):
    db = _session(tmp_path)
    try:
        request = PortraitJobCreate(
            mode=INPAINT_MODE,
            original_image_uri="gs://private/original.png",
            fish_mask_uri="gs://private/fish-mask.png",
            completion_mask_uri="gs://private/completion-mask.png",
            strength=0.15,
            steps=25,
            width=768,
            height=768,
            seed=7,
        )
        created = create_portrait_job(request, BackgroundTasks(), db)
        state = json.loads(db.get(PipelineRun, created["run_id"]).stage_json)
        assert state["request"]["inpaint"] == {
            "strength": 0.15,
            "steps": 25,
            "width": 768,
            "height": 768,
            "seed": 7,
        }
    finally:
        db.close()

