from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.entry import app
from app.models import Batch, BatchCropReview, ImageAsset, SpeciesCatalog
from app.platform.models import FishAsset, PipelineRun, PlatformOperationLog
from app.platform.routes.api import ReviewSelection, platform_batch_confirm, platform_bbox_update, ReviewBBoxSelection
from app.platform.routes import api as platform_api
from app.platform.services import adapters


def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'platform.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)()


def _batch(db, batch_id="BATCH_PLATFORM"):
    db.add(
        Batch(
            batch_id=batch_id,
            source="test",
            manifest_uri="/tmp/manifest.csv",
            raw_uri="/tmp/raw",
            image_count=1,
            status="INGESTED",
        )
    )
    db.add(SpeciesCatalog(species_key="carp", catalog_order=1, common_name_zh="鲤鱼", status="active"))
    db.add(
        ImageAsset(
            batch_id=batch_id,
            image_id="image-1",
            file_name="fish.jpg",
            object_name="fish.jpg",
            gcs_uri="gs://private/fish.jpg",
            claimed_species="鲤鱼",
            review_status="pending",
        )
    )
    db.commit()


def test_platform_empty_adapters_and_allowed_tables(tmp_path):
    db = _session(tmp_path)
    try:
        assert adapters.dashboard(db)["datasets"] == 0
        dataset_payload = adapters.datasets(db)
        assert dataset_payload["datasets"] == []
        assert dataset_payload["recent_batches"] == []
        assert dataset_payload["summary"] == {
            "dataset_count": 0,
            "batch_count": 0,
            "total_images": 0,
            "pending_review": 0,
        }
        assert adapters.review_queue(db)["total"] == 0
        assert adapters.models(db) == []
        assert adapters.pipelines(db) == []
        assert adapters.assets(db) == []
        assert {"pipeline_run", "fish_asset", "platform_operation_log"} <= set(inspect(db.bind).get_table_names())
        paths = app.openapi()["paths"]
        assert "/api/platform/dashboard" in paths
        assert "/api/platform/review/batch-confirm" in paths
        assert "/api/platform/training/create" in paths
    finally:
        db.close()


def test_platform_review_mutations_reuse_existing_gates(tmp_path):
    db = _session(tmp_path)
    try:
        _batch(db)
        image = db.query(ImageAsset).one()
        image.truth_species = "鲤鱼"
        image.truth_status = "UNCERTAIN"
        db.commit()
        db.add(
            BatchCropReview(
                batch_id=image.batch_id,
                image_asset_id=image.id,
                image_id=image.image_id,
                accepted_bbox_json="[0.1,0.1,0.7,0.7]",
                status="ACCEPTED",
            )
        )
        db.commit()

        result = platform_batch_confirm(ReviewSelection(ids=["BATCH_PLATFORM:image-1"]), db)
        assert result["updated"] == 1
        assert db.query(ImageAsset).one().review_status == "approved"

        result = platform_bbox_update(
            ReviewBBoxSelection(ids=["BATCH_PLATFORM:image-1"], accepted_bbox=[0.2, 0.2, 0.5, 0.5]),
            db,
        )
        assert result["updated"] == 1
        assert db.query(BatchCropReview).one().accepted_bbox_json == "[0.2,0.2,0.5,0.5]"
        assert db.query(PlatformOperationLog).count() == 2
    finally:
        db.close()


def test_platform_pipeline_and_asset_payloads_hide_internal_uris(tmp_path):
    db = _session(tmp_path)
    try:
        db.add(
            PipelineRun(
                run_id="PIPE_1",
                pipeline_type="FISH_ASSET",
                status="SUCCESS",
                stage_json='{"stages":[{"name":"Detector","status":"SUCCESS"}]}',
                created_at=datetime.now(timezone.utc),
            )
        )
        db.add(
            FishAsset(
                asset_id="ASSET_1",
                pipeline_run_id="PIPE_1",
                species="鲤鱼",
                original_uri="gs://private/original.jpg",
                mask_uri="gs://private/mask.png",
                transparent_uri="gs://private/fish.png",
                sticker_uri="gs://private/sticker.png",
            )
        )
        db.commit()
        pipeline = adapters.pipelines(db)[0]
        asset = adapters.assets(db)[0]
        assert pipeline["id"] == "PIPE_1"
        assert "gs://" not in str(pipeline)
        assert "gs://" not in str(asset)
        assert asset["transparent_url"].endswith("/media/transparent")
    finally:
        db.close()


def test_platform_evaluation_adds_error_driven_fields_without_replacing_legacy_payload(monkeypatch):
    monkeypatch.setattr(
        adapters,
        "evaluation",
        lambda _db, _model_id: {
            "model_id": "MODEL_CROP_M1_v0.1",
            "model_version": "MODEL_CROP_M1_v0.1",
            "metrics": {"f1": 0.69},
            "confusion_matrix": [[5, 1]],
            "errors": [],
        },
    )
    monkeypatch.setattr(
        "app.intelligence_api.build_intelligence_payload",
        lambda _db, model_version: {
            "confusion_report": {"top_confusions": []},
            "data_gaps": {"quantity_gaps": [], "scene_gaps": []},
            "scene_gaps": [],
            "production_tasks": [],
            "training_recommendations": [],
        },
    )

    result = platform_api.platform_model_evaluation("MODEL_CROP_M1_v0.1", object())

    assert result["metrics"]["f1"] == 0.69
    assert result["confusion_matrix"] == [[5, 1]]
    assert result["production_tasks"] == []
    assert result["training_recommendations"] == []
