from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models import DatasetVersion, TrainingRun
from app.pipeline_contract import CROP_CLASSIFIER_V1
from app.training_api import TrainingCreate, queue_training_run


def test_ready_for_training_crop_dataset_can_queue_crop_run(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'registry.db'}")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    try:
        db.add(
            DatasetVersion(
                dataset_version="DS_CROP_M1_v0.1",
                manifest_uri="gs://bucket/datasets/DS_CROP_M1_v0.1/metadata/crop_manifest.csv",
                class_map_uri="gs://bucket/datasets/DS_CROP_M1_v0.1/metadata/class_map.json",
                train_count=2,
                val_count=1,
                test_count=1,
                species_count=2,
                git_commit="test",
                selection_mode="ACCEPTED_BBOX_CROP",
                status="READY_FOR_TRAINING",
                pipeline_type=CROP_CLASSIFIER_V1,
                metadata_json='{"source":"accepted_bbox"}',
            )
        )
        db.commit()
        launched = {}

        def launcher(run_id, dataset_version, model_version, model_family, params):
            launched.update(
                run_id=run_id,
                dataset_version=dataset_version,
                model_version=model_version,
                model_family=model_family,
                params=params,
            )
            return {"name": "operations/crop-test"}

        result = queue_training_run(
            db,
            TrainingCreate(
                dataset_version="DS_CROP_M1_v0.1",
                run_id="RUN_CROP_M1_v0.1_001",
                model_version="MODEL_CROP_M1_v0.1",
                pipeline_type=CROP_CLASSIFIER_V1,
            ),
            launcher=launcher,
        )
        assert result["status"] == "QUEUED"
        assert launched["model_version"] == "MODEL_CROP_M1_v0.1"
        assert launched["params"]["pipeline_type"] == CROP_CLASSIFIER_V1
        assert db.get(TrainingRun, "RUN_CROP_M1_v0.1_001").pipeline_type == CROP_CLASSIFIER_V1
    finally:
        db.close()
