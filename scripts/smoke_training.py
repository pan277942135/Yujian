#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("REGISTRY_DB_URL", "sqlite:///:memory:")

from app.db import SessionLocal, init_db  # noqa: E402
from app.entry import app  # noqa: E402
from app.models import DatasetVersion, TrainingRun  # noqa: E402
from app.training_api import TrainingCreate, queue_training_run  # noqa: E402


def fake_launcher(run_id: str, dataset_version: str, model_version: str, model_family: str, params: dict) -> dict:
    assert run_id == "RUN_M1_v0.1_001"
    assert dataset_version == "DS_M1_v0.1"
    assert model_version == "MODEL_M1_v0.1"
    assert model_family == "mobilenet_v3_small"
    assert params["epochs"] == 12
    return {"name": "projects/test/locations/test/operations/op-1"}


def main() -> None:
    init_db()
    db = SessionLocal()
    try:
        db.add(
            DatasetVersion(
                dataset_version="DS_M1_v0.1",
                manifest_uri="gs://test-bucket/datasets/DS_M1_v0.1/dataset_manifest.csv",
                class_map_uri="gs://test-bucket/datasets/DS_M1_v0.1/class_map.json",
                train_count=100,
                val_count=20,
                test_count=20,
                species_count=9,
                git_commit="smoke",
                selection_mode="ALL_APPROVED",
                status="FROZEN",
            )
        )
        db.commit()

        payload = TrainingCreate(
            dataset_version="DS_M1_v0.1",
            run_id="RUN_M1_v0.1_001",
            model_version="MODEL_M1_v0.1",
        )
        result = queue_training_run(db, payload, launcher=fake_launcher)
        assert result["status"] == "QUEUED", result
        assert result["model_version"] == "MODEL_M1_v0.1", result
        assert result["cloud_run_operation"].endswith("op-1"), result

        row = db.get(TrainingRun, "RUN_M1_v0.1_001")
        assert row is not None
        assert row.status == "QUEUED"
        assert "cloud_run_operation" in row.params_json

        paths = app.openapi()["paths"]
        assert "/training" in paths
        assert "/api/training/runs" in paths
        assert "/api/training/runs/{run_id}" in paths
        assert "/api/training/runs/{run_id}/metrics" in paths
        assert "/api/models" in paths
        print("Classifier training API smoke test: OK")
    finally:
        db.close()


if __name__ == "__main__":
    main()
