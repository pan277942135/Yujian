#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("REGISTRY_DB_URL", "sqlite:///:memory:")

from starlette.datastructures import Headers, UploadFile  # noqa: E402

from app.db import SessionLocal, init_db  # noqa: E402
from app.entry import app  # noqa: E402
from app import inference_api  # noqa: E402
from app.models import DatasetVersion, ModelVersion, TrainingRun  # noqa: E402


def fake_predict(_db, model_version: str, _data: bytes) -> dict:
    assert model_version == "MODEL_SMOKE"
    return {
        "model_version": model_version,
        "model_status": "CANDIDATE",
        "image_size": 224,
        "input": {"width": 1080, "height": 1440, "format": "JPEG"},
        "top1": {"class_index": 0, "species": "草鱼", "species_key": "grass_carp", "confidence": 0.72},
        "top3": [
            {"class_index": 0, "species": "草鱼", "species_key": "grass_carp", "confidence": 0.72},
            {"class_index": 8, "species": "青鱼", "species_key": "black_carp", "confidence": 0.18},
            {"class_index": 3, "species": "鲤鱼", "species_key": "common_carp", "confidence": 0.06},
        ],
        "low_confidence": False,
        "low_confidence_threshold": 0.55,
        "latency_ms": 41.2,
    }


def fake_persist(**_kwargs) -> str:
    return "gs://smoke-bucket/inference-tests/MODEL_SMOKE/20260828/test.jpg"


def upload(name: str) -> UploadFile:
    return UploadFile(
        file=io.BytesIO(b"fake-image-bytes"),
        filename=name,
        headers=Headers({"content-type": "image/jpeg"}),
    )


def main() -> None:
    init_db()
    db = SessionLocal()
    old_predict = inference_api._predict_bytes
    old_persist = inference_api._persist_image
    try:
        db.add(
            DatasetVersion(
                dataset_version="DS_SMOKE",
                manifest_uri="gs://smoke-bucket/datasets/DS_SMOKE/dataset_manifest.csv",
                class_map_uri="gs://smoke-bucket/datasets/DS_SMOKE/class_map.json",
                train_count=10,
                val_count=2,
                test_count=2,
                species_count=2,
                git_commit="smoke",
                selection_mode="ALL_APPROVED_VERIFIED_TRUTH",
                status="FROZEN",
            )
        )
        db.add(
            TrainingRun(
                run_id="RUN_SMOKE",
                dataset_version="DS_SMOKE",
                git_commit="smoke",
                model_family="mobilenet_v3_small",
                params_json="{}",
                seed=1,
                status="COMPLETED",
                artifact_uri="gs://smoke-bucket/models/MODEL_SMOKE/model_torchscript.pt",
                metrics_uri="gs://smoke-bucket/models/MODEL_SMOKE/metrics.json",
            )
        )
        db.add(
            ModelVersion(
                model_version="MODEL_SMOKE",
                run_id="RUN_SMOKE",
                artifact_uri="gs://smoke-bucket/models/MODEL_SMOKE/model_torchscript.pt",
                metrics_uri="gs://smoke-bucket/models/MODEL_SMOKE/metrics.json",
                status="CANDIDATE",
            )
        )
        db.commit()

        models = inference_api.inference_models(db)
        assert models[0]["model_version"] == "MODEL_SMOKE", models
        assert models[0]["dataset_version"] == "DS_SMOKE", models

        inference_api._predict_bytes = fake_predict
        inference_api._persist_image = fake_persist

        single = asyncio.run(inference_api.inference_predict("MODEL_SMOKE", upload("one.jpg"), db))
        assert single["top1"]["species"] == "草鱼", single
        assert len(single["top3"]) == 3, single
        assert single["image_gcs_uri"].startswith("gs://smoke-bucket/inference-tests/"), single
        assert single["inference_id"].startswith("INF_"), single

        batch = asyncio.run(inference_api.inference_batch("MODEL_SMOKE", [upload("a.jpg"), upload("b.jpg")], db))
        assert batch["count"] == 2, batch
        assert batch["low_confidence_count"] == 0, batch
        assert len(batch["results"]) == 2, batch

        template = (ROOT / "app" / "templates" / "inference.html").read_text(encoding="utf-8")
        for marker in ["模型实测", "单张测试", "批量测试", "Top-3", "保存已标注反馈", "saveBatchFeedback"]:
            assert marker in template, marker

        paths = app.openapi()["paths"]
        assert "/inference" in paths
        assert "/api/inference/models" in paths
        assert "/api/inference/predict" in paths
        assert "/api/inference/batch" in paths
        print("Inference V1 single + batch smoke test: OK")
    finally:
        inference_api._predict_bytes = old_predict
        inference_api._persist_image = old_persist
        db.close()


if __name__ == "__main__":
    main()
