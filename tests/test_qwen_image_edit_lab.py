from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette.datastructures import Headers, UploadFile

from app.dataset_models import DatasetItem
from app.db import Base
from app.entry import app
from app.models import DatasetVersion
from app.platform.models import PipelineRun
from app.platform.routes import qwen_image_edit_lab as lab


def _session(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'qwen_lab.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)()


def _upload() -> UploadFile:
    return UploadFile(
        file=io.BytesIO(b"fake-image-bytes"),
        filename="fish.jpg",
        headers=Headers({"content-type": "image/jpeg"}),
    )


def test_qwen_image_edit_lab_routes_are_additive():
    paths = app.openapi()["paths"]
    assert "/fish-portrait/qwen-lab" in paths
    assert "/api/fish-portrait/qwen-lab/generate" in paths
    assert "/api/fish-portrait/qwen-lab/runs" in paths
    assert "/api/fish-portrait/qwen-lab/runs/{run_id}" in paths
    assert "/api/fish-portrait/qwen-lab/runs/{run_id}/media/{kind}" in paths
    assert lab.PIPELINE_TYPE == "QWEN_IMAGE_EDIT_LAB"
    assert lab.MODEL_ID == "qwen-image-edit-2511"


def test_qwen_image_edit_lab_defaults_preserve_original_fish():
    assert "严格保留原图中的真实鱼体" in lab.DEFAULT_PROMPT
    assert "鱼体必须横向放置" in lab.DEFAULT_PROMPT
    assert "不要改变鱼种" in lab.DEFAULT_NEGATIVE_PROMPT
    assert "不要只显示半条鱼" in lab.DEFAULT_NEGATIVE_PROMPT


def test_qwen_image_edit_lab_history_lists_only_lab_runs(tmp_path):
    db = _session(tmp_path)
    try:
        db.add(
            PipelineRun(
                run_id="QWEN_HISTORY_1",
                pipeline_type=lab.PIPELINE_TYPE,
                status="SUCCESS",
                stage_json=json.dumps(
                    {
                        "request": {
                            "input_image_uri": "gs://bucket/input.jpg",
                            "prompt": lab.DEFAULT_PROMPT,
                            "negative_prompt": lab.DEFAULT_NEGATIVE_PROMPT,
                            "seed": 42,
                        },
                        "result": {
                            "output_image_uri": "gs://bucket/output.png",
                            "seed": 42,
                            "elapsed_ms": 321,
                        },
                    },
                    ensure_ascii=False,
                ),
            )
        )
        db.add(
            PipelineRun(
                run_id="OTHER_PIPELINE_1",
                pipeline_type="FISH_PORTRAIT_POC",
                status="SUCCESS",
                stage_json="{}",
            )
        )
        db.commit()

        rows = lab.qwen_image_edit_lab_runs(limit=10, db=db)

        assert [row["run_id"] for row in rows] == ["QWEN_HISTORY_1"]
        assert rows[0]["input_image_url"].endswith("/runs/QWEN_HISTORY_1/media/input")
        assert rows[0]["output_image_url"].endswith("/runs/QWEN_HISTORY_1/media/output")
        assert rows[0]["time_ms"] == 321
    finally:
        db.close()


def test_qwen_image_edit_lab_template_supports_dataset_selection():
    template = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "templates"
        / "platform"
        / "lab"
        / "qwen_image_edit.html"
    ).read_text(encoding="utf-8")

    assert 'id="qwenLabDataset"' in template
    assert 'id="qwenLabDatasetGrid"' in template
    assert "/api/platform/datasets" in template
    assert "/items?page=1&size=60" in template
    assert "URLSearchParams" in template
    assert "dataset_item_id" in template
    assert "selectedDatasetSource" in template
    assert "new File([blob]" not in template


def test_qwen_image_edit_lab_dataset_source_reads_server_side(tmp_path, monkeypatch):
    db = _session(tmp_path)
    try:
        db.add(
            DatasetVersion(
                dataset_version="DS_TEST",
                manifest_uri="gs://bucket/manifest.json",
                git_commit="test",
                status="FROZEN",
                pipeline_type="WHOLE_IMAGE_V1",
            )
        )
        db.add(
            DatasetItem(
                dataset_version="DS_TEST",
                image_asset_id=1,
                batch_id="BATCH_TEST",
                image_id="IMG00012",
                gcs_uri="gs://bucket/IMG00012.jpg",
                species_key="crucian_carp",
                species_name="鲫鱼",
                class_index=0,
                split="train",
            )
        )
        db.commit()

        stored = {}

        def fake_store(run_id, kind, data, media_type, extension):
            uri = "local://qwen-image-edit-lab/" + run_id + "/" + kind + extension
            stored[kind] = (uri, data, media_type)
            return uri

        def fake_managed(uri):
            assert uri == "gs://bucket/IMG00012.jpg"
            return b"dataset-image", "image/jpeg"

        def fake_read(uri, *, label):
            assert uri == "http://worker/output.png"
            assert label == "qwen_lab_output"
            return b"generated-image", "image/png"

        def fake_worker(**kwargs):
            assert kwargs["visible_fish_refined_image_uri"].startswith("local://qwen-image-edit-lab/")
            return {
                "result_uri": "http://worker/output.png",
                "worker_model": "Qwen-Image-Edit-2511",
                "worker_status": "WORKER_EXECUTED",
                "worker_http_status": 200,
                "seed": 456,
                "elapsed_ms": 789,
            }

        monkeypatch.setattr(lab, "_store_bytes", fake_store)
        monkeypatch.setattr(lab, "_read_managed_uri", fake_managed)
        monkeypatch.setattr(lab, "_read_image_uri", fake_read)
        monkeypatch.setattr(lab, "invoke_qwen_refine_worker", fake_worker)

        response = asyncio.run(
            lab.generate_qwen_image_edit_lab(
                image=None,
                prompt="dataset prompt",
                negative_prompt="dataset negative",
                seed="456",
                dataset_id="DS_TEST",
                dataset_item_id="1",
                db=db,
            )
        )

        assert response["status"] == "SUCCESS"
        assert response["input_source"] == "DATASET"
        assert response["dataset_id"] == "DS_TEST"
        assert response["dataset_item_id"] == 1
        assert response["image_id"] == "IMG00012"
        run = db.get(PipelineRun, response["run_id"])
        state = json.loads(run.stage_json)
        assert state["request"]["input_source"] == "DATASET"
        assert state["request"]["dataset_id"] == "DS_TEST"
        assert state["request"]["dataset_item_id"] == 1
        assert state["request"]["image_id"] == "IMG00012"
        assert stored["input"][1] == b"dataset-image"
    finally:
        db.close()


def test_qwen_image_edit_lab_direct_original_to_worker_and_records_run(tmp_path, monkeypatch):
    db = _session(tmp_path)
    try:
        stored = {}

        def fake_store(run_id, kind, data, media_type, extension):
            uri = "local://qwen-image-edit-lab/" + run_id + "/" + kind + extension
            stored[kind] = (uri, data, media_type)
            return uri

        def fake_read(uri, *, label):
            assert uri == "http://worker/output.png"
            assert label == "qwen_lab_output"
            return b"generated-image", "image/png"

        def fake_worker(**kwargs):
            assert kwargs["visible_fish_refined_image_uri"].startswith("local://qwen-image-edit-lab/")
            assert kwargs["prompt"] == "test prompt"
            assert kwargs["negative_prompt"] == "avoid fish change"
            assert kwargs["seed"] == 123
            return {
                "result_uri": "http://worker/output.png",
                "worker_model": "Qwen-Image-Edit-2511",
                "worker_status": "WORKER_EXECUTED",
                "worker_http_status": 200,
                "seed": 123,
                "elapsed_ms": 456,
            }

        monkeypatch.setattr(lab, "_store_bytes", fake_store)
        monkeypatch.setattr(lab, "_read_image_uri", fake_read)
        monkeypatch.setattr(lab, "invoke_qwen_refine_worker", fake_worker)

        response = asyncio.run(
            lab.generate_qwen_image_edit_lab(
                _upload(),
                prompt="test prompt",
                negative_prompt="avoid fish change",
                seed="123",
                db=db,
            )
        )

        assert response["status"] == "SUCCESS"
        assert response["model"] == "qwen-image-edit-2511"
        assert response["seed"] == 123
        assert response["time_ms"] == 456
        assert response["output_image_uri"] == stored["output"][0]

        run = db.get(PipelineRun, response["run_id"])
        assert run is not None
        assert run.pipeline_type == "QWEN_IMAGE_EDIT_LAB"
        assert run.model_version == "qwen-image-edit-2511"
        assert run.status == "SUCCESS"
        state = json.loads(run.stage_json)
        assert state["request"]["input_image_uri"] == stored["input"][0]
        assert state["result"]["output_image_uri"] == stored["output"][0]
        assert state["result"]["seed"] == 123
        assert state["stages"][-1]["status"] == "DONE"
        assert all("visible" not in json.dumps(item).lower() for item in state["stages"])
    finally:
        db.close()


def test_qwen_image_edit_lab_rejects_invalid_seed(tmp_path):
    db = _session(tmp_path)
    try:
        with pytest.raises(HTTPException) as error:
            asyncio.run(
                lab.generate_qwen_image_edit_lab(
                    _upload(),
                    prompt=lab.DEFAULT_PROMPT,
                    negative_prompt=lab.DEFAULT_NEGATIVE_PROMPT,
                    seed="not-an-integer",
                    db=db,
                )
            )
        assert error.value.status_code == 422
    finally:
        db.close()
