from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path

import pytest
from fastapi import HTTPException
from PIL import Image
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


def _generated_png() -> bytes:
    image = Image.new("RGB", (64, 48), (235, 240, 237))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _fake_qwen_artifacts(data: bytes):
    from app.platform.services.qwen_output import QwenOutputArtifacts

    return QwenOutputArtifacts(
        qwen_result_rgb=data,
        fish_mask_raw=b"raw-mask",
        fish_mask=b"final-mask",
        transparent_fish=b"rgba-png",
        metadata={
            "mode": "RGBA",
            "channels": 4,
            "alpha_min": 0,
            "alpha_max": 255,
            "alpha_coverage": 0.25,
        },
    )


def test_qwen_image_edit_lab_routes_are_additive():
    paths = app.openapi()["paths"]
    assert "/fish-portrait/qwen-lab" in paths
    assert "/api/fish-portrait/qwen-lab/generate" in paths
    assert "/api/fish-portrait/qwen-lab/runs" in paths
    assert "/api/fish-portrait/qwen-lab/runs/{run_id}" in paths
    assert "/api/fish-portrait/qwen-lab/runs/{run_id}/media/{kind}" in paths
    assert "/api/fish-portrait/qwen-lab/runs/{run_id}/extract-transparent" in paths
    assert lab.PIPELINE_TYPE == "QWEN_IMAGE_EDIT_LAB"
    assert lab.MODEL_ID == "qwen-image-edit-2511"


def test_qwen_image_edit_lab_defaults_preserve_original_fish():
    assert "最高优先级：必须保留原图中的同一条真实鱼" in lab.DEFAULT_PROMPT
    assert "原图中已经清晰可见的鱼体区域保持不变" in lab.DEFAULT_PROMPT
    assert "输出透明背景的真实鱼体资产" in lab.DEFAULT_PROMPT
    assert "new fish" in lab.DEFAULT_NEGATIVE_PROMPT
    assert "different fish species" in lab.DEFAULT_NEGATIVE_PROMPT
    assert "cropped fish" in lab.DEFAULT_NEGATIVE_PROMPT


def test_qwen_image_edit_lab_template_uses_asset_prompt_defaults():
    template = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "templates"
        / "platform"
        / "lab"
        / "qwen_image_edit.html"
    ).read_text(encoding="utf-8")

    assert "最高优先级：必须保留原图中的同一条真实鱼" in template
    assert "输出透明背景的真实鱼体资产" in template
    assert "new fish," in template
    assert "3D render" in template


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

        payload = lab.qwen_image_edit_lab_runs(page=1, size=10, db=db)

        assert [row["run_id"] for row in payload["items"]] == ["QWEN_HISTORY_1"]
        assert payload["total"] == 1
        assert payload["page"] == 1
        assert payload["size"] == 10
        assert payload["has_next"] is False
        assert payload["items"][0]["input_image_url"].endswith("/runs/QWEN_HISTORY_1/media/input")
        assert payload["items"][0]["output_image_url"].endswith("/runs/QWEN_HISTORY_1/media/output")
        assert payload["items"][0]["time_ms"] == 321
    finally:
        db.close()


def test_qwen_image_edit_lab_history_paginates(tmp_path):
    db = _session(tmp_path)
    try:
        for index in range(11):
            db.add(
                PipelineRun(
                    run_id=f"QWEN_PAGE_{index:02d}",
                    pipeline_type=lab.PIPELINE_TYPE,
                    status="SUCCESS",
                    stage_json="{}",
                )
            )
        db.commit()

        first = lab.qwen_image_edit_lab_runs(page=1, size=10, db=db)
        second = lab.qwen_image_edit_lab_runs(page=2, size=10, db=db)

        assert len(first["items"]) == 10
        assert first["total"] == 11
        assert first["page_count"] == 2
        assert first["has_next"] is True
        assert len(second["items"]) == 1
        assert second["page"] == 2
        assert second["has_next"] is False
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
    assert "/items?page=" in template
    assert "DATASET_PAGE_SIZE = 10" in template
    assert "HISTORY_PAGE_SIZE = 10" in template
    assert 'id="qwenLabDatasetPager"' in template
    assert 'id="qwenLabHistoryPager"' in template
    assert "URLSearchParams" in template
    assert "dataset_item_id" in template
    assert "selectedDatasetSource" in template
    assert "new File([blob]" not in template
    assert 'id="qwenLabTransparentFishButton"' in template
    assert "/extract-transparent" in template
    assert "transparent_status" in template
    assert "B面视觉生成" in template
    generate_handler_start = template.index("generateButton.addEventListener")
    generate_handler_end = template.index("datasetSelect.addEventListener")
    generate_handler = template[generate_handler_start:generate_handler_end]
    assert "/extract-transparent" not in generate_handler
    assert "bside-visual" not in generate_handler


def test_qwen_image_edit_lab_template_inline_script_has_no_doubled_string_terminator():
    template = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "templates"
        / "platform"
        / "lab"
        / "qwen_image_edit.html"
    ).read_text(encoding="utf-8")

    # A duplicated quote here prevents every page initializer from executing.
    assert "</div>'' +" not in template


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
        processed = []

        def fake_store(run_id, kind, data, media_type, extension):
            uri = "local://qwen-image-edit-lab/" + run_id + "/" + kind + extension
            stored[kind] = (uri, data, media_type)
            return uri

        def fake_managed(uri):
            if uri == "gs://bucket/IMG00012.jpg":
                return b"dataset-image", "image/jpeg"
            if uri.endswith("/qwen_result_rgb.png"):
                return _generated_png(), "image/png"
            raise AssertionError(f"unexpected managed URI: {uri}")

        def fake_read(uri, *, label):
            assert uri == "http://worker/output.png"
            assert label == "qwen_lab_output"
            return _generated_png(), "image/png"

        def fake_process(data):
            processed.append(data)
            return _fake_qwen_artifacts(data)

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
        monkeypatch.setattr(lab, "process_qwen_output", fake_process)
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
        assert response["qwen_status"] == "SUCCESS"
        assert response["transparent_status"] == "NOT_STARTED"
        assert response["transparent_asset_status"] == "NOT_STARTED"
        assert response["transparent_fish_uri"] is None
        assert "qwen_fish_rgba" not in stored
        assert processed == []
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
        assert state["result"]["transparent_status"] == "NOT_STARTED"
        assert all(item["name"] != "transparent_fish_export" for item in state["stages"])
        assert stored["input"][1] == b"dataset-image"

        transparent_response = lab.extract_qwen_image_edit_lab_transparent(
            response["run_id"],
            db=db,
        )

        assert transparent_response["qwen_status"] == "SUCCESS"
        assert transparent_response["transparent_status"] == "SUCCESS"
        assert transparent_response["transparent_fish_uri"] == stored["qwen_fish_rgba"][0]
        assert transparent_response["fish_mask_uri"] == stored["qwen_fish_mask"][0]
        assert len(processed) == 1
    finally:
        db.close()


def test_qwen_image_edit_lab_direct_original_to_worker_and_records_run(tmp_path, monkeypatch):
    db = _session(tmp_path)
    try:
        stored = {}
        worker_calls = []
        processed = []

        def fake_store(run_id, kind, data, media_type, extension):
            uri = "local://qwen-image-edit-lab/" + run_id + "/" + kind + extension
            stored[kind] = (uri, data, media_type)
            return uri

        def fake_managed(uri):
            assert uri.endswith("/qwen_result_rgb.png")
            return _generated_png(), "image/png"

        def fake_read(uri, *, label):
            assert uri == "http://worker/output.png"
            assert label == "qwen_lab_output"
            return _generated_png(), "image/png"

        def fake_process(data):
            processed.append(data)
            return _fake_qwen_artifacts(data)

        def fake_worker(**kwargs):
            worker_calls.append(kwargs)
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
        monkeypatch.setattr(lab, "_read_managed_uri", fake_managed)
        monkeypatch.setattr(lab, "_read_image_uri", fake_read)
        monkeypatch.setattr(lab, "process_qwen_output", fake_process)
        monkeypatch.setattr(lab, "invoke_qwen_refine_worker", fake_worker)

        qwen_response = asyncio.run(
            lab.generate_qwen_image_edit_lab(
                _upload(),
                prompt="test prompt",
                negative_prompt="avoid fish change",
                seed="123",
                db=db,
            )
        )

        assert qwen_response["status"] == "SUCCESS"
        assert qwen_response["qwen_status"] == "SUCCESS"
        assert qwen_response["transparent_status"] == "NOT_STARTED"
        assert qwen_response["transparent_fish_uri"] is None
        assert "qwen_fish_rgba" not in stored
        assert processed == []
        assert len(worker_calls) == 1

        transparent_response = lab.extract_qwen_image_edit_lab_transparent(
            qwen_response["run_id"],
            db=db,
        )

        assert transparent_response["status"] == "SUCCESS"
        assert transparent_response["qwen_status"] == "SUCCESS"
        assert transparent_response["transparent_status"] == "SUCCESS"
        assert transparent_response["transparent_asset_status"] == "SUCCESS"
        assert transparent_response["output_image_uri"] == stored["qwen_result_rgb"][0]
        assert transparent_response["transparent_fish_uri"] == stored["qwen_fish_rgba"][0]
        assert transparent_response["fish_mask_uri"] == stored["qwen_fish_mask"][0]
        assert len(worker_calls) == 1
        assert len(processed) == 1

        run = db.get(PipelineRun, qwen_response["run_id"])
        assert run is not None
        assert run.pipeline_type == "QWEN_IMAGE_EDIT_LAB"
        assert run.model_version == "qwen-image-edit-2511"
        assert run.status == "SUCCESS"
        state = json.loads(run.stage_json)
        assert state["request"]["input_image_uri"] == stored["input"][0]
        assert state["result"]["output_image_uri"] == stored["qwen_result_rgb"][0]
        assert state["result"]["qwen_status"] == "SUCCESS"
        assert state["result"]["transparent_status"] == "SUCCESS"
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
