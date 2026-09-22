from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException
from PIL import Image, ImageDraw
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.entry import app
from app.platform.models import BsideVisualStep, PipelineRun
from app.platform.routes import bside_visual as lab
from app.platform.services.bside_visual import outline, standardize
from app.platform.services.bside_visual.style_registry import STYLES


def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'bside.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)()


def _fish_bytes(*, transparent: bool = True) -> bytes:
    mode = "RGBA" if transparent else "RGB"
    image = Image.new(mode, (360, 220), (0, 0, 0, 0) if transparent else (240, 240, 240))
    draw = ImageDraw.Draw(image)
    fill = (214, 155, 59, 255) if transparent else (214, 155, 59)
    draw.ellipse((42, 72, 300, 148), fill=fill)
    draw.polygon([(298, 110), (344, 70), (344, 150)], fill=fill)
    draw.ellipse((78, 93, 91, 106), fill=(20, 20, 20, 255) if transparent else (20, 20, 20))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _slanted_fish_bytes(angle: float) -> bytes:
    image = Image.new("RGBA", (420, 260), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    body = (214, 155, 59, 255)
    draw.ellipse((82, 105, 316, 155), fill=body)
    draw.polygon([(310, 130), (384, 82), (384, 178)], fill=body)
    draw.polygon([(182, 108), (228, 64), (260, 108)], fill=(54, 132, 151, 255))
    draw.ellipse((106, 119, 120, 133), fill=(18, 18, 18, 255))
    rotated = image.rotate(angle, resample=Image.Resampling.NEAREST, expand=True, fillcolor=(0, 0, 0, 0))
    output = io.BytesIO()
    rotated.save(output, format="PNG")
    return output.getvalue()


def _axis_angle(image: Image.Image) -> float:
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    yx = np.column_stack(np.nonzero(rgba[:, :, 3] >= 16))
    xy = yx[:, [1, 0]].astype(np.float64)
    xy -= xy.mean(axis=0, keepdims=True)
    covariance = np.cov(xy, rowvar=False, bias=True)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    vector = eigenvectors[:, int(np.argmax(eigenvalues))]
    angle = float(np.degrees(np.arctan2(vector[1], vector[0])))
    return ((angle + 90.0) % 180.0) - 90.0


def _add_qwen_run(db, tmp_path, *, transparent=True):
    source = tmp_path / ("fish.png" if transparent else "rgb.png")
    source.write_bytes(_fish_bytes(transparent=transparent))
    db.add(
        PipelineRun(
            run_id="QWEN_TEST_RUN",
            pipeline_type="QWEN_IMAGE_EDIT_LAB",
            status="SUCCESS",
            stage_json=json.dumps({"result": {"output_image_uri": str(source)}}),
        )
    )
    db.commit()
    return source


def _local_store(tmp_path, session_id, step_key, version, filename, data, media_type):
    path = tmp_path / session_id / step_key / f"v{version}" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


def test_bside_routes_are_additive_and_options_are_registry_backed():
    paths = app.openapi()["paths"]
    assert "/api/qwen-lab/runs/{run_id}/bside-visual" in paths
    assert "/api/qwen-lab/bside-visual/{session_id}" in paths
    assert "/api/qwen-lab/bside-visual/{session_id}/standardize" in paths
    assert "/api/qwen-lab/bside-visual/{session_id}/outline" in paths
    assert "/api/qwen-lab/bside-visual/{session_id}/compose" in paths
    options = lab.bside_visual_options()
    assert [item["style_id"] for item in options["styles"]] == [item.style_id for item in STYLES]
    assert options["source_contract"]["gpu_required"] is False


def test_session_get_or_create_is_unique_and_invalid_source_is_rejected(tmp_path):
    db = _session(tmp_path)
    try:
        _add_qwen_run(db, tmp_path)
        first = lab.create_or_get_bside_visual("QWEN_TEST_RUN", db)
        second = lab.create_or_get_bside_visual("QWEN_TEST_RUN", db)
        assert first["created"] is True
        assert second["created"] is False
        assert first["session_id"] == second["session_id"]
        assert len(first["steps"]) == 3
        assert [asset["label"] for asset in first["assets"]] == ["Qwen完整鱼体", "标准姿态鱼", "特色描边鱼", "B面最终视觉"]

        db.add(
            PipelineRun(
                run_id="QWEN_RGB_RUN",
                pipeline_type="QWEN_IMAGE_EDIT_LAB",
                status="SUCCESS",
                stage_json=json.dumps({"result": {"output_image_uri": str(tmp_path / "rgb.png")}}),
            )
        )
        (tmp_path / "rgb.png").write_bytes(_fish_bytes(transparent=False))
        db.commit()
        with pytest.raises(HTTPException) as error:
            lab.create_or_get_bside_visual("QWEN_RGB_RUN", db)
        assert error.value.status_code == 422
        assert error.value.detail["error"] == "INVALID_TRANSPARENT_FISH"
    finally:
        db.close()


def test_standardize_preserves_alpha_and_does_not_flip_direction():
    artifact = standardize(_fish_bytes())
    image = Image.open(io.BytesIO(artifact.data)).convert("RGBA")
    assert image.mode == "RGBA"
    assert artifact.metadata["direction_flipped"] is False
    assert artifact.metadata["output_width"] <= 1600
    assert np.asarray(image)[:, :, 3].max() > 0


@pytest.mark.parametrize("source_angle", [60.0, -45.0, 0.0])
def test_standardize_auto_aligns_rgba_and_preserves_real_fish_rgb(source_angle):
    source = Image.open(io.BytesIO(_slanted_fish_bytes(source_angle))).convert("RGBA")
    before = _axis_angle(source)
    artifact = standardize(_slanted_fish_bytes(source_angle), manual_rotation_offset_deg=0.0)
    output = Image.open(io.BytesIO(artifact.data)).convert("RGBA")
    after = _axis_angle(output)
    rgba = np.asarray(output, dtype=np.uint8)
    foreground = rgba[rgba[:, :, 3] >= 16, :3]

    assert output.mode == "RGBA"
    assert abs(before - artifact.metadata["detected_axis_angle_deg"]) <= 2.0
    assert abs(after) <= 2.0
    assert abs(artifact.metadata["residual_axis_angle_deg"]) <= 2.0
    assert artifact.metadata["pose_validation"] == "PASS"
    assert artifact.metadata["rgb_preserved_inside_fish"] is True
    assert artifact.metadata["transparent_background"] is True
    assert float(np.var(foreground.astype(np.float32), axis=0).mean()) > 1.0
    assert not np.all(foreground == 255)
    assert int(rgba[:, :, 3].max()) > 0
    assert int(rgba[:, :, 3].min()) == 0


def test_standardize_rejects_white_mask_as_formal_fish_asset():
    image = Image.new("RGBA", (220, 120), (0, 0, 0, 0))
    ImageDraw.Draw(image).ellipse((24, 40, 196, 80), fill=(255, 255, 255, 255))
    output = io.BytesIO()
    image.save(output, format="PNG")
    with pytest.raises(ValueError) as error:
        standardize(output.getvalue())
    assert getattr(error.value, "code", None) == "POSE_RGBA_EXPORT_FAILED"


def test_standardize_manual_offset_is_added_to_auto_rotation():
    artifact = standardize(_slanted_fish_bytes(60.0), manual_rotation_offset_deg=5.0)
    metadata = artifact.metadata
    assert metadata["manual_rotation_offset_deg"] == 5.0
    assert metadata["applied_rotation_deg"] == pytest.approx(
        metadata["auto_rotation_deg"] + 5.0,
        abs=0.01,
    )
    assert metadata["expected_residual_axis_angle_deg"] == -5.0
    assert metadata["residual_axis_error_deg"] <= 2.0


def test_all_outline_styles_preserve_source_fish_pixels():
    standardized = standardize(_fish_bytes()).data
    source = np.asarray(Image.open(io.BytesIO(standardized)).convert("RGBA"))
    for style in STYLES:
        result = outline(standardized, style)
        output = np.asarray(Image.open(io.BytesIO(result.data)).convert("RGBA"))
        mask = source[:, :, 3] > 0
        assert np.array_equal(output[mask], source[mask])
        assert result.metadata["source_rgb_preserved"] is True


def test_step_dependencies_and_rerun_invalidation(tmp_path, monkeypatch):
    db = _session(tmp_path)
    try:
        _add_qwen_run(db, tmp_path)
        monkeypatch.setattr(lab, "_store_bytes", lambda *args: _local_store(tmp_path, *args))
        session = lab.create_or_get_bside_visual("QWEN_TEST_RUN", db)
        session_id = session["session_id"]
        with pytest.raises(HTTPException) as locked:
            lab.run_bside_outline(session_id, lab.OutlineRequest(style_id="lake_mist"), db)
        assert locked.value.status_code == 409
        lab.run_bside_standardize(session_id, lab.StandardizeRequest(), db)
        lab.run_bside_outline(session_id, lab.OutlineRequest(style_id="lake_mist"), db)
        composed = lab.run_bside_compose(session_id, lab.ComposeRequest(template_id="lake_dawn_01"), db)
        final_step = next(item for item in composed["steps"] if item["step"] == "compose")
        assert final_step["status"] == "COMPLETE"
        assert final_step["metadata"]["width"] == 1080
        assert final_step["metadata"]["height"] == 1350
        assert final_step["metadata"]["source_asset"] == "outlined_fish_rgba"
        assert final_step["metadata"]["outlined_uri"]
        rerun = lab.run_bside_outline(session_id, lab.OutlineRequest(style_id="soft_gold"), db)
        assert next(item for item in rerun["steps"] if item["step"] == "compose")["status"] == "STALE"
        assert next(item for item in rerun["steps"] if item["step"] == "standardize")["status"] == "COMPLETE"
        steps = db.query(BsideVisualStep).filter(BsideVisualStep.session_id == session_id).all()
        assert {row.step_key for row in steps} == {"standardize", "outline", "compose"}
    finally:
        db.close()


def test_bside_page_contains_locked_three_step_ui_and_no_gpu_dependency():
    template = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "templates"
        / "platform"
        / "lab"
        / "qwen_bside_visual.html"
    ).read_text(encoding="utf-8")
    assert "姿态标准化" in template
    assert "姿态微调" in template
    assert "检测主轴" in template
    assert "特色描边" in template
    assert "融入水体背景" in template
    assert "Qwen完整鱼体" in template
    assert "data-asset" in template
    assert "棋盘格" in template
    assert "/api/qwen-lab/bside-visual/" in template
    assert "Detector" in template
    assert "GPU" not in template
