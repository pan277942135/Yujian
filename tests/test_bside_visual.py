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
from app.platform.models import BsideBackground, BsideVisualSession, BsideVisualStep, PipelineRun
from app.platform.routes import bside_visual as lab
from app.platform.services.bside_assets import seed_bside_asset_registry
from app.platform.services.bside_visual import outline, standardize
from app.platform.services.bside_visual.standardizer import _detect_head_direction
from app.platform.services.bside_visual.asset_registry import outline_renderer_style
from app.platform.services.bside_visual.style_registry import STYLES
from app.platform.services.bside_visual.water_renderer import compose_bside
from app.platform.services.bside_visual.template_registry import get_template


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


def _head_tail_fish_bytes(*, head_side: str = "left") -> bytes:
    """A deliberately asymmetric RGBA fish: broad rounded head, narrow tail."""

    image = Image.new("RGBA", (360, 180), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((28, 34, 144, 146), fill=(214, 155, 59, 255))
    draw.rectangle((106, 58, 276, 122), fill=(214, 155, 59, 255))
    draw.polygon([(270, 90), (334, 66), (334, 114)], fill=(214, 155, 59, 255))
    draw.ellipse((62, 72, 78, 88), fill=(18, 18, 18, 255))
    if head_side == "right":
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    output = io.BytesIO()
    image.save(output, format="PNG")
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


def _fake_transparent_artifacts(data: bytes):
    from app.platform.services.qwen_output import QwenOutputArtifacts

    return QwenOutputArtifacts(
        qwen_result_rgb=data,
        fish_mask_raw=b"raw-mask",
        fish_mask=b"final-mask",
        transparent_fish=_fish_bytes(),
        metadata={
            "mode": "RGBA",
            "channels": 4,
            "alpha_min": 0,
            "alpha_max": 255,
            "alpha_coverage": 0.25,
            "fish_interior_alpha_mean": 255.0,
            "fish_interior_alpha_median": 255.0,
            "fish_interior_opaque_ratio": 1.0,
        },
    )


def _local_store(tmp_path, session_id, step_key, version, filename, data, media_type):
    path = tmp_path / session_id / step_key / f"v{version}" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


def test_bside_routes_are_additive_and_options_are_registry_backed():
    paths = app.openapi()["paths"]
    assert "/api/qwen-lab/runs/{run_id}/bside-visual" in paths
    assert "/api/qwen-lab/bside-visual/{session_id}" in paths
    assert "/api/qwen-lab/bside-visual/{session_id}/extract-transparent" in paths
    assert "/api/qwen-lab/bside-visual/{session_id}/standardize" in paths
    assert "/api/qwen-lab/bside-visual/{session_id}/outline" in paths
    assert "/api/qwen-lab/bside-visual/{session_id}/compose" in paths
    options = lab.bside_visual_options()
    assert [item["style_id"] for item in options["styles"]] == [item.style_id for item in STYLES]
    assert options["source_contract"]["gpu_required"] is False
    assert [item["step"] for item in options["steps"]] == ["transparent", "standardize", "outline", "compose"]


def test_session_get_or_create_is_unique_and_accepts_qwen_rgb_source(tmp_path):
    db = _session(tmp_path)
    try:
        _add_qwen_run(db, tmp_path)
        first = lab.create_or_get_bside_visual("QWEN_TEST_RUN", db)
        second = lab.create_or_get_bside_visual("QWEN_TEST_RUN", db)
        assert first["created"] is True
        assert second["created"] is False
        assert first["session_id"] == second["session_id"]
        assert len(first["steps"]) == 4
        assert [asset["label"] for asset in first["assets"]] == ["Qwen Result · RGB", "透明背景鱼体", "标准姿态鱼", "特色描边鱼", "B面最终视觉"]
        assert first["transparent_status"] == "NOT_STARTED"
        assert first["source_qwen_rgb_uri"] == str(tmp_path / "fish.png")

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
        rgb_session = lab.create_or_get_bside_visual("QWEN_RGB_RUN", db)
        assert rgb_session["created"] is True
        assert rgb_session["transparent_status"] == "NOT_STARTED"
        assert rgb_session["source_qwen_rgb_uri"] == str(tmp_path / "rgb.png")
    finally:
        db.close()


def test_standardize_flips_confident_left_head_to_right_and_preserves_rgba_pixels():
    source = Image.open(io.BytesIO(_head_tail_fish_bytes(head_side="left"))).convert("RGBA")
    artifact = standardize(_head_tail_fish_bytes(head_side="left"))
    image = Image.open(io.BytesIO(artifact.data)).convert("RGBA")
    source_rgba = np.asarray(source, dtype=np.uint8)
    output_rgba = np.asarray(image, dtype=np.uint8)
    source_foreground = source_rgba[source_rgba[:, :, 3] > 0]
    output_foreground = output_rgba[output_rgba[:, :, 3] > 0]

    assert image.mode == "RGBA"
    assert artifact.metadata["head_side_before_flip"] == "left"
    assert artifact.metadata["head_direction_after"] == "right"
    assert artifact.metadata["head_confidence"] >= 0.75
    assert artifact.metadata["flip_horizontal"] is True
    assert artifact.metadata["flip_vertical"] is False
    assert artifact.metadata["direction_flipped"] is True
    assert artifact.metadata["pose_status"] == "PASS"
    assert sorted(map(tuple, output_foreground)) == sorted(map(tuple, source_foreground))
    assert artifact.metadata["output_width"] <= 1600
    assert output_rgba[:, :, 3].max() > 0


def test_standardize_keeps_confident_right_head_without_flip():
    artifact = standardize(_head_tail_fish_bytes(head_side="right"))

    assert artifact.metadata["head_side_before_flip"] == "right"
    assert artifact.metadata["head_direction_after"] == "right"
    assert artifact.metadata["head_confidence"] >= 0.75
    assert artifact.metadata["flip_horizontal"] is False
    assert artifact.metadata["flip_vertical"] is False
    assert artifact.metadata["pose_status"] == "PASS"


def test_head_direction_low_confidence_warns_without_forcing_flip():
    symmetric_mask = np.zeros((100, 240), dtype=bool)
    symmetric_mask[28:72, 20:220] = True
    detection = _detect_head_direction(symmetric_mask)

    assert detection["head_side"] == "unknown"
    assert detection["confidence"] < 0.75
    assert detection["reason"] == "HEAD_DIRECTION_LOW_CONFIDENCE"


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


def test_registry_outline_modes_are_local_and_none_is_a_successful_passthrough(tmp_path):
    from app.platform.models import BsideBackgroundOutlineProfile, BsideOutlineStyle

    db = _session(tmp_path)
    try:
        seed_bside_asset_registry(db)
        source = standardize(_fish_bytes()).data
        source_rgba = np.asarray(Image.open(io.BytesIO(source)).convert("RGBA"))
        for code in ("directional_rim", "bottom_water_glow", "none"):
            style_row = db.query(BsideOutlineStyle).filter_by(code=code).one()
            profile = db.query(BsideBackgroundOutlineProfile).filter_by(
                outline_style_id=style_row.id
            ).first()
            assert profile is not None
            style = outline_renderer_style(style_row, profile)
            result = outline(source, style)
            output_rgba = np.asarray(Image.open(io.BytesIO(result.data)).convert("RGBA"))
            mask = source_rgba[:, :, 3] > 0
            assert np.array_equal(output_rgba[mask], source_rgba[mask])
            if code == "none":
                assert result.metadata["effect_edge_ratio"] == 0.0
            else:
                assert 0.20 <= result.metadata["effect_edge_ratio"] <= 0.45
                assert result.metadata["outline_mode"] == code
    finally:
        db.close()


def test_compose_uses_formal_rgba_layers_in_locked_order_and_canvas():
    standardized = standardize(_fish_bytes()).data
    style = STYLES[0]
    outlined = outline(standardized, style).data
    background = Image.new("RGB", (1080, 1350), (30, 90, 88))
    light = Image.new("RGBA", (1080, 1350), (255, 255, 255, 48))
    foreground = Image.new("RGBA", (1080, 1350), (15, 45, 52, 24))

    def encode(image: Image.Image, image_format: str) -> bytes:
        output = io.BytesIO()
        image.save(output, format=image_format)
        return output.getvalue()

    rendered = compose_bside(
        standardized,
        style,
        get_template("lake_dawn_01"),
        outlined_fish=outlined,
        background_bytes=encode(background, "WEBP"),
        light_bytes=encode(light, "PNG"),
        foreground_bytes=encode(foreground, "PNG"),
    )
    result = Image.open(io.BytesIO(rendered["master"])).convert("RGBA")
    assert result.size == (1080, 1350)
    assert rendered["metadata"]["layer_order"] == [
        "Background",
        "Light",
        "Fish Depth Shadow",
        "outlined_fish_rgba",
        "Foreground",
    ]
    assert rendered["metadata"]["fish_opacity"] == 1.0
    assert rendered["metadata"]["fish_transform"] == "uniform_scale_translate"
    assert rendered["metadata"]["light_asset_used"] is True
    assert rendered["metadata"]["foreground_asset_used"] is True
    assert rendered["metadata"]["fish_width"] == round(1080 * 0.72)


def test_bside_session_starts_from_saved_qwen_rgb_without_auto_transparent_processing(tmp_path, monkeypatch):
    db = _session(tmp_path)
    try:
        _add_qwen_run(db, tmp_path)
        calls = []
        monkeypatch.setattr(lab, "process_qwen_output", lambda data: calls.append(data))
        session = lab.create_or_get_bside_visual("QWEN_TEST_RUN", db)
        assert calls == []
        assert session["progress"] == {"completed": 0, "total": 4}
        assert session["transparent_status"] == "NOT_STARTED"
        transparent = next(item for item in session["steps"] if item["step"] == "transparent")
        assert transparent["status"] == "NOT_STARTED"
        assert transparent["available"] is False
    finally:
        db.close()


def test_step_dependencies_and_rerun_invalidation(tmp_path, monkeypatch):
    db = _session(tmp_path)
    try:
        _add_qwen_run(db, tmp_path)
        monkeypatch.setattr(lab, "_store_bytes", lambda *args: _local_store(tmp_path, *args))
        monkeypatch.setattr(lab, "process_qwen_output", lambda data: _fake_transparent_artifacts(data))
        session = lab.create_or_get_bside_visual("QWEN_TEST_RUN", db)
        session_id = session["session_id"]
        with pytest.raises(HTTPException) as locked:
            lab.run_bside_outline(session_id, lab.OutlineRequest(style_id="lake_mist"), db)
        assert locked.value.status_code == 409
        with pytest.raises(HTTPException) as standardize_locked:
            lab.run_bside_standardize(session_id, lab.StandardizeRequest(), db)
        assert standardize_locked.value.status_code == 409
        transparent = lab.run_bside_transparent(session_id, db)
        assert transparent["transparent_status"] == "SUCCESS"
        lab.run_bside_standardize(session_id, lab.StandardizeRequest(), db)
        lab.run_bside_outline(session_id, lab.OutlineRequest(style_id="lake_mist"), db)
        composed = lab.run_bside_compose(session_id, lab.ComposeRequest(template_id="lake_dawn_01"), db)
        final_step = next(item for item in composed["steps"] if item["step"] == "compose")
        assert final_step["status"] == "SUCCESS"
        assert final_step["metadata"]["width"] == 1080
        assert final_step["metadata"]["height"] == 1350
        assert final_step["metadata"]["source_asset"] == "outlined_fish_rgba"
        assert final_step["metadata"]["outlined_uri"]
        rerun = lab.run_bside_outline(session_id, lab.OutlineRequest(style_id="soft_gold"), db)
        assert next(item for item in rerun["steps"] if item["step"] == "compose")["status"] == "STALE"
        assert next(item for item in rerun["steps"] if item["step"] == "standardize")["status"] == "SUCCESS"
        steps = db.query(BsideVisualStep).filter(BsideVisualStep.session_id == session_id).all()
        assert {row.step_key for row in steps} == {"transparent", "standardize", "outline", "compose"}
    finally:
        db.close()


def test_step3_and_step4_consume_one_persisted_registry_plan(tmp_path, monkeypatch):
    db = _session(tmp_path)
    try:
        _add_qwen_run(db, tmp_path)
        seed_bside_asset_registry(db)
        background = db.query(BsideBackground).filter_by(code="lake_dawn_01").one()

        def save_layer(filename: str, mode: str, image_format: str) -> str:
            image = Image.new(mode, (1080, 1350), (36, 96, 92, 255) if mode == "RGBA" else (36, 96, 92))
            path = tmp_path / filename
            image.save(path, format=image_format)
            return str(path)

        background.background_uri = save_layer("registry-background.webp", "RGB", "WEBP")
        background.foreground_uri = save_layer("registry-foreground.png", "RGBA", "PNG")
        background.light_uri = save_layer("registry-light.png", "RGBA", "PNG")
        background.status = "ACTIVE"
        db.commit()

        monkeypatch.setattr(lab, "_store_bytes", lambda *args: _local_store(tmp_path, *args))
        monkeypatch.setattr(lab, "process_qwen_output", lambda data: _fake_transparent_artifacts(data))
        session_payload = lab.create_or_get_bside_visual("QWEN_TEST_RUN", db)
        session = db.get(BsideVisualSession, session_payload["session_id"])
        assert session is not None
        session.style_seed = 12345
        db.commit()

        session_id = session.session_id
        lab.run_bside_transparent(session_id, db)
        lab.run_bside_standardize(session_id, lab.StandardizeRequest(), db)
        outlined = lab.run_bside_outline(session_id, lab.OutlineRequest(style_id="soft_gold"), db)
        first_plan = outlined["style_plan"]
        assert first_plan["background_code"] == "lake_dawn_01"
        assert first_plan["outline_code"] in {"directional_rim", "bottom_water_glow", "none"}

        rerun = lab.run_bside_outline(session_id, lab.OutlineRequest(style_id="mist_white"), db)
        assert rerun["style_plan"] == first_plan
        assert next(item for item in rerun["steps"] if item["step"] == "compose")["status"] == "NOT_STARTED"

        composed = lab.run_bside_compose(session_id, lab.ComposeRequest(template_id="night_fishing_01"), db)
        final_step = next(item for item in composed["steps"] if item["step"] == "compose")
        assert final_step["status"] == "SUCCESS"
        assert final_step["metadata"]["template_id"] == "lake_dawn_01"
        assert final_step["metadata"]["style_plan"] == first_plan
        assert final_step["metadata"]["layer_order"] == [
            "Background",
            "Light",
            "Fish Depth Shadow",
            "outlined_fish_rgba",
            "Foreground",
        ]
    finally:
        db.close()


def test_bside_page_contains_locked_four_step_ui_without_angle_input_or_gpu_dependency():
    template = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "templates"
        / "platform"
        / "lab"
        / "qwen_bside_visual.html"
    ).read_text(encoding="utf-8")
    assert "四步处理" in template
    assert "透明背景鱼体" in template
    assert "姿态标准化" in template
    assert "检测主轴" in template
    assert "鱼头方向" in template
    assert "水平翻转" in template
    assert "方向置信度" in template
    assert "特色描边" in template
    assert "融入水体背景" in template
    assert "Qwen Result · RGB" in template
    assert "data-asset" in template
    assert "棋盘格" in template
    assert "extract-transparent" in template
    assert "stepOrder = ['transparent','standardize','outline','compose']" in template
    assert "姿态微调" not in template
    assert "bsideRotationOffset" not in template
    assert "不自动镜像" not in template
    assert "三步处理" not in template
    assert "/api/qwen-lab/bside-visual/" in template
    assert "Detector" in template
    assert "GPU" not in template
    assert "bsideStyles" not in template
    assert "bsideTemplates" not in template
    assert "selectedStyle" not in template
    assert "selectedTemplate" not in template
    assert "背景 × 描边权重" in template
    assert "同一 Style Plan" in template
