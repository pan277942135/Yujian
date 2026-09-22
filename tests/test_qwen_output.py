from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.platform.services import qwen_output


def _rgb_source() -> tuple[bytes, np.ndarray]:
    image = Image.new("RGB", (96, 64), (236, 240, 238))
    draw = ImageDraw.Draw(image)
    draw.ellipse((14, 18, 78, 48), fill=(205, 143, 51))
    draw.polygon([(76, 33), (91, 22), (91, 44)], fill=(205, 143, 51))
    draw.polygon([(42, 19), (52, 7), (58, 20)], fill=(205, 143, 51))
    draw.ellipse((24, 27, 29, 32), fill=(20, 20, 20))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue(), np.asarray(image)


def _run_with_mask(monkeypatch, mask: np.ndarray):
    data, source = _rgb_source()
    monkeypatch.setattr(
        qwen_output,
        "_resolve_bbox",
        lambda _source: (
            qwen_output.BBox(0.05, 0.05, 0.95, 0.95),
            {"status": "PASS", "model": "DET_FISH_v0.1"},
        ),
    )
    monkeypatch.setattr(
        qwen_output,
        "generate_fish_cutout",
        lambda _source, _bbox: SimpleNamespace(mask=mask),
    )
    return qwen_output.process_qwen_output(data), source


def test_qwen_result_resegmentation_exports_rgba_and_preserves_rgb(monkeypatch):
    mask = np.zeros((64, 96), dtype=bool)
    mask[18:49, 14:79] = True
    artifacts, source = _run_with_mask(monkeypatch, mask)

    with Image.open(io.BytesIO(artifacts.qwen_result_rgb)) as rgb_image:
        assert rgb_image.mode == "RGB"
        rgb = np.asarray(rgb_image)

    with Image.open(io.BytesIO(artifacts.fish_mask_raw)) as raw:
        assert raw.mode == "L"
        assert np.asarray(raw).max() == 255

    with Image.open(io.BytesIO(artifacts.fish_mask)) as final:
        assert final.mode == "L"
        final_mask = np.asarray(final) > 0
        assert final_mask.shape == mask.shape

    with Image.open(io.BytesIO(artifacts.transparent_fish)) as transparent:
        assert transparent.format == "PNG"
        assert transparent.mode == "RGBA"
        rgba = np.asarray(transparent)

    assert rgba.shape == (64, 96, 4)
    assert int(rgba[:, :, 3].min()) == 0
    assert int(rgba[:, :, 3].max()) == 255
    assert artifacts.metadata["channels"] == 4
    assert artifacts.metadata["mode"] == "RGBA"
    assert artifacts.metadata["alpha_coverage"] > 0
    assert artifacts.metadata["transparent_pixel_ratio"] > 0
    assert artifacts.metadata["fish_interior_alpha_mean"] == pytest.approx(255.0)
    assert artifacts.metadata["fish_interior_alpha_median"] == pytest.approx(255.0)
    assert artifacts.metadata["fish_interior_opaque_ratio"] == pytest.approx(1.0)
    assert np.array_equal(rgba[mask & (rgba[:, :, 3] == 255), :3], source[mask & (rgba[:, :, 3] == 255)])
    assert np.all(rgba[rgba[:, :, 3] == 0, :3] == 0)


def test_qwen_result_resegmentation_rejects_empty_mask(monkeypatch):
    mask = np.zeros((64, 96), dtype=bool)
    with pytest.raises(qwen_output.QwenOutputError) as error:
        _run_with_mask(monkeypatch, mask)
    assert error.value.error_code == "QWEN_SEGMENTATION_NO_FISH"


def test_qwen_result_resegmentation_rejects_full_frame_mask(monkeypatch):
    mask = np.ones((64, 96), dtype=bool)
    with pytest.raises(qwen_output.QwenOutputError) as error:
        _run_with_mask(monkeypatch, mask)
    assert error.value.error_code == "QWEN_ALPHA_FULL_FRAME"


def test_detector_no_fish_uses_one_sam_fallback_bbox(monkeypatch):
    def fail_detector(_image):
        raise RuntimeError("detector unavailable")

    monkeypatch.setattr(qwen_output, "detect", fail_detector)
    bbox, info = qwen_output._resolve_bbox(Image.new("RGB", (96, 64), "white"))

    assert info["status"] == "FALLBACK"
    assert info["reason"] == "detector_unavailable"
    assert bbox.x1 == pytest.approx(0.06)
    assert bbox.y2 == pytest.approx(0.94)
