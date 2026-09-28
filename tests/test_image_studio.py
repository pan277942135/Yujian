from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from app.image_studio_worker_client import IMAGE_STUDIO_MODE, MAX_REFERENCES
from app.platform.routes.image_studio import PIPELINE_TYPE, _mask_composite
from app.platform.services.image_studio_prompt import compile_image_studio_prompt


def _png(size=(4, 4), color=(0, 0, 0), mode="RGB") -> bytes:
    image = Image.new(mode, size, color)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_image_studio_contract_is_separate_from_fish_lab():
    assert PIPELINE_TYPE == "IMAGE_STUDIO_V1"
    assert IMAGE_STUDIO_MODE == "image_studio_v1"
    assert MAX_REFERENCES == 2


def test_identity_lock_compiler_assigns_reference_authority():
    compiled = compile_image_studio_prompt(
        "Change the outfit to a dark coat.",
        mode="IDENTITY_LOCK",
        preservation="MAX",
        reference_roles=["IDENTITY", "OUTFIT"],
    )
    assert compiled.mode == "IDENTITY_LOCK"
    assert compiled.preservation == "MAX"
    assert compiled.reference_roles == ("IDENTITY", "OUTFIT")
    assert "Do not blend, average" in compiled.prompt
    assert "Picture 2 role = IDENTITY" in compiled.prompt
    assert "Picture 3 role = OUTFIT" in compiled.prompt
    assert "face averaging" in compiled.negative_prompt


def test_prompt_compiler_rejects_more_than_two_references():
    with pytest.raises(ValueError, match="at most two"):
        compile_image_studio_prompt(
            "test",
            reference_roles=["IDENTITY", "OUTFIT", "SCENE"],
        )


def test_mask_composite_restores_base_pixels_outside_mask():
    base = Image.new("RGB", (4, 4), (10, 20, 30))
    generated = Image.new("RGB", (4, 4), (200, 210, 220))
    mask = Image.new("L", (4, 4), 0)
    mask.putpixel((1, 1), 255)

    def encode(image):
        output = io.BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()

    result = Image.open(
        io.BytesIO(_mask_composite(encode(base), encode(generated), encode(mask)))
    ).convert("RGB")

    assert result.getpixel((0, 0)) == (10, 20, 30)
    assert result.getpixel((3, 3)) == (10, 20, 30)
    assert result.getpixel((1, 1)) == (200, 210, 220)


def test_worker_source_keeps_legacy_mode_and_adds_multi_reference_inputs():
    worker = (
        Path(__file__).resolve().parents[1]
        / "workers"
        / "fish-qwen-refine-worker"
        / "worker.py"
    ).read_text(encoding="utf-8")

    assert 'QWEN_MODE = "fish_preserve_refine_qwen_v1"' in worker
    assert 'IMAGE_STUDIO_MODE = "image_studio_v1"' in worker
    assert "SUPPORTED_MODES = {QWEN_MODE, IMAGE_STUDIO_MODE}" in worker
    assert "references: list[UploadFile] | None" in worker
    assert 'f"image{offset}"' in worker
    assert '"reference_count": len(reference_inputs)' in worker
