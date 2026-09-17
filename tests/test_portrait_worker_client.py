from __future__ import annotations

import base64
import io
import urllib.error

import pytest

from app.portrait_worker_client import (
    INPAINT_DEFAULT_NEGATIVE_PROMPT,
    INPAINT_DEFAULT_PROMPT,
    PortraitWorkerError,
    _knowledge_media_object_name,
    _multipart_body,
    _read_image_uri,
    _result_uri,
    _worker_error,
    invoke_portrait_worker,
    invoke_portrait_inpaint_worker,
)


def test_read_image_uri_supports_data_url():
    encoded = base64.b64encode(b"png-bytes").decode("ascii")
    data, media_type = _read_image_uri(f"data:image/png;base64,{encoded}", label="source")
    assert data == b"png-bytes"
    assert media_type == "image/png"


def test_read_image_uri_supports_local_lab_uri(tmp_path, monkeypatch):
    image_path = tmp_path / "mask.png"
    image_path.write_bytes(b"mask-bytes")
    monkeypatch.chdir(tmp_path)
    data, media_type = _read_image_uri("local://mask.png", label="mask")
    assert data == b"mask-bytes"
    assert media_type == "image/png"


def test_multipart_body_contains_source_reference_and_params():
    body, content_type = _multipart_body(
        fields=[("task", "fish_portrait_generate"), ("params", '{"steps":25}')],
        files=[
            ("image", "source.png", b"A_BYTES", "image/png"),
            ("reference_image", "reference.png", b"B_BYTES", "image/png"),
        ],
    )
    assert content_type.startswith("multipart/form-data; boundary=")
    assert b'name="image"; filename="source.png"' in body
    assert b'name="reference_image"; filename="reference.png"' in body
    assert b"A_BYTES" in body
    assert b"B_BYTES" in body
    assert b'{"steps":25}' in body


def test_managed_cover_uri_maps_to_canonical_gcs_object():
    assert _knowledge_media_object_name(
        "/api/v1/fish/knowledge-media/grass_carp/cover/cover.webp"
    ) == "fish-assets/grass_carp/cover/cover.webp"


def test_result_uri_is_promoted_to_worker_url():
    assert _result_uri({"result_uri": "/output/result.png"}, "http://34.69.75.199:8000") == (
        "http://34.69.75.199:8000/output/result.png"
    )
    assert _result_uri({"result_path": "/output/result.png"}, "http://34.69.75.199:8000") == (
        "http://34.69.75.199:8000/output/result.png"
    )
    assert _result_uri({"image_url": "https://cdn.example/result.png"}, "http://worker") == (
        "https://cdn.example/result.png"
    )


def test_invoke_portrait_worker_transmits_dual_adapter_scales(monkeypatch):
    captured = {}

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"status":"success","result_uri":"/output/result.png"}'

    def _urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setenv("FISH_PORTRAIT_WORKER_URL", "http://worker")
    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    source = "data:image/png;base64," + base64.b64encode(b"A_BYTES").decode("ascii")
    reference = "data:image/png;base64," + base64.b64encode(b"B_BYTES").decode("ascii")

    result = invoke_portrait_worker(
        source_image_uri=source,
        reference_image_uri=reference,
        dataset_id="DS_PORTRAIT",
        source_item_id="1",
        reference_asset_id="REF_CRUCIAN",
        model="sdxl_ip_adapter",
        params={
            "source_scale": 0.9,
            "reference_scale": 0.15,
            "steps": 25,
            "width": 768,
            "height": 768,
        },
    )

    body = captured["request"].data
    assert b'"source_scale":0.9' in body
    assert b'"reference_scale":0.15' in body
    assert b'name="image"' in body
    assert b'name="reference_image"' in body
    assert result["result_uri"] == "http://worker/output/result.png"


def test_invoke_portrait_inpaint_worker_transmits_json_contract(monkeypatch):
    captured = {}

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"status":"success","result_uri":"/output/inpaint.png"}'

    def _urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setenv("FISH_PORTRAIT_WORKER_URL", "http://worker")
    monkeypatch.delenv("FISH_PORTRAIT_INPAINT_WORKER_PATH", raising=False)
    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    result = invoke_portrait_inpaint_worker(
        original_image_uri="gs://bucket/original.png",
        fish_mask_uri="gs://bucket/fish-mask.png",
        completion_mask_uri="gs://bucket/completion-mask.png",
        species="鲫鱼",
        prompt=None,
        negative_prompt=None,
        strength=0.25,
        steps=25,
        width=768,
        height=768,
        seed=12345,
    )

    body = captured["request"].data.decode("utf-8")
    assert captured["request"].full_url == "http://worker/portrait/generate"
    assert captured["request"].get_header("Content-type") == "application/json"
    assert '"mode":"fish_preserve_inpaint_v2"' in body
    assert '"fish_mask_uri":"gs://bucket/fish-mask.png"' in body
    assert '"completion_mask_uri":"gs://bucket/completion-mask.png"' in body
    assert '"strength":0.25' in body
    assert '"seed":12345' in body
    assert result["result_uri"] == "http://worker/output/inpaint.png"


def test_inpaint_default_prompts_are_stable():
    assert "preserve" not in INPAINT_DEFAULT_PROMPT.lower()
    assert "different fish species" in INPAINT_DEFAULT_NEGATIVE_PROMPT


@pytest.mark.parametrize(
    ("status_code", "error_code"),
    [
        (422, "PORTRAIT_WORKER_PARAMETER_ERROR"),
        (500, "PORTRAIT_WORKER_INFERENCE_ERROR"),
    ],
)
def test_worker_http_errors_are_typed(status_code, error_code):
    response = io.BytesIO(b"worker failed")
    error = urllib.error.HTTPError(
        "http://worker/portrait",
        status_code,
        "failed",
        {"Content-Type": "text/plain"},
        response,
    )
    with pytest.raises(PortraitWorkerError) as raised:
        raise _worker_error(error)
    assert raised.value.error_code == error_code
    assert raised.value.status_code == status_code

