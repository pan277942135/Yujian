from __future__ import annotations

import base64
import io
import urllib.error

import pytest

from app.portrait_worker_client import (
    PortraitWorkerError,
    _knowledge_media_object_name,
    _multipart_body,
    _read_image_uri,
    _result_uri,
    _worker_error,
)


def test_read_image_uri_supports_data_url():
    encoded = base64.b64encode(b"png-bytes").decode("ascii")
    data, media_type = _read_image_uri(f"data:image/png;base64,{encoded}", label="source")
    assert data == b"png-bytes"
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
