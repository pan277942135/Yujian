"""HTTP client for the dedicated SDXL + IP-Adapter Fish Portrait worker.

The portrait worker is intentionally separate from the existing PowerPaint
completion worker. Cloud Run orchestrates the request and materializes the
Dataset/Fish Asset bytes; image generation runs on the configured GPU worker.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


MAX_IMAGE_BYTES = 50 * 1024 * 1024


class PortraitWorkerError(RuntimeError):
    """A typed, safe-to-display worker failure."""

    def __init__(self, error_code: str, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.error_code = error_code
        self.status_code = status_code


def _base_url() -> str:
    return os.getenv("FISH_PORTRAIT_WORKER_URL", "").strip().rstrip("/")


def _worker_path() -> str:
    path = os.getenv("FISH_PORTRAIT_WORKER_PATH", "/portrait").strip() or "/portrait"
    return path if path.startswith("/") else "/" + path


def _timeout(default: float = 900.0) -> float:
    try:
        return max(1.0, float(os.getenv("FISH_PORTRAIT_WORKER_TIMEOUT_SECONDS", str(default))))
    except (TypeError, ValueError):
        return default


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/json"}
    token = os.getenv("FISH_PORTRAIT_WORKER_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _json_response(response: Any) -> tuple[int, dict[str, Any]]:
    status_code = int(getattr(response, "status", 200))
    try:
        body = response.read()
    except Exception as exc:
        raise PortraitWorkerError(
            "PORTRAIT_WORKER_READ_FAILED",
            "Portrait worker response could not be read",
            status_code=status_code,
        ) from exc
    try:
        payload = json.loads(body.decode("utf-8")) if body else {}
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PortraitWorkerError(
            "PORTRAIT_WORKER_INVALID_JSON",
            "Portrait worker returned a non-JSON response",
            status_code=status_code,
        ) from exc
    if not isinstance(payload, dict):
        raise PortraitWorkerError(
            "PORTRAIT_WORKER_INVALID_RESPONSE",
            "Portrait worker response is not an object",
            status_code=status_code,
        )
    return status_code, payload


def check_portrait_worker() -> dict[str, Any]:
    """Return a small health DTO without exposing credentials."""

    base_url = _base_url()
    if not base_url:
        return {
            "endpoint_configured": False,
            "status": "WORKER_UNAVAILABLE",
            "reason": "worker_endpoint_not_configured",
            "worker_url": None,
        }
    request = urllib.request.Request(
        f"{base_url}/health",
        method="GET",
        headers=_headers(),
    )
    try:
        with urllib.request.urlopen(request, timeout=min(_timeout(30.0), 30.0)) as response:
            status_code, payload = _json_response(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:1000] or str(exc)
        raise PortraitWorkerError("PORTRAIT_WORKER_HEALTH_ERROR", detail, status_code=exc.code) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise PortraitWorkerError("PORTRAIT_WORKER_HEALTH_UNREACHABLE", str(exc), status_code=None) from exc
    return {
        "endpoint_configured": True,
        "status": "READY" if 200 <= status_code < 300 else "WORKER_UNAVAILABLE",
        "health_status": status_code,
        "health": payload,
        "worker_url": base_url,
    }


def _decode_data_url(value: Any) -> tuple[bytes, str] | None:
    if not isinstance(value, str) or not value.startswith("data:") or "," not in value:
        return None
    header, encoded = value.split(",", 1)
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError):
        return None
    media_type = header[5:].split(";", 1)[0].strip().lower() or "application/octet-stream"
    return data, media_type


def _guard_image_size(data: bytes, *, label: str) -> bytes:
    if not data:
        raise PortraitWorkerError(
            f"PORTRAIT_{label.upper()}_IMAGE_EMPTY",
            f"{label} image is empty",
        )
    if len(data) > MAX_IMAGE_BYTES:
        raise PortraitWorkerError(
            f"PORTRAIT_{label.upper()}_IMAGE_TOO_LARGE",
            f"{label} image exceeds the 50 MiB worker input limit",
        )
    return data


def _read_image_uri(uri: str, *, label: str) -> tuple[bytes, str]:
    """Read a source/reference URI into bytes for the multipart Worker API."""

    value = str(uri or "").strip()
    if not value:
        raise PortraitWorkerError(
            f"PORTRAIT_{label.upper()}_IMAGE_MISSING",
            f"{label} image URI is empty",
        )
    data_url = _decode_data_url(value)
    if data_url:
        return _guard_image_size(data_url[0], label=label), data_url[1]

    try:
        if value.startswith("gs://"):
            object_path = value[5:]
            bucket_name, object_name = object_path.split("/", 1)
            if not bucket_name or not object_name:
                raise ValueError("invalid gs URI")
            from google.cloud import storage

            data = storage.Client().bucket(bucket_name).blob(object_name).download_as_bytes()
            media_type = mimetypes.guess_type(object_name)[0] or "application/octet-stream"
        elif value.startswith(("http://", "https://")):
            request = urllib.request.Request(value, method="GET", headers={"Accept": "image/*"})
            with urllib.request.urlopen(request, timeout=min(_timeout(120.0), 120.0)) as response:
                data = response.read(MAX_IMAGE_BYTES + 1)
                media_type = response.headers.get_content_type() or mimetypes.guess_type(value)[0] or "application/octet-stream"
        elif value.startswith("/"):
            path = Path(value)
            if not path.is_file():
                raise FileNotFoundError(value)
            data = path.read_bytes()
            media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        else:
            raise ValueError("unsupported image URI")
    except PortraitWorkerError:
        raise
    except Exception as exc:
        raise PortraitWorkerError(
            f"PORTRAIT_{label.upper()}_IMAGE_READ_FAILED",
            f"Could not read {label} image for Portrait Worker",
        ) from exc
    return _guard_image_size(data, label=label), media_type


def _multipart_body(
    *,
    fields: list[tuple[str, str]],
    files: list[tuple[str, str, bytes, str]],
) -> tuple[bytes, str]:
    boundary = "----YuJianPortrait" + uuid.uuid4().hex
    chunks: list[bytes] = []
    for name, value in fields:
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )
    for name, filename, data, media_type in files:
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                (
                    f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                    f"Content-Type: {media_type}\r\n\r\n"
                ).encode("utf-8"),
                data,
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _worker_error(exc: urllib.error.HTTPError) -> PortraitWorkerError:
    detail = exc.read().decode("utf-8", "replace")[:2000] or f"HTTP {exc.code}"
    if exc.code == 422:
        code = "PORTRAIT_WORKER_PARAMETER_ERROR"
    elif exc.code >= 500:
        code = "PORTRAIT_WORKER_INFERENCE_ERROR"
    else:
        code = "PORTRAIT_WORKER_HTTP_ERROR"
    return PortraitWorkerError(code, detail, status_code=exc.code)


def _result_uri(result: dict[str, Any], base_url: str) -> str | None:
    value = (
        result.get("generated_image_uri")
        or result.get("result_uri")
        or result.get("output_uri")
        or result.get("generated_roi_uri")
        or result.get("image_url")
    )
    if value:
        return str(value).strip()
    path_value = result.get("result_path") or result.get("output_path") or result.get("path")
    if not path_value:
        return None
    path = str(path_value).strip()
    if path.startswith(("data:", "gs://", "http://", "https://")):
        return path
    if path.startswith("/"):
        return f"{base_url}{path}"
    return f"{base_url}/{path}"


def invoke_portrait_worker(
    *,
    source_image_uri: str,
    reference_image_uri: str,
    dataset_id: str,
    source_item_id: str,
    reference_asset_id: str,
    model: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Invoke the Worker with A/B multipart compatibility fields.

    The deployed VM currently documents a required "image" file. We keep
    that field as the source photo A, and attach "reference_image" plus
    semantic metadata so the same request is forwards-compatible with the
    A+B Worker contract.
    """

    base_url = _base_url()
    if not base_url:
        raise PortraitWorkerError(
            "PORTRAIT_WORKER_NOT_CONFIGURED",
            "FISH_PORTRAIT_WORKER_URL is not configured",
        )

    source_data, source_media_type = _read_image_uri(source_image_uri, label="source")
    reference_data, reference_media_type = _read_image_uri(reference_image_uri, label="reference")
    fields = [
        ("task", "fish_portrait_generate"),
        ("model", model),
        ("dataset_id", dataset_id),
        ("source_item_id", source_item_id),
        ("reference_asset_id", reference_asset_id),
        ("source_image_uri", source_image_uri),
        ("reference_image_uri", reference_image_uri),
        ("image_uri", source_image_uri),
        ("reference_uri", reference_image_uri),
        ("params", json.dumps(params, separators=(",", ":"))),
    ]
    body, content_type = _multipart_body(
        fields=fields,
        files=[
            ("image", "source_image", source_data, source_media_type),
            ("reference_image", "reference_image", reference_data, reference_media_type),
        ],
    )
    headers = _headers()
    headers["Content-Type"] = content_type
    request = urllib.request.Request(
        f"{base_url}{_worker_path()}",
        data=body,
        method="POST",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=_timeout()) as response:
            status_code, result = _json_response(response)
    except urllib.error.HTTPError as exc:
        raise _worker_error(exc) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise PortraitWorkerError("PORTRAIT_WORKER_CONNECTION_FAILED", str(exc)) from exc

    generated_image = (
        _decode_data_url(result.get("generated_image"))
        or _decode_data_url(result.get("generated"))
        or _decode_data_url(result.get("image"))
    )
    result_uri = _result_uri(result, base_url)
    if not result_uri and not generated_image:
        raise PortraitWorkerError(
            "PORTRAIT_WORKER_INVALID_RESPONSE",
            "Portrait worker response is missing generated_image_uri/result_uri/image_url",
            status_code=status_code,
        )
    result["result_uri"] = result_uri
    result["generated_image"] = (
        "data:" + generated_image[1] + ";base64," + base64.b64encode(generated_image[0]).decode("ascii")
        if generated_image
        else None
    )
    result["worker_status"] = "WORKER_EXECUTED"
    result["worker_http_status"] = status_code
    result["worker_protocol"] = {
        "request": "multipart/form-data",
        "source_field": "image",
        "reference_field": "reference_image",
        "source_bytes": len(source_data),
        "reference_bytes": len(reference_data),
    }
    return result


__all__ = [
    "PortraitWorkerError",
    "check_portrait_worker",
    "invoke_portrait_worker",
]
