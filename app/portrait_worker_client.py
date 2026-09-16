"""HTTP client for the dedicated SDXL + IP-Adapter Fish Portrait worker.

The portrait worker is intentionally separate from the existing PowerPaint
completion worker. Cloud Run only orchestrates the request; image generation
runs on the configured GPU worker.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from typing import Any


class PortraitWorkerError(RuntimeError):
    """A typed, safe-to-display worker failure."""

    def __init__(self, error_code: str, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.error_code = error_code
        self.status_code = status_code


def _base_url() -> str:
    return os.getenv("FISH_PORTRAIT_WORKER_URL", "").strip().rstrip("/")


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
        raise PortraitWorkerError("PORTRAIT_WORKER_HEALTH_UNREACHABLE", str(exc)) from exc
    return {
        "endpoint_configured": True,
        "status": "READY" if 200 <= status_code < 300 else "WORKER_UNAVAILABLE",
        "health_status": status_code,
        "health": payload,
    }


def _data_url(value: Any) -> str | None:
    if not isinstance(value, str) or not value.startswith("data:") or "," not in value:
        return None
    try:
        base64.b64decode(value.split(",", 1)[1], validate=True)
    except (ValueError, TypeError):
        return None
    return value


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
    """Invoke the explicit Fish Portrait worker contract.

    The aliases in the payload make the contract forwards-compatible with the
    first GPU worker implementation while keeping the semantic names stable.
    """

    base_url = _base_url()
    if not base_url:
        raise PortraitWorkerError(
            "PORTRAIT_WORKER_NOT_CONFIGURED",
            "FISH_PORTRAIT_WORKER_URL is not configured",
        )
    payload = json.dumps(
        {
            "task": "fish_portrait",
            "model": model,
            "dataset_id": dataset_id,
            "source_item_id": source_item_id,
            "reference_asset_id": reference_asset_id,
            "source_image_uri": source_image_uri,
            "reference_image_uri": reference_image_uri,
            "image_uri": source_image_uri,
            "reference_uri": reference_image_uri,
            "params": params,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    headers = _headers()
    headers["Content-Type"] = "application/json"
    path = os.getenv("FISH_PORTRAIT_WORKER_PATH", "/portrait").strip() or "/portrait"
    if not path.startswith("/"):
        path = "/" + path
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=payload,
        method="POST",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=_timeout()) as response:
            status_code, result = _json_response(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:2000] or str(exc)
        raise PortraitWorkerError("PORTRAIT_WORKER_HTTP_ERROR", detail, status_code=exc.code) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise PortraitWorkerError("PORTRAIT_WORKER_UNREACHABLE", str(exc)) from exc

    result_uri = (
        result.get("generated_image_uri")
        or result.get("result_uri")
        or result.get("output_uri")
        or result.get("generated_roi_uri")
    )
    generated_image = (
        _data_url(result.get("generated_image"))
        or _data_url(result.get("generated"))
        or _data_url(result.get("image"))
    )
    if not result_uri and not generated_image:
        raise PortraitWorkerError(
            "PORTRAIT_WORKER_INVALID_RESPONSE",
            "Portrait worker response is missing generated_image_uri/result_uri",
            status_code=status_code,
        )
    result["result_uri"] = result_uri
    result["generated_image"] = generated_image
    result["worker_status"] = "WORKER_EXECUTED"
    return result


__all__ = [
    "PortraitWorkerError",
    "check_portrait_worker",
    "invoke_portrait_worker",
]
