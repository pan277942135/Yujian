"""Strict HTTP client for the experimental GPU Completion Worker."""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from typing import Any


class CompletionWorkerError(RuntimeError):
    def __init__(self, error_code: str, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.error_code = error_code
        self.status_code = status_code


def _request_headers() -> dict[str, str]:
    headers = {"Accept": "application/json"}
    token = os.getenv("FISH_COMPLETION_WORKER_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _worker_base_url() -> str:
    return os.getenv("FISH_COMPLETION_WORKER_URL", "").strip().rstrip("/")


def _timeout(default: float = 900.0) -> float:
    try:
        return max(1.0, float(os.getenv("FISH_COMPLETION_WORKER_TIMEOUT_SECONDS", str(default))))
    except ValueError:
        return default


def check_completion_worker() -> dict[str, Any]:
    base_url = _worker_base_url()
    if not base_url:
        return {
            "endpoint_configured": False,
            "status": "WORKER_UNAVAILABLE",
            "reason": "worker_endpoint_not_configured",
        }
    req = urllib.request.Request(
        f"{base_url}/health",
        method="GET",
        headers=_request_headers(),
    )
    try:
        with urllib.request.urlopen(req, timeout=min(_timeout(30.0), 30.0)) as response:
            body = response.read()
            status_code = response.status
    except urllib.error.HTTPError as exc:
        raise CompletionWorkerError(
            "COMPLETION_WORKER_HEALTH_ERROR",
            exc.read().decode("utf-8", "replace")[:2000] or str(exc),
            status_code=exc.code,
        ) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise CompletionWorkerError("COMPLETION_WORKER_HEALTH_UNREACHABLE", str(exc)) from exc
    try:
        payload = json.loads(body.decode("utf-8")) if body else {}
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompletionWorkerError(
            "COMPLETION_WORKER_HEALTH_INVALID_JSON",
            "Worker health returned non-JSON",
            status_code=status_code,
        ) from exc
    return {
        "endpoint_configured": True,
        "status": "READY" if 200 <= status_code < 300 else "WORKER_UNAVAILABLE",
        "health_status": status_code,
        "health": payload if isinstance(payload, dict) else {},
    }


def _decode_data_url(value: str) -> bytes | None:
    if not isinstance(value, str) or not value.startswith("data:"):
        return None
    try:
        return base64.b64decode(value.split(",", 1)[1])
    except (IndexError, ValueError):
        return None


def invoke_completion_worker(
    *,
    image_uri: str,
    mask_uri: str,
    prompt: str,
    task: str = "fish_completion",
) -> dict[str, Any]:
    base_url = _worker_base_url()
    if not base_url:
        raise CompletionWorkerError(
            "COMPLETION_WORKER_NOT_CONFIGURED",
            "FISH_COMPLETION_WORKER_URL is not configured",
        )

    # Keep URI names for the existing Worker contract and also expose the
    # semantic ROI names required by V0.2.
    payload = json.dumps(
        {
            "task": task,
            "image_uri": image_uri,
            "mask_uri": mask_uri,
            "image_roi": image_uri,
            "mask_roi": mask_uri,
            "image": image_uri,
            "mask": mask_uri,
            "prompt": prompt,
        },
        separators=(",", ":"),
    ).encode()
    headers = _request_headers()
    headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        f"{base_url}/completion",
        data=payload,
        method="POST",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=_timeout()) as response:
            status_code = response.status
            body = response.read()
    except urllib.error.HTTPError as exc:
        raise CompletionWorkerError(
            "COMPLETION_WORKER_HTTP_ERROR",
            exc.read().decode("utf-8", "replace")[:2000] or str(exc),
            status_code=exc.code,
        ) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise CompletionWorkerError("COMPLETION_WORKER_UNREACHABLE", str(exc)) from exc
    try:
        result = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompletionWorkerError(
            "COMPLETION_WORKER_INVALID_JSON",
            "Worker returned a non-JSON response",
            status_code=status_code,
        ) from exc
    if not isinstance(result, dict):
        raise CompletionWorkerError(
            "COMPLETION_WORKER_INVALID_RESPONSE",
            "Worker response is not an object",
            status_code=status_code,
        )

    result_uri = (
        result.get("result_uri")
        or result.get("generated_roi_uri")
        or result.get("output_uri")
    )
    generated_roi = result.get("generated_roi")
    if not result_uri and not _decode_data_url(generated_roi or ""):
        raise CompletionWorkerError(
            "COMPLETION_WORKER_INVALID_RESPONSE",
            "Worker response is missing generated_roi/result_uri",
            status_code=status_code,
        )
    if not result.get("model_version"):
        raise CompletionWorkerError(
            "COMPLETION_WORKER_INVALID_RESPONSE",
            "Worker response is missing model_version",
            status_code=status_code,
        )
    if result.get("inference_time_ms") is None:
        raise CompletionWorkerError(
            "COMPLETION_WORKER_INVALID_RESPONSE",
            "Worker response is missing inference_time_ms",
            status_code=status_code,
        )
    result["result_uri"] = result_uri
    result["generated_roi"] = generated_roi
    result["worker_status"] = "WORKER_EXECUTED"
    return result
