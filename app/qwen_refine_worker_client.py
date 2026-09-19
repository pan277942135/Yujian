"""HTTP client for the independent Qwen-Image-Edit-2511 refine worker.

The Qwen worker is intentionally separate from the legacy Fish Portrait worker.
Cloud Run materializes the prepared Visible Fish Refined artifact and sends
exactly one multipart image part to the worker's /refine endpoint.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

from app.portrait_worker_client import (
    PortraitWorkerError,
    _json_response,
    _multipart_body,
    _read_image_uri,
    _result_uri,
)

logger = logging.getLogger(__name__)

QWEN_MODEL_ID = "qwen_image_edit_2511"
QWEN_MODEL_LABEL = "Qwen-Image-Edit-2511"
QWEN_MODE = "fish_preserve_refine_qwen_v1"
DEFAULT_STEPS = 20
DEFAULT_PROMPT = (
    "Restore and refine this fish into a clean, complete, realistic fish portrait. "
    "Keep the exact same fish identity, species traits, body proportions, head shape, "
    "fin structure, scale texture, and natural color pattern. "
    "Use the visible fish as the only identity reference. "
    "Complete the missing or occluded fish body parts naturally. "
    "Remove all non-fish objects and do not keep any human hand, fingers, tools, nets, "
    "hooks, ropes, buckets, or other foreign objects. "
    "The result should be a single complete fish, realistic, clean, natural, and high-detail."
)
DEFAULT_NEGATIVE_PROMPT = (
    "human hand, fingers, arm, person, tool, fishing net, hook, rope, bucket, "
    "extra fish, duplicated fish, changed species, wrong anatomy, deformed body, "
    "cartoon, painting, fake texture, unrealistic fins, broken tail"
)


def _base_url() -> str:
    return os.getenv("FISH_QWEN_REFINE_WORKER_URL", "").strip().rstrip("/")


def _path() -> str:
    value = os.getenv("FISH_QWEN_REFINE_WORKER_PATH", "/refine").strip() or "/refine"
    return value if value.startswith("/") else "/" + value


def _timeout(default: float = 1200.0) -> float:
    try:
        return max(1.0, float(os.getenv("FISH_QWEN_REFINE_WORKER_TIMEOUT_SECONDS", str(default))))
    except (TypeError, ValueError):
        return default


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/json"}
    token = os.getenv("FISH_QWEN_REFINE_WORKER_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def check_qwen_refine_worker() -> dict[str, Any]:
    """Return worker health without exposing the bearer token."""

    base_url = _base_url()
    if not base_url:
        return {
            "endpoint_configured": False,
            "status": "WORKER_UNAVAILABLE",
            "reason": "qwen_worker_endpoint_not_configured",
            "worker_url": None,
            "worker_model": QWEN_MODEL_LABEL,
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
        raise PortraitWorkerError("QWEN_WORKER_HEALTH_ERROR", detail, status_code=exc.code) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise PortraitWorkerError("QWEN_WORKER_HEALTH_UNREACHABLE", str(exc)) from exc
    return {
        "endpoint_configured": True,
        "status": "READY" if 200 <= status_code < 300 else "WORKER_UNAVAILABLE",
        "health_status": status_code,
        "health": payload,
        "worker_url": base_url,
        "worker_model": payload.get("model") or QWEN_MODEL_LABEL,
    }


def invoke_qwen_refine_worker(
    *,
    visible_fish_refined_image_uri: str | None = None,
    sam_visible_image_uri: str | None = None,
    source_run_id: str | None = None,
    prompt: str | None = None,
    negative_prompt: str | None = None,
    steps: int = DEFAULT_STEPS,
    seed: int | None = None,
    auto_straighten: bool = False,
) -> dict[str, Any]:
    """Invoke Qwen using Visible Fish Refined, never raw SAM.

    sam_visible_image_uri remains a compatibility alias for callers that have
    already migrated their stored request; the route only supplies the new
    visible_fish_refined_image_uri field.
    """

    base_url = _base_url()
    if not base_url:
        raise PortraitWorkerError(
            "QWEN_WORKER_NOT_CONFIGURED",
            "FISH_QWEN_REFINE_WORKER_URL is not configured",
        )
    input_uri = str(visible_fish_refined_image_uri or sam_visible_image_uri or "").strip()
    if not input_uri:
        raise PortraitWorkerError(
            "QWEN_VISIBLE_FISH_REFINED_REQUIRED",
            "Visible Fish Refined is required; raw SAM cannot be sent to Qwen",
        )
    sam_data, sam_media_type = _read_image_uri(input_uri, label="visible_fish_refined")
    safe_steps = max(1, min(int(steps), 100))
    safe_seed = int(seed) if seed is not None else None
    params = {
        "mode": QWEN_MODE,
        "source_run_id": str(source_run_id or "") or None,
        "steps": safe_steps,
        "seed": safe_seed,
        "auto_straighten": bool(auto_straighten),
        "prompt": str(prompt or DEFAULT_PROMPT).strip(),
        "negative_prompt": str(negative_prompt or DEFAULT_NEGATIVE_PROMPT).strip(),
    }
    fields = [
        ("mode", QWEN_MODE),
        ("source_run_id", str(source_run_id or "")),
        ("params", json.dumps(params, ensure_ascii=False, separators=(",", ":"))),
    ]
    body, content_type = _multipart_body(
        fields=fields,
        files=[("image", "visible_fish_refined.png", sam_data, sam_media_type)],
    )
    headers = _headers()
    headers["Content-Type"] = content_type
    request = urllib.request.Request(
        f"{base_url}{_path()}",
        data=body,
        method="POST",
        headers=headers,
    )
    logger.info(
        "qwen_refine_worker_request mode=%s source_run_id=%s visible_fish_refined_bytes=%d "
        "steps=%d seed=%s auto_straighten=%s",
        QWEN_MODE,
        source_run_id,
        len(sam_data),
        safe_steps,
        safe_seed,
        str(bool(auto_straighten)).lower(),
    )
    try:
        with urllib.request.urlopen(request, timeout=_timeout()) as response:
            status_code, result = _json_response(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:2000] or f"HTTP {exc.code}"
        code = "QWEN_WORKER_PARAMETER_ERROR" if exc.code == 422 else (
            "QWEN_WORKER_INFERENCE_ERROR" if exc.code >= 500 else "QWEN_WORKER_HTTP_ERROR"
        )
        raise PortraitWorkerError(code, detail, status_code=exc.code) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise PortraitWorkerError("QWEN_WORKER_CONNECTION_FAILED", str(exc)) from exc

    refined_uri = _result_uri(
        {
            "result_uri": result.get("refine_result_uri")
            or result.get("refined_result_uri")
            or result.get("refined_image_uri")
            or result.get("refined_path"),
        },
        base_url,
    )
    final_uri = _result_uri(
        {
            "result_uri": result.get("final_asset_uri")
            or result.get("final_result_uri")
            or result.get("final_image_uri")
            or result.get("result_uri")
            or result.get("output_uri")
            or result.get("output_path"),
        },
        base_url,
    )
    if not refined_uri and not final_uri:
        raise PortraitWorkerError(
            "QWEN_WORKER_INVALID_RESPONSE",
            "Qwen worker response is missing refine_result_uri/final_asset_uri",
            status_code=status_code,
        )
    result["refine_result_uri"] = refined_uri or final_uri
    result["final_asset_uri"] = final_uri or refined_uri
    result["result_uri"] = final_uri or refined_uri
    result["worker_model"] = result.get("worker_model") or QWEN_MODEL_LABEL
    result["worker_status"] = "WORKER_EXECUTED"
    result["worker_http_status"] = status_code
    result["elapsed_ms"] = result.get("elapsed_ms")
    if result["elapsed_ms"] is None:
        result["elapsed_ms"] = result.get("inference_time_ms")
    result["worker_protocol"] = {
        "request": "multipart/form-data",
        "path": _path(),
        "mode": QWEN_MODE,
        "input_source": "visible_fish_refined",
        "source_field": "image",
        "reference_field": None,
        "visible_fish_refined_bytes": len(sam_data),
        "steps": safe_steps,
        "seed": safe_seed,
        "auto_straighten": bool(auto_straighten),
    }
    return result


__all__ = [
    "QWEN_MODE",
    "QWEN_MODEL_ID",
    "QWEN_MODEL_LABEL",
    "DEFAULT_STEPS",
    "DEFAULT_PROMPT",
    "DEFAULT_NEGATIVE_PROMPT",
    "check_qwen_refine_worker",
    "invoke_qwen_refine_worker",
]