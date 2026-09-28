"""Client for the generic Image Studio mode on the existing Qwen worker."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Iterable

from app.portrait_worker_client import (
    PortraitWorkerError,
    _json_response,
    _multipart_body,
    _read_image_uri,
    _result_uri,
)
from app.qwen_refine_worker_client import (
    QWEN_MODEL_LABEL,
    _base_url,
    _headers,
    _path,
    _timeout,
)

IMAGE_STUDIO_MODE = "image_studio_v1"
MAX_REFERENCES = 2


def invoke_image_studio_worker(
    *,
    base_image_uri: str,
    reference_image_uris: Iterable[str] = (),
    source_run_id: str | None = None,
    prompt: str,
    negative_prompt: str,
    steps: int = 25,
    seed: int | None = None,
    resolution_mode: str = "real_768",
) -> dict[str, Any]:
    base_url = _base_url()
    if not base_url:
        raise PortraitWorkerError(
            "QWEN_WORKER_NOT_CONFIGURED",
            "FISH_QWEN_REFINE_WORKER_URL is not configured",
        )

    references = [str(uri or "").strip() for uri in reference_image_uris if str(uri or "").strip()]
    if len(references) > MAX_REFERENCES:
        raise PortraitWorkerError(
            "IMAGE_STUDIO_TOO_MANY_REFERENCES",
            f"Image Studio V1 supports at most {MAX_REFERENCES} reference images",
        )

    base_data, base_media_type = _read_image_uri(base_image_uri, label="image_studio_base")
    files = [("image", "base_image.png", base_data, base_media_type)]
    reference_sizes: list[int] = []
    for index, uri in enumerate(references, start=1):
        data, media_type = _read_image_uri(uri, label=f"image_studio_reference_{index}")
        reference_sizes.append(len(data))
        files.append(("references", f"reference_{index}.png", data, media_type))

    safe_steps = max(1, min(int(steps), 100))
    safe_seed = int(seed) if seed is not None else None
    params = {
        "mode": IMAGE_STUDIO_MODE,
        "source_run_id": str(source_run_id or "") or None,
        "steps": safe_steps,
        "seed": safe_seed,
        "prompt": str(prompt or "").strip(),
        "negative_prompt": str(negative_prompt or "").strip(),
        "resolution_mode": str(resolution_mode or "real_768").strip().lower(),
    }
    fields = [
        ("mode", IMAGE_STUDIO_MODE),
        ("source_run_id", str(source_run_id or "")),
        ("params", json.dumps(params, ensure_ascii=False, separators=(",", ":"))),
    ]
    body, content_type = _multipart_body(fields=fields, files=files)
    headers = _headers()
    headers["Content-Type"] = content_type
    request = urllib.request.Request(
        f"{base_url}{_path()}",
        data=body,
        method="POST",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=_timeout()) as response:
            status_code, result = _json_response(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:2000] or f"HTTP {exc.code}"
        worker_code = ""
        worker_message = detail
        try:
            payload = json.loads(detail)
            nested = payload.get("detail") if isinstance(payload, dict) else None
            if isinstance(nested, dict):
                worker_code = str(nested.get("code") or "")
                worker_message = str(nested.get("message") or detail)
        except json.JSONDecodeError:
            pass
        if worker_code == "QWEN_WORKER_NOT_READY":
            raise PortraitWorkerError(
                "QWEN_WORKER_NOT_READY",
                worker_message,
                status_code=503,
            ) from exc
        code = "IMAGE_STUDIO_WORKER_PARAMETER_ERROR" if exc.code == 422 else (
            "IMAGE_STUDIO_WORKER_INFERENCE_ERROR" if exc.code >= 500 else "IMAGE_STUDIO_WORKER_HTTP_ERROR"
        )
        raise PortraitWorkerError(code, detail, status_code=exc.code) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise PortraitWorkerError("IMAGE_STUDIO_WORKER_CONNECTION_FAILED", str(exc)) from exc

    result_uri = _result_uri(
        {
            "result_uri": result.get("result_uri")
            or result.get("final_asset_uri")
            or result.get("refine_result_uri")
        },
        base_url,
    )
    if not result_uri:
        raise PortraitWorkerError(
            "IMAGE_STUDIO_WORKER_INVALID_RESPONSE",
            "Qwen worker response is missing result_uri",
            status_code=status_code,
        )

    result["result_uri"] = result_uri
    result["worker_model"] = result.get("worker_model") or QWEN_MODEL_LABEL
    result["worker_status"] = "WORKER_EXECUTED"
    result["worker_http_status"] = status_code
    result["worker_protocol"] = {
        "request": "multipart/form-data",
        "path": _path(),
        "mode": IMAGE_STUDIO_MODE,
        "base_bytes": len(base_data),
        "reference_count": len(references),
        "reference_bytes": reference_sizes,
        "steps": safe_steps,
        "seed": safe_seed,
        "resolution_mode": params["resolution_mode"],
    }
    return result


__all__ = ["IMAGE_STUDIO_MODE", "MAX_REFERENCES", "invoke_image_studio_worker"]
