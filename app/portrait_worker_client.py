"""HTTP client for the dedicated SDXL + IP-Adapter Fish Portrait worker.

The portrait worker is intentionally separate from the existing PowerPaint
completion worker. Cloud Run orchestrates the request and materializes the
Dataset/Fish Asset bytes; image generation runs on the configured GPU worker.
"""
from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import re
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


MAX_IMAGE_BYTES = 50 * 1024 * 1024

INPAINT_DEFAULT_PROMPT = (
    "professional wildlife fish portrait, realistic photography, natural fish texture, "
    "detailed scales, clean natural background, soft lighting, high resolution"
)
INPAINT_DEFAULT_NEGATIVE_PROMPT = (
    "different fish species, changed body shape, wrong fish anatomy, extra fins, "
    "missing fins, deformed fish, cartoon, illustration, fake texture, duplicate fish"
)
REFINE_DEFAULT_PROMPT = (
    "professional wildlife fish portrait, realistic photography, "
    "preserve original fish identity, preserve original fish species characteristics, "
    "preserve original body proportions, preserve original head shape, "
    "preserve original fin structure, preserve original color and texture, "
    "complete fish body, natural fish anatomy, clean natural background, "
    "soft lighting, premium realistic fishing asset"
)
REFINE_DEFAULT_NEGATIVE_PROMPT = (
    "different fish species, different fish identity, changed body shape, changed head shape, "
    "changed body proportions, wrong fish anatomy, extra fins, missing fins, deformed fins, "
    "mutated fish, duplicate fish, cartoon, illustration, fake texture, obvious AI artifacts"
)

logger = logging.getLogger(__name__)


class PortraitWorkerError(RuntimeError):
    """A typed, safe-to-display worker failure."""

    def __init__(self, error_code: str, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.error_code = error_code
        self.status_code = status_code


def _base_url() -> str:
    return os.getenv("FISH_PORTRAIT_WORKER_URL", "").strip().rstrip("/")


def _worker_path(default: str = "/portrait", env_name: str = "FISH_PORTRAIT_WORKER_PATH") -> str:
    path = os.getenv(env_name, default).strip() or default
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


def _download_gcs_object(
    bucket_name: str,
    object_name: str,
    *,
    label: str,
) -> tuple[bytes, str]:
    """Read one managed object directly so relative media URLs stay internal."""

    from google.cloud import storage

    data = (
        storage.Client()
        .bucket(bucket_name)
        .blob(object_name)
        .download_as_bytes(timeout=120)
    )
    media_type = mimetypes.guess_type(object_name)[0] or "application/octet-stream"
    return _guard_image_size(data, label=label), media_type


def _knowledge_media_object_name(uri: str) -> str:
    """Map a managed Fish Knowledge media path to its canonical GCS object."""

    prefix = "/api/v1/fish/knowledge-media/"
    parts = uri[len(prefix):].strip("/").split("/")
    if len(parts) != 3 or not all(parts):
        raise ValueError("invalid managed knowledge media URI")
    species_id, asset_type, asset_key = parts
    normalized_type = asset_type.strip().lower()
    hashed = bool(re.fullmatch(r"[a-f0-9]{64}\.(?:jpg|png|webp)", asset_key))

    if normalized_type == "cover":
        if asset_key == "cover.webp":
            return f"fish-assets/{species_id}/cover/cover.webp"
        if hashed:
            return f"fish_knowledge/{species_id}/cover/{asset_key}"
    elif normalized_type in {"hero", "identification", "eco", "gear", "skill"}:
        if asset_key == f"{normalized_type}.webp":
            return f"fish-assets/{species_id}/cards/{normalized_type}.webp"
        if hashed:
            return f"fish_knowledge/{species_id}/{normalized_type}/{asset_key}"

    # Versioned media paths are normally converted to gs:// by the route
    # because their DB row carries the exact object_name. Do not guess them.
    raise ValueError("versioned managed media URI has no object mapping")


def _local_uri_path(uri: str) -> Path:
    """Resolve the debug lab's local:// URI without allowing path escape."""

    relative = str(uri or "")[len("local://") :].lstrip("/")
    root = Path.cwd().resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError("local image URI escapes the application workspace")
    return path


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
            data, media_type = _download_gcs_object(
                bucket_name,
                object_name,
                label=label,
            )
        elif value.startswith("/api/v1/fish/knowledge-media/"):
            bucket_name = os.getenv("GCS_BUCKET", "").strip()
            if not bucket_name:
                raise ValueError("GCS_BUCKET is not configured")
            data, media_type = _download_gcs_object(
                bucket_name,
                _knowledge_media_object_name(value),
                label=label,
            )
        elif value.startswith("local://"):
            path = _local_uri_path(value)
            if not path.is_file():
                raise FileNotFoundError(value)
            data = path.read_bytes()
            media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
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
        value = str(value).strip()
        if value.startswith(("/", "./")):
            return f"{base_url}/{value.lstrip('/')}"
        if not value.startswith(("data:", "gs://", "http://", "https://")):
            return f"{base_url}/{value.lstrip('/')}"
        return value
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

    The deployed VM receives both images as real multipart file parts. URI
    values are used only by Cloud Run to materialize bytes; they are not sent
    as image fields to the Worker.
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
        ("params", json.dumps(params, separators=(",", ":"))),
    ]
    body, content_type = _multipart_body(
        fields=fields,
        files=[
            ("image", "source_image", source_data, source_media_type),
            ("reference_image", "reference_image", reference_data, reference_media_type),
        ],
    )
    logger.info(
        "portrait_worker_request: has_image=%s has_reference_image=%s "
        "image_size=%d reference_size=%d",
        str(bool(source_data)).lower(),
        str(bool(reference_data)).lower(),
        len(source_data),
        len(reference_data),
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


def invoke_portrait_inpaint_worker(
    *,
    original_image_uri: str,
    fish_mask_uri: str | None,
    completion_mask_uri: str | None,
    species: str | None,
    prompt: str | None,
    negative_prompt: str | None,
    strength: float,
    steps: int,
    width: int,
    height: int,
    seed: int | None,
) -> dict[str, Any]:
    """Invoke the preserve-inpaint worker with managed image/mask URIs."""

    base_url = _base_url()
    if not base_url:
        raise PortraitWorkerError(
            "PORTRAIT_WORKER_NOT_CONFIGURED",
            "FISH_PORTRAIT_WORKER_URL is not configured",
        )
    payload = {
        "mode": "fish_preserve_inpaint_v2",
        "original_image_uri": str(original_image_uri or "").strip(),
        "fish_mask_uri": str(fish_mask_uri or "").strip() or None,
        "completion_mask_uri": str(completion_mask_uri or "").strip() or None,
        "species": str(species or "").strip() or None,
        "prompt": str(prompt or INPAINT_DEFAULT_PROMPT).strip(),
        "negative_prompt": str(negative_prompt or INPAINT_DEFAULT_NEGATIVE_PROMPT).strip(),
        "strength": float(strength),
        "steps": int(steps),
        "width": int(width),
        "height": int(height),
        "seed": int(seed) if seed is not None else None,
    }
    if not payload["original_image_uri"]:
        raise PortraitWorkerError("PORTRAIT_ORIGINAL_IMAGE_URI_MISSING", "original_image_uri is required")
    headers = _headers()
    headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{base_url}{_worker_path('/portrait/generate', 'FISH_PORTRAIT_INPAINT_WORKER_PATH')}",
        data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        method="POST",
        headers=headers,
    )
    logger.info(
        "portrait_inpaint_worker_request: mode=%s has_original=%s has_fish_mask=%s "
        "has_completion_mask=%s strength=%.2f steps=%d size=%sx%s seed=%s",
        payload["mode"],
        str(bool(payload["original_image_uri"])).lower(),
        str(bool(payload["fish_mask_uri"])).lower(),
        str(bool(payload["completion_mask_uri"])).lower(),
        payload["strength"],
        payload["steps"],
        payload["width"],
        payload["height"],
        payload["seed"],
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
            "Portrait inpaint worker response is missing generated_image_uri/result_uri/image_url",
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
        "request": "application/json",
        "mode": payload["mode"],
        "original_image_uri": bool(payload["original_image_uri"]),
        "fish_mask_uri": bool(payload["fish_mask_uri"]),
        "completion_mask_uri": bool(payload["completion_mask_uri"]),
    }
    return result


def invoke_portrait_refine_worker(
    *,
    sam_visible_image_uri: str,
    original_image_uri: str | None,
    source_run_id: str | None,
    species: str | None,
    prompt: str | None,
    negative_prompt: str | None,
    preserve_strength: float,
    refine_strength: float,
    auto_straighten: bool,
    steps: int,
    seed: int | None,
) -> dict[str, Any]:
    """Run Fish Preserve Refine V2 from an existing SAM Visible artifact.

    The worker receives exactly one image part: the already prepared SAM
    Visible fish.  Detector/SAM and the Original -> SAM Visible preparation
    remain owned by PowerPaint Direct Lab V0.2.
    """

    base_url = _base_url()
    if not base_url:
        raise PortraitWorkerError(
            "PORTRAIT_WORKER_NOT_CONFIGURED",
            "FISH_PORTRAIT_WORKER_URL is not configured",
        )
    sam_data, sam_media_type = _read_image_uri(sam_visible_image_uri, label="sam_visible")
    params = {
        "preserve_strength": float(preserve_strength),
        "refine_strength": float(refine_strength),
        "auto_straighten": bool(auto_straighten),
        "steps": int(steps),
        "seed": int(seed) if seed is not None else None,
    }
    fields = [
        ("task", "fish_preserve_refine"),
        ("mode", "fish_preserve_refine_v2"),
        ("species", str(species or "")),
        ("source_run_id", str(source_run_id or "")),
        ("original_image_uri", str(original_image_uri or "")),
        ("preserve_strength", str(params["preserve_strength"])),
        ("refine_strength", str(params["refine_strength"])),
        ("auto_straighten", "true" if params["auto_straighten"] else "false"),
        ("steps", str(params["steps"])),
        ("seed", "" if params["seed"] is None else str(params["seed"])),
        ("params", json.dumps(params, separators=(",", ":"))),
    ]
    body, content_type = _multipart_body(
        fields=fields,
        files=[("image", "sam_visible.png", sam_data, sam_media_type)],
    )
    logger.info(
        "portrait_refine_worker_request: mode=fish_preserve_refine_v2 "
        "input_source=sam_visible sam_visible_bytes=%d species=%s "
        "preserve_strength=%.2f refine_strength=%.2f auto_straighten=%s steps=%d seed=%s",
        len(sam_data),
        str(species or ""),
        params["preserve_strength"],
        params["refine_strength"],
        str(params["auto_straighten"]).lower(),
        params["steps"],
        params["seed"],
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

    refined_uri = _result_uri(
        {
            "result_uri": result.get("refine_result_uri")
            or result.get("refined_fish_uri")
            or result.get("refined_image_uri")
            or result.get("refined_result_uri")
            or result.get("refined_image")
            or result.get("refined_path")
            or result.get("refined_result_path"),
        },
        base_url,
    )
    final_uri = _result_uri(
        {
            "result_uri": result.get("final_asset_uri")
            or result.get("final_result_uri")
            or result.get("final_image_uri")
            or result.get("final_asset_path")
            or result.get("result_uri")
            or result.get("generated_image_uri")
            or result.get("image_url")
            or result.get("result_path"),
        },
        base_url,
    )
    if not final_uri and not refined_uri:
        raise PortraitWorkerError(
            "PORTRAIT_WORKER_INVALID_RESPONSE",
            "Portrait refine worker response is missing refined/final output URI",
            status_code=status_code,
        )
    result["refine_result_uri"] = refined_uri or final_uri
    result["final_asset_uri"] = final_uri or refined_uri
    result["result_uri"] = final_uri or refined_uri
    result["worker_status"] = "WORKER_EXECUTED"
    result["worker_http_status"] = status_code
    result["worker_protocol"] = {
        "request": "multipart/form-data",
        "mode": "fish_preserve_refine_v2",
        "input_source": "sam_visible",
        "source_field": "image",
        "reference_field": None,
        "sam_visible_bytes": len(sam_data),
        "preserve_strength": params["preserve_strength"],
        "refine_strength": params["refine_strength"],
        "auto_straighten": params["auto_straighten"],
        "steps": params["steps"],
        "seed": params["seed"],
    }
    return result


__all__ = [
    "PortraitWorkerError",
    "check_portrait_worker",
    "invoke_portrait_worker",
    "invoke_portrait_inpaint_worker",
    "invoke_portrait_refine_worker",
    "INPAINT_DEFAULT_PROMPT",
    "INPAINT_DEFAULT_NEGATIVE_PROMPT",
    "REFINE_DEFAULT_PROMPT",
    "REFINE_DEFAULT_NEGATIVE_PROMPT",
]
