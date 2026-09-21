"""Production-ready Qwen-Image-Edit-2511 worker for Fish Portrait Lab.

The worker keeps the existing ComfyUI-backed protocol, but turns the first-request
model load into an explicit startup warmup. ComfyUI remains the single owner of
the Qwen model and its GPU cache; this service never creates a second model copy.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import secrets
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("fish-qwen-refine-worker")

SERVICE_NAME = "fish-qwen-refine-worker"
MODEL_LABEL = "Qwen-Image-Edit-2511"
QWEN_MODE = "fish_preserve_refine_qwen_v1"
DEFAULT_CONFIG_PATH = "/opt/fish-qwen-refine-worker/config.yaml"
DEFAULT_WORKFLOW_PATH = "/opt/fish-qwen-refine-worker/qwen2511_api.json"
DEFAULT_OUTPUT_DIR = "/opt/fish-qwen-refine-worker/output"
DEFAULT_COMFY_URL = "http://127.0.0.1:8188"
DEFAULT_TIMEOUT_SECONDS = 1200.0
DEFAULT_POLL_SECONDS = 2.0
DEFAULT_STEPS = 25
DEFAULT_WIDTH = 768
DEFAULT_HEIGHT = 768
DEFAULT_DTYPE = "float16"
DEFAULT_DEVICE = "cuda"
DEFAULT_COMFY_READY_TIMEOUT_SECONDS = 120.0
DEFAULT_COMFY_READY_POLL_SECONDS = 2.0
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
# A valid fallback image used only when Pillow is not available in a local smoke test.
_FALLBACK_WARMUP_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


@dataclass(frozen=True)
class WorkerConfig:
    model_name: str = MODEL_LABEL
    unet_name: str = "qwen_image_edit_2511_int8_convrot.safetensors"
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    steps: int = DEFAULT_STEPS
    dtype: str = DEFAULT_DTYPE
    device: str = DEFAULT_DEVICE
    warmup_enabled: bool = True
    warmup_steps: int = 1
    warmup_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    @property
    def workflow_weight_dtype(self) -> str:
        return {
            "float16": "fp16",
            "bfloat16": "bf16",
            "float32": "default",
        }.get(self.dtype, "default")

    @property
    def precision_mode(self) -> str:
        # The L4 deployment uses the verified int8 Qwen weight file. ComfyUI
        # performs CUDA compute in the configured fp16 path while retaining the
        # quantized weights that fit in 24 GiB of VRAM.
        if "int8" in self.unet_name.lower() and self.dtype == "float16":
            return "fp16_compute_int8_weights"
        return self.dtype

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "WorkerConfig":
        model = _section(payload, "model")
        generation = _section(payload, "generation")
        runtime = _section(payload, "runtime")

        model_name = str(model.get("name") or MODEL_LABEL).strip()
        unet_name = str(
            model.get("unet_name") or cls.unet_name
        ).strip()
        width = _bounded_int(generation.get("width", DEFAULT_WIDTH), "width", 64, 2048)
        height = _bounded_int(generation.get("height", DEFAULT_HEIGHT), "height", 64, 2048)
        steps = _bounded_int(generation.get("steps", DEFAULT_STEPS), "steps", 1, 100)
        warmup_steps = _bounded_int(
            runtime.get("warmup_steps", 1), "warmup_steps", 1, 100
        )
        warmup_timeout = float(
            runtime.get("warmup_timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
        )
        if warmup_timeout < 30:
            raise ValueError("warmup_timeout_seconds must be at least 30")
        dtype = _normalise_dtype(generation.get("dtype", DEFAULT_DTYPE))
        device = str(runtime.get("device") or DEFAULT_DEVICE).strip().lower()
        if device not in {"cuda", "cuda:0"}:
            raise ValueError("runtime.device must be cuda or cuda:0")
        warmup_enabled = _as_bool(runtime.get("warmup", True))
        return cls(
            model_name=model_name or MODEL_LABEL,
            unet_name=unet_name or cls.unet_name,
            width=width,
            height=height,
            steps=steps,
            dtype=dtype,
            device=device,
            warmup_enabled=warmup_enabled,
            warmup_steps=warmup_steps,
            warmup_timeout_seconds=warmup_timeout,
        )


def _section(payload: dict[str, Any], name: str) -> dict[str, Any]:
    value = payload.get(name) or {}
    if not isinstance(value, dict):
        raise ValueError(f"config.{name} must be a mapping")
    return value


def _bounded_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"config.{name} must be an integer") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(f"config.{name} must be between {minimum} and {maximum}")
    return parsed


def _normalise_dtype(value: Any) -> str:
    normalised = str(value or DEFAULT_DTYPE).strip().lower()
    aliases = {
        "torch.float16": "float16",
        "fp16": "float16",
        "torch.bfloat16": "bfloat16",
        "bf16": "bfloat16",
        "torch.float32": "float32",
        "fp32": "float32",
    }
    normalised = aliases.get(normalised, normalised)
    if normalised not in {"float16", "bfloat16", "float32"}:
        raise ValueError("generation.dtype must be float16, bfloat16 or float32")
    return normalised


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _load_config(path: Path) -> tuple[WorkerConfig, str | None]:
    if not path.exists():
        logger.warning("Qwen config not found at %s; using built-in defaults", path)
        return WorkerConfig(), None
    try:
        import yaml

        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError("config root must be a mapping")
        return WorkerConfig.from_mapping(payload), None
    except Exception as exc:
        logger.exception("Qwen config load failed: %s", path)
        return WorkerConfig(), str(exc)


CONFIG_PATH = Path(os.getenv("QWEN_CONFIG_PATH", DEFAULT_CONFIG_PATH))
CONFIG, CONFIG_ERROR = _load_config(CONFIG_PATH)
OUTPUT_DIR = Path(os.getenv("QWEN_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
WORKFLOW_PATH = Path(os.getenv("QWEN_WORKFLOW_PATH", DEFAULT_WORKFLOW_PATH))
COMFY_URL = os.getenv("COMFYUI_URL", DEFAULT_COMFY_URL).rstrip("/")
COMFY_READY_TIMEOUT_SECONDS = max(
    30.0,
    float(os.getenv("COMFYUI_READY_TIMEOUT_SECONDS", str(DEFAULT_COMFY_READY_TIMEOUT_SECONDS))),
)
COMFY_READY_POLL_SECONDS = max(
    0.5,
    float(os.getenv("COMFYUI_READY_POLL_SECONDS", str(DEFAULT_COMFY_READY_POLL_SECONDS))),
)
WORKER_TOKEN = os.getenv("QWEN_REFINE_WORKER_TOKEN", "").strip()
REQUEST_TIMEOUT = max(
    30.0,
    float(os.getenv("QWEN_REQUEST_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))),
)
POLL_SECONDS = max(
    0.5,
    float(os.getenv("QWEN_POLL_SECONDS", str(DEFAULT_POLL_SECONDS))),
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _detect_gpu_name() -> str:
    configured = os.getenv("QWEN_GPU_NAME", "").strip()
    if configured:
        return configured
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        first_line = next(
            (line.strip() for line in completed.stdout.splitlines() if line.strip()),
            "",
        )
        if first_line:
            return first_line
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        pass
    return "NVIDIA L4"


GPU_NAME = _detect_gpu_name()
app = FastAPI(title=SERVICE_NAME, version="2.0.0")
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")


_RUNTIME_LOCK = threading.Lock()
_RUNTIME: dict[str, Any] = {
    "status": "starting",
    "stage": "waiting_comfyui",
    "model_loaded": False,
    "error_code": "QWEN_CONFIG_INVALID" if CONFIG_ERROR else None,
    "error": CONFIG_ERROR,
    "warmup_request_id": None,
    "warmup_time": None,
}
_WARMUP_STARTED = False


class WorkerError(RuntimeError):
    def __init__(
        self,
        message: str,
        status_code: int = 502,
        error_code: str = "QWEN_WORKER_ERROR",
    ):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


def _runtime_snapshot() -> dict[str, Any]:
    with _RUNTIME_LOCK:
        return dict(_RUNTIME)


def _set_runtime(**values: Any) -> None:
    with _RUNTIME_LOCK:
        _RUNTIME.update(values)


def _worker_ready() -> bool:
    with _RUNTIME_LOCK:
        return _RUNTIME["status"] == "ready" and bool(_RUNTIME["model_loaded"])


def _health_payload() -> dict[str, Any]:
    state = _runtime_snapshot()
    payload: dict[str, Any] = {
        "status": state["status"],
        "stage": state.get("stage"),
        "service": SERVICE_NAME,
        "model": CONFIG.model_name or MODEL_LABEL,
        "gpu": GPU_NAME,
        "device": CONFIG.device,
        "dtype": CONFIG.dtype,
        "precision_mode": CONFIG.precision_mode,
        "model_loaded": bool(state["model_loaded"]),
        "config": {
            "width": CONFIG.width,
            "height": CONFIG.height,
            "steps": CONFIG.steps,
            "dtype": CONFIG.dtype,
            "device": CONFIG.device,
            "config_path": str(CONFIG_PATH),
        },
    }
    if state.get("error_code"):
        payload["error_code"] = state["error_code"]
    if state.get("warmup_request_id"):
        payload["warmup_request_id"] = state["warmup_request_id"]
    if state.get("warmup_time") is not None:
        payload["warmup_time"] = state["warmup_time"]
    if state.get("error"):
        payload["error"] = state["error"]
    return payload


def _require_token(authorization: str | None = Header(default=None)) -> None:
    if WORKER_TOKEN and authorization != "Bearer " + WORKER_TOKEN:
        raise HTTPException(status_code=401, detail="invalid worker token")


def _json_request(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    headers: dict[str, str] | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    request_headers = {"Accept": "application/json", **(headers or {})}
    body = None
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    request = urllib.request.Request(
        COMFY_URL + path,
        data=body,
        method=method,
        headers=request_headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout or REQUEST_TIMEOUT) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:2000]
        raise WorkerError(
            f"ComfyUI {method} {path}: {detail or exc}",
            502,
            "COMFYUI_ERROR",
        ) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise WorkerError(
            f"ComfyUI {method} {path} unavailable: {exc}",
            502,
            "COMFYUI_UNAVAILABLE",
        ) from exc
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkerError(
            f"ComfyUI returned non-JSON for {method} {path}",
            502,
            "COMFYUI_INVALID_RESPONSE",
        ) from exc
    if not isinstance(decoded, dict):
        raise WorkerError(
            f"ComfyUI returned a non-object for {method} {path}",
            502,
            "COMFYUI_INVALID_RESPONSE",
        )
    return decoded


def _multipart_upload(filename: str, content: bytes, content_type: str) -> dict[str, Any]:
    boundary = "----FishQwen" + secrets.token_hex(12)
    parts = [
        (
            "--" + boundary + "\r\n"
            + 'Content-Disposition: form-data; name="image"; filename="' + filename + '"\r\n'
            + "Content-Type: " + content_type + "\r\n\r\n"
        ).encode("utf-8"),
        content,
        ("\r\n--" + boundary + "--\r\n").encode("utf-8"),
    ]
    request = urllib.request.Request(
        COMFY_URL + "/upload/image",
        data=b"".join(parts),
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": "multipart/form-data; boundary=" + boundary,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:2000]
        raise WorkerError(
            "ComfyUI image upload failed: " + (detail or str(exc)),
            502,
            "COMFYUI_ERROR",
        ) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise WorkerError(
            "ComfyUI image upload unavailable: " + str(exc),
            502,
            "COMFYUI_UNAVAILABLE",
        ) from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkerError(
            "ComfyUI image upload returned non-JSON",
            502,
            "COMFYUI_INVALID_RESPONSE",
        ) from exc
    if not isinstance(payload, dict) or not payload.get("name"):
        raise WorkerError(
            "ComfyUI image upload did not return a name",
            502,
            "COMFYUI_INVALID_RESPONSE",
        )
    return payload


def _download_view(image_ref: dict[str, Any]) -> bytes:
    query = urllib.parse.urlencode(
        {
            "filename": image_ref.get("filename", ""),
            "subfolder": image_ref.get("subfolder", ""),
            "type": image_ref.get("type", "output"),
        }
    )
    request = urllib.request.Request(
        COMFY_URL + "/view?" + query,
        method="GET",
        headers={"Accept": "image/png, image/jpeg, image/webp, */*"},
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:2000]
        raise WorkerError(
            "ComfyUI result download failed: " + (detail or str(exc)),
            502,
            "COMFYUI_ERROR",
        ) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise WorkerError(
            "ComfyUI result download unavailable: " + str(exc),
            502,
            "COMFYUI_UNAVAILABLE",
        ) from exc


def _wait_for_comfyui_ready() -> None:
    """Wait for the local ComfyUI HTTP process before submitting warmup."""

    deadline = time.monotonic() + COMFY_READY_TIMEOUT_SECONDS
    probe_url = COMFY_URL + "/system_stats"
    last_error = "no response"
    _set_runtime(status="loading", stage="waiting_comfyui", error=None, error_code=None)
    while time.monotonic() < deadline:
        request = urllib.request.Request(
            probe_url,
            method="GET",
            headers={"Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=min(5.0, COMFY_READY_TIMEOUT_SECONDS),
            ) as response:
                if 200 <= int(getattr(response, "status", 200)) < 500:
                    logger.info("ComfyUI ready url=%s", COMFY_URL)
                    return
                last_error = "HTTP %s" % getattr(response, "status", "unknown")
        except urllib.error.HTTPError as exc:
            # A responsive ComfyUI process may return 404 for an endpoint
            # changed by its installed version; any HTTP response below 500
            # still proves that port 8188 is serving.
            if exc.code < 500:
                logger.info("ComfyUI ready url=%s http_status=%s", COMFY_URL, exc.code)
                return
            last_error = "HTTP %s" % exc.code
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        time.sleep(COMFY_READY_POLL_SECONDS)
    raise WorkerError(
        "ComfyUI readiness timeout at %s after %.0fs: %s"
        % (COMFY_URL, COMFY_READY_TIMEOUT_SECONDS, last_error),
        503,
        "COMFYUI_UNAVAILABLE",
    )


def _load_workflow() -> dict[str, Any]:
    try:
        workflow = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise WorkerError("workflow file not found: " + str(WORKFLOW_PATH), 500) from exc
    except json.JSONDecodeError as exc:
        raise WorkerError("workflow file is invalid JSON: " + str(exc), 500) from exc
    if not isinstance(workflow, dict):
        raise WorkerError("workflow file root must be an object", 500)
    return workflow


def _set_input(workflow: dict[str, Any], node_id: str, key: str, value: Any) -> None:
    node = workflow.get(node_id)
    if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict):
        raise WorkerError("workflow node %s is missing inputs" % node_id, 500)
    node["inputs"][key] = value


def _set_optional_input(
    workflow: dict[str, Any], node_id: str, key: str, value: Any
) -> bool:
    node = workflow.get(node_id)
    inputs = node.get("inputs") if isinstance(node, dict) else None
    if not isinstance(inputs, dict) or key not in inputs:
        return False
    inputs[key] = value
    return True


def _patch_workflow(
    workflow: dict[str, Any],
    *,
    uploaded_file: str,
    prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    output_prefix: str,
    width: int,
    height: int,
    cfg_scale: float,
    negative_prompt_sent: bool,
    resolution_mode: str,
    diffusion_width: int | None,
    diffusion_height: int | None,
) -> None:
    _set_input(workflow, "41", "image", uploaded_file)
    _set_input(workflow, "170:151", "prompt", prompt)
    _set_input(
        workflow,
        "170:149",
        "prompt",
        negative_prompt if negative_prompt_sent else "",
    )
    _set_input(workflow, "170:169", "seed", seed)
    _set_input(workflow, "170:154", "value", cfg_scale)
    _set_input(workflow, "170:165", "value", 4)
    _set_input(workflow, "170:166", "value", steps)
    # The validated workflow keeps the Lightning branch opt-in. The page uses
    # the non-Lightning path so Steps remains user-controlled.
    _set_input(workflow, "170:168", "value", False)
    _set_input(workflow, "195", "filename_prefix", output_prefix)

    if resolution_mode == "real_768":
        if diffusion_width is None or diffusion_height is None:
            raise WorkerError("real_768 dimensions were not calculated", 500)
        workflow["196"] = {
            "inputs": {
                "image": ["41", 0],
                "upscale_method": "lanczos",
                "width": diffusion_width,
                "height": diffusion_height,
                "crop": "disabled",
            },
            "class_type": "ImageScale",
            "_meta": {"title": "Real-768 aspect-preserving input scale"},
        }
        _set_input(workflow, "170:160", "image", ["196", 0])
    elif resolution_mode != "current":
        raise WorkerError("unsupported resolution_mode: " + resolution_mode, 422)

    _set_optional_input(workflow, "170:161", "unet_name", CONFIG.unet_name)
    if "int8" not in CONFIG.unet_name.lower():
        _set_optional_input(
            workflow,
            "170:161",
            "weight_dtype",
            CONFIG.workflow_weight_dtype,
        )


def _wait_for_output(prompt_id: str, timeout: float | None = None) -> dict[str, Any]:
    deadline = time.monotonic() + (timeout or REQUEST_TIMEOUT)
    while time.monotonic() < deadline:
        history = _json_request(
            "GET", "/history/" + urllib.parse.quote(prompt_id, safe="")
        )
        entry = history.get(prompt_id) or history.get(str(prompt_id))
        if entry:
            status = entry.get("status") or {}
            if status.get("status_str") == "error":
                messages = status.get("messages") or []
                raise WorkerError("ComfyUI workflow failed: " + str(messages), 502)
            outputs = entry.get("outputs") or {}
            result = outputs.get("195") or {}
            images = result.get("images") or []
            if images:
                return images[0]
        time.sleep(POLL_SECONDS)
    raise WorkerError(
        "ComfyUI workflow timed out after %.0f seconds"
        % (timeout or REQUEST_TIMEOUT),
        504,
    )


def _safe_prefix(source_run_id: str | None) -> str:
    raw = "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in (source_run_id or "run")
    )
    return raw[:48] or "run"


def _make_warmup_image() -> bytes:
    try:
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (CONFIG.width, CONFIG.height), (8, 20, 18))
        draw = ImageDraw.Draw(image)
        draw.ellipse(
            (
                CONFIG.width * 0.18,
                CONFIG.height * 0.36,
                CONFIG.width * 0.82,
                CONFIG.height * 0.64,
            ),
            fill=(150, 170, 160),
        )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    except ImportError:
        return _FALLBACK_WARMUP_PNG


def _read_image_size(content: bytes) -> tuple[int, int] | None:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(content)) as image:
            return int(image.width), int(image.height)
    except Exception:
        return None


def _real_768_dimensions(content: bytes) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(content)) as image:
            source_width, source_height = image.size
    except Exception as exc:
        raise WorkerError(
            "real_768 requires a readable image: " + str(exc), 422
        ) from exc

    if source_width <= 0 or source_height <= 0:
        raise WorkerError("real_768 requires a non-empty image", 422)

    scale = 768.0 / max(source_width, source_height)
    width = max(16, int(round(source_width * scale / 16.0)) * 16)
    height = max(16, int(round(source_height * scale / 16.0)) * 16)
    return width, height


def _execute_generation(
    *,
    content: bytes,
    content_type: str,
    filename: str,
    prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    width: int,
    height: int,
    source_run_id: str | None,
    request_id: str,
    timeout: float,
    cfg_scale: float = 4.0,
    negative_prompt_sent: bool = True,
    resolution_mode: str = "current",
    diffusion_width: int | None = None,
    diffusion_height: int | None = None,
) -> dict[str, Any]:
    uploaded_name = "QWEN_" + request_id + "_" + Path(filename).name
    uploaded = _multipart_upload(uploaded_name, content, content_type)
    uploaded_file = uploaded["name"]
    if uploaded.get("subfolder"):
        uploaded_file = str(uploaded["subfolder"]).strip("/") + "/" + uploaded_file

    workflow = _load_workflow()
    _patch_workflow(
        workflow,
        uploaded_file=uploaded_file,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        steps=steps,
        output_prefix="Qwen_Refine_" + _safe_prefix(source_run_id) + "_" + request_id[:8],
        width=width,
        height=height,
        cfg_scale=cfg_scale,
        negative_prompt_sent=negative_prompt_sent,
        resolution_mode=resolution_mode,
        diffusion_width=diffusion_width,
        diffusion_height=diffusion_height,
    )

    inference_started = time.monotonic()
    prompt_response = _json_request(
        "POST",
        "/prompt",
        {"prompt": workflow, "client_id": SERVICE_NAME + "-" + request_id},
        timeout=timeout,
    )
    prompt_id = prompt_response.get("prompt_id")
    if not prompt_id:
        raise WorkerError("ComfyUI /prompt did not return prompt_id", 502)
    result_ref = _wait_for_output(str(prompt_id), timeout=timeout)
    inference_seconds = round(time.monotonic() - inference_started, 3)

    download_started = time.monotonic()
    result_bytes = _download_view(result_ref)
    download_seconds = round(time.monotonic() - download_started, 3)
    return {
        "prompt_id": str(prompt_id),
        "result_bytes": result_bytes,
        "inference_seconds": inference_seconds,
        "download_seconds": download_seconds,
    }


def _warmup_worker() -> None:
    request_id = "warmup-" + secrets.token_hex(6)
    _set_runtime(
        status="starting",
        stage="waiting_comfyui",
        model_loaded=False,
        error=None,
        error_code=None,
        warmup_request_id=request_id,
    )
    started = time.monotonic()
    logger.info(
        "Loading Qwen Image Edit model... model=%s device=%s dtype=torch.%s "
        "resolution=%sx%s steps=%s",
        CONFIG.model_name,
        CONFIG.device,
        CONFIG.dtype,
        CONFIG.width,
        CONFIG.height,
        CONFIG.warmup_steps,
    )
    if CONFIG_ERROR:
        _set_runtime(
            status="error",
            stage="config",
            model_loaded=False,
            error_code="QWEN_CONFIG_INVALID",
            error=CONFIG_ERROR,
        )
        logger.error("Worker config error: %s", CONFIG_ERROR)
        return
    try:
        _wait_for_comfyui_ready()
        _set_runtime(
            status="loading",
            stage="qwen_warmup",
            model_loaded=False,
            error=None,
            error_code=None,
        )
        logger.info("Qwen warmup started request_id=%s", request_id)
        result = _execute_generation(
            content=_make_warmup_image(),
            content_type="image/png",
            filename="qwen_worker_warmup.png",
            prompt="Warm up the Qwen image edit pipeline.",
            negative_prompt="deformed, duplicate, artifact",
            seed=0,
            steps=CONFIG.warmup_steps,
            width=CONFIG.width,
            height=CONFIG.height,
            source_run_id="worker_warmup",
            request_id=request_id,
            timeout=CONFIG.warmup_timeout_seconds,
        )
        warmup_seconds = round(time.monotonic() - started, 3)
        _set_runtime(
            status="ready",
            stage="ready",
            model_loaded=True,
            error=None,
            error_code=None,
            warmup_time=warmup_seconds,
        )
        logger.info(
            "Model loaded successfully model=%s device=%s dtype=torch.%s "
            "precision_mode=%s warmup_time=%.3fs inference_time=%.3fs",
            CONFIG.model_name,
            CONFIG.device,
            CONFIG.dtype,
            CONFIG.precision_mode,
            warmup_seconds,
            result["inference_seconds"],
        )
        logger.info("Worker ready")
    except WorkerError as exc:
        _set_runtime(
            status="error",
            stage="qwen_warmup",
            model_loaded=False,
            error_code=exc.error_code,
            error=str(exc),
            warmup_time=round(time.monotonic() - started, 3),
        )
        logger.exception("Qwen model warmup failed code=%s", exc.error_code)
    except Exception as exc:
        _set_runtime(
            status="error",
            stage="qwen_warmup",
            model_loaded=False,
            error_code="QWEN_WARMUP_FAILED",
            error=str(exc),
            warmup_time=round(time.monotonic() - started, 3),
        )
        logger.exception("Qwen model warmup failed")
 
 
@app.on_event("startup")
def _start_warmup() -> None:
    global _WARMUP_STARTED
    if _WARMUP_STARTED:
        return
    _WARMUP_STARTED = True
    if not CONFIG.warmup_enabled:
        _set_runtime(
            status="error",
            stage="config",
            model_loaded=False,
            error_code="QWEN_WARMUP_DISABLED",
            error="runtime.warmup is disabled; worker cannot become ready",
        )
        logger.error("runtime.warmup is disabled; refusing inference")
        return
    threading.Thread(
        target=_warmup_worker,
        name="qwen-model-warmup",
        daemon=True,
    ).start()


@app.get("/health", dependencies=[Depends(_require_token)])
def health() -> JSONResponse:
    return JSONResponse(_health_payload(), status_code=200)


@app.post("/refine", dependencies=[Depends(_require_token)])
def refine(
    image: UploadFile = File(...),
    params: str = Form(default="{}"),
    mode: str = Form(default=QWEN_MODE),
    source_run_id: str = Form(default=""),
) -> JSONResponse:
    request_received = time.monotonic()
    request_id = "req-" + secrets.token_hex(8)
    if not _worker_ready():
        state = _runtime_snapshot()
        raise HTTPException(
            status_code=503,
            detail={
                "code": "QWEN_WORKER_NOT_READY",
                "status": state["status"],
                "model_loaded": bool(state["model_loaded"]),
                "message": state.get("error") or "Qwen model warmup is still running",
            },
        )
    try:
        options = json.loads(params or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="params must be a JSON object") from exc
    if not isinstance(options, dict):
        raise HTTPException(status_code=422, detail="params must be a JSON object")

    requested_mode = str(options.get("mode") or mode or QWEN_MODE)
    if requested_mode != QWEN_MODE:
        raise HTTPException(status_code=422, detail="unsupported mode: " + requested_mode)
    try:
        steps = _bounded_int(
            options.get("steps", CONFIG.steps),
            "steps",
            1,
            100,
        )
        width = _bounded_int(
            options.get("width", CONFIG.width),
            "width",
            64,
            2048,
        )
        height = _bounded_int(
            options.get("height", CONFIG.height),
            "height",
            64,
            2048,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    experiment_no_cfg = _as_bool(options.get("experimental_no_cfg", False))
    resolution_mode = str(options.get("resolution_mode") or "current").strip().lower()
    if resolution_mode not in {"current", "real_768"}:
        raise HTTPException(
            status_code=422,
            detail="resolution_mode must be current or real_768",
        )
    cfg_scale = 1.0 if experiment_no_cfg else 4.0
    negative_prompt_sent = not experiment_no_cfg

    raw_seed = options.get("seed")
    try:
        seed = (
            int(raw_seed)
            if raw_seed not in (None, "")
            else secrets.randbelow(2**63 - 1)
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail="seed must be an integer or null"
        ) from exc

    auto_straighten = bool(options.get("auto_straighten", False))
    prompt = str(options.get("prompt") or DEFAULT_PROMPT).strip()
    negative_prompt = (
        str(options.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT).strip()
        if negative_prompt_sent
        else ""
    )
    source_id = str(options.get("source_run_id") or source_run_id or "") or None
    content = image.file.read()
    if not content:
        raise HTTPException(status_code=422, detail="image is empty")
    content_type = image.content_type or "image/png"
    original_filename = Path(image.filename or "visible_fish.png").name
    diffusion_width = None
    diffusion_height = None
    if resolution_mode == "real_768":
        try:
            diffusion_width, diffusion_height = _real_768_dimensions(content)
        except WorkerError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    try:
        result = _execute_generation(
            content=content,
            content_type=content_type,
            filename=original_filename,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            steps=steps,
            width=width,
            height=height,
            source_run_id=source_id,
            request_id=request_id,
            timeout=REQUEST_TIMEOUT,
            cfg_scale=cfg_scale,
            negative_prompt_sent=negative_prompt_sent,
            resolution_mode=resolution_mode,
            diffusion_width=diffusion_width,
            diffusion_height=diffusion_height,
        )
        save_started = time.monotonic()
        output_name = "REFINE_" + secrets.token_hex(12) + ".png"
        output_path = OUTPUT_DIR / output_name
        output_path.write_bytes(result["result_bytes"])
        save_seconds = round(time.monotonic() - save_started, 3)
        total_seconds = round(time.monotonic() - request_received, 3)
        result_uri = "/output/" + output_name
        output_size = _read_image_size(result["result_bytes"])
        decoded_size = list(output_size) if output_size else None
        effective_diffusion_size = (
            [diffusion_width, diffusion_height]
            if diffusion_width is not None and diffusion_height is not None
            else decoded_size
        )
        logger.info(
            "[QWEN] request_id=%s resolution=%sx%s steps=%s model_loaded=%s "
            "dtype=torch.%s device=%s inference_time=%.3fs total_time=%.3fs",
            request_id,
            width,
            height,
            steps,
            str(_worker_ready()).lower(),
            CONFIG.dtype,
            CONFIG.device,
            result["inference_seconds"],
            total_seconds,
        )
        logger.info(
            "[QWEN] request_id=%s cfg_scale=%.2f negative_prompt_sent=%s "
            "resolution_mode=%s diffusion_size=%s output_size=%s",
            request_id,
            cfg_scale,
            str(negative_prompt_sent).lower(),
            resolution_mode,
            effective_diffusion_size,
            decoded_size,
        )
        payload = {
            "status": "success",
            "service": SERVICE_NAME,
            "mode": QWEN_MODE,
            "request_id": request_id,
            "source_run_id": source_id,
            "refine_result_uri": result_uri,
            "final_asset_uri": result_uri,
            "result_uri": result_uri,
            "steps": steps,
            "seed": seed,
            "true_cfg_scale": cfg_scale,
            "cfg_scale": cfg_scale,
            "negative_prompt_sent": negative_prompt_sent,
            "resolution_mode": resolution_mode,
            "diffusion_size": effective_diffusion_size,
            "decoded_size": decoded_size,
            "saved_size": decoded_size,
            "auto_straighten": auto_straighten,
            "auto_straighten_applied": False,
            "worker_model": MODEL_LABEL,
            "prompt_id": result["prompt_id"],
            "elapsed_ms": round(total_seconds * 1000, 2),
            "input_source": "visible_fish_refined",
            "performance": {
                "inference_time": result["inference_seconds"],
                "download_time": result["download_seconds"],
                "save_time": save_seconds,
                "total_time": total_seconds,
            },
            "runtime": {
                "model_loaded": True,
                "gpu": GPU_NAME,
                "device": CONFIG.device,
                "dtype": CONFIG.dtype,
                "precision_mode": CONFIG.precision_mode,
                "width": width,
                "height": height,
                "cfg_scale": cfg_scale,
                "negative_prompt_sent": negative_prompt_sent,
                "resolution_mode": resolution_mode,
                "diffusion_size": effective_diffusion_size,
                "decoded_size": decoded_size,
            },
        }
        return JSONResponse(payload)
    except WorkerError as exc:
        logger.exception(
            "[QWEN] request_id=%s failed total_time=%.3fs",
            request_id,
            time.monotonic() - request_received,
        )
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(
            "[QWEN] request_id=%s unexpected failure total_time=%.3fs",
            request_id,
            time.monotonic() - request_received,
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.exception_handler(WorkerError)
def worker_error_handler(_, exc: WorkerError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("worker:app", host="0.0.0.0", port=int(os.getenv("PORT", "8002")), workers=1)
