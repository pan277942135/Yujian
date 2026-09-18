"""Independent Qwen-Image-Edit-2511 worker for Fish Portrait Lab.

The service accepts the SAM Visible fish as its only image input, patches the
validated ComfyUI API workflow, waits for the ComfyUI history record, and
persists a browser-readable result under /output.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("fish-qwen-refine-worker")

SERVICE_NAME = "fish-qwen-refine-worker"
MODEL_LABEL = "Qwen-Image-Edit-2511"
QWEN_MODE = "fish_preserve_refine_qwen_v1"
DEFAULT_WORKFLOW_PATH = "/opt/fish-qwen-refine-worker/qwen2511_api.json"
DEFAULT_OUTPUT_DIR = "/opt/fish-qwen-refine-worker/output"
DEFAULT_COMFY_URL = "http://127.0.0.1:8188"
DEFAULT_TIMEOUT_SECONDS = 1200.0
DEFAULT_POLL_SECONDS = 2.0
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

OUTPUT_DIR = Path(os.getenv("QWEN_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
WORKFLOW_PATH = Path(os.getenv("QWEN_WORKFLOW_PATH", DEFAULT_WORKFLOW_PATH))
COMFY_URL = os.getenv("COMFYUI_URL", DEFAULT_COMFY_URL).rstrip("/")
WORKER_TOKEN = os.getenv("QWEN_REFINE_WORKER_TOKEN", "").strip()
REQUEST_TIMEOUT = max(
    30.0,
    float(os.getenv("QWEN_REQUEST_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))),
)
POLL_SECONDS = max(0.5, float(os.getenv("QWEN_POLL_SECONDS", str(DEFAULT_POLL_SECONDS))))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app = FastAPI(title=SERVICE_NAME, version="1.0.0")
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")


class WorkerError(RuntimeError):
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


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
        raise WorkerError("ComfyUI %s %s: %s" % (method, path, detail or exc, 502)) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise WorkerError("ComfyUI %s %s unavailable: %s" % (method, path, exc), 502) from exc
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkerError("ComfyUI returned non-JSON for %s %s" % (method, path), 502) from exc


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
        raise WorkerError("ComfyUI image upload failed: %s" % (detail or exc), 502) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise WorkerError("ComfyUI image upload unavailable: %s" % exc, 502) from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkerError("ComfyUI image upload returned non-JSON", 502) from exc
    if not payload.get("name"):
        raise WorkerError("ComfyUI image upload did not return a name", 502)
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
        raise WorkerError("ComfyUI result download failed: %s" % (detail or exc), 502) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise WorkerError("ComfyUI result download unavailable: %s" % exc, 502) from exc


def _load_workflow() -> dict[str, Any]:
    try:
        return json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise WorkerError("workflow file not found: %s" % WORKFLOW_PATH, 500) from exc
    except json.JSONDecodeError as exc:
        raise WorkerError("workflow file is invalid JSON: %s" % exc, 500) from exc


def _set_input(workflow: dict[str, Any], node_id: str, key: str, value: Any) -> None:
    node = workflow.get(node_id)
    if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict):
        raise WorkerError("workflow node %s is missing inputs" % node_id, 500)
    node["inputs"][key] = value


def _wait_for_output(prompt_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + REQUEST_TIMEOUT
    while time.monotonic() < deadline:
        history = _json_request("GET", "/history/" + urllib.parse.quote(prompt_id, safe=""))
        entry = history.get(prompt_id) or history.get(str(prompt_id))
        if entry:
            status = entry.get("status") or {}
            if status.get("status_str") == "error" or status.get("completed") is False and status.get("status_str") == "error":
                messages = status.get("messages") or []
                raise WorkerError("ComfyUI workflow failed: %s" % messages, 502)
            outputs = entry.get("outputs") or {}
            result = outputs.get("195") or {}
            images = result.get("images") or []
            if images:
                return images[0]
        time.sleep(POLL_SECONDS)
    raise WorkerError("ComfyUI workflow timed out after %.0f seconds" % REQUEST_TIMEOUT, 504)


def _safe_prefix(source_run_id: str | None) -> str:
    raw = "".join(char if char.isalnum() or char in "-_" else "_" for char in (source_run_id or "run"))
    return raw[:48] or "run"


@app.get("/health", dependencies=[Depends(_require_token)])
def health() -> dict[str, str]:
    return {"status": "ok", "service": SERVICE_NAME, "model": MODEL_LABEL}


@app.post("/refine", dependencies=[Depends(_require_token)])
def refine(
    image: UploadFile = File(...),
    params: str = Form(default="{}"),
    mode: str = Form(default=QWEN_MODE),
    source_run_id: str = Form(default=""),
) -> JSONResponse:
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
        steps = int(options.get("steps") if options.get("steps") is not None else DEFAULT_STEPS)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="steps must be an integer") from exc
    if not 1 <= steps <= 100:
        raise HTTPException(status_code=422, detail="steps must be between 1 and 100")
    raw_seed = options.get("seed")
    try:
        seed = int(raw_seed) if raw_seed not in (None, "") else secrets.randbelow(2**63 - 1)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="seed must be an integer or null") from exc
    auto_straighten = bool(options.get("auto_straighten", False))
    prompt = str(options.get("prompt") or DEFAULT_PROMPT).strip()
    negative_prompt = str(options.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT).strip()
    source_id = str(options.get("source_run_id") or source_run_id or "") or None
    content = image.file.read()
    if not content:
        raise HTTPException(status_code=422, detail="image is empty")
    content_type = image.content_type or "image/png"
    original_filename = Path(image.filename or "sam_visible.png").name
    uploaded_name = "SAM_VISIBLE_" + secrets.token_hex(8) + "_" + original_filename
    started = time.monotonic()
    try:
        uploaded = _multipart_upload(uploaded_name, content, content_type)
        uploaded_file = uploaded["name"]
        if uploaded.get("subfolder"):
            uploaded_file = str(uploaded["subfolder"]).strip("/") + "/" + uploaded_file
        workflow = _load_workflow()
        _set_input(workflow, "41", "image", uploaded_file)
        _set_input(workflow, "170:151", "prompt", prompt)
        _set_input(workflow, "170:149", "prompt", negative_prompt)
        _set_input(workflow, "170:169", "seed", seed)
        _set_input(workflow, "170:165", "value", 4)
        _set_input(workflow, "170:166", "value", steps)
        # The validated workflow keeps the Lightning branch opt-in. The page
        # uses the non-Lightning path so Steps remains user-controlled.
        _set_input(workflow, "170:168", "value", False)
        output_prefix = "Qwen_Refine_" + _safe_prefix(source_id) + "_" + secrets.token_hex(4)
        _set_input(workflow, "195", "filename_prefix", output_prefix)
        prompt_response = _json_request(
            "POST",
            "/prompt",
            {"prompt": workflow, "client_id": "fish-qwen-refine-worker"},
        )
        prompt_id = prompt_response.get("prompt_id")
        if not prompt_id:
            raise WorkerError("ComfyUI /prompt did not return prompt_id", 502)
        result_ref = _wait_for_output(str(prompt_id))
        result_bytes = _download_view(result_ref)
        output_name = "REFINE_" + secrets.token_hex(12) + ".png"
        output_path = OUTPUT_DIR / output_name
        output_path.write_bytes(result_bytes)
        elapsed_ms = round((time.monotonic() - started) * 1000, 2)
        result_uri = "/output/" + output_name
        payload = {
            "status": "success",
            "service": SERVICE_NAME,
            "mode": QWEN_MODE,
            "source_run_id": source_id,
            "refine_result_uri": result_uri,
            "final_asset_uri": result_uri,
            "result_uri": result_uri,
            "steps": steps,
            "seed": seed,
            "auto_straighten": auto_straighten,
            "auto_straighten_applied": False,
            "worker_model": MODEL_LABEL,
            "prompt_id": str(prompt_id),
            "elapsed_ms": elapsed_ms,
            "input_source": "sam_visible",
        }
        logger.info(
            "qwen_refine_success source_run_id=%s prompt_id=%s steps=%s seed=%s elapsed_ms=%s",
            source_id,
            prompt_id,
            steps,
            seed,
            elapsed_ms,
        )
        return JSONResponse(payload)
    except WorkerError as exc:
        logger.exception("qwen_refine_failed source_run_id=%s", source_id)
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@app.exception_handler(WorkerError)
def worker_error_handler(_, exc: WorkerError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"status": "error", "message": str(exc)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("worker:app", host="0.0.0.0", port=int(os.getenv("PORT", "8002")), workers=1)
