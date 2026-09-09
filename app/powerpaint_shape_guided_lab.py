"""PowerPaint official Shape-guided Object Inpainting experiment.

This module is deliberately separate from all fish-completion production and
debug pipelines. It builds a small crop, a semantic fish completion mask, and
sends the Shape Guided contract directly to the existing Worker without
changing the shared Worker client.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import secrets
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from google.cloud import storage
from PIL import Image
from sqlalchemy import select

from app.dataset_models import DatasetItem
from app.db import get_db
from app.detector_runtime import detect, normalize_android_source
from app.models import DatasetVersion
from app.recognition_pipeline import assess_detections
from app.segmentation.service import generate_fish_cutout

router = APIRouter(tags=["powerpaint-shape-guided-lab"])
templates = Jinja2Templates(directory="app/templates")
VERSION = "POWERPAINT_SHAPE_GUIDED_V0.3"
PROMPT_ID = "FIXED_FISH_SHAPE_GUIDED_V0.3"
TASK_MODE = "SHAPE_GUIDED"
FITTING_DEGREES = (0.6, 0.8, 0.95)
PREFIX = "powerpaint_shape_guided_lab/v0.3"
MAX_BYTES = 25 * 1024 * 1024
logger = logging.getLogger(__name__)
PROMPT = """Complete only the missing biological parts of this fish.

Use the existing visible fish as the identity reference.

Preserve:
- species
- head shape
- body proportion
- scale pattern
- fin structure
- tail anatomy
- original color

Only generate inside the provided completion mask.
Do not modify visible fish pixels.
Do not change background.
Do not create another fish."""


def _bucket():
    name = os.getenv("GCS_BUCKET", "").strip()
    return storage.Client().bucket(name) if name else None


def _persist(test_id: str, name: str, data: bytes, content_type: str) -> str:
    bucket = _bucket()
    object_name = f"{PREFIX}/{test_id}/{name}"
    if bucket is None:
        path = os.path.join("var", "powerpaint_shape_guided_lab", test_id, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(data)
        return f"local://{path}"
    bucket.blob(object_name).upload_from_string(data, content_type=content_type)
    return f"gs://{bucket.name}/{object_name}"


def _read_dataset_image(item: DatasetItem) -> bytes:
    uri = (item.gcs_uri or "").strip()
    if not uri.startswith("gs://") or "/" not in uri[5:]:
        raise HTTPException(422, "DATASET_IMAGE_URI_INVALID")
    bucket_name, object_name = uri[5:].split("/", 1)
    try:
        return storage.Client().bucket(bucket_name).blob(object_name).download_as_bytes(timeout=120)
    except Exception as exc:
        raise HTTPException(502, "DATASET_IMAGE_READ_FAILED") from exc


def _png(image: Image.Image) -> bytes:
    out = io.BytesIO()
    image.save(out, format="PNG", optimize=True)
    return out.getvalue()


def _mask_png(mask: np.ndarray) -> bytes:
    return _png(Image.fromarray(np.where(mask, 255, 0).astype("uint8"), "L"))


def _data_url(data: bytes, media_type: str) -> str:
    return f"data:{media_type};base64,{base64.b64encode(data).decode()}"


def _decode_data_url(value: str | None) -> bytes | None:
    if not isinstance(value, str) or not value.startswith("data:") or "," not in value:
        return None
    try:
        return base64.b64decode(value.split(",", 1)[1])
    except (ValueError, TypeError):
        return None


def _read_uri(uri: str) -> bytes:
    if uri.startswith("gs://"):
        bucket_name, object_name = uri[5:].split("/", 1)
        return storage.Client().bucket(bucket_name).blob(object_name).download_as_bytes(timeout=120)
    if uri.startswith("local://"):
        with open(uri.removeprefix("local://"), "rb") as handle:
            return handle.read()
    if uri.startswith(("http://", "https://")):
        with urllib.request.urlopen(uri, timeout=120) as response:
            return response.read()
    raise ValueError(f"unsupported result URI: {uri}")


def _runtime(test_id: str) -> dict[str, Any]:
    return {"test_id": test_id, "timestamp": datetime.now(timezone.utc).isoformat(), "service": os.getenv("K_SERVICE", "unknown"), "revision": os.getenv("K_REVISION", "unknown"), "commit": os.getenv("APP_GIT_COMMIT", "unknown")}


def _new_test_id() -> str:
    return "PSG_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(2)


def _datasets(db):
    rows = db.scalars(select(DatasetVersion).where(DatasetVersion.status == "FROZEN").order_by(DatasetVersion.created_at.desc())).all()
    return [{"dataset_version": row.dataset_version, "status": row.status, "image_count": row.train_count + row.val_count + row.test_count, "train_count": row.train_count, "val_count": row.val_count, "test_count": row.test_count} for row in rows]


def _images(db, dataset_version: str, split: str | None, limit: int, offset: int):
    dataset = db.get(DatasetVersion, dataset_version)
    if not dataset:
        raise HTTPException(404, "DATASET_VERSION_NOT_FOUND")
    if dataset.status != "FROZEN":
        raise HTTPException(409, "DATASET_VERSION_NOT_FROZEN")
    stmt = select(DatasetItem).where(DatasetItem.dataset_version == dataset_version)
    if split:
        stmt = stmt.where(DatasetItem.split == split)
    rows = db.scalars(stmt.order_by(DatasetItem.id).offset(offset).limit(limit)).all()
    return [{"dataset_version": row.dataset_version, "dataset_item_id": row.id, "image_id": row.image_id, "species": row.species_name, "split": row.split, "preview_url": f"/media/{row.batch_id}/{row.image_id}"} for row in rows]


def _crop_box(box, width: int, height: int) -> tuple[int, int, int, int]:
    b = box.normalized()
    x1 = max(0, min(width - 1, int(np.floor(b.x1 * width))))
    y1 = max(0, min(height - 1, int(np.floor(b.y1 * height))))
    x2 = max(x1 + 1, min(width, int(np.ceil(b.x2 * width))))
    y2 = max(y1 + 1, min(height, int(np.ceil(b.y2 * height))))
    return x1, y1, x2, y2


def _binary_dilation(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    for _ in range(max(0, int(iterations))):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = (
            padded[:-2, :-2] | padded[:-2, 1:-1] | padded[:-2, 2:]
            | padded[1:-1, :-2] | padded[1:-1, 1:-1] | padded[1:-1, 2:]
            | padded[2:, :-2] | padded[2:, 1:-1] | padded[2:, 2:]
        )
    return result


def _binary_fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed background pixels with a small border flood fill."""
    source = np.asarray(mask, dtype=bool)
    if source.ndim != 2:
        raise ValueError("mask must be 2D")
    background = ~source
    reachable = np.zeros_like(source, dtype=bool)
    height, width = source.shape
    queue: list[tuple[int, int]] = []
    for x in range(width):
        queue.extend(((0, x), (height - 1, x)))
    for y in range(height):
        queue.extend(((y, 0), (y, width - 1)))
    while queue:
        y, x = queue.pop()
        if y < 0 or y >= height or x < 0 or x >= width or reachable[y, x] or not background[y, x]:
            continue
        reachable[y, x] = True
        queue.extend(((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)))
    return source | (background & ~reachable)


def build_completion_mask(visible_mask: np.ndarray, *, max_ratio: float = 0.20) -> np.ndarray:
    """Create a small biological envelope gap mask, never overlapping visible fish."""
    visible = np.asarray(visible_mask, dtype=bool)
    if visible.ndim != 2:
        raise ValueError("visible mask must be 2D")
    # Hole filling estimates an envelope only where visible fish surrounds a gap.
    envelope = _binary_fill_holes(visible)
    # A narrow local contour band makes complete-fish cases testable without
    # opening the whole crop to the Worker. It is not a large edit region.
    band = _binary_dilation(visible, iterations=2)
    candidate = (envelope | band) & ~visible
    denominator = int((envelope | visible).sum())
    if denominator and candidate.sum() / denominator > max_ratio:
        # Keep the closest boundary pixels only, deterministically.
        ys, xs = np.where(candidate)
        boundary = _binary_dilation(visible, iterations=1) & ~visible
        distance = np.full(visible.shape, visible.shape[0] + visible.shape[1], dtype=np.int32)
        frontier = list(zip(*np.where(boundary)))
        for y, x in frontier:
            distance[y, x] = 1
        for step in range(2, max(visible.shape) + 2):
            next_frontier: list[tuple[int, int]] = []
            for y, x in frontier:
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < visible.shape[0] and 0 <= nx < visible.shape[1] and distance[ny, nx] > step:
                        distance[ny, nx] = step
                        next_frontier.append((ny, nx))
            if not next_frontier:
                break
            frontier = next_frontier
        order = np.argsort(distance[ys, xs])
        keep = max(1, int(denominator * max_ratio))
        limited = np.zeros_like(candidate)
        limited[ys[order[:keep]], xs[order[:keep]]] = True
        candidate = limited & ~visible
    return candidate


def shape_guided_report_entry(*, fitting_degree: float, visible_pixel_change_ratio: float | None, completion_area_ratio: float, generated_area_pixels: int, status: str, fish_identity_check: str = "PENDING", background_change: str = "PENDING") -> dict[str, Any]:
    return {
        "task_mode": TASK_MODE,
        "fitting_degree": float(fitting_degree),
        "visible_pixel_change_ratio": visible_pixel_change_ratio,
        "completion_area_ratio": float(completion_area_ratio),
        "generated_area_pixels": int(generated_area_pixels),
        "fish_identity_check": fish_identity_check,
        "background_change": background_change,
        "status": status,
    }


def normalize_fitting_degrees(requested: Any) -> list[float]:
    values = requested if isinstance(requested, (list, tuple, set)) else str(requested or "").split(",")
    degrees = sorted({float(value) for value in values if str(value).strip()})
    if not degrees or any(value not in FITTING_DEGREES for value in degrees):
        raise ValueError("支持 0.6、0.8、0.95")
    return degrees


def validate_completion_mask(mask: np.ndarray, visible: np.ndarray, *, max_ratio: float = 0.20) -> dict[str, Any]:
    completion = np.asarray(mask, dtype=bool)
    visible = np.asarray(visible, dtype=bool)
    if completion.shape != visible.shape:
        raise ValueError("completion and visible masks must have identical shape")
    overlap = int((completion & visible).sum())
    area = int(completion.sum())
    ratio = area / max(int((completion | visible).sum()), 1)
    return {"valid": overlap == 0 and ratio <= max_ratio, "completion_area_pixels": area, "completion_area_ratio": round(ratio, 6), "visible_overlap_pixels": overlap}


def _worker_base_url() -> str:
    return os.getenv("FISH_COMPLETION_WORKER_URL", "").strip().rstrip("/")


def _invoke_shape_guided(*, image_uri: str, mask_uri: str, fitting_degree: float) -> dict[str, Any]:
    base_url = _worker_base_url()
    if not base_url:
        raise RuntimeError("COMPLETION_WORKER_NOT_CONFIGURED")
    payload = json.dumps({"task": "fish_completion", "task_mode": TASK_MODE, "fitting_degree": fitting_degree, "image_uri": image_uri, "mask_uri": mask_uri, "image": image_uri, "mask": mask_uri, "prompt": PROMPT}, separators=(",", ":")).encode()
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    token = os.getenv("FISH_COMPLETION_WORKER_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"{base_url}/completion", data=payload, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=float(os.getenv("FISH_COMPLETION_WORKER_TIMEOUT_SECONDS", "900"))) as response:
            status_code = response.status
            body = response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"SHAPE_GUIDED_WORKER_HTTP_{exc.code}: {exc.read().decode('utf-8', 'replace')[:2000]}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"SHAPE_GUIDED_WORKER_UNREACHABLE: {exc}") from exc
    try:
        result = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"SHAPE_GUIDED_WORKER_INVALID_JSON_HTTP_{status_code}") from exc
    if not isinstance(result, dict):
        raise RuntimeError("SHAPE_GUIDED_WORKER_INVALID_RESPONSE")
    result["result_uri"] = result.get("result_uri") or result.get("generated_roi_uri") or result.get("output_uri")
    result["generated_roi"] = result.get("generated_roi")
    if not result.get("result_uri") and not _decode_data_url(result.get("generated_roi")):
        raise RuntimeError("SHAPE_GUIDED_WORKER_MISSING_RESULT_URI")
    return result


def _compose(original: Image.Image, generated: Image.Image, completion_mask: np.ndarray) -> tuple[bytes, float]:
    base = np.asarray(original.convert("RGB"), dtype=np.uint8)
    out = np.asarray(generated.convert("RGB").resize(original.size), dtype=np.uint8)
    mask = np.asarray(completion_mask, dtype=bool)
    final = base.copy()
    final[mask] = out[mask]
    visible = ~mask
    changed = np.any(final != base, axis=2)
    ratio = float(changed[visible].sum()) / max(int(visible.sum()), 1)
    return _png(Image.fromarray(final, "RGB")), round(ratio, 6)


def _error(status: int, test_id: str, stage: str, code: str, message: str, report: dict[str, Any] | None = None) -> JSONResponse:
    body: dict[str, Any] = {"status": "error", "stage": stage, "test_id": test_id, "error": {"error_code": code, "message": message}}
    if report is not None:
        body["report"] = report
    return JSONResponse(status_code=status, content=body)


async def _payload(request: Request) -> dict[str, Any]:
    if "application/json" in (request.headers.get("content-type") or "").lower():
        data = await request.json()
    else:
        data = dict(await request.form())
    if not isinstance(data, dict):
        raise ValueError("request payload must be an object")
    return data


@router.get("/debug/powerpaint-shape-guided-lab", response_class=HTMLResponse)
def page(request: Request):
    return templates.TemplateResponse(request=request, name="powerpaint_shape_guided_lab.html", context={})


@router.get("/api/debug/powerpaint-shape-guided-lab/datasets")
def datasets(db=Depends(get_db)):
    return _datasets(db)


@router.get("/api/debug/powerpaint-shape-guided-lab/datasets/{dataset_version}/images")
def images(dataset_version: str, split: str | None = None, limit: int = Query(60, ge=1, le=200), offset: int = Query(0, ge=0), db=Depends(get_db)):
    return _images(db, dataset_version, split, limit, offset)


@router.post("/api/debug/powerpaint-shape-guided-lab/run")
async def run(request: Request, db=Depends(get_db)):
    started = time.perf_counter()
    test_id = _new_test_id()
    source = None
    try:
        data = await _payload(request)
        dataset_version = str(data.get("dataset_version") or "").strip()
        item_id = int(data.get("dataset_item_id"))
        requested = data.get("fitting_degrees") or data.get("fitting_degree") or [str(x) for x in FITTING_DEGREES]
        try:
            degrees = normalize_fitting_degrees(requested)
        except ValueError as exc:
            return _error(400, test_id, "input", "INVALID_FITTING_DEGREE", str(exc))
        item = db.scalar(select(DatasetItem).where(DatasetItem.dataset_version == dataset_version, DatasetItem.id == item_id))
        dataset = db.get(DatasetVersion, dataset_version)
        if not dataset or dataset.status != "FROZEN":
            return _error(409, test_id, "input", "DATASET_VERSION_NOT_FROZEN", dataset_version)
        if not item:
            return _error(404, test_id, "input", "DATASET_ITEM_NOT_FOUND", str(item_id))
        raw = _read_dataset_image(item)
        with Image.open(io.BytesIO(raw)) as uploaded:
            source = normalize_android_source(uploaded)
        detector_run = detect(source)
        assessment = assess_detections(detector_run.detections)
        if assessment.primary is None:
            return _error(422, test_id, "detector", "NO_RELIABLE_PRIMARY_FISH", assessment.status.value)
        primary = assessment.primary
        x1, y1, x2, y2 = _crop_box(primary.box, source.width, source.height)
        crop = source.crop((x1, y1, x2, y2)).convert("RGB")
        segmentation = generate_fish_cutout(source, primary.box)
        visible_full = np.asarray(segmentation.mask, dtype=bool)
        visible = visible_full[y1:y2, x1:x2]
        completion = build_completion_mask(visible)
        validation = validate_completion_mask(completion, visible)
        if not validation["valid"]:
            return _error(422, test_id, "completion_mask", "INVALID_COMPLETION_MASK", json.dumps(validation))
        original_bytes = _png(crop)
        detector_uri = _persist(test_id, "detector_crop.png", original_bytes, "image/png")
        sam_visible_bytes = _png(Image.fromarray(np.where(visible, np.asarray(crop), 0).astype("uint8"), "RGB"))
        sam_visible_uri = _persist(test_id, "sam_visible.png", sam_visible_bytes, "image/png")
        sam_mask_uri = _persist(test_id, "sam_mask.png", _mask_png(visible), "image/png")
        completion_mask_uri = _persist(test_id, "completion_mask.png", _mask_png(completion), "image/png")
        report: dict[str, Any] = {
            "experiment": VERSION,
            "runtime": _runtime(test_id),
            "input": {"dataset_version": dataset_version, "dataset_item_id": item.id, "image_id": item.image_id, "width": crop.width, "height": crop.height, "source_type": "dataset_freeze", "image_hash": hashlib.sha256(raw).hexdigest()[:16]},
            "detector": {"model": detector_run.model_version, "confidence": float(primary.confidence), "bbox_pixels": [x1, y1, x2, y2], "assessment": assessment.status.value},
            "sam": {"model": "SAM_VIT_B", "quality": segmentation.quality.value, "mask_area_pixels": int(visible.sum())},
            "task_mode": TASK_MODE,
            "prompt_id": PROMPT_ID,
            "completion_mask": validation,
            "assets": {"original": detector_uri, "detector_crop": detector_uri, "sam_visible": sam_visible_uri, "sam_mask": sam_mask_uri, "completion_mask": completion_mask_uri},
            "preview_original": _data_url(original_bytes, "image/png"),
            "preview_sam": _data_url(sam_visible_bytes, "image/png"),
            "preview_completion_mask": _data_url(_mask_png(completion), "image/png"),
            "results": [],
            "progress": [
                {"stage": "input", "label": "Input", "status": "READY"},
                {"stage": "detector", "label": "Detector", "status": "READY", "result": assessment.status.value},
                {"stage": "sam", "label": "SAM", "status": "READY", "result": segmentation.quality.value},
                {"stage": "completion_mask", "label": "Completion Mask", "status": "READY", "result": validation},
                {"stage": "shape_guided", "label": "PowerPaint Shape Guided", "status": "RUNNING"},
            ],
            "timings": {"total_ms": None},
        }
        for degree in degrees:
            result_started = time.perf_counter()
            item_result: dict[str, Any] = {"task_mode": TASK_MODE, "fitting_degree": degree, "status": "PENDING", "visible_pixel_change_ratio": None, "generated_area_pixels": int(completion.sum()), "result_uri": None}
            try:
                worker = _invoke_shape_guided(image_uri=detector_uri, mask_uri=completion_mask_uri, fitting_degree=degree)
                generated = _decode_data_url(worker.get("result_uri")) or _decode_data_url(worker.get("generated_roi"))
                if generated is None and worker.get("result_uri"):
                    generated = _read_uri(worker["result_uri"])
                if not generated:
                    raise RuntimeError("SHAPE_GUIDED_WORKER_EMPTY_OUTPUT")
                output_uri = _persist(test_id, f"powerpaint_output_{degree:g}.png", generated, "image/png")
                with Image.open(io.BytesIO(generated)) as generated_image:
                    final_bytes, visible_change = _compose(crop, generated_image, completion)
                final_uri = _persist(test_id, f"final_result_{degree:g}.png", final_bytes, "image/png")
                item_result.update({"status": "SUCCESS", "result_uri": worker.get("result_uri"), "output_asset": output_uri, "final_asset": final_uri, "worker_ms": round((time.perf_counter() - result_started) * 1000, 2), "inference_time_ms": worker.get("inference_time_ms"), "visible_pixel_change_ratio": visible_change, "fish_identity_check": "PENDING", "background_change": "PENDING"})
                report["assets"][f"powerpaint_output_{degree:g}"] = output_uri
                report["assets"][f"final_result_{degree:g}"] = final_uri
                item_result["shape_guided_report"] = shape_guided_report_entry(
                    fitting_degree=degree,
                    visible_pixel_change_ratio=visible_change,
                    completion_area_ratio=validation["completion_area_ratio"],
                    generated_area_pixels=validation["completion_area_pixels"],
                    status="SUCCESS",
                )
                report["result_preview"] = _data_url(final_bytes, "image/png")
            except Exception as exc:
                item_result.update({"status": "SHAPE_GUIDED_FAILED", "error_code": "SHAPE_GUIDED_FAILED", "error": str(exc), "worker_ms": round((time.perf_counter() - result_started) * 1000, 2)})
            report["results"].append(item_result)
        successful = [x for x in report["results"] if x["status"] == "SUCCESS"]
        if successful:
            best = successful[-1]
            output_alias = _persist(test_id, "powerpaint_output.png", _read_uri(best["output_asset"]), "image/png")
            final_alias = _persist(test_id, "final_result.png", _read_uri(best["final_asset"]), "image/png")
            report["assets"]["powerpaint_output"] = output_alias
            report["assets"]["final_result"] = final_alias
            report["status"] = "SUCCESS"
            report["shape_guided_report"] = shape_guided_report_entry(
                fitting_degree=best["fitting_degree"],
                visible_pixel_change_ratio=best["visible_pixel_change_ratio"],
                completion_area_ratio=validation["completion_area_ratio"],
                generated_area_pixels=validation["completion_area_pixels"],
                status="SUCCESS",
            )
        else:
            report["status"] = "SHAPE_GUIDED_FAILED"
            report["shape_guided_report"] = shape_guided_report_entry(
                fitting_degree=degrees[-1],
                visible_pixel_change_ratio=None,
                completion_area_ratio=validation["completion_area_ratio"],
                generated_area_pixels=validation["completion_area_pixels"],
                status="SHAPE_GUIDED_FAILED",
            )
        report["progress"][-1].update({"status": "SUCCESS" if successful else "SHAPE_GUIDED_FAILED", "result": {"successful_degrees": [x["fitting_degree"] for x in successful]}})
        report["timings"]["total_ms"] = round((time.perf_counter() - started) * 1000, 2)
        report_bytes = json.dumps(report, ensure_ascii=False, indent=2).encode()
        report_uri = _persist(test_id, "shape_guided_report.json", report_bytes, "application/json")
        _persist(test_id, "report.json", report_bytes, "application/json")
        report["assets"]["report"] = report_uri
        return {"status": "ok" if successful else "error", "stage": "shape_guided_complete" if successful else "shape_guided", "test_id": test_id, "report": report}
    except (TypeError, ValueError) as exc:
        return _error(400, test_id, "input", "INVALID_REQUEST", str(exc))
    except HTTPException as exc:
        return _error(exc.status_code, test_id, "input", str(exc.detail), str(exc.detail))
    except Exception as exc:
        logger.exception("Shape Guided Lab failed test_id=%s", test_id)
        return _error(500, test_id, "shape_guided", "SHAPE_GUIDED_FAILED", f"{exc.__class__.__name__}: {exc}")
    finally:
        if source is not None:
            source.close()
