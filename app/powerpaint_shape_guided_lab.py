"""PowerPaint official Shape-guided Object Inpainting experiment.

This module is deliberately separate from all fish-completion production and
debug pipelines. It builds a small crop, a semantic fish completion mask, and
sends the Shape Guided contract directly to the existing Worker without
changing the shared Worker client.
"""
from __future__ import annotations

import base64
import hashlib
from collections import deque
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
PROMPT_ID = "FIXED_FISH_SHAPE_GUIDED_V0.3.1"
TASK_MODE = "SHAPE_GUIDED"
FITTING_DEGREES = (0.6, 0.8, 0.95)
PREFIX = "experiments/powerpaint_shape_guided_lab/v0.3"
MAX_BYTES = 25 * 1024 * 1024
logger = logging.getLogger(__name__)
EXPERIMENT_STAGES = (
    "INIT",
    "INPUT_READY",
    "DETECTOR_READY",
    "SAM_READY",
    "RAW_SAM_READY",
    "REFINED_VISIBLE_READY",
    "MASK_READY",
    "COMPLETION_MASK_READY",
    "WORKER_READY",
    "POWERPAINT_RUNNING",
    "FINAL_COMPOSE_READY",
    "SUCCESS",
    "FAILED",
)
PROMPT = "a realistic fish body matching the visible fish"
P2_DEFAULT_FITTING_DEGREE = 0.8
MASK_MODES = ("AUTO_V1", "MANUAL_V2")
QUALITY_FIELDS = ("BODY_CONTINUITY", "SCALE_TEXTURE", "COLOR_MATCH", "EDGE_SEAM", "ANATOMY", "BACKGROUND_PRESERVATION")
QUALITY_VALUES = ("PASS", "WARNING", "FAIL", "UNRATED")


def _quality_scores(data: dict[str, Any]) -> dict[str, str]:
    scores: dict[str, str] = {}
    for field in QUALITY_FIELDS:
        value = str(data.get(field.lower()) or data.get(field) or "UNRATED").strip().upper()
        scores[field] = value if value in QUALITY_VALUES else "UNRATED"
    return scores


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


def _persist_json(test_id: str, name: str, value: Any) -> str:
    return _persist(test_id, name, json.dumps(value, ensure_ascii=False, indent=2, default=str).encode("utf-8"), "application/json")


def _placeholder_png() -> bytes:
    return _png(Image.new("RGB", (1, 1), (0, 0, 0)))


class ExperimentFailure(RuntimeError):
    def __init__(self, http_status: int, stage: str, error_code: str, message: str, classification: str = "FAILED_VALIDATION"):
        super().__init__(message)
        self.http_status = http_status
        self.stage = stage
        self.error_code = error_code
        self.classification = classification


def _set_stage(report: dict[str, Any], stage: str) -> None:
    if stage not in EXPERIMENT_STAGES:
        raise ValueError(f"unknown experiment stage: {stage}")
    report["experiment_stage"] = stage
    if stage == "FAILED":
        report["status"] = "FAILED"
    elif stage == "SUCCESS":
        report["status"] = "SUCCESS"


def _safe_persist(test_id: str, name: str, data: bytes, content_type: str, report: dict[str, Any]) -> str | None:
    try:
        uri = _persist(test_id, name, data, content_type)
        report.setdefault("assets", {})[name.rsplit(".", 1)[0]] = uri
        return uri
    except Exception as exc:
        report.setdefault("persistence_errors", []).append({"asset": name, "error": f"{exc.__class__.__name__}: {exc}"})
        logger.exception("Shape Guided artifact persistence failed test_id=%s asset=%s", test_id, name)
        return None


def _safe_persist_json(test_id: str, name: str, value: Any, report: dict[str, Any] | None = None) -> str | None:
    try:
        uri = _persist_json(test_id, name, value)
        if report is not None:
            report.setdefault("assets", {})[name.rsplit(".", 1)[0]] = uri
        return uri
    except Exception as exc:
        if report is not None:
            report.setdefault("persistence_errors", []).append({"asset": name, "error": f"{exc.__class__.__name__}: {exc}"})
        logger.exception("Shape Guided JSON artifact persistence failed test_id=%s asset=%s", test_id, name)
        return None


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



def _visible_fish_png(crop: Image.Image, mask: np.ndarray) -> bytes:
    rgb = np.asarray(crop.convert("RGB"), dtype=np.uint8)
    visible = np.asarray(mask, dtype=bool)
    return _png(Image.fromarray(np.where(visible[..., None], rgb, 0).astype("uint8"), "RGB"))


def _decode_mask_data_url(value: Any, shape: tuple[int, int], field: str) -> np.ndarray:
    if not value:
        return np.zeros(shape, dtype=bool)
    data = _decode_data_url(str(value))
    if data is None:
        raise ValueError(f"{field} must be a PNG data URL")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image = image.convert("L")
            if image.size != (shape[1], shape[0]):
                raise ValueError(f"{field} dimensions must be {shape[1]}x{shape[0]}")
            return np.asarray(image, dtype=np.uint8) > 0
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"{field} is not a readable mask") from exc


def validate_manual_completion_mask(mask: np.ndarray, visible: np.ndarray) -> dict[str, Any]:
    completion = np.asarray(mask, dtype=bool)
    visible = np.asarray(visible, dtype=bool)
    if completion.shape != visible.shape:
        raise ValueError("completion and visible masks must have identical shape")
    overlap = int((completion & visible).sum())
    area = int(completion.sum())
    denominator = max(int((completion | visible).sum()), 1)
    ratio = area / denominator
    return {
        "valid": overlap == 0 and area > 0 and ratio <= 0.60,
        "warning": bool(0.35 < ratio <= 0.60),
        "completion_area_pixels": area,
        "completion_area_ratio": round(ratio, 6),
        "visible_overlap_pixels": overlap,
        "ratio_warning_threshold": 0.35,
        "ratio_fail_threshold": 0.60,
    }

def _prepare_input(db, dataset_version: str, item_id: int) -> dict[str, Any]:
    item = db.scalar(select(DatasetItem).where(DatasetItem.dataset_version == dataset_version, DatasetItem.id == item_id))
    dataset = db.get(DatasetVersion, dataset_version)
    if not dataset or dataset.status != "FROZEN":
        raise ExperimentFailure(409, "INIT", "DATASET_VERSION_NOT_FROZEN", dataset_version)
    if not item:
        raise ExperimentFailure(404, "INIT", "DATASET_ITEM_NOT_FOUND", str(item_id))
    raw = _read_dataset_image(item)
    with Image.open(io.BytesIO(raw)) as uploaded:
        source = normalize_android_source(uploaded)
    detector_run = detect(source)
    assessment = assess_detections(detector_run.detections)
    if assessment.primary is None:
        source.close()
        raise ExperimentFailure(422, "DETECTOR_READY", "NO_RELIABLE_PRIMARY_FISH", assessment.status.value, "FAILED_VALIDATION")
    primary = assessment.primary
    x1, y1, x2, y2 = _crop_box(primary.box, source.width, source.height)
    crop = source.crop((x1, y1, x2, y2)).convert("RGB")
    segmentation = generate_fish_cutout(source, primary.box)
    visible_full = np.asarray(segmentation.mask, dtype=bool)
    visible = visible_full[y1:y2, x1:x2]
    return {
        "item": item,
        "raw": raw,
        "source": source,
        "detector_run": detector_run,
        "assessment": assessment,
        "primary": primary,
        "bbox": [x1, y1, x2, y2],
        "crop": crop,
        "segmentation": segmentation,
        "visible": visible,
    }


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
    # Hole filling estimates an envelope only where visible fish surrounds a
    # gap. A dilation/ring is deliberately not a completion mask: a complete
    # fish must produce an empty mask and therefore skip PowerPaint.
    envelope = _binary_fill_holes(visible)
    candidate = envelope & ~visible
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


def _check_worker_health() -> dict[str, Any]:
    base_url = _worker_base_url()
    if not base_url:
        raise RuntimeError("SHAPE_GUIDED_WORKER_NOT_CONFIGURED")
    req = urllib.request.Request(f"{base_url}/health", method="GET", headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            status_code = response.status
            body = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"SHAPE_GUIDED_WORKER_HEALTH_UNREACHABLE: {exc}") from exc
    if status_code != 200:
        raise RuntimeError(f"SHAPE_GUIDED_WORKER_HEALTH_HTTP_{status_code}")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("SHAPE_GUIDED_WORKER_HEALTH_INVALID_JSON") from exc
    return {"status_code": status_code, **payload} if isinstance(payload, dict) else {"status_code": status_code}


def _invoke_shape_guided(*, image_uri: str, mask_uri: str, fitting_degree: float, visible_reference_uri: str | None = None) -> dict[str, Any]:
    base_url = _worker_base_url()
    if not base_url:
        raise RuntimeError("COMPLETION_WORKER_NOT_CONFIGURED")
    payload_data = {"task": "fish_completion", "task_mode": TASK_MODE, "fitting_degree": fitting_degree, "image_uri": image_uri, "mask_uri": mask_uri, "image": image_uri, "mask": mask_uri, "prompt": PROMPT}
    if visible_reference_uri:
        payload_data["visible_reference_uri"] = visible_reference_uri
        payload_data["visible_reference"] = visible_reference_uri
    payload = json.dumps(payload_data, separators=(",", ":")).encode()
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    token = os.getenv("FISH_COMPLETION_WORKER_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"{base_url}/completion", data=payload, method="POST", headers=headers)
    try:
        configured_timeout = float(os.getenv("FISH_COMPLETION_WORKER_TIMEOUT_SECONDS", "120"))
        with urllib.request.urlopen(req, timeout=min(max(configured_timeout, 1.0), 120.0)) as response:
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
    result["http_status"] = status_code
    result["result_uri"] = result.get("result_uri") or result.get("generated_roi_uri") or result.get("output_uri")
    result["generated_roi"] = result.get("generated_roi")
    if not result.get("result_uri") and not _decode_data_url(result.get("generated_roi")):
        raise RuntimeError("SHAPE_GUIDED_WORKER_MISSING_RESULT_URI")
    return result



def _mask_bbox(mask: np.ndarray) -> dict[str, int] | None:
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return {"x": x1, "y": y1, "width": x2 - x1 + 1, "height": y2 - y1 + 1, "x2": x2, "y2": y2}


def _mask_contour(mask: np.ndarray) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    if not source.any():
        return np.zeros_like(source)
    padded = np.pad(source, 1, mode="constant", constant_values=False)
    eroded = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return source & ~eroded


def _distance_map(target: np.ndarray) -> np.ndarray:
    source = np.asarray(target, dtype=bool)
    distance = np.full(source.shape, -1, dtype=np.int32)
    ys, xs = np.where(source)
    queue: deque[tuple[int, int]] = deque(zip(ys.tolist(), xs.tolist()))
    if not queue:
        return distance
    distance[ys, xs] = 0
    height, width = source.shape
    while queue:
        y, x = queue.popleft()
        next_distance = int(distance[y, x]) + 1
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < height and 0 <= nx < width and distance[ny, nx] < 0:
                distance[ny, nx] = next_distance
                queue.append((ny, nx))
    return distance


def _completion_boundary_metrics(completion: np.ndarray, visible: np.ndarray) -> dict[str, Any]:
    completion = np.asarray(completion, dtype=bool)
    visible = np.asarray(visible, dtype=bool)
    bbox = _mask_bbox(completion)
    distances = _distance_map(visible)
    values = distances[completion]
    finite = values[values >= 0]
    min_distance = int(finite.min()) if finite.size else None
    return {
        "completion_area_pixels": int(completion.sum()),
        "completion_bbox": bbox,
        "completion_width": int(bbox["width"]) if bbox else 0,
        "completion_height": int(bbox["height"]) if bbox else 0,
        "distance_to_visible_fish": min_distance,
        "distance_to_visible_fish_stats": {
            "min_px": min_distance,
            "mean_px": round(float(finite.mean()), 3) if finite.size else None,
            "max_px": int(finite.max()) if finite.size else None,
        },
        "completion_contour_pixels": int(_mask_contour(completion).sum()),
        "refined_visible_contour_pixels": int(_mask_contour(visible).sum()),
    }


def _mask_boundary_preview(original: Image.Image, completion: np.ndarray, visible: np.ndarray) -> bytes:
    base = np.asarray(original.convert("RGB"), dtype=np.float32)
    completion = np.asarray(completion, dtype=bool)
    visible = np.asarray(visible, dtype=bool)
    rendered = base.copy()
    if completion.any():
        rendered[completion] = rendered[completion] * 0.58 + np.asarray([255, 196, 0], dtype=np.float32) * 0.42
    visible_contour = _mask_contour(visible)
    completion_contour = _mask_contour(completion)
    if visible_contour.any():
        rendered[visible_contour] = np.asarray([24, 190, 110], dtype=np.float32)
    if completion_contour.any():
        rendered[completion_contour] = np.asarray([235, 55, 55], dtype=np.float32)
    return _png(Image.fromarray(np.clip(rendered, 0, 255).astype("uint8"), "RGB"))


def _feather_alpha(mask: np.ndarray, radius: int) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    if not source.any():
        return np.zeros(source.shape, dtype=np.float32)
    if source.all():
        return np.ones(source.shape, dtype=np.float32)
    distance = _distance_map(~source)
    alpha = np.clip(distance.astype(np.float32) / max(int(radius), 1), 0.0, 1.0)
    alpha[~source] = 0.0
    return alpha


def _compose_feather(original: Image.Image, generated: Image.Image, completion_mask: np.ndarray, radius: int) -> bytes:
    base = np.asarray(original.convert("RGB"), dtype=np.float32)
    out = np.asarray(generated.convert("RGB").resize(original.size), dtype=np.float32)
    alpha = _feather_alpha(completion_mask, radius)[..., None]
    blended = base * (1.0 - alpha) + out * alpha
    return _png(Image.fromarray(np.clip(np.rint(blended), 0, 255).astype("uint8"), "RGB"))


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


def _load_base_case_assets(base_test_id: str, dataset_version: str, item_id: int) -> dict[str, bytes]:
    bucket = _bucket()
    if bucket is None:
        raise ValueError("P2.5 base assets require GCS_BUCKET")
    report_uri = f"gs://{bucket.name}/{PREFIX}/{base_test_id}/report.json"
    try:
        report = json.loads(_read_uri(report_uri).decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"P2.5 base test report unavailable: {base_test_id}") from exc
    base_input = report.get("input") or {}
    if str(base_input.get("dataset_version")) != str(dataset_version) or int(base_input.get("dataset_item_id")) != int(item_id):
        raise ValueError("P2.5 base test input does not match selected dataset item")
    assets = report.get("assets") or {}
    result: dict[str, bytes] = {}
    for key in ("visible_add_mask", "visible_remove_mask", "manual_completion_mask"):
        uri = assets.get(key)
        if not uri:
            raise ValueError(f"P2.5 base test asset missing: {key}")
        result[key] = _read_uri(uri)
    return result


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


@router.post("/api/debug/powerpaint-shape-guided-lab/prepare")
async def prepare(request: Request, db=Depends(get_db)):
    test_id = _new_test_id()
    source = None
    try:
        data = await _payload(request)
        dataset_version = str(data.get("dataset_version") or "").strip()
        try:
            item_id = int(data.get("dataset_item_id"))
        except (TypeError, ValueError) as exc:
            raise ExperimentFailure(400, "INIT", "INVALID_DATASET_ITEM", "dataset_item_id is required") from exc
        base_test_id = str(data.get("base_test_id") or "").strip()
        base_assets = _load_base_case_assets(base_test_id, dataset_version, item_id) if base_test_id else {}
        prepared = _prepare_input(db, dataset_version, item_id)
        source = prepared["source"]
        crop = prepared["crop"]
        visible = prepared["visible"]
        original_bytes = _png(crop)
        raw_sam_mask_bytes = _mask_png(visible)
        raw_sam_visible_bytes = _visible_fish_png(crop, visible)
        report = {
            "experiment": VERSION,
            "runtime": _runtime(test_id),
            "test_id": test_id,
            "status": "READY_FOR_MANUAL_EDIT",
            "experiment_stage": "RAW_SAM_READY",
            "mask_mode": "MANUAL_V2",
            "prompt_id": PROMPT_ID,
            "fitting_degree": P2_DEFAULT_FITTING_DEGREE,
            "input": {
                "dataset_version": dataset_version,
                "dataset_item_id": item_id,
                "image_id": prepared["item"].image_id,
                "species": prepared["item"].species_name,
                "source_type": "dataset_freeze",
                "base_test_id": base_test_id or None,
            },
            "raw_sam_pixels": int(visible.sum()),
            "progress": [
                {"stage": "raw_sam", "label": "RAW_SAM_READY", "status": "READY"},
                {"stage": "refined_visible", "label": "REFINED_VISIBLE_READY", "status": "PENDING"},
                {"stage": "completion_mask", "label": "COMPLETION_MASK_READY", "status": "PENDING"},
                {"stage": "worker", "label": "WORKER_READY", "status": "PENDING"},
                {"stage": "powerpaint", "label": "POWERPAINT_SUCCESS", "status": "PENDING"},
                {"stage": "final_compose", "label": "FINAL_COMPOSE_READY", "status": "PENDING"},
            ],
            "assets": {},
        }
        report["assets"]["original"] = _safe_persist(test_id, "original.png", original_bytes, "image/png", report)
        report["assets"]["raw_sam_mask"] = _safe_persist(test_id, "raw_sam_mask.png", raw_sam_mask_bytes, "image/png", report)
        report["assets"]["raw_sam_visible"] = _safe_persist(test_id, "raw_sam_visible.png", raw_sam_visible_bytes, "image/png", report)
        report["assets"]["sam_mask"] = report["assets"]["raw_sam_mask"]
        report["assets"]["sam_visible"] = report["assets"]["raw_sam_visible"]
        return {
            "status": "ok",
            "stage": "raw_sam_ready",
            "test_id": test_id,
            "input": report["input"],
            "report": report,
            "width": crop.width,
            "height": crop.height,
            "preview_original": _data_url(original_bytes, "image/png"),
            "preview_raw_sam": _data_url(raw_sam_visible_bytes, "image/png"),
            "raw_sam_mask": _data_url(raw_sam_mask_bytes, "image/png"),
            "visible_add_mask": _data_url(_mask_png(np.zeros_like(visible)), "image/png"),
            "visible_remove_mask": _data_url(_mask_png(np.zeros_like(visible)), "image/png"),
            "manual_completion_mask": _data_url(base_assets.get("manual_completion_mask") or _mask_png(np.zeros_like(visible)), "image/png"),
            "base_test_id": base_test_id or None,
            "base_visible_add_mask": _data_url(base_assets["visible_add_mask"], "image/png") if base_assets else None,
            "base_visible_remove_mask": _data_url(base_assets["visible_remove_mask"], "image/png") if base_assets else None,
            "base_manual_completion_mask": _data_url(base_assets["manual_completion_mask"], "image/png") if base_assets else None,
        }
    except ExperimentFailure as exc:
        return _error(exc.http_status, test_id, exc.stage, exc.error_code, str(exc))
    except Exception as exc:
        logger.exception("Shape Guided prepare failed test_id=%s", test_id)
        return _error(500, test_id, "INIT", "SHAPE_GUIDED_PREPARE_FAILED", f"{exc.__class__.__name__}: {exc}")
    finally:
        if source is not None:
            source.close()


@router.post("/api/debug/powerpaint-shape-guided-lab/run")
async def run(request: Request, db=Depends(get_db)):
    started = time.perf_counter()
    test_id = _new_test_id()
    source = None
    raw = None
    crop = None
    visible = None
    original_bytes = None
    detector_bytes = None
    sam_visible_bytes = None
    sam_mask_bytes = None
    raw_sam_mask_bytes = None
    raw_sam_visible_bytes = None
    visible_add_mask_bytes = None
    visible_remove_mask_bytes = None
    refined_visible_mask_bytes = None
    refined_visible_fish_bytes = None
    manual_completion_mask_bytes = None
    completion_mask_bytes = None
    completion_overlay_bytes = None
    request_log: list[dict[str, Any]] = []
    response_log: list[dict[str, Any]] = []
    error_info: dict[str, Any] | None = None
    response_status = 200
    report: dict[str, Any] = {
        "experiment": VERSION,
        "runtime": _runtime(test_id),
        "test_id": test_id,
        "status": "RUNNING",
        "experiment_stage": "INIT",
        "result_classification": None,
        "completion_required": None,
        "comparison_mode": False,
        "base_test_id": None,
        "quality_scores": {},
        "mask_mode": None,
        "assets": {},
        "results": [],
        "progress": [
            {"stage": "raw_sam", "label": "RAW_SAM_READY", "status": "PENDING"},
            {"stage": "refined_visible", "label": "REFINED_VISIBLE_READY", "status": "PENDING"},
            {"stage": "completion_mask", "label": "COMPLETION_MASK_READY", "status": "PENDING"},
            {"stage": "worker", "label": "WORKER_READY", "status": "PENDING"},
            {"stage": "powerpaint", "label": "POWERPAINT_SUCCESS", "status": "PENDING"},
            {"stage": "final_compose", "label": "FINAL_COMPOSE_READY", "status": "PENDING"},
        ],
        "timings": {"total_ms": None},
    }

    def mark_progress(stage: str, status: str, result: Any = None) -> None:
        for entry in report["progress"]:
            if entry["stage"] == stage:
                entry["status"] = status
                if result is not None:
                    entry["result"] = result

    try:
        data = await _payload(request)
        dataset_version = str(data.get("dataset_version") or "").strip()
        try:
            item_id = int(data.get("dataset_item_id"))
        except (TypeError, ValueError) as exc:
            raise ExperimentFailure(400, "INIT", "INVALID_DATASET_ITEM", "dataset_item_id is required") from exc
        mask_mode = str(data.get("mask_mode") or "AUTO_V1").strip().upper()
        if mask_mode not in MASK_MODES:
            raise ExperimentFailure(400, "INIT", "INVALID_MASK_MODE", "mask_mode must be AUTO_V1 or MANUAL_V2")
        comparison_mode = str(data.get("p25_compare") or data.get("comparison_mode") or "").strip().lower() in {"1", "true", "yes", "on"}
        base_test_id = str(data.get("base_test_id") or "").strip()
        quality_scores = _quality_scores(data)
        requested = data.get("fitting_degrees") if mask_mode == "MANUAL_V2" and comparison_mode else (data.get("fitting_degree") if mask_mode == "MANUAL_V2" else (data.get("fitting_degrees") or data.get("fitting_degree") or [str(x) for x in FITTING_DEGREES]))
        try:
            degrees = normalize_fitting_degrees(requested)
        except ValueError as exc:
            raise ExperimentFailure(400, "INIT", "INVALID_FITTING_DEGREE", str(exc)) from exc
        if mask_mode == "MANUAL_V2" and not comparison_mode and degrees != [P2_DEFAULT_FITTING_DEGREE]:
            raise ExperimentFailure(400, "INIT", "P2_FIXED_FITTING_DEGREE", "MANUAL_V2 requires fitting_degree=0.8")
        if comparison_mode and mask_mode != "MANUAL_V2":
            raise ExperimentFailure(400, "INIT", "P25_REQUIRES_MANUAL_V2", "P2.5 comparison requires MANUAL_V2")
        report["mask_mode"] = mask_mode
        report["comparison_mode"] = comparison_mode
        report["base_test_id"] = base_test_id or None
        report["quality_scores"] = quality_scores
        report["prompt_id"] = PROMPT_ID
        report["input"] = {
            "dataset_version": dataset_version,
            "dataset_item_id": item_id,
            "source_type": "dataset_freeze",
            "mask_mode": mask_mode,
            "prompt_id": PROMPT_ID,
            "fitting_degree": degrees[0] if mask_mode == "MANUAL_V2" and not comparison_mode else None,
            "fitting_degrees": degrees,
            "comparison_mode": comparison_mode,
            "base_test_id": base_test_id or None,
        }
        prepared = _prepare_input(db, dataset_version, item_id)
        source = prepared["source"]
        raw = prepared["raw"]
        crop = prepared["crop"]
        visible = prepared["visible"]
        detector_run = prepared["detector_run"]
        assessment = prepared["assessment"]
        primary = prepared["primary"]
        report["input"].update({
            "image_id": prepared["item"].image_id,
            "image_hash": hashlib.sha256(raw).hexdigest()[:16],
            "width": crop.width,
            "height": crop.height,
        })
        original_bytes = _png(crop)
        detector_bytes = original_bytes
        report["detector"] = {"model": detector_run.model_version, "assessment": assessment.status.value, "detections": len(detector_run.detections), "confidence": float(primary.confidence), "bbox_pixels": prepared["bbox"]}
        report["assets"]["original"] = _persist(test_id, "original.png", original_bytes, "image/png")
        report["assets"]["detector_crop"] = _persist(test_id, "detector_crop.png", detector_bytes, "image/png")
        mark_progress("input", "READY")
        _set_stage(report, "INPUT_READY")
        mark_progress("detector", "READY", assessment.status.value)
        _set_stage(report, "DETECTOR_READY")

        raw_sam_mask_bytes = _mask_png(visible)
        raw_sam_visible_bytes = _visible_fish_png(crop, visible)
        sam_visible_bytes = raw_sam_visible_bytes
        sam_mask_bytes = raw_sam_mask_bytes
        report["sam"] = {"model": "SAM_VIT_B", "quality": prepared["segmentation"].quality.value, "raw_sam_pixels": int(visible.sum()), "mask_area_pixels": int(visible.sum())}
        report["assets"]["raw_sam_mask"] = _persist(test_id, "raw_sam_mask.png", raw_sam_mask_bytes, "image/png")
        report["assets"]["raw_sam_visible"] = _persist(test_id, "raw_sam_visible.png", raw_sam_visible_bytes, "image/png")
        report["assets"]["sam_mask"] = _persist(test_id, "sam_mask.png", sam_mask_bytes, "image/png")
        report["assets"]["sam_visible"] = _persist(test_id, "sam_visible.png", sam_visible_bytes, "image/png")
        _persist_json(test_id, "sam_report.json", report["sam"])
        mark_progress("raw_sam", "READY", {"pixels": int(visible.sum())})
        _set_stage(report, "RAW_SAM_READY")

        if mask_mode == "MANUAL_V2":
            visible_add = _decode_mask_data_url(data.get("visible_add_mask"), visible.shape, "visible_add_mask")
            visible_remove = _decode_mask_data_url(data.get("visible_remove_mask"), visible.shape, "visible_remove_mask")
            refined_visible = (visible | visible_add) & ~visible_remove
            visible_add_mask_bytes = _mask_png(visible_add)
            visible_remove_mask_bytes = _mask_png(visible_remove)
            refined_visible_mask_bytes = _mask_png(refined_visible)
            refined_visible_fish_bytes = _visible_fish_png(crop, refined_visible)
            report["sam"].update({
                "refined_visible_pixels": int(refined_visible.sum()),
                "visible_add_pixels": int(visible_add.sum()),
                "visible_remove_pixels": int(visible_remove.sum()),
            })
            report["assets"]["visible_add_mask"] = _persist(test_id, "visible_add_mask.png", _mask_png(visible_add), "image/png")
            report["assets"]["visible_remove_mask"] = _persist(test_id, "visible_remove_mask.png", _mask_png(visible_remove), "image/png")
            report["assets"]["refined_visible_mask"] = _persist(test_id, "refined_visible_mask.png", refined_visible_mask_bytes, "image/png")
            report["assets"]["refined_visible_fish"] = _persist(test_id, "refined_visible_fish.png", refined_visible_fish_bytes, "image/png")
            visible = refined_visible
            mark_progress("refined_visible", "READY", {"pixels": int(visible.sum()), "added": int(visible_add.sum()), "removed": int(visible_remove.sum())})
            _set_stage(report, "REFINED_VISIBLE_READY")
            completion = _decode_mask_data_url(data.get("manual_completion_mask") or data.get("completion_mask"), visible.shape, "manual_completion_mask")
            requested_overlap = int((completion & visible).sum())
            completion = completion & ~visible
            validation = validate_manual_completion_mask(completion, visible)
            validation["requested_visible_overlap_pixels"] = requested_overlap
            validation["mask_mode"] = "MANUAL_V2"
            if not validation["valid"]:
                raise ExperimentFailure(422, "COMPLETION_MASK_READY", "INVALID_MANUAL_COMPLETION_MASK", json.dumps(validation), "FAILED_MASK")
            completion_mask_bytes = _mask_png(completion)
            manual_completion_mask_bytes = completion_mask_bytes
            report["completion_mask"] = validation
            report["raw_sam_pixels"] = int(prepared["visible"].sum())
            report["refined_visible_pixels"] = int(visible.sum())
            report["visible_add_pixels"] = int(visible_add.sum())
            report["visible_remove_pixels"] = int(visible_remove.sum())
            report["completion_area_pixels"] = validation["completion_area_pixels"]
            report["completion_area_ratio"] = validation["completion_area_ratio"]
            report["visible_overlap_pixels"] = validation["visible_overlap_pixels"]
            report["assets"]["manual_completion_mask"] = _persist(test_id, "manual_completion_mask.png", manual_completion_mask_bytes, "image/png")
            report["assets"]["completion_mask"] = _persist(test_id, "completion_mask.png", completion_mask_bytes, "image/png")
            _persist_json(test_id, "completion_mask_report.json", validation)
            _persist_json(test_id, "completion_report.json", validation)
            report["preview_raw_sam"] = _data_url(raw_sam_visible_bytes, "image/png")
            report["preview_refined_visible"] = _data_url(refined_visible_fish_bytes, "image/png")
            report["preview_sam"] = report["preview_refined_visible"]
            report["preview_completion_mask"] = _data_url(completion_mask_bytes, "image/png")
            mark_progress("completion_mask", "READY", validation)
            _set_stage(report, "COMPLETION_MASK_READY")
            if not comparison_mode:
                degrees = [P2_DEFAULT_FITTING_DEGREE]
        else:
            completion = build_completion_mask(visible)
            validation = validate_completion_mask(completion, visible)
            if not validation["valid"]:
                raise ExperimentFailure(422, "MASK_READY", "INVALID_COMPLETION_MASK", json.dumps(validation), "FAILED_MASK")
            completion_mask_bytes = _mask_png(completion)
            manual_completion_mask_bytes = completion_mask_bytes
            report["completion_mask"] = validation
            report["completion_area_pixels"] = validation["completion_area_pixels"]
            report["completion_area_ratio"] = validation["completion_area_ratio"]
            report["visible_overlap_pixels"] = validation["visible_overlap_pixels"]
            report["assets"]["completion_mask"] = _persist(test_id, "completion_mask.png", completion_mask_bytes, "image/png")
            _persist_json(test_id, "completion_report.json", validation)
            report["preview_raw_sam"] = _data_url(raw_sam_visible_bytes, "image/png")
            report["preview_refined_visible"] = _data_url(raw_sam_visible_bytes, "image/png")
            report["preview_sam"] = report["preview_refined_visible"]
            report["preview_completion_mask"] = _data_url(completion_mask_bytes, "image/png")
            mark_progress("refined_visible", "READY", {"pixels": int(visible.sum()), "added": 0, "removed": 0})
            mark_progress("completion_mask", "READY", validation)
            _set_stage(report, "MASK_READY")
        boundary_metrics = _completion_boundary_metrics(completion, visible)
        completion_overlay_bytes = _mask_boundary_preview(crop, completion, visible)
        report["mask_boundary"] = boundary_metrics
        report["assets"]["completion_mask_overlay"] = _persist(test_id, "completion_mask_overlay.png", completion_overlay_bytes, "image/png")
        report["assets"]["completion_mask_boundary"] = _persist(test_id, "completion_mask_boundary.png", completion_overlay_bytes, "image/png")
        report["preview_completion_overlay"] = _data_url(completion_overlay_bytes, "image/png")
        report["preview_original"] = _data_url(original_bytes, "image/png")
        report["completion_required"] = bool(validation["completion_area_pixels"] > 0 and validation["completion_area_ratio"] >= 0.001)
        if mask_mode == "MANUAL_V2" and not report["completion_required"]:
            raise ExperimentFailure(422, "COMPLETION_MASK_READY", "EMPTY_MANUAL_COMPLETION_MASK", "MANUAL_V2 requires a non-empty completion mask", "FAILED_MASK")
        if not report["completion_required"]:
            for degree in degrees:
                report["results"].append({"task_mode": TASK_MODE, "fitting_degree": degree, "status": "NOT_REQUIRED", "result": "NOT_REQUIRED", "worker_called": False, "result_uri": None, "generated_area_pixels": 0, "visible_pixel_change_ratio": 0.0})
            report["result"] = "NOT_REQUIRED"
            report["result_classification"] = "SUCCESS_NOT_REQUIRED"
            report["completion_case"] = "COMPLETE_FISH"
            report["assets"]["powerpaint_output"] = _persist(test_id, "powerpaint_output.png", original_bytes, "image/png")
            report["assets"]["final_result"] = _persist(test_id, "final_result.png", original_bytes, "image/png")
            report["result_preview"] = _data_url(original_bytes, "image/png")
            mark_progress("powerpaint", "SUCCESS", "NOT_REQUIRED")
            mark_progress("final_compose", "READY", "NOT_REQUIRED")
            _set_stage(report, "FINAL_COMPOSE_READY")
            _set_stage(report, "SUCCESS")
        else:
            try:
                report["worker"] = {"health": _check_worker_health()}
                mark_progress("worker", "READY", report["worker"]["health"])
                _set_stage(report, "WORKER_READY")
            except Exception as exc:
                report["worker"] = {"health": {"status": "unreachable", "error": str(exc)}}
                request_log.append({"task_mode": TASK_MODE, "mask_mode": mask_mode, "fitting_degree": degrees[0], "prompt_version": PROMPT_ID, "image_uri": report["assets"]["detector_crop"], "mask_uri": report["assets"]["completion_mask"], "visible_reference_uri": report["assets"].get("refined_visible_fish"), "worker_called": False})
                response_log.append({"fitting_degree": degrees[0], "worker_called": False, "http_status": None, "result_uri": None, "latency_ms": 0, "error": str(exc)})
                raise ExperimentFailure(503, "WORKER_READY", "SHAPE_GUIDED_WORKER_FAILED", str(exc), "FAILED_WORKER") from exc
            _set_stage(report, "POWERPAINT_RUNNING")
            mark_progress("powerpaint", "RUNNING", {"fitting_degree": degrees[0]})
            for degree in degrees:
                result_started = time.perf_counter()
                request_entry = {"task_mode": TASK_MODE, "mask_mode": mask_mode, "fitting_degree": degree, "prompt_version": PROMPT_ID, "image_uri": report["assets"]["detector_crop"], "mask_uri": report["assets"]["completion_mask"], "visible_reference_uri": report["assets"].get("refined_visible_fish"), "mask_ratio": validation["completion_area_ratio"], "worker_called": True}
                request_log.append(request_entry)
                item_result: dict[str, Any] = {"task_mode": TASK_MODE, "mask_mode": mask_mode, "fitting_degree": degree, "status": "PENDING", "worker_called": True, "visible_pixel_change_ratio": None, "generated_area_pixels": int(completion.sum()), "result_uri": None}
                try:
                    worker = _invoke_shape_guided(image_uri=report["assets"]["detector_crop"], mask_uri=report["assets"]["completion_mask"], visible_reference_uri=report["assets"].get("refined_visible_fish"), fitting_degree=degree)
                    generated = _decode_data_url(worker.get("result_uri")) or _decode_data_url(worker.get("generated_roi"))
                    if generated is None and worker.get("result_uri"):
                        generated = _read_uri(worker["result_uri"])
                    if not generated:
                        raise RuntimeError("SHAPE_GUIDED_WORKER_EMPTY_OUTPUT")
                    output_uri = _persist(test_id, f"powerpaint_output_{degree:g}.png", generated, "image/png")
                    suffix = f"{degree:g}"
                    with Image.open(io.BytesIO(generated)) as generated_image:
                        hard_bytes, visible_change = _compose(crop, generated_image, completion)
                        feather3_bytes = _compose_feather(crop, generated_image, completion, 3)
                        feather5_bytes = _compose_feather(crop, generated_image, completion, 5)
                    hard_name = "final_hard_compose.png" if degree == P2_DEFAULT_FITTING_DEGREE else f"final_hard_compose_{suffix}.png"
                    feather3_name = "final_feather_3px.png" if degree == P2_DEFAULT_FITTING_DEGREE else f"final_feather_3px_{suffix}.png"
                    feather5_name = "final_feather_5px.png" if degree == P2_DEFAULT_FITTING_DEGREE else f"final_feather_5px_{suffix}.png"
                    hard_uri = _persist(test_id, hard_name, hard_bytes, "image/png")
                    feather3_uri = _persist(test_id, feather3_name, feather3_bytes, "image/png")
                    feather5_uri = _persist(test_id, feather5_name, feather5_bytes, "image/png")
                    latency_ms = round((time.perf_counter() - result_started) * 1000, 2)
                    item_result.update({"status": "SUCCESS", "result_uri": worker.get("result_uri"), "output_asset": output_uri, "final_asset": hard_uri, "hard_compose_asset": hard_uri, "feather_3px_asset": feather3_uri, "feather_5px_asset": feather5_uri, "worker_ms": latency_ms, "inference_time_ms": worker.get("inference_time_ms"), "model_version": worker.get("model_version"), "raw_output_preview": _data_url(generated, "image/png"), "visible_pixel_change_ratio": visible_change, "fish_identity_check": "PENDING", "background_change": "PENDING", "result_preview": _data_url(hard_bytes, "image/png"), "hard_compose_preview": _data_url(hard_bytes, "image/png"), "feather_3px_preview": _data_url(feather3_bytes, "image/png"), "feather_5px_preview": _data_url(feather5_bytes, "image/png")})
                    response_log.append({"fitting_degree": degree, "worker_called": True, "http_status": worker.get("http_status"), "result_uri": worker.get("result_uri"), "inference_time_ms": worker.get("inference_time_ms"), "model_version": worker.get("model_version"), "latency_ms": latency_ms, "hard_compose_asset": hard_uri, "feather_3px_asset": feather3_uri, "feather_5px_asset": feather5_uri, "error": None})
                    report["assets"][f"powerpaint_output_{suffix}"] = output_uri
                    report["assets"][f"final_hard_compose_{suffix}"] = hard_uri
                    report["assets"][f"final_feather_3px_{suffix}"] = feather3_uri
                    report["assets"][f"final_feather_5px_{suffix}"] = feather5_uri
                    if degree == P2_DEFAULT_FITTING_DEGREE:
                        report["assets"]["final_hard_compose"] = hard_uri
                        report["assets"]["final_feather_3px"] = feather3_uri
                        report["assets"]["final_feather_5px"] = feather5_uri
                    report["result_preview"] = _data_url(hard_bytes, "image/png")
                    report["preview_hard_compose"] = _data_url(hard_bytes, "image/png")
                    report["preview_feather_3px"] = _data_url(feather3_bytes, "image/png")
                    report["preview_feather_5px"] = _data_url(feather5_bytes, "image/png")
                    mark_progress("powerpaint", "SUCCESS", {"fitting_degree": degree, "inference_time_ms": worker.get("inference_time_ms"), "model_version": worker.get("model_version")})
                    mark_progress("final_compose", "READY", {"fitting_degree": degree, "hard_compose_asset": hard_uri, "feather_3px_asset": feather3_uri})
                except Exception as exc:
                    latency_ms = round((time.perf_counter() - result_started) * 1000, 2)
                    item_result.update({"status": "FAILED_POWERPAINT", "error_code": "SHAPE_GUIDED_POWERPAINT_FAILED", "error": str(exc), "worker_ms": latency_ms})
                    response_log.append({"fitting_degree": degree, "worker_called": True, "http_status": None, "result_uri": None, "latency_ms": latency_ms, "error": str(exc)})
                report["results"].append(item_result)
            successful = [x for x in report["results"] if x["status"] == "SUCCESS"]
            if successful:
                best = successful[-1]
                report["assets"]["powerpaint_output"] = _persist(test_id, "powerpaint_output.png", _read_uri(best["output_asset"]), "image/png")
                report["assets"]["final_result"] = _persist(test_id, "final_result.png", _read_uri(best["hard_compose_asset"]), "image/png")
                report["assets"]["final_hard_compose"] = best.get("hard_compose_asset")
                report["assets"]["final_feather_3px"] = best.get("feather_3px_asset")
                report["assets"]["final_feather_5px"] = best.get("feather_5px_asset")
                report["result_uri"] = best.get("result_uri")
                report["model_version"] = best.get("model_version")
                report["inference_time_ms"] = best.get("inference_time_ms")
                report["preview_hard_compose"] = best.get("hard_compose_preview")
                report["preview_feather_3px"] = best.get("feather_3px_preview")
                report["preview_feather_5px"] = best.get("feather_5px_preview")
                report["result_classification"] = "SUCCESS_COMPLETED"
                _set_stage(report, "FINAL_COMPOSE_READY")
                mark_progress("powerpaint", "SUCCESS", {"successful_degrees": [x["fitting_degree"] for x in successful]})
                mark_progress("final_compose", "READY", {"successful_degrees": [x["fitting_degree"] for x in successful]})
                _set_stage(report, "SUCCESS")
            else:
                raise ExperimentFailure(502, "POWERPAINT_RUNNING", "SHAPE_GUIDED_POWERPAINT_FAILED", "all fitting degrees failed", "FAILED_POWERPAINT")
        report["shape_guided_report"] = shape_guided_report_entry(fitting_degree=degrees[-1], visible_pixel_change_ratio=(0.0 if not report["completion_required"] else next((x.get("visible_pixel_change_ratio") for x in report["results"] if x.get("status") == "SUCCESS"), None)), completion_area_ratio=validation["completion_area_ratio"], generated_area_pixels=validation["completion_area_pixels"], status=report["result_classification"] or "SUCCESS")
        report["p2"] = {
            "mask_mode": mask_mode,
            "raw_sam_pixels": report.get("raw_sam_pixels", int(prepared["visible"].sum())),
            "refined_visible_pixels": report.get("refined_visible_pixels", int(visible.sum())),
            "visible_add_pixels": report.get("visible_add_pixels", 0),
            "visible_remove_pixels": report.get("visible_remove_pixels", 0),
            "completion_area_pixels": validation["completion_area_pixels"],
            "completion_area_ratio": validation["completion_area_ratio"],
            "visible_overlap_pixels": validation["visible_overlap_pixels"],
            "fitting_degree": degrees[0] if mask_mode == "MANUAL_V2" else degrees[-1],
            "result_uri": report.get("result_uri"),
        }
        report["p25"] = {
            "base_test_id": base_test_id or None,
            "comparison_mode": comparison_mode,
            "mask_boundary": report.get("mask_boundary"),
            "quality_scores": quality_scores,
            "hard_compose_asset": report["assets"].get("final_hard_compose"),
            "feather_3px_asset": report["assets"].get("final_feather_3px"),
            "feather_5px_asset": report["assets"].get("final_feather_5px"),
            "degrees": degrees,
        }
    except ExperimentFailure as exc:
        response_status = exc.http_status
        error_info = {"error_code": exc.error_code, "message": str(exc), "stage": exc.stage, "classification": exc.classification}
        report["result_classification"] = exc.classification
        report["error"] = error_info
        report["completion_required"] = report.get("completion_required")
        mark_progress("completion_mask" if exc.stage in {"COMPLETION_MASK_READY", "MASK_READY"} else "worker", "FAILED", str(exc))
        _set_stage(report, "FAILED")
    except (TypeError, ValueError) as exc:
        response_status = 400
        error_info = {"error_code": "INVALID_REQUEST", "message": str(exc), "stage": "INIT", "classification": "FAILED_VALIDATION"}
        report["result_classification"] = "FAILED_VALIDATION"
        report["error"] = error_info
        _set_stage(report, "FAILED")
    except HTTPException as exc:
        response_status = exc.status_code
        error_info = {"error_code": str(exc.detail), "message": str(exc.detail), "stage": report.get("experiment_stage", "INIT"), "classification": "FAILED_VALIDATION"}
        report["result_classification"] = "FAILED_VALIDATION"
        report["error"] = error_info
        _set_stage(report, "FAILED")
    except Exception as exc:
        logger.exception("Shape Guided Lab failed test_id=%s", test_id)
        response_status = 500
        error_info = {"error_code": "SHAPE_GUIDED_FAILED", "message": f"{exc.__class__.__name__}: {exc}", "stage": report.get("experiment_stage", "INIT"), "classification": "FAILED_VALIDATION"}
        report["result_classification"] = "FAILED_VALIDATION"
        report["error"] = error_info
        _set_stage(report, "FAILED")
    finally:
        if source is not None:
            source.close()
        if original_bytes is None:
            original_bytes = _placeholder_png()
        if detector_bytes is None:
            detector_bytes = original_bytes
        if raw_sam_mask_bytes is None:
            raw_sam_mask_bytes = _mask_png(np.zeros((1, 1), dtype=bool))
        if raw_sam_visible_bytes is None:
            raw_sam_visible_bytes = original_bytes
        if visible_add_mask_bytes is None:
            visible_add_mask_bytes = _mask_png(np.zeros((1, 1), dtype=bool))
        if visible_remove_mask_bytes is None:
            visible_remove_mask_bytes = _mask_png(np.zeros((1, 1), dtype=bool))
        if refined_visible_mask_bytes is None:
            refined_visible_mask_bytes = raw_sam_mask_bytes
        if refined_visible_fish_bytes is None:
            refined_visible_fish_bytes = raw_sam_visible_bytes
        if manual_completion_mask_bytes is None:
            manual_completion_mask_bytes = _mask_png(np.zeros((1, 1), dtype=bool))
        if completion_mask_bytes is None:
            completion_mask_bytes = manual_completion_mask_bytes
        if completion_overlay_bytes is None:
            completion_overlay_bytes = _placeholder_png()
        _safe_persist(test_id, "original.png, original_bytes, "image/png", report)
        _safe_persist(test_id, "detector_crop.png", detector_bytes, "image/png", report)
        _safe_persist(test_id, "raw_sam_mask.png", raw_sam_mask_bytes, "image/png", report)
        _safe_persist(test_id, "raw_sam_visible.png", raw_sam_visible_bytes, "image/png", report)
        _safe_persist(test_id, "visible_add_mask.png", visible_add_mask_bytes, "image/png", report)
        _safe_persist(test_id, "visible_remove_mask.png", visible_remove_mask_bytes, "image/png", report)
        _safe_persist(test_id, "refined_visible_mask.png", refined_visible_mask_bytes, "image/png", report)
        _safe_persist(test_id, "refined_visible_fish.png", refined_visible_fish_bytes, "image/png", report)
        _safe_persist(test_id, "sam_visible.png", refined_visible_fish_bytes, "image/png", report)
        _safe_persist(test_id, "sam_mask.png", refined_visible_mask_bytes, "image/png", report)
        _safe_persist(test_id, "manual_completion_mask.png", manual_completion_mask_bytes, "image/png", report)
        _safe_persist(test_id, "completion_mask.png", completion_mask_bytes, "image/png", report)
        _safe_persist(test_id, "completion_mask_overlay.png", completion_overlay_bytes, "image/png", report)
        _safe_persist(test_id, "completion_mask_boundary.png", completion_overlay_bytes, "image/png", report)
        report["assets"]["detector_report"] = _safe_persist_json(test_id, "detector_report.json", report.get("detector", {"status": "NOT_REACHED"}))
        report["assets"]["sam_report"] = _safe_persist_json(test_id, "sam_report.json", report.get("sam", {"status": "NOT_REACHED"}))
        report["assets"]["completion_report"] = _safe_persist_json(test_id, "completion_report.json", report.get("completion_mask", {"status": "NOT_REACHED"}))
        report["assets"]["completion_mask_report"] = _safe_persist_json(test_id, "completion_mask_report.json", report.get("completion_mask", {"status": "NOT_REACHED"}))
        report["assets"]["shape_guided_request"] = _safe_persist_json(test_id, "shape_guided_request.json", {"requests": request_log})
        report["assets"]["shape_guided_response"] = _safe_persist_json(test_id, "shape_guided_response.json", {"responses": response_log})
        if "powerpaint_output" not in report["assets"]:
            report["assets"]["powerpaint_output"] = _safe_persist(test_id, "powerpaint_output.png", original_bytes, "image/png", report)
        if "final_result" not in report["assets"]:
            report["assets"]["final_result"] = _safe_persist(test_id, "final_result.png", original_bytes, "image/png", report)
        report["timings"]["total_ms"] = round((time.perf_counter() - started) * 1000, 2)
        report["artifacts_ready"] = not bool(report.get("persistence_errors"))
        report["error"] = error_info
        report["status"] = "SUCCESS" if report.get("experiment_stage") == "SUCCESS" else "FAILED"
        report["assets"]["shape_guided_report"] = _safe_persist_json(test_id, "shape_guided_report.json", report)
        report["assets"]["error"] = _safe_persist_json(test_id, "error.json", error_info or {"error": None, "status": report["status"]})
        report["assets"]["report"] = _safe_persist_json(test_id, "report.json", report)
    if error_info:
        return JSONResponse(status_code=response_status, content={"status": "error", "stage": report["experiment_stage"], "test_id": test_id, "error": error_info, "report": report})
    return {"status": "ok", "stage": "shape_guided_complete", "test_id": test_id, "report": report}