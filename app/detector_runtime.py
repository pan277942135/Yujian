from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

import numpy as np
from google.cloud import storage
from PIL import Image, ImageOps

from app.recognition_pipeline import BBox, Detection, load_contract


DETECTOR_MODEL_VERSION = os.getenv("DETECTOR_MODEL_VERSION", "DET_FISH_v0.1").strip()
DETECTOR_MODEL_FILENAME = "fish_detector_yolox_nano_v0_1.onnx"
YOLOX_LETTERBOX_FILL = 114
ANDROID_MAX_SOURCE_DIMENSION = 2048


class DetectorRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class DetectorModel:
    model_version: str
    onnx_sha256: str
    onnx_bytes: int
    input_name: str
    input_size: int
    session: Any


@dataclass(frozen=True)
class DetectorRun:
    model_version: str
    onnx_sha256: str
    input_size: int
    input_scale: float
    input_draw_width: int
    input_draw_height: int
    latency_ms: float
    detections: tuple[Detection, ...]


_MODEL_LOCK = threading.Lock()


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://") or "/" not in uri[5:]:
        raise DetectorRuntimeError(f"invalid detector artifact URI: {uri}")
    return tuple(uri[5:].split("/", 1))  # type: ignore[return-value]


def _artifact_uri() -> str:
    configured = os.getenv("DETECTOR_ONNX_URI", "").strip()
    if configured:
        return configured
    bucket = os.getenv("GCS_BUCKET", "").strip()
    if not bucket:
        raise DetectorRuntimeError("GCS_BUCKET is required for the production fish detector")
    return f"gs://{bucket}/models/{DETECTOR_MODEL_VERSION}/{DETECTOR_MODEL_FILENAME}"


def _download_bytes(client: storage.Client, uri: str) -> bytes:
    bucket, object_name = _parse_gs_uri(uri)
    return client.bucket(bucket).blob(object_name).download_as_bytes(timeout=180)


def _metadata_uri(onnx_uri: str) -> str:
    bucket, name = _parse_gs_uri(onnx_uri)
    parent = name.rsplit("/", 1)[0]
    return f"gs://{bucket}/{parent}/detector_metadata.json"


def _contract_uri(onnx_uri: str) -> str:
    bucket, name = _parse_gs_uri(onnx_uri)
    parent = name.rsplit("/", 1)[0]
    return f"gs://{bucket}/{parent}/recognition_pipeline_v1.json"


def _same_json(left: dict, right: dict) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


@lru_cache(maxsize=1)
def load_detector() -> DetectorModel:
    """Load only the signed-by-metadata DET_FISH_v0.1 ONNX artifact from GCS."""
    uri = _artifact_uri()
    client = storage.Client()
    onnx_bytes = _download_bytes(client, uri)
    try:
        metadata = json.loads(_download_bytes(client, _metadata_uri(uri)).decode("utf-8"))
        artifact_contract = json.loads(_download_bytes(client, _contract_uri(uri)).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DetectorRuntimeError("detector metadata or recognition contract is invalid") from exc

    expected_contract = load_contract()
    if not _same_json(artifact_contract, expected_contract):
        raise DetectorRuntimeError("detector recognition contract does not match backend contract")
    if metadata.get("model_version") != DETECTOR_MODEL_VERSION:
        raise DetectorRuntimeError(
            f"detector model version mismatch: expected {DETECTOR_MODEL_VERSION}, got {metadata.get('model_version')}"
        )
    if metadata.get("model_family") != "YOLOX_NANO":
        raise DetectorRuntimeError("production detector must be YOLOX_NANO")
    actual_sha = hashlib.sha256(onnx_bytes).hexdigest()
    if metadata.get("onnx_sha256") != actual_sha:
        raise DetectorRuntimeError("detector ONNX SHA256 mismatch")
    if int(metadata.get("onnx_bytes") or 0) != len(onnx_bytes):
        raise DetectorRuntimeError("detector ONNX byte-size mismatch")

    onnx_contract = metadata.get("onnx") or {}
    input_shape = onnx_contract.get("input_shape") or []
    if len(input_shape) != 4 or input_shape[0] != 1 or input_shape[1] != 3:
        raise DetectorRuntimeError(f"unexpected detector input shape metadata: {input_shape}")
    input_size = int(input_shape[2])
    if input_size <= 0 or int(input_shape[3]) != input_size:
        raise DetectorRuntimeError(f"detector input must be square, got {input_shape}")
    if int(expected_contract["detector"]["input_size"]) != input_size:
        raise DetectorRuntimeError("detector ONNX input size does not match recognition contract")

    try:
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover - validated in runtime image CI
        raise DetectorRuntimeError("onnxruntime is required for production fish detection") from exc

    session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise DetectorRuntimeError("detector ONNX must expose exactly one input and one output")
    if list(inputs[0].shape) != [1, 3, input_size, input_size]:
        raise DetectorRuntimeError(f"unexpected detector ONNX input shape: {inputs[0].shape}")

    return DetectorModel(
        model_version=DETECTOR_MODEL_VERSION,
        onnx_sha256=actual_sha,
        onnx_bytes=len(onnx_bytes),
        input_name=inputs[0].name,
        input_size=input_size,
        session=session,
    )


def reset_detector_cache() -> None:
    """Test-only cache reset; production uses a single immutable verified artifact per process."""
    with _MODEL_LOCK:
        load_detector.cache_clear()


def normalize_android_source(image: Image.Image) -> Image.Image:
    """Normalize a source image to the bitmap Android sends to the detector.

    Android applies EXIF orientation and decodes camera/gallery inputs with an
    ``inSampleSize`` that keeps the longest side at or below 2048 pixels.  The
    detector contract itself is unchanged; this helper makes the source raster
    contract explicit for Frozen Dataset inference as well.
    """

    oriented = ImageOps.exif_transpose(image).convert("RGB")
    max_dimension = max(oriented.width, oriented.height)
    sample = 1
    while max_dimension / sample > ANDROID_MAX_SOURCE_DIMENSION:
        sample *= 2
    if sample > 1:
        target_size = (
            max(1, oriented.width // sample),
            max(1, oriented.height // sample),
        )
        resized = oriented.resize(target_size, Image.Resampling.BILINEAR)
        oriented.close()
        return resized
    return oriented


def _prepare_yolox_input(image: Image.Image, input_size: int) -> tuple[np.ndarray, float, int, int]:
    """YOLOX 0.3 validation preprocess: BGR, top-left letterbox, 114 fill, float32 NCHW."""
    if image.width <= 0 or image.height <= 0:
        raise DetectorRuntimeError("detector source image has invalid dimensions")
    scale = min(input_size / float(image.width), input_size / float(image.height))
    draw_width = max(1, int(image.width * scale))
    draw_height = max(1, int(image.height * scale))
    resized = image.convert("RGB").resize((draw_width, draw_height), Image.Resampling.BILINEAR)
    canvas = np.full((input_size, input_size, 3), YOLOX_LETTERBOX_FILL, dtype=np.uint8)
    canvas[:draw_height, :draw_width] = np.asarray(resized, dtype=np.uint8)
    # YOLOX trains from OpenCV BGR frames. Android mirrors this RGB→BGR conversion.
    bgr = canvas[:, :, ::-1]
    tensor = np.ascontiguousarray(bgr.transpose(2, 0, 1), dtype=np.float32)[None, ...]
    return tensor, scale, draw_width, draw_height


def _iou(left: BBox, right: BBox) -> float:
    a = left.normalized()
    b = right.normalized()
    overlap_x = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    overlap_y = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    overlap = overlap_x * overlap_y
    union = a.area_ratio + b.area_ratio - overlap
    return overlap / union if union > 0.0 else 0.0


def nms(detections: Iterable[Detection], iou_threshold: float) -> tuple[Detection, ...]:
    kept: list[Detection] = []
    for candidate in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if all(_iou(candidate.box, accepted.box) <= iou_threshold for accepted in kept):
            kept.append(candidate)
    return tuple(kept)


def decode_yolox_output(
    output: np.ndarray,
    *,
    scale: float,
    source_width: int,
    source_height: int,
    nms_iou: float,
    min_confidence: float = 0.0,
) -> tuple[Detection, ...]:
    """Decode YOLOX's exported [cx, cy, w, h, objectness, fish_probability] rows."""
    if scale <= 0.0 or source_width <= 0 or source_height <= 0:
        raise DetectorRuntimeError("invalid detector decode dimensions")
    rows = np.asarray(output, dtype=np.float32)
    if rows.ndim == 3:
        if rows.shape[0] != 1:
            raise DetectorRuntimeError(f"unexpected detector batch size: {rows.shape}")
        rows = rows[0]
    if rows.ndim != 2 or rows.shape[1] < 6:
        raise DetectorRuntimeError(f"unexpected YOLOX output shape: {rows.shape}")

    decoded: list[Detection] = []
    for row in rows:
        cx, cy, width, height, objectness, class_probability = (float(v) for v in row[:6])
        confidence = objectness * class_probability
        if not np.isfinite(confidence) or confidence < min_confidence or not all(
            np.isfinite(value) for value in (cx, cy, width, height)
        ):
            continue
        left = (cx - width / 2.0) / scale / source_width
        top = (cy - height / 2.0) / scale / source_height
        right = (cx + width / 2.0) / scale / source_width
        bottom = (cy + height / 2.0) / scale / source_height
        box = BBox(left, top, right, bottom).normalized()
        if box.area_ratio <= 0.0:
            continue
        decoded.append(Detection(confidence=confidence, box=box, class_name="fish"))

    return nms(decoded, nms_iou)


def detect(image: Image.Image) -> DetectorRun:
    started = time.perf_counter()
    model = load_detector()
    tensor, scale, draw_width, draw_height = _prepare_yolox_input(image, model.input_size)
    output = model.session.run(None, {model.input_name: tensor})[0]
    contract = load_contract()
    detections = decode_yolox_output(
        output,
        scale=scale,
        source_width=image.width,
        source_height=image.height,
        nms_iou=float(contract["detector"]["nms_iou"]),
        # Scores below the weak gate cannot change NMS or any quality-gate outcome.
        min_confidence=float(contract["detector"]["weak_confidence"]),
    )
    return DetectorRun(
        model_version=model.model_version,
        onnx_sha256=model.onnx_sha256,
        input_size=model.input_size,
        input_scale=scale,
        input_draw_width=draw_width,
        input_draw_height=draw_height,
        latency_ms=round((time.perf_counter() - started) * 1000.0, 1),
        detections=detections,
    )
