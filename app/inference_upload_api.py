from __future__ import annotations

import hashlib
import io
import json
import math
import mimetypes
import re
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from google.cloud import storage
from pydantic import BaseModel, Field
from PIL import Image, ImageOps, UnidentifiedImageError
from sqlalchemy.orm import Session

from app.db import get_db
from app.factory import get_bucket_name
from app.models import InferenceAsset

router = APIRouter(tags=["inference-assets"])

MAX_RECORD_BYTES = 1 * 1024 * 1024
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
SUFFIXES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}


class InferenceContractError(ValueError):
    """A client record failed the additive inference contract."""


class InferenceReviewRequest(BaseModel):
    decision: str = Field(min_length=3, max_length=32)
    reviewer: str = Field(default="web-review", min_length=1, max_length=256)
    accepted_bbox: list[float] | None = Field(default=None, min_length=4, max_length=4)
    accepted_species: str | None = Field(default=None, max_length=128)


def _safe_image_id(value: Any) -> str:
    image_id = str(value or "").strip()
    if not re.fullmatch(r"yj_img_[A-Za-z0-9][A-Za-z0-9_.-]{1,191}", image_id):
        raise InferenceContractError("image_id must be a generated yj_img UUID")
    return image_id


def _safe_component(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())[:192]
    return result or "image"


def _content_type(file: UploadFile, data: bytes) -> str:
    value = (file.content_type or "").lower().strip()
    if value in ALLOWED_CONTENT_TYPES:
        return value
    guessed = mimetypes.guess_type(file.filename or "")[0]
    if guessed in ALLOWED_CONTENT_TYPES:
        return guessed
    try:
        with Image.open(io.BytesIO(data)) as source:
            format_name = (source.format or "").upper()
    except Exception:
        format_name = ""
    return {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}.get(format_name, "")


def _finite_between_zero_one(value: Any) -> bool:
    try:
        number = float(value)
        return math.isfinite(number) and 0.0 <= number <= 1.0
    except (TypeError, ValueError):
        return False


def _validate_bbox(value: Any, field: str = "bbox") -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise InferenceContractError(f"{field} must be normalized [x, y, width, height]")
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise InferenceContractError(f"{field} must contain numbers") from exc
    if not all(math.isfinite(item) and 0.0 <= item <= 1.0 for item in values):
        raise InferenceContractError(f"{field} must be normalized between 0 and 1")
    if values[0] + values[2] > 1.00001 or values[1] + values[3] > 1.00001:
        raise InferenceContractError(f"{field} must stay inside the image")
    if values[2] <= 0 or values[3] <= 0:
        raise InferenceContractError(f"{field} must have positive size")
    return values


def _read_record(data: bytes) -> dict[str, Any]:
    if not data:
        raise InferenceContractError("InferenceRecord.json is empty")
    if len(data) > MAX_RECORD_BYTES:
        raise InferenceContractError("InferenceRecord.json exceeds 1 MB")
    try:
        document = json.loads(data.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InferenceContractError("InferenceRecord.json is not valid UTF-8 JSON") from exc
    if not isinstance(document, dict):
        raise InferenceContractError("InferenceRecord.json must be an object")
    version = str(document.get("contract_version") or "")
    if version not in {"INFERENCE_RECORD_V2", "INFERENCE_RECORD_V1"}:
        raise InferenceContractError("unsupported inference contract_version")
    image_id = _safe_image_id(document.get("image_id"))
    document["image_id"] = image_id

    detection = document.get("detection")
    if detection is not None:
        if not isinstance(detection, dict):
            raise InferenceContractError("detection must be an object or null")
        if "ground_truth_bbox" in detection or "ground_truth" in detection:
            raise InferenceContractError("detector output must use candidate_bbox; ground truth requires review")
        bbox = detection.get("candidate_bbox")
        if bbox is not None:
            _validate_bbox(bbox, "candidate_bbox")
    crop = document.get("crop")
    if crop is not None:
        if not isinstance(crop, dict):
            raise InferenceContractError("crop must be an object or null")
        if str(crop.get("source_image_id") or "").strip() != image_id:
            raise InferenceContractError("crop.source_image_id must match image_id")
        try:
            if int(crop.get("crop_width")) <= 0 or int(crop.get("crop_height")) <= 0:
                raise InferenceContractError("crop dimensions must be positive")
        except (TypeError, ValueError) as exc:
            raise InferenceContractError("crop dimensions are invalid") from exc
    classification = document.get("classification")
    if classification is not None:
        if not isinstance(classification, dict) or not str(classification.get("model_version") or "").strip():
            raise InferenceContractError("classification.model_version is required")
        if not _finite_between_zero_one(classification.get("confidence")):
            raise InferenceContractError("classification.confidence must be between 0 and 1")
    return document


async def _read_upload(file: UploadFile, limit: int) -> bytes:
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise InferenceContractError(f"{file.filename or 'upload'} exceeds the size limit")
    return data


def _inspect_image(data: bytes, expected: dict[str, Any] | None = None) -> tuple[int, int, str]:
    if not data:
        raise InferenceContractError("image is empty")
    try:
        with Image.open(io.BytesIO(data)) as source:
            source = ImageOps.exif_transpose(source)
            width, height = source.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise InferenceContractError("image dimensions are invalid or too large")
            image_format = (source.format or "JPEG").upper()
    except (UnidentifiedImageError, OSError) as exc:
        raise InferenceContractError("image must be JPEG, PNG, or WEBP") from exc
    if expected:
        try:
            if int(expected.get("image_width")) != width or int(expected.get("image_height")) != height:
                raise InferenceContractError("record image dimensions do not match uploaded image")
        except (TypeError, ValueError) as exc:
            raise InferenceContractError("record image dimensions are invalid") from exc
    return width, height, image_format


def _uri(bucket: str, object_name: str) -> str:
    return f"gs://{bucket}/{object_name}"


def _put_if_absent(blob: Any, data: bytes, *, content_type: str, client: Any, digest: str) -> str:
    if bool(blob.exists(client)):
        existing = blob.download_as_bytes()
        if hashlib.sha256(existing).hexdigest() == digest:
            return "SKIP"
        raise InferenceContractError("object already exists with a different hash")
    blob.upload_from_string(data, content_type=content_type, if_generation_match=0)
    return "CREATED"


def _asset_dict(row: InferenceAsset, *, duplicate: bool = False) -> dict[str, Any]:
    return {
        "image_id": row.image_id,
        "status": row.status,
        "duplicate": duplicate,
        "source": row.source,
        "record_gcs_uri": row.record_gcs_uri,
        "image_gcs_uri": row.image_gcs_uri,
        "crop_gcs_uri": row.crop_gcs_uri,
        "detector_version": row.detector_version,
        "classifier_version": row.classifier_version,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "reviewed_at": row.reviewed_at.isoformat() if row.reviewed_at else None,
    }


@router.post("/api/v1/inference/upload")
async def upload_inference_asset(
    record: UploadFile = File(...),
    image: UploadFile = File(...),
    crop: UploadFile | None = File(default=None),
    db: Session = Depends(get_db),
):
    try:
        record_bytes = await _read_upload(record, MAX_RECORD_BYTES)
        document = _read_record(record_bytes)
        image_bytes = await _read_upload(image, MAX_IMAGE_BYTES)
        image_type = _content_type(image, image_bytes)
        if image_type not in ALLOWED_CONTENT_TYPES:
            raise InferenceContractError("image must be JPEG, PNG, or WEBP")
        image_width, image_height, image_format = _inspect_image(image_bytes, document.get("detection"))
        crop_bytes = await _read_upload(crop, MAX_IMAGE_BYTES) if crop is not None else None
        crop_type = _content_type(crop, crop_bytes) if crop is not None and crop_bytes else ""
        if crop_bytes is not None:
            if crop_type not in ALLOWED_CONTENT_TYPES:
                raise InferenceContractError("crop must be JPEG, PNG, or WEBP")
            _inspect_image(crop_bytes)

        image_id = document["image_id"]
        existing = db.get(InferenceAsset, image_id)
        record_digest = hashlib.sha256(record_bytes).hexdigest()
        image_digest = hashlib.sha256(image_bytes).hexdigest()
        crop_digest = hashlib.sha256(crop_bytes).hexdigest() if crop_bytes is not None else None
        if existing:
            if (
                existing.record_sha256 == record_digest
                and existing.image_sha256 == image_digest
                and existing.crop_sha256 == crop_digest
            ):
                return _asset_dict(existing, duplicate=True)
            raise HTTPException(
                status_code=409,
                detail={"error": "INFERENCE_CONFLICT", "reason": "image_id already has different content"},
            )

        bucket_name = get_bucket_name()
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        date_path = datetime.now(timezone.utc).strftime("%Y/%m/%d")
        safe_id = _safe_component(image_id)
        prefix = f"app_feedback/inference/{date_path}"
        record_name = f"{prefix}/{safe_id}.json"
        image_name = f"{prefix}/images/{safe_id}{SUFFIXES[image_type]}"
        crop_name = f"{prefix}/crops/{safe_id}.jpg" if crop_bytes is not None else None

        detector = document.get("detection") or {}
        classifier = document.get("classification") or {}
        storage_results = {
            "record": _put_if_absent(bucket.blob(record_name), record_bytes, content_type="application/json", client=storage_client, digest=record_digest),
            "image": _put_if_absent(bucket.blob(image_name), image_bytes, content_type=image_type, client=storage_client, digest=image_digest),
        }
        if crop_name and crop_bytes is not None:
            storage_results["crop"] = _put_if_absent(
                bucket.blob(crop_name),
                crop_bytes,
                content_type="image/jpeg",
                client=storage_client,
                digest=crop_digest or "",
            )

        row = InferenceAsset(
            image_id=image_id,
            source=str(document.get("source") or "android_detector"),
            status="REVIEW_REQUIRED" if detector.get("candidate_bbox") else "CANDIDATE",
            record_gcs_uri=_uri(bucket_name, record_name),
            image_gcs_uri=_uri(bucket_name, image_name),
            crop_gcs_uri=_uri(bucket_name, crop_name) if crop_name else None,
            record_sha256=record_digest,
            image_sha256=image_digest,
            crop_sha256=crop_digest,
            detector_version=str(detector.get("detector_version") or "") or None,
            classifier_version=str(classifier.get("model_version") or "") or None,
        )
        db.add(row)
        db.commit()
        return {
            **_asset_dict(row),
            "storage": storage_results,
            "image": {"width": image_width, "height": image_height, "format": image_format},
        }
    except HTTPException:
        raise
    except InferenceContractError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail={"error": "INFERENCE_INVALID", "reason": str(exc)}) from exc
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail={"error": "INFERENCE_UPLOAD_FAILED", "reason": str(exc)}) from exc


@router.post("/api/v1/inference/{image_id}/review")
def review_inference_asset(image_id: str, payload: InferenceReviewRequest, db: Session = Depends(get_db)):
    try:
        image_id = _safe_image_id(image_id)
    except InferenceContractError as exc:
        raise HTTPException(status_code=400, detail={"error": "INFERENCE_INVALID", "reason": str(exc)}) from exc
    row = db.get(InferenceAsset, image_id)
    if not row:
        raise HTTPException(status_code=404, detail="inference asset not found")
    decision = payload.decision.strip().upper()
    if decision not in {"REVIEW_REQUIRED", "ACCEPTED", "REJECTED", "TRAINING_READY"}:
        raise HTTPException(status_code=400, detail={"error": "INFERENCE_INVALID", "reason": "unsupported review decision"})
    if decision in {"ACCEPTED", "TRAINING_READY"}:
        if payload.accepted_bbox is None:
            raise HTTPException(status_code=400, detail={"error": "INFERENCE_INVALID", "reason": "accepted_bbox is required"})
        try:
            bbox = _validate_bbox(payload.accepted_bbox, "accepted_bbox")
        except InferenceContractError as exc:
            raise HTTPException(status_code=400, detail={"error": "INFERENCE_INVALID", "reason": str(exc)}) from exc
        if decision == "TRAINING_READY" and row.status != "ACCEPTED":
            raise HTTPException(status_code=409, detail={"error": "INFERENCE_STATE_INVALID", "reason": "only ACCEPTED assets can become TRAINING_READY"})
        row.accepted_bbox_json = json.dumps(bbox, separators=(",", ":"))
        if payload.accepted_species:
            row.accepted_species = payload.accepted_species.strip()
    row.status = decision
    row.reviewed_by = payload.reviewer.strip()
    row.reviewed_at = datetime.now(timezone.utc)
    db.commit()
    return _asset_dict(row)
