"""BBox-only review endpoints for immutable Frozen Dataset rows."""

from __future__ import annotations

import json
import math
import random
import re
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.detector_runtime import normalize_android_source
from app.frozen_crop_bridge import _read_uri, load_frozen_dataset
from app.models import DatasetCropReview, DatasetCropReviewEvent
from app.recognition_pipeline import assess_detections, load_contract

router = APIRouter(prefix="/api/dataset-crop-review", tags=["dataset-crop-review"])


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _box(value: Any) -> list[float] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(0 <= item <= 1 for item in result) or result[2] <= 0 or result[3] <= 0:
        return None
    if result[0] + result[2] > 1.00001 or result[1] + result[3] > 1.00001:
        return None
    return result


def _json_value(value: str | None) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _quality_status(assessment: Any) -> str:
    status = str(getattr(assessment.status, "value", assessment.status)).lower()
    if status == "ready":
        return "GOOD"
    if status in {"uncertain", "incomplete_fish"}:
        return "WARNING"
    return "BAD"


def _detector_payload(run: Any, assessment: Any) -> tuple[list[float] | None, dict[str, Any]]:
    primary = assessment.primary
    candidate = None
    confidence = None
    area_ratio = None
    aspect_ratio = None
    quality_score = None
    touch_edge = None
    if primary is not None:
        box = primary.box.normalized()
        candidate = [round(value, 6) for value in (box.x1, box.y1, box.width, box.height)]
        confidence = float(primary.confidence)
        area_ratio = float(box.area_ratio)
        aspect_ratio = float(box.width / box.height) if box.height > 0 else None
        quality_score = confidence * math.sqrt(max(area_ratio, 0.0))
        edge_margin = float(load_contract()["quality_gate"]["incomplete_edge_margin_ratio"])
        touch_edge = bool(box.touches_edge(edge_margin))
    detections = []
    for detection in run.detections:
        box = detection.box.normalized()
        detections.append(
            {
                "confidence": round(float(detection.confidence), 6),
                "bbox": [box.x1, box.y1, box.width, box.height],
                "class_name": detection.class_name,
            }
        )
    return candidate, {
        "detector_confidence": confidence,
        "bbox_area_ratio": area_ratio,
        "aspect_ratio": aspect_ratio,
        "quality_score": quality_score,
        "quality_status": _quality_status(assessment),
        "touch_edge": touch_edge,
        "all_detections": detections,
    }


class DatasetCropReviewUpdate(BaseModel):
    decision: str = Field(default="ACCEPTED", max_length=32)
    accepted_bbox: list[float] | None = Field(default=None, min_length=4, max_length=4)
    reviewer: str = Field(default="crop-review", max_length=256)
    notes: str | None = Field(default=None, max_length=4000)


def _ensure_rows(db: Session, dataset_version: str) -> list[dict[str, Any]]:
    loaded = load_frozen_dataset(db, dataset_version)
    existing = {
        row.image_id: row
        for row in db.scalars(
            select(DatasetCropReview).where(DatasetCropReview.source_dataset_version == dataset_version)
        ).all()
    }
    for base in loaded["rows"]:
        if base["image_id"] in existing:
            continue
        row = DatasetCropReview(
            source_dataset_version=dataset_version,
            source_manifest_uri=base["source_manifest_uri"],
            image_id=base["image_id"],
            source_image_id=base["source_image_id"],
            source_image_gcs_uri=base["source_image_gcs_uri"],
            species_key=base["species_key"],
            species_name=base["species_name"],
            class_index=base["class_index"],
            split=base["split"],
            group_id=base.get("group_id") or None,
            review_status="BBOX_REQUIRED",
        )
        db.add(row)
        existing[base["image_id"]] = row
    db.commit()
    return loaded["rows"]


def _item(base: dict[str, Any], review: DatasetCropReview | None) -> dict[str, Any]:
    crop_uri = review.crop_uri if review else None
    crop_status = (review.crop_status if review else None) or ("READY" if crop_uri else "NOT_GENERATED")
    return {
        **base,
        "source_type": "FROZEN_DATASET",
        "candidate_bbox": _box(review.candidate_bbox_json) if review else None,
        "accepted_bbox": _box(review.accepted_bbox_json) if review else None,
        "bbox_source": review.bbox_source if review else None,
        "detector_version": review.detector_version if review else None,
        "detector_confidence": review.detector_confidence if review else None,
        "bbox_area_ratio": review.bbox_area_ratio if review else None,
        "aspect_ratio": review.aspect_ratio if review else None,
        "quality_score": review.quality_score if review else None,
        "quality_status": review.quality_status if review else None,
        "all_detections": _json_value(review.all_detections_json) if review else None,
        "detector_error": review.detector_error if review else None,
        "crop_uri": crop_uri,
        "crop_status": crop_status,
        "crop_error": review.crop_error if review else None,
        "preview_url": f"/api/dataset-crop-review/{base['source_dataset_version']}/{base['image_id']}/crop"
        if crop_uri and crop_status == "READY"
        else None,
        "status": review.review_status if review else "BBOX_REQUIRED",
        "reviewer": review.reviewer if review else None,
        "reviewed_at": review.reviewed_at.isoformat() if review and review.reviewed_at else None,
        "media_url": f"/api/dataset-crop-review/{base['source_dataset_version']}/{base['image_id']}/image",
    }


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("._") or "image"


def _crop_preview_bytes(data: bytes, box: list[float], expand_ratio: float = 0.15) -> bytes:
    from PIL import Image

    with Image.open(BytesIO(data)) as source:
        image = source.convert("RGB")
    x, y, width, height = box
    left = max(0.0, x - width * expand_ratio)
    top = max(0.0, y - height * expand_ratio)
    right = min(1.0, x + width + width * expand_ratio)
    bottom = min(1.0, y + height + height * expand_ratio)
    pixel_box = (
        max(0, int(left * image.width)),
        max(0, int(top * image.height)),
        min(image.width, max(1, int(round(right * image.width)))),
        min(image.height, max(1, int(round(bottom * image.height)))),
    )
    if pixel_box[2] <= pixel_box[0] or pixel_box[3] <= pixel_box[1]:
        raise ValueError("accepted bbox produced an empty crop")
    output = BytesIO()
    image.crop(pixel_box).save(output, format="JPEG", quality=92)
    return output.getvalue()


def _persist_crop_preview(base: dict[str, Any], box: list[float]) -> str:
    """Materialize a review-only crop preview without invoking Dataset Build."""

    source_uri = str(base["source_image_gcs_uri"])
    source_bytes, _ = _read_uri(source_uri)
    crop_bytes = _crop_preview_bytes(source_bytes, box)
    if source_uri.startswith("gs://"):
        from google.cloud import storage
        from app.factory import get_bucket_name

        bucket_name = get_bucket_name()
        object_name = (
            f"crop_review/{_safe_component(base['source_dataset_version'])}/"
            f"{_safe_component(base['image_id'])}_preview.jpg"
        )
        client = storage.Client()
        client.bucket(bucket_name).blob(object_name).upload_from_string(crop_bytes, content_type="image/jpeg")
        return f"gs://{bucket_name}/{object_name}"
    source_path = Path(source_uri)
    crop_path = source_path.with_name(f"{source_path.stem}_crop_preview.jpg")
    crop_path.write_bytes(crop_bytes)
    return str(crop_path)


def _populate_candidate(review: DatasetCropReview, base: dict[str, Any], db: Session) -> None:
    # Rows created by the pre-C.5-E path may already contain a candidate bbox
    # without the detector contract/quality metadata.  Treat those as stale
    # and recompute from the registered source image.  This deliberately does
    # not touch accepted_bbox_json or review_status, so a human decision remains
    # immutable while the read-only candidate evidence is refreshed.
    if review.candidate_bbox_json and review.detector_version and review.quality_status and review.all_detections_json:
        return
    try:
        from PIL import Image
        from app.detector_runtime import detect
        data, _ = _read_uri(base["source_image_gcs_uri"])
        with Image.open(BytesIO(data)) as source_image:
            detector_image = normalize_android_source(source_image)
        try:
            run = detect(detector_image)
        finally:
            detector_image.close()
        assessment = assess_detections(run.detections)
        candidate, metadata = _detector_payload(run, assessment)
        review.detector_version = run.model_version
        review.detector_error = None
        review.detector_confidence = metadata["detector_confidence"]
        review.bbox_area_ratio = metadata["bbox_area_ratio"]
        review.aspect_ratio = metadata["aspect_ratio"]
        review.quality_score = metadata["quality_score"]
        review.quality_status = metadata["quality_status"]
        review.all_detections_json = json.dumps(metadata["all_detections"], separators=(",", ":"))
        if candidate is not None:
            review.candidate_bbox_json = json.dumps(candidate, separators=(",", ":"))
        db.add(review)
        db.commit()
    except Exception as exc:
        db.rollback()
        review.detector_error = str(exc)[:4000]
        review.quality_status = "ERROR"
        db.add(review)
        db.commit()


@router.get("/{dataset_version}/summary")
def summary(dataset_version: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    rows = _ensure_rows(db, dataset_version)
    counts_rows = db.execute(
        select(DatasetCropReview.review_status, func.count())
        .where(DatasetCropReview.source_dataset_version == dataset_version)
        .group_by(DatasetCropReview.review_status)
    ).all()
    counts = {str(status): int(count) for status, count in counts_rows}
    accepted = counts.get("ACCEPTED", 0) + counts.get("TRAINING_READY", 0)
    candidate_bbox_count = int(
        db.scalar(
            select(func.count())
            .select_from(DatasetCropReview)
            .where(
                DatasetCropReview.source_dataset_version == dataset_version,
                DatasetCropReview.candidate_bbox_json.is_not(None),
            )
        )
        or 0
    )
    accepted_bbox_count = int(
        db.scalar(
            select(func.count())
            .select_from(DatasetCropReview)
            .where(
                DatasetCropReview.source_dataset_version == dataset_version,
                DatasetCropReview.accepted_bbox_json.is_not(None),
            )
        )
        or 0
    )
    return {
        "dataset_version": dataset_version,
        "source_type": "FROZEN_DATASET",
        "total_images": len(rows),
        "bbox_required": max(len(rows) - accepted, 0),
        "accepted": accepted,
        "rejected": counts.get("REJECTED", 0),
        "candidate_bbox_count": candidate_bbox_count,
        "accepted_bbox_count": accepted_bbox_count,
        "counts": counts,
    }


@router.get("/{dataset_version}/detector-audit")
def detector_audit(
    dataset_version: str,
    sample_size: int = Query(default=100, ge=1, le=100),
    seed: int = Query(default=20260902),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Run a bounded, read-only Detector audit over Frozen Dataset images.

    This endpoint intentionally does not call ``_ensure_rows`` and never writes
    DatasetCropReview rows.  It evaluates the registered Frozen Dataset source
    with the Android-compatible source normalization and the shared quality gate.
    """

    loaded = load_frozen_dataset(db, dataset_version, verify_source_images=False)
    rows = list(loaded["rows"])
    sample = random.Random(seed).sample(rows, min(sample_size, len(rows)))
    results: list[dict[str, Any]] = []
    for base in sample:
        result: dict[str, Any] = {
            "image_id": base["image_id"],
            "source_image_id": base["source_image_id"],
            "split": base["split"],
            "species_key": base["species_key"],
        }
        try:
            from app.detector_runtime import detect

            data, _ = _read_uri(base["source_image_gcs_uri"])
            with Image.open(BytesIO(data)) as source_image:
                detector_image = normalize_android_source(source_image)
            try:
                run = detect(detector_image)
            finally:
                detector_image.close()
            assessment = assess_detections(run.detections)
            candidate, metadata = _detector_payload(run, assessment)
            result.update(
                {
                    "candidate_bbox": candidate,
                    "detected": candidate is not None,
                    "detection_count": len(run.detections),
                    "detector_version": run.model_version,
                    "detector_confidence": metadata["detector_confidence"],
                    "bbox_area_ratio": metadata["bbox_area_ratio"],
                    "aspect_ratio": metadata["aspect_ratio"],
                    "touch_edge": metadata["touch_edge"],
                    "quality_score": metadata["quality_score"],
                    "quality_status": metadata["quality_status"] if candidate is not None else "BAD",
                    "all_detections": metadata["all_detections"],
                }
            )
        except Exception as exc:
            result.update({"detected": False, "quality_status": "ERROR", "error": str(exc)[:4000]})
        results.append(result)

    counts = {
        "detected": sum(1 for item in results if item.get("detected")),
        "no_detection": sum(1 for item in results if not item.get("detected")),
        "quality_good": sum(1 for item in results if item.get("quality_status") == "GOOD"),
        "quality_warning": sum(1 for item in results if item.get("quality_status") == "WARNING"),
        "quality_bad": sum(1 for item in results if item.get("quality_status") == "BAD"),
        "errors": sum(1 for item in results if item.get("quality_status") == "ERROR"),
    }
    return {
        "dataset_version": dataset_version,
        "source_type": "FROZEN_DATASET",
        "total": len(rows),
        "sample_size": len(results),
        "seed": seed,
        **counts,
        "items": results,
        "read_only": True,
    }


@router.get("/{dataset_version}/items")
def items(
    dataset_version: str,
    status: str = Query(default="BBOX_REQUIRED", max_length=32),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    # Direct unit callers do not receive FastAPI's coerced query values.
    if not isinstance(page, int):
        page = 1
    if not isinstance(page_size, int):
        page_size = 50
    rows = _ensure_rows(db, dataset_version)
    reviews = {
        row.image_id: row
        for row in db.scalars(
            select(DatasetCropReview).where(DatasetCropReview.source_dataset_version == dataset_version)
        ).all()
    }
    if not isinstance(status, str):
        status = "BBOX_REQUIRED"
    normalized = status.strip().upper()
    allowed = {"BBOX_REQUIRED", "ACCEPTED", "REJECTED", "TRAINING_READY", "ALL"}
    if normalized not in allowed:
        raise HTTPException(status_code=400, detail="invalid dataset crop review status")
    selected = [
        _item(base, reviews.get(base["image_id"]))
        for base in rows
        if normalized == "ALL" or (reviews.get(base["image_id"]) is not None and reviews[base["image_id"]].review_status == normalized)
    ]
    total = len(selected)
    start = (page - 1) * page_size
    page_items = selected[start : start + page_size]
    row_by_image_id = {row["image_id"]: row for row in rows}
    for item in page_items:
        review = reviews.get(item["image_id"])
        if review:
            base = row_by_image_id[item["image_id"]]
            _populate_candidate(review, base, db)
            item.update(_item(base, review))
    return {"dataset_version": dataset_version, "source_type": "FROZEN_DATASET", "total": total, "page": page, "page_size": page_size, "items": page_items}


@router.patch("/{dataset_version}/{image_id}")
def update(
    dataset_version: str,
    image_id: str,
    payload: DatasetCropReviewUpdate,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    rows = _ensure_rows(db, dataset_version)
    base = next((row for row in rows if row["image_id"] == image_id), None)
    if not base:
        raise HTTPException(status_code=404, detail="image not found in frozen dataset")
    decision = payload.decision.strip().upper()
    if decision == "SKIP":
        decision = "BBOX_REQUIRED"
    if decision not in {"BBOX_REQUIRED", "ACCEPTED", "REJECTED", "TRAINING_READY"}:
        raise HTTPException(status_code=400, detail="invalid dataset crop review decision")
    box = _box(payload.accepted_bbox)
    if decision in {"ACCEPTED", "TRAINING_READY"} and box is None:
        raise HTTPException(status_code=400, detail={"error": "ACCEPTED_BBOX_REQUIRED", "reason": "Frozen Dataset 只需人工确认 bbox"})
    row = db.scalar(
        select(DatasetCropReview).where(
            DatasetCropReview.source_dataset_version == dataset_version,
            DatasetCropReview.image_id == image_id,
        )
    )
    before = {"status": row.review_status, "accepted_bbox_json": row.accepted_bbox_json}
    row.accepted_bbox_json = json.dumps(box, separators=(",", ":")) if box is not None else None
    row.bbox_source = "accepted_review" if box is not None else None
    row.review_status = decision
    if decision in {"ACCEPTED", "TRAINING_READY"} and box is not None:
        row.crop_status = "PROCESSING"
        row.crop_error = None
        db.flush()
        try:
            row.crop_uri = _persist_crop_preview(base, box)
            row.crop_status = "READY"
        except Exception as exc:
            row.crop_uri = None
            row.crop_status = "ERROR"
            row.crop_error = str(exc)[:4000]
    elif decision in {"BBOX_REQUIRED", "REJECTED"}:
        row.crop_uri = None
        row.crop_status = "NOT_GENERATED"
        row.crop_error = None
    row.reviewer = payload.reviewer.strip() or "crop-review"
    row.reviewed_at = _now()
    row.updated_at = _now()
    db.add(
        DatasetCropReviewEvent(
            source_dataset_version=dataset_version,
            image_id=image_id,
            action="dataset_crop_bbox_review",
            reviewer=row.reviewer,
            before_json=json.dumps(before, ensure_ascii=False),
            after_json=json.dumps({"status": decision, "accepted_bbox": box}, ensure_ascii=False),
        )
    )
    db.commit()
    return _item(base, row)


@router.get("/{dataset_version}/{image_id}/image")
def image(dataset_version: str, image_id: str, db: Session = Depends(get_db)) -> Response:
    loaded = load_frozen_dataset(db, dataset_version)
    base = next((row for row in loaded["rows"] if row["image_id"] == image_id), None)
    if not base:
        raise HTTPException(status_code=404, detail="image not found in frozen dataset")
    try:
        data, _ = _read_uri(base["source_image_gcs_uri"])
    except Exception as exc:
        raise HTTPException(status_code=404, detail="source image unavailable") from exc
    media_type = "image/jpeg"
    uri = base["source_image_gcs_uri"].lower()
    if uri.endswith(".png"):
        media_type = "image/png"
    elif uri.endswith(".webp"):
        media_type = "image/webp"
    return Response(content=data, media_type=media_type)


@router.get("/{dataset_version}/{image_id}/crop")
def crop_preview(dataset_version: str, image_id: str, db: Session = Depends(get_db)) -> Response:
    row = db.scalar(
        select(DatasetCropReview).where(
            DatasetCropReview.source_dataset_version == dataset_version,
            DatasetCropReview.image_id == image_id,
        )
    )
    if not row or not row.crop_uri or row.crop_status != "READY":
        raise HTTPException(status_code=404, detail="crop preview is not ready")
    try:
        data, _ = _read_uri(row.crop_uri)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="crop preview unavailable") from exc
    return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=300"})


__all__ = ["router"]
