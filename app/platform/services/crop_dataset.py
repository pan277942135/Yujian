"""Accepted fish pool -> fixed-size crop dataset V0.1.

This service is intentionally platform-scoped.  It reads only human accepted
bbox reviews, writes immutable GCS artifacts, and registers the existing
DatasetVersion model.  It never changes review rows and never starts training.
"""

from __future__ import annotations

import csv
import io
import json
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any

from PIL import Image, ImageOps
from sqlalchemy import func, select

from app.db import SessionLocal
from app.factory import get_bucket_name
from app.models import BatchCropReview, DatasetVersion, ImageAsset
from app.presence import FishPresenceResult

CROP_DATASET_VERSION = "DS_CROP_M1_v0.1"
CROP_DATASET_TYPE = "CROP_IMAGE_V1"
CROP_PIPELINE_TYPE = "CROP_CLASSIFIER_V1"
CROP_EXPAND_RATIO = 1.25
CROP_OUTPUT_SIZE = 416
ACCEPTED_STATUSES = {"ACCEPTED", "TRAINING_READY"}
JOB_STATES = {"PENDING", "RUNNING", "SUCCESS", "FAILED"}

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="crop-dataset")
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    return value


def _bbox(value: Any) -> list[float] | None:
    parsed = _json(value)
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 4:
        return None
    try:
        x, y, width, height = [float(item) for item in parsed]
    except (TypeError, ValueError):
        return None
    if not all(0.0 <= item <= 1.0 for item in (x, y, width, height)):
        return None
    if width <= 0.0 or height <= 0.0 or x + width > 1.00001 or y + height > 1.00001:
        return None
    return [round(x, 6), round(y, 6), round(width, 6), round(height, 6)]


def _slug(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return clean.strip("._-")[:180] or "image"


def _parse_gs(uri: str) -> tuple[str, str]:
    value = str(uri or "").strip()
    if not value.startswith("gs://") or "/" not in value[5:]:
        raise ValueError("source image is not a valid gs:// URI")
    return tuple(value[5:].split("/", 1))  # type: ignore[return-value]


def _pool_rows(db) -> list[tuple[BatchCropReview, ImageAsset, FishPresenceResult | None]]:
    statement = (
        select(BatchCropReview, ImageAsset, FishPresenceResult)
        .join(ImageAsset, ImageAsset.id == BatchCropReview.image_asset_id)
        .outerjoin(FishPresenceResult, FishPresenceResult.image_asset_id == ImageAsset.id)
        .where(BatchCropReview.status.in_(ACCEPTED_STATUSES))
        .order_by(BatchCropReview.id)
    )
    return list(db.execute(statement).all())


def accepted_pool_count(db) -> int:
    return int(
        db.scalar(
            select(func.count())
            .select_from(BatchCropReview)
            .where(
                BatchCropReview.status.in_(ACCEPTED_STATUSES),
                BatchCropReview.accepted_bbox_json.is_not(None),
            )
        )
        or 0
    )


def accepted_pool_snapshot(db) -> dict[str, Any]:
    count = accepted_pool_count(db)
    return {
        "count": count,
        "ready": count > 0,
        "source": "accepted_bbox_pool",
        "accepted_statuses": sorted(ACCEPTED_STATUSES),
    }


def _expanded_box(box: list[float], width: int, height: int) -> tuple[int, int, int, int, bool]:
    x, y, box_width, box_height = box
    center_x = (x + box_width / 2.0) * width
    center_y = (y + box_height / 2.0) * height
    crop_width = box_width * CROP_EXPAND_RATIO * width
    crop_height = box_height * CROP_EXPAND_RATIO * height
    left = max(0, int(round(center_x - crop_width / 2.0)))
    top = max(0, int(round(center_y - crop_height / 2.0)))
    right = min(width, max(left + 1, int(round(center_x + crop_width / 2.0))))
    bottom = min(height, max(top + 1, int(round(center_y + crop_height / 2.0))))
    clipped = left == 0 or top == 0 or right == width or bottom == height
    return left, top, right, bottom, clipped


def _letterbox(data: bytes, box: list[float]) -> tuple[bytes, tuple[int, int, int, int], bool, tuple[int, int]]:
    with Image.open(io.BytesIO(data)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        pixel_box = _expanded_box(box, image.width, image.height)
        left, top, right, bottom, clipped = pixel_box
        crop = image.crop((left, top, right, bottom))
        resized = ImageOps.contain(crop, (CROP_OUTPUT_SIZE, CROP_OUTPUT_SIZE), method=Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (CROP_OUTPUT_SIZE, CROP_OUTPUT_SIZE), (255, 255, 255))
        canvas.paste(resized, ((CROP_OUTPUT_SIZE - resized.width) // 2, (CROP_OUTPUT_SIZE - resized.height) // 2))
        output = io.BytesIO()
        canvas.save(output, format="JPEG", quality=92)
        return output.getvalue(), (left, top, right, bottom), clipped, image.size


def _quality_status(
    review: BatchCropReview,
    presence: FishPresenceResult | None,
    clipped: bool,
    box: list[float] | None,
) -> tuple[str, str]:
    if box is None:
        return "INVALID", "accepted_bbox 无效"
    presence_status = str(getattr(presence, "status", "") or "").lower()
    if presence_status in {"no_fish", "multi_fish"} or int(getattr(presence, "fish_count", 1) or 1) != 1:
        return "INVALID", "自动鱼体检查不是单鱼"
    if clipped:
        return "WARNING", "扩展框触及原图边缘，需人工抽查"
    return "GOOD", ""


def _download(client, uri: str) -> bytes:
    bucket, object_name = _parse_gs(uri)
    return client.bucket(bucket).blob(object_name).download_as_bytes(timeout=180)


def _write_gcs_json(bucket, name: str, document: dict[str, Any]) -> None:
    bucket.blob(name).upload_from_string(
        json.dumps(document, ensure_ascii=False, indent=2).encode("utf-8"),
        content_type="application/json",
    )


def _set_job(job_id: str, **values: Any) -> dict[str, Any]:
    with _jobs_lock:
        current = dict(_jobs.get(job_id, {"job_id": job_id}))
        current.update(values)
        _jobs[job_id] = current
        return dict(current)


def _job_prefix(job_id: str) -> str:
    return f"datasets/_crop_jobs/{job_id}.json"


def _persist_job(job_id: str, **values: Any) -> dict[str, Any]:
    job = _set_job(job_id, **values)
    try:
        from google.cloud import storage

        bucket = storage.Client().bucket(get_bucket_name())
        _write_gcs_json(bucket, _job_prefix(job_id), job)
    except Exception:
        # Job status remains available in-process; a storage error must not
        # hide the actual crop build error.
        pass
    return job


def _make_dataset(
    job_id: str,
    dataset_name: str,
    expand_ratio: float,
    size: int,
) -> None:
    db = SessionLocal()
    started = _now()
    try:
        if dataset_name != CROP_DATASET_VERSION:
            raise ValueError(f"only {CROP_DATASET_VERSION} is supported in V0.1")
        if abs(expand_ratio - CROP_EXPAND_RATIO) > 1e-9 or size != CROP_OUTPUT_SIZE:
            raise ValueError("V0.1 requires expand_ratio=1.25 and size=416")
        if db.get(DatasetVersion, dataset_name):
            raise ValueError(f"dataset already exists: {dataset_name}")

        from google.cloud import storage

        client = storage.Client()
        bucket_name = get_bucket_name()
        bucket = client.bucket(bucket_name)
        prefix = f"datasets/{dataset_name}/"
        marker = bucket.blob(prefix + "metadata.json")
        if marker.exists(client):
            raise ValueError(f"dataset already exists in GCS: gs://{bucket_name}/{prefix}")

        rows = _pool_rows(db)
        source_count = len(rows)
        if source_count == 0:
            raise ValueError("accepted_bbox_pool is empty")
        _persist_job(job_id, status="RUNNING", source_count=source_count, processed=0, started_at=started)

        manifest_rows: list[dict[str, Any]] = []
        quality_counts = {"GOOD": 0, "WARNING": 0, "INVALID": 0}
        reasons: dict[str, int] = {}
        for index, (review, image, presence) in enumerate(rows, start=1):
            box = _bbox(review.accepted_bbox_json)
            species = str(review.species_name or review.species_key or image.truth_species or image.claimed_species or "").strip()
            quality, reason = _quality_status(review, presence, False, box)
            crop_path = ""
            if box is not None and species:
                try:
                    data = _download(client, image.gcs_uri)
                    encoded, pixel_box, clipped, source_size = _letterbox(data, box)
                    quality, reason = _quality_status(review, presence, clipped, box)
                    crop_path = f"images/{_slug(image.batch_id)}__{_slug(image.image_id)}_crop.jpg"
                    blob = bucket.blob(prefix + crop_path)
                    blob.upload_from_string(encoded, content_type="image/jpeg")
                    manifest_rows.append(
                        {
                            "image_id": image.image_id,
                            "crop_path": crop_path,
                            "species": species,
                            "source_image": image.gcs_uri,
                            "bbox": json.dumps(box, separators=(",", ":")),
                            "expand_ratio": f"{CROP_EXPAND_RATIO:.2f}",
                            "quality_status": quality,
                            "batch_id": image.batch_id,
                            "pixel_bbox": json.dumps(pixel_box, separators=(",", ":")),
                            "source_size": json.dumps(source_size, separators=(",", ":")),
                        }
                    )
                except Exception as exc:
                    quality, reason = "INVALID", f"裁剪失败: {type(exc).__name__}"
            elif not species:
                quality, reason = "INVALID", "accepted species 缺失"

            quality_counts[quality] = quality_counts.get(quality, 0) + 1
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
            _persist_job(job_id, processed=index, source_count=source_count, status="RUNNING")

        manifest_io = io.StringIO()
        fields = ["image_id", "crop_path", "species", "source_image", "bbox", "expand_ratio", "quality_status"]
        writer = csv.DictWriter(manifest_io, fieldnames=fields)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow({field: row.get(field, "") for field in fields})
        bucket.blob(prefix + "manifest.csv").upload_from_string(manifest_io.getvalue(), content_type="text/csv")

        generated_count = len(manifest_rows)
        metadata = {
            "dataset_version": dataset_name,
            "type": CROP_DATASET_TYPE,
            "pipeline_type": CROP_PIPELINE_TYPE,
            "source": "accepted_bbox_pool",
            "source_count": source_count,
            "generated_count": generated_count,
            "expand_ratio": CROP_EXPAND_RATIO,
            "input_size": f"{CROP_OUTPUT_SIZE}x{CROP_OUTPUT_SIZE}",
            "resize_mode": "crop_resize_letterbox",
            "created_by": "system",
            "created_at": _now(),
            "quality_counts": quality_counts,
            "gcs_prefix": f"gs://{bucket_name}/{prefix}",
            "safety": {
                "candidate_bbox_used": False,
                "accepted_bbox_only": True,
                "auto_train": False,
            },
        }
        quality_report = {
            "dataset_version": dataset_name,
            "source_count": source_count,
            "generated_count": generated_count,
            "quality_counts": quality_counts,
            "reasons": reasons,
            "good_for_training": quality_counts.get("GOOD", 0),
            "warning_for_review": quality_counts.get("WARNING", 0),
            "invalid_filtered": quality_counts.get("INVALID", 0),
        }
        _write_gcs_json(bucket, prefix + "metadata.json", metadata)
        _write_gcs_json(bucket, prefix + "quality_report.json", quality_report)

        train_count = sum(1 for row in manifest_rows if row["quality_status"] == "GOOD")
        review_count = sum(1 for row in manifest_rows if row["quality_status"] == "WARNING")
        dataset = DatasetVersion(
            dataset_version=dataset_name,
            manifest_uri=f"gs://{bucket_name}/{prefix}manifest.csv",
            class_map_uri=None,
            train_count=train_count,
            val_count=review_count,
            test_count=0,
            species_count=len({row["species"] for row in manifest_rows}),
            git_commit="platform-crop-v0.1",
            selection_mode="ACCEPTED_BBOX_CROP",
            source_cutoff_at=datetime.now(timezone.utc),
            status="READY_FOR_TRAINING",
            pipeline_type=CROP_PIPELINE_TYPE,
            metadata_json=json.dumps(metadata, ensure_ascii=False),
        )
        db.add(dataset)
        db.commit()
        _persist_job(
            job_id,
            status="SUCCESS",
            finished_at=_now(),
            source_count=source_count,
            processed=source_count,
            generated_count=generated_count,
            dataset_version=dataset_name,
            manifest_uri=metadata["gcs_prefix"] + "manifest.csv",
            quality_report_uri=metadata["gcs_prefix"] + "quality_report.json",
            metadata_uri=metadata["gcs_prefix"] + "metadata.json",
            quality_counts=quality_counts,
        )
    except Exception as exc:
        db.rollback()
        _persist_job(
            job_id,
            status="FAILED",
            finished_at=_now(),
            error_code="CROP_DATASET_GENERATION_FAILED",
            error=str(exc)[:1000],
        )
    finally:
        db.close()


def start_crop_dataset_job(
    *,
    dataset_name: str = CROP_DATASET_VERSION,
    expand_ratio: float = CROP_EXPAND_RATIO,
    size: int = CROP_OUTPUT_SIZE,
) -> dict[str, Any]:
    job_id = "crop_dataset_" + uuid.uuid4().hex[:16]
    _persist_job(
        job_id,
        status="PENDING",
        dataset_version=dataset_name,
        expand_ratio=expand_ratio,
        size=size,
        created_at=_now(),
    )
    _executor.submit(_make_dataset, job_id, dataset_name, expand_ratio, size)
    return get_crop_dataset_job(job_id)


def get_crop_dataset_job(job_id: str) -> dict[str, Any] | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is not None:
        return dict(job)
    try:
        from google.cloud import storage

        bucket = storage.Client().bucket(get_bucket_name())
        blob = bucket.blob(_job_prefix(job_id))
        if not blob.exists():
            return None
        return json.loads(blob.download_as_text(encoding="utf-8"))
    except Exception:
        return None


__all__ = [
    "CROP_DATASET_VERSION",
    "CROP_DATASET_TYPE",
    "CROP_EXPAND_RATIO",
    "CROP_OUTPUT_SIZE",
    "accepted_pool_count",
    "accepted_pool_snapshot",
    "get_crop_dataset_job",
    "start_crop_dataset_job",
]
