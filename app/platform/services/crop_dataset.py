"""Resumable Accepted BBox crop dataset builder V0.1."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
import re
import threading
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from PIL import Image, ImageDraw, ImageOps
from sqlalchemy import func, select

from app.db import SessionLocal
from app.factory import get_bucket_name
from app.freeze_policy import SPLIT_STRATEGY, _assign_stratified_group_splits
from app.models import BatchCropReview, DatasetVersion, ImageAsset
from app.presence import FishPresenceResult

CROP_DATASET_VERSION = "DS_CROP_M1_v0.1"
CROP_DATASET_TYPE = "CROP_IMAGE_V1"
CROP_PIPELINE_TYPE = "CROP_CLASSIFIER_V1"
CROP_EXPAND_RATIO = 1.25
CROP_OUTPUT_SIZE = 416
CROP_SPLIT_SEED = 20260827
CROP_CHUNK_SIZE = 25
QA_SCHEMA_VERSION = "RANDOM_50_QA_V1"
QA_SAMPLE_SIZE = 50
QA_SOURCE_MANIFEST = "manifest.csv"
QA_SPLIT_PLAN = (("train", 35), ("val", 8), ("test", 7))
QA_DECISIONS = {"PASS", "ISSUE", "CRITICAL"}
QUALITY_ANALYSIS_SCHEMA_VERSION = "QUALITY_GATE_ANALYSIS_V1"
QUALITY_ANALYSIS_SAMPLE_SIZE = 10
ACCEPTED_STATUSES = {"ACCEPTED", "TRAINING_READY"}
JOB_STATES = {"PENDING", "RUNNING", "SUCCESS", "FAILED"}

MANIFEST_FIELDS = (
    "image_id", "batch_id", "crop_path", "species", "source_image", "bbox",
    "pixel_bbox", "source_size", "expand_ratio", "fish_bbox_ratio",
    "crop_clipped", "quality_status", "quality_reason", "split",
)

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_job_locks: dict[str, threading.Lock] = {}
_job_locks_guard = threading.Lock()


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


def _pool_rows(db):
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


def _expanded_box(box: list[float], width: int, height: int):
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


def _letterbox(data: bytes, box: list[float]):
    with Image.open(io.BytesIO(data)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        left, top, right, bottom, clipped = _expanded_box(box, image.width, image.height)
        crop = image.crop((left, top, right, bottom))
        crop_area = max(1, (right - left) * (bottom - top))
        bbox_area = box[2] * image.width * box[3] * image.height
        ratio = bbox_area / crop_area
        resized = ImageOps.contain(crop, (CROP_OUTPUT_SIZE, CROP_OUTPUT_SIZE), method=Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (CROP_OUTPUT_SIZE, CROP_OUTPUT_SIZE), (255, 255, 255))
        canvas.paste(resized, ((CROP_OUTPUT_SIZE - resized.width) // 2, (CROP_OUTPUT_SIZE - resized.height) // 2))
        output = io.BytesIO()
        canvas.save(output, format="JPEG", quality=92)
        return output.getvalue(), (left, top, right, bottom), clipped, image.size, ratio


def evaluate_quality(*, box, species, presence_status, fish_count, clipped, crop_ok, bbox_area_ratio):
    invalid = []
    warnings = []
    if box is None:
        invalid.append("accepted_bbox_invalid")
    if not species:
        invalid.append("species_missing")
    status = str(presence_status or "").strip().lower()
    count = 1 if fish_count is None else int(fish_count)
    if status in {"no_fish", "multi_fish"} or count != 1:
        invalid.append("presence_not_single_fish")
    if box is not None and not crop_ok:
        invalid.append("crop_generation_failed")
    if clipped and not invalid:
        warnings.append("expanded_crop_touches_source_edge")
    if bbox_area_ratio is not None and (bbox_area_ratio < 0.10 or bbox_area_ratio > 0.95):
        warnings.append("bbox_area_ratio_outlier")
    if invalid:
        return "INVALID", ";".join(invalid)
    if warnings:
        return "WARNING", ";".join(warnings)
    return "GOOD", ""


def _storage():
    from google.cloud import storage
    client = storage.Client()
    return client, client.bucket(get_bucket_name())


def _download(client, uri: str) -> bytes:
    bucket, object_name = _parse_gs(uri)
    return client.bucket(bucket).blob(object_name).download_as_bytes(timeout=180)


def _write_json(bucket, name, value):
    bucket.blob(name).upload_from_string(
        json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8"),
        content_type="application/json",
    )


def _job_name(job_id):
    return f"datasets/_crop_jobs/{job_id}.json"


def _set_job(job_id, **values):
    with _jobs_lock:
        current = dict(_jobs.get(job_id, {"job_id": job_id}))
        current.update(values)
        _jobs[job_id] = current
        return dict(current)


def _persist_job(job_id, **values):
    values = dict(values)
    values.setdefault("updated_at", _now())
    job = _set_job(job_id, **values)
    _client, bucket = _storage()
    _write_json(bucket, _job_name(job_id), job)
    return job


def _raw_job(job_id):
    try:
        client, bucket = _storage()
        blob = bucket.blob(_job_name(job_id))
        if blob.exists(client):
            remote = json.loads(blob.download_as_text(encoding="utf-8"))
            if isinstance(remote, dict):
                _set_job(job_id, **remote)
                return remote
    except Exception:
        pass
    with _jobs_lock:
        return dict(_jobs[job_id]) if job_id in _jobs else None


def _public_job(job):
    if job is None:
        return None
    result = {key: value for key, value in job.items() if key != "source_refs"}
    result.setdefault("source_count", 0)
    result.setdefault("processed", int(result.get("cursor", 0) or 0))
    result["has_more"] = (
        result.get("status") != "SUCCESS"
        and int(result.get("cursor", 0) or 0) < int(result.get("source_count", 0) or 0)
    )
    result["quality_counts"] = {
        "GOOD": int(result.get("good_count", 0) or 0),
        "WARNING": int(result.get("warning_count", 0) or 0),
        "INVALID": int(result.get("invalid_count", 0) or 0),
    }
    return result


def get_crop_dataset_job(job_id):
    return _public_job(_raw_job(job_id))


def _job_lock(job_id):
    with _job_locks_guard:
        return _job_locks.setdefault(job_id, threading.Lock())


def _refs(rows):
    return [
        {
            "review_id": int(review.id),
            "image_asset_id": int(image.id),
            "batch_id": str(image.batch_id),
            "image_id": str(image.image_id),
        }
        for review, image, _presence in rows
    ]


def _has_successful_smoke():
    try:
        client, bucket = _storage()
        for blob in bucket.list_blobs(prefix="datasets/_crop_jobs/"):
            if not blob.name.endswith(".json"):
                continue
            try:
                job = json.loads(blob.download_as_text(encoding="utf-8"))
            except Exception:
                continue
            if (
                str(job.get("mode", "")).upper() == "SMOKE"
                and str(job.get("status", "")).upper() == "SUCCESS"
                and int(job.get("processed", 0) or 0) == int(job.get("source_count", 0) or 0)
            ):
                return True
    except Exception:
        pass
    with _jobs_lock:
        return any(
            str(job.get("mode", "")).upper() == "SMOKE"
            and str(job.get("status", "")).upper() == "SUCCESS"
            for job in _jobs.values()
        )


def _chunk_rows(db, refs):
    ids = [int(ref["review_id"]) for ref in refs]
    statement = (
        select(BatchCropReview, ImageAsset, FishPresenceResult)
        .join(ImageAsset, ImageAsset.id == BatchCropReview.image_asset_id)
        .outerjoin(FishPresenceResult, FishPresenceResult.image_asset_id == ImageAsset.id)
        .where(BatchCropReview.id.in_(ids))
    )
    by_id = {int(review.id): (review, image, presence) for review, image, presence in db.execute(statement).all()}
    missing = [review_id for review_id in ids if review_id not in by_id]
    if missing:
        raise ValueError(f"source snapshot rows missing: {missing[:5]}")
    return [by_id[review_id] for review_id in ids]


def _make_row(client, bucket, job, review, image, presence):
    box = _bbox(review.accepted_bbox_json)
    species = str(review.species_name or review.species_key or image.truth_species or image.claimed_species or "").strip()
    presence_status = str(getattr(presence, "status", "") or "").strip().lower()
    fish_count_value = getattr(presence, "fish_count", None)
    fish_count = None if fish_count_value is None else int(fish_count_value)
    row = {
        "image_id": str(image.image_id),
        "batch_id": str(image.batch_id),
        "crop_path": "",
        "species": species,
        "source_image": str(image.gcs_uri or ""),
        "bbox": json.dumps(box, separators=(",", ":")) if box else "",
        "pixel_bbox": "",
        "source_size": "",
        "expand_ratio": "1.25",
        "fish_bbox_ratio": "",
        "crop_clipped": "false",
        "quality_status": "INVALID",
        "quality_reason": "",
        "split": "",
    }
    encoded = None
    try:
        if box is not None and species:
            encoded, pixel_box, clipped, source_size, ratio = _letterbox(_download(client, image.gcs_uri), box)
            row["pixel_bbox"] = json.dumps(pixel_box, separators=(",", ":"))
            row["source_size"] = json.dumps(source_size, separators=(",", ":"))
            row["fish_bbox_ratio"] = f"{ratio:.6f}"
            row["crop_clipped"] = "true" if clipped else "false"
    except Exception as exc:
        row["quality_reason"] = f"crop_generation_failed:{type(exc).__name__}"
        row["quality_status"], reason = evaluate_quality(
            box=box, species=species, presence_status=presence_status, fish_count=fish_count,
            clipped=False, crop_ok=False, bbox_area_ratio=None,
        )
        row["quality_reason"] = ";".join(filter(None, [row["quality_reason"], reason]))
        return row

    if encoded is not None:
        prefix = str(job["artifact_prefix"]).rstrip("/")
        crop_path = f"images/{_slug(image.batch_id)}__{_slug(image.image_id)}_crop.jpg"
        bucket.blob(f"{prefix}/{crop_path}").upload_from_string(encoded, content_type="image/jpeg")
        row["crop_path"] = crop_path
        row["quality_status"], row["quality_reason"] = evaluate_quality(
            box=box, species=species, presence_status=presence_status, fish_count=fish_count,
            clipped=row["crop_clipped"] == "true", crop_ok=True,
            bbox_area_ratio=float(row["fish_bbox_ratio"]) if row["fish_bbox_ratio"] else None,
        )
        return row

    row["quality_status"], row["quality_reason"] = evaluate_quality(
        box=box, species=species, presence_status=presence_status, fish_count=fish_count,
        clipped=False, crop_ok=False, bbox_area_ratio=None,
    )
    return row


def _write_csv(bucket, name, rows):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})
    bucket.blob(name).upload_from_string(output.getvalue().encode("utf-8"), content_type="text/csv")


def _read_chunks(bucket, prefix):
    rows = []
    chunk_prefix = f"{prefix.rstrip('/')}/chunks/"
    blobs = sorted(
        [blob for blob in bucket.list_blobs(prefix=chunk_prefix) if blob.name.endswith(".csv")],
        key=lambda blob: blob.name,
    )
    for blob in blobs:
        rows.extend(csv.DictReader(io.StringIO(blob.download_as_text(encoding="utf-8"))))
    return rows


def _split_report(rows, dataset_name):
    quality = Counter(str(row.get("quality_status", "")).upper() for row in rows)
    for key in ("GOOD", "WARNING", "INVALID"):
        quality.setdefault(key, 0)
    selected = []
    for row in rows:
        if row.get("quality_status") != "GOOD":
            continue
        species = str(row.get("species") or "").strip()
        selected.append({
            "catalog": SimpleNamespace(species_key=species, common_name_zh=species),
            "group_key": f"{row.get('batch_id', '')}:{row.get('image_id', '')}",
            "row": row,
        })
    if selected:
        split = _assign_stratified_group_splits(selected, seed=CROP_SPLIT_SEED, train=0.70, val=0.15)
        for item in selected:
            item["row"]["split"] = item["split"]
    else:
        split = {"strategy": SPLIT_STRATEGY, "targets": {}, "per_species": {}, "warnings": [], "blockers": [], "group_count": 0}
    counts = {name: sum(1 for row in rows if row.get("split") == name) for name in ("train", "val", "test")}
    groups = defaultdict(set)
    for row in rows:
        if row.get("quality_status") == "GOOD" and row.get("split"):
            groups[f"{row.get('batch_id', '')}:{row.get('image_id', '')}"].add(row["split"])
    leaks = sorted(key for key, values in groups.items() if len(values) > 1)
    species_report = {}
    for row in rows:
        species = str(row.get("species") or "").strip() or "__MISSING__"
        item = species_report.setdefault(species, {"total": 0, "good": 0, "warning": 0, "invalid": 0, "train": 0, "val": 0, "test": 0})
        item["total"] += 1
        item[str(row.get("quality_status", "")).lower()] += 1
        if row.get("split") in counts:
            item[row["split"]] += 1
    source_count = len(rows)
    quality_sum = sum(quality[key] for key in ("GOOD", "WARNING", "INVALID")) == source_count
    split_sum = sum(counts.values()) == quality["GOOD"]
    report = {
        "dataset_version": dataset_name,
        "source_count": source_count,
        "quality": {key: int(quality[key]) for key in ("GOOD", "WARNING", "INVALID")},
        "split": counts,
        "species": dict(sorted(species_report.items())),
        "strategy": split.get("strategy", SPLIT_STRATEGY),
        "seed": CROP_SPLIT_SEED,
        "targets": split.get("targets", {}),
        "split_warnings": split.get("warnings", []),
        "split_blockers": split.get("blockers", []),
        "group_count": split.get("group_count", 0),
        "source_group_leak_groups": leaks,
        "quality_sum_check": quality_sum,
        "split_sum_check": split_sum,
        "source_group_leak_check": not leaks,
    }
    return report, counts


def _finalize(job, db):
    _client, bucket = _storage()
    prefix = str(job["artifact_prefix"]).rstrip("/")
    rows = _read_chunks(bucket, prefix)
    if len(rows) != int(job["source_count"]):
        raise ValueError(f"manifest source count mismatch: expected {job['source_count']}, got {len(rows)}")
    report, counts = _split_report(rows, str(job["dataset_version"]))
    if not report["quality_sum_check"] or not report["split_sum_check"]:
        raise ValueError("quality/split sum validation failed")
    if not report["source_group_leak_check"]:
        raise ValueError("source image group crossed split")
    all_name = f"{prefix}/manifest_all.csv"
    train_name = f"{prefix}/training_manifest.csv"
    manifest_name = f"{prefix}/manifest.csv"
    _write_csv(bucket, all_name, rows)
    training_rows = [row for row in rows if row.get("quality_status") == "GOOD"]
    _write_csv(bucket, train_name, training_rows)
    _write_csv(bucket, manifest_name, training_rows)
    quality = report["quality"]
    generated = sum(1 for row in rows if row.get("crop_path"))
    ratios = [float(row["fish_bbox_ratio"]) for row in rows if row.get("fish_bbox_ratio")]
    metadata = {
        "dataset_version": job["dataset_version"],
        "type": CROP_DATASET_TYPE,
        "pipeline_type": CROP_PIPELINE_TYPE,
        "source": "accepted_bbox_pool",
        "source_count": len(rows),
        "generated_count": generated,
        "good_count": quality["GOOD"],
        "warning_count": quality["WARNING"],
        "invalid_count": quality["INVALID"],
        "expand_ratio": CROP_EXPAND_RATIO,
        "input_size": f"{CROP_OUTPUT_SIZE}x{CROP_OUTPUT_SIZE}",
        "resize_mode": "crop_resize_letterbox",
        "split_strategy": SPLIT_STRATEGY,
        "split_seed": CROP_SPLIT_SEED,
        "accepted_bbox_only": True,
        "candidate_bbox_used": False,
        "auto_train": False,
        "fish_bbox_ratio": {
            "definition": "accepted bbox pixel area / expanded crop pixel area",
            "is_mask": False,
            "count": len(ratios),
            "min": min(ratios) if ratios else None,
            "max": max(ratios) if ratios else None,
            "mean": sum(ratios) / len(ratios) if ratios else None,
        },
        "split_counts": counts,
        "quality_sum_check": report["quality_sum_check"],
        "split_sum_check": report["split_sum_check"],
        "source_group_leak_check": report["source_group_leak_check"],
        "created_by": "system",
        "created_at": _now(),
        "gcs_prefix": f"gs://{get_bucket_name()}/{prefix}/",
    }
    _write_json(bucket, f"{prefix}/quality_report.json", {
        "dataset_version": job["dataset_version"], "source_count": len(rows),
        "generated_count": generated, "quality_counts": quality,
        "good_for_training": quality["GOOD"], "warning_for_review": quality["WARNING"],
        "invalid_filtered": quality["INVALID"], "fish_bbox_ratio": metadata["fish_bbox_ratio"],
        "reasons": dict(Counter(row.get("quality_reason", "") for row in rows if row.get("quality_reason"))),
    })
    _write_json(bucket, f"{prefix}/split_report.json", report)
    _write_json(bucket, f"{prefix}/metadata.json", metadata)
    uris = {
        "manifest_uri": f"gs://{get_bucket_name()}/{manifest_name}",
        "manifest_all_uri": f"gs://{get_bucket_name()}/{all_name}",
        "quality_report_uri": f"gs://{get_bucket_name()}/{prefix}/quality_report.json",
        "split_report_uri": f"gs://{get_bucket_name()}/{prefix}/split_report.json",
        "metadata_uri": f"gs://{get_bucket_name()}/{prefix}/metadata.json",
    }
    if str(job.get("mode", "")).upper() == "SMOKE":
        return _persist_job(
            job["job_id"], status="SUCCESS", finished_at=_now(), cursor=len(rows),
            processed=len(rows), generated_count=generated, good_count=quality["GOOD"],
            warning_count=quality["WARNING"], invalid_count=quality["INVALID"],
            split_counts=counts, quality_sum_check=report["quality_sum_check"],
            split_sum_check=report["split_sum_check"],
            source_group_leak_check=report["source_group_leak_check"], **uris,
        )
    dataset_name = str(job["dataset_version"])
    dataset = db.get(DatasetVersion, dataset_name)
    if dataset is None:
        dataset = DatasetVersion(dataset_version=dataset_name)
        db.add(dataset)
    dataset.manifest_uri = uris["manifest_uri"]
    dataset.class_map_uri = None
    dataset.train_count = int(counts["train"])
    dataset.val_count = int(counts["val"])
    dataset.test_count = int(counts["test"])
    dataset.species_count = len({row.get("species") for row in training_rows if row.get("species")})
    dataset.git_commit = os.getenv("APP_GIT_COMMIT", "platform-crop-v0.1")
    dataset.selection_mode = "ACCEPTED_BBOX_CROP"
    dataset.source_cutoff_at = datetime.now(timezone.utc)
    dataset.status = "READY_FOR_TRAINING"
    dataset.pipeline_type = CROP_PIPELINE_TYPE
    dataset.metadata_json = json.dumps(metadata, ensure_ascii=False)
    db.commit()
    return _persist_job(
        job["job_id"], status="SUCCESS", finished_at=_now(), cursor=len(rows),
        processed=len(rows), generated_count=generated, good_count=quality["GOOD"],
        warning_count=quality["WARNING"], invalid_count=quality["INVALID"],
        split_counts=counts, quality_sum_check=report["quality_sum_check"],
        split_sum_check=report["split_sum_check"],
        source_group_leak_check=report["source_group_leak_check"],
        dataset_status="READY_FOR_TRAINING", **uris,
    )


def step_crop_dataset_job(job_id):
    with _job_lock(job_id):
        job = _raw_job(job_id)
        if job is None:
            raise ValueError("crop dataset job not found")
        if job.get("status") == "SUCCESS":
            return _public_job(job)
        if job.get("status") == "FAILED":
            job = _persist_job(job_id, status="RUNNING", error=None, error_code=None)
        if not job.get("started_at"):
            job = _persist_job(job_id, status="RUNNING", started_at=_now())
        db = SessionLocal()
        try:
            cursor = int(job.get("cursor", 0) or 0)
            source_count = int(job.get("source_count", 0) or 0)
            if cursor >= source_count:
                return _public_job(_finalize(job, db))
            refs = list(job.get("source_refs") or [])
            end = min(source_count, cursor + int(job.get("chunk_size", CROP_CHUNK_SIZE)))
            records = _chunk_rows(db, refs[cursor:end])
            client, bucket = _storage()
            chunk = [_make_row(client, bucket, job, review, image, presence) for review, image, presence in records]
            prefix = str(job["artifact_prefix"]).rstrip("/")
            _write_csv(bucket, f"{prefix}/chunks/chunk_{cursor:08d}_{end:08d}.csv", chunk)
            counts = Counter(row.get("quality_status") for row in chunk)
            next_job = _persist_job(
                job_id, status="RUNNING", cursor=end, processed=end,
                generated_count=int(job.get("generated_count", 0) or 0) + sum(1 for row in chunk if row.get("crop_path")),
                good_count=int(job.get("good_count", 0) or 0) + counts["GOOD"],
                warning_count=int(job.get("warning_count", 0) or 0) + counts["WARNING"],
                invalid_count=int(job.get("invalid_count", 0) or 0) + counts["INVALID"],
            )
            if end < source_count:
                return _public_job(next_job)
            return _public_job(_finalize(next_job, db))
        except Exception as exc:
            db.rollback()
            try:
                failed = _persist_job(
                    job_id, status="FAILED", error_code="CROP_DATASET_STEP_FAILED",
                    error=str(exc)[:1000], cursor=int(job.get("cursor", 0) or 0),
                    processed=int(job.get("cursor", 0) or 0),
                )
            except Exception:
                failed = _set_job(job_id, status="FAILED", error_code="CROP_DATASET_STEP_FAILED", error=str(exc)[:1000])
            return _public_job(failed)
        finally:
            db.close()


def start_crop_dataset_job(*, source="accepted_bbox", dataset_name=CROP_DATASET_VERSION,
                           expand_ratio=CROP_EXPAND_RATIO, size=CROP_OUTPUT_SIZE,
                           mode="FULL", limit=None):
    if str(source).strip().lower() != "accepted_bbox":
        raise ValueError("source must be accepted_bbox")
    mode = str(mode or "FULL").strip().upper()
    if mode not in {"SMOKE", "FULL"}:
        raise ValueError("mode must be SMOKE or FULL")
    if abs(float(expand_ratio) - CROP_EXPAND_RATIO) > 1e-9 or int(size) != CROP_OUTPUT_SIZE:
        raise ValueError("V0.1 requires expand_ratio=1.25 and size=416")
    if mode == "FULL" and dataset_name != CROP_DATASET_VERSION:
        raise ValueError(f"FULL requires dataset_name={CROP_DATASET_VERSION}")
    if mode == "SMOKE":
        dataset_name = str(dataset_name or f"{CROP_DATASET_VERSION}_SMOKE")
        limit = max(1, min(int(limit or 20), 20))
    else:
        limit = None
    db = SessionLocal()
    try:
        rows = _pool_rows(db)
        if not rows:
            raise ValueError("accepted_bbox_pool is empty")
        if mode == "FULL":
            if not _has_successful_smoke():
                raise ValueError("SMOKE_REQUIRED_BEFORE_FULL")
            existing = db.get(DatasetVersion, dataset_name)
            if existing is not None and str(existing.status).upper() == "READY_FOR_TRAINING":
                raise ValueError(f"dataset already registered: {dataset_name}")
        selected = rows[:limit] if limit else rows
        job_id = "crop_dataset_" + uuid.uuid4().hex[:16]
        prefix = f"datasets/_crop_smoke/{job_id}" if mode == "SMOKE" else f"datasets/{dataset_name}"
        job = {
            "job_id": job_id, "dataset_version": dataset_name, "mode": mode, "status": "PENDING",
            "source_count": len(selected), "cursor": 0, "processed": 0,
            "generated_count": 0, "good_count": 0, "warning_count": 0, "invalid_count": 0,
            "chunk_size": CROP_CHUNK_SIZE, "expand_ratio": float(expand_ratio), "size": int(size),
            "split_strategy": SPLIT_STRATEGY, "split_seed": CROP_SPLIT_SEED,
            "artifact_prefix": prefix, "source_refs": _refs(selected),
            "created_at": _now(), "started_at": None, "updated_at": _now(),
            "finished_at": None, "error_code": None, "error": None,
        }
        _persist_job(job_id, **{key: value for key, value in job.items() if key != "job_id"})
        return _public_job(job)
    finally:
        db.close()



def _qa_artifact_prefix(dataset_name: str) -> str:
    return f"datasets/{dataset_name}/qa"


def _qa_uri(dataset_name: str, filename: str) -> str:
    return f"gs://{get_bucket_name()}/{_qa_artifact_prefix(dataset_name)}/{filename}"


def _qa_blob_name(dataset_name: str, filename: str) -> str:
    return f"{_qa_artifact_prefix(dataset_name)}/{filename}"


def _qa_seed(dataset_name: str) -> int:
    digest = hashlib.sha256(f"{QA_SCHEMA_VERSION}:{dataset_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 2147483647


def _qa_item_id(row: dict[str, Any]) -> str:
    image_id = str(row.get("image_id") or row.get("id") or row.get("crop_path") or row.get("source_image") or "").strip()
    batch_id = str(row.get("batch_id") or row.get("batch") or "").strip()
    return f"{batch_id}:{image_id}" if batch_id else image_id


def _qa_split(row: dict[str, Any]) -> str:
    return str(row.get("split") or row.get("dataset_split") or "").strip().lower()


def _qa_read(dataset_name: str) -> dict[str, Any] | None:
    try:
        client, bucket = _storage()
        blob = bucket.blob(_qa_blob_name(dataset_name, "random_50_qa.json"))
        if not blob.exists(client):
            return None
        value = json.loads(blob.download_as_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _qa_is_frozen_manifest(qa: dict[str, Any] | None) -> bool:
    return isinstance(qa, dict) and str(qa.get("source_manifest") or "") == QA_SOURCE_MANIFEST



def _qa_write_csv(bucket, dataset_name: str, qa: dict[str, Any]) -> None:
    output = io.StringIO(newline="")
    fields = ("qa_index", "item_id", *MANIFEST_FIELDS, "decision", "note", "reviewed_at")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for item in qa.get("items", []):
        writer.writerow({field: item.get(field, "") for field in fields})
    bucket.blob(_qa_blob_name(dataset_name, "random_50_qa.csv")).upload_from_string(
        output.getvalue().encode("utf-8"),
        content_type="text/csv",
    )


def _qa_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    reviewed = [item for item in items if str(item.get("decision") or "").upper() in QA_DECISIONS]
    pass_count = sum(1 for item in reviewed if str(item.get("decision")).upper() == "PASS")
    issue_count = sum(1 for item in reviewed if str(item.get("decision")).upper() in {"ISSUE", "CRITICAL"})
    critical_count = sum(1 for item in reviewed if str(item.get("decision")).upper() == "CRITICAL")
    if not reviewed:
        status, final_gate = "PENDING", "PARTIAL_PASS"
    elif len(reviewed) < len(items):
        status, final_gate = "IN_PROGRESS", "PARTIAL_PASS"
    elif issue_count:
        status, final_gate = "FAIL", "FAIL"
    else:
        status, final_gate = "PASS", "PASS"
    return {"status": status, "final_release_gate": final_gate, "reviewed_count": len(reviewed), "pass_count": pass_count, "issue_count": issue_count, "critical_count": critical_count}


def _qa_unique_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: (str(item.get("image_id") or item.get("id") or ""), str(item.get("batch_id") or item.get("batch") or ""), str(item.get("crop_path") or ""))):
        identity = _qa_item_id(row)
        if identity and identity not in unique:
            unique[identity] = row
    return list(unique.values())


def select_random_50_qa_rows(rows: list[dict[str, Any]], dataset_name: str) -> list[dict[str, Any]]:
    """Select a deterministic QA snapshot from the frozen training manifest.

    The production caller passes manifest.csv, which contains only GOOD rows and
    therefore uses the frozen train/val/test coverage plan. The legacy mixed
    manifest shape remains supported for existing unit fixtures and callers.
    """
    rng = random.Random(_qa_seed(dataset_name))

    def choose(pool: list[dict[str, Any]], count: int, label: str) -> list[dict[str, Any]]:
        candidates = _qa_unique_rows(pool)
        if len(candidates) < count:
            raise ValueError(f"RANDOM_50_QA_INSUFFICIENT_{label}_AVAILABLE_{len(candidates)}")
        return [candidates[index] for index in sorted(rng.sample(range(len(candidates)), count))]

    statuses = {str(row.get("quality_status") or "").upper() for row in rows}
    good = [row for row in rows if str(row.get("quality_status") or "GOOD").upper() == "GOOD"]
    selected: list[dict[str, Any]] = []
    if statuses - {"", "GOOD"}:
        # Compatibility for legacy mixed-manifest unit fixtures. Production QA
        # never takes this branch because it reads the frozen manifest.csv.
        warning = [row for row in rows if str(row.get("quality_status") or "").upper() == "WARNING"]
        invalid = [row for row in rows if str(row.get("quality_status") or "").upper() == "INVALID"]
        for split, count in (("train", 20), ("val", 5), ("test", 5)):
            selected.extend(choose([row for row in good if _qa_split(row) == split], count, f"GOOD_{split.upper()}"))
        selected.extend(choose(warning, 10, "WARNING"))
        selected.extend(choose(invalid, 10, "INVALID"))
    else:
        for split, count in QA_SPLIT_PLAN:
            selected.extend(choose([row for row in good if _qa_split(row) == split], count, f"GOOD_{split.upper()}"))
    if len({_qa_item_id(row) for row in selected}) != QA_SAMPLE_SIZE:
        raise ValueError("RANDOM_50_QA_DUPLICATE_SAMPLE")
    return selected


def _qa_payload(dataset_name: str, selected: list[dict[str, Any]]) -> dict[str, Any]:
    items = []
    for index, row in enumerate(selected):
        item = {field: row.get(field, "") for field in MANIFEST_FIELDS}
        item.update({"qa_index": index, "item_id": _qa_item_id(row), "decision": None, "note": "", "reviewed_at": None})
        items.append(item)
    payload = {"schema_version": QA_SCHEMA_VERSION, "dataset_version": dataset_name, "source_manifest": QA_SOURCE_MANIFEST, "sample_plan": {split: count for split, count in QA_SPLIT_PLAN}, "seed": _qa_seed(dataset_name), "sample_size": len(items), "created_at": _now(), "items": items}
    payload.update(_qa_summary(items))
    return payload


def _update_release_gate_metadata(db, dataset_name: str, qa: dict[str, Any]) -> None:
    dataset = db.get(DatasetVersion, dataset_name)
    if dataset is None:
        raise ValueError("dataset not found")
    metadata = _json(dataset.metadata_json) or {}
    metadata["release_gate"] = {
        "required": True,
        "random_50_qa": {
            "schema_version": qa.get("schema_version", QA_SCHEMA_VERSION),
            "status": qa.get("status", "NOT_PERFORMED"),
            "sample_size": int(qa.get("sample_size", QA_SAMPLE_SIZE) or QA_SAMPLE_SIZE),
            "reviewed_count": int(qa.get("reviewed_count", 0) or 0),
            "pass_count": int(qa.get("pass_count", 0) or 0),
            "issue_count": int(qa.get("issue_count", 0) or 0),
            "critical_count": int(qa.get("critical_count", 0) or 0),
            "source_manifest": qa.get("source_manifest", QA_SOURCE_MANIFEST),
            "sample_plan": qa.get("sample_plan", {split: count for split, count in QA_SPLIT_PLAN}),
            "qa_uri": _qa_uri(dataset_name, "random_50_qa.json"),
            "qa_csv_uri": _qa_uri(dataset_name, "random_50_qa.csv"),
        },
        "final_release_gate": qa.get("final_release_gate", "PARTIAL_PASS"),
    }
    dataset.metadata_json = json.dumps(metadata, ensure_ascii=False)
    db.commit()


def _persist_qa(dataset_name: str, qa: dict[str, Any], db) -> dict[str, Any]:
    _client, bucket = _storage()
    _write_json(bucket, _qa_blob_name(dataset_name, "random_50_qa.json"), qa)
    _qa_write_csv(bucket, dataset_name, qa)
    _update_release_gate_metadata(db, dataset_name, qa)
    return qa


def get_release_gate_summary(db, dataset_name: str) -> dict[str, Any] | None:
    dataset = db.get(DatasetVersion, dataset_name)
    if dataset is None:
        return None
    if str(getattr(dataset, "pipeline_type", "") or "").upper() != CROP_PIPELINE_TYPE:
        return None
    qa = _qa_read(dataset_name)
    if not _qa_is_frozen_manifest(qa):
        return {
            "required": True,
            "schema_version": QA_SCHEMA_VERSION,
            "source_manifest": QA_SOURCE_MANIFEST,
            "status": "NOT_PERFORMED",
            "sample_size": QA_SAMPLE_SIZE,
            "reviewed_count": 0,
            "pass_count": 0,
            "issue_count": 0,
            "critical_count": 0,
            "final_release_gate": "PARTIAL_PASS",
            "qa_uri": _qa_uri(dataset_name, "random_50_qa.json"),
            "qa_csv_uri": _qa_uri(dataset_name, "random_50_qa.csv"),
        }
    return {
        "required": True,
        "schema_version": qa.get("schema_version", QA_SCHEMA_VERSION),
        "source_manifest": QA_SOURCE_MANIFEST,
        "sample_plan": qa.get("sample_plan", {split: count for split, count in QA_SPLIT_PLAN}),
        "status": qa.get("status", "PENDING"),
        "sample_size": int(qa.get("sample_size", QA_SAMPLE_SIZE) or QA_SAMPLE_SIZE),
        "reviewed_count": int(qa.get("reviewed_count", 0) or 0),
        "pass_count": int(qa.get("pass_count", 0) or 0),
        "issue_count": int(qa.get("issue_count", 0) or 0),
        "critical_count": int(qa.get("critical_count", 0) or 0),
        "final_release_gate": qa.get("final_release_gate", "PARTIAL_PASS"),
        "qa_uri": _qa_uri(dataset_name, "random_50_qa.json"),
        "qa_csv_uri": _qa_uri(dataset_name, "random_50_qa.csv"),
    }


def get_random_50_qa(dataset_name: str) -> dict[str, Any]:
    qa = _qa_read(dataset_name)
    if not _qa_is_frozen_manifest(qa):
        return {
            "schema_version": QA_SCHEMA_VERSION,
            "dataset_version": dataset_name,
            "source_manifest": QA_SOURCE_MANIFEST,
            "sample_plan": {split: count for split, count in QA_SPLIT_PLAN},
            "seed": _qa_seed(dataset_name),
            "sample_size": QA_SAMPLE_SIZE,
            "status": "NOT_PERFORMED",
            "final_release_gate": "PARTIAL_PASS",
            "reviewed_count": 0,
            "pass_count": 0,
            "issue_count": 0,
            "critical_count": 0,
            "items": [],
        }
    return qa


def _qa_manifest_row_key(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("image_id") or row.get("id") or "").strip(),
        str(row.get("crop_path") or "").strip(),
    )


def _qa_enrich_frozen_rows(client, bucket, dataset_name: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Enrich legacy manifest.csv rows from the same frozen audit artifact.

    The sample universe remains manifest.csv. manifest_all.csv is only used to
    restore split/audit columns that were absent from an older frozen export.
    """
    if all(_qa_split(row) for row in rows):
        return rows
    audit_blob = bucket.blob(f"datasets/{dataset_name}/manifest_all.csv")
    if not audit_blob.exists(client):
        return rows
    audit_rows = list(csv.DictReader(audit_blob.download_as_text(encoding="utf-8")))
    by_key = {_qa_manifest_row_key(row): row for row in audit_rows}
    enriched: list[dict[str, Any]] = []
    for row in rows:
        current = dict(row)
        audit = by_key.get(_qa_manifest_row_key(row))
        if audit:
            for field in MANIFEST_FIELDS:
                if not str(current.get(field) or "").strip() and str(audit.get(field) or "").strip():
                    current[field] = audit[field]
        if not str(current.get("quality_status") or "").strip():
            current["quality_status"] = "GOOD"
        enriched.append(current)
    return enriched


def start_random_50_qa(dataset_name: str, db) -> dict[str, Any]:
    dataset = db.get(DatasetVersion, dataset_name)
    if dataset is None:
        raise ValueError("dataset not found")
    if str(getattr(dataset, "pipeline_type", "") or "").upper() != CROP_PIPELINE_TYPE:
        raise ValueError("RANDOM_50_QA_ONLY_FOR_CROP_DATASET")
    existing = _qa_read(dataset_name)
    if _qa_is_frozen_manifest(existing):
        return existing
    client, bucket = _storage()
    manifest_blob = bucket.blob(f"datasets/{dataset_name}/{QA_SOURCE_MANIFEST}")
    if not manifest_blob.exists(client):
        raise FileNotFoundError(f"{QA_SOURCE_MANIFEST} not found")
    rows = list(csv.DictReader(manifest_blob.download_as_text(encoding="utf-8")))
    rows = _qa_enrich_frozen_rows(client, bucket, dataset_name, rows)
    selected = select_random_50_qa_rows(rows, dataset_name)
    return _persist_qa(dataset_name, _qa_payload(dataset_name, selected), db)


def review_random_50_qa(dataset_name: str, qa_index: int, decision: str, note: str, db) -> dict[str, Any]:
    qa = _qa_read(dataset_name)
    if qa is None:
        raise ValueError("RANDOM_50_QA_NOT_STARTED")
    items = qa.get("items") or []
    if qa_index < 0 or qa_index >= len(items):
        raise ValueError("RANDOM_50_QA_ITEM_NOT_FOUND")
    decision = str(decision or "").strip().upper()
    if decision not in QA_DECISIONS:
        raise ValueError("RANDOM_50_QA_DECISION_INVALID")
    item = items[qa_index]
    item["decision"] = decision
    item["note"] = str(note or "").strip()[:1000]
    item["reviewed_at"] = _now()
    qa.update(_qa_summary(items))
    return _persist_qa(dataset_name, qa, db)


def _qa_source_bytes(client, bucket, dataset_name: str, source_image: str) -> bytes:
    source_image = str(source_image or "").strip()
    if not source_image:
        raise FileNotFoundError("RANDOM_50_QA_SOURCE_NOT_AVAILABLE")
    if source_image.startswith("gs://"):
        return _download(client, source_image)
    candidate = source_image.lstrip("/")
    if candidate.startswith("datasets/"):
        blob = bucket.blob(candidate)
    else:
        blob = bucket.blob(f"datasets/{dataset_name}/{candidate}")
    if not blob.exists(client):
        raise FileNotFoundError("RANDOM_50_QA_SOURCE_NOT_AVAILABLE")
    return blob.download_as_bytes(timeout=180)


def _qa_bbox_pixels(item: dict[str, Any]) -> tuple[int, int, int, int] | None:
    pixel_bbox = _json(item.get("pixel_bbox"))
    if isinstance(pixel_bbox, (list, tuple)) and len(pixel_bbox) == 4:
        try:
            left, top, right, bottom = [int(round(float(value))) for value in pixel_bbox]
            return left, top, right, bottom
        except (TypeError, ValueError):
            pass
    bbox = _bbox(item.get("bbox"))
    source_size = _json(item.get("source_size"))
    if bbox is None or not isinstance(source_size, (list, tuple)) or len(source_size) != 2:
        return None
    try:
        width, height = [float(value) for value in source_size]
        x, y, box_width, box_height = bbox
        return (
            int(round(x * width)),
            int(round(y * height)),
            int(round((x + box_width) * width)),
            int(round((y + box_height) * height)),
        )
    except (TypeError, ValueError):
        return None


def read_random_50_qa_media(dataset_name: str, qa_index: int, kind: str = "crop") -> bytes:
    qa = _qa_read(dataset_name)
    if not _qa_is_frozen_manifest(qa):
        raise FileNotFoundError("RANDOM_50_QA_NOT_STARTED")
    items = qa.get("items") or []
    if qa_index < 0 or qa_index >= len(items):
        raise FileNotFoundError("RANDOM_50_QA_ITEM_NOT_FOUND")
    item = items[qa_index]
    client, bucket = _storage()
    kind = str(kind or "crop").lower()
    if kind == "crop":
        crop_path = str(item.get("crop_path") or "")
        if not crop_path or crop_path.startswith("/") or ".." in crop_path:
            raise FileNotFoundError("RANDOM_50_QA_MEDIA_NOT_AVAILABLE")
        blob = bucket.blob(f"datasets/{dataset_name}/{crop_path}")
        if not blob.exists(client):
            raise FileNotFoundError("RANDOM_50_QA_MEDIA_NOT_AVAILABLE")
        return blob.download_as_bytes(timeout=120)
    if kind != "source_bbox":
        raise FileNotFoundError("RANDOM_50_QA_MEDIA_KIND_INVALID")
    source = Image.open(io.BytesIO(_qa_source_bytes(client, bucket, dataset_name, item.get("source_image")))).convert("RGB")
    bbox = _qa_bbox_pixels(item)
    if bbox is not None:
        draw = ImageDraw.Draw(source)
        left, top, right, bottom = bbox
        draw.rectangle((left, top, right, bottom), outline=(220, 38, 38), width=max(3, min(source.size) // 180))
    output = io.BytesIO()
    source.save(output, format="JPEG", quality=90)
    return output.getvalue()


def _analysis_seed(dataset_name: str, label: str = "") -> int:
    digest = hashlib.sha256(
        f"{QUALITY_ANALYSIS_SCHEMA_VERSION}:{dataset_name}:{label}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % 2147483647


def _analysis_reason_parts(value: Any) -> list[str]:
    parts = [part.strip() for part in str(value or "").split(";") if part.strip()]
    return parts or ["NO_REASON"]


def _analysis_row_key(row: dict[str, Any]) -> str:
    return ":".join(str(row.get(field) or "") for field in ("batch_id", "image_id", "crop_path"))


def _analysis_train_candidate(status: str, reason: str) -> str:
    if status != "WARNING":
        return "NOT_APPLICABLE"
    return "PENDING_HUMAN_REVIEW"


def _analysis_uri(dataset_name: str, filename: str) -> str:
    return f"gs://{get_bucket_name()}/datasets/{dataset_name}/reports/{filename}"


def _analysis_blob(dataset_name: str, filename: str) -> str:
    return f"datasets/{dataset_name}/reports/{filename}"


def _analysis_read(dataset_name: str) -> dict[str, Any] | None:
    try:
        client, bucket = _storage()
        blob = bucket.blob(_analysis_blob(dataset_name, "quality_gate_analysis.json"))
        if not blob.exists(client):
            return None
        value = json.loads(blob.download_as_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _analysis_recommendation_markdown(dataset_name, totals, reason_rows, warning_coverage) -> str:
    lines = [
        f"# Quality Gate V1.1 分布分析：{dataset_name}",
        "",
        "> 本报告只读分析冻结 Dataset 产物，不修改 quality_status、manifest、split 或 DatasetVersion。",
        "",
        "## 总体分布",
        "",
        f"- 总样本：{totals['TOTAL']}",
        f"- GOOD：{totals['GOOD']}（{totals['GOOD'] / max(1, totals['TOTAL']):.1%}）",
        f"- WARNING：{totals['WARNING']}（{totals['WARNING'] / max(1, totals['TOTAL']):.1%}）",
        f"- INVALID：{totals['INVALID']}（{totals['INVALID'] / max(1, totals['TOTAL']):.1%}）",
        "",
        "## WARNING 原因",
        "",
        "| 原因 | 数量 | 占 WARNING | 当前分析建议 |",
        "|---|---:|---:|---|",
    ]
    for item in reason_rows:
        if item["status"] != "WARNING":
            continue
        lines.append(
            f"| {item['reason']} | {item['count']} | "
            f"{item['ratio_within_status']:.1%} | {item['train_candidate']} |"
        )
    lines.extend(
        [
            "",
            f"WARNING 原因覆盖：{warning_coverage['rows_with_reason']} / "
            f"{warning_coverage['warning_count']}（{warning_coverage['coverage_ratio']:.1%}）。",
            "",
            "## 训练候选评估",
            "",
            "当前字段只能确认几何/流程异常，不能可靠判断鱼体是否完整。因此本报告不把 WARNING 自动提升为 GOOD；所有 WARNING 的 train_candidate 暂定为 PENDING_HUMAN_REVIEW。",
            "",
            "下一步建议：先查看 reports/quality_examples/ 中按原因导出的样例，再基于人工抽查结果制定 Quality Gate V1.1 规则；不要直接修改 DS_CROP_M1_v0.1。",
            "",
        ]
    )
    return "\n".join(lines)


def generate_quality_gate_analysis(dataset_name: str) -> dict[str, Any]:
    existing = _analysis_read(dataset_name)
    if existing is not None:
        return existing
    client, bucket = _storage()
    manifest_name = f"datasets/{dataset_name}/manifest_all.csv"
    manifest_blob = bucket.blob(manifest_name)
    if not manifest_blob.exists(client):
        raise FileNotFoundError("manifest_all.csv not found")
    rows = list(csv.DictReader(manifest_blob.download_as_text(encoding="utf-8")))
    totals = Counter(str(row.get("quality_status") or "").upper() for row in rows)
    total = len(rows)
    totals_payload = {
        "TOTAL": total,
        "GOOD": int(totals.get("GOOD", 0)),
        "WARNING": int(totals.get("WARNING", 0)),
        "INVALID": int(totals.get("INVALID", 0)),
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    warning_with_reason = 0
    for row in rows:
        status = str(row.get("quality_status") or "UNKNOWN").upper()
        reasons = _analysis_reason_parts(row.get("quality_reason"))
        if status == "WARNING" and reasons != ["NO_REASON"]:
            warning_with_reason += 1
        for reason in reasons:
            grouped[(status, reason)].append(row)

    example_uri_by_key: dict[str, list[str]] = {}
    example_exported = 0
    example_failed = 0
    for status, reason in sorted(grouped):
        candidates = _qa_unique_rows(grouped[(status, reason)])
        sample_size = min(QUALITY_ANALYSIS_SAMPLE_SIZE, len(candidates))
        rng = random.Random(_analysis_seed(dataset_name, f"{status}:{reason}"))
        selected = [candidates[i] for i in sorted(rng.sample(range(len(candidates)), sample_size))]
        safe_reason = _slug(reason)
        for index, row in enumerate(selected, start=1):
            key = _analysis_row_key(row)
            crop_path = str(row.get("crop_path") or "").strip()
            if not crop_path or crop_path.startswith("/") or ".." in crop_path:
                example_failed += 1
                continue
            source_blob = bucket.blob(f"datasets/{dataset_name}/{crop_path}")
            destination = f"datasets/{dataset_name}/reports/quality_examples/{status}_{safe_reason}/{index:03d}.jpg"
            try:
                if not source_blob.exists(client):
                    raise FileNotFoundError(crop_path)
                source_blob.copy_to(bucket, destination)
                example_uri_by_key.setdefault(key, []).append(f"gs://{get_bucket_name()}/{destination}")
                example_exported += 1
            except Exception:
                example_failed += 1

    reason_rows = []
    for (status, reason), group in sorted(grouped.items()):
        count = len(group)
        reason_rows.append(
            {
                "status": status,
                "reason": reason,
                "count": count,
                "ratio_within_status": count / max(1, totals_payload.get(status, count)),
                "ratio_total": count / max(1, total),
                "train_candidate": _analysis_train_candidate(status, reason),
                "sample_size": min(QUALITY_ANALYSIS_SAMPLE_SIZE, count),
                "example_uris": sorted({uri for row in group for uri in example_uri_by_key.get(_analysis_row_key(row), [])}),
            }
        )
    warning_coverage = {
        "warning_count": totals_payload["WARNING"],
        "rows_with_reason": warning_with_reason,
        "coverage_ratio": warning_with_reason / max(1, totals_payload["WARNING"]),
    }

    csv_output = io.StringIO(newline="")
    csv_fields = (
        "image_id", "batch_id", "species", "quality_status", "quality_reason",
        "train_candidate", "split", "crop_path", "source_image", "bbox",
        "pixel_bbox", "fish_bbox_ratio", "crop_clipped", "example_uri",
    )
    writer = csv.DictWriter(csv_output, fieldnames=csv_fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        status = str(row.get("quality_status") or "UNKNOWN").upper()
        reason = ";".join(_analysis_reason_parts(row.get("quality_reason")))
        writer.writerow({
            **{field: row.get(field, "") for field in csv_fields},
            "quality_status": status,
            "quality_reason": reason,
            "train_candidate": _analysis_train_candidate(status, reason),
            "example_uri": ";".join(example_uri_by_key.get(_analysis_row_key(row), [])),
        })

    recommendation = _analysis_recommendation_markdown(dataset_name, totals_payload, reason_rows, warning_coverage)
    report = {
        "schema_version": QUALITY_ANALYSIS_SCHEMA_VERSION,
        "dataset_version": dataset_name,
        "generated_at": _now(),
        "source": {
            "manifest_uri": f"gs://{get_bucket_name()}/{manifest_name}",
            "source_count": total,
            "source_is_frozen_manifest_all": True,
        },
        "totals": totals_payload,
        "ratios": {
            "GOOD": totals_payload["GOOD"] / max(1, total),
            "WARNING": totals_payload["WARNING"] / max(1, total),
            "INVALID": totals_payload["INVALID"] / max(1, total),
        },
        "warning_reason_coverage": warning_coverage,
        "reasons": reason_rows,
        "training_value": {
            "warning_train_candidate": "PENDING_HUMAN_REVIEW",
            "warning_count": totals_payload["WARNING"],
            "automatic_promotion": False,
        },
        "artifacts": {
            "analysis_json_uri": _analysis_uri(dataset_name, "quality_gate_analysis.json"),
            "analysis_csv_uri": _analysis_uri(dataset_name, "quality_gate_analysis.csv"),
            "recommendation_uri": _analysis_uri(dataset_name, "quality_gate_recommendation.md"),
            "examples_prefix": _analysis_uri(dataset_name, "quality_examples/"),
            "example_requested": sum(item["sample_size"] for item in reason_rows),
            "example_exported": example_exported,
            "example_failed": example_failed,
        },
    }
    _write_json(bucket, _analysis_blob(dataset_name, "quality_gate_analysis.json"), report)
    bucket.blob(_analysis_blob(dataset_name, "quality_gate_analysis.csv")).upload_from_string(
        csv_output.getvalue().encode("utf-8"), content_type="text/csv"
    )
    bucket.blob(_analysis_blob(dataset_name, "quality_gate_recommendation.md")).upload_from_string(
        recommendation.encode("utf-8"), content_type="text/markdown"
    )
    return report



def validate_crop_split_summary(rows):
    report, counts = _split_report(rows, "TEST")
    return {
        "quality_sum_check": report["quality_sum_check"],
        "split_sum_check": report["split_sum_check"],
        "source_group_leak_check": report["source_group_leak_check"],
        "counts": counts,
        "report": report,
    }


__all__ = [
    "CROP_DATASET_VERSION", "CROP_DATASET_TYPE", "CROP_PIPELINE_TYPE",
    "CROP_EXPAND_RATIO", "CROP_OUTPUT_SIZE", "CROP_SPLIT_SEED", "CROP_CHUNK_SIZE",
    "SPLIT_STRATEGY", "MANIFEST_FIELDS", "QA_SCHEMA_VERSION", "QA_SAMPLE_SIZE", "accepted_pool_count",
    "accepted_pool_snapshot", "evaluate_quality", "get_crop_dataset_job",
    "start_crop_dataset_job", "step_crop_dataset_job", "validate_crop_split_summary",
    "get_release_gate_summary", "get_random_50_qa", "start_random_50_qa",
    "review_random_50_qa", "read_random_50_qa_media", "select_random_50_qa_rows",
    "generate_quality_gate_analysis",
]
