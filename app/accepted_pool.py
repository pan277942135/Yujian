"""Persistent, incremental materialisation of the legacy Accepted Pool.

The review tables are the source of truth for the pool.  This module keeps the
derived crop files and manifest in GCS, so Dataset Freeze and the existing
Release QA workflow can consume the result without adding another database
model.  A pool key is stable for ``(batch_id, image_id)`` and a source
fingerprint makes the job idempotent: a second sync only processes new or
changed accepted bboxes.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models import BatchCropReview, DatasetVersion, ImageAsset, SpeciesCatalog
from app.platform.services import crop_dataset


ACCEPTED_POOL_SOURCE = "ACCEPTED_POOL"
ACCEPTED_POOL_DATASET_VERSION = "DS_CROP_M1_v0.2"
ACCEPTED_POOL_PREFIX = "datasets/accepted_pool"
ACCEPTED_POOL_MANIFEST_NAME = f"{ACCEPTED_POOL_PREFIX}/manifest.csv"
ACCEPTED_POOL_METADATA_NAME = f"{ACCEPTED_POOL_PREFIX}/metadata.json"
ACCEPTED_POOL_CLASS_MAP_NAME = f"{ACCEPTED_POOL_PREFIX}/class_map.json"
ACCEPTED_POOL_JOB_PREFIX = f"{ACCEPTED_POOL_PREFIX}/jobs"
ACCEPTED_POOL_CHUNK_SIZE = 25
ACCEPTED_POOL_CROP_SCALE = 1.0
ACCEPTED_POOL_CROP_SIZE = crop_dataset.CROP_OUTPUT_SIZE
ACCEPTED_STATUSES = {"ACCEPTED", "TRAINING_READY"}
JOB_STATES = {"PENDING", "RUNNING", "SUCCESS", "FAILED"}

# Keep the existing crop-manifest contract and add only provenance fields for
# the cumulative index.  The classifier and Release QA readers ignore the
# additive fields, while future syncs can decide whether a row is reusable.
CUMULATIVE_MANIFEST_FIELDS = tuple(
    dict.fromkeys(
        (
            *crop_dataset.ACCEPTED_POOL_MANIFEST_FIELDS,
            "pool_key",
            "source_fingerprint",
            "source_review_id",
            "pool_status",
            "pool_updated_at",
        )
    )
)

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_job_locks: dict[str, threading.Lock] = {}
_job_locks_guard = threading.Lock()
_worker_threads: dict[str, threading.Thread] = {}
_worker_threads_guard = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pool_key(batch_id: Any, image_id: Any) -> str:
    return f"{str(batch_id or '').strip()}:{str(image_id or '').strip()}"


def _json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    return value


def _bbox(value: Any) -> list[float] | None:
    return crop_dataset._bbox(value)


def _storage():
    """Use the existing crop service storage hook so tests and dev storage stay shared."""

    return crop_dataset._storage()


def _download(client, uri: str) -> bytes:
    return crop_dataset._download(client, uri)


def _write_json(bucket, name: str, value: dict[str, Any]) -> None:
    bucket.blob(name).upload_from_string(
        json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8"),
        content_type="application/json",
    )


def _job_name(job_id: str) -> str:
    return f"{ACCEPTED_POOL_JOB_PREFIX}/{job_id}.json"


def _chunk_name(job_id: str, start: int, end: int) -> str:
    return f"{ACCEPTED_POOL_JOB_PREFIX}/{job_id}/chunks/chunk_{start:08d}_{end:08d}.json"


def _set_job(job_id: str, **values: Any) -> dict[str, Any]:
    with _jobs_lock:
        current = dict(_jobs.get(job_id, {"job_id": job_id}))
        current.update(values)
        _jobs[job_id] = current
        return dict(current)


def _persist_job(job_id: str, **values: Any) -> dict[str, Any]:
    values = dict(values)
    values.setdefault("updated_at", _now())
    job = _set_job(job_id, **values)
    _client, bucket = _storage()
    _write_json(bucket, _job_name(job_id), job)
    return job


def _read_job(job_id: str) -> dict[str, Any] | None:
    with _jobs_lock:
        local = dict(_jobs[job_id]) if job_id in _jobs else None
    try:
        client, bucket = _storage()
        blob = bucket.blob(_job_name(job_id))
        if blob.exists(client):
            remote = json.loads(blob.download_as_text(encoding="utf-8"))
            if isinstance(remote, dict):
                _set_job(job_id, **{key: value for key, value in remote.items() if key != "job_id"})
                return remote
    except Exception:
        pass
    return local


def _public_job(job: dict[str, Any] | None) -> dict[str, Any] | None:
    if job is None:
        return None
    result = {key: value for key, value in job.items() if key not in {"source_refs", "active_keys"}}
    pending_count = int(result.get("pending_count", len(job.get("source_refs") or [])) or 0)
    cursor = int(result.get("cursor", 0) or 0)
    result.setdefault("source_count", 0)
    result["pending_count"] = pending_count
    result.setdefault("processed", cursor)
    result["has_more"] = result.get("status") not in {"SUCCESS", "FAILED"} and cursor < pending_count
    result.setdefault("bbox_generated", 0)
    result.setdefault("crop_generated", 0)
    result.setdefault("bbox_total", int(result.get("dataset_count", 0) or 0))
    result.setdefault("crop_total", int(result.get("dataset_count", 0) or 0))
    result.setdefault("dataset_count", 0)
    result.setdefault("failure_count", len(result.get("failure_records") or []))
    result.setdefault("manifest_created", False)
    result.setdefault("quality_analysis_mode", "RISK_ONLY")
    result.setdefault("source", ACCEPTED_POOL_SOURCE)
    return result


def get_accepted_pool_job(job_id: str) -> dict[str, Any] | None:
    return _public_job(_read_job(job_id))


def _job_lock(job_id: str) -> threading.Lock:
    with _job_locks_guard:
        return _job_locks.setdefault(job_id, threading.Lock())


def _read_csv(bucket, client, name: str) -> list[dict[str, str]]:
    blob = bucket.blob(name)
    if not blob.exists(client):
        return []
    text = blob.download_as_text(encoding="utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CUMULATIVE_MANIFEST_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in CUMULATIVE_MANIFEST_FIELDS})
    return output.getvalue().encode("utf-8")


def _write_manifest_files(bucket, rows: list[dict[str, Any]]) -> None:
    data = _csv_bytes(rows)
    # This is the only durable Accepted Pool index.  A formal DatasetVersion
    # is created later by the explicit Dataset Freeze action; sync must not
    # create a version-shaped artifact or database record.
    bucket.blob(ACCEPTED_POOL_MANIFEST_NAME).upload_from_string(data, content_type="text/csv")


def _manifest_uri(name: str) -> str:
    return f"gs://{crop_dataset.get_bucket_name()}/{name}"


def _choose_pool_split(key: str) -> str:
    # Reuse the canonical legacy deterministic splitter; importing lazily
    # avoids a module-level cycle because flywheel exposes the pool summary.
    from app.flywheel import _choose_split

    return _choose_split(key, crop_dataset.CROP_SPLIT_SEED, 0.70, 0.15)


def _source_rows(db: Session) -> tuple[list[dict[str, Any]], int]:
    """Snapshot accepted bbox rows in one DB query.

    The query intentionally uses ``BatchCropReview.accepted_bbox_json`` rather
    than the older approved-image selector.  An approved image without a
    confirmed bbox is not yet part of this pool.
    """

    statement = (
        select(BatchCropReview, ImageAsset)
        .join(ImageAsset, ImageAsset.id == BatchCropReview.image_asset_id)
        .where(
            ImageAsset.review_status == "approved",
            BatchCropReview.status.in_(ACCEPTED_STATUSES),
            BatchCropReview.accepted_bbox_json.is_not(None),
        )
        .order_by(BatchCropReview.id)
    )
    refs: list[dict[str, Any]] = []
    maps = _species_values(db)
    invalid_bbox_count = 0
    for review, image in db.execute(statement).all():
        box = _bbox(review.accepted_bbox_json)
        if box is None:
            invalid_bbox_count += 1
            continue
        refs.append(_source_ref(review, image, box, maps=maps, db=db))
    return refs, invalid_bbox_count


def _species_values(db: Session) -> tuple[dict[str, str], dict[str, str]]:
    rows = db.scalars(select(SpeciesCatalog)).all()
    by_key = {
        str(row.species_key).strip(): str(row.common_name_zh).strip()
        for row in rows
        if str(row.species_key or "").strip() and str(row.common_name_zh or "").strip()
    }
    return by_key, {name: key for key, name in by_key.items()}


def _species_for(db: Session, review: BatchCropReview, image: ImageAsset, maps=None) -> tuple[str, str]:
    by_key, by_name = maps or _species_values(db)
    raw = str(review.species_name or review.species_key or image.truth_species or image.claimed_species or "").strip()
    if raw in by_key:
        return raw, by_key[raw]
    if raw in by_name:
        return by_name[raw], raw
    return raw, raw


def _source_fingerprint(*, pool_key: str, source_image: str, species_key: str, species_name: str, box: list[float]) -> str:
    payload = {
        "pool_key": pool_key,
        "source_image": source_image,
        "species_key": species_key,
        "species_name": species_name,
        "accepted_bbox": box,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _source_ref(review: BatchCropReview, image: ImageAsset, box: list[float], maps=None, db: Session | None = None) -> dict[str, Any]:
    if maps is None and db is not None:
        maps = _species_values(db)
    species_key, species_name = _species_for(db, review, image, maps=maps)
    pool_key = _pool_key(image.batch_id, image.image_id)
    source_image = str(image.gcs_uri or "").strip()
    return {
        "review_id": int(review.id),
        "image_asset_id": int(image.id),
        "batch_id": str(image.batch_id),
        "image_id": str(image.image_id),
        "pool_key": pool_key,
        "accepted_bbox": box,
        "species_key": species_key,
        "species_name": species_name,
        "source_image": source_image,
        "source_fingerprint": _source_fingerprint(
            pool_key=pool_key,
            source_image=source_image,
            species_key=species_key,
            species_name=species_name,
            box=box,
        ),
    }


def _source_ref_map(db: Session, review_ids: list[int]) -> dict[int, dict[str, Any]]:
    if not review_ids:
        return {}
    statement = (
        select(BatchCropReview, ImageAsset)
        .join(ImageAsset, ImageAsset.id == BatchCropReview.image_asset_id)
        .where(BatchCropReview.id.in_(review_ids))
    )
    maps = _species_values(db)
    result = {}
    for review, image in db.execute(statement).all():
        box = _bbox(review.accepted_bbox_json)
        if box is None:
            continue
        result[int(review.id)] = _source_ref(review, image, box, maps=maps, db=db)
    return result


def _existing_key(row: dict[str, Any]) -> str:
    return str(row.get("pool_key") or _pool_key(row.get("source_batch") or row.get("batch_id"), row.get("image_id"))).strip()


def _existing_fingerprint(row: dict[str, Any]) -> str:
    stored = str(row.get("source_fingerprint") or "").strip()
    if stored:
        return stored
    box = _bbox(row.get("accepted_bbox") or row.get("bbox"))
    source = str(row.get("source_image") or row.get("source_image_gcs_uri") or row.get("image_path") or "").strip()
    species_key = str(row.get("species_key") or row.get("species") or "").strip()
    species_name = str(row.get("species_name") or row.get("species") or "").strip()
    if not box or not source:
        return ""
    return _source_fingerprint(
        pool_key=_existing_key(row),
        source_image=source,
        species_key=species_key,
        species_name=species_name,
        box=box,
    )


def _needs_materialisation(ref: dict[str, Any], row: dict[str, Any] | None) -> bool:
    if not row:
        return True
    if str(row.get("pool_status") or "ACTIVE").upper() != "ACTIVE":
        return True
    if str(row.get("bbox_source") or "").strip().lower() != "accepted_bbox":
        return True
    if not str(row.get("crop_path") or "").strip():
        return True
    return _existing_fingerprint(row) != str(ref.get("source_fingerprint") or "")


def _read_existing_manifest() -> tuple[list[dict[str, str]], str]:
    client, bucket = _storage()
    rows = _read_csv(bucket, client, ACCEPTED_POOL_MANIFEST_NAME)
    digest = hashlib.sha256(_csv_bytes(rows)).hexdigest() if rows else ""
    return rows, digest


def _read_class_map(bucket, client) -> list[dict[str, Any]]:
    blob = bucket.blob(ACCEPTED_POOL_CLASS_MAP_NAME)
    if not blob.exists(client):
        return []
    try:
        value = json.loads(blob.download_as_text(encoding="utf-8"))
    except Exception:
        return []
    return list(value.get("classes") or []) if isinstance(value, dict) else []


def _classes(existing_rows: list[dict[str, str]], refs: list[dict[str, Any]], prior: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for item in prior:
        key = str(item.get("species_key") or "").strip()
        if not key:
            continue
        try:
            index = int(item.get("class_index", len(by_key)))
        except (TypeError, ValueError):
            index = len(by_key)
        by_key[key] = {
            "class_index": index,
            "species_key": key,
            "common_name_zh": str(item.get("common_name_zh") or key),
            "common_name_en": item.get("common_name_en"),
            "status": item.get("status") or "active",
        }
    names: dict[str, str] = {}
    for row in existing_rows:
        key = str(row.get("species_key") or "").strip()
        if key:
            names.setdefault(key, str(row.get("species_name") or row.get("species") or key))
    for ref in refs:
        key = str(ref.get("species_key") or "").strip()
        if key:
            names.setdefault(key, str(ref.get("species_name") or key))
    next_index = max((int(item["class_index"]) for item in by_key.values()), default=-1) + 1
    for key in sorted(names):
        if key in by_key:
            continue
        by_key[key] = {
            "class_index": next_index,
            "species_key": key,
            "common_name_zh": names[key],
            "common_name_en": None,
            "status": "active",
        }
        next_index += 1
    return sorted(by_key.values(), key=lambda item: int(item["class_index"]))


def _class_map(classes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item.get("species_key")): item for item in classes if str(item.get("species_key") or "").strip()}


def _make_row(client, bucket, ref: dict[str, Any], job: dict[str, Any], class_by_key: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source_uri = str(ref.get("source_image") or "").strip()
    if not source_uri:
        raise ValueError("source image URI is missing")
    box = _bbox(ref.get("accepted_bbox"))
    if box is None:
        raise ValueError("accepted bbox is invalid")
    species_key = str(ref.get("species_key") or "").strip()
    species_name = str(ref.get("species_name") or species_key).strip()
    if not species_name:
        raise ValueError("accepted species is missing")
    class_item = class_by_key.get(species_key)
    if class_item is None:
        raise ValueError(f"species class is missing: {species_key or species_name}")
    data = _download(client, source_uri)
    encoded, pixel_box, clipped, source_size, ratio = crop_dataset._letterbox(
        data,
        box,
        crop_scale=ACCEPTED_POOL_CROP_SCALE,
    )
    batch_id = str(ref.get("batch_id") or "")
    image_id = str(ref.get("image_id") or "")
    crop_path = f"images/{crop_dataset._slug(batch_id)}__{crop_dataset._slug(image_id)}_crop.jpg"
    object_name = f"{ACCEPTED_POOL_PREFIX}/{crop_path}"
    bucket.blob(object_name).upload_from_string(encoded, content_type="image/jpeg")
    crop_uri = f"gs://{crop_dataset.get_bucket_name()}/{object_name}"
    quality_status, quality_reason = crop_dataset.evaluate_quality(
        box=box,
        species=species_name,
        presence_status="",
        fish_count=1,
        clipped=clipped,
        crop_ok=True,
        bbox_area_ratio=ratio,
    )
    created_at = str(ref.get("created_at") or "") or _now()
    return {
        "image_id": image_id,
        "image_path": source_uri,
        "species": species_name,
        "file_name": f"{crop_dataset._slug(image_id)}_crop.jpg",
        "species_key": species_key,
        "species_name": species_name,
        "class_index": int(class_item["class_index"]),
        "gcs_uri": crop_uri,
        "local_path": crop_path,
        "crop_image_path": crop_uri,
        "crop_path": crop_path,
        "input_type": "crop_image",
        "pipeline_type": crop_dataset.CROP_PIPELINE_TYPE,
        "source_image_id": image_id,
        "source_batch": batch_id,
        "batch_id": batch_id,
        "source_dataset": "",
        "source_manifest_uri": _manifest_uri(ACCEPTED_POOL_MANIFEST_NAME),
        "source_manifest_sha256": str(job.get("base_manifest_sha256") or ""),
        "source_image": source_uri,
        "source_image_path": source_uri,
        "source_image_gcs_uri": source_uri,
        "source_image_exists": "true",
        "detector_version": "",
        "split": _choose_pool_split(str(ref.get("pool_key") or _pool_key(batch_id, image_id))),
        "bbox": json.dumps(box, separators=(",", ":")),
        "detector_bbox": "",
        "pixel_bbox": json.dumps(pixel_box, separators=(",", ":")),
        "source_size": json.dumps(source_size, separators=(",", ":")),
        "accepted_bbox": json.dumps(box, separators=(",", ":")),
        "bbox_source": "accepted_bbox",
        "expand_ratio": ACCEPTED_POOL_CROP_SCALE,
        "crop_width": ACCEPTED_POOL_CROP_SIZE,
        "crop_height": ACCEPTED_POOL_CROP_SIZE,
        "crop_left": pixel_box[0],
        "crop_top": pixel_box[1],
        "crop_right": pixel_box[2],
        "crop_bottom": pixel_box[3],
        "fish_bbox_ratio": f"{ratio:.6f}",
        "crop_clipped": "true" if clipped else "false",
        "quality_status": quality_status,
        "quality_reason": quality_reason,
        "review_status": "ACCEPTED",
        "created_at": created_at,
        "pool_key": str(ref.get("pool_key") or _pool_key(batch_id, image_id)),
        "source_fingerprint": str(ref.get("source_fingerprint") or ""),
        "source_review_id": int(ref.get("review_id") or 0),
        "pool_status": "ACTIVE",
        "pool_updated_at": _now(),
    }


def _chunk_payload(bucket, client, job_id: str, start: int, end: int) -> dict[str, Any] | None:
    blob = bucket.blob(_chunk_name(job_id, start, end))
    if not blob.exists(client):
        return None
    try:
        value = json.loads(blob.download_as_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _write_chunk(bucket, job_id: str, start: int, end: int, rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    _write_json(
        bucket,
        _chunk_name(job_id, start, end),
        {"start": start, "end": end, "rows": rows, "failures": failures, "created_at": _now()},
    )


def _read_job_chunks(bucket, client, job_id: str) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    rows: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    prefix = f"{ACCEPTED_POOL_JOB_PREFIX}/{job_id}/chunks/"
    for blob in sorted(bucket.list_blobs(prefix=prefix), key=lambda item: item.name):
        if not blob.name.endswith(".json"):
            continue
        try:
            payload = json.loads(blob.download_as_text(encoding="utf-8"))
        except Exception:
            continue
        for row in payload.get("rows") or []:
            key = _existing_key(row)
            if key:
                rows[key] = row
        failures.extend(payload.get("failures") or [])
    return rows, failures


def _invalidate_release_qa(bucket) -> None:
    """Deprecated compatibility hook.

    Accepted Pool sync no longer owns a DatasetVersion or its Release QA
    snapshot.  Every frozen version has its own QA artifact and is therefore
    invalidated only when that version is explicitly regenerated (which is
    forbidden); retaining this no-op keeps old imports safe during rollout.
    """

    return None


def _metadata_for(job: dict[str, Any], rows: list[dict[str, Any]], *, changed: bool, previous: dict[str, Any]) -> dict[str, Any]:
    split_counts = {name: sum(1 for row in rows if str(row.get("split") or "") == name) for name in ("train", "val", "test")}
    risk_counts = Counter(str(row.get("quality_status") or "").upper() for row in rows)
    source_batches = sorted({str(row.get("source_batch") or row.get("batch_id") or "").strip() for row in rows if str(row.get("source_batch") or row.get("batch_id") or "").strip()})
    metadata = dict(previous) if isinstance(previous, dict) else {}
    metadata.update(
        {
            "pool_name": ACCEPTED_POOL_SOURCE,
            "type": "ACCEPTED_POOL_INDEX",
            "pipeline_type": crop_dataset.CROP_PIPELINE_TYPE,
            "source": ACCEPTED_POOL_SOURCE,
            "source_type": ACCEPTED_POOL_SOURCE,
            "source_selector": "BatchCropReview.ACCEPTED + ImageAsset.approved",
            "accepted_pool_mode": "CUMULATIVE_INCREMENTAL",
            "accepted_pool_manifest_uri": _manifest_uri(ACCEPTED_POOL_MANIFEST_NAME),
            "accepted_pool_count": len(rows),
            "current_accepted_bbox_count": int(job.get("source_count", 0) or 0),
            "source_count": int(job.get("source_count", 0) or 0),
            "bbox_generated": len(rows),
            "crop_generated": len(rows),
            "dataset_count": len(rows),
            "added_count": int(job.get("added_count", 0) or 0),
            "updated_count": int(job.get("updated_count", 0) or 0),
            "reused_count": int(job.get("reused_count", 0) or 0),
            "removed_count": int(job.get("removed_count", 0) or 0),
            "failure_count": int(job.get("failure_count", 0) or 0),
            "failures": list(job.get("failure_records") or []),
            "quality_analysis_mode": "RISK_ONLY",
            "quality_filter_applied": False,
            "quality_counts": {key: int(risk_counts.get(key, 0)) for key in ("GOOD", "WARNING", "INVALID")},
            "split_counts": split_counts,
            "split_strategy": "stable_hash_per_pool_key",
            "split_seed": crop_dataset.CROP_SPLIT_SEED,
            "bbox_source": "accepted_bbox",
            "bbox_expansion": False,
            "expand_ratio": ACCEPTED_POOL_CROP_SCALE,
            "input_size": f"{ACCEPTED_POOL_CROP_SIZE}x{ACCEPTED_POOL_CROP_SIZE}",
            "source_batches": source_batches,
            "manifest_uri": _manifest_uri(ACCEPTED_POOL_MANIFEST_NAME),
            "processing_status": "POOL_READY",
            "dataset_freeze_required": True,
            "incremental_sync": True,
            "last_sync_at": _now(),
        }
    )
    metadata.pop("release_gate", None)
    metadata.pop("release_qa_status", None)
    metadata.pop("training_gate_status", None)
    return metadata


def _finalize(job: dict[str, Any], db: Session) -> dict[str, Any]:
    client, bucket = _storage()
    chunk_rows, chunk_failures = _read_job_chunks(bucket, client, job["job_id"])
    failures = list(job.get("failure_records") or [])
    if chunk_failures:
        failures = chunk_failures
    if failures:
        failed = _persist_job(
            job["job_id"],
            status="FAILED",
            finished_at=_now(),
            failure_records=failures,
            failure_count=len(failures),
            error_code="ACCEPTED_POOL_MATERIALISATION_FAILED",
            error=(
                f"Accepted Pool 增量处理失败 {len(failures)} 条；首个失败 "
                f"{failures[0].get('image_id')}: {failures[0].get('error')}"
            )[:1000],
            dataset_status="FAILED",
        )
        return failed

    # Retain existing rows while merging the newly materialised chunks.  The
    # pool manifest is the durable source of truth; DatasetVersion artifacts
    # are written only by the explicit Freeze operation below.
    existing_rows, _ = _read_existing_manifest()
    existing = {_existing_key(row): row for row in existing_rows if _existing_key(row)}
    active_keys = set(job.get("active_keys") or [])
    final_by_key: dict[str, dict[str, Any]] = {}
    for key in active_keys:
        if key in chunk_rows:
            final_by_key[key] = chunk_rows[key]
        elif key in existing:
            final_by_key[key] = dict(existing[key])
        else:
            raise ValueError(f"accepted pool source row missing from result: {key}")
    rows = []
    class_by_key = _class_map(list(job.get("classes") or []))
    for key in sorted(final_by_key):
        row = dict(final_by_key[key])
        row["pool_key"] = key
        row["pool_status"] = "ACTIVE"
        if not str(row.get("split") or "").strip():
            row["split"] = _choose_pool_split(key)
        species_key = str(row.get("species_key") or row.get("species") or "").strip()
        if species_key in class_by_key:
            row["class_index"] = int(class_by_key[species_key]["class_index"])
        rows.append(row)

    previous_metadata: dict[str, Any] = {}
    try:
        metadata_blob = bucket.blob(ACCEPTED_POOL_METADATA_NAME)
        if metadata_blob.exists(client):
            previous_metadata = _json(metadata_blob.download_as_text(encoding="utf-8")) or {}
    except Exception:
        previous_metadata = {}
    if not isinstance(previous_metadata, dict):
        previous_metadata = {}
    added_count = int(job.get("added_count", 0) or 0)
    updated_count = int(job.get("updated_count", 0) or 0)
    removed_count = int(job.get("removed_count", 0) or 0)
    changed = bool(added_count or updated_count or removed_count or not previous_metadata.get("last_sync_at"))
    metadata = _metadata_for(job, rows, changed=changed, previous=previous_metadata)
    classes = list(job.get("classes") or [])

    class_map = {
        "pool_name": ACCEPTED_POOL_SOURCE,
        "pipeline_type": crop_dataset.CROP_PIPELINE_TYPE,
        "source": ACCEPTED_POOL_SOURCE,
        "classes": classes,
    }
    _write_manifest_files(bucket, rows)
    _write_json(bucket, ACCEPTED_POOL_CLASS_MAP_NAME, class_map)
    _write_json(bucket, ACCEPTED_POOL_METADATA_NAME, metadata)
    pool_manifest_sha256 = hashlib.sha256(_csv_bytes(rows)).hexdigest() if rows else ""

    return _persist_job(
        job["job_id"],
        status="SUCCESS",
        finished_at=_now(),
        cursor=int(job.get("pending_count", 0) or 0),
        processed=int(job.get("pending_count", 0) or 0),
        bbox_generated=int(job.get("generated_count", 0) or 0),
        crop_generated=int(job.get("generated_count", 0) or 0),
        bbox_total=len(rows),
        crop_total=len(rows),
        dataset_count=len(rows),
        manifest_created=True,
        failure_records=[],
        failure_count=0,
        dataset_status="POOL_READY",
        manifest_uri=_manifest_uri(ACCEPTED_POOL_MANIFEST_NAME),
        pool_manifest_uri=_manifest_uri(ACCEPTED_POOL_MANIFEST_NAME),
        pool_manifest_sha256=pool_manifest_sha256,
        classes=classes,
        split_counts=metadata["split_counts"],
        added_count=added_count,
        updated_count=updated_count,
        removed_count=removed_count,
        reused_count=int(job.get("reused_count", 0) or 0),
        accepted_pool_count=len(rows),
        quality_analysis_mode="RISK_ONLY",
        dataset_version=None,
    )


def _step(job_id: str) -> dict[str, Any]:
    with _job_lock(job_id):
        job = _read_job(job_id)
        if job is None:
            raise ValueError("Accepted Pool sync job not found")
        if job.get("status") in {"SUCCESS", "FAILED"}:
            return _public_job(job) or {}
        if not job.get("started_at"):
            job = _persist_job(job_id, status="RUNNING", started_at=_now(), dataset_status="BBOX_PROCESSING")
        db = SessionLocal()
        try:
            cursor = int(job.get("cursor", 0) or 0)
            pending = list(job.get("source_refs") or [])
            pending_count = int(job.get("pending_count", len(pending)) or 0)
            if cursor >= pending_count:
                return _public_job(_finalize(job, db)) or {}
            end = min(pending_count, cursor + int(job.get("chunk_size", ACCEPTED_POOL_CHUNK_SIZE)))
            client, bucket = _storage()
            cached = _chunk_payload(bucket, client, job_id, cursor, end)
            if cached is not None:
                rows = list(cached.get("rows") or [])
                failures = list(cached.get("failures") or [])
            else:
                refs = pending[cursor:end]
                by_review = _source_ref_map(db, [int(ref["review_id"]) for ref in refs])
                class_by_key = _class_map(list(job.get("classes") or []))
                rows = []
                failures = []
                for ref in refs:
                    current = by_review.get(int(ref["review_id"]))
                    if current is None:
                        failures.append({"image_id": ref.get("image_id"), "batch_id": ref.get("batch_id"), "stage": "SOURCE", "error": "accepted_bbox source row disappeared"})
                        continue
                    if current.get("source_fingerprint") != ref.get("source_fingerprint"):
                        failures.append({"image_id": ref.get("image_id"), "batch_id": ref.get("batch_id"), "stage": "SOURCE", "error": "accepted_bbox changed during sync; retry the sync"})
                        continue
                    image = db.get(ImageAsset, int(current["image_asset_id"]))
                    if image is None:
                        failures.append({"image_id": ref.get("image_id"), "batch_id": ref.get("batch_id"), "stage": "SOURCE", "error": "image asset disappeared"})
                        continue
                    try:
                        rows.append(_make_row(client, bucket, current, job, class_by_key))
                    except Exception as exc:
                        failures.append({"image_id": ref.get("image_id"), "batch_id": ref.get("batch_id"), "stage": "CROP", "error": str(exc)[:500]})
                _write_chunk(bucket, job_id, cursor, end, rows, failures)
            all_failures = list(job.get("failure_records") or []) + failures
            next_job = _persist_job(
                job_id,
                status="RUNNING",
                cursor=end,
                processed=end,
                generated_count=int(job.get("generated_count", 0) or 0) + len(rows),
                bbox_generated=int(job.get("bbox_generated", 0) or 0) + len(rows),
                crop_generated=int(job.get("crop_generated", 0) or 0) + len(rows),
                failure_records=all_failures,
                failure_count=len(all_failures),
                dataset_status="BBOX_PROCESSING" if end < pending_count else "POOL_READY",
            )
            if end < pending_count:
                return _public_job(next_job) or {}
            return _public_job(_finalize(next_job, db)) or {}
        except Exception as exc:
            db.rollback()
            failed = _persist_job(
                job_id,
                status="FAILED",
                finished_at=_now(),
                error_code="ACCEPTED_POOL_SYNC_FAILED",
                error=str(exc)[:1000],
                dataset_status="FAILED",
            )
            return _public_job(failed) or {}
        finally:
            db.close()


def step_accepted_pool_job(job_id: str) -> dict[str, Any]:
    return _step(job_id)


def _drain_accepted_pool_job(job_id: str) -> None:
    """Run a queued sync to completion outside the review request.

    The job and every completed chunk are persisted before the next chunk is
    started.  If a Cloud Run instance is recycled, the next Dataset page sync
    can resume from the persisted cursor without rebuilding completed crops.
    """

    terminal_job: dict[str, Any] | None = None
    try:
        try:
            for _ in range(100000):
                job = step_accepted_pool_job(job_id)
                if str(job.get("status") or "").upper() in {"SUCCESS", "FAILED"}:
                    terminal_job = job
                    break
            else:
                _persist_job(
                    job_id,
                    status="FAILED",
                    finished_at=_now(),
                    error_code="ACCEPTED_POOL_WORKER_LIMIT",
                    error="Accepted Pool 后台任务超过最大步数，请重新同步",
                    dataset_status="FAILED",
                )
        except Exception as exc:
            try:
                _persist_job(
                    job_id,
                    status="FAILED",
                    finished_at=_now(),
                    error_code="ACCEPTED_POOL_WORKER_FAILED",
                    error=str(exc)[:1000],
                    dataset_status="FAILED",
                )
            except Exception:
                # The persisted job is the recovery record; do not let a secondary
                # storage failure escape from a daemon thread.
                pass
        if str((terminal_job or {}).get("status") or "").upper() == "SUCCESS":
            # A review can land while the current snapshot is being processed.
            # Rescan once after finalisation so that those rows start a follow-up
            # incremental job automatically instead of waiting for a page visit.
            db = SessionLocal()
            try:
                if accepted_pool_summary(db).get("pending_count", 0):
                    follow_up = start_accepted_pool_sync(db)
                    _schedule_job(follow_up)
            except Exception:
                pass
            finally:
                db.close()
    finally:
        with _worker_threads_guard:
            _worker_threads.pop(job_id, None)


def _schedule_job(job: dict[str, Any]) -> None:
    job_id = str(job.get("job_id") or "").strip()
    if not job_id or str(job.get("status") or "").upper() in {"SUCCESS", "FAILED"}:
        return
    with _worker_threads_guard:
        current = _worker_threads.get(job_id)
        if current is not None and current.is_alive():
            return
        worker = threading.Thread(
            target=_drain_accepted_pool_job,
            args=(job_id,),
            name=f"accepted-pool-{job_id[-8:]}",
            daemon=True,
        )
        _worker_threads[job_id] = worker
        worker.start()


def _active_job() -> dict[str, Any] | None:
    with _jobs_lock:
        local = [dict(job) for job in _jobs.values() if job.get("status") in {"PENDING", "RUNNING"}]
    for job in local:
        return job
    try:
        client, bucket = _storage()
        for blob in sorted(bucket.list_blobs(prefix=f"{ACCEPTED_POOL_JOB_PREFIX}/"), key=lambda item: item.name, reverse=True):
            if not blob.name.endswith(".json") or "/chunks/" in blob.name:
                continue
            try:
                job = json.loads(blob.download_as_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(job, dict) and job.get("status") in {"PENDING", "RUNNING"}:
                job_id = str(job.get("job_id") or "")
                if job_id:
                    _set_job(job_id, **{key: value for key, value in job.items() if key != "job_id"})
                return job
    except Exception:
        pass
    return None


def start_accepted_pool_sync(db: Session) -> dict[str, Any]:
    """Create an idempotent sync job; the caller advances it via ``step``."""

    active = _active_job()
    if active is not None:
        result = _public_job(active) or {}
        result["idempotent_reuse"] = True
        return result
    refs, invalid_bbox_count = _source_rows(db)
    if not refs:
        raise ValueError("accepted_bbox_pool is empty or has no valid accepted_bbox")
    existing_rows, manifest_sha256 = _read_existing_manifest()
    existing = {_existing_key(row): row for row in existing_rows if _existing_key(row)}
    client, bucket = _storage()
    prior_classes = _read_class_map(bucket, client)
    classes = _classes(existing_rows, refs, prior_classes)
    pending_refs = [ref for ref in refs if _needs_materialisation(ref, existing.get(ref["pool_key"]))]
    added_count = sum(1 for ref in pending_refs if ref["pool_key"] not in existing)
    updated_count = len(pending_refs) - added_count
    active_keys = [ref["pool_key"] for ref in refs]
    removed_count = sum(1 for key, row in existing.items() if key not in set(active_keys) and str(row.get("pool_status") or "ACTIVE").upper() == "ACTIVE")
    job_id = "accepted_pool_sync_" + uuid.uuid4().hex[:16]
    job = {
        "job_id": job_id,
        "job_type": "ACCEPTED_POOL_INCREMENTAL_SYNC",
        "source": ACCEPTED_POOL_SOURCE,
        "source_type": ACCEPTED_POOL_SOURCE,
        # A sync materialises the shared pool only.  A DatasetVersion is
        # intentionally absent until the operator submits Dataset Freeze.
        "dataset_version": None,
        "status": "PENDING",
        "dataset_status": "BBOX_PROCESSING" if pending_refs else "POOL_READY",
        "source_count": len(refs),
        "invalid_bbox_count": invalid_bbox_count,
        "pending_count": len(pending_refs),
        "cursor": 0,
        "processed": 0,
        "generated_count": 0,
        "bbox_generated": 0,
        "crop_generated": 0,
        "bbox_total": len(existing_rows),
        "crop_total": len(existing_rows),
        "dataset_count": len(existing_rows),
        "added_count": added_count,
        "updated_count": updated_count,
        "removed_count": removed_count,
        "reused_count": max(len(refs) - len(pending_refs), 0),
        "failure_count": 0,
        "failure_records": [],
        "chunk_size": ACCEPTED_POOL_CHUNK_SIZE,
        "expand_ratio": ACCEPTED_POOL_CROP_SCALE,
        "size": ACCEPTED_POOL_CROP_SIZE,
        "bbox_source": "accepted_bbox",
        "quality_analysis_mode": "RISK_ONLY",
        "quality_filter_applied": False,
        "artifact_prefix": ACCEPTED_POOL_PREFIX,
        "pool_manifest_uri": _manifest_uri(ACCEPTED_POOL_MANIFEST_NAME),
        "manifest_uri": _manifest_uri(ACCEPTED_POOL_MANIFEST_NAME),
        "base_manifest_sha256": manifest_sha256,
        "active_keys": active_keys,
        "source_refs": pending_refs,
        "classes": classes,
        "created_at": _now(),
        "started_at": None,
        "updated_at": _now(),
        "finished_at": None,
        "error_code": None,
        "error": None,
    }
    _persist_job(job_id, **{key: value for key, value in job.items() if key != "job_id"})
    return _public_job(job) or {}


def accepted_pool_summary(db: Session) -> dict[str, Any]:
    """Return source/materialised counts for the old Dataset page card."""

    statement = (
        select(BatchCropReview, ImageAsset)
        .join(ImageAsset, ImageAsset.id == BatchCropReview.image_asset_id)
        .where(
            ImageAsset.review_status == "approved",
            BatchCropReview.status.in_(ACCEPTED_STATUSES),
            BatchCropReview.accepted_bbox_json.is_not(None),
        )
        .order_by(BatchCropReview.id)
    )
    maps = _species_values(db)
    species = Counter()
    refs = []
    for review, image in db.execute(statement).all():
        box = _bbox(review.accepted_bbox_json)
        if box is None:
            continue
        ref = _source_ref(review, image, box, maps=maps, db=db)
        refs.append(ref)
        species[str(ref.get("species_name") or "未标注")] += 1
    try:
        existing_rows, _ = _read_existing_manifest()
    except Exception:
        existing_rows = []
    existing = {_existing_key(row): row for row in existing_rows if _existing_key(row)}
    pending = sum(1 for ref in refs if _needs_materialisation(ref, existing.get(ref["pool_key"])))
    metadata: dict[str, Any] = {}
    try:
        client, bucket = _storage()
        blob = bucket.blob(ACCEPTED_POOL_METADATA_NAME)
        if blob.exists(client):
            value = json.loads(blob.download_as_text(encoding="utf-8"))
            if isinstance(value, dict):
                metadata = value
    except Exception:
        metadata = {}
    active_materialised = sum(1 for row in existing_rows if str(row.get("pool_status") or "ACTIVE").upper() == "ACTIVE")
    latest = _active_job()
    return {
        "source": ACCEPTED_POOL_SOURCE,
        "source_count": len(refs),
        "current_accepted_bbox_count": len(refs),
        "pool_count": active_materialised,
        "materialized_count": active_materialised,
        "pending_count": pending,
        "species": [{"species": name, "count": count} for name, count in sorted(species.items())],
        "accepted_pool_species": [{"species": name, "count": count} for name, count in sorted(species.items())],
        "manifest_uri": _manifest_uri(ACCEPTED_POOL_MANIFEST_NAME),
        "last_sync_at": metadata.get("last_sync_at"),
        "job": _public_job(latest),
    }


def _pool_manifest_rows() -> tuple[list[dict[str, str]], str]:
    """Read the materialised pool index and its content hash."""

    client, bucket = _storage()
    rows = _read_csv(bucket, client, ACCEPTED_POOL_MANIFEST_NAME)
    active = [
        row
        for row in rows
        if str(row.get("pool_status") or "ACTIVE").strip().upper() == "ACTIVE"
    ]
    active.sort(key=lambda row: _existing_key(row))
    digest = hashlib.sha256(_csv_bytes(active)).hexdigest() if active else ""
    return active, digest


def _pool_parent_version(db: Session, parent_version: str | None) -> str | None:
    if parent_version:
        parent = db.get(DatasetVersion, parent_version)
        if parent is None:
            raise ValueError(f"父版本不存在：{parent_version}")
        return parent.dataset_version
    parent = db.scalar(select(DatasetVersion).order_by(DatasetVersion.created_at.desc()).limit(1))
    return parent.dataset_version if parent else None


def _pool_split(row: dict[str, Any]) -> str:
    split = str(row.get("split") or "").strip().lower()
    if split in {"train", "val", "test"}:
        return split
    return _choose_pool_split(_existing_key(row))


def _species_token(value: Any) -> str:
    return str(value or "").strip().casefold()


def _candidate_species_tokens(db: Session) -> set[str]:
    """Return catalog identities that are explicitly marked as candidates.

    Accepted Pool remains the cumulative human-confirmed source of truth.  The
    candidate rule is applied only when a user creates a training Dataset
    Freeze, so changing a catalog status never deletes or rewrites pool rows.
    Both the stable key and the displayed Chinese name are indexed because old
    pool manifests may have been written before the catalog key was attached.
    """

    return {
        token
        for item in db.scalars(select(SpeciesCatalog)).all()
        if _species_token(item.status) == "candidate"
        for token in (_species_token(item.species_key), _species_token(item.common_name_zh))
        if token
    }


def _training_rows_from_pool(
    db: Session,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Exclude candidate species from this Freeze without changing the pool."""

    candidate_tokens = _candidate_species_tokens(db)
    eligible: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    for row in rows:
        key = _species_token(row.get("species_key") or row.get("species"))
        name = _species_token(row.get("species_name") or row.get("species"))
        if candidate_tokens.intersection({key, name}):
            label = str(row.get("species_name") or row.get("species_key") or row.get("species") or "未标注").strip()
            excluded[label] += 1
            continue
        eligible.append(row)
    return eligible, dict(sorted(excluded.items()))


def _copy_blob(bucket, source_blob, destination_name: str) -> None:
    """Copy a GCS object using the supported Bucket API.

    ``google.cloud.storage.Blob`` does not expose ``copy_to`` in the runtime
    client used by Cloud Run.  The supported operation is
    ``Bucket.copy_blob(source, destination_bucket, new_name)``.  The fallback
    keeps the lightweight in-memory storage adapters used by older tests and
    local development working during the rollout.
    """

    copy_blob = getattr(bucket, "copy_blob", None)
    if callable(copy_blob):
        copy_blob(source_blob, bucket, destination_name)
        return
    copy_to = getattr(source_blob, "copy_to", None)
    if callable(copy_to):
        copy_to(bucket, destination_name)
        return
    raise AttributeError("storage bucket does not support object copy")


def accepted_pool_freeze_preview(
    db: Session,
    *,
    dataset_version: str,
    parent_version: str | None = None,
    seed: int = crop_dataset.CROP_SPLIT_SEED,
    train: float = 0.70,
    val: float = 0.15,
) -> dict[str, Any]:
    """Return the exact Accepted Pool snapshot a formal Freeze would use.

    ``seed``/ratios are retained in the API contract for the old Freeze form,
    but the pool's stable per-key split is authoritative.  That keeps existing
    samples in the same split when later batches are appended.
    """

    if not str(dataset_version or "").startswith("DS_"):
        raise ValueError("数据集版本必须以 DS_ 开头")
    if db.get(DatasetVersion, dataset_version) is not None:
        raise ValueError(f"数据集版本已存在：{dataset_version}")
    if not (0 < train < 1 and 0 <= val < 1 and train + val < 1):
        raise ValueError("训练集/验证集比例不合法")

    parent = _pool_parent_version(db, parent_version)
    pool_rows, manifest_sha256 = _pool_manifest_rows()
    if not pool_rows:
        raise ValueError("Accepted Pool 尚未完成同步，请先同步 accepted_bbox")
    rows, candidate_excluded_species = _training_rows_from_pool(db, pool_rows)
    if not rows:
        raise ValueError("Accepted Pool 中没有可用于训练的非候选鱼种")

    missing_crop = [
        _existing_key(row)
        for row in rows
        if not str(row.get("crop_path") or "").strip()
    ]
    split_counts = Counter(_pool_split(row) for row in rows)
    species_counts = Counter(
        str(row.get("species_name") or row.get("species") or "未标注").strip()
        for row in rows
    )
    source_batches = sorted(
        {
            str(row.get("source_batch") or row.get("batch_id") or "").strip()
            for row in rows
            if str(row.get("source_batch") or row.get("batch_id") or "").strip()
        }
    )
    snapshot = {
        "dataset_version": dataset_version,
        "parent_version": parent,
        "source": ACCEPTED_POOL_SOURCE,
        "manifest_sha256": manifest_sha256,
        "items": [
            [
                _existing_key(row),
                str(row.get("source_fingerprint") or ""),
                _pool_split(row),
            ]
            for row in rows
        ],
    }
    selection_hash = hashlib.sha256(
        json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    blockers = []
    if missing_crop:
        blockers.append(
            {
                "code": "POOL_CROP_MISSING",
                "message": f"{len(missing_crop)} 条 Accepted Pool 尚未生成 Crop",
                "keys": missing_crop[:20],
            }
        )
    return {
        "dataset_version": dataset_version,
        "parent_version": parent,
        "source_mode": ACCEPTED_POOL_SOURCE,
        "selection_mode": "ACCEPTED_POOL_BBOX_CROP",
        "source_manifest_uri": _manifest_uri(ACCEPTED_POOL_MANIFEST_NAME),
        "source_manifest_sha256": manifest_sha256,
        "image_count": len(rows),
        "accepted_pool_count": len(pool_rows),
        "source_count": len(pool_rows),
        "training_input_count": len(rows),
        "candidate_excluded_count": sum(candidate_excluded_species.values()),
        "candidate_excluded_species": candidate_excluded_species,
        "species_count": len(species_counts),
        "species_counts": dict(species_counts),
        "source_batches": source_batches,
        "split_counts": {name: int(split_counts.get(name, 0)) for name in ("train", "val", "test")},
        "split_strategy": "stable_hash_per_pool_key",
        "split_seed": crop_dataset.CROP_SPLIT_SEED,
        "train_ratio": train,
        "val_ratio": val,
        "test_ratio": round(1.0 - train - val, 6),
        "quality_analysis_mode": "RISK_ONLY",
        "quality_filter_applied": False,
        "split_blockers": blockers,
        "freeze_ready": not blockers,
        "selection_hash": selection_hash,
    }


def _freeze_pool_manifest(rows: list[dict[str, Any]], dataset_version: str, bucket_name: str, pool_sha256: str) -> bytes:
    fields = ("dataset_version", *CUMULATIVE_MANIFEST_FIELDS)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for source in rows:
        row = dict(source)
        row["dataset_version"] = dataset_version
        row["split"] = _pool_split(row)
        row["source_manifest_uri"] = _manifest_uri(ACCEPTED_POOL_MANIFEST_NAME)
        row["source_manifest_sha256"] = pool_sha256
        crop_path = str(row.get("crop_path") or "").strip()
        destination_uri = f"gs://{bucket_name}/datasets/{dataset_version}/{crop_path}"
        row["gcs_uri"] = destination_uri
        row["crop_image_path"] = destination_uri
        row["local_path"] = crop_path
        writer.writerow({field: row.get(field, "") for field in fields})
    return output.getvalue().encode("utf-8")


def freeze_accepted_pool_dataset(
    db: Session,
    *,
    dataset_version: str,
    preview_hash: str,
    git_commit: str,
    parent_version: str | None = None,
    seed: int = crop_dataset.CROP_SPLIT_SEED,
    train: float = 0.70,
    val: float = 0.15,
    bucket_name: str | None = None,
) -> dict[str, Any]:
    """Create a formal immutable DatasetVersion from the cumulative pool.

    This function is called only by the explicit legacy Dataset Freeze route.
    It copies already materialised Crop objects; it never calls Detector or
    regenerates Crop data.
    """

    preview = accepted_pool_freeze_preview(
        db,
        dataset_version=dataset_version,
        parent_version=parent_version,
        seed=seed,
        train=train,
        val=val,
    )
    if preview.get("selection_hash") != str(preview_hash or ""):
        raise ValueError("Accepted Pool 冻结预览已失效，请重新生成预览")
    if not preview.get("freeze_ready"):
        raise ValueError("Accepted Pool 尚有 Crop 未完成，不能创建 Dataset Freeze")

    pool_rows, pool_sha256 = _pool_manifest_rows()
    rows, candidate_excluded_species = _training_rows_from_pool(db, pool_rows)
    if not rows:
        raise ValueError("Accepted Pool 中没有可用于训练的非候选鱼种")
    bucket_name = bucket_name or crop_dataset.get_bucket_name()
    client, bucket = _storage()
    out_prefix = f"datasets/{dataset_version}"
    marker = bucket.blob(f"{out_prefix}/dataset.json")
    if marker.exists(client):
        raise ValueError(f"数据集已存在：gs://{bucket_name}/{out_prefix}")

    classes = _classes(rows, [], _read_class_map(bucket, client))
    eligible_class_keys = {
        str(row.get("species_key") or row.get("species") or "").strip()
        for row in rows
        if str(row.get("species_key") or row.get("species") or "").strip()
    }
    classes = [item for item in classes if str(item.get("species_key") or "").strip() in eligible_class_keys]
    classes = [dict(item, class_index=index) for index, item in enumerate(classes)]
    class_by_key = _class_map(classes)
    frozen_rows: list[dict[str, Any]] = []
    for source in rows:
        crop_path = str(source.get("crop_path") or "").strip()
        if not crop_path or crop_path.startswith("/") or ".." in crop_path:
            raise ValueError(f"Accepted Pool Crop 路径无效：{_existing_key(source)}")
        source_name = f"{ACCEPTED_POOL_PREFIX}/{crop_path}"
        source_blob = bucket.blob(source_name)
        if not source_blob.exists(client):
            # Accept an already materialised URI from a pre-change pool as a
            # compatibility fallback, without rerunning Crop generation.
            source_uri = str(source.get("crop_image_path") or source.get("gcs_uri") or "").strip()
            if source_uri.startswith("gs://"):
                body = source_uri[5:]
                source_bucket, source_object = body.split("/", 1) if "/" in body else ("", "")
                if source_bucket == bucket_name and source_object:
                    source_blob = bucket.blob(source_object)
            if not source_blob.exists(client):
                raise ValueError(f"Accepted Pool Crop 不存在：{_existing_key(source)}")
        destination_name = f"{out_prefix}/{crop_path}"
        _copy_blob(bucket, source_blob, destination_name)

        row = dict(source)
        row["dataset_version"] = dataset_version
        row["split"] = _pool_split(source)
        row["source_manifest_uri"] = _manifest_uri(ACCEPTED_POOL_MANIFEST_NAME)
        row["source_manifest_sha256"] = pool_sha256
        destination_uri = f"gs://{bucket_name}/{destination_name}"
        row["gcs_uri"] = destination_uri
        row["crop_image_path"] = destination_uri
        row["local_path"] = crop_path
        species_key = str(row.get("species_key") or row.get("species") or "").strip()
        if species_key in class_by_key:
            row["class_index"] = int(class_by_key[species_key]["class_index"])
        frozen_rows.append(row)

    split_counts = Counter(str(row.get("split") or "") for row in frozen_rows)
    species_counts = Counter(str(row.get("species_name") or row.get("species") or "未标注") for row in frozen_rows)
    cutoff = datetime.now(timezone.utc)
    manifest_uri = f"gs://{bucket_name}/{out_prefix}/manifest.csv"
    class_map_uri = f"gs://{bucket_name}/{out_prefix}/class_map.json"
    metadata = {
        "dataset_version": dataset_version,
        "parent_version": preview.get("parent_version"),
        "created_at": cutoff.isoformat(),
        "source_cutoff_at": cutoff.isoformat(),
        "source": ACCEPTED_POOL_SOURCE,
        "source_type": ACCEPTED_POOL_SOURCE,
        "source_selector": "cumulative Accepted Pool",
        "selection_mode": "ACCEPTED_POOL_BBOX_CROP",
        "pipeline_type": crop_dataset.CROP_PIPELINE_TYPE,
        "pool_manifest_uri": _manifest_uri(ACCEPTED_POOL_MANIFEST_NAME),
        "pool_manifest_sha256": pool_sha256,
        "accepted_pool_count": len(pool_rows),
        "source_count": len(pool_rows),
        "training_input_count": len(frozen_rows),
        "candidate_excluded_count": sum(candidate_excluded_species.values()),
        "candidate_excluded_species": candidate_excluded_species,
        "bbox_generated": len(frozen_rows),
        "crop_generated": len(frozen_rows),
        "dataset_count": len(frozen_rows),
        "source_batches": preview.get("source_batches") or [],
        "quality_analysis_mode": "RISK_ONLY",
        "quality_filter_applied": False,
        "quality_counts": Counter(str(row.get("quality_status") or "").upper() for row in frozen_rows),
        "split_counts": {name: int(split_counts.get(name, 0)) for name in ("train", "val", "test")},
        "split_strategy": "stable_hash_per_pool_key",
        "split_seed": crop_dataset.CROP_SPLIT_SEED,
        "processing_status": "READY_FOR_TRAINING",
        "manifest_uri": manifest_uri,
        "class_map_uri": class_map_uri,
        "git_commit": git_commit or "unknown",
        "immutable": True,
        "selection_hash": preview.get("selection_hash"),
    }
    metadata["quality_counts"] = dict(metadata["quality_counts"])
    manifest_data = _freeze_pool_manifest(frozen_rows, dataset_version, bucket_name, pool_sha256)
    bucket.blob(f"{out_prefix}/manifest.csv").upload_from_string(manifest_data, content_type="text/csv")
    bucket.blob(f"{out_prefix}/manifest_all.csv").upload_from_string(manifest_data, content_type="text/csv")
    bucket.blob(f"{out_prefix}/training_manifest.csv").upload_from_string(manifest_data, content_type="text/csv")
    bucket.blob(f"{out_prefix}/class_map.json").upload_from_string(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "parent_version": preview.get("parent_version"),
                "source": ACCEPTED_POOL_SOURCE,
                "classes": classes,
            },
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8"),
        content_type="application/json",
    )
    marker.upload_from_string(json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"), content_type="application/json")

    db.add(
        DatasetVersion(
            dataset_version=dataset_version,
            parent_version=preview.get("parent_version"),
            manifest_uri=manifest_uri,
            class_map_uri=class_map_uri,
            train_count=int(split_counts.get("train", 0)),
            val_count=int(split_counts.get("val", 0)),
            test_count=int(split_counts.get("test", 0)),
            species_count=len(classes),
            git_commit=git_commit or "unknown",
            selection_mode="ACCEPTED_POOL_BBOX_CROP",
            source_cutoff_at=cutoff,
            status="FROZEN",
            pipeline_type=crop_dataset.CROP_PIPELINE_TYPE,
            metadata_json=json.dumps(metadata, ensure_ascii=False),
        )
    )
    db.commit()
    return {
        **preview,
        "status": "FROZEN",
        "manifest_uri": manifest_uri,
        "class_map_uri": class_map_uri,
        "image_count": len(frozen_rows),
        "split_counts": {name: int(split_counts.get(name, 0)) for name in ("train", "val", "test")},
        "species_counts": dict(species_counts),
        "git_commit": git_commit or "unknown",
    }


def enqueue_accepted_pool_sync(db: Session) -> dict[str, Any] | None:
    """Best-effort enqueue used after review commits.

    Review writes must remain usable in local environments without GCS.  The
    explicit Dataset page sync endpoint remains the retryable source of truth.
    """

    try:
        job = start_accepted_pool_sync(db)
        _schedule_job(job)
        return job
    except Exception:
        return None


__all__ = [
    "ACCEPTED_POOL_DATASET_VERSION",
    "ACCEPTED_POOL_MANIFEST_NAME",
    "ACCEPTED_POOL_SOURCE",
    "CUMULATIVE_MANIFEST_FIELDS",
    "accepted_pool_summary",
    "accepted_pool_freeze_preview",
    "enqueue_accepted_pool_sync",
    "freeze_accepted_pool_dataset",
    "get_accepted_pool_job",
    "start_accepted_pool_sync",
    "step_accepted_pool_job",
]
