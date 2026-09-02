"""Read-only Dataset Freeze bridge for Crop Annotation Review.

This module deliberately resolves the parent DatasetVersion's registered
manifest and class-map URIs.  It never guesses a GCS path and never uses a raw
Batch as a Frozen Dataset source.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from app.models import DatasetVersion


def _read_uri(uri: str, storage_client: Any = None) -> tuple[bytes, str | None]:
    value = str(uri or "").strip()
    if not value:
        raise ValueError("registered URI is empty")
    if value.startswith("gs://"):
        if storage_client is None:
            from google.cloud import storage

            storage_client = storage.Client()
        body = value[5:]
        if "/" not in body:
            raise ValueError(f"invalid GCS URI: {value}")
        bucket_name, object_name = body.split("/", 1)
        blob = storage_client.bucket(bucket_name).blob(object_name)
        if not blob.exists(storage_client):
            raise FileNotFoundError(value)
        data = blob.download_as_bytes()
        generation = str(getattr(blob, "generation", "") or "") or None
        return data, generation
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(value)
    return path.read_bytes(), None


def _uri_exists(uri: str, storage_client: Any = None) -> bool:
    value = str(uri or "").strip()
    if value.startswith("gs://"):
        if storage_client is None:
            from google.cloud import storage

            storage_client = storage.Client()
        body = value[5:]
        if "/" not in body:
            return False
        bucket_name, object_name = body.split("/", 1)
        return bool(storage_client.bucket(bucket_name).blob(object_name).exists(storage_client))
    return Path(value).is_file()


def _parse_class_map(data: bytes) -> dict[str, dict[str, Any]]:
    document = json.loads(data.decode("utf-8"))
    classes = document.get("classes") if isinstance(document, dict) else None
    if not isinstance(classes, list) or not classes:
        raise ValueError("registered class_map has no classes")
    result: dict[str, dict[str, Any]] = {}
    indexes: set[int] = set()
    for item in classes:
        if not isinstance(item, Mapping):
            raise ValueError("class_map contains an invalid class row")
        key = str(item.get("species_key") or item.get("key") or "").strip()
        if not key:
            raise ValueError("class_map class is missing species_key")
        index = int(item.get("class_index"))
        if index in indexes or index < 0:
            raise ValueError("class_map class_index is duplicated or invalid")
        indexes.add(index)
        result[key] = {"class_index": index, **dict(item)}
    return result


def load_frozen_dataset(db: Any, dataset_version: str, storage_client: Any = None, *, verify_source_images: bool = False) -> dict[str, Any]:
    dataset = db.get(DatasetVersion, dataset_version)
    if not dataset:
        raise ValueError(f"Frozen Dataset 不存在：{dataset_version}")
    status = str(dataset.status or "").upper()
    if status != "FROZEN":
        raise ValueError(f"Dataset 尚未冻结：{dataset_version} ({dataset.status})")
    if not dataset.manifest_uri or not dataset.class_map_uri:
        raise ValueError("Dataset Freeze 缺少 manifest_uri 或 class_map_uri")
    manifest_bytes, generation = _read_uri(dataset.manifest_uri, storage_client)
    class_map_bytes, class_generation = _read_uri(dataset.class_map_uri, storage_client)
    class_map = _parse_class_map(class_map_bytes)
    rows = list(csv.DictReader(io.StringIO(manifest_bytes.decode("utf-8-sig"))))
    if not rows:
        raise ValueError("Frozen Dataset manifest 为空")
    normalized: list[dict[str, Any]] = []
    errors: list[str] = []
    for row_number, raw in enumerate(rows, 2):
        row = {str(k): str(v or "").strip() for k, v in raw.items()}
        image_id = row.get("image_id")
        source_uri = row.get("gcs_uri") or row.get("source_image_gcs_uri")
        species_key = row.get("species_key")
        species_name = row.get("species") or row.get("species_name")
        split = row.get("split")
        try:
            class_index = int(row.get("class_index", ""))
        except ValueError:
            class_index = -1
        if not image_id or not source_uri or not species_key or not species_name or split not in {"train", "val", "test"}:
            errors.append(f"row {row_number}: missing image/source/species/split")
            continue
        if species_key not in class_map or class_map[species_key]["class_index"] != class_index:
            errors.append(f"row {row_number}: class_map mismatch for {species_key}")
            continue
        truth = row.get("truth_species") or species_name
        review_status = row.get("review_status") or ""
        if not truth or review_status.lower() not in {"approved", "accepted"}:
            errors.append(f"row {row_number}: ground truth is not confirmed")
            continue
        if verify_source_images and not _uri_exists(source_uri, storage_client):
            errors.append(f"row {row_number}: source image does not exist")
            continue
        normalized.append(
            {
                "source_dataset_version": dataset_version,
                "source_manifest_uri": dataset.manifest_uri,
                "image_id": image_id,
                "source_image_id": row.get("source_image_id") or image_id,
                "source_image_gcs_uri": source_uri,
                "species_key": species_key,
                "species_name": species_name,
                "class_index": class_index,
                "split": split,
                "group_id": row.get("group_id") or row.get("duplicate_group") or "",
                "batch_id": row.get("batch_id") or "",
                "review_status": "BBOX_REQUIRED",
            }
        )
    return {
        "dataset": dataset,
        "rows": normalized,
        "errors": errors,
        "manifest_uri": dataset.manifest_uri,
        "class_map_uri": dataset.class_map_uri,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest_generation": generation,
        "class_map_generation": class_generation,
        "class_map": class_map,
    }


def crop_readiness(db: Any, dataset_version: str, storage_client: Any = None) -> dict[str, Any]:
    try:
        loaded = load_frozen_dataset(db, dataset_version, storage_client, verify_source_images=True)
    except Exception as exc:
        return {
            "dataset_version": dataset_version,
            "source_type": "FROZEN_DATASET",
            "ground_truth_confirmed": False,
            "bbox_status": "REQUIRED",
            "crop_ready": False,
            "errors": [str(exc)],
        }
    from sqlalchemy import func, select
    from app.models import DatasetCropReview

    rows = loaded["rows"]
    reviewed = db.execute(
        select(DatasetCropReview.review_status, func.count())
        .where(DatasetCropReview.source_dataset_version == dataset_version)
        .group_by(DatasetCropReview.review_status)
    ).all()
    counts = {str(status): int(count) for status, count in reviewed}
    accepted = counts.get("ACCEPTED", 0) + counts.get("TRAINING_READY", 0)
    errors = list(loaded["errors"])
    crop_ready = bool(rows) and not errors and accepted == len(rows)
    splits = Counter(row["split"] for row in rows)
    return {
        "dataset_version": dataset_version,
        "source_type": "FROZEN_DATASET",
        "manifest_uri": loaded["manifest_uri"],
        "class_map_uri": loaded["class_map_uri"],
        "manifest_sha256": loaded["manifest_sha256"],
        "images": len(rows),
        "species_count": len({row["species_key"] for row in rows}),
        "split_counts": {name: splits.get(name, 0) for name in ("train", "val", "test")},
        "ground_truth_confirmed": not errors and bool(rows),
        "bbox_status": "ACCEPTED" if accepted == len(rows) and rows else "REQUIRED",
        "bbox_counts": {**counts, "accepted": accepted, "pending": max(len(rows) - accepted, 0)},
        "crop_ready": crop_ready,
        "errors": errors,
    }


__all__ = ["crop_readiness", "load_frozen_dataset"]
