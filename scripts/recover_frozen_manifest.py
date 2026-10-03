"""Recover one missing Frozen Dataset manifest by byte-identical GCS copy.

This operator is deliberately limited to the immutable-artifact recovery gate
for DS_CROP_M1_v0.2. It never regenerates Dataset data or writes to SQL.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from google.cloud import storage
from sqlalchemy import select

from app.db import SessionLocal
from app.dataset_models import DatasetItem
from app.models import DatasetVersion


VERSION = os.environ.get("DATASET_VERSION", "DS_CROP_M1_v0.2").strip()
BUCKET_NAME = os.environ["GCS_BUCKET"].strip()
PREFIX = f"datasets/{VERSION}/"
MISSING_NAME = f"{PREFIX}manifest.csv"
RECOVERY_OUTPUT_PREFIX = os.environ["RECOVERY_OUTPUT_PREFIX"].rstrip("/")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def canonical_hash(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=json_default)
    return sha256_bytes(body.encode("utf-8"))


def blob_metadata(blob: Any) -> dict[str, Any]:
    return {
        "name": blob.name,
        "generation": str(blob.generation or ""),
        "size": int(blob.size or 0),
        "updated": blob.updated.isoformat() if blob.updated else None,
        "md5_hash": blob.md5_hash,
        "crc32c": blob.crc32c,
    }


def inventory(bucket: Any) -> list[dict[str, Any]]:
    result = [blob_metadata(blob) for blob in bucket.list_blobs(prefix=PREFIX, versions=True)]
    result.sort(key=lambda item: (item["name"], item["generation"]))
    return result


def exact_blobs(bucket: Any, name: str) -> list[Any]:
    return [blob for blob in bucket.list_blobs(prefix=name, versions=True) if blob.name == name]


def current_blob(bucket: Any, name: str) -> Any | None:
    return bucket.get_blob(name)


def download(bucket: Any, name: str, generation: str | None = None) -> bytes:
    blob = bucket.get_blob(name, generation=int(generation)) if generation else bucket.get_blob(name)
    if blob is None:
        raise RuntimeError(f"missing object: gs://{BUCKET_NAME}/{name}")
    return blob.download_as_bytes(timeout=300)


def split_counts(data: bytes) -> tuple[list[str], list[dict[str, str]], Counter[str]]:
    text = data.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if not reader.fieldnames:
        raise RuntimeError("manifest has no header")
    required = {"dataset_version", "split"}
    missing = sorted(required - set(reader.fieldnames))
    if missing:
        raise RuntimeError(f"manifest missing columns: {','.join(missing)}")
    rows = list(reader)
    versions = {str(row.get("dataset_version") or "").strip() for row in rows}
    if versions != {VERSION}:
        raise RuntimeError(f"manifest dataset_version mismatch: {sorted(versions)}")
    counts = Counter(str(row.get("split") or "").strip() for row in rows)
    return list(reader.fieldnames), rows, counts


def dataset_snapshot(db: Any) -> dict[str, Any]:
    dataset = db.get(DatasetVersion, VERSION)
    if dataset is None:
        raise RuntimeError(f"DatasetVersion not found: {VERSION}")
    record = {
        "dataset_version": dataset.dataset_version,
        "status": dataset.status,
        "manifest_uri": dataset.manifest_uri,
        "class_map_uri": dataset.class_map_uri,
        "train_count": dataset.train_count,
        "val_count": dataset.val_count,
        "test_count": dataset.test_count,
        "species_count": dataset.species_count,
        "git_commit": dataset.git_commit,
        "selection_mode": dataset.selection_mode,
        "source_cutoff_at": dataset.source_cutoff_at,
        "pipeline_type": dataset.pipeline_type,
        "metadata_json": dataset.metadata_json,
    }
    items = db.scalars(
        select(DatasetItem)
        .where(DatasetItem.dataset_version == VERSION)
        .order_by(DatasetItem.id)
    ).all()
    item_rows = [
        {
            "id": item.id,
            "dataset_version": item.dataset_version,
            "image_asset_id": item.image_asset_id,
            "batch_id": item.batch_id,
            "image_id": item.image_id,
            "gcs_uri": item.gcs_uri,
            "species_key": item.species_key,
            "species_name": item.species_name,
            "class_index": item.class_index,
            "split": item.split,
        }
        for item in items
    ]
    return {
        "dataset_version": record,
        "dataset_item_count": len(item_rows),
        "dataset_item_sha256": canonical_hash(item_rows),
        "dataset_items": item_rows,
    }


def validate_manifest(data: bytes, snapshot: dict[str, Any], label: str) -> dict[str, Any]:
    header, rows, counts = split_counts(data)
    db_row = snapshot["dataset_version"]
    expected_counts = {
        "train": int(db_row["train_count"]),
        "val": int(db_row["val_count"]),
        "test": int(db_row["test_count"]),
    }
    actual_counts = {name: int(counts.get(name, 0)) for name in ("train", "val", "test")}
    if actual_counts != expected_counts:
        raise RuntimeError(f"{label} split counts mismatch: {actual_counts} != {expected_counts}")
    if len(rows) != sum(expected_counts.values()):
        raise RuntimeError(f"{label} row count mismatch: {len(rows)}")

    db_items = snapshot["dataset_items"]
    db_by_key = {(str(row["batch_id"]), str(row["image_id"])): row for row in db_items}
    manifest_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        batch_id = str(row.get("batch_id") or row.get("source_batch") or "").strip()
        image_id = str(row.get("image_id") or row.get("source_image_id") or "").strip()
        key = (batch_id, image_id)
        if not batch_id or not image_id or key in manifest_by_key:
            raise RuntimeError(f"{label} has invalid or duplicate lineage key: {key}")
        manifest_by_key[key] = row
    if set(manifest_by_key) != set(db_by_key):
        raise RuntimeError(f"{label} DatasetItem membership mismatch")
    for key, item in db_by_key.items():
        row = manifest_by_key[key]
        checks = {
            "gcs_uri": str(row.get("gcs_uri") or row.get("crop_image_path") or "").strip(),
            "species_key": str(row.get("species_key") or "").strip(),
            "species_name": str(row.get("species_name") or "").strip(),
            "class_index": int(row.get("class_index") or 0),
            "split": str(row.get("split") or "").strip(),
        }
        expected = {
            "gcs_uri": str(item["gcs_uri"]),
            "species_key": str(item["species_key"]),
            "species_name": str(item["species_name"]),
            "class_index": int(item["class_index"]),
            "split": str(item["split"]),
        }
        if checks != expected:
            raise RuntimeError(f"{label} DatasetItem row mismatch for {key}")
    return {
        "sha256": sha256_bytes(data),
        "size": len(data),
        "header": header,
        "row_count": len(rows),
        "split_counts": actual_counts,
        "dataset_version_validation": True,
        "dataset_item_validation": True,
    }


def upload_json(client: Any, value: dict[str, Any]) -> None:
    body = json.dumps(value, ensure_ascii=False, indent=2, default=json_default).encode("utf-8") + b"\n"
    prefix_body = RECOVERY_OUTPUT_PREFIX[5:]
    bucket_name, object_prefix = prefix_body.split("/", 1)
    object_name = f"{object_prefix.rstrip('/')}/{VERSION}_recovery.json"
    client.bucket(bucket_name).blob(object_name).upload_from_string(body, content_type="application/json")


def main() -> int:
    client = storage.Client(project=os.getenv("GCP_PROJECT_ID") or None)
    bucket = client.bucket(BUCKET_NAME)
    evidence: dict[str, Any] = {
        "dataset_version": VERSION,
        "missing_object": f"gs://{BUCKET_NAME}/{MISSING_NAME}",
        "recovery_method": "BYTE_IDENTICAL_GCS_COPY",
        "business_data_changed": False,
    }
    try:
        before_inventory = inventory(bucket)
        with SessionLocal() as db:
            before_db = dataset_snapshot(db)
        db_row = before_db["dataset_version"]
        if db_row["status"] != "FROZEN":
            raise RuntimeError(f"DatasetVersion status is not FROZEN: {db_row['status']}")
        if db_row["selection_mode"] != "ACCEPTED_POOL_BBOX_CROP":
            raise RuntimeError(f"unexpected selection_mode: {db_row['selection_mode']}")
        if not str(db_row["manifest_uri"]).endswith(f"datasets/{VERSION}/manifest.csv"):
            raise RuntimeError(f"unexpected manifest_uri: {db_row['manifest_uri']}")

        dataset_json = download(bucket, f"{PREFIX}dataset.json")
        class_map_json = download(bucket, f"{PREFIX}class_map.json")
        all_blob = current_blob(bucket, f"{PREFIX}manifest_all.csv")
        training_blob = current_blob(bucket, f"{PREFIX}training_manifest.csv")
        immutable_hashes_before = {
            "dataset.json": sha256_bytes(dataset_json),
            "class_map.json": sha256_bytes(class_map_json),
        }
        if all_blob is not None:
            immutable_hashes_before["manifest_all.csv"] = sha256_bytes(all_blob.download_as_bytes(timeout=300))
        if training_blob is not None:
            immutable_hashes_before["training_manifest.csv"] = sha256_bytes(
                training_blob.download_as_bytes(timeout=300)
            )
        dataset_doc = json.loads(dataset_json.decode("utf-8"))
        class_map_doc = json.loads(class_map_json.decode("utf-8"))
        if dataset_doc.get("dataset_version") != VERSION or dataset_doc.get("immutable") is not True:
            raise RuntimeError("dataset.json immutable identity gate failed")
        if not str(dataset_doc.get("manifest_uri") or "").endswith(f"datasets/{VERSION}/manifest.csv"):
            raise RuntimeError("dataset.json manifest_uri gate failed")
        if class_map_doc.get("dataset_version") != VERSION:
            raise RuntimeError("class_map.json dataset_version gate failed")

        if all_blob is None and training_blob is None:
            raise RuntimeError("no current authoritative sibling manifest exists")
        source_blob = all_blob or training_blob
        source_name = source_blob.name
        source_bytes = source_blob.download_as_bytes(timeout=300)
        source_validation = validate_manifest(source_bytes, before_db, source_name)
        training_validation = None
        if all_blob is not None and training_blob is not None:
            training_bytes = training_blob.download_as_bytes(timeout=300)
            training_validation = validate_manifest(training_bytes, before_db, training_blob.name)
            if source_bytes != training_bytes:
                raise RuntimeError("manifest_all.csv and training_manifest.csv are not byte-identical")

        source_metadata = blob_metadata(source_blob)
        destination = bucket.blob(MISSING_NAME)
        existing_destination = current_blob(bucket, MISSING_NAME)
        copy_performed = existing_destination is None
        if copy_performed:
            token = None
            while True:
                token, _, _ = destination.rewrite(source_blob, token=token, if_generation_match=0)
                if not token:
                    break
        else:
            existing_bytes = existing_destination.download_as_bytes(timeout=300)
            if existing_bytes != source_bytes:
                raise RuntimeError("existing destination manifest differs from authoritative source")
        restored_bytes = download(bucket, MISSING_NAME)
        destination_metadata = blob_metadata(bucket.get_blob(MISSING_NAME))
        if restored_bytes != source_bytes:
            raise RuntimeError("destination bytes differ from authoritative source")

        restored_validation = validate_manifest(restored_bytes, before_db, MISSING_NAME)
        after_inventory = inventory(bucket)
        expected_inventory = [item for item in before_inventory if item["name"] != MISSING_NAME]
        actual_inventory = [item for item in after_inventory if item["name"] != MISSING_NAME]
        if actual_inventory != expected_inventory:
            raise RuntimeError("an existing Frozen Dataset object changed during recovery")
        with SessionLocal() as db:
            after_db = dataset_snapshot(db)
        if canonical_hash(before_db) != canonical_hash(after_db):
            raise RuntimeError("DatasetVersion or DatasetItem lineage changed during recovery")
        for name, expected_hash in immutable_hashes_before.items():
            after_hash = sha256_bytes(download(bucket, f"{PREFIX}{name}"))
            if after_hash != expected_hash:
                raise RuntimeError(f"immutable object changed during recovery: {name}")

        evidence.update(
            {
                "manifest_recovery": "RECOVERED",
                "recovery_source": source_name.rsplit("/", 1)[-1],
                "source_generation": source_metadata["generation"],
                "source_size": source_metadata["size"],
                "source_sha256": source_validation["sha256"],
                "destination_generation": destination_metadata["generation"],
                "destination_sha256": sha256_bytes(restored_bytes),
                "manifest_all_sha256": immutable_hashes_before.get("manifest_all.csv"),
                "training_manifest_sha256": immutable_hashes_before.get("training_manifest.csv"),
                "dataset_json_sha256": immutable_hashes_before["dataset.json"],
                "class_map_sha256": immutable_hashes_before["class_map.json"],
                "db_train_count": int(db_row["train_count"]),
                "db_val_count": int(db_row["val_count"]),
                "db_test_count": int(db_row["test_count"]),
                "db_total_count": int(before_db["dataset_item_count"]),
                "csv_train_count": restored_validation["split_counts"]["train"],
                "csv_val_count": restored_validation["split_counts"]["val"],
                "csv_test_count": restored_validation["split_counts"]["test"],
                "row_count": restored_validation["row_count"],
                "dataset_version_validation": True,
                "immutable_validation": True,
                "dataset_item_validation": True,
                "inventory_before": before_inventory,
                "inventory_after": after_inventory,
                "dataset_version_before": before_db["dataset_version"],
                "dataset_version_after": after_db["dataset_version"],
                "dataset_item_sha256_before": before_db["dataset_item_sha256"],
                "dataset_item_sha256_after": after_db["dataset_item_sha256"],
                "only_allowed_object_created": copy_performed,
                "copy_performed_in_this_execution": copy_performed,
                "gcs_source_images_deleted": 0,
            }
        )
        upload_json(client, evidence)
        print(json.dumps({key: evidence[key] for key in ("manifest_recovery", "recovery_source", "source_sha256", "destination_sha256", "row_count")}, ensure_ascii=False))
        return 0
    except Exception as exc:
        evidence.update({"manifest_recovery": "NOT_RECOVERABLE", "error": f"{type(exc).__name__}: {exc}"})
        upload_json(client, evidence)
        print(json.dumps(evidence, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
