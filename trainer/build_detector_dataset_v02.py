"""Build the reviewed detector dataset version selected by DETECTOR_DATASET_VERSION.

This is deliberately separate from ``build_detector_dataset`` (the legacy
Google Vision pseudo-label builder) and ``build_reviewed_datasets`` (the crop
materializer). A frozen package is published only after its real source rows,
human review provenance, grouped split, image bytes, COCO labels, and all
training gates validate.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageOps, UnidentifiedImageError

from trainer.detector_dataset_gate import (
    DATASET_VERSION,
    _HELD_OUT_SHA256,
    validate_v02_dataset,
    validate_v02_materialized_dataset,
)

HUMAN_LABEL_SOURCE = "accepted_bbox_human_review"
GCS_PREFIX = f"detector-datasets/{DATASET_VERSION}"
SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}


def _json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    return value


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bbox_xywh(value: Any) -> list[float] | None:
    parsed = _json(value)
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 4:
        return None
    try:
        x, y, width, height = [float(item) for item in parsed]
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(0.0 <= item <= 1.0 for item in (x, y, width, height)):
        return None
    if width <= 0 or height <= 0 or x + width > 1.00001 or y + height > 1.00001:
        return None
    return [round(x, 8), round(y, 8), round(width, 8), round(height, 8)]


def _source_group(image: Any) -> tuple[str, str]:
    group = str(getattr(image, "group_id", None) or "").strip()
    if group:
        return f"group:{group}", "ImageAsset.group_id"
    # batch_id identifies an ingestion/import batch, not necessarily a capture
    # session. Grouping every asset in that batch can join hundreds of unrelated
    # scenes and strand most negatives in validation/test. Use a stable per-asset
    # fallback; exact source SHA and explicit group_id still join true duplicates.
    asset_id = str(getattr(image, "id", None) or "").strip()
    if not asset_id:
        raise ValueError("missing source group and stable ImageAsset.id")
    return f"asset:{asset_id}", "ImageAsset.id_fallback"


def collect_source_records(db) -> dict[str, Any]:
    """Read accepted bbox rows and active, explicitly confirmed DET_FISH pool rows.

    Crop paths and Google Vision boxes are intentionally never selected. A
    negative is admitted only through an active DET_FISH-specific pool member
    and its most recent ADD/RE_ADD audit event. The UI operation records the
    actor, time, source state, and explicit confirmation without changing the
    source presence/review rows.
    """

    from sqlalchemy import select

    from app.models import BatchCropReview, DetectorNegativePoolEntry, DetectorNegativePoolEvent, ImageAsset

    positive_query = (
        select(BatchCropReview, ImageAsset)
        .join(ImageAsset, ImageAsset.id == BatchCropReview.image_asset_id)
        .where(
            ImageAsset.review_status == "approved",
            ImageAsset.training_eligible.is_(True),
            BatchCropReview.status.in_({"ACCEPTED", "TRAINING_READY"}),
            BatchCropReview.accepted_bbox_json.is_not(None),
        )
        .order_by(BatchCropReview.id)
    )
    positives: list[dict[str, Any]] = []
    for review, image in db.execute(positive_query).all():
        group_id, group_source = _source_group(image)
        positives.append({
            "kind": "positive",
            "review_record_id": str(review.id),
            "batch_id": str(image.batch_id),
            "image_id": str(image.image_id),
            "source_uri": str(image.gcs_uri or "").strip(),
            "source_group_id": group_id,
            "source_group_basis": group_source,
            "bbox_xywh": _bbox_xywh(review.accepted_bbox_json),
            "raw_bbox_value": review.accepted_bbox_json,
            "review_status": str(review.status or "").upper(),
            "reviewer_id": str(review.reviewer or "").strip(),
            "reviewed_at": review.reviewed_at.isoformat() if review.reviewed_at else "",
            "training_eligibility_basis": (
                "database_training_eligible=true; ImageAsset.review_status=approved; "
                "BatchCropReview.status=accepted; accepted_bbox_json is present"
            ),
        })

    negatives: list[dict[str, Any]] = []
    negative_pool_errors: list[dict[str, str]] = []
    active_pool_rows = db.execute(
        select(DetectorNegativePoolEntry, ImageAsset)
        .join(ImageAsset, ImageAsset.id == DetectorNegativePoolEntry.image_asset_id)
        .where(DetectorNegativePoolEntry.state == "ACTIVE")
        .order_by(DetectorNegativePoolEntry.id)
    ).all()
    for pool_entry, image in active_pool_rows:
        if not image.training_eligible:
            negative_pool_errors.append({"record_id": str(pool_entry.id), "error": "active negative pool member is not training eligible"})
            continue
        if image.duplicate_of_image_asset_id:
            negative_pool_errors.append({"record_id": str(pool_entry.id), "error": "active negative pool member is a known duplicate"})
            continue
        if image.review_status == "approved" and str(image.truth_species or "").strip():
            negative_pool_errors.append({"record_id": str(pool_entry.id), "error": "negative pool member conflicts with approved positive truth"})
            continue
        positive_review = db.scalar(select(BatchCropReview.id).where(
            BatchCropReview.image_asset_id == image.id,
            BatchCropReview.status.in_({"ACCEPTED", "TRAINING_READY"}),
            BatchCropReview.accepted_bbox_json.is_not(None),
        ))
        if positive_review is not None:
            negative_pool_errors.append({"record_id": str(pool_entry.id), "error": "negative pool member conflicts with accepted positive bbox"})
            continue
        events = db.scalars(
            select(DetectorNegativePoolEvent)
            .where(DetectorNegativePoolEvent.pool_entry_id == pool_entry.id)
            .order_by(DetectorNegativePoolEvent.occurred_at.desc(), DetectorNegativePoolEvent.id.desc())
        ).all()
        event = events[0] if events else None
        snapshot = _json(event.source_snapshot_json) if event else {}
        if (
            event is None
            or event.action not in {"ADD", "RE_ADD"}
            or event.after_state != "ACTIVE"
            or not str(event.actor or "").strip()
            or not event.occurred_at
            or snapshot.get("det_fish_manual_confirmation") is not True
            or not _is_sha256(str(pool_entry.source_image_sha256 or ""))
            or str(pool_entry.source_uri or "") != str(image.gcs_uri or "")
        ):
            negative_pool_errors.append({"record_id": str(pool_entry.id), "error": "active pool member lacks latest valid manual ADD/RE_ADD provenance, source URI or source SHA"})
            continue
        group_id, group_source = _source_group(image)
        negatives.append({
            "kind": "negative",
            "review_record_id": f"det_fish_negative_pool_entry:{pool_entry.id}:event:{event.id}",
            "batch_id": str(image.batch_id),
            "image_id": str(image.image_id),
            "source_uri": str(image.gcs_uri or "").strip(),
            "source_group_id": group_id,
            "source_group_basis": group_source,
            "review_status": "NO_FISH_CONFIRMED",
            "reviewer_id": str(event.actor or "").strip(),
            "reviewed_at": event.occurred_at.isoformat() if event.occurred_at else "",
            "pool_entry_id": int(pool_entry.id),
            "pool_event_id": int(event.id),
            "pool_action": str(event.action),
            "pool_source_kind": str(pool_entry.source_kind),
            "pool_source_record_id": str(pool_entry.source_record_id),
            "pool_source_dataset_version": pool_entry.source_dataset_version,
            "pool_source_image_sha256": str(pool_entry.source_image_sha256),
            "pool_source_provider": pool_entry.source_provider,
            "pool_source_model_version": pool_entry.source_model_version,
            "pool_source_status": pool_entry.source_status,
            "pool_confirmation_snapshot": snapshot,
            "training_eligibility_basis": (
                "ImageAsset.training_eligible=true; active DET_FISH negative pool membership; "
                "latest ADD/RE_ADD audit event records explicit manual no-fish confirmation, actor and time"
            ),
        })

    return {
        "positives": positives,
        "negatives": negatives,
        "accepted_bbox_record_count": len(positives),
        "explicit_human_negative_record_count": len(negatives),
        "negative_pool_active_member_count": len(active_pool_rows),
        "negative_pool_member_errors": negative_pool_errors,
        "source_record_errors": negative_pool_errors,
    }


def read_accepted_pool_gcs(bucket_name: str, *, client=None) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, str]]:
    """Read the authoritative Accepted Pool CSV and metadata from GCS."""

    if client is None:
        from google.cloud import storage

        client = storage.Client()
    bucket = client.bucket(bucket_name)
    manifest_uri = f"gs://{bucket_name}/datasets/accepted_pool/manifest.csv"
    metadata_uri = f"gs://{bucket_name}/datasets/accepted_pool/metadata.json"
    manifest_bytes = bucket.blob("datasets/accepted_pool/manifest.csv").download_as_bytes(timeout=120)
    metadata_bytes = bucket.blob("datasets/accepted_pool/metadata.json").download_as_bytes(timeout=120)
    manifest_rows = list(csv.DictReader(io.StringIO(manifest_bytes.decode("utf-8-sig"))))
    metadata = json.loads(metadata_bytes.decode("utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Accepted Pool metadata.json must be a JSON object")
    return manifest_rows, metadata, {
        "manifest_uri": manifest_uri,
        "manifest_sha256": _sha256(manifest_bytes),
        "metadata_uri": metadata_uri,
        "metadata_sha256": _sha256(metadata_bytes),
    }


def reconcile_accepted_pool_export(
    pool_rows: list[dict[str, Any]],
    db_snapshot: dict[str, Any],
    source_files: dict[str, str],
) -> dict[str, Any]:
    """Reconcile each materialized Accepted Pool row with its DB review.

    Accepted Pool crop files are ignored. Only the source-scene URI and the
    exact accepted_bbox are carried forward; reviewer and review timestamp are
    reloaded from the authoritative review row.
    """

    db_by_id = {str(row.get("review_record_id")): row for row in db_snapshot.get("positives") or []}
    problems: list[str] = []
    source_image_audit_records = [
        {
            "review_record_id": str(row.get("source_review_id") or "").strip(),
            "source_uri": str(row.get("source_image_gcs_uri") or "").strip(),
            "kind": "positive",
        }
        for row in pool_rows
    ]
    source_image_audit_records.extend(
        {
            "review_record_id": str(row.get("review_record_id") or "").strip(),
            "source_uri": str(row.get("source_uri") or "").strip(),
            "kind": "negative",
        }
        for row in db_snapshot.get("negatives") or []
    )
    pool_by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids = 0
    nonaccepted = 0
    invalid_accepted_bbox = 0
    bbox_alias_conflicts = 0
    database_bbox_conflicts = 0
    uri_conflicts = 0
    detector_bbox_nonempty = 0
    pixel_bbox_invalid = 0
    crop_bbox_invalid = 0
    missing_source_uri = 0
    missing_source_claim = 0
    metadata_claimed_count = None
    for field in ("current_accepted_bbox_count", "source_count", "accepted_pool_count"):
        value = source_files.get(f"metadata_{field}")
        try:
            metadata_claimed_count = int(value)
        except (TypeError, ValueError):
            continue
        break
    if metadata_claimed_count is None:
        problems.append("Accepted Pool metadata has no recognized accepted record count")
    if metadata_claimed_count is not None and metadata_claimed_count != len(pool_rows):
        problems.append(
            f"Accepted Pool metadata declares {metadata_claimed_count} rows but CSV has {len(pool_rows)}"
        )
    for row in pool_rows:
        record_id = str(row.get("source_review_id") or "").strip()
        if not record_id or record_id in pool_by_id:
            duplicate_ids += 1
            continue
        pool_by_id[record_id] = row
        if str(row.get("review_status") or "").strip().upper() != "ACCEPTED":
            nonaccepted += 1
            continue
        if str(row.get("bbox_source") or "").strip().lower() != "accepted_bbox":
            problems.append(f"review_record_id={record_id}: bbox_source is not accepted_bbox")
            continue
        if not str(row.get("source_image_gcs_uri") or "").strip():
            missing_source_uri += 1
            continue
        if str(row.get("source_image_exists") or "").strip().lower() != "true":
            missing_source_claim += 1
        source_bbox = _bbox_xywh(row.get("accepted_bbox"))
        alias_bbox = _bbox_xywh(row.get("bbox"))
        if source_bbox is None:
            invalid_accepted_bbox += 1
            problems.append(f"review_record_id={record_id}: accepted_bbox invalid")
            continue
        if alias_bbox is None or alias_bbox != source_bbox:
            bbox_alias_conflicts += 1
            problems.append(f"review_record_id={record_id}: bbox alias differs from accepted_bbox")
            continue
        if str(row.get("detector_bbox") or "").strip():
            detector_bbox_nonempty += 1
        pixel_bbox = _json(row.get("pixel_bbox"))
        source_size = _json(row.get("source_size"))
        pixel_bbox_error = False
        if not isinstance(pixel_bbox, (list, tuple)) or len(pixel_bbox) != 4 or not isinstance(source_size, (list, tuple)) or len(source_size) != 2:
            pixel_bbox_invalid += 1
            pixel_bbox_error = True
        else:
            try:
                width, height = int(source_size[0]), int(source_size[1])
                left, top, right, bottom = (int(value) for value in pixel_bbox)
                x, y, box_width, box_height = source_bbox
                expected = (
                    round((x + box_width / 2) * width - box_width * width / 2),
                    round((y + box_height / 2) * height - box_height * height / 2),
                    round((x + box_width / 2) * width + box_width * width / 2),
                    round((y + box_height / 2) * height + box_height * height / 2),
                )
                # The accepted pool implementation uses the same center/round
                # rule to create this source-scene pixel box (crop_scale=1.0).
                if max(abs(a - b) for a, b in zip((left, top, right, bottom), expected)) > 1:
                    pixel_bbox_invalid += 1
                    pixel_bbox_error = True
            except (TypeError, ValueError, OverflowError):
                pixel_bbox_invalid += 1
                pixel_bbox_error = True
        crop_coords = [_json(row.get(key)) for key in ("crop_left", "crop_top", "crop_right", "crop_bottom")]
        try:
            if len(crop_coords) != 4 or any(value is None for value in crop_coords):
                raise ValueError
            crop_coords = [int(value) for value in crop_coords]
            pixel_coords = [int(value) for value in pixel_bbox]
            if crop_coords != pixel_coords:
                crop_bbox_invalid += 1
        except (TypeError, ValueError, OverflowError):
            crop_bbox_invalid += 1
        db_row = db_by_id.get(record_id)
        if db_row is None:
            problems.append(f"review_record_id={record_id}: missing database review row")
            continue
        db_bbox = _bbox_xywh(db_row.get("bbox_xywh"))
        if db_bbox != source_bbox:
            database_bbox_conflicts += 1
            problems.append(f"review_record_id={record_id}: GCS accepted_bbox differs from database review")
            continue
        if str(row.get("source_image_gcs_uri") or "").strip() != str(db_row.get("source_uri") or "").strip():
            uri_conflicts += 1
            problems.append(f"review_record_id={record_id}: source image URI differs from database")
            continue
        enriched = dict(db_row)
        enriched["pool_manifest_row"] = {
            field: row.get(field)
            for field in (
                "pool_key", "batch_id", "image_id", "source_review_id", "source_image_gcs_uri",
                "source_image_exists", "accepted_bbox", "bbox", "detector_bbox", "pixel_bbox",
                "source_size", "crop_left", "crop_top", "crop_right", "crop_bottom", "review_status",
                "bbox_source", "split",
            )
        }
        enriched["pool_bbox_coordinate_audit"] = {
            "accepted_bbox_and_bbox": "same normalized xywh original-scene coordinates",
            "pixel_bbox_and_source_size": "source-scene pixel xyxy and EXIF-transposed source dimensions; checked within 1 pixel",
            "crop_left_top_right_bottom": "source-scene crop rectangle used to materialize the classifier crop; not detector labels",
            "detector_bbox": "legacy detector candidate field; ignored as a label",
        }
        enriched["bbox_coordinate_audit_ok"] = not pixel_bbox_error
        # Carry fields used for actual-byte verification against the pool row.
        enriched["expected_source_size"] = source_size
        enriched["expected_pixel_bbox"] = pixel_bbox
        enriched["pool_crop_coordinates"] = [row.get(key) for key in ("crop_left", "crop_top", "crop_right", "crop_bottom")]
        enriched["detector_bbox_was_present"] = bool(str(row.get("detector_bbox") or "").strip())
        yield_record = enriched
        db_by_id[record_id] = yield_record

    missing_db_records = sorted(set(db_by_id) - set(pool_by_id))
    extra_pool_records = sorted(set(pool_by_id) - set({str(row.get("review_record_id")) for row in db_snapshot.get("positives") or []}))
    if duplicate_ids:
        problems.append(f"Accepted Pool manifest has {duplicate_ids} duplicate/missing source_review_id rows")
    if nonaccepted:
        problems.append(f"Accepted Pool manifest has {nonaccepted} rows outside ACCEPTED")
    if missing_source_uri:
        problems.append(f"Accepted Pool manifest has {missing_source_uri} rows without source_image_gcs_uri")
    if missing_source_claim:
        problems.append(f"Accepted Pool manifest has {missing_source_claim} rows without source_image_exists=true")
    bbox_conflicts = bbox_alias_conflicts + database_bbox_conflicts
    if bbox_conflicts or uri_conflicts:
        problems.append(f"Accepted Pool/DB conflict counts: bbox={bbox_conflicts}, source_uri={uri_conflicts}")
    if invalid_accepted_bbox:
        problems.append(f"Accepted Pool has {invalid_accepted_bbox} invalid accepted_bbox records")
    if pixel_bbox_invalid:
        problems.append(f"Accepted Pool has {pixel_bbox_invalid} pixel_bbox/source_size discrepancies")
    if crop_bbox_invalid:
        problems.append(f"Accepted Pool has {crop_bbox_invalid} source-scene crop rectangle mismatches")
    if missing_db_records:
        problems.append(f"Accepted Pool/DB reconciliation missing {len(missing_db_records)} DB source records")
    if extra_pool_records:
        problems.append(f"Accepted Pool/DB reconciliation has {len(extra_pool_records)} unmatched source records")

    positives = [
        db_by_id[record_id]
        for record_id in sorted(pool_by_id)
        if record_id in db_by_id and db_by_id[record_id].get("pool_manifest_row")
    ]
    return {
        "positives": positives,
        "negatives": list(db_snapshot.get("negatives") or []),
        "source_image_audit_records": source_image_audit_records,
        "accepted_bbox_record_count": len(pool_rows),
        "explicit_human_negative_record_count": int(db_snapshot.get("explicit_human_negative_record_count") or 0),
        "source_reconciliation": {
            "status": "PASS" if not problems else "BLOCKED",
            "problems": problems,
            "accepted_pool_rows": len(pool_rows),
            "metadata_claimed_record_count": metadata_claimed_count,
            "unique_source_review_ids": len(pool_by_id),
            "matched_database_review_rows": len(positives),
            "invalid_accepted_bbox_count": invalid_accepted_bbox,
            "bbox_alias_conflicts": bbox_alias_conflicts,
            "database_bbox_conflicts": database_bbox_conflicts,
            "bbox_conflicts": bbox_conflicts,
            "source_uri_conflicts": uri_conflicts,
            "pixel_bbox_or_source_size_errors": pixel_bbox_invalid,
            "crop_rectangle_errors": crop_bbox_invalid,
            "detector_bbox_nonempty_ignored_count": detector_bbox_nonempty,
            "source_image_exists_claim_false_count": missing_source_claim,
            "bbox_field_semantics": {
                "accepted_bbox": "human accepted normalized xywh on complete source scene after EXIF transpose; sole training label",
                "bbox": "legacy alias; must equal accepted_bbox",
                "detector_bbox": "legacy detector candidate; audited and ignored as a training label",
                "pixel_bbox": "source-scene pixel xyxy derived from accepted_bbox and source_size; compared within 1 px",
                "source_size": "EXIF-transposed complete source-scene width and height; verified against downloaded bytes",
                "crop_left_top_right_bottom": "source-scene crop rectangle for classifier crop only; verified against pixel_bbox and never used as detector input",
            },
            "manifest_sha256": source_files.get("manifest_sha256"),
            "manifest_uri": source_files.get("manifest_uri"),
            "metadata_sha256": source_files.get("metadata_sha256"),
            "metadata_uri": source_files.get("metadata_uri"),
        },
    }


def _read_source(uri: str, *, storage_client=None) -> bytes:
    value = str(uri or "").strip()
    if value.startswith("gs://"):
        if storage_client is None:
            from google.cloud import storage

            storage_client = storage.Client()
        bucket_name, object_name = value[5:].split("/", 1)
        return storage_client.bucket(bucket_name).blob(object_name).download_as_bytes(timeout=120)
    if value.startswith("file://"):
        return Path(value[7:]).read_bytes()
    path = Path(value)
    if path.is_file():
        return path.read_bytes()
    raise ValueError("source_uri must be a readable local path or gs:// URI")


def _audit_source_objects(
    records: dict[str, Any],
    fetch_source: Callable[[str], bytes],
    staging_dir: Path,
) -> dict[str, Any]:
    """Download and decode every unique source URI before considering labels.

    The Accepted Pool manifest is the inventory for this pass, so rows that
    later fail review/BBox reconciliation still receive a real object read,
    SHA-256, and EXIF-normalized dimension audit.
    """

    supplied_rows = records.get("source_image_audit_records")
    if not isinstance(supplied_rows, list):
        supplied_rows = [
            {
                "review_record_id": row.get("review_record_id"),
                "source_uri": row.get("source_uri"),
                "kind": row.get("kind"),
            }
            for row in (list(records.get("positives") or []) + list(records.get("negatives") or []))
        ]
    staging_dir.mkdir(parents=True, exist_ok=True)
    by_uri: dict[str, dict[str, Any]] = {}
    by_sha: dict[str, dict[str, Any]] = {}
    audit_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for row in supplied_rows:
        record_id = str(row.get("review_record_id") or "").strip()
        uri = str(row.get("source_uri") or row.get("source_image_gcs_uri") or "").strip()
        if not uri:
            errors.append({"record_id": record_id or "unknown", "error": "missing source image URI for source-byte audit"})
            audit_rows.append({"review_record_id": record_id, "source_uri": "", "status": "BLOCKED_MISSING_URI"})
            continue
        if uri not in by_uri:
            try:
                raw = fetch_source(uri)
                source_sha = _sha256(raw)
                metadata = by_sha.get(source_sha)
                if metadata is None:
                    with Image.open(BytesIO(raw)) as source:
                        orientation = int(source.getexif().get(274, 1) or 1)
                        image = ImageOps.exif_transpose(source).convert("RGB")
                        width, height = image.size
                        metadata = {
                            "source_image_sha256": source_sha,
                            "width": width,
                            "height": height,
                            "exif_orientation": orientation,
                            "staging_path": "",
                            "materialized_image_sha256": "",
                            "file_extension": "png" if str(source.format or "").upper() == "PNG" else "jpg",
                        }
                        if source_sha not in _HELD_OUT_SHA256:
                            image_path = staging_dir / f"{source_sha}.{metadata['file_extension']}"
                            if metadata["file_extension"] == "png":
                                image.save(image_path, format="PNG", optimize=False, compress_level=9)
                            else:
                                image.save(image_path, format="JPEG", quality=95, optimize=False)
                            metadata["staging_path"] = str(image_path)
                            metadata["materialized_image_sha256"] = _sha256_path(image_path)
                    by_sha[source_sha] = metadata
                by_uri[uri] = {**metadata, "status": "READ_OK"}
            except Exception as exc:
                by_uri[uri] = {
                    "status": "READ_FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        source = by_uri[uri]
        row_result = {
            "review_record_id": record_id,
            "kind": str(row.get("kind") or ""),
            "source_uri": uri,
            "status": source.get("status"),
            "source_image_sha256": source.get("source_image_sha256"),
            "width": source.get("width"),
            "height": source.get("height"),
            "exif_orientation": source.get("exif_orientation"),
            "file_extension": source.get("file_extension"),
        }
        audit_rows.append(row_result)
        if source.get("status") != "READ_OK":
            errors.append({"record_id": record_id or uri, "error": str(source.get("error") or source["status"])})

    successful_uris = sum(1 for value in by_uri.values() if value.get("status") == "READ_OK")
    failed_uris = len(by_uri) - successful_uris
    missing_uri_rows = sum(1 for value in audit_rows if value.get("status") == "BLOCKED_MISSING_URI")
    return {
        "by_uri": by_uri,
        "rows": audit_rows,
        "errors": errors,
        "summary": {
            "unique_source_uris_attempted": len(by_uri),
            "unique_source_uris_read_and_decoded": successful_uris,
            "unique_source_uris_failed": failed_uris + missing_uri_rows,
            "source_record_rows": len(audit_rows),
            "source_records_missing_uri": missing_uri_rows,
            "unique_original_sha256": len(by_sha),
            "unique_exif_normalized_images": len(by_sha),
        },
    }


def _stable_split(
    groups: dict[str, set[str]],
    fish_presence_by_sha: dict[str, bool],
) -> dict[str, str]:
    """Assign connected source-group/hash components with deterministic class balance."""

    target_ratios = SPLIT_RATIOS
    split_order = {name: index for index, name in enumerate(target_ratios)}
    class_totals = {
        fish_present: sum(1 for value in fish_presence_by_sha.values() if value is fish_present)
        for fish_present in (True, False)
    }
    target = {
        split: {
            fish_present: class_totals[fish_present] * ratio
            for fish_present in (True, False)
        }
        for split, ratio in target_ratios.items()
    }
    current = {
        split: {True: 0, False: 0}
        for split in target_ratios
    }
    assignments: dict[str, str] = {}
    components = sorted(
        groups,
        key=lambda key: (-len(groups[key]), hashlib.sha256(f"{DATASET_VERSION}:{key}".encode()).hexdigest()),
    )

    for component in components:
        missing_labels = set(groups[component]) - set(fish_presence_by_sha)
        if missing_labels:
            raise ValueError(f"missing class labels for source component {component}")
        component_counts = {
            fish_present: sum(
                1 for image_sha in groups[component]
                if fish_presence_by_sha[image_sha] is fish_present
            )
            for fish_present in (True, False)
        }

        def candidate_score(candidate: str) -> float:
            score = 0.0
            for split in target_ratios:
                for fish_present in (True, False):
                    count = current[split][fish_present]
                    if split == candidate:
                        count += component_counts[fish_present]
                    expected = target[split][fish_present]
                    score += ((count - expected) / max(expected, 1.0)) ** 2
            return score

        split = min(
            target_ratios,
            key=lambda name: (candidate_score(name), split_order[name]),
        )
        assignments[component] = split
        for fish_present in (True, False):
            current[split][fish_present] += component_counts[fish_present]
    return assignments


def _group_components(images: dict[str, dict[str, Any]]) -> dict[str, str]:
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        if parent[item] != item:
            parent[item] = find(parent[item])
        return parent[item]

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    image_nodes: dict[str, list[str]] = defaultdict(list)
    group_nodes: dict[str, list[str]] = defaultdict(list)
    for image_sha, item in images.items():
        sha_node = f"sha:{image_sha}"
        find(sha_node)
        for group_id in item["source_group_ids"]:
            group_node = f"group:{group_id}"
            union(sha_node, group_node)
            image_nodes[find(sha_node)].append(image_sha)
            group_nodes[find(sha_node)].append(group_id)

    # Rebuild components after all unions have settled.
    components: dict[str, set[str]] = defaultdict(set)
    for image_sha, item in images.items():
        root = find(f"sha:{image_sha}")
        components[root].add(image_sha)
        for group_id in item["source_group_ids"]:
            components[root].add(f"@group:{group_id}")
    stable_ids: dict[str, str] = {}
    for root, nodes in components.items():
        stable_ids[root] = "source-component:" + hashlib.sha256("\n".join(sorted(nodes)).encode()).hexdigest()
    return {image_sha: stable_ids[find(f"sha:{image_sha}")] for image_sha in images}


def _deterministic_archive(package_root: Path, archive_path: Path) -> str:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with archive_path.open("wb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                for path in sorted(package_root.rglob("*")):
                    if not path.is_file():
                        continue
                    relative = Path(DATASET_VERSION) / path.relative_to(package_root)
                    info = archive.gettarinfo(str(path), arcname=relative.as_posix())
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    info.mode = 0o644
                    with path.open("rb") as handle:
                        archive.addfile(info, handle)
    return _sha256_path(archive_path)


def _materialize(
    records: dict[str, Any],
    package_root: Path,
    source_object_audit: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_records = list(records.get("positives") or [])
    negatives = list(records.get("negatives") or [])
    errors: list[dict[str, str]] = list(records.get("source_record_errors") or [])
    errors.extend(source_object_audit.get("errors") or [])
    held_out_positive_ids: list[str] = []
    held_out_negative_ids: list[str] = []
    images: dict[str, dict[str, Any]] = {}
    positive_record_ids: set[str] = set()
    source_by_uri = source_object_audit.get("by_uri") or {}
    source_uris_attempted = set(source_by_uri)
    source_uris_read = {
        uri for uri, value in source_by_uri.items() if value.get("status") == "READ_OK"
    }
    source_uri_failures = {
        uri: str(value.get("error") or "source read failed")
        for uri, value in source_by_uri.items()
        if value.get("status") != "READ_OK"
    }
    source_hashes_read = {
        str(value.get("source_image_sha256"))
        for value in source_by_uri.values()
        if value.get("status") == "READ_OK" and value.get("source_image_sha256")
    }
    source_image_metadata = {
        str(value["source_image_sha256"]): {
            "orientation": int(value["exif_orientation"]),
            "width": int(value["width"]),
            "height": int(value["height"]),
            "file_extension": str(value.get("file_extension") or "png"),
        }
        for value in source_by_uri.values()
        if value.get("status") == "READ_OK" and value.get("source_image_sha256")
    }
    source_staging_paths = {
        str(value["source_image_sha256"]): Path(value["staging_path"])
        for value in source_by_uri.values()
        if value.get("status") == "READ_OK" and value.get("source_image_sha256") and value.get("staging_path")
    }
    materialized_staging = package_root / ".source-images"
    materialized_staging.mkdir(parents=True, exist_ok=True)

    for record in source_records:
        record_id = str(record.get("review_record_id") or "").strip()
        try:
            if not record_id:
                raise ValueError("missing accepted_bbox review record id")
            if record_id in positive_record_ids:
                raise ValueError("duplicate accepted_bbox review record id")
            positive_record_ids.add(record_id)
            if str(record.get("review_status") or "").upper() not in {"ACCEPTED", "TRAINING_READY"}:
                raise ValueError("review status is not accepted")
            if not str(record.get("reviewer_id") or "").strip() or not str(record.get("reviewed_at") or "").strip():
                raise ValueError("missing human reviewer or review timestamp")
            if not str(record.get("training_eligibility_basis") or "").strip():
                raise ValueError("missing training eligibility evidence")
            if not str(record.get("source_uri") or "").strip():
                raise ValueError("missing full-scene source URI")
            bbox = _bbox_xywh(record.get("bbox_xywh"))
            if bbox is None:
                raise ValueError("accepted_bbox is invalid or unavailable")
            source_uri = str(record["source_uri"])
            source_audit = source_by_uri.get(source_uri)
            if not isinstance(source_audit, dict) or source_audit.get("status") != "READ_OK":
                raise ValueError(f"source image was not successfully downloaded and decoded: {source_uri}")
            source_sha = str(source_audit.get("source_image_sha256") or "")
            if not _is_sha256(source_sha):
                raise ValueError("source image audit has no valid SHA-256")
            metadata = source_image_metadata.get(source_sha)
            if metadata is None:
                raise ValueError("source image metadata missing from complete source-object audit")
            width, height = metadata["width"], metadata["height"]
            expected_size = record.get("expected_source_size")
            if isinstance(expected_size, (list, tuple)) and len(expected_size) == 2:
                try:
                    if (int(expected_size[0]), int(expected_size[1])) != (width, height):
                        errors.append({"record_id": record_id, "error": "actual EXIF-normalized dimensions differ from Accepted Pool source_size"})
                except (TypeError, ValueError):
                    errors.append({"record_id": record_id, "error": "Accepted Pool source_size is invalid"})
            expected_pixel_box = record.get("expected_pixel_bbox")
            if isinstance(expected_pixel_box, (list, tuple)) and len(expected_pixel_box) == 4:
                x, y, box_width, box_height = bbox
                calculated = [
                    round((x + box_width / 2) * width - box_width * width / 2),
                    round((y + box_height / 2) * height - box_height * height / 2),
                    round((x + box_width / 2) * width + box_width * width / 2),
                    round((y + box_height / 2) * height + box_height * height / 2),
                ]
                try:
                    stored = [int(value) for value in expected_pixel_box]
                    if max(abs(a - b) for a, b in zip(calculated, stored)) > 1:
                        errors.append({"record_id": record_id, "error": "actual dimensions and accepted_bbox do not reproduce pixel_bbox"})
                except (TypeError, ValueError):
                    errors.append({"record_id": record_id, "error": "Accepted Pool pixel_bbox is invalid"})
            if source_sha in _HELD_OUT_SHA256:
                held_out_positive_ids.append(record_id)
                continue
            item = images.get(source_sha)
            if item is None:
                if source_sha not in source_staging_paths:
                    raise ValueError("source image has no normalized full-scene materialization")
                image_path = source_staging_paths[source_sha]
                item = {
                    "source_sha256": source_sha,
                    "materialized_path": image_path,
                    "materialized_sha256": _sha256_path(image_path),
                    "width": width,
                    "height": height,
                    "orientation": metadata["orientation"],
                    "file_extension": metadata["file_extension"],
                    "source_group_ids": set(),
                    "source_group_bases": set(),
                    "source_uris": set(),
                    "review_records": [],
                    "box_map": {},
                    "negative_record": None,
                }
                images[source_sha] = item
            group_id = str(record.get("source_group_id") or "").strip()
            if not group_id:
                raise ValueError("missing source group id")
            item["source_group_ids"].add(group_id)
            item["source_group_bases"].add(str(record.get("source_group_basis") or "unknown"))
            item["source_uris"].add(str(record["source_uri"]))
            x, y, box_width, box_height = bbox
            xyxy = [round(x, 8), round(y, 8), round(x + box_width, 8), round(y + box_height, 8)]
            box_key = json.dumps(xyxy, separators=(",", ":"))
            box = item["box_map"].setdefault(box_key, {"bbox_xyxy": xyxy, "review_record_ids": [], "review_provenance": []})
            box["review_record_ids"].append(record_id)
            provenance = {
                "review_record_id": record_id,
                "reviewer_id": str(record["reviewer_id"]).strip(),
                "reviewed_at": str(record["reviewed_at"]).strip(),
                "review_status": str(record["review_status"]).upper(),
                "accepted_bbox_xywh": bbox,
            }
            box["review_provenance"].append(provenance)
            item["review_records"].append(provenance)
        except (KeyError, OSError, TypeError, ValueError, UnidentifiedImageError) as exc:
            errors.append({"record_id": record_id or "unknown", "error": str(exc)})

    # Active DET_FISH pool members must carry a matching current source SHA and
    # append-only manual confirmation event. Machine labels alone never pass.
    negative_record_ids: set[str] = set()
    for record in negatives:
        record_id = str(record.get("review_record_id") or "").strip()
        try:
            if not record_id or record_id in negative_record_ids:
                raise ValueError("missing or duplicate human no-fish review event id")
            negative_record_ids.add(record_id)
            if str(record.get("review_status") or "").upper() != "NO_FISH_CONFIRMED":
                raise ValueError("negative row is not explicitly human-confirmed")
            if not str(record.get("reviewer_id") or "").strip() or not str(record.get("reviewed_at") or "").strip():
                raise ValueError("missing human no-fish reviewer or timestamp")
            if not str(record.get("training_eligibility_basis") or "").strip():
                raise ValueError("missing training eligibility evidence")
            source_uri = str(record.get("source_uri") or "")
            source_audit = source_by_uri.get(source_uri)
            if not isinstance(source_audit, dict) or source_audit.get("status") != "READ_OK":
                raise ValueError(f"source image was not successfully downloaded and decoded: {source_uri}")
            source_sha = str(source_audit.get("source_image_sha256") or "")
            if not _is_sha256(source_sha):
                raise ValueError("source image audit has no valid SHA-256")
            expected_pool_sha = str(record.get("pool_source_image_sha256") or "")
            if not _is_sha256(expected_pool_sha) or source_sha != expected_pool_sha:
                raise ValueError("negative pool source SHA does not match the actual original scene bytes")
            metadata = source_image_metadata.get(source_sha)
            if metadata is None:
                raise ValueError("source image metadata missing from complete source-object audit")
            if source_sha in _HELD_OUT_SHA256:
                held_out_negative_ids.append(record_id)
                continue
            item = images.get(source_sha)
            if item is None:
                image_path = source_staging_paths[source_sha]
                item = {
                    "source_sha256": source_sha,
                    "materialized_path": image_path,
                    "materialized_sha256": _sha256_path(image_path),
                    "width": metadata["width"],
                    "height": metadata["height"],
                    "orientation": metadata["orientation"],
                    "file_extension": metadata["file_extension"],
                    "source_group_ids": set(),
                    "source_group_bases": set(),
                    "source_uris": set(),
                    "review_records": [],
                    "box_map": {},
                    "negative_record": None,
                }
                images[source_sha] = item
            if item["box_map"]:
                raise ValueError("the same source image has both positive and human no-fish labels")
            group_id = str(record.get("source_group_id") or "").strip()
            if item["negative_record"] or item["box_map"]:
                raise ValueError("duplicate or conflicting label for the same source image")
            item["source_group_ids"].add(group_id)
            item["source_group_bases"].add(str(record.get("source_group_basis") or "unknown"))
            item["source_uris"].add(str(record.get("source_uri") or ""))
            item["negative_record"] = {
                "review_record_id": record_id,
                "reviewer_id": str(record["reviewer_id"]).strip(),
                "reviewed_at": str(record["reviewed_at"]).strip(),
                "review_status": "NO_FISH_CONFIRMED",
                "pool_entry_id": int(record["pool_entry_id"]),
                "pool_event_id": int(record["pool_event_id"]),
                "pool_action": str(record.get("pool_action") or ""),
                "pool_source_kind": str(record.get("pool_source_kind") or ""),
                "pool_source_record_id": str(record.get("pool_source_record_id") or ""),
                "pool_source_dataset_version": record.get("pool_source_dataset_version"),
                "pool_source_image_sha256": expected_pool_sha,
                "pool_source_provider": record.get("pool_source_provider"),
                "pool_source_model_version": record.get("pool_source_model_version"),
                "pool_source_status": record.get("pool_source_status"),
            }
        except (KeyError, OSError, TypeError, ValueError, UnidentifiedImageError) as exc:
            errors.append({"record_id": record_id or "unknown", "error": str(exc)})

    if not images:
        return [], {
            "source_accepted_bbox_record_count": len(source_records),
            "source_human_negative_record_count": len(negatives),
            "negative_pool_active_member_count": int(records.get("negative_pool_active_member_count") or len(negatives)),
            "negative_pool_member_errors": list(records.get("negative_pool_member_errors") or []),
            "excluded_held_out_accepted_bbox_record_count": len(held_out_positive_ids),
            "excluded_held_out_accepted_bbox_record_ids": sorted(held_out_positive_ids),
            "excluded_held_out_negative_record_count": len(held_out_negative_ids),
            "source_record_errors": errors,
            "source_reconciliation": records.get("source_reconciliation"),
            "accepted_pool_metadata_summary": records.get("accepted_pool_metadata_summary"),
            "source_uris_attempted": len(source_uris_attempted),
            "source_uris_read": len(source_uris_read),
            "source_uri_failures": len(source_uri_failures),
            "unique_source_sha256_read": len(source_hashes_read),
            "source_image_audit_summary": source_object_audit.get("summary") or {},
            "source_image_audit_rows": source_object_audit.get("rows") or [],
        }

    # Union exact source hashes and source groups before choosing splits.
    component_by_sha = _group_components(images)
    component_images: dict[str, set[str]] = defaultdict(set)
    for image_sha, component in component_by_sha.items():
        component_images[component].add(image_sha)
    fish_presence_by_sha = {image_sha: bool(item["box_map"]) for image_sha, item in images.items()}
    split_by_component = _stable_split(component_images, fish_presence_by_sha)

    manifest_rows: list[dict[str, Any]] = []
    split_counts = {split: {"images": 0, "annotations": 0, "negative_images": 0} for split in SPLIT_RATIOS}
    coco = {
        split: {"images": [], "annotations": [], "categories": [{"id": 1, "name": "fish"}]}
        for split in SPLIT_RATIOS
    }
    for image_sha in sorted(images):
        item = images[image_sha]
        split = split_by_component[component_by_sha[image_sha]]
        file_name = f"{image_sha}.{item['file_extension']}"
        image_path = package_root / f"{split}2017" / file_name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        # All staged and final files live under the same temporary directory.
        # Move the normalized scene rather than keeping two full copies in the
        # Cloud Run task's writable filesystem while materializing the split.
        Path(item["materialized_path"]).replace(image_path)
        positive_boxes = sorted(item["box_map"].values(), key=lambda box: box["bbox_xyxy"])
        fish_present = bool(positive_boxes)
        negative = item["negative_record"]
        if not fish_present and not negative:
            errors.append({"record_id": image_sha, "error": "source image has neither an accepted bbox nor an audited no-fish review"})
            continue
        if fish_present and negative:
            errors.append({"record_id": image_sha, "error": "positive and negative labels conflict for source image"})
            continue
        provenance = [record for box in positive_boxes for record in box["review_provenance"]] if fish_present else [negative]
        reviewers = sorted({str(record["reviewer_id"]) for record in provenance})
        dates = sorted({str(record["reviewed_at"]) for record in provenance})
        record_ids = sorted({str(record["review_record_id"]) for record in provenance})
        boxes = [
            {
                "bbox_xyxy": box["bbox_xyxy"],
                "review_record_ids": sorted(box["review_record_ids"]),
                "review_provenance": sorted(box["review_provenance"], key=lambda value: value["review_record_id"]),
                "label_source": HUMAN_LABEL_SOURCE,
            }
            for box in positive_boxes
        ]
        source_uris = sorted(uri for uri in item["source_uris"] if uri)
        row = {
            "image_id": image_sha[:24],
            "image_sha256": image_sha,
            "source_image_sha256": image_sha,
            "materialized_image_sha256": item["materialized_sha256"],
            "source_group_id": component_by_sha[image_sha],
            "source_group_ids": sorted(item["source_group_ids"]),
            "source_group_basis": sorted(item["source_group_bases"]),
            "split": split,
            "file_name": file_name,
            "image_kind": "complete_scene",
            "width": item["width"],
            "height": item["height"],
            "exif_orientation": item["orientation"],
            "orientation_policy": "Pillow ImageOps.exif_transpose; boxes use EXIF-normalized source coordinates",
            "reviewer_id": ";".join(reviewers),
            "reviewer_ids": reviewers,
            "reviewed_at": ";".join(dates),
            "review_record_ids": record_ids,
            "review_provenance": provenance,
            "source_uri": source_uris[0] if source_uris else "",
            "source_uris": source_uris,
            "training_eligibility_basis": ";".join(sorted({str(record.get("training_eligibility_basis") or "") for record in (source_records + negatives) if str(record.get("source_uri") or "") in item["source_uris"] and str(record.get("training_eligibility_basis") or "")})),
            "review_status": "ACCEPTED" if fish_present else "NO_FISH_CONFIRMED",
            "fish_present": fish_present,
            "label_source": HUMAN_LABEL_SOURCE if fish_present else "det_fish_negative_pool_human_confirmation",
            "bbox_coordinate_space": "exif_transposed_normalized_xywh",
            "boxes": boxes,
        }
        manifest_rows.append(row)
        split_counts[split]["images"] += 1
        split_counts[split]["annotations"] += len(boxes)
        split_counts[split]["negative_images"] += int(not fish_present)
        split_index = len(coco[split]["images"]) + 1
        coco[split]["images"].append({"id": split_index, "file_name": file_name, "width": item["width"], "height": item["height"]})
        for box in boxes:
            x1, y1, x2, y2 = box["bbox_xyxy"]
            left, top = x1 * item["width"], y1 * item["height"]
            width, height = (x2 - x1) * item["width"], (y2 - y1) * item["height"]
            coco[split]["annotations"].append({
                "id": sum(len(value["annotations"]) for value in coco.values()) + 1,
                "image_id": split_index,
                "category_id": 1,
                "bbox": [left, top, width, height],
                "area": width * height,
                "iscrowd": 0,
                "accepted_bbox_review_record_ids": box["review_record_ids"],
            })

    shutil.rmtree(materialized_staging, ignore_errors=True)
    annotations_dir = package_root / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    for split, document in coco.items():
        (annotations_dir / f"instances_{split}2017.json").write_text(
            json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
    manifest_rows.sort(key=lambda row: row["image_sha256"])
    manifest_bytes = json.dumps(manifest_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    (package_root / "bootstrap_manifest.json").write_bytes(manifest_bytes)
    actual_manifest_ids = {
        record_id
        for row in manifest_rows
        if row["fish_present"]
        for record_id in row["review_record_ids"]
    }
    held_ids = set(held_out_positive_ids)
    source_id_set = positive_record_ids
    unaccounted_ids = source_id_set - actual_manifest_ids - held_ids
    if unaccounted_ids:
        errors.extend({"record_id": record_id, "error": "accepted review record did not materialize"} for record_id in sorted(unaccounted_ids))

    return manifest_rows, {
        "manifest_bytes": manifest_bytes,
        "coco": coco,
        "splits": split_counts,
        "source_accepted_bbox_record_count": len(source_records),
        "source_human_negative_record_count": len(negatives),
        "negative_pool_active_member_count": int(records.get("negative_pool_active_member_count") or len(negatives)),
        "negative_pool_member_errors": list(records.get("negative_pool_member_errors") or []),
        "manifested_accepted_bbox_record_count": len(actual_manifest_ids),
        "excluded_held_out_accepted_bbox_record_count": len(held_out_positive_ids),
        "excluded_held_out_accepted_bbox_record_ids": sorted(held_out_positive_ids),
        "excluded_held_out_negative_record_count": len(held_out_negative_ids),
        "human_negative_count": sum(1 for row in manifest_rows if not row["fish_present"]),
        "unique_complete_scene_image_count": len(manifest_rows),
        "unique_positive_image_count": sum(1 for row in manifest_rows if row["fish_present"]),
        "unique_bbox_count": sum(len(row["boxes"]) for row in manifest_rows),
        "source_record_errors": errors,
        "source_reconciliation": records.get("source_reconciliation"),
        "accepted_pool_metadata_summary": records.get("accepted_pool_metadata_summary"),
        "source_uris_attempted": len(source_uris_attempted),
        "source_uris_read": len(source_uris_read),
        "source_uri_failures": len(source_uri_failures),
        "unique_source_sha256_read": len(source_hashes_read),
        "source_image_audit_summary": source_object_audit.get("summary") or {},
        "source_image_audit_rows": source_object_audit.get("rows") or [],
    }


def _build_report(data: dict[str, Any], manifest_rows: list[dict[str, Any]], archive_sha256: str | None) -> dict[str, Any]:
    manifest_bytes = data.get("manifest_bytes")
    split_integrity_ok = bool(manifest_rows) and not data.get("source_record_errors")
    all_positive_reviews = sum(len(row.get("review_record_ids") or []) for row in manifest_rows if row.get("fish_present"))
    all_have_review = all(
        row.get("reviewer_id") and row.get("reviewed_at") and row.get("training_eligibility_basis")
        for row in manifest_rows
    ) if manifest_rows else False
    negatives = int(data.get("human_negative_count") or 0)
    reconciliation = data.get("source_reconciliation")
    source_audit_rows = data.get("source_image_audit_rows") or []
    source_audit_summary = data.get("source_image_audit_summary") or {}
    expected_audit_rows = (
        int((reconciliation or {}).get("accepted_pool_rows") or 0)
        + int(data.get("source_human_negative_record_count") or 0)
    ) if isinstance(reconciliation, dict) else None
    source_image_audit_verified = bool(
        source_audit_rows
        and expected_audit_rows == len(source_audit_rows)
        and not data.get("source_record_errors")
        and all(
            row.get("status") == "READ_OK"
            and _is_sha256(str(row.get("source_image_sha256") or ""))
            and int(row.get("width") or 0) > 0
            and int(row.get("height") or 0) > 0
            and 1 <= int(row.get("exif_orientation") or 0) <= 8
            for row in source_audit_rows
        )
    )
    candidate_status = (
        "FROZEN_READY_FOR_TRAINING"
        if isinstance(reconciliation, dict)
        and reconciliation.get("status") == "PASS"
        and not data.get("source_record_errors")
        else "BLOCKED"
    )
    report = {
        "dataset_version": DATASET_VERSION,
        "status": candidate_status,
        "source": HUMAN_LABEL_SOURCE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "human_review_verified": bool(all_have_review and all_positive_reviews == data.get("manifested_accepted_bbox_record_count")),
        "held_out_exclusions_verified": not data.get("source_record_errors"),
        "negative_coverage_verified": negatives > 0,
        "negative_pool_provenance_verified": bool(negatives) and all(
            row.get("fish_present")
            or bool(row.get("review_provenance")) and all(
                str(provenance.get("pool_entry_id") or "").isdigit()
                and int(provenance.get("pool_entry_id") or 0) > 0
                and str(provenance.get("pool_event_id") or "").isdigit()
                and int(provenance.get("pool_event_id") or 0) > 0
                and str(provenance.get("pool_source_image_sha256") or "") == str(row.get("image_sha256") or "")
                for provenance in row.get("review_provenance") or []
            )
            for row in manifest_rows
        ),
        "split_integrity": "PASS" if split_integrity_ok else "BLOCKED",
        "source_accepted_bbox_record_count": int(data.get("source_accepted_bbox_record_count") or 0),
        "manifested_accepted_bbox_record_count": int(data.get("manifested_accepted_bbox_record_count") or 0),
        "excluded_held_out_accepted_bbox_record_count": int(data.get("excluded_held_out_accepted_bbox_record_count") or 0),
        "excluded_held_out_accepted_bbox_record_ids": list(data.get("excluded_held_out_accepted_bbox_record_ids") or []),
        "source_human_negative_record_count": int(data.get("source_human_negative_record_count") or 0),
        "source_negative_pool_active_member_count": int(data.get("negative_pool_active_member_count") or 0),
        "negative_pool_member_errors": list(data.get("negative_pool_member_errors") or []),
        "negative_pool_source_kinds": dict(sorted(Counter(
            str(provenance.get("pool_source_kind") or "unknown")
            for row in manifest_rows if not row.get("fish_present")
            for provenance in row.get("review_provenance") or []
        ).items())),
        "excluded_held_out_negative_record_count": int(data.get("excluded_held_out_negative_record_count") or 0),
        "source_manifest_sha256": (data.get("source_reconciliation") or {}).get("manifest_sha256") if isinstance(data.get("source_reconciliation"), dict) else None,
        "source_metadata_sha256": (data.get("source_reconciliation") or {}).get("metadata_sha256") if isinstance(data.get("source_reconciliation"), dict) else None,
        "source_manifest_uri": (data.get("source_reconciliation") or {}).get("manifest_uri") if isinstance(data.get("source_reconciliation"), dict) else None,
        "source_metadata_uri": (data.get("source_reconciliation") or {}).get("metadata_uri") if isinstance(data.get("source_reconciliation"), dict) else None,
        "accepted_pool_metadata_summary": data.get("accepted_pool_metadata_summary"),
        "source_reconciliation": data.get("source_reconciliation"),
        "source_image_audit_verified": source_image_audit_verified,
        "source_image_audit_summary": source_audit_summary,
        "source_image_audit": source_audit_rows,
        "negatives": negatives,
        "unique_complete_scene_images": int(data.get("unique_complete_scene_image_count") or 0),
        "unique_positive_images": sum(1 for row in manifest_rows if row.get("fish_present")),
        "unique_bbox_count": int(data.get("unique_bbox_count") or 0),
        "original_source_images_read": int((data.get("source_image_audit_summary") or {}).get("unique_source_uris_read_and_decoded") or 0),
        "actual_source_images_read": int((data.get("source_image_audit_summary") or {}).get("unique_source_uris_read_and_decoded") or 0),
        "source_uris_attempted": int((data.get("source_image_audit_summary") or {}).get("unique_source_uris_attempted") or 0),
        "source_uris_read": int((data.get("source_image_audit_summary") or {}).get("unique_source_uris_read_and_decoded") or 0),
        "source_uri_failures": int((data.get("source_image_audit_summary") or {}).get("unique_source_uris_failed") or 0),
        "actual_source_images_failed": int((data.get("source_image_audit_summary") or {}).get("unique_source_uris_failed") or 0),
        "duplicate_accepted_bbox_records_over_unique_sources": max(
            int(data.get("source_accepted_bbox_record_count") or 0)
            - int(data.get("unique_positive_image_count") or 0)
            - int(data.get("excluded_held_out_accepted_bbox_record_count") or 0),
            0,
        ),
        "bbox_or_provenance_conflict_count": sum(
            1 for item in (data.get("source_record_errors") or []) if "conflict" in str(item.get("error") or "").lower()
        ),
        "invalid_bbox_record_count": int((reconciliation or {}).get("invalid_accepted_bbox_count") or 0)
        + sum(
            1 for item in (data.get("source_record_errors") or [])
            if "bbox" in str(item.get("error") or "").lower()
        ),
        "label_integrity_counts": {
            "invalid_accepted_bbox_records": int((reconciliation or {}).get("invalid_accepted_bbox_count") or 0),
            "bbox_alias_conflicts": int((reconciliation or {}).get("bbox_alias_conflicts") or 0),
            "database_bbox_conflicts": int((reconciliation or {}).get("database_bbox_conflicts") or 0),
            "source_uri_conflicts": int((reconciliation or {}).get("source_uri_conflicts") or 0),
            "pixel_bbox_or_source_size_errors": int((reconciliation or {}).get("pixel_bbox_or_source_size_errors") or 0),
            "crop_rectangle_errors": int((reconciliation or {}).get("crop_rectangle_errors") or 0),
        },
        "build_code_sha": os.environ.get("APP_GIT_COMMIT", "unknown"),
        "build_started_at": data.get("build_started_at"),
        "build_completed_at": datetime.now(timezone.utc).isoformat(),
        "source_record_error_count": len(data.get("source_record_errors") or []),
        "splits": data.get("splits") or {name: {"images": 0, "annotations": 0} for name in SPLIT_RATIOS},
        "manifest_sha256": _sha256(manifest_bytes) if manifest_bytes else None,
        "archive_sha256": archive_sha256,
        "source_record_errors": data.get("source_record_errors") or [],
        "source_scope": "approved ImageAsset + training_eligible + accepted BatchCropReview; full-scene ImageAsset.gcs_uri; no crops and no Google Vision labels",
        "split_method": "deterministic class-aware connected components of exact source SHA and ImageAsset.group_id; ImageAsset.id fallback when group_id is absent; 80/10/10 target",
        "materialization_policy": "EXIF-transposed RGB complete scenes; original PNGs are retained as lossless PNG and other formats are stored as high-quality JPEG quality=95; source-byte SHA remains separately recorded",
        "held_out_sha256": sorted(_HELD_OUT_SHA256),
    }
    return report


def _atomic_write(path: Path, data: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    if temp.exists():
        temp.unlink()
    temp.write_bytes(data)
    os.replace(temp, path)


def _same_frozen_report(existing: Any, current: Any) -> bool:
    """Compare immutable dataset identity while allowing build timestamps to differ."""

    if not isinstance(existing, dict) or not isinstance(current, dict):
        return False
    immutable_fields = (
        "dataset_version",
        "status",
        "source",
        "source_accepted_bbox_record_count",
        "source_human_negative_record_count",
        "source_manifest_sha256",
        "source_metadata_sha256",
        "manifest_sha256",
        "archive_sha256",
        "splits",
        "unique_complete_scene_images",
        "unique_bbox_count",
        "held_out_sha256",
    )
    return all(existing.get(field) == current.get(field) for field in immutable_fields)


def build_dataset(
    records: dict[str, Any],
    output_dir: str | Path,
    *,
    fetch_source: Callable[[str], bytes] = _read_source,
) -> dict[str, Any]:
    """Build and locally freeze only a gate-passing dataset.

    If input evidence is insufficient, a diagnostic BLOCKED report is emitted
    at the requested output path; no dataset archive or manifest is blessed.
    Existing output paths are never overwritten.
    """

    build_started_at = datetime.now(timezone.utc).isoformat()
    destination = Path(output_dir)
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise FileExistsError(f"refusing to overwrite existing dataset output: {destination}")
        destination.rmdir()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".det-ds-v02-stage-", dir=destination.parent) as temp_name:
        work_root = Path(temp_name)
        stage = Path(temp_name) / DATASET_VERSION
        stage.mkdir()
        staged_output = work_root / "output"
        staged_output.mkdir()
        source_object_audit = _audit_source_objects(records, fetch_source, stage / ".source-images")
        source_reconciliation = records.get("source_reconciliation")
        if isinstance(source_reconciliation, dict) and source_reconciliation.get("status") != "PASS":
            data = {
                "source_accepted_bbox_record_count": len(records.get("positives") or []),
                "source_human_negative_record_count": len(records.get("negatives") or []),
            "negative_pool_active_member_count": int(records.get("negative_pool_active_member_count") or len(records.get("negatives") or [])),
            "negative_pool_member_errors": list(records.get("negative_pool_member_errors") or []),
                "source_record_errors": list(records.get("source_record_errors") or []) + list(source_object_audit.get("errors") or []),
                "source_reconciliation": source_reconciliation,
                "accepted_pool_metadata_summary": records.get("accepted_pool_metadata_summary"),
                "splits": {name: {"images": 0, "annotations": 0} for name in SPLIT_RATIOS},
                "source_image_audit_summary": source_object_audit.get("summary") or {},
                "source_image_audit_rows": source_object_audit.get("rows") or [],
            }
            manifest_rows = []
        else:
            manifest_rows, data = _materialize(records, stage, source_object_audit)
        data.setdefault("build_started_at", build_started_at)
        archive_temp = Path(temp_name) / f"{DATASET_VERSION}.tar.gz"
        archive_sha = _deterministic_archive(stage, archive_temp) if manifest_rows else None
        report = _build_report(data, manifest_rows, archive_sha)
        manifest_bytes = data.get("manifest_bytes")

        gate = validate_v02_dataset(report, manifest_bytes, archive_sha)
        materialized_blockers = validate_v02_materialized_dataset(stage, manifest_bytes, gate["counts"]) if manifest_bytes else ["缺少有效逐图 Manifest"]
        blockers = list(dict.fromkeys(gate["blockers"] + materialized_blockers))
        if blockers:
            report["status"] = "BLOCKED"
            report["blockers"] = blockers
            report_bytes = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
            _atomic_write(staged_output / "bootstrap_report.json", report_bytes)
            os.replace(staged_output, destination)
            return {"ready": False, "blockers": blockers, "report": report, "output_dir": str(destination)}

        report["status"] = "FROZEN_READY_FOR_TRAINING"
        report["gate_blockers"] = []
        # This report hash is not included in the archive, avoiding a circular
        # digest. It is uploaded last and serves as the immutable commit marker.
        report_bytes = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        final_archive = staged_output / f"{DATASET_VERSION}.tar.gz"
        final_manifest = staged_output / "bootstrap_manifest.json"
        final_report = staged_output / "bootstrap_report.json"
        if final_archive.exists():
            raise FileExistsError(f"refusing to overwrite frozen artifact: {final_archive}")
        shutil.copyfile(archive_temp, final_archive)
        if _sha256_path(final_archive) != archive_sha:
            raise IOError("copied archive SHA-256 changed before local freeze")
        _atomic_write(final_manifest, manifest_bytes or b"")
        _atomic_write(final_report, report_bytes)
        os.replace(staged_output, destination)
    return {"ready": True, "blockers": [], "report": report, "output_dir": str(destination)}


def publish_gcs(output_dir: str | Path, bucket_name: str, *, client=None) -> dict[str, str]:
    """Publish immutable objects with GCS generation preconditions.

    The report is written last, so incomplete uploads cannot appear frozen to
    the reader. Existing objects are accepted only when their bytes match.
    """

    if client is None:
        from google.cloud import storage

        client = storage.Client()
    root = Path(output_dir)
    report_path = root / "bootstrap_report.json"
    manifest_path = root / "bootstrap_manifest.json"
    archive_path = root / f"{DATASET_VERSION}.tar.gz"
    if not all(path.is_file() for path in (report_path, manifest_path, archive_path)):
        raise FileNotFoundError("refusing GCS publish: frozen archive, manifest, and report are all required")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "FROZEN_READY_FOR_TRAINING":
        raise ValueError("refusing GCS publish: report status is not FROZEN_READY_FOR_TRAINING")
    manifest_bytes = manifest_path.read_bytes()
    archive_sha = _sha256_path(archive_path)
    gate = validate_v02_dataset(report, manifest_bytes, archive_sha)
    if not gate["ready"]:
        raise ValueError("refusing GCS publish: local frozen evidence failed validation: " + "; ".join(gate["blockers"]))
    artifacts = [
        (archive_path, f"{GCS_PREFIX}/{archive_path.name}", "application/gzip"),
        (manifest_path, f"{GCS_PREFIX}/bootstrap_manifest.json", "application/json"),
        (report_path, f"{GCS_PREFIX}/bootstrap_report.json", "application/json"),
    ]
    bucket = client.bucket(bucket_name)
    results: dict[str, str] = {}
    # Check every existing object before writing any new one.
    for source, object_name, _content_type in artifacts:
        blob = bucket.blob(object_name)
        if blob.exists(client):
            existing = blob.download_as_bytes()
            if _sha256(existing) != _sha256(source.read_bytes()):
                if object_name.endswith("/bootstrap_report.json"):
                    try:
                        existing_report = json.loads(existing.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        existing_report = None
                    existing_gate = validate_v02_dataset(existing_report, manifest_bytes, archive_sha)
                    if existing_gate["ready"] and _same_frozen_report(existing_report, report):
                        # A successful previous run may have frozen identical
                        # data with different timestamps. Keep its immutable
                        # report instead of rejecting a safe idempotent retry.
                        results[object_name] = "UNCHANGED"
                        continue
                raise FileExistsError(f"refusing to overwrite different frozen GCS object gs://{bucket_name}/{object_name}")
            results[object_name] = "UNCHANGED"
        else:
            results[object_name] = "PENDING"
    for source, object_name, content_type in artifacts:
        if results[object_name] == "UNCHANGED":
            continue
        blob = bucket.blob(object_name)
        blob.upload_from_filename(str(source), content_type=content_type, if_generation_match=0)
        results[object_name] = "CREATED"
    return {f"gs://{bucket_name}/{name}": status for name, status in results.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=f"/tmp/{DATASET_VERSION}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    parser.add_argument("--publish-gcs", action="store_true", help="publish gate-passing immutable artifacts to GCS")
    parser.add_argument("--bucket", default=os.environ.get("GCS_BUCKET", ""))
    args = parser.parse_args(argv)
    try:
        if not args.bucket:
            raise RuntimeError("GCS_BUCKET must name the Accepted Pool and frozen dataset bucket")
        from google.cloud import storage
        from app.db import SessionLocal

        storage_client = storage.Client()
        pool_rows, pool_metadata, source_files = read_accepted_pool_gcs(args.bucket, client=storage_client)
        for field in ("current_accepted_bbox_count", "source_count", "accepted_pool_count"):
            if field in pool_metadata:
                source_files[f"metadata_{field}"] = pool_metadata.get(field)
        with SessionLocal() as db:
            db_snapshot = collect_source_records(db)
        source_records = reconcile_accepted_pool_export(pool_rows, db_snapshot, source_files)
        source_records["source_files"] = source_files
        source_records["accepted_pool_metadata_summary"] = {
            key: pool_metadata.get(key)
            for key in (
                "pool_name", "type", "source", "accepted_pool_mode", "accepted_pool_count",
                "current_accepted_bbox_count", "source_count", "dataset_count", "bbox_generated",
                "crop_generated", "split_strategy", "split_counts", "manifest_uri", "last_sync_at",
            )
            if key in pool_metadata
        }

        def source_fetch(uri: str) -> bytes:
            return _read_source(uri, storage_client=storage_client)

        result = build_dataset(source_records, args.output_dir, fetch_source=source_fetch)
    except Exception as exc:
        destination = Path(args.output_dir)
        blocked = {
            "dataset_version": DATASET_VERSION,
            "status": "BLOCKED",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "build_code_sha": os.environ.get("APP_GIT_COMMIT", "unknown"),
            "blockers": [f"无法读取 Accepted Pool/审核数据或原始工件：{type(exc).__name__}: {exc}"],
            "actual_source_images_read": 0,
            "unique_complete_scene_images": 0,
            "unique_bbox_count": 0,
            "source_record_errors": [],
        }
        try:
            if not destination.exists():
                destination.mkdir(parents=True)
            _atomic_write(destination / "bootstrap_report.json", json.dumps(blocked, ensure_ascii=False, indent=2).encode("utf-8"))
        except FileExistsError:
            pass
        print(json.dumps({
            "status": "BLOCKED",
            "dataset_version": DATASET_VERSION,
            "build_code_sha": blocked["build_code_sha"],
            "error_type": type(exc).__name__,
            "detail": "无法完成 Accepted Pool/审核数据读取；检查本次任务权限与受控诊断报告。",
        }, ensure_ascii=False))
        return 2
    report = result["report"]
    reconciliation = report.get("source_reconciliation") or {}
    source_error_categories = Counter(
        str(item.get("error") or "unknown").split(":", 1)[0][:100]
        for item in (report.get("source_record_errors") or [])
        if isinstance(item, dict)
    )
    summary = {
        "status": "FROZEN_READY_FOR_TRAINING" if result["ready"] else "BLOCKED",
        "dataset_version": DATASET_VERSION,
        "build_code_sha": report.get("build_code_sha"),
        "source_accepted_bbox_record_count": report.get("source_accepted_bbox_record_count"),
        "unique_complete_scene_images": report.get("unique_complete_scene_images"),
        "unique_bbox_count": report.get("unique_bbox_count"),
        "human_negative_count": report.get("negatives"),
        "source_images_read": report.get("actual_source_images_read"),
        "source_images_failed": report.get("actual_source_images_failed"),
        "source_image_audit_summary": report.get("source_image_audit_summary"),
        "source_manifest_sha256": report.get("source_manifest_sha256"),
        "source_metadata_sha256": report.get("source_metadata_sha256"),
        "manifest_sha256": report.get("manifest_sha256"),
        "archive_sha256": report.get("archive_sha256"),
        "label_integrity_counts": report.get("label_integrity_counts"),
        "source_reconciliation_status": reconciliation.get("status"),
        "source_reconciliation_problem_count": len(reconciliation.get("problems") or []),
        "source_reconciliation_problem_sample": list(reconciliation.get("problems") or [])[:20],
        "source_record_error_count": report.get("source_record_error_count"),
        "source_record_error_categories": dict(sorted(source_error_categories.items())),
        "splits": report.get("splits"),
        "blockers": result["blockers"],
    }
    print(json.dumps(summary, ensure_ascii=False))
    if not result["ready"]:
        return 2
    if args.publish_gcs:
        if not args.bucket:
            print("GCS_BUCKET is required for --publish-gcs")
            return 2
        try:
            print(json.dumps(publish_gcs(args.output_dir, args.bucket), ensure_ascii=False))
        except Exception as exc:
            print(json.dumps({"status": "PUBLISH_BLOCKED", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False))
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
