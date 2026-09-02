"""Production builder and explicit Freeze step for reviewed crop datasets.

This module is intentionally separate from the cumulative whole-image Freeze
policy.  A crop classifier dataset is a derivative of reviewed inference
assets, so its only admissible label source is ``accepted_bbox`` plus an
explicit human-reviewed species.  Building a dataset never changes review
state; calling :func:`freeze_crop_dataset` is the explicit operator gate.
"""

from __future__ import annotations

import csv
import io
import json
import mimetypes
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from trainer.build_reviewed_datasets import build_crop_dataset, load_reviewed_inference_records
from trainer.crop_dataset_validator import CropDatasetValidationError, validate_crop_dataset


CROP_DATASET_VERSION = "DS_CROP_M1_v0.1"
CROP_PIPELINE_TYPE = "CROP_CLASSIFIER_V1"
CROP_INPUT_TYPE = "crop_image"
CROP_READY_STATUS = "READY_FOR_TRAINING"
CROP_SELECTION_MODE = "ACCEPTED_BBOX_CROP"
CROP_EXPAND_RATIO = 0.15


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://") or "/" not in uri[5:]:
        raise ValueError(f"invalid GCS URI: {uri}")
    return tuple(uri[5:].split("/", 1))  # type: ignore[return-value]


def _blob_exists(blob: Any, client: Any = None) -> bool:
    try:
        return bool(blob.exists(client))
    except TypeError:
        return bool(blob.exists())


def _upload_bytes(blob: Any, data: bytes, *, content_type: str) -> None:
    try:
        blob.upload_from_string(data, content_type=content_type, if_generation_match=0)
    except TypeError:  # lightweight test doubles may not accept generation guards
        blob.upload_from_string(data, content_type=content_type)


def _upload_file(blob: Any, path: Path, *, content_type: str) -> None:
    try:
        blob.upload_from_filename(str(path), content_type=content_type, if_generation_match=0)
    except TypeError:  # lightweight test doubles may not accept generation guards
        blob.upload_from_filename(str(path), content_type=content_type)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    return fields


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _species_name_map(db: Any) -> dict[str, str]:
    if db is None:
        return {}
    try:
        from app.models import SpeciesCatalog
        from sqlalchemy import select

        rows = db.scalars(select(SpeciesCatalog)).all()
    except Exception:
        return {}
    return {
        str(row.species_key).strip(): str(row.common_name_zh).strip()
        for row in rows
        if str(row.species_key or "").strip() and str(row.common_name_zh or "").strip()
    }


def load_reviewed_crop_records(db: Any, storage_client: Any = None, *, limit: int = 5000) -> list[dict[str, Any]]:
    """Load only ACCEPTED/TRAINING_READY inference records and join catalog names."""

    records = load_reviewed_inference_records(db, storage_client=storage_client, limit=limit)
    names = _species_name_map(db)
    reverse = {name: key for key, name in names.items()}
    for record in records:
        reviewed = str(record.get("accepted_species") or "").strip()
        if reviewed and reviewed in reverse:
            record.setdefault("accepted_species_key", reverse[reviewed])
            record.setdefault("accepted_species_name", reviewed)
        elif reviewed and reviewed in names:
            record.setdefault("accepted_species_key", reviewed)
            record.setdefault("accepted_species_name", names[reviewed])
    return records


def build_reviewed_crop_dataset(
    records: Iterable[Mapping[str, Any]],
    output_root: str | Path,
    *,
    dataset_version: str = CROP_DATASET_VERSION,
    expand_ratio: float = CROP_EXPAND_RATIO,
    image_loader: Any = None,
    species_name_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Generate a local, validated crop dataset from reviewed records."""

    return build_crop_dataset(
        [dict(record) for record in records],
        output_root,
        dataset_version=dataset_version,
        expand_ratio=expand_ratio,
        image_loader=image_loader,
        species_name_map=species_name_map,
        input_type=CROP_INPUT_TYPE,
    )


def build_reviewed_crop_dataset_from_db(
    db: Any,
    output_root: str | Path,
    *,
    dataset_version: str = CROP_DATASET_VERSION,
    expand_ratio: float = CROP_EXPAND_RATIO,
    storage_client: Any = None,
    limit: int = 5000,
) -> dict[str, Any]:
    records = load_reviewed_crop_records(db, storage_client=storage_client, limit=limit)
    return build_reviewed_crop_dataset(
        records,
        output_root,
        dataset_version=dataset_version,
        expand_ratio=expand_ratio,
        species_name_map=_species_name_map(db),
    )


def _content_type(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _prepare_published_manifest(root: Path, bucket_name: str, dataset_version: str) -> tuple[Path, list[dict[str, Any]]]:
    manifest_path = root / "metadata" / "crop_manifest.csv"
    rows = _read_rows(manifest_path)
    prefix = f"datasets/{dataset_version}/"
    for row in rows:
        crop_rel = str(
            row.get("crop_image_path")
            or row.get("crop_path")
            or row.get("local_path")
            or ""
        ).replace("\\", "/").lstrip("/")
        if not crop_rel:
            raise CropDatasetValidationError(
                {
                    "valid": False,
                    "errors": [
                        {
                            "row": None,
                            "image_id": row.get("image_id"),
                            "code": "MISSING_CROP_IMAGE_PATH",
                            "message": "crop_image_path is required before Freeze",
                        }
                    ],
                }
            )
        crop_uri = f"gs://{bucket_name}/{prefix}{crop_rel}"
        row["crop_image_path"] = crop_rel
        row["crop_path"] = crop_rel
        row["gcs_uri"] = crop_uri
        row["crop_gcs_uri"] = crop_uri
        # The trainer may materialize a local copy, but the immutable manifest
        # must identify the crop object as its classifier input.
        row["input_type"] = CROP_INPUT_TYPE
        row["pipeline_type"] = CROP_PIPELINE_TYPE
    _write_rows(manifest_path, rows)
    return manifest_path, rows


def freeze_crop_dataset(
    dataset_root: str | Path,
    *,
    dataset_version: str = CROP_DATASET_VERSION,
    bucket_name: str | None = None,
    storage_client: Any = None,
    db: Any = None,
    git_commit: str = "unknown",
    status: str = CROP_READY_STATUS,
    preprocess_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Explicitly publish and register a reviewed crop dataset.

    The function refuses to publish an invalid manifest.  It never changes an
    inference review status and never starts a training job.
    """

    if not dataset_version.startswith("DS_"):
        raise ValueError("dataset_version must start with DS_")
    if status != CROP_READY_STATUS:
        raise ValueError(f"crop dataset Freeze status must be {CROP_READY_STATUS}")
    root = Path(dataset_root)
    manifest_path = root / "metadata" / "crop_manifest.csv"
    validation = validate_crop_dataset(root, require_metadata=True, check_source_image=True)
    ratios: set[float] = set()
    invalid_ratio = False
    if manifest_path.is_file():
        for row in _read_rows(manifest_path):
            try:
                ratios.add(round(float(row.get("expand_ratio")), 6))
            except (TypeError, ValueError):
                invalid_ratio = True
    if invalid_ratio or any(abs(ratio - CROP_EXPAND_RATIO) > 1e-6 for ratio in ratios):
        validation["valid"] = False
        validation.setdefault("checks", {})["expand_ratio_contract"] = False
        validation.setdefault("checks", {})["metadata_complete"] = False
        validation.setdefault("errors", []).append(
            {
                "row": None,
                "image_id": None,
                "code": "EXPAND_RATIO_MISMATCH",
                "message": f"production crop Freeze requires expand_ratio={CROP_EXPAND_RATIO}",
            }
        )
    else:
        validation.setdefault("checks", {})["expand_ratio_contract"] = True
    if not validation.get("valid"):
        raise CropDatasetValidationError(validation)
    rows = _read_rows(manifest_path)
    if not rows:
        raise CropDatasetValidationError(
            {"valid": False, "errors": [{"code": "MANIFEST_EMPTY", "message": "crop manifest is empty"}]}
        )

    published_manifest = manifest_path
    manifest_uri = str(manifest_path)
    class_map_uri = str(root / "metadata" / "class_map.json")
    now = utcnow_iso()
    if bucket_name:
        if storage_client is None:
            from google.cloud import storage

            storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        prefix = f"datasets/{dataset_version}/"
        marker = bucket.blob(prefix + "dataset.json")
        if _blob_exists(marker, storage_client):
            raise ValueError(f"dataset already exists in GCS: gs://{bucket_name}/{prefix}")

        published_manifest, rows = _prepare_published_manifest(root, bucket_name, dataset_version)
        # Upload crops and metadata.  The marker is written last and is the
        # immutable publication point consumed by operators and training.
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path == published_manifest:
                continue
            rel = path.relative_to(root).as_posix()
            _upload_file(bucket.blob(prefix + rel), path, content_type=_content_type(path))
        manifest_uri = f"gs://{bucket_name}/{prefix}metadata/crop_manifest.csv"
        class_map_uri = f"gs://{bucket_name}/{prefix}metadata/class_map.json"
        _upload_file(bucket.blob(prefix + "metadata/crop_manifest.csv"), published_manifest, content_type="text/csv")
    else:
        # Local mode is useful for CI and operator dry-runs.  It still writes
        # the same metadata contract, but cannot register a trainable GCS URI.
        _prepare_published_manifest(root, "local", dataset_version)

    split_counts = Counter(str(row.get("split") or "") for row in rows)
    species_counts = Counter(str(row.get("species_name") or row.get("species_key") or "") for row in rows)
    class_keys = sorted({str(row.get("species_key") or "") for row in rows if str(row.get("species_key") or "")})
    metadata = {
        "dataset_version": dataset_version,
        "pipeline": CROP_PIPELINE_TYPE,
        "pipeline_type": CROP_PIPELINE_TYPE,
        "input_type": CROP_INPUT_TYPE,
        "source": "accepted_bbox",
        "review_status_required": sorted({str(row.get("review_status") or "") for row in rows}),
        "crop_expand_ratio": CROP_EXPAND_RATIO,
        "image_count": len(rows),
        "class_count": len(class_keys),
        "classes": class_keys,
        "species_counts": dict(species_counts),
        "split_counts": {name: split_counts.get(name, 0) for name in ("train", "val", "test")},
        "manifest_uri": manifest_uri,
        "class_map_uri": class_map_uri,
        "created_at": now,
        "git_commit": git_commit or "unknown",
        "status": status,
        "immutable": True,
        "validation": validation,
        "preprocess_contract": dict(preprocess_contract or {}),
        "safety": {
            "source_is_accepted_bbox": True,
            "candidate_bbox_used": False,
            "original_image_classifier_input": False,
            "auto_train": False,
        },
    }

    if bucket_name:
        prefix = f"datasets/{dataset_version}/"
        bucket = storage_client.bucket(bucket_name)
        _upload_bytes(
            bucket.blob(prefix + "metadata/dataset.json"),
            json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"),
            content_type="application/json",
        )
        _upload_bytes(
            bucket.blob(prefix + "metadata/freeze_report.json"),
            json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"),
            content_type="application/json",
        )
        _upload_bytes(
            bucket.blob(prefix + "dataset.json"),
            json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"),
            content_type="application/json",
        )
    else:
        # Keep local dry-runs inspectable with the same Freeze markers as the
        # published GCS tree.  This does not register or train anything.
        _write_json(root / "metadata" / "dataset.json", metadata)
        _write_json(root / "metadata" / "freeze_report.json", metadata)
        _write_json(root / "dataset.json", metadata)

    if db is not None:
        from app.models import DatasetVersion

        existing = db.get(DatasetVersion, dataset_version)
        if existing:
            raise ValueError(f"dataset already registered: {dataset_version}")
        dataset = DatasetVersion(
            dataset_version=dataset_version,
            parent_version=None,
            manifest_uri=manifest_uri,
            class_map_uri=class_map_uri,
            train_count=split_counts.get("train", 0),
            val_count=split_counts.get("val", 0),
            test_count=split_counts.get("test", 0),
            species_count=len(class_keys),
            git_commit=git_commit or "unknown",
            selection_mode=CROP_SELECTION_MODE,
            source_cutoff_at=datetime.now(timezone.utc),
            status=status,
            pipeline_type=CROP_PIPELINE_TYPE,
            metadata_json=json.dumps(metadata, ensure_ascii=False),
        )
        db.add(dataset)
        db.commit()

    return {
        "dataset_version": dataset_version,
        "status": status,
        "pipeline": CROP_PIPELINE_TYPE,
        "source": "accepted_bbox",
        "image_count": len(rows),
        "class_count": len(class_keys),
        "split_counts": {name: split_counts.get(name, 0) for name in ("train", "val", "test")},
        "manifest_uri": manifest_uri,
        "class_map_uri": class_map_uri,
        "freeze_metadata": metadata,
        "validation": validation,
        "auto_train": False,
    }


__all__ = [
    "CROP_DATASET_VERSION",
    "CROP_EXPAND_RATIO",
    "CROP_INPUT_TYPE",
    "CROP_PIPELINE_TYPE",
    "CROP_READY_STATUS",
    "CROP_SELECTION_MODE",
    "build_reviewed_crop_dataset",
    "build_reviewed_crop_dataset_from_db",
    "freeze_crop_dataset",
    "load_reviewed_crop_records",
]
