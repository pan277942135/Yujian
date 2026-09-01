"""Validation for the reviewed crop-classifier dataset contract.

Crop training is deliberately stricter than the App inference contract: a
row must point at a materialised crop and carry the box and species accepted
by a human reviewer.  In particular, a detector ``candidate_bbox`` is never
accepted as a substitute for ``accepted_bbox`` here.

The module is dependency-light so it can be used by dataset builders, the
training worker, and offline CI checks without importing the training stack.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


CROP_VALIDATOR_VERSION = "CROP_DATASET_VALIDATOR_V1"
CROP_PIPELINE_TYPE = "CROP_CLASSIFIER_V1"


class CropDatasetValidationError(ValueError):
    """Raised when a crop manifest cannot be used for classifier training."""

    def __init__(self, report: Mapping[str, Any]):
        self.report = dict(report)
        errors = self.report.get("errors") or []
        if errors:
            first = errors[0]
            message = f"{first.get('code', 'INVALID')}: {first.get('message', 'crop dataset is invalid')}"
        else:
            message = "crop dataset is invalid"
        super().__init__(message)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _first(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and _text(value):
            return value
    return None


def _parse_bbox(value: Any) -> list[float] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) and 0.0 <= item <= 1.0 for item in values):
        return None
    if values[2] <= 0 or values[3] <= 0:
        return None
    if values[0] + values[2] > 1.00001 or values[1] + values[3] > 1.00001:
        return None
    return values


def _resolve_local_path(value: str, dataset_root: Path | None) -> Path:
    path = Path(value)
    if not path.is_absolute() and dataset_root is not None:
        path = dataset_root / path
    return path


def validate_crop_rows(
    rows: Iterable[Mapping[str, Any]],
    dataset_root: str | Path | None = None,
    *,
    require_bbox: bool = True,
    allow_remote: bool = False,
    image_exists: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    """Validate crop manifest rows and return an auditable report.

    ``dataset_root`` is used to resolve relative ``local_path`` values.  A
    remote URI is considered unverifiable unless ``image_exists`` is supplied
    (or ``allow_remote`` is explicitly enabled by a caller that has already
    checked the object).  This keeps a training job from silently accepting a
    missing crop object.
    """

    materialized = [dict(row) for row in rows]
    root = Path(dataset_root) if dataset_root is not None else None
    errors: list[dict[str, Any]] = []
    seen: dict[str, int] = {}

    def add(index: int, image_id: str, code: str, message: str) -> None:
        errors.append({"row": index, "image_id": image_id or None, "code": code, "message": message})

    if not materialized:
        add(1, "", "MANIFEST_EMPTY", "crop manifest must contain at least one row")

    for index, row in enumerate(materialized, start=2):
        image_id = _text(row.get("image_id"))
        if not image_id:
            add(index, image_id, "MISSING_IMAGE_ID", "image_id is required")
        elif image_id in seen:
            add(index, image_id, "DUPLICATE_IMAGE_ID", f"image_id already appears on manifest row {seen[image_id]}")
        else:
            seen[image_id] = index

        # ``accepted_species`` is the review-facing name; generated manifests
        # use ``species_key``.  ``species`` is accepted for older exported
        # manifests, but only after the explicit reviewed fields.
        species = _first(row, "accepted_species", "species_key", "species")
        if not _text(species):
            add(index, image_id, "MISSING_SPECIES", "accepted_species, species_key, or species is required")

        if require_bbox:
            raw_bbox = _first(row, "accepted_bbox", "accepted_bbox_json")
            if raw_bbox is None:
                add(index, image_id, "MISSING_ACCEPTED_BBOX", "accepted_bbox is required; candidate_bbox cannot be used")
            elif _parse_bbox(raw_bbox) is None:
                add(index, image_id, "INVALID_ACCEPTED_BBOX", "accepted_bbox must be normalized [x, y, width, height]")

        input_type = _text(row.get("input_type"))
        if input_type and input_type.lower() != "crop":
            add(index, image_id, "ORIGINAL_INPUT_FORBIDDEN", "CROP_CLASSIFIER_V1 accepts crop inputs only")
        pipeline_type = _text(row.get("pipeline_type"))
        if pipeline_type and pipeline_type != CROP_PIPELINE_TYPE:
            add(index, image_id, "PIPELINE_TYPE_INVALID", f"pipeline_type must be {CROP_PIPELINE_TYPE}")

        # Crop-specific paths must win over generic paths.  This prevents a
        # manifest containing both ``local_path`` (original) and ``crop_path``
        # (crop) from silently training on the original image.
        crop_ref = _first(row, "crop_gcs_uri", "crop_path", "local_path", "file_path", "gcs_uri")
        if not _text(crop_ref):
            add(index, image_id, "MISSING_CROP", "crop path or URI is required")
        else:
            crop_value = _text(crop_ref)
            source_refs = {
                _text(row.get(key))
                for key in ("source_image_gcs_uri", "source_image_path", "original_gcs_uri", "original_path", "image_gcs_uri", "image_path")
            }
            if crop_value in source_refs:
                add(index, image_id, "ORIGINAL_INPUT_FORBIDDEN", "crop reference must not equal the source/original image")
            if crop_value.startswith("gs://"):
                if image_exists is not None:
                    try:
                        exists = bool(image_exists(crop_value))
                    except Exception as exc:  # pragma: no cover - defensive callback boundary
                        exists = False
                        add(index, image_id, "CROP_CHECK_FAILED", f"crop existence check failed: {exc}")
                    if not exists:
                        add(index, image_id, "CROP_NOT_FOUND", "crop object does not exist")
                elif not allow_remote:
                    add(index, image_id, "CROP_NOT_VERIFIABLE", "remote crop requires an existence check")
            else:
                path = _resolve_local_path(crop_value, root)
                try:
                    exists = path.is_file() and path.stat().st_size > 0
                except OSError:
                    exists = False
                if not exists:
                    add(index, image_id, "CROP_NOT_FOUND", f"crop file does not exist: {path}")

    codes = {error["code"] for error in errors}
    checks = {
        "crop_exists": bool(materialized) and not bool(codes & {"MISSING_CROP", "CROP_NOT_FOUND", "CROP_NOT_VERIFIABLE", "CROP_CHECK_FAILED"}),
        "species_present": bool(materialized) and "MISSING_SPECIES" not in codes,
        "accepted_bbox_present": bool(materialized) and not bool(codes & {"MISSING_ACCEPTED_BBOX", "INVALID_ACCEPTED_BBOX"}),
        "image_id_unique": bool(materialized) and "DUPLICATE_IMAGE_ID" not in codes and "MISSING_IMAGE_ID" not in codes,
    }
    valid_rows = len(materialized) - len({error["row"] for error in errors})
    return {
        "validator_version": CROP_VALIDATOR_VERSION,
        "valid": not errors,
        "rows": len(materialized),
        "valid_rows": max(0, valid_rows),
        "unique_image_ids": len(seen),
        "errors": errors,
        "checks": checks,
        "safety": {
            "accepted_bbox_required": require_bbox,
            "candidate_bbox_accepted": False,
            "original_image_input_accepted": False,
        },
    }


def validate_crop_dataset(
    dataset_root: str | Path,
    manifest_path: str | Path | None = None,
    *,
    require_bbox: bool = True,
    allow_remote: bool = False,
    image_exists: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    """Load and validate a dataset's crop manifest."""

    root = Path(dataset_root)
    path = Path(manifest_path) if manifest_path is not None else root / "metadata" / "crop_manifest.csv"
    if not path.is_absolute():
        path = root / path
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except FileNotFoundError:
        return {
            "validator_version": CROP_VALIDATOR_VERSION,
            "valid": False,
            "rows": 0,
            "valid_rows": 0,
            "unique_image_ids": 0,
            "errors": [{"row": 1, "image_id": None, "code": "MANIFEST_NOT_FOUND", "message": f"manifest does not exist: {path}"}],
            "checks": {"crop_exists": False, "species_present": False, "accepted_bbox_present": False, "image_id_unique": False},
            "safety": {"accepted_bbox_required": require_bbox, "candidate_bbox_accepted": False, "original_image_input_accepted": False},
        }
    return validate_crop_rows(
        rows,
        root,
        require_bbox=require_bbox,
        allow_remote=allow_remote,
        image_exists=image_exists,
    )


def validate_crop_manifest(
    manifest: str | Path | Iterable[Mapping[str, Any]],
    dataset_root: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Validate either a crop CSV path/root or already-loaded manifest rows.

    Older workers used ``validate_crop_manifest(rows, root)`` while the
    dataset-oriented API uses ``validate_crop_dataset(root, manifest_path)``.
    Keeping both forms avoids forcing callers to rewrite their import layer.
    """

    if isinstance(manifest, (str, Path)):
        path = Path(manifest)
        if path.suffix.lower() == ".csv":
            root = dataset_root or path.parent.parent
            return validate_crop_dataset(root, path, **kwargs)
        return validate_crop_dataset(path, dataset_root, **kwargs)
    return validate_crop_rows(manifest, dataset_root, **kwargs)


def raise_if_invalid(report: Mapping[str, Any]) -> dict[str, Any]:
    """Raise a concise contract error while retaining the full report."""

    if not report.get("valid"):
        raise CropDatasetValidationError(report)
    return dict(report)


class CropDatasetValidator:
    """Small object facade for dependency-injection and worker callers."""

    def __init__(self, *, require_bbox: bool = True, allow_remote: bool = False, image_exists: Callable[[str], bool] | None = None):
        self.require_bbox = require_bbox
        self.allow_remote = allow_remote
        self.image_exists = image_exists

    def validate_rows(self, rows: Iterable[Mapping[str, Any]], dataset_root: str | Path | None = None) -> dict[str, Any]:
        return validate_crop_rows(
            rows,
            dataset_root,
            require_bbox=self.require_bbox,
            allow_remote=self.allow_remote,
            image_exists=self.image_exists,
        )

    def validate(self, dataset_root: str | Path, manifest_path: str | Path | None = None) -> dict[str, Any]:
        return validate_crop_dataset(
            dataset_root,
            manifest_path,
            require_bbox=self.require_bbox,
            allow_remote=self.allow_remote,
            image_exists=self.image_exists,
        )


__all__ = [
    "CROP_PIPELINE_TYPE",
    "CROP_VALIDATOR_VERSION",
    "CropDatasetValidationError",
    "CropDatasetValidator",
    "raise_if_invalid",
    "validate_crop_dataset",
    "validate_crop_manifest",
    "validate_crop_rows",
]
