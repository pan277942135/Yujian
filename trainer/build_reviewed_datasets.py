"""Build reviewed Detector and Crop datasets from App inference assets.

The App detector output is intentionally only a candidate.  This module will
never fall back to ``detection.candidate_bbox`` when an accepted review box is
missing.  A caller must supply a record with status ACCEPTED/TRAINING_READY and
an explicit accepted_bbox, keeping the human gate between inference and labels.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable

from PIL import Image, ImageOps


ACCEPTED_STATUSES = {"ACCEPTED", "TRAINING_READY"}
DEFAULT_EXPAND_RATIO = 0.15


def _value(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def _storage_value(record: dict[str, Any], key: str) -> Any:
    storage = record.get("storage")
    if isinstance(storage, dict) and storage.get(key):
        return storage[key]
    return record.get(key)


def _record_id(record: dict[str, Any]) -> str:
    image_id = str(record.get("image_id") or "").strip()
    if not image_id:
        raise ValueError("inference record missing image_id")
    return image_id


def _status(record: dict[str, Any]) -> str:
    return str(record.get("status") or record.get("review_status") or "").strip().upper()


def _accepted_bbox(record: dict[str, Any]) -> list[float] | None:
    raw: Any = record.get("accepted_bbox")
    if raw is None:
        raw = record.get("accepted_bbox_json")
        if isinstance(raw, str) and raw.strip():
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = None
    if raw is None and isinstance(record.get("review"), dict):
        raw = record["review"].get("accepted_bbox")
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        values = [float(value) for value in raw]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        return None
    if values[2] <= 0 or values[3] <= 0 or values[0] + values[2] > 1.00001 or values[1] + values[3] > 1.00001:
        return None
    return values


def _accepted_species(record: dict[str, Any]) -> str:
    for key in ("accepted_species", "truth_species"):
        value = str(record.get(key) or "").strip()
        if value:
            return value
    review = record.get("review")
    if isinstance(review, dict):
        return str(review.get("accepted_species") or "").strip()
    return ""


def _split(image_id: str) -> str:
    bucket = int.from_bytes(hashlib.sha256(image_id.encode("utf-8")).digest()[:2], "big") % 100
    return "train" if bucket < 80 else "val" if bucket < 90 else "test"


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://") or "/" not in uri[5:]:
        raise ValueError(f"invalid GCS URI: {uri}")
    return tuple(uri[5:].split("/", 1))  # type: ignore[return-value]


def _read_image_bytes(record: dict[str, Any], image_loader: Callable[[str], bytes] | None = None) -> bytes:
    local = _storage_value(record, "source_image_path") or _storage_value(record, "image_path")
    if local and not str(local).startswith("gs://"):
        return Path(str(local)).read_bytes()
    uri = _storage_value(record, "image_gcs_uri") or _storage_value(record, "gcs_uri") or local
    if not uri:
        raise ValueError(f"record {record.get('image_id')} has no source image")
    if image_loader:
        return image_loader(str(uri))
    bucket, object_name = _parse_gs_uri(str(uri))
    from google.cloud import storage

    return storage.Client().bucket(bucket).blob(object_name).download_as_bytes(timeout=180)


def _save_jpeg(data: bytes, path: Path) -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(data)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.save(path, format="JPEG", quality=92)
        return image.size


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _reviewed_records(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    accepted: list[dict[str, Any]] = []
    counts = {"accepted": 0, "candidate_excluded": 0, "rejected_excluded": 0, "invalid_accepted": 0}
    for raw in records:
        record = dict(raw)
        status = _status(record)
        bbox = _accepted_bbox(record)
        if status not in ACCEPTED_STATUSES:
            counts["candidate_excluded" if status in {"CANDIDATE", "REVIEW_REQUIRED", "RECEIVED"} else "rejected_excluded"] += 1
            continue
        if bbox is None:
            counts["invalid_accepted"] += 1
            continue
        record["accepted_bbox"] = bbox
        accepted.append(record)
        counts["accepted"] += 1
    return accepted, counts


def build_reviewed_detector_dataset(
    records: Iterable[dict[str, Any]],
    output_root: str | Path,
    *,
    dataset_version: str = "DS_DET_FISH_v0.1",
    image_loader: Callable[[str], bytes] | None = None,
) -> dict[str, Any]:
    """Write one-class YOLO images/labels from explicitly accepted bboxes."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    accepted, excluded = _reviewed_records(records)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    split_counts = {name: {"images": 0, "labels": 0} for name in ("train", "val", "test")}
    for record in accepted:
        image_id = _record_id(record)
        split = _split(image_id)
        try:
            data = _read_image_bytes(record, image_loader)
            image_path = root / "images" / split / f"{image_id}.jpg"
            width, height = _save_jpeg(data, image_path)
            bbox = record["accepted_bbox"]
            label_path = root / "labels" / split / f"{image_id}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            x, y, w, h = bbox
            label_path.write_text(
                f"0 {(x + w / 2):.6f} {(y + h / 2):.6f} {w:.6f} {h:.6f}\n",
                encoding="utf-8",
            )
            split_counts[split]["images"] += 1
            split_counts[split]["labels"] += 1
            rows.append(
                {
                    "image_id": image_id,
                    "file_name": image_path.name,
                    "split": split,
                    "image_path": str(image_path.relative_to(root)),
                    "label_path": str(label_path.relative_to(root)),
                    "bbox_source": "accepted_review",
                    "accepted_bbox": json.dumps(bbox, separators=(",", ":")),
                    "pipeline_type": "DETECTOR_V1",
                }
            )
        except Exception as exc:
            failures.append({"image_id": image_id, "error": str(exc)})

    fields = ["image_id", "file_name", "split", "image_path", "label_path", "bbox_source", "accepted_bbox", "pipeline_type"]
    _write_csv(root / "metadata" / "manifest.csv", fields, rows)
    report = {
        "dataset_version": dataset_version,
        "pipeline_type": "DETECTOR_V1",
        "class_map": {"0": "fish"},
        "accepted": excluded["accepted"],
        "written": len(rows),
        "excluded": excluded,
        "failures": failures,
        "splits": split_counts,
        "safety": {
            "candidate_bbox_used_as_label": False,
            "requires_human_review": True,
            "auto_freeze": False,
            "auto_train": False,
        },
    }
    _write_json(root / "metadata" / "report.json", report)
    return report


def _crop_pixels(bbox: list[float], width: int, height: int, expand_ratio: float) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    x1 = max(0.0, x - w * expand_ratio)
    y1 = max(0.0, y - h * expand_ratio)
    x2 = min(1.0, x + w + w * expand_ratio)
    y2 = min(1.0, y + h + h * expand_ratio)
    left = max(0, min(width - 1, math.floor(x1 * width)))
    top = max(0, min(height - 1, math.floor(y1 * height)))
    right = max(left + 1, min(width, math.ceil(x2 * width)))
    bottom = max(top + 1, min(height, math.ceil(y2 * height)))
    return left, top, right, bottom


def build_crop_dataset(
    records: Iterable[dict[str, Any]],
    output_root: str | Path,
    *,
    dataset_version: str = "DS_CROP_M1_v0.1",
    expand_ratio: float = DEFAULT_EXPAND_RATIO,
    image_loader: Callable[[str], bytes] | None = None,
) -> dict[str, Any]:
    """Regenerate classifier crops from accepted boxes, never App candidate crops."""

    if not 0.0 <= expand_ratio <= 1.0:
        raise ValueError("expand_ratio must be between 0 and 1")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    accepted, excluded = _reviewed_records(records)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for record in accepted:
        image_id = _record_id(record)
        species = _accepted_species(record)
        if not species:
            failures.append({"image_id": image_id, "error": "accepted_species is required for crop dataset"})
            continue
        try:
            data = _read_image_bytes(record, image_loader)
            with Image.open(io.BytesIO(data)) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                left, top, right, bottom = _crop_pixels(record["accepted_bbox"], image.width, image.height, expand_ratio)
                crop = image.crop((left, top, right, bottom))
                safe_species = species.replace("/", "_").replace("\\", "_").strip() or "unknown"
                crop_path = root / "images" / safe_species / f"{image_id}_crop.jpg"
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                crop.save(crop_path, format="JPEG", quality=92)
            rows.append(
                {
                    "image_id": image_id,
                    "file_name": crop_path.name,
                    "species_key": species,
                    "gcs_uri": "",
                    "local_path": str(crop_path.relative_to(root)),
                    "input_type": "crop",
                    "pipeline_type": "CROP_CLASSIFIER_V1",
                    "source_image_id": image_id,
                    "split": _split(image_id),
                    "accepted_bbox": json.dumps(record["accepted_bbox"], separators=(",", ":")),
                    "expand_ratio": expand_ratio,
                }
            )
        except Exception as exc:
            failures.append({"image_id": image_id, "error": str(exc)})

    fields = ["image_id", "file_name", "species_key", "gcs_uri", "local_path", "input_type", "pipeline_type", "source_image_id", "split", "accepted_bbox", "expand_ratio"]
    _write_csv(root / "metadata" / "crop_manifest.csv", fields, rows)
    report = {
        "dataset_version": dataset_version,
        "pipeline_type": "CROP_CLASSIFIER_V1",
        "input_type": "crop",
        "expand_ratio": expand_ratio,
        "written": len(rows),
        "excluded": excluded,
        "failures": failures,
        "safety": {
            "original_images_used_for_classifier": False,
            "candidate_bbox_used": False,
            "requires_human_review": True,
            "auto_freeze": False,
            "auto_train": False,
        },
    }
    _write_json(root / "metadata" / "report.json", report)
    return report


def load_record_directory(root: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(Path(root).rglob("*.json")):
        if path.name == "report.json":
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(document, dict) and document.get("image_id"):
            records.append(document)
    return records


__all__ = [
    "ACCEPTED_STATUSES",
    "build_crop_dataset",
    "build_reviewed_detector_dataset",
    "load_record_directory",
]
