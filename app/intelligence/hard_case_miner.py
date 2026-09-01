from __future__ import annotations

import csv
import io
import json
import mimetypes
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .confusion_analyzer import _PRED_KEYS, _TRUE_KEYS, _first, _text

HARD_CASE_FIELDS = (
    "image_id",
    "file_name",
    "true_species",
    "pred_species",
    "confidence",
    "error_group",
    "model_version",
    "hard_case_type",
)
_SAFE_PART = re.compile(r"[^A-Za-z0-9._-]+")


def _load_json(source: Any) -> Any:
    if isinstance(source, (str, Path)):
        with Path(source).open("r", encoding="utf-8-sig") as handle:
            return json.load(handle)
    return source


def _rows_from_evaluation(source: Any) -> list[Mapping[str, Any]]:
    source = _load_json(source)
    if isinstance(source, list):
        return [row for row in source if isinstance(row, Mapping)]
    if not isinstance(source, Mapping):
        return []
    for key in ("samples", "evaluation_samples", "predictions", "results", "items", "errors"):
        rows = source.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, Mapping)]
    for key in ("test", "validation", "evaluation", "metrics", "report"):
        nested = source.get(key)
        rows = _rows_from_evaluation(nested) if isinstance(nested, Mapping) else []
        if rows:
            return rows
    return []


def _safe_part(value: str, fallback: str = "unknown") -> str:
    value = _SAFE_PART.sub("_", (value or "").strip()).strip("._-")
    return value or fallback


def pair_slug(true_species: str, pred_species: str) -> str:
    return f"{_safe_part(true_species)}_vs_{_safe_part(pred_species)}"


def default_error_group(true_species: str, pred_species: str) -> str:
    carp_family = {"grass_carp", "common_carp", "black_carp", "bighead_carp", "silver_carp"}
    if true_species in carp_family and pred_species in carp_family:
        return "carp_family_boundary"
    if "whitefish" in {true_species.lower(), pred_species.lower()}:
        return "whitefish_multi_class"
    return pair_slug(true_species, pred_species)


def _report_pairs(report: Any) -> dict[tuple[str, str], Mapping[str, Any]]:
    document = _load_json(report)
    if not isinstance(document, Mapping):
        return {}
    pairs = document.get("top_confusions") or document.get("confusions") or []
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in pairs:
        if not isinstance(row, Mapping):
            continue
        truth = _text(_first(row, _TRUE_KEYS))
        pred = _text(_first(row, _PRED_KEYS))
        if truth and pred and truth != pred:
            result[(truth, pred)] = row
    return result


def _confidence(row: Mapping[str, Any]) -> str:
    value = _first(row, ("confidence", "predicted_confidence", "prediction_confidence", "score"))
    if value in (None, ""):
        return ""
    try:
        return str(round(float(value), 6))
    except (TypeError, ValueError):
        return str(value).strip()


def _source_path(row: Mapping[str, Any]) -> tuple[Path | None, str]:
    source_value = _first(row, ("local_path", "source_path", "file_path", "image_path", "file_name", "filename"))
    source_uri = _text(_first(row, ("source_uri", "gcs_uri", "uri", "image_uri")))
    if source_value not in (None, ""):
        candidate = Path(str(source_value))
        if candidate.exists() and candidate.is_file():
            return candidate, source_uri or str(candidate)
    if source_uri and not source_uri.startswith("gs://"):
        candidate = Path(source_uri)
        if candidate.exists() and candidate.is_file():
            return candidate, source_uri
    return None, source_uri or _text(source_value)


def _copy_gcs(source_uri: str, destination: Path, storage_client: Any = None) -> bool:
    if not source_uri.startswith("gs://"):
        return False
    try:
        from google.cloud import storage

        client = storage_client or storage.Client()
        body = source_uri[5:]
        bucket_name, object_name = body.split("/", 1)
        destination.parent.mkdir(parents=True, exist_ok=True)
        client.bucket(bucket_name).blob(object_name).download_to_filename(str(destination))
        return destination.exists()
    except Exception:
        return False


def _destination_name(file_name: str, image_id: str, used: set[str]) -> str:
    raw = Path(file_name or "").name
    suffix = Path(raw).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}:
        suffix = ".jpg"
    stem = _safe_part(Path(raw).stem or image_id, fallback="image")
    candidate = f"{stem}{suffix}"
    if candidate not in used:
        used.add(candidate)
        return candidate
    index = 2
    while f"{stem}_{index}{suffix}" in used:
        index += 1
    candidate = f"{stem}_{index}{suffix}"
    used.add(candidate)
    return candidate


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(HARD_CASE_FIELDS))
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in HARD_CASE_FIELDS} for row in rows)


def mine_hard_cases(
    evaluation_samples: Any,
    confusion_report: Any,
    output_root: str | Path,
    model_version: str | None = None,
    *,
    include_priorities: Iterable[str] = ("P0", "P1", "P2"),
    storage_client: Any = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Extract misclassified evaluation images into an immutable reviewable set.

    ``output_root`` is intentionally caller-controlled (local staging or a new
    hard-case artifact prefix).  No existing Dataset/Batch object is modified.
    Missing image bytes are retained in the manifest with their source URI by
    default so an operator can remediate them; ``strict=True`` turns that into
    a validation error.
    """

    report = _load_json(confusion_report)
    report_pairs = _report_pairs(report)
    if model_version is None and isinstance(report, Mapping):
        model_version = _text(report.get("model_version")) or None
    model_version = model_version or "unknown"
    allowed = {str(value).upper() for value in include_priorities}
    rows = _rows_from_evaluation(evaluation_samples)
    root = Path(output_root) / _safe_part(model_version)
    used_names: dict[str, set[str]] = defaultdict(set)
    manifests_by_pair: dict[str, list[dict[str, str]]] = defaultdict(list)
    copied = missing = skipped = 0

    for sample in rows:
        truth = _text(_first(sample, _TRUE_KEYS))
        pred = _text(_first(sample, _PRED_KEYS))
        if not truth or not pred or truth == pred:
            continue
        pair = report_pairs.get((truth, pred))
        if pair is not None and _text(pair.get("priority")).upper() not in allowed:
            continue
        image_id = _text(_first(sample, ("image_id", "id", "asset_id"))) or f"case_{len(manifests_by_pair) + 1:06d}"
        directory = pair_slug(truth, pred)
        pair_root = root / directory
        file_name = _text(_first(sample, ("file_name", "filename", "image_name", "image_path"))) or f"{image_id}.jpg"
        destination_name = _destination_name(file_name, image_id, used_names[directory])
        destination = pair_root / "images" / destination_name
        source_path, source_uri = _source_path(sample)
        extracted = False
        if source_path:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)
            extracted = True
            copied += 1
        elif source_uri.startswith("gs://"):
            extracted = _copy_gcs(source_uri, destination, storage_client)
            if extracted:
                copied += 1
        if not extracted:
            missing += 1
            if strict:
                raise FileNotFoundError(f"hard case image bytes unavailable: {source_uri or file_name}")
        row = {
            "image_id": image_id,
            "file_name": destination_name,
            "true_species": truth,
            "pred_species": pred,
            "confidence": _confidence(sample),
            "error_group": _text(_first(sample, ("error_group", "hard_pair_type", "pair_type"))) or default_error_group(truth, pred),
            "model_version": model_version,
            "hard_case_type": _text(_first(sample, ("hard_case_type", "case_type"))) or "confusion_pair",
        }
        manifests_by_pair[directory].append(row)

    all_rows = [row for pair_rows in manifests_by_pair.values() for row in pair_rows]
    for directory, pair_rows in manifests_by_pair.items():
        _write_manifest(root / directory / "metadata" / "hard_case_manifest.csv", pair_rows)
    aggregate_path = root / "metadata" / "hard_case_manifest.csv"
    _write_manifest(aggregate_path, all_rows)
    # A root-level copy makes the artifact easy to locate from object browsers
    # while the per-pair manifests mirror the requested directory layout.
    _write_manifest(root / "hard_case_manifest.csv", all_rows)
    return {
        "model_version": model_version,
        "hard_case_root": str(root),
        "manifest_path": str(aggregate_path),
        "pair_count": len(manifests_by_pair),
        "row_count": len(all_rows),
        "copied_count": copied,
        "missing_image_count": missing,
        "pairs": {key: len(value) for key, value in sorted(manifests_by_pair.items())},
        "rows": all_rows,
    }


def build_hard_case_set(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return mine_hard_cases(*args, **kwargs)


class HardCaseMiner:
    def __init__(self, *, storage_client: Any = None, strict: bool = False):
        self.storage_client = storage_client
        self.strict = strict

    def mine(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("storage_client", self.storage_client)
        kwargs.setdefault("strict", self.strict)
        return mine_hard_cases(*args, **kwargs)

    __call__ = mine


__all__ = [
    "HARD_CASE_FIELDS",
    "HardCaseMiner",
    "build_hard_case_set",
    "default_error_group",
    "mine_hard_cases",
    "pair_slug",
]
