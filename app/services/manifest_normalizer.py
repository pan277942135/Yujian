from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


OUTPUT_FIELDS = ("image_path", "image_id", "claimed_species", "species_key", "source")

IMAGE_FIELD_ALIASES = ("image_path", "file_name", "filename", "image_name")
SPECIES_FIELD_ALIASES = ("claimed_species", "species_name", "fish_name", "label", "species")
SPECIES_KEY_ALIASES = ("species_key", "class_name", "category_key")
SOURCE_FIELD_ALIASES = ("source_platform", "source", "dataset_source")


class ManifestNormalizationError(ValueError):
    """A deterministic, user-actionable manifest contract error."""

    code = "MANIFEST_INVALID"

    def __init__(self, reason: str, *, source_path: str | Path | None = None, row_number: int | None = None):
        self.reason = reason
        self.source_path = str(source_path) if source_path is not None else None
        self.row_number = row_number
        super().__init__(reason)

    def as_dict(self) -> dict[str, str]:
        payload = {"error": self.code, "reason": self.reason}
        if self.row_number is not None:
            payload["row"] = str(self.row_number)
        return payload


@dataclass(frozen=True)
class ManifestNormalizationResult:
    output_path: Path
    source_path: Path
    rows: int
    generated: bool

    @property
    def status(self) -> str:
        return "MANIFEST_READY"

    @property
    def manifest_rows(self) -> int:
        return self.rows

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "output_path": str(self.output_path),
            "source_path": str(self.source_path),
            "rows": self.rows,
            "manifest_rows": self.rows,
            "generated": self.generated,
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.as_dict().get(key, default)


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _field_lookup(fieldnames: Iterable[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for field in fieldnames:
        name = _clean(field)
        if not name:
            continue
        lookup.setdefault(name.casefold(), name)
    return lookup


def _pick(row: Mapping[str, Any], lookup: Mapping[str, str], aliases: Iterable[str]) -> str:
    for alias in aliases:
        original = lookup.get(alias.casefold())
        if original is None:
            continue
        value = _clean(row.get(original))
        if value:
            return value
    return ""


def _csv_reader(text: str, *, source_name: str) -> tuple[csv.DictReader, dict[str, str]]:
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff"), newline=""))
    if not reader.fieldnames:
        raise ManifestNormalizationError("manifest has no header", source_path=source_name)
    fieldnames = [_clean(field) for field in reader.fieldnames]
    if any(not field for field in fieldnames):
        raise ManifestNormalizationError("manifest has an empty header field", source_path=source_name)
    if len({field.casefold() for field in fieldnames}) != len(fieldnames):
        raise ManifestNormalizationError("manifest has duplicate header fields", source_path=source_name)
    reader.fieldnames = fieldnames
    return reader, _field_lookup(fieldnames)


def _normalize_image_path(value: str, *, row_number: int | None = None, source_name: str = "manifest.csv") -> str:
    raw = _clean(value).replace("\\", "/")
    if not raw:
        raise ManifestNormalizationError("missing image field", source_path=source_name, row_number=row_number)
    path = PurePosixPath(raw)
    if raw.startswith("/") or any(part in {"..", "."} for part in path.parts):
        raise ManifestNormalizationError("invalid image path", source_path=source_name, row_number=row_number)
    normalized = str(path)
    if len(path.parts) == 1:
        normalized = f"images/{normalized}"
    return normalized


def _iter_rows(reader: csv.DictReader, lookup: Mapping[str, str], *, source_name: str):
    seen_rows = 0
    for row_number, row in enumerate(reader, start=2):
        if None in row:
            raise ManifestNormalizationError("malformed CSV row", source_path=source_name, row_number=row_number)
        seen_rows += 1
        image_id = _pick(row, lookup, ("image_id",))
        if not image_id:
            raise ManifestNormalizationError("missing image_id field", source_path=source_name, row_number=row_number)
        image_value = _pick(row, lookup, IMAGE_FIELD_ALIASES)
        image_path = _normalize_image_path(image_value, row_number=row_number, source_name=source_name)
        claimed_species = _pick(row, lookup, SPECIES_FIELD_ALIASES)
        if not claimed_species:
            raise ManifestNormalizationError("missing species field", source_path=source_name, row_number=row_number)
        species_key = _pick(row, lookup, SPECIES_KEY_ALIASES)
        source = _pick(row, lookup, SOURCE_FIELD_ALIASES) or "unknown"
        yield {
            "image_path": image_path,
            "image_id": image_id,
            "claimed_species": claimed_species,
            "species_key": species_key,
            "source": source,
        }
    if seen_rows == 0:
        raise ManifestNormalizationError("manifest is empty", source_path=source_name)


def _render(rows: Iterable[Mapping[str, str]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(OUTPUT_FIELDS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def normalize_manifest_text(source_text: str, *, source_name: str = "metadata/manifest.csv") -> tuple[str, int]:
    """Convert a Data Asset Manifest into the fixed Training Manifest contract."""

    reader, lookup = _csv_reader(source_text, source_name=source_name)
    rows = list(_iter_rows(reader, lookup, source_name=source_name))
    return _render(rows), len(rows)


def validate_fish_manifest_text(source_text: str, *, source_name: str = "metadata/fish_manifest.csv") -> int:
    """Validate an existing Training Manifest without rewriting it."""

    reader, lookup = _csv_reader(source_text, source_name=source_name)
    rows = list(reader)
    if not rows:
        raise ManifestNormalizationError("manifest is empty", source_path=source_name)
    for row_number, row in enumerate(rows, start=2):
        if None in row:
            raise ManifestNormalizationError("malformed CSV row", source_path=source_name, row_number=row_number)
        if not _pick(row, lookup, ("image_path", "file_name", "filename", "image_name")):
            raise ManifestNormalizationError("missing image field", source_path=source_name, row_number=row_number)
        if not _pick(row, lookup, ("image_id",)):
            raise ManifestNormalizationError("missing image_id field", source_path=source_name, row_number=row_number)
        if not _pick(row, lookup, ("claimed_species",)):
            raise ManifestNormalizationError("missing species field", source_path=source_name, row_number=row_number)
    return len(rows)


def _write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with path.open("x", encoding="utf-8", newline="") as handle:
            created = True
            handle.write(text)
    except Exception:
        if created and path.exists():
            path.unlink()
        raise


def _read_manifest_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ManifestNormalizationError("manifest is not valid UTF-8", source_path=path) from exc


def _existing_fish_manifest(root: Path, canonical: Path) -> Path | None:
    if canonical.exists():
        return canonical
    candidates = sorted(path for path in root.rglob("fish_manifest.csv") if path.is_file())
    if len(candidates) > 1:
        raise ManifestNormalizationError(f"multiple fish_manifest.csv files found: {len(candidates)}", source_path=root)
    return candidates[0] if candidates else None


def normalize_manifest(batch_root: str | Path) -> ManifestNormalizationResult:
    """Ensure ``batch_root/metadata/fish_manifest.csv`` exists and is valid.

    Existing manifests are validated in place and never overwritten. A single legacy
    fish_manifest.csv outside metadata is accepted for backward compatibility; new
    conversions always materialize under metadata/.
    """

    root = Path(batch_root)
    if not root.exists() or not root.is_dir():
        raise ManifestNormalizationError("batch root does not exist", source_path=root)
    metadata = root / "metadata"
    canonical = metadata / "fish_manifest.csv"
    existing = _existing_fish_manifest(root, canonical)
    if existing is not None:
        count = validate_fish_manifest_text(_read_manifest_file(existing), source_name=str(existing))
        return ManifestNormalizationResult(output_path=existing, source_path=existing, rows=count, generated=False)

    source = metadata / "manifest.csv"
    if not source.exists():
        legacy_source = root / "manifest.csv"
        source = legacy_source if legacy_source.exists() else source
    if not source.exists() or not source.is_file():
        raise ManifestNormalizationError("missing metadata/manifest.csv", source_path=source)

    normalized, count = normalize_manifest_text(_read_manifest_file(source), source_name=str(source))
    try:
        _write_new(canonical, normalized)
    except FileExistsError:
        count = validate_fish_manifest_text(_read_manifest_file(canonical), source_name=str(canonical))
        return ManifestNormalizationResult(output_path=canonical, source_path=canonical, rows=count, generated=False)
    return ManifestNormalizationResult(output_path=canonical, source_path=source, rows=count, generated=True)


class ManifestNormalizer:
    """Small OO facade for callers that prefer an injectable service object."""

    def normalize(self, batch_root: str | Path) -> ManifestNormalizationResult:
        return normalize_manifest(batch_root)


normalize_batch_manifest = normalize_manifest


__all__ = [
    "ManifestNormalizationError",
    "ManifestNormalizationResult",
    "ManifestNormalizer",
    "OUTPUT_FIELDS",
    "normalize_batch_manifest",
    "normalize_manifest",
    "normalize_manifest_text",
    "validate_fish_manifest_text",
]
