from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from google.cloud import storage
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from sqlalchemy import CheckConstraint, Column, DateTime, ForeignKey, Index, Integer, String, Text, select
from sqlalchemy.orm import Session, relationship

from app.db import Base, get_db
from app.factory import get_bucket_name
from app.fish_knowledge.asset_types import asset_direction, normalize_asset_type, upsert_fish_asset_index
from app.fish_knowledge.cards import CARD_TYPE_ORDER, FishCard, normalize_card_type
from app.fish_knowledge.cover import FishSpeciesCover
from app.fish_knowledge.gallery import GalleryUploadError, inspect_knowledge_asset
from app.fish_knowledge.species import FishSpecies, SPECIES_ID_ALIASES
from app.models import SpeciesCatalog, utcnow


BATCH_STATUSES = ("CREATED", "SCANNING", "READY", "IMPORTING", "COMPLETED", "FAILED", "CANCELLED")
ITEM_STATUSES = ("VALID", "WARNING", "INVALID", "IMPORTED", "FAILED")
ASSET_TYPES = (
    "COVER",
    "COVER_CARD",
    "COVER_CARD_TRANSPARENT_LEFT",
    "COVER_CARD_TRANSPARENT_RIGHT",
    "HERO",
    "IDENTIFICATION",
    "ECO",
    "GEAR",
    "SKILL",
)
SOURCE_PREFIX = "fish-assets/imports/"
TARGET_ROOT = "fish-assets/fish-knowledge/"
ASSET_DIR = {
    "COVER": "cover",
    "COVER_CARD": "cover-card",
    "COVER_CARD_TRANSPARENT_LEFT": "cover-card/transparent-left",
    "COVER_CARD_TRANSPARENT_RIGHT": "cover-card/transparent-right",
    "HERO": "hero",
    "IDENTIFICATION": "identification",
    "ECO": "ecology",
    "GEAR": "gear",
    "SKILL": "skill",
}
EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
IGNORED_FILES = {"readme.txt", "asset_manifest.csv", "manifest.csv"}
ASSET_PATTERNS = (
    (re.compile(r"^00_cover_list(?:_.*)?$", re.I), "COVER_CARD"),
    (re.compile(r"^00_cover_card(?:_.*)?$", re.I), "COVER_CARD"),
    (re.compile(r"^00_cover(?:_.*)?$", re.I), "COVER"),
    (re.compile(r"^01_transparent_main(?:_.*)?$", re.I), "COVER_CARD_TRANSPARENT_LEFT"),
    (re.compile(r"^02_transparent_alt(?:_.*)?$", re.I), "COVER_CARD_TRANSPARENT_RIGHT"),
    (re.compile(r"^cover_card(?:_.*)?$", re.I), "COVER_CARD"),
    (re.compile(r"^cover_card_left(?:_.*)?$", re.I), "COVER_CARD_TRANSPARENT_LEFT"),
    (re.compile(r"^cover_card_right(?:_.*)?$", re.I), "COVER_CARD_TRANSPARENT_RIGHT"),
    (re.compile(r"^01_hero(?:_.*)?$", re.I), "HERO"),
    (re.compile(r"^02_identification(?:_.*)?$", re.I), "IDENTIFICATION"),
    (re.compile(r"^03_(?:ecology|eco)(?:_.*)?$", re.I), "ECO"),
    (re.compile(r"^04_gear(?:_.*)?$", re.I), "GEAR"),
    (re.compile(r"^05_(?:skill|fishing)(?:_.*)?$", re.I), "SKILL"),
)


class FishAssetImportBatch(Base):
    __tablename__ = "fish_asset_import_batches"
    __table_args__ = (CheckConstraint("status IN ('CREATED','SCANNING','READY','IMPORTING','COMPLETED','FAILED','CANCELLED')", name="ck_fish_asset_import_batch_status"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_id = Column(String(128), nullable=False, unique=True, index=True)
    source_gcs_uri = Column(Text, nullable=False)
    status = Column(String(16), nullable=False, default="CREATED", index=True)
    created_by = Column(String(256), nullable=False, default="admin")
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)
    total_files = Column(Integer, nullable=False, default=0)
    recognized_files = Column(Integer, nullable=False, default=0)
    valid_files = Column(Integer, nullable=False, default=0)
    warning_files = Column(Integer, nullable=False, default=0)
    failed_files = Column(Integer, nullable=False, default=0)
    species_count = Column(Integer, nullable=False, default=0)
    error_summary = Column(Text, nullable=False, default="{}")
    result_json = Column(Text, nullable=False, default="{}")

    items = relationship(
        "FishAssetImportItem",
        primaryjoin=lambda: FishAssetImportBatch.batch_id == FishAssetImportItem.batch_id,
        cascade="all, delete-orphan",
        order_by="FishAssetImportItem.id",
    )


class FishAssetImportItem(Base):
    __tablename__ = "fish_asset_import_items"
    __table_args__ = (
        CheckConstraint("asset_type IN ('COVER','COVER_CARD','COVER_CARD_TRANSPARENT_LEFT','COVER_CARD_TRANSPARENT_RIGHT','HERO','IDENTIFICATION','ECO','GEAR','SKILL')", name="ck_fish_asset_import_item_type"),
        CheckConstraint("validation_status IN ('VALID','WARNING','INVALID','IMPORTED','FAILED')", name="ck_fish_asset_import_item_status"),
        Index("ix_fish_asset_import_item_batch_species_type", "batch_id", "species_id", "asset_type"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_id = Column(String(128), ForeignKey("fish_asset_import_batches.batch_id", ondelete="CASCADE"), nullable=False, index=True)
    species_id = Column(String(128), ForeignKey("fish_species.id", ondelete="RESTRICT"), nullable=True, index=True)
    source_object = Column(Text, nullable=False)
    asset_type = Column(String(64), nullable=True)
    direction = Column(String(16))
    source_filename = Column(String(512), nullable=False)
    mime_type = Column(String(128))
    width = Column(Integer)
    height = Column(Integer)
    aspect_ratio = Column(Text)
    file_size = Column(Integer)
    sha256 = Column(String(64), index=True)
    validation_status = Column(String(16), nullable=False, default="INVALID", index=True)
    validation_errors = Column(Text, nullable=False, default="[]")
    validation_warnings = Column(Text, nullable=False, default="[]")
    target_object = Column(Text)
    version_id = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)


class FishKnowledgeAssetVersion(Base):
    __tablename__ = "fish_knowledge_asset_versions"
    __table_args__ = (
        CheckConstraint("asset_type IN ('COVER','COVER_CARD','COVER_CARD_TRANSPARENT_LEFT','COVER_CARD_TRANSPARENT_RIGHT','HERO','IDENTIFICATION','ECO','GEAR','SKILL')", name="ck_fish_knowledge_asset_version_type"),
        CheckConstraint("status IN ('DRAFT','ACTIVE','ARCHIVED')", name="ck_fish_knowledge_asset_version_status"),
        Index("uq_fish_knowledge_asset_version_slot", "species_id", "asset_type", "version", unique=True),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    species_id = Column(String(128), ForeignKey("fish_species.id", ondelete="CASCADE"), nullable=False, index=True)
    asset_type = Column(String(64), nullable=False)
    direction = Column(String(16))
    version = Column(Integer, nullable=False)
    object_name = Column(Text, nullable=False, unique=True)
    image_url = Column(Text, nullable=False, unique=True)
    status = Column(String(16), nullable=False, default="DRAFT", index=True)
    sha256 = Column(String(64), nullable=False, index=True)
    metadata_json = Column(Text, nullable=False, default="{}")
    batch_id = Column(String(128), ForeignKey("fish_asset_import_batches.batch_id", ondelete="SET NULL"), nullable=True, index=True)
    item_id = Column(Integer, nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)


class CreateBatchPayload(BaseModel):
    source_gcs_uri: str = Field(min_length=1, max_length=2048)


class CreateLocalBatchPayload(BaseModel):
    batch_id: str = Field(min_length=3, max_length=128)


def _normalize_upload_path(value: str) -> str:
    raw = (value or "").replace("\\", "/").strip()
    if not raw or raw.startswith("/") or re.match(r"^[A-Za-z]:", raw):
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_UPLOAD_PATH", "message": "relative_path 必须是本地文件夹内的相对路径"},
        )
    parts = raw.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_UPLOAD_PATH", "message": "relative_path 不允许包含空目录、. 或 .."},
        )
    return "/".join(parts)


class ExecuteBatchPayload(BaseModel):
    allow_warnings: bool = False


router = APIRouter(prefix="/api/v1/admin/fish/assets/import-batches", tags=["fish-knowledge-asset-import"])
page_router = APIRouter(tags=["fish-knowledge-asset-import"])


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _read_json(value: str | None, default: Any) -> Any:
    try:
        return json.loads(value or "")
    except (TypeError, ValueError):
        return default


def _source_parts(uri: str) -> tuple[str, str, str]:
    value = uri.strip()
    if not value.startswith("gs://"):
        raise HTTPException(status_code=400, detail={"code": "INVALID_SOURCE_URI", "message": "source_gcs_uri 必须是 gs:// URI"})
    raw = value[5:]
    bucket, separator, prefix = raw.partition("/")
    if not bucket or not separator or not prefix:
        raise HTTPException(status_code=400, detail={"code": "INVALID_SOURCE_URI", "message": "source_gcs_uri 必须包含 bucket 和目录"})
    expected_bucket = get_bucket_name()
    if bucket != expected_bucket:
        raise HTTPException(status_code=400, detail={"code": "SOURCE_BUCKET_NOT_ALLOWED", "message": f"只允许使用 gs://{expected_bucket}/"})
    prefix = prefix.strip("/") + "/"
    if not prefix.startswith(SOURCE_PREFIX) or prefix == SOURCE_PREFIX:
        raise HTTPException(status_code=400, detail={"code": "SOURCE_PREFIX_NOT_ALLOWED", "message": f"导入源必须位于 gs://{expected_bucket}/{SOURCE_PREFIX}"})
    batch_id = prefix.rstrip("/").split("/")[-1]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{2,127}", batch_id):
        raise HTTPException(status_code=400, detail={"code": "INVALID_BATCH_ID", "message": "导入目录最后一级必须是合法 batch_id"})
    return bucket, prefix, batch_id


def _parse_source_uri(uri: str) -> tuple[str, str, str]:
    return _source_parts(uri)


def _folder_candidates(folder: str) -> list[str]:
    raw = folder.strip()
    candidates = [raw, re.sub(r"^\d+[_-]", "", raw)]
    return [item.strip().lower() for item in candidates if item.strip()]


def _resolve_species(db: Session, folder: str) -> FishSpecies | None:
    candidates = _folder_candidates(folder)
    rows = db.scalars(select(FishSpecies).where(FishSpecies.status != "DELETED")).all()
    for row in rows:
        values = {row.id.lower(), row.name_cn.strip().lower()}
        values.update(str(alias).strip().lower() for alias in (row.alias or []) if str(alias).strip())
        if row.id.lower() == "sharpbelly":
            values.update({"baitiao", "白条"})
        if any(candidate in values for candidate in candidates):
            return row
    for alias, canonical in SPECIES_ID_ALIASES.items():
        if alias in candidates:
            return db.get(FishSpecies, canonical)
    catalog = db.scalars(select(SpeciesCatalog)).all()
    for row in catalog:
        if row.species_key.lower() in candidates or row.common_name_zh.strip().lower() in candidates:
            return db.get(FishSpecies, row.species_key)
    return None


def _resolve_species_name(db: Session, folder: str) -> str | None:
    row = _resolve_species(db, folder)
    return row.id if row else None


def _asset_type_for_filename(filename: str) -> str | None:
    stem = filename.rsplit(".", 1)[0]
    for pattern, asset_type in ASSET_PATTERNS:
        if pattern.fullmatch(stem):
            return asset_type
    return None


def _asset_direction_for_filename(filename: str, asset_type: str | None) -> str:
    return asset_direction(asset_type)


def _manifest_entries(blobs: list[Any], prefix: str) -> dict[str, dict[str, str]]:
    """Read optional Manifest V2 rows keyed by relative path and basename."""

    manifest_blob = next(
        (
            blob for blob in blobs
            if blob.name.startswith(prefix)
            and blob.name.rsplit("/", 1)[-1].lower() in {"asset_manifest.csv", "manifest.csv"}
        ),
        None,
    )
    if manifest_blob is None:
        return {}
    raw = manifest_blob.download_as_bytes(timeout=120).decode("utf-8-sig")
    entries: dict[str, dict[str, str]] = {}
    for row in csv.DictReader(io.StringIO(raw)):
        normalized = {
            str(key or "").strip().lower(): str(value or "").strip()
            for key, value in row.items()
        }
        file_name = normalized.get("file_name") or normalized.get("file") or normalized.get("filename")
        if not file_name:
            continue
        value = {
            "species_id": normalized.get("species_id", ""),
            "species_name": normalized.get("species_name", ""),
            "file_name": file_name.replace("\\", "/").lstrip("/"),
            "asset_type": normalized.get("asset_type", ""),
            "direction": normalized.get("direction", ""),
        }
        keys = {value["file_name"].lower(), value["file_name"].rsplit("/", 1)[-1].lower()}
        for key in keys:
            if key in entries:
                entries[f"{key}#manifest{len(entries)}"] = value
            else:
                entries[key] = value
    return entries


def _manifest_entry(entries: dict[str, dict[str, str]], relative: str, filename: str) -> dict[str, str] | None:
    exact = entries.get(relative.lower())
    if exact is not None:
        return exact
    candidates = [
        value for value in entries.values()
        if value.get("file_name", "").rsplit("/", 1)[-1].lower() == filename.lower()
    ]
    if len(candidates) <= 1:
        return candidates[0] if candidates else None
    folder_hint = relative.rsplit("/", 1)[0].lower()
    for value in candidates:
        if any(
            hint and hint.lower() in folder_hint
            for hint in (value.get("species_id"), value.get("species_name"))
        ):
            return value
    return candidates[0]


def _image_extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def _validation(error_code: str, message: str) -> dict[str, str]:
    return {"code": error_code, "message": message}


def _next_version(db: Session, client: Any, bucket: Any, species_id: str, asset_type: str) -> int:
    current = db.scalar(
        select(FishKnowledgeAssetVersion.version)
        .where(
            FishKnowledgeAssetVersion.species_id == species_id,
            FishKnowledgeAssetVersion.asset_type == asset_type,
        )
        .order_by(FishKnowledgeAssetVersion.version.desc())
    ) or 0
    prefix = f"{TARGET_ROOT}{species_id}/{ASSET_DIR[asset_type]}/"
    try:
        for blob in client.list_blobs(bucket, prefix=prefix):
            match = re.fullmatch(rf"{re.escape(prefix)}v(\d+)\.webp", blob.name)
            if match:
                current = max(current, int(match.group(1)))
    except Exception:
        # Object listing failure is surfaced as a scan error by the caller.
        raise
    return int(current) + 1


def _existing_duplicate(db: Session, species_id: str, asset_type: str, sha256: str) -> FishKnowledgeAssetVersion | None:
    return db.scalar(
        select(FishKnowledgeAssetVersion)
        .where(
            FishKnowledgeAssetVersion.species_id == species_id,
            FishKnowledgeAssetVersion.asset_type == asset_type,
            FishKnowledgeAssetVersion.sha256 == sha256,
        )
        .order_by(FishKnowledgeAssetVersion.id.desc())
    )


def _item_dict(item: FishAssetImportItem, *, base: str) -> dict[str, Any]:
    errors = _read_json(item.validation_errors, [])
    warnings = _read_json(item.validation_warnings, [])
    return {
        "id": item.id,
        "species_id": item.species_id,
        "asset_type": item.asset_type,
        "direction": item.direction or asset_direction(item.asset_type),
        "source_object": item.source_object,
        "source_filename": item.source_filename,
        "mime_type": item.mime_type,
        "width": item.width,
        "height": item.height,
        "aspect_ratio": float(item.aspect_ratio) if item.aspect_ratio else None,
        "file_size": item.file_size,
        "sha256": item.sha256,
        "validation_status": item.validation_status,
        "validation_errors": errors,
        "validation_warnings": warnings,
        "target_object": item.target_object,
        "version_id": item.version_id,
        "source_preview_url": f"{base}/items/{item.id}/source",
    }


def _summary(batch: FishAssetImportBatch) -> dict[str, int]:
    return {
        "species_count": batch.species_count,
        "total_files": batch.total_files,
        "recognized_files": batch.recognized_files,
        "valid": batch.valid_files,
        "warning": batch.warning_files,
        "invalid": batch.failed_files,
    }


def _batch_dict(batch: FishAssetImportBatch, *, include_items: bool = False) -> dict[str, Any]:
    items = [_item_dict(item, base=f"/api/v1/admin/fish/assets/import-batches/{batch.batch_id}") for item in batch.items]
    by_species: dict[str, dict[str, Any]] = {}
    for item in items:
        species_id = item.get("species_id") or "UNRESOLVED"
        row = by_species.setdefault(species_id, {"species_id": species_id, "assets": {}, "completion": 0})
        key = item.get("asset_type") or f"INVALID_{item['id']}"
        row["assets"].setdefault(key, []).append(item)
    for row in by_species.values():
        row["completion"] = f"{sum(1 for values in row['assets'].values() if any(x['validation_status'] in {'VALID','WARNING','IMPORTED'} for x in values))}/8"
    payload = {
        "batch_id": batch.batch_id,
        "source_gcs_uri": batch.source_gcs_uri,
        "status": batch.status,
        "created_by": batch.created_by,
        "created_at": batch.created_at.isoformat() if batch.created_at else None,
        "updated_at": batch.updated_at.isoformat() if batch.updated_at else None,
        "summary": _summary(batch),
        "error_summary": _read_json(batch.error_summary, {}),
        "result": _read_json(batch.result_json, {}),
        "species": list(by_species.values()),
    }
    if include_items:
        payload["items"] = items
    return payload


def _batch_or_404(db: Session, batch_id: str) -> FishAssetImportBatch:
    row = db.scalar(select(FishAssetImportBatch).where(FishAssetImportBatch.batch_id == batch_id))
    if row is None:
        raise HTTPException(status_code=404, detail={"code": "BATCH_NOT_FOUND", "message": "导入批次不存在"})
    return row


def _storage(batch: FishAssetImportBatch) -> tuple[Any, Any, str]:
    bucket_name, prefix, _ = _source_parts(batch.source_gcs_uri)
    client = storage.Client()
    return client, client.bucket(bucket_name), prefix


def _commit(db: Session) -> None:
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail={"code": "DB_WRITE_FAILED", "message": str(exc)}) from exc


def _scan_item(
    db: Session,
    client: Any,
    bucket: Any,
    batch: FishAssetImportBatch,
    name: str,
    *,
    manifest_entries: dict[str, dict[str, str]] | None = None,
) -> FishAssetImportItem | None:
    relative = name[len(_source_parts(batch.source_gcs_uri)[1]):].lstrip("/")
    parts = relative.split("/")
    filename = parts[-1]
    if not filename or name.endswith("/"):
        return None
    lowered = filename.lower()
    if lowered in IGNORED_FILES:
        return None

    manifest = _manifest_entry(manifest_entries or {}, relative, filename)
    folder = parts[0] if parts else ""
    for candidate in parts[:-1]:
        if _resolve_species(db, candidate) is not None:
            folder = candidate
            break
    if manifest:
        manifest_folder = manifest.get("species_id") or manifest.get("species_name") or ""
        if manifest_folder:
            folder = manifest_folder
    species = None
    if manifest:
        for candidate in (manifest.get("species_id", ""), manifest.get("species_name", ""), folder):
            if candidate and (species := _resolve_species(db, candidate)) is not None:
                break
    else:
        species = _resolve_species(db, folder) if folder else None

    filename_asset_type = _asset_type_for_filename(filename)
    raw_manifest_type = manifest.get("asset_type", "") if manifest else ""
    asset_type = (
        normalize_asset_type(raw_manifest_type, manifest.get("direction"))
        if raw_manifest_type
        else filename_asset_type
    )
    direction = (
        asset_direction(asset_type, manifest.get("direction"))
        if asset_type
        else (manifest.get("direction") or "NONE")
    )
    suffix = "." + _image_extension(filename) if "." in filename else ""
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    if species is None:
        errors.append(_validation("SPECIES_NOT_FOUND", f"Unknown species folder: {folder or relative}"))
    if asset_type is None:
        errors.append(_validation("UNKNOWN_ASSET_TYPE", f"Unknown asset filename: {filename}"))
    if suffix not in EXTENSIONS:
        errors.append(_validation("UNSUPPORTED_IMAGE_FORMAT", f"Only PNG, JPEG and WEBP are supported: {filename}"))
    blob = bucket.blob(name)
    file_size = int(blob.size or 0)
    if not file_size:
        try:
            blob.reload(client)
            file_size = int(blob.size or 0)
        except Exception as exc:
            errors.append(_validation("GCS_READ_FAILED", f"Cannot read object metadata: {exc}"))
    metadata: dict[str, Any] = {}
    if file_size > 10 * 1024 * 1024:
        errors.append(_validation("FILE_TOO_LARGE", f"{filename} exceeds the 10 MB limit"))
    if not errors or all(error["code"] not in {"UNSUPPORTED_IMAGE_FORMAT", "FILE_TOO_LARGE", "SPECIES_NOT_FOUND", "UNKNOWN_ASSET_TYPE"} for error in errors):
        try:
            data = blob.download_as_bytes(timeout=120)
            metadata = inspect_knowledge_asset(data)
            actual_ext = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}[str(metadata["original_content_type"])]
            if suffix not in {actual_ext, ".jpeg"}:
                warnings.append(_validation("MIME_EXTENSION_MISMATCH", f"{filename} content is {actual_ext}, extension is {suffix or 'missing'}"))
        except (GalleryUploadError, UnidentifiedImageError, OSError, KeyError) as exc:
            errors.append(_validation("IMAGE_DECODE_FAILED", f"{filename}: {exc}"))
        except Exception as exc:
            errors.append(_validation("GCS_READ_FAILED", f"{filename}: {exc}"))
    width = int(metadata.get("width") or 0)
    height = int(metadata.get("height") or 0)
    ratio = (width / height) if width and height else None
    if ratio is not None:
        if abs(ratio - 1.0) > 0.05:
            errors.append(_validation("INVALID_ASPECT_RATIO", f"{filename} = {width}x{height}"))
        elif abs(ratio - 1.0) > 0.02:
            warnings.append(_validation("ASPECT_RATIO_WARNING", f"{filename} = {width}x{height}"))
        if min(width, height) < 512:
            errors.append(_validation("RESOLUTION_TOO_LOW", f"{filename} = {width}x{height}; minimum is 512px"))
        elif min(width, height) < 1024:
            warnings.append(_validation("RESOLUTION_WARNING", f"{filename} = {width}x{height}; recommended minimum is 1024px"))
    digest = str(metadata.get("sha256") or "")
    if species is not None and asset_type is not None and digest:
        duplicate = _existing_duplicate(db, species.id, asset_type, digest)
        if duplicate is not None:
            warnings.append(_validation("ASSET_ALREADY_EXISTS", f"Same SHA-256 already exists as {duplicate.object_name}"))
            target_object = duplicate.object_name
        else:
            target_object = None
    else:
        target_object = None
    status = "INVALID" if errors else ("WARNING" if warnings else "VALID")
    return FishAssetImportItem(
        batch_id=batch.batch_id,
        species_id=species.id if species else None,
        source_object=name,
        asset_type=asset_type,
        direction=direction,
        source_filename=filename,
        mime_type=str(metadata.get("original_content_type") or ""),
        width=width or None,
        height=height or None,
        aspect_ratio=str(round(ratio, 6)) if ratio is not None else None,
        file_size=file_size or None,
        sha256=digest or None,
        validation_status=status,
        validation_errors=_json(errors),
        validation_warnings=_json(warnings),
        target_object=target_object,
    )

def _mark_duplicate_slots(items: list[FishAssetImportItem]) -> None:
    slots: dict[tuple[str | None, str | None], list[FishAssetImportItem]] = {}
    for item in items:
        key = (item.species_id, item.asset_type)
        if item.species_id and item.asset_type:
            slots.setdefault(key, []).append(item)
    for values in slots.values():
        if len(values) < 2:
            continue
        for item in values:
            errors = _read_json(item.validation_errors, [])
            if not any(error.get("code") == "DUPLICATE_ASSET_SLOT" for error in errors):
                errors.append(_validation("DUPLICATE_ASSET_SLOT", f"{item.species_id} {item.asset_type} has {len(values)} files"))
            item.validation_errors = _json(errors)
            item.validation_status = "INVALID"


def _assign_targets(db: Session, client: Any, bucket: Any, items: list[FishAssetImportItem]) -> None:
    allocated: set[tuple[str, str]] = set()
    for item in items:
        if item.validation_status == "INVALID" or not item.species_id or not item.asset_type:
            continue
        if item.target_object:
            continue
        key = (item.species_id, item.asset_type)
        if key not in allocated:
            version = _next_version(db, client, bucket, item.species_id, item.asset_type)
            allocated.add(key)
        else:
            continue
        item.target_object = f"{TARGET_ROOT}{item.species_id}/{ASSET_DIR[item.asset_type]}/v{version}.webp"


def _counts(batch: FishAssetImportBatch, items: list[FishAssetImportItem]) -> None:
    batch.total_files = len(items)
    batch.recognized_files = sum(1 for item in items if item.asset_type is not None)
    batch.valid_files = sum(1 for item in items if item.validation_status == "VALID")
    batch.warning_files = sum(1 for item in items if item.validation_status == "WARNING")
    batch.failed_files = sum(1 for item in items if item.validation_status == "INVALID")
    batch.species_count = len({item.species_id for item in items if item.species_id})
    batch.error_summary = _json({
        code: sum(1 for item in items if any(error.get("code") == code for error in _read_json(item.validation_errors, [])))
        for code in {error.get("code") for item in items for error in _read_json(item.validation_errors, []) if error.get("code")}
    })


def _version_url(species_id: str, asset_type: str, version: int) -> str:
    return f"/api/v1/fish/knowledge-media/{species_id}/{asset_type.lower()}/v{version}.webp"


def _next_version_row(db: Session, species_id: str, asset_type: str) -> int:
    return int(db.scalar(select(FishKnowledgeAssetVersion.version).where(
        FishKnowledgeAssetVersion.species_id == species_id,
        FishKnowledgeAssetVersion.asset_type == asset_type,
    ).order_by(FishKnowledgeAssetVersion.version.desc())) or 0) + 1


def _bind_imported_version(db: Session, version: FishKnowledgeAssetVersion) -> str:
    """Bind an imported version while indexing the same slot in fish_asset."""

    species = db.get(FishSpecies, version.species_id)
    if species is None:
        raise RuntimeError(f"species {version.species_id} not found while binding imported asset")

    indexed_type = "COVER_CARD" if version.asset_type == "COVER" else version.asset_type
    upsert_fish_asset_index(
        db,
        species_id=species.id,
        asset_type=indexed_type,
        direction=version.direction,
        url=version.image_url,
        object_name=version.object_name,
        status=version.status,
        version=f"v{version.version}",
        source_batch_id=version.batch_id,
    )

    if version.asset_type in {"COVER", "COVER_CARD"}:
        current = db.scalar(select(FishSpeciesCover).where(FishSpeciesCover.species_id == species.id))
        if current is None:
            db.add(
                FishSpeciesCover(
                    species_id=species.id,
                    image_url=version.image_url,
                    style="ANIME_CARD",
                    title=f"{species.name_cn}图鉴卡",
                    status="DRAFT",
                )
            )
            return "BOUND"
        if current.status == "ACTIVE" and (current.image_url or "").strip():
            return "ACTIVE_PRESERVED"
        current.image_url = version.image_url
        current.status = "DRAFT"
        if not (current.title or "").strip():
            current.title = f"{species.name_cn}图鉴卡"
        return "BOUND"

    if version.asset_type in {"COVER_CARD_TRANSPARENT_LEFT", "COVER_CARD_TRANSPARENT_RIGHT"}:
        return "BOUND"

    rows = db.scalars(
        select(FishCard)
        .where(FishCard.species_id == species.id)
        .order_by(FishCard.sort_order, FishCard.id)
    ).all()
    card_type = normalize_card_type(version.asset_type)
    candidate = next(
        (
            row for row in rows
            if normalize_card_type(row.card_type) == card_type
            and row.status == "DRAFT"
            and not (row.image_url or "").strip()
        ),
        None,
    )
    active = next(
        (
            row for row in rows
            if normalize_card_type(row.card_type) == card_type
            and row.status == "ACTIVE"
        ),
        None,
    )
    if candidate is None:
        candidate = FishCard(
            species_id=species.id,
            card_type=card_type,
            title=(active.title if active else f"{species.name_cn}{card_type}卡"),
            image_url=version.image_url,
            description=(active.description if active else ""),
            sort_order=CARD_TYPE_ORDER.index(card_type),
            status="DRAFT",
        )
        db.add(candidate)
    else:
        candidate.image_url = version.image_url
        candidate.status = "DRAFT"
    return "BOUND"

def _import_item(db: Session, client: Any, bucket: Any, batch: FishAssetImportBatch, item: FishAssetImportItem) -> str:
    if item.validation_status == "IMPORTED":
        return "SKIP_IMPORTED"
    warnings = _read_json(item.validation_warnings, [])
    if any(warning.get("code") == "ASSET_ALREADY_EXISTS" for warning in warnings):
        return "SKIP_DUPLICATE"
    if item.validation_status == "INVALID":
        item.validation_status = "FAILED"
        return "SKIP_INVALID"
    if not item.target_object or not item.species_id or not item.asset_type:
        item.validation_status = "FAILED"
        item.validation_errors = _json([_validation("TARGET_NOT_READY", "Validated item has no target object")])
        return "FAILED"
    try:
        blob = bucket.blob(item.source_object)
        data = blob.download_as_bytes(timeout=120)
        metadata = inspect_knowledge_asset(data)
        stored = bytes(metadata["webp_data"])
        target = bucket.blob(item.target_object)
        if target.exists(client):
            existing = target.download_as_bytes(timeout=120)
            if hashlib.sha256(existing).hexdigest() != hashlib.sha256(stored).hexdigest():
                raise RuntimeError("target object exists with different content")
        else:
            target.metadata = {
                "source_sha256": str(item.sha256 or ""),
                "source_object": item.source_object,
                "batch_id": batch.batch_id,
                "asset_type": item.asset_type,
                "direction": item.direction or "NONE",
            }
            target.upload_from_string(stored, content_type="image/webp", if_generation_match=0)
        version = _next_version_row(db, item.species_id, item.asset_type)
        existing_version = db.scalar(select(FishKnowledgeAssetVersion).where(FishKnowledgeAssetVersion.object_name == item.target_object))
        if existing_version is None:
            version_row = FishKnowledgeAssetVersion(
                species_id=item.species_id,
                asset_type=item.asset_type,
                direction=item.direction,
                version=version,
                object_name=item.target_object,
                image_url=_version_url(item.species_id, item.asset_type, version),
                status="DRAFT",
                sha256=str(item.sha256),
                metadata_json=_json({
                    "width": metadata.get("width"),
                    "height": metadata.get("height"),
                    "original_content_type": metadata.get("original_content_type"),
                    "stored_size_bytes": metadata.get("stored_size_bytes"),
                }),
                batch_id=batch.batch_id,
                item_id=item.id,
            )
            db.add(version_row)
            db.flush()
            item.version_id = version_row.id
        else:
            version_row = existing_version
            if not version_row.direction:
                version_row.direction = item.direction
            item.version_id = existing_version.id

        binding = _bind_imported_version(db, version_row)
        item.validation_status = "IMPORTED"
        return "IMPORTED_ACTIVE_PRESERVED" if binding == "ACTIVE_PRESERVED" else "IMPORTED"
    except Exception as exc:
        item.validation_status = "FAILED"
        item.validation_errors = _json([_validation("IMPORT_FAILED", f"{item.source_object}: {exc}")])
        return "FAILED"


def _run_import(db: Session, batch: FishAssetImportBatch, *, retry_failed: bool = False) -> dict[str, int]:
    client, bucket, _ = _storage(batch)
    items = db.scalars(select(FishAssetImportItem).where(FishAssetImportItem.batch_id == batch.batch_id).order_by(FishAssetImportItem.id)).all()
    totals = {
        "imported": 0,
        "skipped_duplicate": 0,
        "skipped_imported": 0,
        "active_preserved": 0,
        "failed": 0,
    }
    for item in items:
        if retry_failed and item.validation_status != "FAILED":
            continue
        result = _import_item(db, client, bucket, batch, item)
        totals[{
            "SKIP_DUPLICATE": "skipped_duplicate",
            "SKIP_IMPORTED": "skipped_imported",
            "IMPORTED": "imported",
            "IMPORTED_ACTIVE_PRESERVED": "active_preserved",
            "FAILED": "failed",
            "SKIP_INVALID": "failed",
        }.get(result, "failed")] += 1
        db.commit()
    batch.result_json = _json(totals)
    batch.failed_files = totals["failed"]
    batch.status = "FAILED" if totals["failed"] else "COMPLETED"
    db.commit()
    return totals


def _build_detail_items(db: Session, batch: FishAssetImportBatch) -> list[FishAssetImportItem]:
    items = db.scalars(select(FishAssetImportItem).where(FishAssetImportItem.batch_id == batch.batch_id).order_by(FishAssetImportItem.id)).all()
    return items


@router.post("", status_code=201)
def create_batch(payload: CreateBatchPayload, db: Session = Depends(get_db)) -> dict[str, Any]:
    _, _, batch_id = _source_parts(payload.source_gcs_uri)
    existing = db.scalar(select(FishAssetImportBatch).where(FishAssetImportBatch.batch_id == batch_id))
    if existing is not None:
        raise HTTPException(status_code=409, detail={"code": "BATCH_EXISTS", "message": "batch_id 已存在，请使用新的导入目录"})
    row = FishAssetImportBatch(batch_id=batch_id, source_gcs_uri=payload.source_gcs_uri.strip(), status="CREATED", created_by="admin")
    db.add(row)
    _commit(db)
    return {"batch_id": row.batch_id, "status": row.status, "source_gcs_uri": row.source_gcs_uri}


@router.post("/local", status_code=201)
def create_local_batch(payload: CreateLocalBatchPayload, db: Session = Depends(get_db)) -> dict[str, Any]:
    batch_id = payload.batch_id.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{2,127}", batch_id):
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_BATCH_ID", "message": "batch_id 只能包含字母、数字、下划线和连字符，长度 3-128"},
        )
    source_gcs_uri = f"gs://{get_bucket_name()}/{SOURCE_PREFIX}{batch_id}/"
    existing = db.scalar(select(FishAssetImportBatch).where(FishAssetImportBatch.batch_id == batch_id))
    if existing is not None:
        raise HTTPException(status_code=409, detail={"code": "BATCH_EXISTS", "message": "batch_id 已存在，请使用新的导入批次 ID"})
    row = FishAssetImportBatch(batch_id=batch_id, source_gcs_uri=source_gcs_uri, status="CREATED", created_by="admin")
    db.add(row)
    _commit(db)
    return {"batch_id": row.batch_id, "status": row.status, "source_gcs_uri": row.source_gcs_uri}


@router.post("/{batch_id}/upload")
async def upload_batch_file(
    batch_id: str,
    relative_path: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    batch = _batch_or_404(db, batch_id)
    if batch.status != "CREATED":
        raise HTTPException(status_code=409, detail={"code": "BATCH_NOT_UPLOADABLE", "message": "只有 CREATED 批次可以继续上传文件"})
    normalized_path = _normalize_upload_path(relative_path)
    size = getattr(file, "size", None)
    if size is not None and size > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail={"code": "FILE_TOO_LARGE", "message": "单个图片文件不能超过 10 MB"})
    try:
        client, bucket, prefix = _storage(batch)
        blob = bucket.blob(prefix + normalized_path)
        blob.upload_from_file(
            file.file,
            content_type=file.content_type or "application/octet-stream",
            rewind=True,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail={"code": "GCS_UPLOAD_FAILED", "message": str(exc)}) from exc
    finally:
        await file.close()
    return {
        "batch_id": batch.batch_id,
        "relative_path": normalized_path,
        "source_object": prefix + normalized_path,
        "size": size,
    }


@router.post("/{batch_id}/scan")
def scan_batch(batch_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    batch = _batch_or_404(db, batch_id)
    if batch.status in {"IMPORTING"}:
        raise HTTPException(status_code=409, detail={"code": "BATCH_BUSY", "message": "批次正在导入"})
    client, bucket, prefix = _storage(batch)
    batch.status = "SCANNING"
    db.query(FishAssetImportItem).filter(FishAssetImportItem.batch_id == batch.batch_id).delete(synchronize_session=False)
    db.commit()
    items: list[FishAssetImportItem] = []
    try:
        blobs = list(client.list_blobs(bucket, prefix=prefix))
        manifest_entries = _manifest_entries(blobs, prefix)
        for blob in blobs:
            item = _scan_item(db, client, bucket, batch, blob.name, manifest_entries=manifest_entries)
            if item is not None:
                items.append(item)
        _mark_duplicate_slots(items)
        _assign_targets(db, client, bucket, items)
        db.add_all(items)
        _counts(batch, items)
        batch.status = "READY"
        batch.result_json = _json({"scanned_at": datetime.now(timezone.utc).isoformat()})
        db.commit()
        return {"batch_id": batch.batch_id, "status": batch.status, "summary": _summary(batch)}
    except Exception as exc:
        db.rollback()
        batch.status = "FAILED"
        batch.error_summary = _json({"SCAN_FAILED": str(exc)})
        db.commit()
        raise HTTPException(status_code=502, detail={"code": "SCAN_FAILED", "message": str(exc)}) from exc


@router.get("/{batch_id}")
def get_batch(batch_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    return _batch_dict(_batch_or_404(db, batch_id), include_items=True)


@router.get("")
def list_batches(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    rows = db.scalars(select(FishAssetImportBatch).order_by(FishAssetImportBatch.created_at.desc()).limit(100)).all()
    return [_batch_dict(row) for row in rows]


@router.get("/{batch_id}/items/{item_id}/source")
def preview_source(batch_id: str, item_id: int, db: Session = Depends(get_db)) -> Response:
    batch = _batch_or_404(db, batch_id)
    item = db.scalar(select(FishAssetImportItem).where(FishAssetImportItem.id == item_id, FishAssetImportItem.batch_id == batch.batch_id))
    if item is None:
        raise HTTPException(status_code=404, detail={"code": "ITEM_NOT_FOUND", "message": "导入项不存在"})
    try:
        client, bucket, _ = _storage(batch)
        blob = bucket.blob(item.source_object)
        data = blob.download_as_bytes(timeout=120)
    except Exception as exc:
        raise HTTPException(status_code=502, detail={"code": "GCS_READ_FAILED", "message": str(exc)}) from exc
    mime = item.mime_type or "application/octet-stream"
    return Response(content=data, media_type=mime, headers={"Cache-Control": "private, max-age=300", "X-Content-Type-Options": "nosniff"})


@router.post("/{batch_id}/execute")
def execute_batch(batch_id: str, payload: ExecuteBatchPayload, db: Session = Depends(get_db)) -> dict[str, Any]:
    batch = _batch_or_404(db, batch_id)
    if batch.status != "READY":
        raise HTTPException(status_code=409, detail={"code": "BATCH_NOT_READY", "message": "必须先完成 Scan 并处于 READY"})
    if batch.failed_files:
        raise HTTPException(status_code=409, detail={"code": "INVALID_ITEMS_PRESENT", "message": "存在 INVALID 图片，不能执行"})
    if batch.warning_files and not payload.allow_warnings:
        raise HTTPException(status_code=409, detail={"code": "WARNINGS_REQUIRE_CONFIRMATION", "message": "存在 WARNING 图片，请明确 allow_warnings=true"})
    batch.status = "IMPORTING"
    db.commit()
    result = _run_import(db, batch)
    payload = _batch_dict(batch, include_items=True)
    payload["result"] = result
    return payload


@router.post("/{batch_id}/retry")
def retry_batch(batch_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    batch = _batch_or_404(db, batch_id)
    if batch.status not in {"FAILED", "COMPLETED"}:
        raise HTTPException(status_code=409, detail={"code": "RETRY_NOT_ALLOWED", "message": "只有已执行批次可以重试"})
    failed = db.scalar(select(FishAssetImportItem.id).where(FishAssetImportItem.batch_id == batch.batch_id, FishAssetImportItem.validation_status == "FAILED"))
    if failed is None:
        return {"batch_id": batch.batch_id, "status": batch.status, "result": _read_json(batch.result_json, {})}
    batch.status = "IMPORTING"
    db.commit()
    result = _run_import(db, batch, retry_failed=True)
    payload = _batch_dict(batch, include_items=True)
    payload["result"] = result
    return payload


@router.post("/{batch_id}/sync-content")
def sync_content(batch_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Bind already imported DRAFT versions to the Fish Knowledge CMS slots."""

    batch = _batch_or_404(db, batch_id)
    if batch.status not in {"COMPLETED", "FAILED"}:
        raise HTTPException(status_code=409, detail={"code": "SYNC_NOT_ALLOWED", "message": "只有已执行批次可以同步到鱼鉴内容"})
    items = db.scalars(
        select(FishAssetImportItem)
        .where(
            FishAssetImportItem.batch_id == batch.batch_id,
            FishAssetImportItem.validation_status == "IMPORTED",
            FishAssetImportItem.version_id.is_not(None),
        )
        .order_by(FishAssetImportItem.id)
    ).all()
    totals = {"bound": 0, "active_preserved": 0, "missing_version": 0}
    for item in items:
        version = db.get(FishKnowledgeAssetVersion, item.version_id)
        if version is None:
            totals["missing_version"] += 1
            continue
        binding = _bind_imported_version(db, version)
        totals["active_preserved" if binding == "ACTIVE_PRESERVED" else "bound"] += 1
    previous = _read_json(batch.result_json, {})
    previous["content_sync"] = totals
    batch.result_json = _json(previous)
    db.commit()
    payload = _batch_dict(batch, include_items=True)
    payload["result"] = totals
    return payload


@router.get("/{batch_id}/versions/{version_id}/preview")
def preview_version(batch_id: str, version_id: int, db: Session = Depends(get_db)) -> Response:
    """Serve a DRAFT version to the authenticated Admin CMS only."""

    batch = _batch_or_404(db, batch_id)
    version = db.scalar(
        select(FishKnowledgeAssetVersion).where(
            FishKnowledgeAssetVersion.id == version_id,
            FishKnowledgeAssetVersion.batch_id == batch.batch_id,
        )
    )
    if version is None:
        raise HTTPException(status_code=404, detail={"code": "VERSION_NOT_FOUND", "message": "素材版本不存在"})
    try:
        client, bucket, _ = _storage(batch)
        blob = bucket.blob(version.object_name)
        data = blob.download_as_bytes(timeout=120)
    except Exception as exc:
        raise HTTPException(status_code=502, detail={"code": "GCS_READ_FAILED", "message": str(exc)}) from exc
    return Response(
        content=data,
        media_type="image/webp",
        headers={"Cache-Control": "private, max-age=300", "X-Content-Type-Options": "nosniff"},
    )


@router.post("/{batch_id}/versions/{version_id}/activate")
def activate_version(batch_id: str, version_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    batch = _batch_or_404(db, batch_id)
    version = db.scalar(select(FishKnowledgeAssetVersion).where(FishKnowledgeAssetVersion.id == version_id, FishKnowledgeAssetVersion.batch_id == batch.batch_id))
    if version is None:
        raise HTTPException(status_code=404, detail={"code": "VERSION_NOT_FOUND", "message": "DRAFT 素材版本不存在"})
    if version.status != "DRAFT":
        raise HTTPException(status_code=409, detail={"code": "VERSION_NOT_DRAFT", "message": "只有 DRAFT 版本可以发布"})
    species = db.get(FishSpecies, version.species_id)
    if species is None:
        raise HTTPException(status_code=404, detail={"code": "SPECIES_NOT_FOUND", "message": "鱼种不存在"})
    if version.asset_type == "COVER":
        current = db.scalar(select(FishSpeciesCover).where(FishSpeciesCover.species_id == species.id))
        if current is None:
            current = FishSpeciesCover(species_id=species.id, image_url=version.image_url, title=f"{species.name_cn}封面", status="ACTIVE")
            db.add(current)
        else:
            current.image_url = version.image_url
            current.status = "ACTIVE"
        version.status = "ACTIVE"
    else:
        active = next((row for row in db.scalars(select(FishCard).where(FishCard.species_id == species.id)).all() if normalize_card_type(row.card_type) == version.asset_type and row.status == "ACTIVE"), None)
        if active is not None:
            active.status = "DRAFT"
        candidate = next((row for row in db.scalars(select(FishCard).where(FishCard.species_id == species.id)).all() if normalize_card_type(row.card_type) == version.asset_type and row.status == "DRAFT" and not (row.image_url or "").strip()), None)
        if candidate is None:
            source = active
            candidate = FishCard(
                species_id=species.id,
                card_type=version.asset_type,
                title=(source.title if source else f"{species.name_cn}{version.asset_type}卡"),
                image_url=version.image_url,
                description=(source.description if source else ""),
                sort_order=CARD_TYPE_ORDER.index(version.asset_type),
                status="DRAFT",
            )
            db.add(candidate)
        else:
            candidate.image_url = version.image_url
        candidate.status = "ACTIVE"
        version.status = "ACTIVE"
    _commit(db)
    return {"success": True, "batch_id": batch.batch_id, "version_id": version.id, "species_id": version.species_id, "asset_type": version.asset_type, "status": "ACTIVE", "image_url": version.image_url}


@page_router.get("/fish-knowledge/assets/import", response_class=HTMLResponse)
def import_page() -> HTMLResponse:
    with open("app/templates/fish_asset_import.html", encoding="utf-8") as handle:
        return HTMLResponse(handle.read())
