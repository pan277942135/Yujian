from __future__ import annotations

import hashlib
import io
import json
import csv
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from google.cloud import storage
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from sqlalchemy import Boolean, CheckConstraint, Column, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, select
from sqlalchemy.orm import Session, relationship

from app.db import Base, get_db
from app.factory import get_bucket_name
from app.fish_knowledge.cards import CARD_TYPE_ORDER, FishCard, normalize_card_type
from app.fish_knowledge.content import card_description, parse_card_content
from app.fish_knowledge.cover import FishSpeciesCover
from app.fish_knowledge.gallery import GalleryUploadError, inspect_knowledge_asset, validate_knowledge_asset_role
from app.fish_knowledge.species import FishSpecies, SPECIES_ID_ALIASES
from app.models import SpeciesCatalog, utcnow


BATCH_STATUSES = ("CREATED", "SCANNING", "READY", "IMPORTING", "COMPLETED", "FAILED", "CANCELLED")
ITEM_STATUSES = ("VALID", "WARNING", "INVALID", "IMPORTED", "FAILED")
ASSET_TYPES = ("COVER", "HERO", "IDENTIFICATION", "ECO", "GEAR", "SKILL")
SOURCE_PREFIX = "fish-assets/imports/"
TARGET_ROOT = "fish-assets/fish-knowledge/"
ASSET_DIR = {
    "COVER": "cover",
    "COVER_LIST": "cover_list",
    "COVER_HERO": "cover_hero",
    "TRANSPARENT_MAIN": "transparent_main",
    "TRANSPARENT_ALT": "transparent_alt",
    "HERO": "hero",
    "IDENTIFICATION": "identification",
    "ECO": "ecology",
    "GEAR": "gear",
    "SKILL": "skill",
}
EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
IGNORED_FILES = {"readme.txt", "asset_manifest.csv", "manifest.csv"}
# Historical cover rows still use the constrained legacy asset_type=COVER.
# Their old three-way mapping stays in metadata_json; V1.3 asset_role and
# role-specific object paths keep all nine slots versioned independently.
COVER_VARIANT_ORDER = (
    "COVER_CARD",
    "COVER_CARD_TRANSPARENT_LEFT",
    "COVER_CARD_TRANSPARENT_RIGHT",
)
ASSET_PATTERNS = (
    (re.compile(r"^00_cover(?:_list)?(?:_.*)?$", re.I), "COVER"),
    (re.compile(r"^01_transparent_main(?:_.*)?$", re.I), "COVER"),
    (re.compile(r"^02_transparent_alt(?:_.*)?$", re.I), "COVER"),
    (re.compile(r"^01_hero(?:_.*)?$", re.I), "HERO"),
    (re.compile(r"^02_identification(?:_.*)?$", re.I), "IDENTIFICATION"),
    (re.compile(r"^03_(?:ecology|eco)(?:_.*)?$", re.I), "ECO"),
    (re.compile(r"^04_gear(?:_.*)?$", re.I), "GEAR"),
    (re.compile(r"^05_(?:skill|fishing)(?:_.*)?$", re.I), "SKILL"),
)
ASSET_ROLES = (
    "COVER_LIST", "COVER_HERO", "TRANSPARENT_MAIN", "TRANSPARENT_ALT",
    "HERO", "IDENTIFICATION", "ECO", "GEAR", "SKILL",
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
    warnings_acknowledged = Column(Boolean, nullable=False, default=False)
    warnings_acknowledged_at = Column(DateTime(timezone=True), nullable=True)
    warnings_acknowledged_by = Column(String(256), nullable=True)

    items = relationship(
        "FishAssetImportItem",
        primaryjoin=lambda: FishAssetImportBatch.batch_id == FishAssetImportItem.batch_id,
        cascade="all, delete-orphan",
        order_by="FishAssetImportItem.id",
    )


class FishAssetImportItem(Base):
    __tablename__ = "fish_asset_import_items"
    __table_args__ = (
        CheckConstraint("asset_type IN ('COVER','HERO','IDENTIFICATION','ECO','GEAR','SKILL')", name="ck_fish_asset_import_item_type"),
        CheckConstraint("validation_status IN ('VALID','WARNING','INVALID','IMPORTED','FAILED')", name="ck_fish_asset_import_item_status"),
        Index("ix_fish_asset_import_item_batch_species_type", "batch_id", "species_id", "asset_type"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_id = Column(String(128), ForeignKey("fish_asset_import_batches.batch_id", ondelete="CASCADE"), nullable=False, index=True)
    species_id = Column(String(128), ForeignKey("fish_species.id", ondelete="RESTRICT"), nullable=True, index=True)
    source_object = Column(Text, nullable=False)
    asset_type = Column(String(32), nullable=True)
    asset_role = Column(String(32), nullable=True, index=True)
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
        CheckConstraint("asset_type IN ('COVER','HERO','IDENTIFICATION','ECO','GEAR','SKILL')", name="ck_fish_knowledge_asset_version_type"),
        CheckConstraint("status IN ('DRAFT','ACTIVE','ARCHIVED')", name="ck_fish_knowledge_asset_version_status"),
        Index("uq_fish_knowledge_asset_role_version", "species_id", "asset_role", "version", unique=True),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    species_id = Column(String(128), ForeignKey("fish_species.id", ondelete="CASCADE"), nullable=False, index=True)
    asset_type = Column(String(32), nullable=False)
    asset_role = Column(String(32), nullable=True, index=True)
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


class FishKnowledgeAssetReview(Base):
    __tablename__ = "fish_knowledge_asset_reviews"
    __table_args__ = (
        UniqueConstraint("batch_id", "species_id", "asset_role", name="uq_fish_knowledge_review_batch_slot"),
        CheckConstraint("visual_qa_result IN ('PENDING','PASS','BLOCKED_VISUAL_QA')", name="ck_fish_knowledge_review_visual"),
        CheckConstraint("content_qa_result IN ('PENDING','PASS','BLOCKED_CONTENT_MISMATCH')", name="ck_fish_knowledge_review_content"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_id = Column(String(128), ForeignKey("fish_asset_import_batches.batch_id", ondelete="CASCADE"), nullable=False, index=True)
    species_id = Column(String(128), ForeignKey("fish_species.id", ondelete="RESTRICT"), nullable=False, index=True)
    asset_role = Column(String(32), nullable=False)
    version_id = Column(Integer, ForeignKey("fish_knowledge_asset_versions.id", ondelete="RESTRICT"), nullable=False, index=True)
    source_filename = Column(String(512), nullable=False)
    source_sha256 = Column(String(64), nullable=False)
    derived_media_sha256 = Column(String(64), nullable=False)
    object_name = Column(Text, nullable=False)
    object_generation = Column(String(64))
    validation_result = Column(String(16), nullable=False, default="PASS")
    validation_warnings_json = Column(Text, nullable=False, default="[]")
    visual_qa_result = Column(String(32), nullable=False, default="PENDING")
    content_qa_result = Column(String(32), nullable=False, default="PENDING")
    review_note = Column(Text, nullable=False, default="")
    reviewer = Column(String(256), nullable=False, default="admin")
    visual_qa_note = Column(Text, nullable=False, default="")
    visual_qa_reviewer = Column(String(256), nullable=False, default="admin")
    visual_qa_reviewed_at = Column(DateTime(timezone=True))
    content_qa_note = Column(Text, nullable=False, default="")
    content_qa_reviewer = Column(String(256), nullable=False, default="admin")
    content_qa_reviewed_at = Column(DateTime(timezone=True))
    binding_type = Column(String(32), nullable=True)
    binding_id = Column(Integer, nullable=True)
    binding_status = Column(String(16), nullable=True)
    binding_image_url = Column(Text, nullable=True)
    cms_content_sha256 = Column(String(64), nullable=True)
    asset_status = Column(String(16), nullable=True)
    warnings_acknowledged_at = Column(DateTime(timezone=True), nullable=True)
    warnings_acknowledged_by = Column(String(256), nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    frozen_at = Column(DateTime(timezone=True), nullable=True, index=True)
    code_head = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class FishKnowledgeAssetQAAudit(Base):
    """Immutable, per-stage QA evidence tied to an exact asset/content revision."""

    __tablename__ = "fish_knowledge_asset_qa_audits"
    __table_args__ = (
        CheckConstraint("qa_stage IN ('VISUAL','CONTENT')", name="ck_fish_knowledge_qa_stage"),
        Index("ix_fish_knowledge_asset_qa_version_stage", "version_id", "qa_stage"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    version_id = Column(Integer, ForeignKey("fish_knowledge_asset_versions.id", ondelete="RESTRICT"), nullable=False, index=True)
    species_id = Column(String(128), ForeignKey("fish_species.id", ondelete="RESTRICT"), nullable=False, index=True)
    asset_role = Column(String(32), nullable=False)
    qa_stage = Column(String(16), nullable=False)
    result = Column(String(32), nullable=False)
    reviewer = Column(String(256), nullable=False)
    evidence_note = Column(Text, nullable=False, default="")
    content_revision = Column(Integer, nullable=True)
    card_id = Column(Integer, ForeignKey("fish_cards.id", ondelete="RESTRICT"), nullable=True)
    source_sha256 = Column(String(64), nullable=False)
    derived_media_sha256 = Column(String(64), nullable=False)
    object_name = Column(Text, nullable=False)
    object_generation = Column(String(64), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class FishCardContentRevision(Base):
    """Immutable structured-content snapshot associated with an asset version."""

    __tablename__ = "fish_card_content_revisions"
    __table_args__ = (
        UniqueConstraint("card_id", "content_revision", name="uq_fish_card_content_revision"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    card_id = Column(Integer, ForeignKey("fish_cards.id", ondelete="RESTRICT"), nullable=False, index=True)
    asset_version_id = Column(Integer, ForeignKey("fish_knowledge_asset_versions.id", ondelete="RESTRICT"), nullable=False, index=True)
    content_revision = Column(Integer, nullable=False)
    title = Column(String(256), nullable=False, default="")
    description = Column(Text, nullable=False, default="")
    image_url = Column(Text, nullable=False)
    created_by = Column(String(256), nullable=False, default="admin")
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class FishKnowledgePublicationAudit(Base):
    """Durable record of a unified image/content publication decision."""

    __tablename__ = "fish_knowledge_publication_audits"

    id = Column(Integer, primary_key=True, autoincrement=True)
    species_id = Column(String(128), ForeignKey("fish_species.id", ondelete="RESTRICT"), nullable=False, index=True)
    asset_role = Column(String(32), nullable=False, index=True)
    asset_version_id = Column(Integer, ForeignKey("fish_knowledge_asset_versions.id", ondelete="RESTRICT"), nullable=False, index=True)
    card_id = Column(Integer, ForeignKey("fish_cards.id", ondelete="RESTRICT"), nullable=True, index=True)
    previous_version_id = Column(Integer, nullable=True)
    previous_card_id = Column(Integer, nullable=True)
    publication_status = Column(String(16), nullable=False)
    validation_json = Column(Text, nullable=False, default="{}")
    actor = Column(String(256), nullable=False, default="admin")
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


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


class AssetReviewPayload(BaseModel):
    batch_id: str | None = Field(default=None, min_length=3, max_length=128)
    visual_qa_result: Literal["PASS", "BLOCKED_VISUAL_QA"] | None = None
    visual_qa_note: str = Field(default="", max_length=4000)
    visual_qa_reviewer: str | None = Field(default=None, min_length=1, max_length=256)
    content_qa_result: Literal["PASS", "BLOCKED_CONTENT_MISMATCH"] | None = None
    content_qa_note: str = Field(default="", max_length=4000)
    content_qa_reviewer: str | None = Field(default=None, min_length=1, max_length=256)
    review_note: str = Field(default="", max_length=4000)
    reviewer: str = Field(default="admin", min_length=1, max_length=256)


class PublicAPIReadbackPayload(BaseModel):
    status: Literal["PUBLIC_API_OK", "API_MISMATCH", "IMAGE_UNREADABLE"]
    reviewer: str = Field(min_length=1, max_length=256)
    observed_version_id: int | None = None
    public_image_sha256: str | None = Field(default=None, max_length=64)
    preview_image_sha256: str | None = Field(default=None, max_length=64)
    detail: str = Field(default="", max_length=4000)


class ClientAcceptancePayload(BaseModel):
    result: Literal["CLIENT_PASSED", "CLIENT_FAILED"]
    reviewer: str = Field(min_length=1, max_length=256)
    observed_version_id: int
    detail: str = Field(default="", max_length=4000)


router = APIRouter(prefix="/api/v1/admin/fish/assets/import-batches", tags=["fish-knowledge-asset-import"])
asset_router = APIRouter(prefix="/api/v1/admin/fish/assets", tags=["fish-knowledge-assets-v13"])
page_router = APIRouter(tags=["fish-knowledge-asset-import"])


class BoundCardContentPayload(BaseModel):
    version_id: int
    title: str = Field(default="", max_length=256)
    structured_content: dict[str, Any]
    actor: str = Field(default="admin", max_length=256)


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


def _cover_variant_for_filename(filename: str) -> str | None:
    stem = filename.rsplit(".", 1)[0]
    if re.fullmatch(r"00_cover_hero(?:_.*)?", stem, re.I):
        return "COVER_HERO"
    if re.fullmatch(r"00_cover(?:_list)?(?:_.*)?", stem, re.I):
        return "COVER_CARD"
    if re.fullmatch(r"01_transparent_main(?:_.*)?", stem, re.I):
        return "COVER_CARD_TRANSPARENT_LEFT"
    if re.fullmatch(r"02_transparent_alt(?:_.*)?", stem, re.I):
        return "COVER_CARD_TRANSPARENT_RIGHT"
    return None


def _cover_variant_for_version(version: FishKnowledgeAssetVersion) -> str:
    metadata = _read_json(version.metadata_json, {})
    metadata = metadata if isinstance(metadata, dict) else {}
    value = str(
        metadata.get("cover_variant")
        or metadata.get("asset_role")
        or metadata.get("reference_variant")
        or ""
    ).strip().upper()
    aliases = {
        "COVER_LIST": "COVER_CARD",
        "COVER_CARD_TRANSPARENT_MAIN": "COVER_CARD_TRANSPARENT_LEFT",
        "TRANSPARENT_MAIN": "COVER_CARD_TRANSPARENT_LEFT",
        "TRANSPARENT_LEFT": "COVER_CARD_TRANSPARENT_LEFT",
        "COVER_CARD_TRANSPARENT_ALT": "COVER_CARD_TRANSPARENT_RIGHT",
        "TRANSPARENT_ALT": "COVER_CARD_TRANSPARENT_RIGHT",
        "TRANSPARENT_RIGHT": "COVER_CARD_TRANSPARENT_RIGHT",
    }
    value = aliases.get(value, value)
    return value if value in COVER_VARIANT_ORDER or value == "COVER_HERO" else "COVER_CARD"


def _asset_role_for_filename(filename: str) -> str | None:
    stem = filename.rsplit(".", 1)[0]
    if re.fullmatch(r"00_cover_hero(?:_.*)?", stem, re.I):
        return "COVER_HERO"
    if re.fullmatch(r"00_cover(?:_list)?(?:_.*)?", stem, re.I):
        return "COVER_LIST"
    if re.fullmatch(r"01_transparent_main(?:_.*)?", stem, re.I):
        return "TRANSPARENT_MAIN"
    if re.fullmatch(r"02_transparent_alt(?:_.*)?", stem, re.I):
        return "TRANSPARENT_ALT"
    return {
        "HERO": "HERO",
        "IDENTIFICATION": "IDENTIFICATION",
        "ECO": "ECO",
        "GEAR": "GEAR",
        "SKILL": "SKILL",
    }.get(_asset_type_for_filename(filename) or "")


def _asset_role_for_version(version: FishKnowledgeAssetVersion) -> str:
    value = str(getattr(version, "asset_role", None) or "").strip().upper()
    if value in ASSET_ROLES:
        return value
    if version.asset_type != "COVER":
        return str(version.asset_type).strip().upper()
    metadata = _read_json(version.metadata_json, {})
    metadata = metadata if isinstance(metadata, dict) else {}
    stored_role = str(metadata.get("asset_role") or "").strip().upper()
    if stored_role in ASSET_ROLES:
        return stored_role
    variant = str(metadata.get("cover_variant") or "").strip().upper()
    if variant in {"COVER_CARD_TRANSPARENT_LEFT", "TRANSPARENT_MAIN", "TRANSPARENT_LEFT"}:
        return "TRANSPARENT_MAIN"
    if variant in {"COVER_CARD_TRANSPARENT_RIGHT", "TRANSPARENT_ALT", "TRANSPARENT_RIGHT"}:
        return "TRANSPARENT_ALT"
    if variant in {"COVER_HERO", "COVER_CARD_HERO"}:
        return "COVER_HERO"
    return "COVER_LIST"


def _asset_type_for_filename(filename: str) -> str | None:
    stem = filename.rsplit(".", 1)[0]
    for pattern, asset_type in ASSET_PATTERNS:
        if pattern.fullmatch(stem):
            return asset_type
    return None


def _image_extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def _validation(error_code: str, message: str) -> dict[str, str]:
    return {"code": error_code, "message": message}


def _next_version(db: Session, client: Any, bucket: Any, species_id: str, asset_role: str) -> int:
    current = db.scalar(
        select(FishKnowledgeAssetVersion.version)
        .where(
            FishKnowledgeAssetVersion.species_id == species_id,
            FishKnowledgeAssetVersion.asset_role == asset_role,
        )
        .order_by(FishKnowledgeAssetVersion.version.desc())
    ) or 0
    prefix = f"{TARGET_ROOT}{species_id}/{ASSET_DIR[asset_role]}/"
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
            FishKnowledgeAssetVersion.asset_role == asset_type,
            FishKnowledgeAssetVersion.sha256 == sha256,
        )
        .order_by(FishKnowledgeAssetVersion.id.desc())
    )


def _existing_sha_elsewhere(db: Session, species_id: str, asset_role: str, sha256: str) -> FishKnowledgeAssetVersion | None:
    return db.scalar(
        select(FishKnowledgeAssetVersion)
        .where(
            FishKnowledgeAssetVersion.species_id == species_id,
            FishKnowledgeAssetVersion.asset_role != asset_role,
            FishKnowledgeAssetVersion.sha256 == sha256,
        )
        .order_by(FishKnowledgeAssetVersion.id.desc())
    )


def _item_dict(item: FishAssetImportItem, *, base: str, db: Session | None = None) -> dict[str, Any]:
    errors = _read_json(item.validation_errors, [])
    warnings = _read_json(item.validation_warnings, [])
    review = db.scalar(select(FishKnowledgeAssetReview).where(
        FishKnowledgeAssetReview.batch_id == item.batch_id,
        FishKnowledgeAssetReview.species_id == item.species_id,
        FishKnowledgeAssetReview.asset_role == (item.asset_role or _asset_role_for_filename(item.source_filename)),
    )) if db and item.species_id else None
    version = db.get(FishKnowledgeAssetVersion, item.version_id) if db and item.version_id else None
    structured_content = None
    role = item.asset_role or _asset_role_for_filename(item.source_filename)
    if db and item.species_id and role in CARD_TYPE_ORDER:
        cards = db.scalars(select(FishCard).where(
            FishCard.species_id == item.species_id,
            FishCard.card_type == normalize_card_type(role),
        ).order_by(FishCard.updated_at.desc(), FishCard.id.desc())).all()
        card = next((value for value in cards if value.status == "ACTIVE"), cards[0] if cards else None)
        if card:
            structured_content = {
                "card_type": normalize_card_type(card.card_type),
                "title": card.title,
                "description": card.description,
                "content": parse_card_content(card.description),
            }
    payload = {
        "id": item.id,
        "species_id": item.species_id,
        "asset_type": item.asset_type,
        "asset_role": item.asset_role or _asset_role_for_filename(item.source_filename),
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
        "review": {
            "validation_result": review.validation_result,
            "visual_qa_result": review.visual_qa_result,
            "content_qa_result": review.content_qa_result,
            "review_note": review.review_note,
            "reviewer": review.reviewer,
            "frozen": bool(review.frozen_at),
        } if review else None,
        "version_status": version.status if version else None,
        "derived_media_sha256": (
            _read_json(version.metadata_json, {}).get("derived_sha256")
            if version and isinstance(_read_json(version.metadata_json, {}), dict)
            else None
        ),
        "structured_content": structured_content,
    }
    if item.version_id:
        payload["derived_preview_url"] = f"/api/v1/admin/fish/assets/versions/{item.version_id}/preview"
    return payload


def _summary(batch: FishAssetImportBatch, *, items: list[FishAssetImportItem] | None = None) -> dict[str, int]:
    items = items or []
    already_exists = sum(
        1 for item in items
        if any(error.get("code") == "ASSET_ALREADY_EXISTS" for error in _read_json(item.validation_warnings, []))
    )
    imported = sum(1 for item in items if item.validation_status == "IMPORTED")
    failed = sum(1 for item in items if item.validation_status == "FAILED")
    invalid = sum(1 for item in items if item.validation_status == "INVALID") if items else batch.failed_files
    return {
        "species_count": batch.species_count,
        "total_files": batch.total_files,
        "recognized_files": batch.recognized_files,
        "valid": batch.valid_files,
        "warning": batch.warning_files,
        "invalid": invalid,
        "imported": max(0, imported - already_exists),
        "already_exists": already_exists,
        "failed": failed,
        "pending": sum(1 for item in items if item.validation_status in {"VALID", "WARNING"}),
    }


def _batch_dict(
    batch: FishAssetImportBatch,
    *,
    include_items: bool = False,
    db: Session | None = None,
    item_rows: list[FishAssetImportItem] | None = None,
) -> dict[str, Any]:
    rows = list(batch.items) if include_items else list(item_rows or [])
    items = [
        _item_dict(
            item,
            base=f"/api/v1/admin/fish/assets/import-batches/{batch.batch_id}",
            db=db if include_items else None,
        )
        for item in rows
    ]
    by_species: dict[str, dict[str, Any]] = {}
    for item in items:
        species_id = item.get("species_id") or "UNRESOLVED"
        row = by_species.setdefault(species_id, {"species_id": species_id, "assets": {}, "completion": 0})
        key = item.get("asset_role") or item.get("asset_type") or f"INVALID_{item['id']}"
        row["assets"].setdefault(key, []).append(item)
    for row in by_species.values():
        row["completion"] = f"{sum(1 for values in row['assets'].values() if any(x['validation_status'] in {'VALID','WARNING','IMPORTED'} for x in values))}/9"
    payload = {
        "batch_id": batch.batch_id,
        "source_gcs_uri": batch.source_gcs_uri,
        "status": batch.status,
        "created_by": batch.created_by,
        "created_at": batch.created_at.isoformat() if batch.created_at else None,
        "updated_at": batch.updated_at.isoformat() if batch.updated_at else None,
        "summary": _summary(batch, items=rows),
        "warning_acknowledgement": {
            "acknowledged": bool(batch.warnings_acknowledged),
            "acknowledged_at": batch.warnings_acknowledged_at.isoformat() if batch.warnings_acknowledged_at else None,
            "acknowledged_by": batch.warnings_acknowledged_by,
        },
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


def _scan_item(db: Session, client: Any, bucket: Any, batch: FishAssetImportBatch, name: str) -> FishAssetImportItem | None:
    relative = name[len(_source_parts(batch.source_gcs_uri)[1]):].lstrip("/")
    parts = relative.split("/")
    filename = parts[-1]
    if not filename or name.endswith("/"):
        return None
    lowered = filename.lower()
    if lowered in IGNORED_FILES:
        return None
    folder = parts[0] if parts else ""
    for candidate in parts[:-1]:
        if _resolve_species(db, candidate) is not None:
            folder = candidate
            break
    asset_type = _asset_type_for_filename(filename)
    asset_role = _asset_role_for_filename(filename)
    suffix = "." + _image_extension(filename) if "." in filename else ""
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    species = _resolve_species(db, folder) if folder else None
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
            metadata, role_errors, role_warnings = validate_knowledge_asset_role(data, asset_role or "")
            errors.extend(role_errors)
            warnings.extend(role_warnings)
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
    digest = str(metadata.get("sha256") or "")
    if species is not None and asset_type is not None and asset_role is not None and digest:
        duplicate = _existing_duplicate(db, species.id, asset_role, digest)
        if duplicate is not None:
            warnings.append(_validation("ASSET_ALREADY_EXISTS", f"Same SHA-256 already exists as {duplicate.object_name}"))
            target_object = duplicate.object_name
        else:
            target_object = None
            same_sha_other_role = _existing_sha_elsewhere(db, species.id, asset_role, digest)
            if same_sha_other_role is not None:
                warnings.append(_validation(
                    "ASSET_SHA_REUSED_OTHER_ROLE",
                    f"Same source SHA-256 exists in {same_sha_other_role.asset_role}; this role remains independently versioned",
                ))
    else:
        target_object = None
    status = "INVALID" if errors else ("WARNING" if warnings else "VALID")
    return FishAssetImportItem(
        batch_id=batch.batch_id,
        species_id=species.id if species else None,
        source_object=name,
        asset_type=asset_type,
        asset_role=asset_role,
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
        role = item.asset_role or _asset_role_for_filename(item.source_filename)
        item.asset_role = role
        key = (item.species_id, role)
        if item.species_id and role:
            slots.setdefault(key, []).append(item)
    for (species_id, role), values in slots.items():
        if len(values) < 2:
            continue
        for item in values:
            errors = _read_json(item.validation_errors, [])
            label = f"{species_id} {role}"
            if not any(error.get("code") == "DUPLICATE_ASSET_SLOT" for error in errors):
                errors.append(_validation("DUPLICATE_ASSET_SLOT", f"{label} has {len(values)} files"))
            item.validation_errors = _json(errors)
            item.validation_status = "INVALID"


def _assign_targets(db: Session, client: Any, bucket: Any, items: list[FishAssetImportItem]) -> None:
    next_versions: dict[tuple[str, str], int] = {}
    for item in items:
        if item.validation_status == "INVALID" or not item.species_id or not item.asset_type:
            continue
        if item.target_object:
            continue
        role = item.asset_role or _asset_role_for_filename(item.source_filename)
        if not role:
            continue
        item.asset_role = role
        key = (item.species_id, role)
        if key not in next_versions:
            next_versions[key] = _next_version(db, client, bucket, item.species_id, role)
        version = next_versions[key]
        next_versions[key] = version + 1
        item.target_object = f"{TARGET_ROOT}{item.species_id}/{ASSET_DIR[role]}/v{version}.webp"


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


def _version_url(species_id: str, asset_role: str, version: int) -> str:
    return f"/api/v1/fish/knowledge-media/{species_id}/{asset_role.lower()}/v{version}.webp"


def _next_version_row(db: Session, species_id: str, asset_role: str) -> int:
    return int(db.scalar(select(FishKnowledgeAssetVersion.version).where(
        FishKnowledgeAssetVersion.species_id == species_id,
        FishKnowledgeAssetVersion.asset_role == asset_role,
    ).order_by(FishKnowledgeAssetVersion.version.desc())) or 0) + 1


def _bind_imported_version(db: Session, version: FishKnowledgeAssetVersion) -> str:
    """Bind an imported DRAFT image to the editable Fish Knowledge slot.

    Existing ACTIVE content is never replaced.  For cards we can keep the
    ACTIVE row and create a new DRAFT row; the cover schema has one row per
    species, so an existing ACTIVE cover remains the live slot until the
    operator explicitly activates the imported version.
    """

    species = db.get(FishSpecies, version.species_id)
    if species is None:
        raise RuntimeError(f"species {version.species_id} not found while binding imported asset")

    role = _asset_role_for_version(version)
    if role in {"COVER_HERO", "TRANSPARENT_MAIN", "TRANSPARENT_ALT"}:
        return "ROLE_ONLY"

    if role == "COVER_LIST":
        # Keep the legacy list-cover binding while the new role remains separately versioned.
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

    if role not in CARD_TYPE_ORDER:
        return "ROLE_ONLY"
    card_type = normalize_card_type(role)
    # An import is idempotent by the immutable asset version ID.  Never guess
    # which blank-image card was intended, and never reuse another version's
    # draft row.  Copying published text creates a new editable revision while
    # leaving the ACTIVE FishCard untouched.
    candidate = db.scalar(select(FishCard).where(FishCard.asset_version_id == version.id))
    if candidate is not None:
        return "ALREADY_BOUND"
    active_rows = db.scalars(select(FishCard).where(
        FishCard.species_id == species.id,
        FishCard.status == "ACTIVE",
    )).all()
    active = next((row for row in active_rows if normalize_card_type(row.card_type) == card_type), None)
    candidate = FishCard(
        species_id=species.id,
        card_type=card_type,
        title=(active.title if active else f"{species.name_cn}{card_type}卡"),
        image_url=version.image_url,
        description=(active.description if active else ""),
        sort_order=CARD_TYPE_ORDER.index(card_type),
        status="DRAFT",
        asset_version_id=version.id,
        content_revision=1,
    )
    db.add(candidate)
    db.flush()
    db.add(FishCardContentRevision(
        card_id=candidate.id,
        asset_version_id=version.id,
        content_revision=1,
        title=candidate.title,
        description=candidate.description,
        image_url=candidate.image_url,
    ))
    return "BOUND"


def _import_item(db: Session, client: Any, bucket: Any, batch: FishAssetImportBatch, item: FishAssetImportItem) -> str:
    if item.validation_status == "IMPORTED":
        return "SKIP_IMPORTED"
    warnings = _read_json(item.validation_warnings, [])
    if any(warning.get("code") == "ASSET_ALREADY_EXISTS" for warning in warnings):
        role = item.asset_role or _asset_role_for_filename(item.source_filename)
        duplicate = _existing_duplicate(db, item.species_id, role or "", str(item.sha256 or "")) if item.species_id and role else None
        if duplicate is None:
            item.validation_status = "FAILED"
            item.validation_errors = _json([_validation("DUPLICATE_VERSION_MISSING", "重复 SHA 记录对应的有效版本不存在")])
            return "FAILED"
        item.version_id = duplicate.id
        item.target_object = duplicate.object_name
        item.validation_status = "IMPORTED"
        _bind_imported_version(db, duplicate)
        return "SKIP_DUPLICATE"
    if item.validation_status == "INVALID":
        item.validation_status = "FAILED"
        return "SKIP_INVALID"
    role = item.asset_role or _asset_role_for_filename(item.source_filename)
    if not item.target_object or not item.species_id or not item.asset_type or not role:
        item.validation_status = "FAILED"
        item.validation_errors = _json([_validation("TARGET_NOT_READY", "Validated item has no target object")])
        return "FAILED"
    try:
        blob = bucket.blob(item.source_object)
        data = blob.download_as_bytes(timeout=120)
        metadata, role_errors, _role_warnings = validate_knowledge_asset_role(data, role)
        if role_errors:
            raise RuntimeError("source failed role validation after scan: " + "; ".join(e["code"] for e in role_errors))
        if str(metadata["sha256"]) != str(item.sha256):
            raise RuntimeError("source content changed after scan")
        stored = bytes(metadata["webp_data"])
        target = bucket.blob(item.target_object)
        if target.exists(client):
            existing = target.download_as_bytes(timeout=120)
            if hashlib.sha256(existing).hexdigest() != str(metadata["derived_sha256"]):
                raise RuntimeError("target object exists with different content")
        else:
            target.metadata = {
                "source_sha256": str(item.sha256 or ""),
                "derived_sha256": str(metadata["derived_sha256"]),
                "source_object": item.source_object,
                "batch_id": batch.batch_id,
                "asset_type": item.asset_type,
                "asset_role": role,
            }
            target.upload_from_string(stored, content_type="image/webp", if_generation_match=0)
        try:
            target.reload(client)
        except Exception:
            pass
        version = _next_version_row(db, item.species_id, role)
        existing_version = db.scalar(select(FishKnowledgeAssetVersion).where(FishKnowledgeAssetVersion.object_name == item.target_object))
        if existing_version is None:
            version_row = FishKnowledgeAssetVersion(
                species_id=item.species_id,
                asset_type=item.asset_type,
                asset_role=role,
                version=version,
                object_name=item.target_object,
                image_url=_version_url(item.species_id, role, version),
                status="DRAFT",
                sha256=str(item.sha256),
                metadata_json=_json({
                    "asset_role": role,
                    "width": metadata.get("width"),
                    "height": metadata.get("height"),
                    "original_content_type": metadata.get("original_content_type"),
                    "source_sha256": metadata.get("sha256"),
                    "derived_sha256": metadata.get("derived_sha256"),
                    "source_size_bytes": metadata.get("size_bytes"),
                    "stored_size_bytes": metadata.get("stored_size_bytes"),
                    "source_filename": item.source_filename,
                    "source_format": metadata.get("original_content_type"),
                    "cover_variant": "COVER_CARD" if role == "COVER_LIST" else (
                        "COVER_CARD_TRANSPARENT_LEFT" if role == "TRANSPARENT_MAIN" else (
                            "COVER_CARD_TRANSPARENT_RIGHT" if role == "TRANSPARENT_ALT" else None
                        )
                    ),
                    "gcs_generation": str(getattr(target, "generation", "") or ""),
                }),
                batch_id=batch.batch_id,
                item_id=item.id,
            )
            db.add(version_row)
            db.flush()
            item.version_id = version_row.id
        else:
            version_row = existing_version
            item.version_id = existing_version.id

        binding = _bind_imported_version(db, version_row)
        item.validation_status = "IMPORTED"
        if binding == "ROLE_ONLY":
            return "IMPORTED_ROLE_ONLY"
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
        "role_only": 0,
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
            "IMPORTED_ROLE_ONLY": "role_only",
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
        if existing.source_gcs_uri.rstrip("/") == payload.source_gcs_uri.strip().rstrip("/"):
            return {"batch_id": existing.batch_id, "status": existing.status, "source_gcs_uri": existing.source_gcs_uri, "resumed": True}
        raise HTTPException(status_code=409, detail={"code": "BATCH_ID_CONFLICT", "message": "batch_id 已绑定到不同的源目录"})
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
        if existing.source_gcs_uri == source_gcs_uri:
            return {"batch_id": existing.batch_id, "status": existing.status, "source_gcs_uri": existing.source_gcs_uri, "resumed": True}
        raise HTTPException(status_code=409, detail={"code": "BATCH_ID_CONFLICT", "message": "batch_id 已绑定到不同的源目录"})
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
    try:
        normalized_path = _normalize_upload_path(relative_path)
    except HTTPException:
        await file.close()
        raise
    if normalized_path.lower().endswith(".zip"):
        await file.close()
        raise HTTPException(status_code=400, detail={"code": "ARCHIVE_UPLOAD_FORBIDDEN", "message": "请在客户端解压 ZIP 后上传图片文件；不得将 ZIP 发送到服务端"})
    try:
        data = await file.read(10 * 1024 * 1024 + 1)
    finally:
        await file.close()
    size = len(data)
    if size > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail={"code": "FILE_TOO_LARGE", "message": "单个图片文件不能超过 10 MiB"})
    if not data:
        raise HTTPException(status_code=400, detail={"code": "EMPTY_FILE", "message": "图片文件为空"})
    if batch.status != "CREATED":
        # A retry after an interrupted request is safe only when the bytes match.
        if batch.status in {"READY", "COMPLETED", "FAILED"}:
            try:
                client, bucket, prefix = _storage(batch)
                existing_blob = bucket.blob(prefix + normalized_path)
                if existing_blob.exists(client):
                    if hashlib.sha256(existing_blob.download_as_bytes(timeout=120)).hexdigest() == hashlib.sha256(data).hexdigest():
                        return {"batch_id": batch.batch_id, "relative_path": normalized_path, "source_object": prefix + normalized_path, "size": size, "idempotent": True}
            except Exception:
                pass
        raise HTTPException(status_code=409, detail={"code": "BATCH_NOT_UPLOADABLE", "message": "该批次已扫描或执行；仅相同源文件允许幂等恢复"})
    try:
        client, bucket, prefix = _storage(batch)
        blob = bucket.blob(prefix + normalized_path)
        if blob.exists(client):
            existing_data = blob.download_as_bytes(timeout=120)
            if hashlib.sha256(existing_data).hexdigest() != hashlib.sha256(data).hexdigest():
                raise HTTPException(status_code=409, detail={"code": "SOURCE_PATH_CONFLICT", "message": "同一路径已有不同内容，请创建新批次或改正本地文件"})
            return {"batch_id": batch.batch_id, "relative_path": normalized_path, "source_object": prefix + normalized_path, "size": size, "idempotent": True}
        blob.upload_from_string(data, content_type=file.content_type or "application/octet-stream", if_generation_match=0)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail={"code": "GCS_UPLOAD_FAILED", "message": str(exc)}) from exc
    return {
        "batch_id": batch.batch_id,
        "relative_path": normalized_path,
        "source_object": prefix + normalized_path,
        "size": size,
        "idempotent": False,
    }


@router.post("/{batch_id}/scan")
def scan_batch(batch_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    batch = _batch_or_404(db, batch_id)
    if batch.status == "COMPLETED":
        return _batch_dict(batch, include_items=True, db=db)
    if batch.status in {"IMPORTING"}:
        raise HTTPException(status_code=409, detail={"code": "BATCH_BUSY", "message": "批次正在导入"})
    client, bucket, prefix = _storage(batch)
    batch.status = "SCANNING"
    db.query(FishAssetImportItem).filter(FishAssetImportItem.batch_id == batch.batch_id).delete(synchronize_session=False)
    db.commit()
    items: list[FishAssetImportItem] = []
    try:
        blobs = list(client.list_blobs(bucket, prefix=prefix))
        for blob in blobs:
            item = _scan_item(db, client, bucket, batch, blob.name)
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
    return _batch_dict(_batch_or_404(db, batch_id), include_items=True, db=db)


@router.get("")
def list_batches(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    rows = db.scalars(select(FishAssetImportBatch).order_by(FishAssetImportBatch.created_at.desc()).limit(100)).all()
    if not rows:
        return []
    batch_ids = [row.batch_id for row in rows]
    item_rows = db.scalars(
        select(FishAssetImportItem)
        .where(FishAssetImportItem.batch_id.in_(batch_ids))
        .order_by(FishAssetImportItem.id)
    ).all()
    items_by_batch: dict[str, list[FishAssetImportItem]] = {batch_id: [] for batch_id in batch_ids}
    for item in item_rows:
        items_by_batch.setdefault(item.batch_id, []).append(item)
    return [_batch_dict(row, db=db, item_rows=items_by_batch.get(row.batch_id, [])) for row in rows]


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
    if batch.status == "COMPLETED":
        return _batch_dict(batch, include_items=True, db=db)
    if batch.status != "READY":
        raise HTTPException(status_code=409, detail={"code": "BATCH_NOT_READY", "message": "必须先完成 Scan 并处于 READY"})
    if batch.failed_files:
        raise HTTPException(status_code=409, detail={"code": "INVALID_ITEMS_PRESENT", "message": "存在 INVALID 图片，不能执行"})
    if batch.warning_files and not payload.allow_warnings:
        raise HTTPException(status_code=409, detail={"code": "WARNINGS_REQUIRE_CONFIRMATION", "message": "存在 WARNING 图片，请明确 allow_warnings=true"})
    if batch.warning_files:
        batch.warnings_acknowledged = True
        batch.warnings_acknowledged_at = utcnow()
        batch.warnings_acknowledged_by = batch.created_by or "admin"
    batch.status = "IMPORTING"
    db.commit()
    result = _run_import(db, batch)
    payload = _batch_dict(batch, include_items=True, db=db)
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
    payload = _batch_dict(batch, include_items=True, db=db)
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
    totals = {"bound": 0, "reference_only": 0, "active_preserved": 0, "missing_version": 0}
    for item in items:
        version = db.get(FishKnowledgeAssetVersion, item.version_id)
        if version is None:
            totals["missing_version"] += 1
            continue
        binding = _bind_imported_version(db, version)
        if binding == "REFERENCE_ONLY":
            totals["reference_only"] += 1
        else:
            totals["active_preserved" if binding == "ACTIVE_PRESERVED" else "bound"] += 1
    previous = _read_json(batch.result_json, {})
    previous["content_sync"] = totals
    batch.result_json = _json(previous)
    db.commit()
    payload = _batch_dict(batch, include_items=True, db=db)
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
    version = db.scalar(select(FishKnowledgeAssetVersion).where(
        FishKnowledgeAssetVersion.id == version_id,
        FishKnowledgeAssetVersion.batch_id == batch.batch_id,
    ))
    if version is None:
        raise HTTPException(status_code=404, detail={"code": "VERSION_NOT_FOUND", "message": "DRAFT 素材版本不存在"})
    from app.fish_knowledge.publication import publish_asset_version

    result = publish_asset_version(
        db,
        version.id,
        expected_batch_id=batch.batch_id,
        actor=batch.created_by,
    )
    return {**result, "batch_id": batch.batch_id, "asset_type": version.asset_type, "asset_role": result["role"], "status": result["publication_status"]}


@page_router.get("/fish-knowledge/assets/import", response_class=HTMLResponse)
def import_page() -> HTMLResponse:
    with open("app/templates/fish_asset_import.html", encoding="utf-8") as handle:
        return HTMLResponse(handle.read())


def _version_admin_dict(version: FishKnowledgeAssetVersion, db: Session) -> dict[str, Any]:
    metadata = _read_json(version.metadata_json, {})
    metadata = metadata if isinstance(metadata, dict) else {}
    role = _asset_role_for_version(version)
    review = db.scalar(select(FishKnowledgeAssetReview).where(
        FishKnowledgeAssetReview.version_id == version.id,
        FishKnowledgeAssetReview.species_id == version.species_id,
        FishKnowledgeAssetReview.asset_role == role,
    ).order_by(
        FishKnowledgeAssetReview.frozen_at.is_not(None).desc(),
        FishKnowledgeAssetReview.created_at.desc(),
    ))
    return {
        "id": version.id,
        "species_id": version.species_id,
        "asset_type": version.asset_type,
        "asset_role": role,
        "version": version.version,
        "status": version.status,
        "image_url": version.image_url,
        "source_filename": metadata.get("source_filename"),
        "source_format": metadata.get("source_format") or metadata.get("original_content_type"),
        "source_sha256": metadata.get("source_sha256") or version.sha256,
        "derived_media_sha256": metadata.get("derived_sha256"),
        "width": metadata.get("width"),
        "height": metadata.get("height"),
        "source_size_bytes": metadata.get("source_size_bytes"),
        "stored_size_bytes": metadata.get("stored_size_bytes"),
        "object_name": version.object_name,
        "object_generation": metadata.get("gcs_generation"),
        "batch_id": version.batch_id,
        "created_at": version.created_at.isoformat() if version.created_at else None,
        "preview_url": f"/api/v1/admin/fish/assets/versions/{version.id}/preview",
        "frozen": bool(review and review.frozen_at),
        "frozen_at": review.frozen_at.isoformat() if review and review.frozen_at else None,
        "review": {
            "validation_result": review.validation_result,
            "visual_qa_result": review.visual_qa_result,
            "content_qa_result": review.content_qa_result,
            "reviewer": review.reviewer,
            "review_note": review.review_note,
            "reviewed_at": review.reviewed_at.isoformat() if review and review.reviewed_at else None,
            "visual_qa_note": review.visual_qa_note,
            "visual_qa_reviewer": review.visual_qa_reviewer,
            "visual_qa_reviewed_at": review.visual_qa_reviewed_at.isoformat() if review and review.visual_qa_reviewed_at else None,
            "content_qa_note": review.content_qa_note,
            "content_qa_reviewer": review.content_qa_reviewer,
            "content_qa_reviewed_at": review.content_qa_reviewed_at.isoformat() if review and review.content_qa_reviewed_at else None,
        } if review else None,
    }


def _asset_slots(db: Session, species_id: str) -> dict[str, Any]:
    species = db.get(FishSpecies, species_id)
    if species is None or species.status == "DELETED":
        raise HTTPException(status_code=404, detail={"code": "SPECIES_NOT_FOUND", "message": "鱼种不存在"})
    versions = db.scalars(select(FishKnowledgeAssetVersion).where(
        FishKnowledgeAssetVersion.species_id == species_id,
    ).order_by(FishKnowledgeAssetVersion.created_at.desc(), FishKnowledgeAssetVersion.id.desc())).all()
    by_role: dict[str, list[FishKnowledgeAssetVersion]] = {role: [] for role in ASSET_ROLES}
    for version in versions:
        role = _asset_role_for_version(version)
        if role in by_role:
            by_role[role].append(version)
    slots = []
    for role in ASSET_ROLES:
        history = [_version_admin_dict(version, db) for version in by_role[role]]
        current = next((row for row in history if row["status"] == "DRAFT"), None)
        if current is None:
            current = next((row for row in history if row["status"] == "ACTIVE"), None)
        available = [row for row in history if row["status"] in {"DRAFT", "ACTIVE"}]
        slots.append({"asset_role": role, "current": current, "history": history, "count": len(history), "available": len(available) > 0})
    populated = sum(1 for slot in slots if slot["available"])
    return {
        "species_id": species.id,
        "species_name_cn": species.name_cn,
        "slots": slots,
        "completion": {
            "cover": sum(1 for slot in slots[:2] if slot["available"]),
            "standard": sum(1 for slot in slots[2:4] if slot["available"]),
            "knowledge": sum(1 for slot in slots[4:] if slot["available"]),
            "total": populated,
            "expected": 9,
        },
    }


@asset_router.get("/species/{species_id}")
def get_species_assets_v13(species_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    return _asset_slots(db, species_id)


@asset_router.get("/species/{species_id}/roles/{asset_role}")
def get_species_asset_role_history(species_id: str, asset_role: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    role = asset_role.strip().upper()
    if role not in ASSET_ROLES:
        raise HTTPException(status_code=400, detail={"code": "INVALID_ASSET_ROLE", "message": "未知资产角色"})
    payload = _asset_slots(db, species_id)
    slot = next(value for value in payload["slots"] if value["asset_role"] == role)
    return {"species_id": species_id, **slot}


@asset_router.get("/species/{species_id}/workspace")
def get_unified_card_workspace(
    species_id: str,
    selected_version_id: int | None = None,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Return one role-indexed view with ACTIVE and DRAFT kept separate."""

    species = db.get(FishSpecies, species_id)
    if species is None or species.status == "DELETED":
        raise HTTPException(status_code=404, detail={"code": "SPECIES_NOT_FOUND", "message": "鱼种不存在"})
    versions = db.scalars(select(FishKnowledgeAssetVersion).where(
        FishKnowledgeAssetVersion.species_id == species.id,
    )).all()
    unbound_cards = db.scalars(select(FishCard).where(
        FishCard.species_id == species.id,
        FishCard.asset_version_id.is_(None),
    ).order_by(FishCard.id)).all()
    legacy_cards_by_role: dict[str, list[FishCard]] = {role: [] for role in CARD_TYPE_ORDER}
    for legacy_card in unbound_cards:
        legacy_role = normalize_card_type(legacy_card.card_type)
        if legacy_role in legacy_cards_by_role:
            legacy_cards_by_role[legacy_role].append(legacy_card)
    by_role: dict[str, list[FishKnowledgeAssetVersion]] = {role: [] for role in ASSET_ROLES}
    for version in versions:
        role = _asset_role_for_version(version)
        if role in by_role:
            by_role[role].append(version)

    roles: dict[str, Any] = {}
    for role in ASSET_ROLES:
        history = sorted(by_role[role], key=lambda value: (value.version, value.id), reverse=True)
        active = [value for value in history if value.status == "ACTIVE"]
        drafts = [value for value in history if value.status == "DRAFT"]
        legacy_cards = legacy_cards_by_role.get(role, [])
        legacy_active_cards = [value for value in legacy_cards if value.status == "ACTIVE"]
        legacy_draft_cards = [value for value in legacy_cards if value.status == "DRAFT"]
        selected = next((value for value in history if value.id == selected_version_id), None) if selected_version_id else None
        if selected is not None and _asset_role_for_version(selected) != role:
            selected = None
        if selected is None and len(drafts) == 1:
            selected = drafts[0]
        if selected is None and not drafts and len(active) == 1:
            selected = active[0]

        active_dict = _version_admin_dict(active[0], db) if len(active) == 1 else None
        draft_dict = _version_admin_dict(selected, db) if selected is not None and selected.status == "DRAFT" else (
            _version_admin_dict(drafts[0], db) if len(drafts) == 1 else None
        )
        selected_dict = _version_admin_dict(selected, db) if selected is not None else None
        card = None
        if selected is not None and role in CARD_TYPE_ORDER:
            cards = db.scalars(select(FishCard).where(FishCard.asset_version_id == selected.id)).all()
            card = cards[0] if len(cards) == 1 else None
        review = None
        if selected is not None:
            review = db.scalar(select(FishKnowledgeAssetReview).where(
                FishKnowledgeAssetReview.version_id == selected.id,
                FishKnowledgeAssetReview.species_id == species.id,
                FishKnowledgeAssetReview.asset_role == role,
            ))

        revisions = []
        publication_history = []
        role_version_ids = [value.id for value in history]
        role_card_ids = db.scalars(select(FishCard.id).where(
            FishCard.asset_version_id.in_(role_version_ids),
        )).all() if role_version_ids else []
        if role_card_ids:
            revisions = db.scalars(select(FishCardContentRevision).where(
                FishCardContentRevision.card_id.in_(role_card_ids),
            ).order_by(FishCardContentRevision.created_at.desc(), FishCardContentRevision.id.desc())).all()
        publication_history = db.scalars(select(FishKnowledgePublicationAudit).where(
            FishKnowledgePublicationAudit.species_id == species.id,
            FishKnowledgePublicationAudit.asset_role == role,
        ).order_by(FishKnowledgePublicationAudit.created_at.desc(), FishKnowledgePublicationAudit.id.desc())).all()
        qa_history = db.scalars(select(FishKnowledgeAssetQAAudit).where(
            FishKnowledgeAssetQAAudit.species_id == species.id,
            FishKnowledgeAssetQAAudit.asset_role == role,
        ).order_by(FishKnowledgeAssetQAAudit.created_at.desc(), FishKnowledgeAssetQAAudit.id.desc())).all()

        roles[role] = {
            "asset_role": role,
            "active_version": active_dict,
            "active_versions": [_version_admin_dict(value, db) for value in active],
            "draft_version": draft_dict,
            "draft_versions": [_version_admin_dict(value, db) for value in drafts],
            "selected_version_id": selected.id if selected is not None else None,
            "selected_version": selected_dict,
            "source_sha256": (selected_dict or {}).get("source_sha256"),
            "image_url": selected.image_url if selected is not None else None,
            "asset_status": selected.status if selected is not None else "UNAVAILABLE",
            "card_id": card.id if card is not None else None,
            "card_title": card.title if card is not None else "",
            "structured_content": parse_card_content(card.description) if card is not None else {},
            "content_revision": int(card.content_revision or 1) if card is not None else None,
            "binding_status": (
                "BOUND_ACTIVE" if card is not None and card.status == "ACTIVE"
                else "BOUND_DRAFT" if card is not None and card.status == "DRAFT"
                else "MISSING" if role in CARD_TYPE_ORDER and selected is not None
                else "LEGACY_READ_ONLY" if legacy_cards
                else "ROLE_ONLY"
            ),
            "legacy_active": [{
                "card_id": value.id,
                "title": value.title,
                "image_url": value.image_url,
                "status": value.status,
            } for value in legacy_active_cards],
            "legacy_drafts": [{
                "card_id": value.id,
                "title": value.title,
                "image_url": value.image_url,
                "status": value.status,
            } for value in legacy_draft_cards],
            "visual_qa": review.visual_qa_result if review else "PENDING",
            "content_qa": review.content_qa_result if review else "PENDING",
            "publication_status": (
                "CONFLICT" if len(active) > 1 else "ACTIVE" if selected is not None and selected.status == "ACTIVE"
                else "DRAFT" if selected is not None and selected.status == "DRAFT"
                else "ARCHIVED" if selected is not None and selected.status == "ARCHIVED"
                else "LEGACY_ACTIVE" if legacy_active_cards
                else "LEGACY_DRAFT" if legacy_draft_cards
                else "UNPUBLISHED"
            ),
            "history": {
                "versions": [_version_admin_dict(value, db) for value in history],
                "content_revisions": [{
                    "id": value.id,
                    "card_id": value.card_id,
                    "version_id": value.asset_version_id,
                    "content_revision": value.content_revision,
                    "title": value.title,
                    "description": value.description,
                    "image_url": value.image_url,
                    "created_at": value.created_at.isoformat() if value.created_at else None,
                } for value in revisions],
                "publication_audits": [{
                    "id": value.id,
                    "version_id": value.asset_version_id,
                    "card_id": value.card_id,
                    "previous_version_id": value.previous_version_id,
                    "previous_card_id": value.previous_card_id,
                    "status": value.publication_status,
                    "validation": _read_json(value.validation_json, {}),
                    "actor": value.actor,
                    "created_at": value.created_at.isoformat() if value.created_at else None,
                } for value in publication_history],
                "qa_audits": [{
                    "id": value.id,
                    "version_id": value.version_id,
                    "stage": value.qa_stage,
                    "result": value.result,
                    "reviewer": value.reviewer,
                    "evidence_note": value.evidence_note,
                    "content_revision": value.content_revision,
                    "card_id": value.card_id,
                    "source_sha256": value.source_sha256,
                    "derived_media_sha256": value.derived_media_sha256,
                    "object_name": value.object_name,
                    "object_generation": value.object_generation,
                    "created_at": value.created_at.isoformat() if value.created_at else None,
                } for value in qa_history],
                "legacy_cards": [{
                    "card_id": value.id,
                    "status": value.status,
                    "title": value.title,
                    "image_url": value.image_url,
                    "structured_content": parse_card_content(value.description),
                } for value in legacy_cards],
            },
        }
    return {"species_id": species.id, "species_name_cn": species.name_cn, "roles": roles}


@asset_router.put("/species/{species_id}/roles/{asset_role}/content")
def update_bound_card_content(
    species_id: str,
    asset_role: str,
    payload: BoundCardContentPayload,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    from app.fish_knowledge.publication import save_bound_card_content

    role = asset_role.strip().upper()
    description = card_description(payload.structured_content)
    return save_bound_card_content(
        db,
        species_id=species_id,
        role=role,
        version_id=payload.version_id,
        title=payload.title,
        description=description,
        actor=payload.actor,
    )


@asset_router.post("/single-upload")
async def upload_single_asset_v13(
    species_id: str = Form(...),
    asset_role: str = Form(...),
    file: UploadFile = File(...),
    allow_warnings: bool = Form(False),
    preflight_only: bool = Form(False),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Stage and import one slot through the same batch scan/execute core."""

    role = asset_role.strip().upper()
    species = db.get(FishSpecies, species_id.strip())
    if role not in ASSET_ROLES:
        await file.close()
        raise HTTPException(status_code=400, detail={"code": "INVALID_ASSET_ROLE", "message": "未知资产角色"})
    if species is None or species.status == "DELETED":
        await file.close()
        raise HTTPException(status_code=404, detail={"code": "SPECIES_NOT_FOUND", "message": "鱼种不存在"})
    original_name = str(file.filename or "asset").replace("\\", "/").split("/")[-1]
    extension = "." + _image_extension(original_name)
    if extension not in EXTENSIONS:
        await file.close()
        raise HTTPException(status_code=400, detail={"code": "UNSUPPORTED_IMAGE_FORMAT", "message": "仅支持 PNG/JPG/JPEG/WEBP"})

    batch_id = "FK_SINGLE_" + uuid.uuid4().hex
    created = create_local_batch(CreateLocalBatchPayload(batch_id=batch_id), db)
    canonical_stems = {
        "COVER_LIST": "00_cover_list", "COVER_HERO": "00_cover_hero",
        "TRANSPARENT_MAIN": "01_transparent_main", "TRANSPARENT_ALT": "02_transparent_alt",
        "HERO": "01_hero", "IDENTIFICATION": "02_identification", "ECO": "03_ecology",
        "GEAR": "04_gear", "SKILL": "05_skill",
    }
    await upload_batch_file(batch_id, f"{species.id}/{canonical_stems[role]}{extension}", file, db)
    scan_batch(batch_id, db)
    batch = _batch_or_404(db, batch_id)
    item = db.scalar(select(FishAssetImportItem).where(FishAssetImportItem.batch_id == batch_id).order_by(FishAssetImportItem.id))
    if item is None:
        raise HTTPException(status_code=502, detail={"code": "SCAN_EMPTY", "message": "上传文件未生成预检记录"})
    item.source_filename = original_name
    db.commit()
    if item.validation_status == "INVALID":
        return {"batch_id": batch_id, "status": "READY", "validation_status": "INVALID", "next_action": "EDIT_OR_REJECT", "item": _item_dict(item, base=f"/api/v1/admin/fish/assets/import-batches/{batch_id}", db=db)}
    warnings = _read_json(item.validation_warnings, [])
    warnings_confirmed = allow_warnings is True
    if warnings and not warnings_confirmed:
        return {"batch_id": batch_id, "status": "READY", "validation_status": "WARNING", "needs_warning_confirmation": True, "next_action": "CONFIRM_WARNING", "item": _item_dict(item, base=f"/api/v1/admin/fish/assets/import-batches/{batch_id}", db=db)}
    if preflight_only is True:
        return {"batch_id": batch_id, "status": "READY", "validation_status": "VALID", "next_action": "UPLOAD_DRAFT", "item": _item_dict(item, base=f"/api/v1/admin/fish/assets/import-batches/{batch_id}", db=db)}
    result = execute_batch(batch_id, ExecuteBatchPayload(allow_warnings=warnings_confirmed), db)
    item = db.get(FishAssetImportItem, item.id)
    version = db.get(FishKnowledgeAssetVersion, item.version_id) if item and item.version_id else None
    if item is None or version is None:
        return {"batch_id": batch_id, "status": result.get("status"), "validation_status": "FAILED", "next_action": "RETRY_OR_DIAGNOSE", "item": _item_dict(item, base=f"/api/v1/admin/fish/assets/import-batches/{batch_id}", db=db) if item else None, "version": None}
    persisted_version = db.get(FishKnowledgeAssetVersion, version.id)
    return {
        "batch_id": batch_id,
        "status": result.get("status"),
        "validation_status": item.validation_status if item else "FAILED",
        "next_action": "DRAFT_CREATED" if persisted_version and persisted_version.status == "DRAFT" else "ALREADY_EXISTS",
        "item": _item_dict(item, base=f"/api/v1/admin/fish/assets/import-batches/{batch_id}", db=db) if item else None,
        "version": _version_admin_dict(persisted_version, db) if persisted_version else None,
    }


@asset_router.get("/versions/{version_id}/preview")
def preview_asset_version_v13(version_id: int, db: Session = Depends(get_db)) -> Response:
    version = db.get(FishKnowledgeAssetVersion, version_id)
    if version is None:
        raise HTTPException(status_code=404, detail={"code": "VERSION_NOT_FOUND", "message": "素材版本不存在"})
    try:
        client = storage.Client()
        blob = client.bucket(get_bucket_name()).blob(version.object_name)
        data = blob.download_as_bytes(timeout=120)
    except Exception as exc:
        raise HTTPException(status_code=502, detail={"code": "GCS_READ_FAILED", "message": str(exc)}) from exc
    return Response(content=data, media_type="image/webp", headers={"Cache-Control": "private, max-age=300", "X-Content-Type-Options": "nosniff"})


@asset_router.post("/versions/{version_id}/activate")
def activate_asset_version_v13(version_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    version = db.get(FishKnowledgeAssetVersion, version_id)
    if version is None or not version.batch_id:
        raise HTTPException(status_code=404, detail={"code": "VERSION_NOT_FOUND", "message": "素材版本不存在或不可发布"})
    return activate_version(version.batch_id, version.id, db)


@asset_router.post("/versions/{version_id}/public-api-check")
def record_public_api_readback_v14(
    version_id: int,
    payload: PublicAPIReadbackPayload,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Persist an operator's real public HTTP/API and image-byte verification."""

    version = db.get(FishKnowledgeAssetVersion, version_id)
    if version is None:
        raise HTTPException(status_code=404, detail={"code": "VERSION_NOT_FOUND", "message": "素材版本不存在"})
    if version.status != "ACTIVE":
        raise HTTPException(status_code=409, detail={"code": "PUBLIC_API_CHECK_REQUIRES_ACTIVE", "message": "公共 API 生效检查只接受真实 ACTIVE 版本"})
    role = _asset_role_for_version(version)
    audit = db.scalar(select(FishKnowledgePublicationAudit).where(
        FishKnowledgePublicationAudit.asset_version_id == version.id,
        FishKnowledgePublicationAudit.species_id == version.species_id,
        FishKnowledgePublicationAudit.asset_role == role,
    ).order_by(FishKnowledgePublicationAudit.created_at.desc(), FishKnowledgePublicationAudit.id.desc()))
    if audit is None:
        raise HTTPException(status_code=409, detail={"code": "PUBLICATION_AUDIT_MISSING", "message": "缺少此 ACTIVE 版本的发布审计记录"})

    if payload.status == "PUBLIC_API_OK":
        expected_media_sha = _read_json(version.metadata_json, {})
        expected_media_sha = str(expected_media_sha.get("derived_sha256") or "").lower() if isinstance(expected_media_sha, dict) else ""
        evidence_hashes = (payload.public_image_sha256 or "", payload.preview_image_sha256 or "")
        if payload.observed_version_id != version.id or any(not re.fullmatch(r"[0-9a-f]{64}", value.lower()) for value in evidence_hashes):
            raise HTTPException(status_code=400, detail={"code": "PUBLIC_API_EVIDENCE_INVALID", "message": "成功核验必须包含目标 version_id 与有效的公共/预览图片 SHA-256"})
        if evidence_hashes[0].lower() != evidence_hashes[1].lower() or evidence_hashes[0].lower() != expected_media_sha:
            raise HTTPException(status_code=409, detail={"code": "PUBLIC_API_IMAGE_MISMATCH", "message": "公共图片、目标版本预览与已存派生 SHA-256 不一致"})
        species = db.get(FishSpecies, version.species_id)
        if species is None or species.status != "ACTIVE":
            raise HTTPException(status_code=409, detail={"code": "SPECIES_NOT_PUBLIC", "message": "非 ACTIVE 鱼种不能记录为公共 API 已生效"})
        from app.fish_knowledge.api import build_species_full_detail, list_fish_species

        detail = build_species_full_detail(species, db)
        if role == "COVER_HERO":
            list_item = next((item for item in list_fish_species(db) if item.id == species.id), None)
            verified = bool(
                list_item
                and list_item.cover_hero_status == "ACTIVE"
                and list_item.cover_hero_version_id == version.id
                and list_item.cover_hero_image == version.image_url
                and detail.cover_hero_status == "ACTIVE"
                and detail.cover_hero_version_id == version.id
                and detail.cover_hero_image == version.image_url
            )
        elif role in {"HERO", "IDENTIFICATION", "ECO", "GEAR", "SKILL"}:
            asset = detail.knowledge_assets.get(role)
            card = next((item for item in detail.cards if item.card_type == role), None)
            verified = bool(
                asset
                and asset.get("asset_status") == "ACTIVE"
                and asset.get("version_id") == version.id
                and asset.get("image_url") == version.image_url
                and card
                and card.status == "ACTIVE"
                and card.species_id == species.id
                and card.card_type == role
                and card.asset_version_id == version.id
                and card.image_url == version.image_url
            )
        else:
            asset = detail.cover_assets.get(role)
            verified = bool(asset and asset.get("asset_status") == "ACTIVE" and asset.get("version_id") == version.id and asset.get("image_url") == version.image_url)
        if not verified:
            raise HTTPException(status_code=409, detail={"code": "PUBLIC_API_PROJECTION_MISMATCH", "message": "数据库公共 API 投影与已观测的 version_id 不一致"})

    validation = _read_json(audit.validation_json, {})
    validation = validation if isinstance(validation, dict) else {}
    checked_at = utcnow()
    validation["public_api_check"] = {
        "status": payload.status,
        "version_id": version.id,
        "species_id": version.species_id,
        "asset_role": role,
        "reviewer": payload.reviewer,
        "checked_at": checked_at.isoformat(),
        "observed_version_id": payload.observed_version_id,
        "public_image_sha256": payload.public_image_sha256,
        "preview_image_sha256": payload.preview_image_sha256,
        "detail": payload.detail,
    }
    audit.validation_json = _json(validation)
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail={"code": "PUBLIC_API_CHECK_NOT_SAVED", "message": "API 检查结果未能保存到发布审计；请重试"}) from exc
    return validation["public_api_check"]


@asset_router.post("/versions/{version_id}/client-check")
def record_client_acceptance_v14(
    version_id: int,
    payload: ClientAcceptancePayload,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Record independent Android/client acceptance for the exact ACTIVE asset."""

    version = db.get(FishKnowledgeAssetVersion, version_id)
    if version is None:
        raise HTTPException(status_code=404, detail={"code": "VERSION_NOT_FOUND", "message": "素材版本不存在"})
    if version.status != "ACTIVE":
        raise HTTPException(status_code=409, detail={"code": "CLIENT_CHECK_REQUIRES_ACTIVE", "message": "客户端实测只接受真实 ACTIVE 版本"})
    if payload.observed_version_id != version.id:
        raise HTTPException(status_code=400, detail={"code": "CLIENT_VERSION_MISMATCH", "message": "客户端实测必须确认此精确 ACTIVE version_id"})
    role = _asset_role_for_version(version)
    audit = db.scalar(select(FishKnowledgePublicationAudit).where(
        FishKnowledgePublicationAudit.asset_version_id == version.id,
        FishKnowledgePublicationAudit.species_id == version.species_id,
        FishKnowledgePublicationAudit.asset_role == role,
    ).order_by(FishKnowledgePublicationAudit.created_at.desc(), FishKnowledgePublicationAudit.id.desc()))
    if audit is None:
        raise HTTPException(status_code=409, detail={"code": "PUBLICATION_AUDIT_MISSING", "message": "缺少此 ACTIVE 版本的发布审计记录"})
    validation = _read_json(audit.validation_json, {})
    validation = validation if isinstance(validation, dict) else {}
    if payload.result == "CLIENT_PASSED":
        api_check = validation.get("public_api_check") if isinstance(validation.get("public_api_check"), dict) else {}
        if api_check.get("status") != "PUBLIC_API_OK" or api_check.get("version_id") != version.id:
            raise HTTPException(status_code=409, detail={"code": "PUBLIC_API_CHECK_REQUIRED", "message": "客户端实测通过前，必须先记录同一 version_id 的 PUBLIC_API_OK"})
    client_check = {
        "status": payload.result,
        "version_id": version.id,
        "species_id": version.species_id,
        "asset_role": role,
        "reviewer": payload.reviewer,
        "checked_at": utcnow().isoformat(),
        "observed_version_id": payload.observed_version_id,
        "detail": payload.detail,
    }
    validation["client_acceptance"] = client_check
    audit.validation_json = _json(validation)
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail={"code": "CLIENT_CHECK_NOT_SAVED", "message": "客户端实测结果未能保存到发布审计；请重试"}) from exc
    return client_check


@asset_router.put("/versions/{version_id}/review")
def review_asset_version_v13(version_id: int, payload: AssetReviewPayload, db: Session = Depends(get_db)) -> dict[str, Any]:
    if payload.visual_qa_result is None and payload.content_qa_result is None:
        raise HTTPException(status_code=400, detail={"code": "QA_STAGE_REQUIRED", "message": "至少提交视觉 QA 或内容 QA 其中一个阶段"})
    version = db.get(FishKnowledgeAssetVersion, version_id)
    if version is None:
        raise HTTPException(status_code=404, detail={"code": "VERSION_NOT_FOUND", "message": "素材版本不存在"})
    if version.status != "DRAFT":
        raise HTTPException(status_code=409, detail={"code": "QA_REQUIRES_DRAFT", "message": "QA 只能更新指定 DRAFT 版本；ACTIVE 和历史版本只读"})
    role = _asset_role_for_version(version)
    batch_id = (payload.batch_id or version.batch_id or "").strip()
    if not batch_id:
        raise HTTPException(status_code=404, detail={"code": "BATCH_NOT_FOUND", "message": "素材版本未关联导入批次"})
    batch = _batch_or_404(db, batch_id)
    review = db.scalar(select(FishKnowledgeAssetReview).where(
        FishKnowledgeAssetReview.batch_id == batch_id,
        FishKnowledgeAssetReview.species_id == version.species_id,
        FishKnowledgeAssetReview.asset_role == role,
    ))
    if review and review.frozen_at:
        raise HTTPException(status_code=409, detail={"code": "ASSET_FROZEN", "message": "冻结版本为只读；更新必须创建新版本"})
    if payload.visual_qa_result is not None and payload.content_qa_result is not None:
        raise HTTPException(status_code=400, detail={"code": "QA_STAGES_MUST_BE_SEPARATE", "message": "视觉 QA 与内容 QA 必须分别提交并留存审核记录"})
    item = db.scalar(select(FishAssetImportItem).where(
        FishAssetImportItem.batch_id == batch_id,
        FishAssetImportItem.species_id == version.species_id,
        FishAssetImportItem.asset_role == role,
        FishAssetImportItem.version_id == version.id,
    ))
    if item is None:
        raise HTTPException(status_code=404, detail={"code": "BATCH_VERSION_NOT_FOUND", "message": "该批次没有绑定到此角色版本的导入项"})
    if review and review.version_id != version.id:
        raise HTTPException(status_code=409, detail={"code": "REVIEW_VERSION_CONFLICT", "message": "该批次角色槽位已审核其他版本"})
    if review is None:
        metadata = _read_json(version.metadata_json, {})
        metadata = metadata if isinstance(metadata, dict) else {}
        source_matches = str(item.sha256 or "") == str(version.sha256 or "")
        derived_sha = str(metadata.get("derived_sha256") or "")
        valid_derived_sha = bool(re.fullmatch(r"[0-9a-fA-F]{64}", derived_sha))
        warnings = _read_json(item.validation_warnings, [])
        warning_acknowledged = not warnings or bool(batch.warnings_acknowledged and batch.warnings_acknowledged_at)
        review = FishKnowledgeAssetReview(
            batch_id=batch_id,
            species_id=version.species_id,
            asset_role=role,
            version_id=version.id,
            source_filename=item.source_filename,
            source_sha256=str(item.sha256 or ""),
            derived_media_sha256=derived_sha,
            object_name=version.object_name,
            object_generation=str(metadata.get("gcs_generation") or ""),
            validation_result="PASS" if item.validation_status == "IMPORTED" and source_matches and valid_derived_sha and warning_acknowledged else "INVALID",
            validation_warnings_json=_json(warnings),
        )
        db.add(review)
    reviewed_at = utcnow()
    if payload.visual_qa_result is not None:
        review.visual_qa_result = payload.visual_qa_result
        review.visual_qa_note = payload.visual_qa_note or payload.review_note
        review.visual_qa_reviewer = payload.visual_qa_reviewer or payload.reviewer
        review.visual_qa_reviewed_at = reviewed_at
    if payload.content_qa_result is not None:
        review.content_qa_result = payload.content_qa_result
        review.content_qa_note = payload.content_qa_note or payload.review_note
        review.content_qa_reviewer = payload.content_qa_reviewer or payload.reviewer
        review.content_qa_reviewed_at = reviewed_at
    review.review_note = payload.review_note or "；".join(filter(None, (review.visual_qa_note, review.content_qa_note)))
    review.reviewer = payload.reviewer
    review.reviewed_at = reviewed_at
    metadata = _read_json(version.metadata_json, {})
    metadata = metadata if isinstance(metadata, dict) else {}
    bound_cards = db.scalars(select(FishCard).where(FishCard.asset_version_id == version.id)).all()
    bound_card = bound_cards[0] if len(bound_cards) == 1 else None
    if payload.visual_qa_result is not None:
        db.add(FishKnowledgeAssetQAAudit(
            version_id=version.id,
            species_id=version.species_id,
            asset_role=role,
            qa_stage="VISUAL",
            result=payload.visual_qa_result,
            reviewer=payload.visual_qa_reviewer or payload.reviewer,
            evidence_note=payload.visual_qa_note or payload.review_note,
            content_revision=int(bound_card.content_revision or 1) if bound_card else None,
            card_id=bound_card.id if bound_card else None,
            source_sha256=str(version.sha256 or ""),
            derived_media_sha256=str(metadata.get("derived_sha256") or ""),
            object_name=version.object_name,
            object_generation=str(metadata.get("gcs_generation") or ""),
            created_at=reviewed_at,
        ))
    if payload.content_qa_result is not None:
        db.add(FishKnowledgeAssetQAAudit(
            version_id=version.id,
            species_id=version.species_id,
            asset_role=role,
            qa_stage="CONTENT",
            result=payload.content_qa_result,
            reviewer=payload.content_qa_reviewer or payload.reviewer,
            evidence_note=payload.content_qa_note or payload.review_note,
            content_revision=int(bound_card.content_revision or 1) if bound_card else None,
            card_id=bound_card.id if bound_card else None,
            source_sha256=str(version.sha256 or ""),
            derived_media_sha256=str(metadata.get("derived_sha256") or ""),
            object_name=version.object_name,
            object_generation=str(metadata.get("gcs_generation") or ""),
            created_at=reviewed_at,
        ))
    db.commit()
    return {
        "version_id": version.id,
        "batch_id": batch_id,
        "species_id": version.species_id,
        "asset_role": role,
        "validation_result": review.validation_result,
        "visual_qa_result": review.visual_qa_result,
        "visual_qa_note": review.visual_qa_note,
        "visual_qa_reviewer": review.visual_qa_reviewer,
        "visual_qa_reviewed_at": review.visual_qa_reviewed_at.isoformat() if review.visual_qa_reviewed_at else None,
        "content_qa_result": review.content_qa_result,
        "content_qa_note": review.content_qa_note,
        "content_qa_reviewer": review.content_qa_reviewer,
        "content_qa_reviewed_at": review.content_qa_reviewed_at.isoformat() if review.content_qa_reviewed_at else None,
        "frozen": bool(review.frozen_at),
    }


def _freeze_manifest_rows(db: Session, batch_id: str) -> list[dict[str, Any]]:
    rows = db.scalars(select(FishKnowledgeAssetReview).where(
        FishKnowledgeAssetReview.batch_id == batch_id,
        FishKnowledgeAssetReview.frozen_at.is_not(None),
    ).order_by(FishKnowledgeAssetReview.species_id, FishKnowledgeAssetReview.asset_role)).all()
    return [{
        "batch_id": row.batch_id,
        "species_id": row.species_id,
        "asset_role": row.asset_role,
        "source_filename": row.source_filename,
        "source_sha256": row.source_sha256,
        "derived_media_sha256": row.derived_media_sha256,
        "gcs_object_name": row.object_name,
        "gcs_generation": row.object_generation,
        "asset_version_id": row.version_id,
        "asset_status": row.asset_status,
        "validation_result": row.validation_result,
        "validation_warnings": _read_json(row.validation_warnings_json, []),
        "visual_qa_result": row.visual_qa_result,
        "visual_qa_note": row.visual_qa_note,
        "visual_qa_reviewer": row.visual_qa_reviewer,
        "visual_qa_reviewed_at": row.visual_qa_reviewed_at.isoformat() if row.visual_qa_reviewed_at else None,
        "content_qa_result": row.content_qa_result,
        "content_qa_note": row.content_qa_note,
        "content_qa_reviewer": row.content_qa_reviewer,
        "content_qa_reviewed_at": row.content_qa_reviewed_at.isoformat() if row.content_qa_reviewed_at else None,
        "warnings_acknowledged_at": row.warnings_acknowledged_at.isoformat() if row.warnings_acknowledged_at else None,
        "warnings_acknowledged_by": row.warnings_acknowledged_by,
        "binding_type": row.binding_type,
        "binding_id": row.binding_id,
        "binding_status": row.binding_status,
        "binding_image_url": row.binding_image_url,
        "cms_content_sha256": row.cms_content_sha256,
        "reviewer": row.reviewer,
        "review_note": row.review_note,
        "freeze_timestamp": row.frozen_at.isoformat() if row.frozen_at else None,
        "code_head": row.code_head,
    } for row in rows]


@asset_router.post("/batches/{batch_id}/freeze")
def freeze_asset_batch_v13(batch_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    batch = _batch_or_404(db, batch_id)
    if batch.status != "COMPLETED":
        raise HTTPException(status_code=409, detail={"code": "BATCH_NOT_COMPLETED", "message": "只有已完成导入的批次可冻结"})
    items = db.scalars(select(FishAssetImportItem).where(FishAssetImportItem.batch_id == batch_id).order_by(FishAssetImportItem.species_id, FishAssetImportItem.asset_role)).all()
    if not items:
        raise HTTPException(status_code=409, detail={"code": "EMPTY_BATCH", "message": "批次没有可冻结资产"})
    if any(item.validation_status != "IMPORTED" or not item.version_id for item in items):
        raise HTTPException(status_code=409, detail={"code": "IMPORT_NOT_CLOSED", "message": "存在未导入、失败或无版本绑定的素材"})
    if len({(item.species_id, item.asset_role) for item in items}) != len(items):
        raise HTTPException(status_code=409, detail={"code": "DUPLICATE_SLOT", "message": "批次存在重复鱼种/角色槽位"})
    if any((item.asset_role or "") not in {"HERO", "IDENTIFICATION", "ECO", "GEAR", "SKILL"} for item in items):
        raise HTTPException(status_code=409, detail={"code": "UNEXPECTED_ROLE", "message": "本冻结批次只能包含五张知识卡角色"})
    if batch_id == "FK_KNOWLEDGE_V2_20X5_20261008_001":
        expected_roles = {"HERO", "IDENTIFICATION", "ECO", "GEAR", "SKILL"}
        by_species: dict[str, set[str]] = {}
        for item in items:
            if item.species_id and item.asset_role:
                by_species.setdefault(item.species_id, set()).add(item.asset_role)
        if len(items) != 100 or len(by_species) != 20 or any(roles != expected_roles for roles in by_species.values()):
            raise HTTPException(status_code=409, detail={"code": "KNOWLEDGE_20X5_INCOMPLETE", "message": "正式 V2 冻结要求恰好 20 个 canonical species × 5 个知识卡角色（100 槽位）"})
    existing = _freeze_manifest_rows(db, batch_id)
    if existing:
        return {"batch_id": batch_id, "frozen": len(existing), "manifest": existing, "idempotent": True}
    code_head = os.getenv("APP_GIT_COMMIT", "").strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", code_head):
        raise HTTPException(status_code=409, detail={"code": "CODE_HEAD_UNAVAILABLE", "message": "冻结要求部署版本提供 40 位 APP_GIT_COMMIT"})
    reviews = []
    binding_snapshots = []
    for item in items:
        version = db.get(FishKnowledgeAssetVersion, item.version_id)
        if version is None or version.species_id != item.species_id or _asset_role_for_version(version) != item.asset_role:
            raise HTTPException(status_code=409, detail={"code": "VERSION_BINDING_MISMATCH", "message": f"{item.species_id}/{item.asset_role} 的版本绑定不一致"})
        bound_cards = db.scalars(select(FishCard).where(
            FishCard.species_id == item.species_id,
            FishCard.card_type == normalize_card_type(item.asset_role),
            FishCard.image_url == version.image_url,
        ).order_by(FishCard.status.desc(), FishCard.updated_at.desc(), FishCard.id.desc())).all()
        binding = next((value for value in bound_cards if value.status == "ACTIVE"), bound_cards[0] if bound_cards else None)
        if binding is None:
            raise HTTPException(status_code=409, detail={"code": "CMS_BINDING_MISSING", "message": f"{item.species_id}/{item.asset_role} 图片版本尚未绑定到 CMS 卡片"})
        structured_content = parse_card_content(binding.description)
        if not structured_content:
            raise HTTPException(status_code=409, detail={"code": "STRUCTURED_CONTENT_MISSING", "message": f"{item.species_id}/{item.asset_role} 缺少可供内容 QA 对照的结构化知识"})
        structured_snapshot = json.dumps(structured_content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        binding_snapshots.append({
            "binding_type": "FISH_CARD",
            "binding_id": binding.id,
            "binding_status": binding.status,
            "binding_image_url": binding.image_url,
            "cms_content_sha256": hashlib.sha256(structured_snapshot.encode("utf-8")).hexdigest(),
        })
        metadata = _read_json(version.metadata_json, {})
        metadata = metadata if isinstance(metadata, dict) else {}
        derived_sha = str(metadata.get("derived_sha256") or "")
        source_sha = str(metadata.get("source_sha256") or version.sha256 or "")
        generation = str(metadata.get("gcs_generation") or "")
        if (
            str(item.sha256 or "") != str(version.sha256 or "")
            or source_sha != str(item.sha256 or "")
            or not re.fullmatch(r"[0-9a-fA-F]{64}", derived_sha)
            or not generation.isdigit()
            or item.target_object != version.object_name
        ):
            raise HTTPException(status_code=409, detail={"code": "ASSET_PROVENANCE_MISMATCH", "message": f"{item.species_id}/{item.asset_role} 的 SHA、GCS generation 或对象路径证据不完整"})
        review = db.scalar(select(FishKnowledgeAssetReview).where(
            FishKnowledgeAssetReview.batch_id == batch_id,
            FishKnowledgeAssetReview.species_id == item.species_id,
            FishKnowledgeAssetReview.asset_role == item.asset_role,
        ))
        if review is None or review.version_id != item.version_id or review.validation_result != "PASS" or review.visual_qa_result != "PASS" or review.content_qa_result != "PASS":
            raise HTTPException(status_code=409, detail={"code": "QA_NOT_PASSED", "message": f"{item.species_id}/{item.asset_role} 尚未完成三项 PASS 审核"})
        reviews.append(review)
    frozen_at = utcnow()
    for review, binding in zip(reviews, binding_snapshots, strict=True):
        review.frozen_at = frozen_at
        review.code_head = code_head
        review.binding_type = binding["binding_type"]
        review.binding_id = binding["binding_id"]
        review.binding_status = binding["binding_status"]
        review.binding_image_url = binding["binding_image_url"]
        review.cms_content_sha256 = binding["cms_content_sha256"]
        review.asset_status = db.get(FishKnowledgeAssetVersion, review.version_id).status
        if _read_json(review.validation_warnings_json, []):
            review.warnings_acknowledged_at = batch.warnings_acknowledged_at
            review.warnings_acknowledged_by = batch.warnings_acknowledged_by
    db.commit()
    manifest = _freeze_manifest_rows(db, batch_id)
    return {"batch_id": batch_id, "frozen": len(manifest), "manifest": manifest, "idempotent": False}


@asset_router.get("/batches/{batch_id}/freeze-manifest")
def get_freeze_manifest_v13(batch_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    rows = _freeze_manifest_rows(db, batch_id)
    if not rows:
        raise HTTPException(status_code=404, detail={"code": "FREEZE_MANIFEST_NOT_FOUND", "message": "该批次尚无冻结清单"})
    return {"batch_id": batch_id, "frozen_count": len(rows), "manifest": rows}


@asset_router.get("/batches/{batch_id}/freeze-manifest.csv")
def download_freeze_manifest_csv_v13(batch_id: str, db: Session = Depends(get_db)) -> Response:
    rows = _freeze_manifest_rows(db, batch_id)
    if not rows:
        raise HTTPException(status_code=404, detail={"code": "FREEZE_MANIFEST_NOT_FOUND", "message": "该批次尚无冻结清单"})
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return Response(content=output.getvalue(), media_type="text/csv; charset=utf-8", headers={"Content-Disposition": f'attachment; filename="{batch_id}-freeze-manifest.csv"'})
