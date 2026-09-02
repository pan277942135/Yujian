from __future__ import annotations

import hashlib
import io

from PIL import Image, UnidentifiedImageError
from sqlalchemy import CheckConstraint, Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship

from app.db import Base
from app.models import utcnow


GALLERY_TYPES = {"standard", "side", "top", "catch", "environment", "action"}
MAX_GALLERY_IMAGES = 5
MAX_GALLERY_IMAGE_BYTES = 15 * 1024 * 1024
MAX_GALLERY_IMAGE_PIXELS = 40_000_000
GALLERY_MIME_SUFFIXES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


class GalleryUploadError(ValueError):
    pass


def inspect_gallery_image(data: bytes) -> dict[str, str | int]:
    if not data:
        raise GalleryUploadError("图片文件为空")
    if len(data) > MAX_GALLERY_IMAGE_BYTES:
        raise GalleryUploadError("图片不能超过 15 MB")
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            image_format = (image.format or "").upper()
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise GalleryUploadError("仅支持 JPEG、PNG 或 WEBP 图片") from exc
    if width <= 0 or height <= 0 or width * height > MAX_GALLERY_IMAGE_PIXELS:
        raise GalleryUploadError("图片尺寸非法或像素数量过大")
    content_type = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
    }.get(image_format)
    if content_type is None:
        raise GalleryUploadError("仅支持 JPEG、PNG 或 WEBP 图片")
    return {
        "content_type": content_type,
        "suffix": GALLERY_MIME_SUFFIXES[content_type],
        "width": width,
        "height": height,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def store_gallery_image(
    *,
    client,
    bucket,
    species_id: str,
    data: bytes,
    metadata: dict[str, str | int],
) -> tuple[str, str]:
    digest = str(metadata["sha256"])
    suffix = str(metadata["suffix"])
    object_name = f"fish_knowledge/{species_id}/gallery/{digest}{suffix}"
    blob = bucket.blob(object_name)
    if blob.exists(client):
        existing = blob.download_as_bytes()
        if hashlib.sha256(existing).hexdigest() != digest:
            raise GalleryUploadError("同名知识库图片已存在但内容不一致")
        return object_name, "SKIP"
    try:
        blob.upload_from_string(
            data,
            content_type=str(metadata["content_type"]),
            if_generation_match=0,
        )
    except Exception:
        if blob.exists(client):
            existing = blob.download_as_bytes()
            if hashlib.sha256(existing).hexdigest() == digest:
                return object_name, "SKIP"
        raise
    return object_name, "CREATED"


def store_knowledge_asset(
    *,
    client,
    bucket,
    species_id: str,
    asset_type: str,
    data: bytes,
    metadata: dict[str, str | int],
) -> tuple[str, str]:
    """Store a reviewed cover/card image under a deterministic GCS path."""

    digest = str(metadata["sha256"])
    suffix = str(metadata["suffix"])
    object_name = f"fish_knowledge/{species_id}/{asset_type.lower()}/{digest}{suffix}"
    blob = bucket.blob(object_name)
    if blob.exists(client):
        existing = blob.download_as_bytes()
        if hashlib.sha256(existing).hexdigest() != digest:
            raise GalleryUploadError("同名知识库素材已存在但内容不一致")
        return object_name, "SKIP"
    try:
        blob.upload_from_string(
            data,
            content_type=str(metadata["content_type"]),
            if_generation_match=0,
        )
    except Exception:
        if blob.exists(client):
            existing = blob.download_as_bytes()
            if hashlib.sha256(existing).hexdigest() == digest:
                return object_name, "SKIP"
        raise
    return object_name, "CREATED"


class FishGalleryImage(Base):
    __tablename__ = "fish_gallery"
    __table_args__ = (
        CheckConstraint(
            "type IN ('standard', 'side', 'top', 'catch', 'environment', 'action')",
            name="ck_fish_gallery_type",
        ),
        UniqueConstraint("species_id", "sort_order", name="uq_fish_gallery_species_order"),
        UniqueConstraint("species_id", "url", name="uq_fish_gallery_species_url"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    species_id = Column(
        String(128),
        ForeignKey("fish_species.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    type = Column(String(32), nullable=False)
    url = Column(Text, nullable=False)
    title = Column(String(256))
    sort_order = Column(Integer, nullable=False, default=0)
    object_name = Column(Text)
    content_type = Column(String(128))
    size_bytes = Column(Integer)
    sha256 = Column(String(64))
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)

    species = relationship("FishSpecies", back_populates="gallery")
