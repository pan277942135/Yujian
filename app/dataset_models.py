from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint

from app.db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DatasetItem(Base):
    """Immutable lineage row linking a frozen Dataset version to source ImageAsset."""

    __tablename__ = "dataset_items"
    __table_args__ = (
        UniqueConstraint("dataset_version", "image_asset_id", name="uq_dataset_item_version_image"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_version = Column(String(128), ForeignKey("datasets.dataset_version"), nullable=False, index=True)
    image_asset_id = Column(Integer, ForeignKey("image_assets.id"), nullable=False, index=True)
    batch_id = Column(String(128), nullable=False, index=True)
    image_id = Column(String(256), nullable=False, index=True)
    gcs_uri = Column(Text, nullable=False)
    species_key = Column(String(128), nullable=False, index=True)
    species_name = Column(String(128), nullable=False, index=True)
    class_index = Column(Integer, nullable=False)
    split = Column(String(16), nullable=False, index=True)
    presence_status = Column(String(32))
    duplicate_group = Column(String(128))
    group_id = Column(String(256))
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
