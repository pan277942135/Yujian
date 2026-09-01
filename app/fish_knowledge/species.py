from __future__ import annotations

from sqlalchemy import CheckConstraint, Column, DateTime, ForeignKey, JSON, String, Text
from sqlalchemy.orm import relationship

from app.db import Base
from app.models import utcnow


class FishSpecies(Base):
    """Consumer-facing fish species master data.

    ``id`` deliberately reuses ``SpeciesCatalog.species_key`` so model output,
    reviewed data, and App knowledge pages share one stable machine identity.
    """

    __tablename__ = "fish_species"
    __table_args__ = (
        CheckConstraint("status IN ('ACTIVE', 'DRAFT')", name="ck_fish_species_status"),
    )

    id = Column(
        String(128),
        ForeignKey("species_catalog.species_key", ondelete="RESTRICT"),
        primary_key=True,
    )
    name_cn = Column(String(128), nullable=False, unique=True, index=True)
    alias = Column(JSON, nullable=False, default=list)
    scientific_name = Column(String(256))
    category = Column(String(64), nullable=False)
    family = Column(String(128))
    genus = Column(String(128))
    summary = Column(Text, nullable=False, default="")
    status = Column(String(16), nullable=False, default="DRAFT", index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)

    gallery = relationship(
        "FishGalleryImage",
        back_populates="species",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="FishGalleryImage.sort_order",
    )
    profile = relationship(
        "FishProfile",
        back_populates="species",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )
    fishing = relationship(
        "FishFishing",
        back_populates="species",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )
    videos = relationship(
        "FishVideo",
        back_populates="species",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="FishVideo.sort_order",
    )
    similarities = relationship(
        "FishSimilarity",
        foreign_keys="FishSimilarity.species_id",
        back_populates="species",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
