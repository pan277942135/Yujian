from __future__ import annotations

from sqlalchemy import CheckConstraint, Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship

from app.db import Base
from app.models import utcnow


GALLERY_TYPES = {"standard", "side", "top", "catch", "environment", "action"}


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
