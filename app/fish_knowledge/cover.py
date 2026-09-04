from __future__ import annotations

from sqlalchemy import CheckConstraint, Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship

from app.db import Base
from app.models import utcnow


COVER_STYLES = frozenset({"ANIME_CARD"})


class FishSpeciesCover(Base):
    """The single square illustrated cover used on the species list page."""

    __tablename__ = "fish_species_cover"
    __table_args__ = (
        CheckConstraint("status IN ('ACTIVE', 'DRAFT')", name="ck_fish_species_cover_status"),
        UniqueConstraint("species_id", name="uq_fish_species_cover_species"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    species_id = Column(
        String(128),
        ForeignKey("fish_species.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    image_url = Column(Text, nullable=False, default="")
    style = Column(String(64), nullable=False, default="ANIME_CARD")
    title = Column(String(256), nullable=False, default="")
    status = Column(String(16), nullable=False, default="DRAFT", index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)

    species = relationship("FishSpecies", back_populates="cover")
