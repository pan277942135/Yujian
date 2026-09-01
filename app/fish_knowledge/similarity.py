from __future__ import annotations

from sqlalchemy import CheckConstraint, Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship

from app.db import Base
from app.models import utcnow


class FishSimilarity(Base):
    __tablename__ = "fish_similarity"
    __table_args__ = (
        CheckConstraint("species_id <> similar_species_id", name="ck_fish_similarity_not_self"),
        UniqueConstraint("species_id", "similar_species_id", name="uq_fish_similarity_pair"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    species_id = Column(
        String(128),
        ForeignKey("fish_species.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    similar_species_id = Column(
        String(128),
        ForeignKey("fish_species.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    difference = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)

    species = relationship("FishSpecies", foreign_keys=[species_id], back_populates="similarities")
    similar_species = relationship("FishSpecies", foreign_keys=[similar_species_id])
