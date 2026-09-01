from __future__ import annotations

from sqlalchemy import CheckConstraint, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from app.db import Base
from app.models import utcnow


RANKING_TYPES = {"MAX_WEIGHT", "MAX_LENGTH", "MOST_CATCHES"}


class FishRanking(Base):
    """Reserved ranking record; v1 intentionally exposes no ranking API."""

    __tablename__ = "fish_ranking"
    __table_args__ = (
        CheckConstraint(
            "type IN ('MAX_WEIGHT', 'MAX_LENGTH', 'MOST_CATCHES')",
            name="ck_fish_ranking_type",
        ),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    species_id = Column(
        String(128),
        ForeignKey("fish_species.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    type = Column(String(32), nullable=False, index=True)
    value = Column(Float, nullable=False)
    location = Column(String(256))
    user_id = Column(String(256), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)

    species = relationship("FishSpecies")
