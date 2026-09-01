from __future__ import annotations

from sqlalchemy import Column, DateTime, ForeignKey, JSON, String, Text
from sqlalchemy.orm import relationship

from app.db import Base
from app.models import utcnow


class FishProfile(Base):
    __tablename__ = "fish_profile"

    species_id = Column(
        String(128),
        ForeignKey("fish_species.id", ondelete="CASCADE"),
        primary_key=True,
    )
    body_shape = Column(Text)
    features = Column(JSON, nullable=False, default=list)
    habitat = Column(JSON, nullable=False, default=list)
    food = Column(Text)
    season = Column(JSON, nullable=False, default=list)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)

    species = relationship("FishSpecies", back_populates="profile")
