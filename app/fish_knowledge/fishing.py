from __future__ import annotations

from sqlalchemy import Column, DateTime, ForeignKey, JSON, String, Text
from sqlalchemy.orm import relationship

from app.db import Base
from app.models import utcnow


class FishFishing(Base):
    __tablename__ = "fish_fishing"

    species_id = Column(
        String(128),
        ForeignKey("fish_species.id", ondelete="CASCADE"),
        primary_key=True,
    )
    water_layer = Column(String(128))
    season = Column(JSON, nullable=False, default=list)
    bait = Column(JSON, nullable=False, default=list)
    method = Column(JSON, nullable=False, default=list)
    summary = Column(Text, nullable=False, default="")
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)

    species = relationship("FishSpecies", back_populates="fishing")
