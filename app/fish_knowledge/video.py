from __future__ import annotations

from sqlalchemy import CheckConstraint, Column, DateTime, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship

from app.db import Base
from app.models import utcnow


VIDEO_TYPES = {"INTRO", "HOW_TO_FISH", "REAL_CATCH", "EQUIPMENT"}


class FishVideo(Base):
    __tablename__ = "fish_video"
    __table_args__ = (
        CheckConstraint(
            "type IN ('INTRO', 'HOW_TO_FISH', 'REAL_CATCH', 'EQUIPMENT')",
            name="ck_fish_video_type",
        ),
        UniqueConstraint("species_id", "video_url", name="uq_fish_video_species_url"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    species_id = Column(
        String(128),
        ForeignKey("fish_species.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title = Column(String(256), nullable=False)
    type = Column(String(32), nullable=False)
    cover_url = Column(Text)
    video_url = Column(Text, nullable=False)
    duration = Column(Integer, nullable=False)
    tags = Column(JSON, nullable=False, default=list)
    sort_order = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)

    species = relationship("FishSpecies", back_populates="videos")
