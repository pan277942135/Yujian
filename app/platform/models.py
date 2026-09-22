"""Additive Platform V1 index tables.

These tables deliberately contain operational indexes only.  The source of
truth for datasets, reviews, training and model versions remains the existing
registry schema.  Platform API responses never expose the internal URI
columns stored here.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint

from app.db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PipelineRun(Base):
    """Trace one production pipeline without changing existing workers."""

    __tablename__ = "pipeline_run"

    run_id = Column(String(128), primary_key=True)
    source_batch_id = Column(String(128), index=True)
    source_image_id = Column(String(256), index=True)
    pipeline_type = Column(String(64), nullable=False, default="FISH_ASSET")
    status = Column(String(32), nullable=False, default="QUEUED", index=True)
    current_stage = Column(String(64))
    stage_json = Column(Text, nullable=False, default="{}")
    model_version = Column(String(128))
    started_at = Column(DateTime(timezone=True))
    finished_at = Column(DateTime(timezone=True))
    duration_ms = Column(Integer)
    error_stage = Column(String(64))
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, index=True)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)


class FishAsset(Base):
    """Index of outputs produced by the existing completion pipeline."""

    __tablename__ = "fish_asset"

    asset_id = Column(String(128), primary_key=True)
    pipeline_run_id = Column(String(128), ForeignKey("pipeline_run.run_id", ondelete="SET NULL"), index=True)
    source_batch_id = Column(String(128), index=True)
    source_image_id = Column(String(256), index=True)
    species = Column(String(128), index=True)
    status = Column(String(32), nullable=False, default="ACTIVE", index=True)
    original_uri = Column(Text)
    mask_uri = Column(Text)
    transparent_uri = Column(Text)
    sticker_uri = Column(Text)
    version = Column(String(64), nullable=False, default="v1")
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, index=True)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)


class PlatformOperationLog(Base):
    """Small audit trail for Platform actions and adapter failures."""

    __tablename__ = "platform_operation_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    operation_type = Column(String(64), nullable=False, index=True)
    resource_type = Column(String(64), nullable=False, index=True)
    resource_id = Column(String(128), index=True)
    status = Column(String(32), nullable=False, default="SUCCESS", index=True)
    message = Column(Text)
    detail_json = Column(Text)
    actor = Column(String(256), default="platform")
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, index=True)


class BsideVisualSession(Base):
    """One CPU-only B-side demo session derived from one Qwen output."""

    __tablename__ = "qwen_bside_visual_session"

    session_id = Column(String(128), primary_key=True)
    source_qwen_run_id = Column(
        String(128),
        ForeignKey("pipeline_run.run_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    # New sessions start from the persisted Qwen RGB result. The legacy
    # transparent URI column remains for additive upgrades and historical
    # sessions that were created before Step 1 moved into this workflow.
    source_qwen_rgb_uri = Column(Text, nullable=True)
    source_transparent_fish_uri = Column(Text, nullable=True)
    status = Column(String(32), nullable=False, default="ACTIVE", index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, index=True)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)


class BsideVisualStep(Base):
    """Latest pointer for one versioned B-side processing step."""

    __tablename__ = "qwen_bside_visual_step"
    __table_args__ = (
        UniqueConstraint("session_id", "step_key", name="uq_qwen_bside_visual_step"),
    )

    step_id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        String(128),
        ForeignKey("qwen_bside_visual_session.session_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    step_key = Column(String(32), nullable=False)
    status = Column(String(32), nullable=False, default="NOT_STARTED", index=True)
    output_uri = Column(Text)
    preview_uri = Column(Text)
    metadata_json = Column(Text, nullable=False, default="{}")
    style_id = Column(String(64))
    template_id = Column(String(64))
    version = Column(Integer, nullable=False, default=0)
    error_code = Column(String(64))
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)


__all__ = [
    "BsideVisualSession",
    "BsideVisualStep",
    "FishAsset",
    "PipelineRun",
    "PlatformOperationLog",
]
