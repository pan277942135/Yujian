from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship

from app.db import Base


def utcnow():
    return datetime.now(timezone.utc)


class Batch(Base):
    __tablename__ = "batches"

    batch_id = Column(String(128), primary_key=True)
    source = Column(String(128), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    image_count = Column(Integer, nullable=False, default=0)
    manifest_uri = Column(Text, nullable=False)
    raw_uri = Column(Text, nullable=False)
    status = Column(String(64), nullable=False, default="INGESTED")
    notes = Column(Text)

    images = relationship("ImageAsset", back_populates="batch", cascade="all, delete-orphan")


class SpeciesCatalog(Base):
    """Stable species identity independent from any one model's class index."""

    __tablename__ = "species_catalog"

    species_key = Column(String(128), primary_key=True)
    catalog_order = Column(Integer, nullable=False, unique=True, index=True)
    common_name_zh = Column(String(128), nullable=False, unique=True, index=True)
    common_name_en = Column(String(256))
    scientific_name = Column(String(256))
    status = Column(String(32), nullable=False, default="candidate", index=True)  # candidate/active/retired
    is_other = Column(Boolean, nullable=False, default=False)
    notes = Column(Text)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)


class ImageAsset(Base):
    __tablename__ = "image_assets"
    __table_args__ = (UniqueConstraint("batch_id", "image_id", name="uq_image_batch_image"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_id = Column(String(128), ForeignKey("batches.batch_id"), nullable=False, index=True)
    image_id = Column(String(256), nullable=False, index=True)
    file_name = Column(String(512), nullable=False)
    object_name = Column(Text, nullable=False)
    gcs_uri = Column(Text, nullable=False)
    source_url = Column(Text)
    source_platform = Column(String(128))
    claimed_species = Column(String(128))
    truth_species = Column(String(128), index=True)
    truth_status = Column(String(64), nullable=False, default="UNCERTAIN", index=True)
    review_status = Column(String(64), nullable=False, default="pending", index=True)
    scene = Column(String(64))
    lighting = Column(String(64))
    quality = Column(String(64))
    group_id = Column(String(256), index=True)
    notes = Column(Text)
    reviewed_by = Column(String(256))
    reviewed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)

    batch = relationship("Batch", back_populates="images")
    review_events = relationship("ReviewEvent", back_populates="image", cascade="all, delete-orphan")


class ReviewEvent(Base):
    __tablename__ = "review_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_asset_id = Column(Integer, ForeignKey("image_assets.id"), nullable=False, index=True)
    action = Column(String(64), nullable=False)
    reviewer = Column(String(256))
    before_json = Column(Text)
    after_json = Column(Text)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

    image = relationship("ImageAsset", back_populates="review_events")


class DatasetVersion(Base):
    __tablename__ = "datasets"

    dataset_version = Column(String(128), primary_key=True)
    parent_version = Column(String(128), ForeignKey("datasets.dataset_version"))
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    manifest_uri = Column(Text, nullable=False)
    class_map_uri = Column(Text)
    train_count = Column(Integer, nullable=False, default=0)
    val_count = Column(Integer, nullable=False, default=0)
    test_count = Column(Integer, nullable=False, default=0)
    species_count = Column(Integer, nullable=False, default=0)
    gold_version = Column(String(128))
    git_commit = Column(String(128), nullable=False)
    selection_mode = Column(String(64), nullable=False, default="ALL_APPROVED")
    source_cutoff_at = Column(DateTime(timezone=True))
    status = Column(String(64), nullable=False)


class FeedbackEvent(Base):
    """Online inference feedback that feeds back into the data factory."""

    __tablename__ = "feedback_events"
    __table_args__ = (UniqueConstraint("source_event_id", name="uq_feedback_source_event"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_event_id = Column(String(256), nullable=False, index=True)
    source = Column(String(64), nullable=False, default="app")
    image_gcs_uri = Column(Text)
    model_version = Column(String(128), index=True)
    predicted_species = Column(String(128), index=True)
    confidence = Column(Float)
    feedback_type = Column(String(64), nullable=False, index=True)  # confirmed/corrected/unknown/new_species_candidate
    corrected_species = Column(String(128), index=True)
    user_note = Column(Text)
    pipeline_status = Column(String(64), nullable=False, default="NEW", index=True)  # NEW/BATCHED/REVIEWED/IGNORED
    materialized_batch_id = Column(String(128), ForeignKey("batches.batch_id"))
    materialized_image_id = Column(String(256))
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)


class TrainingRun(Base):
    __tablename__ = "training_runs"

    run_id = Column(String(128), primary_key=True)
    dataset_version = Column(String(128), ForeignKey("datasets.dataset_version"), nullable=False, index=True)
    git_commit = Column(String(128), nullable=False)
    model_family = Column(String(128), nullable=False)
    params_json = Column(Text, nullable=False)
    seed = Column(Integer)
    started_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    finished_at = Column(DateTime(timezone=True))
    status = Column(String(64), nullable=False)
    artifact_uri = Column(Text)
    metrics_uri = Column(Text)


class ModelVersion(Base):
    __tablename__ = "models"

    model_version = Column(String(128), primary_key=True)
    run_id = Column(String(128), ForeignKey("training_runs.run_id"), nullable=False, unique=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    artifact_uri = Column(Text, nullable=False)
    metrics_uri = Column(Text)
    status = Column(String(64), nullable=False)
    notes = Column(Text)


class Evaluation(Base):
    __tablename__ = "evaluations"

    evaluation_id = Column(String(128), primary_key=True)
    model_version = Column(String(128), ForeignKey("models.model_version"), nullable=False)
    gold_version = Column(String(128))
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    metrics_uri = Column(Text, nullable=False)
    confusion_matrix_uri = Column(Text)
    errors_uri = Column(Text)


class ErrorCase(Base):
    __tablename__ = "error_pool"

    error_id = Column(String(128), primary_key=True)
    evaluation_id = Column(String(128), ForeignKey("evaluations.evaluation_id"), nullable=False)
    image_id = Column(String(256), nullable=False)
    truth_species = Column(String(128))
    predicted_species = Column(String(128))
    confidence = Column(Float)
    hard_pair_type = Column(String(128))
    scene = Column(String(64))
    status = Column(String(64), nullable=False, default="OPEN")
    source_uri = Column(Text)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
