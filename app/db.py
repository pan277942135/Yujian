import os
from pathlib import Path

from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import URL
from sqlalchemy.orm import declarative_base, sessionmaker


CLOUD_SQL_CONNECTION_NAME = os.getenv("CLOUD_SQL_CONNECTION_NAME", "").strip()

if CLOUD_SQL_CONNECTION_NAME:
    db_user = os.getenv("DB_USER", "yujian_console").strip()
    db_password = os.getenv("DB_PASSWORD", "")
    db_name = os.getenv("DB_NAME", "yujian_registry").strip()
    if not db_password:
        raise RuntimeError("DB_PASSWORD is required when CLOUD_SQL_CONNECTION_NAME is configured")

    database_url = URL.create(
        drivername="postgresql+psycopg",
        username=db_user,
        password=db_password,
        database=db_name,
    )
    engine = create_engine(
        database_url,
        connect_args={"host": f"/cloudsql/{CLOUD_SQL_CONNECTION_NAME}"},
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        pool_recycle=1800,
    )
else:
    DATABASE_URL = os.getenv("REGISTRY_DB_URL", "sqlite:///./var/yujian_registry.db")
    if DATABASE_URL.startswith("sqlite:///"):
        sqlite_path = DATABASE_URL.removeprefix("sqlite:///")
        if sqlite_path and sqlite_path != ":memory:":
            Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    else:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from app import models  # noqa: F401
    from app import fish_knowledge  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _ensure_production_pipeline_columns()
    _ensure_fish_knowledge_crud_constraints()


def _ensure_production_pipeline_columns() -> None:
    """Apply additive v2 columns for installations created before the migration.

    The console intentionally has no destructive migration path.  These fixed
    identifiers match schemas/0013_production_pipeline_v2.sql and are safe to
    run at every startup on SQLite and PostgreSQL.
    """

    additions = {
        "datasets": {
            "pipeline_type": "VARCHAR(64) NOT NULL DEFAULT 'WHOLE_IMAGE_V1'",
            "metadata_json": "TEXT",
        },
        "training_runs": {
            "pipeline_type": "VARCHAR(64) NOT NULL DEFAULT 'WHOLE_IMAGE_V1'",
            "detector_version": "VARCHAR(128)",
            "crop_version": "VARCHAR(128)",
            "classifier_version": "VARCHAR(128)",
        },
        "models": {
            "pipeline_type": "VARCHAR(64) NOT NULL DEFAULT 'WHOLE_IMAGE_V1'",
            "detector_version": "VARCHAR(128)",
            "crop_version": "VARCHAR(128)",
            "classifier_version": "VARCHAR(128)",
            "dataset_version": "VARCHAR(128)",
        },
        "batch_crop_reviews": {
            "detector_version": "VARCHAR(128)",
        },
        "dataset_crop_reviews": {
            "detector_version": "VARCHAR(128)",
            "detector_confidence": "DOUBLE PRECISION",
            "bbox_area_ratio": "DOUBLE PRECISION",
            "aspect_ratio": "DOUBLE PRECISION",
            "quality_score": "DOUBLE PRECISION",
            "quality_status": "VARCHAR(32)",
            "all_detections_json": "TEXT",
            "detector_error": "TEXT",
            "crop_uri": "TEXT",
            "crop_status": "VARCHAR(32)",
            "crop_error": "TEXT",
        },
        "inference_assets": {
            "source_batch": "VARCHAR(128)",
        },
    }
    inspector = inspect(engine)
    with engine.begin() as connection:
        for table, columns in additions.items():
            if not inspector.has_table(table):
                continue
            existing = {column["name"] for column in inspect(connection).get_columns(table)}
            for name, definition in columns.items():
                if name not in existing:
                    connection.exec_driver_sql(f'ALTER TABLE "{table}" ADD COLUMN "{name}" {definition}')


def _ensure_fish_knowledge_crud_constraints() -> None:
    """Allow Fish Knowledge species to be soft-deleted on existing installs.

    ``create_all`` does not alter a pre-existing CHECK constraint.  Cloud SQL
    deployments use PostgreSQL, so apply the additive status change at startup
    while leaving SQLite test databases to the current declarative metadata.
    The operation is idempotent and does not touch any other model or state
    machine.
    """

    if not inspect(engine).has_table("fish_species"):
        return
    if engine.dialect.name != "postgresql":
        return
    with engine.begin() as connection:
        connection.exec_driver_sql(
            'ALTER TABLE "fish_species" DROP CONSTRAINT IF EXISTS "ck_fish_species_status"'
        )
        connection.exec_driver_sql(
            'ALTER TABLE "fish_species" ADD CONSTRAINT "ck_fish_species_status" '
            "CHECK (status IN ('ACTIVE', 'DRAFT', 'DELETED'))"
        )
