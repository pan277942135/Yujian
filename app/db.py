import os
import json
import re
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
    from app.platform import models as platform_models  # noqa: F401
    from app.fish_knowledge import import_batch as fish_asset_models  # noqa: F401

    _ensure_fish_knowledge_asset_v13()
    Base.metadata.create_all(bind=engine)
    _ensure_production_pipeline_columns()
    _ensure_fish_knowledge_crud_constraints()
    _ensure_user_catch_columns()
    _ensure_bside_visual_columns()
    _ensure_bside_asset_registry_columns()
    _ensure_account_privacy_columns()
    _ensure_training_eligibility_columns()
    from app.platform.services.bside_assets import seed_bside_asset_registry

    seed_db = SessionLocal()
    try:
        seed_bside_asset_registry(seed_db)
    finally:
        seed_db.close()


def _ensure_fish_knowledge_asset_v13() -> None:
    """Add role-aware Fish Knowledge asset columns and per-role versioning.

    Old COVER rows are mapped from their preserved cover_variant metadata;
    untagged historical COVER rows remain COVER_LIST. The operation is
    additive and idempotent for the supported SQLite and PostgreSQL stores.
    """

    additions = {
        "fish_asset_import_batches": {
            "warnings_acknowledged": "BOOLEAN NOT NULL DEFAULT FALSE",
            "warnings_acknowledged_at": "TIMESTAMP WITH TIME ZONE",
            "warnings_acknowledged_by": "VARCHAR(256)",
        },
        "fish_asset_import_items": {"asset_role": "VARCHAR(32)"},
        "fish_knowledge_asset_versions": {"asset_role": "VARCHAR(32)"},
        "fish_knowledge_asset_reviews": {
            "binding_type": "VARCHAR(32)",
            "binding_id": "INTEGER",
            "binding_status": "VARCHAR(16)",
            "binding_image_url": "TEXT",
            "cms_content_sha256": "VARCHAR(64)",
            "asset_status": "VARCHAR(16)",
            "validation_warnings_json": "TEXT NOT NULL DEFAULT '[]'",
            "warnings_acknowledged_at": "TIMESTAMP WITH TIME ZONE",
            "warnings_acknowledged_by": "VARCHAR(256)",
        },
    }
    with engine.connect() as connection:
        if engine.dialect.name == "sqlite":
            # SQLite cannot drop the historical table-level uniqueness rule on
            # (species_id, asset_type, version). Disable FK enforcement before
            # opening the transaction; the replacement preserves all FK rows.
            connection.exec_driver_sql("PRAGMA foreign_keys=OFF")
            connection.commit()
        try:
            with connection.begin():
                for table, columns in additions.items():
                    if not inspect(connection).has_table(table):
                        continue
                    existing = {column["name"] for column in inspect(connection).get_columns(table)}
                    for name, definition in columns.items():
                        if name not in existing:
                            connection.exec_driver_sql(f'ALTER TABLE "{table}" ADD COLUMN "{name}" {definition}')

                if inspect(connection).has_table("fish_knowledge_asset_versions"):
                    rows = connection.exec_driver_sql(
                        'SELECT id, asset_type, metadata_json, asset_role FROM "fish_knowledge_asset_versions"'
                    ).mappings().all()
                    for row in rows:
                        if row["asset_role"]:
                            continue
                        metadata = {}
                        try:
                            metadata = json.loads(row["metadata_json"] or "{}")
                        except (TypeError, ValueError):
                            pass
                        role = str(metadata.get("asset_role") or "").strip().upper() if isinstance(metadata, dict) else ""
                        variant = str(metadata.get("cover_variant") or "").strip().upper() if isinstance(metadata, dict) else ""
                        if role not in {"COVER_LIST", "COVER_HERO", "TRANSPARENT_MAIN", "TRANSPARENT_ALT", "HERO", "IDENTIFICATION", "ECO", "GEAR", "SKILL"}:
                            role = {
                                "COVER_CARD_TRANSPARENT_LEFT": "TRANSPARENT_MAIN",
                                "COVER_CARD_TRANSPARENT_RIGHT": "TRANSPARENT_ALT",
                                "COVER_HERO": "COVER_HERO",
                            }.get(variant, "COVER_LIST" if row["asset_type"] == "COVER" else row["asset_type"])
                        connection.exec_driver_sql(
                            'UPDATE "fish_knowledge_asset_versions" SET asset_role = ? WHERE id = ?'
                            if engine.dialect.name == "sqlite"
                            else 'UPDATE "fish_knowledge_asset_versions" SET asset_role = %s WHERE id = %s',
                            (role, row["id"]),
                        )

                if inspect(connection).has_table("fish_asset_import_items"):
                    rows = connection.exec_driver_sql(
                        'SELECT id, asset_type, source_filename, asset_role FROM "fish_asset_import_items"'
                    ).mappings().all()
                    for row in rows:
                        if row["asset_role"]:
                            continue
                        stem = re.sub(r"\.[^.]+$", "", str(row["source_filename"] or "")).lower()
                        role = (
                            "COVER_HERO" if re.fullmatch(r"00_cover_hero(?:_.*)?", stem)
                            else "TRANSPARENT_MAIN" if re.fullmatch(r"01_transparent_main(?:_.*)?", stem)
                            else "TRANSPARENT_ALT" if re.fullmatch(r"02_transparent_alt(?:_.*)?", stem)
                            else "COVER_LIST" if row["asset_type"] == "COVER"
                            else row["asset_type"]
                        )
                        connection.exec_driver_sql(
                            'UPDATE "fish_asset_import_items" SET asset_role = ? WHERE id = ?'
                            if engine.dialect.name == "sqlite"
                            else 'UPDATE "fish_asset_import_items" SET asset_role = %s WHERE id = %s',
                            (role, row["id"]),
                        )

                if inspect(connection).has_table("fish_knowledge_asset_versions"):
                    if engine.dialect.name == "postgresql":
                        connection.exec_driver_sql(
                            'ALTER TABLE "fish_knowledge_asset_versions" '
                            'DROP CONSTRAINT IF EXISTS "uq_fish_knowledge_asset_version_slot"'
                        )
                        connection.exec_driver_sql('DROP INDEX IF EXISTS "uq_fish_knowledge_asset_version_slot"')
                    else:
                        unique_constraints = inspect(connection).get_unique_constraints("fish_knowledge_asset_versions")
                        old_rule = any(
                            row.get("column_names") == ["species_id", "asset_type", "version"]
                            for row in unique_constraints
                        )
                        if old_rule:
                            connection.exec_driver_sql(
                                'CREATE TABLE "__fish_knowledge_asset_versions_v13" ('
                                '"id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, '
                                '"species_id" VARCHAR(128) NOT NULL, "asset_type" VARCHAR(32) NOT NULL, '
                                '"asset_role" VARCHAR(32), "version" INTEGER NOT NULL, '
                                '"object_name" TEXT NOT NULL UNIQUE, "image_url" TEXT NOT NULL UNIQUE, '
                                '"status" VARCHAR(16) NOT NULL DEFAULT \'DRAFT\', "sha256" VARCHAR(64) NOT NULL, '
                                '"metadata_json" TEXT NOT NULL DEFAULT \'{}\', "batch_id" VARCHAR(128), '
                                '"item_id" INTEGER, "created_at" DATETIME NOT NULL, "updated_at" DATETIME NOT NULL, '
                                'CONSTRAINT "ck_fish_knowledge_asset_version_type" '
                                'CHECK (asset_type IN (\'COVER\',\'HERO\',\'IDENTIFICATION\',\'ECO\',\'GEAR\',\'SKILL\')), '
                                'CONSTRAINT "ck_fish_knowledge_asset_version_status" '
                                'CHECK (status IN (\'DRAFT\',\'ACTIVE\',\'ARCHIVED\')), '
                                'FOREIGN KEY("species_id") REFERENCES fish_species(id) ON DELETE CASCADE, '
                                'FOREIGN KEY("batch_id") REFERENCES fish_asset_import_batches(batch_id) ON DELETE SET NULL)'
                            )
                            connection.exec_driver_sql(
                                'INSERT INTO "__fish_knowledge_asset_versions_v13" '
                                '(id,species_id,asset_type,asset_role,version,object_name,image_url,status,sha256,metadata_json,batch_id,item_id,created_at,updated_at) '
                                'SELECT id,species_id,asset_type,asset_role,version,object_name,image_url,status,sha256,metadata_json,batch_id,item_id,created_at,updated_at '
                                'FROM "fish_knowledge_asset_versions"'
                            )
                            connection.exec_driver_sql('DROP TABLE "fish_knowledge_asset_versions"')
                            connection.exec_driver_sql(
                                'ALTER TABLE "__fish_knowledge_asset_versions_v13" RENAME TO "fish_knowledge_asset_versions"'
                            )
                    connection.exec_driver_sql(
                        'CREATE UNIQUE INDEX IF NOT EXISTS "uq_fish_knowledge_asset_role_version" '
                        'ON "fish_knowledge_asset_versions" ("species_id", "asset_role", "version")'
                    )
        finally:
            if engine.dialect.name == "sqlite":
                try:
                    if connection.in_transaction():
                        connection.rollback()
                    connection.exec_driver_sql("PRAGMA foreign_keys=ON")
                    connection.commit()
                except Exception:
                    connection.invalidate()
                    raise


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
            "is_production": "BOOLEAN NOT NULL DEFAULT FALSE",
            "published_at": "TIMESTAMP WITH TIME ZONE",
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


def _ensure_user_catch_columns() -> None:
    """Keep the additive MVP tables safe for deployments upgraded in place.

    New installs receive these tables through SQLAlchemy metadata.  The guard is
    intentionally non-destructive and only adds fields introduced after the
    initial MVP table was already live.
    """

    inspector = inspect(engine)
    if not inspector.has_table("fish_catches"):
        return
    existing = {column["name"] for column in inspector.get_columns("fish_catches")}
    additions = {
        "image_object_name": "TEXT",
        "length_cm": "DOUBLE PRECISION CHECK (length_cm IS NULL OR (length_cm > 0 AND length_cm <= 1000))",
        "weight_kg": "DOUBLE PRECISION CHECK (weight_kg IS NULL OR (weight_kg > 0 AND weight_kg <= 1000))",
        "location": "TEXT",
        "story": "TEXT",
        "metadata_version": "INTEGER NOT NULL DEFAULT 0",
        "client_record_id": "VARCHAR(128)",
        "bside_status": "VARCHAR(16) NOT NULL DEFAULT 'NONE'",
        "bside_result_uri": "TEXT",
        "bside_result_object_name": "TEXT",
        "bside_generated_at": "TIMESTAMP WITH TIME ZONE",
        "bside_job_id": "VARCHAR(36)",
    }
    with engine.begin() as connection:
        for name, definition in additions.items():
            if name not in existing:
                connection.exec_driver_sql(f'ALTER TABLE "fish_catches" ADD COLUMN "{name}" {definition}')
        if engine.dialect.name == "postgresql":
            existing_checks = {
                check.get("name") for check in inspect(connection).get_check_constraints("fish_catches")
            }
            for name, expression in (
                ("ck_fish_catches_length_cm_range", "length_cm IS NULL OR (length_cm > 0 AND length_cm <= 1000)"),
                ("ck_fish_catches_weight_kg_range", "weight_kg IS NULL OR (weight_kg > 0 AND weight_kg <= 1000)"),
            ):
                if name not in existing_checks:
                    connection.exec_driver_sql(
                        f'ALTER TABLE "fish_catches" ADD CONSTRAINT "{name}" CHECK ({expression}) NOT VALID'
                    )
        connection.exec_driver_sql(
            'CREATE UNIQUE INDEX IF NOT EXISTS "uq_fish_catches_user_client_record" '
            'ON "fish_catches" ("user_id", "client_record_id") WHERE "client_record_id" IS NOT NULL'
        )

    job_table = "fish_bside_job"
    if not inspect(engine).has_table(job_table):
        return
    job_existing = {column["name"] for column in inspect(engine).get_columns(job_table)}
    if "pose_metadata_json" not in job_existing:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                f'ALTER TABLE "{job_table}" ADD COLUMN "pose_metadata_json" TEXT'
            )


def _ensure_bside_visual_columns() -> None:
    """Add the Qwen RGB source pointer for the four-step B-side workflow.

    The previous release stored a transparent-fish URI on this session table.
    Keep that column and old rows intact; the new pointer is additive and lets
    Step 1 perform transparent extraction only after an explicit user action.
    """

    table = "qwen_bside_visual_session"
    if not inspect(engine).has_table(table):
        return
    existing = {column["name"] for column in inspect(engine).get_columns(table)}
    if "source_qwen_rgb_uri" not in existing:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                f'ALTER TABLE "{table}" ADD COLUMN "source_qwen_rgb_uri" TEXT'
            )


def _ensure_bside_asset_registry_columns() -> None:
    """Add the persisted DB-backed B-side style-plan pointers in place."""

    table = "qwen_bside_visual_session"
    if not inspect(engine).has_table(table):
        return
    additions = {
        "background_id": "INTEGER",
        "outline_style_id": "INTEGER",
        "outline_profile_id": "INTEGER",
        "style_seed": "BIGINT",
    }
    existing = {column["name"] for column in inspect(engine).get_columns(table)}
    missing = {name: definition for name, definition in additions.items() if name not in existing}
    if not missing:
        return
    with engine.begin() as connection:
        for name, definition in missing.items():
            connection.exec_driver_sql(
                f'ALTER TABLE "{table}" ADD COLUMN "{name}" {definition}'
            )


def _ensure_account_privacy_columns() -> None:
    """Apply Account & Privacy v1 additively to existing Cloud SQL installs."""

    inspector = inspect(engine)
    if not inspector.has_table("users"):
        return
    existing = {column["name"] for column in inspector.get_columns("users")}
    if "avatar_object_name" not in existing:
        with engine.begin() as connection:
            connection.exec_driver_sql('ALTER TABLE "users" ADD COLUMN "avatar_object_name" TEXT')


def _ensure_training_eligibility_columns() -> None:
    """Apply the additive historical duplicate eligibility contract in place.

    ``create_all`` handles fresh databases, but production upgrades must use
    explicit additive ALTER statements so an existing ``image_assets`` table
    receives the fields without being recreated or rewritten.
    """

    inspector = inspect(engine)
    if not inspector.has_table("image_assets"):
        return
    existing = {column["name"] for column in inspector.get_columns("image_assets")}
    additions = {
        "training_eligible": "BOOLEAN NOT NULL DEFAULT TRUE",
        "training_exclusion_reason": "VARCHAR(64)",
        "duplicate_of_image_asset_id": "INTEGER",
        "training_eligibility_source": "TEXT",
        "training_eligibility_updated_at": "TIMESTAMP WITH TIME ZONE",
    }
    with engine.begin() as connection:
        for name, definition in additions.items():
            if name not in existing:
                connection.exec_driver_sql(f'ALTER TABLE "image_assets" ADD COLUMN "{name}" {definition}')
        connection.exec_driver_sql(
            'CREATE INDEX IF NOT EXISTS "ix_image_assets_training_eligible" '
            'ON "image_assets" ("training_eligible")'
        )
        connection.exec_driver_sql(
            'CREATE INDEX IF NOT EXISTS "ix_image_assets_training_exclusion_reason" '
            'ON "image_assets" ("training_exclusion_reason")'
        )
        connection.exec_driver_sql(
            'CREATE INDEX IF NOT EXISTS "ix_image_assets_duplicate_of_image_asset_id" '
            'ON "image_assets" ("duplicate_of_image_asset_id")'
        )
        if engine.dialect.name == "postgresql":
            connection.exec_driver_sql(
                """
                DO $$
                BEGIN
                  IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conname = 'fk_image_assets_duplicate_of_image_asset_id'
                  ) THEN
                    ALTER TABLE image_assets
                      ADD CONSTRAINT fk_image_assets_duplicate_of_image_asset_id
                      FOREIGN KEY (duplicate_of_image_asset_id) REFERENCES image_assets(id);
                  END IF;
                END $$;
                """
            )
