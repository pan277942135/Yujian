import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

import app.db as db_module


def _legacy_fish_schema(engine):
    with engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE fish_species (id VARCHAR(128) PRIMARY KEY)")
        connection.exec_driver_sql("CREATE TABLE fish_asset_import_batches (batch_id VARCHAR(128) PRIMARY KEY)")
        connection.exec_driver_sql(
            """CREATE TABLE fish_knowledge_asset_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                species_id VARCHAR(128) NOT NULL,
                asset_type VARCHAR(32) NOT NULL,
                version INTEGER NOT NULL,
                object_name TEXT NOT NULL UNIQUE,
                image_url TEXT NOT NULL UNIQUE,
                status VARCHAR(16) NOT NULL DEFAULT 'DRAFT',
                sha256 VARCHAR(64) NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                batch_id VARCHAR(128),
                item_id INTEGER,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL,
                CONSTRAINT uq_fish_knowledge_asset_version_slot
                    UNIQUE (species_id, asset_type, version),
                FOREIGN KEY(species_id) REFERENCES fish_species(id) ON DELETE CASCADE,
                FOREIGN KEY(batch_id) REFERENCES fish_asset_import_batches(batch_id) ON DELETE SET NULL
            )"""
        )
        connection.exec_driver_sql(
            """CREATE TABLE fish_asset_import_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id VARCHAR(128), species_id VARCHAR(128),
                asset_type VARCHAR(32), source_filename VARCHAR(512) NOT NULL
            )"""
        )
        connection.exec_driver_sql("INSERT INTO fish_species(id) VALUES ('sharpbelly')")
        connection.exec_driver_sql(
            """INSERT INTO fish_knowledge_asset_versions
                (species_id,asset_type,version,object_name,image_url,status,sha256,metadata_json,created_at,updated_at)
                VALUES ('sharpbelly','COVER',1,'old-list','old-list-url','DRAFT','sha-list',
                '{"cover_variant":"COVER_CARD"}',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)"""
        )
        connection.exec_driver_sql(
            """INSERT INTO fish_knowledge_asset_versions
                (species_id,asset_type,version,object_name,image_url,status,sha256,metadata_json,created_at,updated_at)
                VALUES ('sharpbelly','COVER',2,'old-transparent','old-transparent-url','DRAFT','sha-alpha',
                '{"cover_variant":"COVER_CARD_TRANSPARENT_LEFT"}',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)"""
        )
        connection.exec_driver_sql(
            "INSERT INTO fish_asset_import_items(asset_type,source_filename) VALUES ('COVER','00_cover_hero.png')"
        )
        connection.exec_driver_sql(
            """CREATE TABLE fish_knowledge_asset_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id VARCHAR(128), species_id VARCHAR(128), asset_role VARCHAR(32),
                version_id INTEGER REFERENCES fish_knowledge_asset_versions(id),
                frozen_at DATETIME
            )"""
        )
        connection.exec_driver_sql(
            "INSERT INTO fish_knowledge_asset_reviews(batch_id,species_id,asset_role,version_id) VALUES ('old','sharpbelly','COVER_LIST',1)"
        )


def test_v13_migration_preserves_history_and_versions_roles_independently(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy-fish.db'}")
    _legacy_fish_schema(engine)
    monkeypatch.setattr(db_module, "engine", engine)

    db_module._ensure_fish_knowledge_asset_v13()
    db_module._ensure_fish_knowledge_asset_v13()  # startup migration is idempotent

    with engine.connect() as connection:
        roles = connection.execute(
            text("SELECT id, asset_role, version FROM fish_knowledge_asset_versions ORDER BY id")
        ).all()
        assert roles == [(1, "COVER_LIST", 1), (2, "TRANSPARENT_MAIN", 2)]
        assert connection.execute(text("SELECT asset_role FROM fish_asset_import_items")).scalar_one() == "COVER_HERO"
        review_columns = {column["name"] for column in inspect(connection).get_columns("fish_knowledge_asset_reviews")}
        assert {
            "binding_type", "binding_id", "binding_status", "binding_image_url", "cms_content_sha256", "asset_status",
            "validation_warnings_json", "warnings_acknowledged_at", "warnings_acknowledged_by",
            "visual_qa_note", "visual_qa_reviewer", "visual_qa_reviewed_at",
            "content_qa_note", "content_qa_reviewer", "content_qa_reviewed_at",
        } <= review_columns
        assert connection.execute(text("SELECT version_id FROM fish_knowledge_asset_reviews")).scalar_one() == 1
        assert connection.exec_driver_sql("PRAGMA foreign_key_check").all() == []
        assert any(
            index["name"] == "uq_fish_knowledge_asset_role_version"
            for index in inspect(connection).get_indexes("fish_knowledge_asset_versions")
        )
        assert {tuple(row["constrained_columns"]) for row in inspect(connection).get_foreign_keys("fish_knowledge_asset_versions")} == {
            ("species_id",), ("batch_id",)
        }

    with engine.begin() as connection:
        # Distinct roles keep independent v1 slots even though legacy asset_type
        # remains COVER for backwards compatibility with its check constraint.
        connection.exec_driver_sql(
            """INSERT INTO fish_knowledge_asset_versions
                (species_id,asset_type,asset_role,version,object_name,image_url,status,sha256,metadata_json,created_at,updated_at)
                VALUES ('sharpbelly','COVER','COVER_HERO',1,'hero-v1','hero-v1-url','DRAFT','sha-hero','{}',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)"""
        )

    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.exec_driver_sql(
                """INSERT INTO fish_knowledge_asset_versions
                    (species_id,asset_type,asset_role,version,object_name,image_url,status,sha256,metadata_json,created_at,updated_at)
                    VALUES ('sharpbelly','COVER','COVER_HERO',1,'hero-v1-duplicate','hero-v1-duplicate-url','DRAFT','sha-hero-2','{}',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)"""
            )
