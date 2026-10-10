from pathlib import Path

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

import app.db as db_module


def test_integrated_fish_and_catch_migrations_have_unique_ordered_versions():
    schema_dir = Path(__file__).resolve().parents[1] / "schemas"
    migrations = sorted(
        path.name
        for path in schema_dir.glob("003[123]_*.sql")
        if not path.name.endswith(".down.sql")
    )
    assert migrations == [
        "0031_fish_knowledge_publication_v14.sql",
        "0032_fish_knowledge_separate_qa_v14.sql",
        "0033_catch_metadata_p0_v1.sql",
    ]
    assert not (schema_dir / "0032_catch_metadata_p0_v1.sql").exists()


def test_full_startup_schema_upgrade_is_idempotent_and_preserves_tables(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'integrated-startup.db'}")
    monkeypatch.setattr(db_module, "engine", engine)
    monkeypatch.setattr(db_module, "SessionLocal", sessionmaker(bind=engine, autoflush=False, autocommit=False))

    db_module.init_db()
    first_tables = set(inspect(engine).get_table_names())
    db_module.init_db()
    second_tables = set(inspect(engine).get_table_names())

    assert first_tables == second_tables
    catch_columns = {column["name"] for column in inspect(engine).get_columns("fish_catches")}
    assert {"length_cm", "weight_kg", "location", "story", "metadata_version", "client_record_id"} <= catch_columns
    review_columns = {column["name"] for column in inspect(engine).get_columns("fish_knowledge_asset_reviews")}
    assert {"visual_qa_note", "visual_qa_reviewer", "visual_qa_reviewed_at",
            "content_qa_note", "content_qa_reviewer", "content_qa_reviewed_at"} <= review_columns
    assert "fish_knowledge_asset_qa_audits" in second_tables
