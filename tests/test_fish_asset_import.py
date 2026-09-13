from io import BytesIO

from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import models  # noqa: F401
from app.db import Base
from app.fish_knowledge.import_batch import (
    _asset_type_for_filename,
    _parse_source_uri,
    _resolve_species_name,
    _scan_item,
)
from app.fish_knowledge.species import FishSpecies
from app.models import SpeciesCatalog


def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'fish-import.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)()


def _add_species(db):
    db.add(SpeciesCatalog(
        species_key="sharpbelly",
        catalog_order=1,
        common_name_zh="白条",
        status="active",
        is_other=False,
    ))
    db.add(FishSpecies(
        id="sharpbelly",
        name_cn="白条",
        alias=["baitiao"],
        category="淡水鱼",
        summary="",
        status="DRAFT",
    ))
    db.commit()


def test_parse_source_uri_requires_configured_import_prefix(monkeypatch):
    monkeypatch.setattr("app.fish_knowledge.import_batch.get_bucket_name", lambda: "bucket")
    assert _parse_source_uri("gs://bucket/fish-assets/imports/FK_001/") == (
        "bucket",
        "fish-assets/imports/FK_001/",
        "FK_001",
    )
    try:
        _parse_source_uri("gs://other/fish-assets/imports/FK_001/")
    except Exception as exc:
        assert "SOURCE_BUCKET_NOT_ALLOWED" in str(exc)
    else:
        raise AssertionError("expected source bucket validation")


def test_asset_mapping_contract():
    assert _asset_type_for_filename("00_cover.png") == "COVER"
    assert _asset_type_for_filename("01_hero.jpeg") == "HERO"
    assert _asset_type_for_filename("02_identification.webp") == "IDENTIFICATION"
    assert _asset_type_for_filename("03_ecology.png") == "ECO"
    assert _asset_type_for_filename("04_gear.jpg") == "GEAR"
    assert _asset_type_for_filename("05_skill.png") == "SKILL"
    assert _asset_type_for_filename("06_gallery.png") is None


def test_alias_resolves_to_canonical_species(tmp_path):
    db = _session(tmp_path)
    _add_species(db)
    assert _resolve_species_name(db, "baitiao") == "sharpbelly"


def test_valid_image_scan_marks_square_file(tmp_path, monkeypatch):
    db = _session(tmp_path)
    _add_species(db)
    image = BytesIO()
    Image.new("RGB", (1024, 1024), "white").save(image, format="PNG")
    data = image.getvalue()

    class Blob:
        name = "fish-assets/imports/FK_001/baitiao/01_hero.png"
        size = len(data)

        def download_as_bytes(self, timeout=None):
            return data

    class Bucket:
        def blob(self, _):
            return Blob()

    monkeypatch.setattr(
        "app.fish_knowledge.import_batch._source_parts",
        lambda _: ("bucket", "fish-assets/imports/FK_001/", "FK_001"),
    )
    item = _scan_item(
        db,
        None,
        Bucket(),
        type("Batch", (), {"source_gcs_uri": "gs://bucket/fish-assets/imports/FK_001/"})(),
        Blob.name,
    )
    assert item.species_id == "sharpbelly"
    assert item.asset_type == "HERO"
    assert item.validation_status == "VALID"
    assert item.width == 1024
    assert item.height == 1024
