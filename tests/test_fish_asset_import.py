from io import BytesIO

import pytest
from PIL import Image

from app.fish_knowledge.import_batch import (
    _asset_type_for_filename,
    _parse_source_uri,
    _resolve_species_name,
    _scan_item,
)


def test_parse_source_uri_requires_configured_import_prefix(monkeypatch):
    monkeypatch.setattr("app.fish_knowledge.import_batch.get_bucket_name", lambda: "bucket")
    assert _parse_source_uri("gs://bucket/fish-assets/imports/FK_001/") == (
        "bucket",
        "fish-assets/imports/FK_001/",
        "FK_001",
    )
    with pytest.raises(Exception):
        _parse_source_uri("gs://other/fish-assets/imports/FK_001/")


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("00_cover.png", "COVER"),
        ("01_hero.jpeg", "HERO"),
        ("02_identification.webp", "IDENTIFICATION"),
        ("03_ecology.png", "ECO"),
        ("04_gear.jpg", "GEAR"),
        ("05_skill.png", "SKILL"),
    ],
)
def test_asset_mapping(filename, expected):
    assert _asset_type_for_filename(filename) == expected


def test_asset_mapping_rejects_unknown_slot():
    assert _asset_type_for_filename("06_gallery.png") is None


def test_alias_resolves_to_canonical_species(db_session):
    from app.fish_knowledge.species import FishSpecies
    from app.models import SpeciesCatalog

    db_session.add(SpeciesCatalog(
        species_key="sharpbelly",
        catalog_order=1,
        common_name_zh="白条",
        status="active",
        is_other=False,
    ))
    db_session.add(FishSpecies(
        id="sharpbelly",
        name_cn="白条",
        alias=["baitiao"],
        category="淡水鱼",
        summary="",
        status="DRAFT",
    ))
    db_session.commit()
    assert _resolve_species_name(db_session, "baitiao") == "sharpbelly"


def test_valid_image_scan_marks_square_file(db_session, monkeypatch):
    from app.fish_knowledge.species import FishSpecies
    from app.models import SpeciesCatalog

    db_session.add(SpeciesCatalog(
        species_key="sharpbelly",
        catalog_order=1,
        common_name_zh="白条",
        status="active",
        is_other=False,
    ))
    db_session.add(FishSpecies(
        id="sharpbelly",
        name_cn="白条",
        alias=["baitiao"],
        category="淡水鱼",
        summary="",
        status="DRAFT",
    ))
    db_session.commit()
    image = BytesIO()
    Image.new("RGB", (1024, 1024), "white").save(image, format="PNG")
    class Blob:
        name = "fish-assets/imports/FK_001/baitiao/01_hero.png"
        size = len(image.getvalue())
        def download_as_bytes(self, timeout=None):
            return image.getvalue()
    class Bucket:
        def blob(self, _):
            return Blob()
    monkeypatch.setattr("app.fish_knowledge.import_batch._source_parts", lambda _: ("bucket", "fish-assets/imports/FK_001/", "FK_001"))
    item = _scan_item(db_session, None, Bucket(), type("Batch", (), {"source_gcs_uri": "gs://bucket/fish-assets/imports/FK_001/"})(), Blob.name)
    assert item.species_id == "sharpbelly"
    assert item.asset_type == "HERO"
    assert item.validation_status == "VALID"
    assert item.width == 1024
    assert item.height == 1024
