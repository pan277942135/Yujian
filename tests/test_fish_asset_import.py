from io import BytesIO

from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import models  # noqa: F401
from app.db import Base
from app.fish_knowledge.import_batch import (
    _asset_type_for_filename,
    _bind_imported_version,
    _normalize_upload_path,
    _parse_source_uri,
    _resolve_species_name,
    _scan_item,
    FishKnowledgeAssetVersion,
)
from app.fish_knowledge.cards import FishCard
from app.fish_knowledge.cover import FishSpeciesCover
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


def test_local_upload_path_normalization_accepts_folder_selection_and_rejects_traversal():
    assert _normalize_upload_path("Yujian_Fish_Knowledge/01_白条/00_cover.png") == "Yujian_Fish_Knowledge/01_白条/00_cover.png"
    assert _normalize_upload_path(r"01_白条\00_cover.png") == "01_白条/00_cover.png"
    for value in ("../secret.png", "/absolute.png", "C:/secret.png", "species//00_cover.png"):
        try:
            _normalize_upload_path(value)
        except Exception as exc:
            assert "INVALID_UPLOAD_PATH" in str(exc)
        else:
            raise AssertionError("expected upload path validation")


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
    batch = type("Batch", (), {"source_gcs_uri": "gs://bucket/fish-assets/imports/FK_001/", "batch_id": "FK_001"})()
    item = _scan_item(
        db,
        None,
        Bucket(),
        batch,
        Blob.name,
    )
    assert item.species_id == "sharpbelly"
    assert item.asset_type == "HERO"
    assert item.validation_status == "VALID"
    assert item.width == 1024
    assert item.height == 1024


def test_scan_resolves_species_after_browser_selected_outer_folder(tmp_path, monkeypatch):
    db = _session(tmp_path)
    _add_species(db)
    image = BytesIO()
    Image.new("RGB", (1024, 1024), "white").save(image, format="PNG")
    data = image.getvalue()

    class Blob:
        name = "fish-assets/imports/FK_001/Yujian_Fish_Knowledge/01_白条/01_hero.png"
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
    batch = type("Batch", (), {"source_gcs_uri": "gs://bucket/fish-assets/imports/FK_001/", "batch_id": "FK_001"})()
    item = _scan_item(db, None, Bucket(), batch, Blob.name)
    assert item.species_id == "sharpbelly"
    assert item.asset_type == "HERO"
    assert item.validation_status == "VALID"


def test_imported_version_binds_existing_draft_card_without_overwriting_content(tmp_path):
    db = _session(tmp_path)
    _add_species(db)
    card = FishCard(
        species_id="sharpbelly",
        card_type="HERO",
        title="运营标题",
        image_url="",
        description='{"type":"HERO","tag":"保留标签"}',
        sort_order=0,
        status="DRAFT",
    )
    db.add(card)
    version = FishKnowledgeAssetVersion(
        species_id="sharpbelly",
        asset_type="HERO",
        version=1,
        object_name="fish-assets/fish-knowledge/sharpbelly/hero/v1.webp",
        image_url="/api/v1/fish/knowledge-media/sharpbelly/hero/v1.webp",
        status="DRAFT",
        sha256="a" * 64,
        batch_id="FK_001",
    )
    db.add(version)
    db.flush()

    assert _bind_imported_version(db, version) == "BOUND"
    assert card.image_url == version.image_url
    assert card.description == '{"type":"HERO","tag":"保留标签"}'


def test_imported_version_keeps_active_card_and_creates_draft(tmp_path):
    db = _session(tmp_path)
    _add_species(db)
    active = FishCard(
        species_id="sharpbelly",
        card_type="HERO",
        title="当前 ACTIVE",
        image_url="/api/v1/fish/knowledge-media/sharpbelly/hero/v0.webp",
        description='{"type":"HERO","tag":"当前文案"}',
        sort_order=0,
        status="ACTIVE",
    )
    db.add(active)
    version = FishKnowledgeAssetVersion(
        species_id="sharpbelly",
        asset_type="HERO",
        version=1,
        object_name="fish-assets/fish-knowledge/sharpbelly/hero/v1.webp",
        image_url="/api/v1/fish/knowledge-media/sharpbelly/hero/v1.webp",
        status="DRAFT",
        sha256="b" * 64,
        batch_id="FK_002",
    )
    db.add(version)
    db.flush()

    assert _bind_imported_version(db, version) == "BOUND"
    db.flush()
    assert active.status == "ACTIVE"
    drafts = [
        row for row in db.query(FishCard).all()
        if row.species_id == "sharpbelly" and row.status == "DRAFT"
    ]
    assert len(drafts) == 1
    assert drafts[0].image_url == version.image_url
    assert drafts[0].description == active.description


def test_imported_version_binds_existing_draft_cover(tmp_path):
    db = _session(tmp_path)
    _add_species(db)
    cover = FishSpeciesCover(
        species_id="sharpbelly",
        image_url="",
        style="ANIME_CARD",
        title="",
        status="DRAFT",
    )
    db.add(cover)
    version = FishKnowledgeAssetVersion(
        species_id="sharpbelly",
        asset_type="COVER",
        version=1,
        object_name="fish-assets/fish-knowledge/sharpbelly/cover/v1.webp",
        image_url="/api/v1/fish/knowledge-media/sharpbelly/cover/v1.webp",
        status="DRAFT",
        sha256="c" * 64,
        batch_id="FK_003",
    )
    db.add(version)
    db.flush()

    assert _bind_imported_version(db, version) == "BOUND"
    assert cover.image_url == version.image_url
    assert cover.status == "DRAFT"
