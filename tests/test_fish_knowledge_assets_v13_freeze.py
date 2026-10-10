import hashlib
import json

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app import models  # noqa: F401
from app.db import Base
from app.platform import models as platform_models  # noqa: F401
from app.fish_knowledge.cards import FishCard
from app.fish_knowledge.import_batch import (
    AssetReviewPayload,
    FishAssetImportBatch,
    FishAssetImportItem,
    FishKnowledgeAssetReview,
    FishKnowledgeAssetVersion,
    freeze_asset_batch_v13,
    review_asset_version_v13,
)
from app.fish_knowledge.species import FishSpecies
from app.models import SpeciesCatalog


def test_freeze_requires_review_and_records_provenance_without_activation(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'freeze.db'}")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
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
        category="淡水鱼",
        summary="",
        status="DRAFT",
    ))
    batch_id = "FK_REVIEW_TEST_001"
    batch = FishAssetImportBatch(
        batch_id=batch_id,
        source_gcs_uri="gs://test-bucket/fish-assets/imports/FK_REVIEW_TEST_001/",
        status="COMPLETED",
        created_by="tester",
        total_files=1,
        recognized_files=1,
        valid_files=1,
        species_count=1,
    )
    db.add(batch)
    raw_sha = hashlib.sha256(b"source-png").hexdigest()
    derived_sha = hashlib.sha256(b"derived-webp").hexdigest()
    object_name = "fish-assets/fish-knowledge/sharpbelly/hero/v1.webp"
    image_url = "/api/v1/fish/knowledge-media/sharpbelly/hero/v1.webp"
    version = FishKnowledgeAssetVersion(
        species_id="sharpbelly",
        asset_type="HERO",
        asset_role="HERO",
        version=1,
        object_name=object_name,
        image_url=image_url,
        status="DRAFT",
        sha256=raw_sha,
        metadata_json=json.dumps({
            "asset_role": "HERO",
            "source_sha256": raw_sha,
            "derived_sha256": derived_sha,
            "source_filename": "01_hero.png",
            "source_format": "image/png",
            "gcs_generation": "17283940",
        }),
        batch_id=batch_id,
    )
    db.add(version)
    db.flush()
    card = FishCard(
        species_id="sharpbelly",
        card_type="HERO",
        title="识别卡",
        image_url=image_url,
        description='{"tag":"中上层鱼","description":"体形修长"}',
        sort_order=0,
        status="DRAFT",
    )
    item = FishAssetImportItem(
        batch_id=batch_id,
        species_id="sharpbelly",
        source_object="fish-assets/imports/FK_REVIEW_TEST_001/sharpbelly/01_hero.png",
        asset_type="HERO",
        asset_role="HERO",
        source_filename="01_hero.png",
        mime_type="image/png",
        width=1254,
        height=1254,
        file_size=123,
        sha256=raw_sha,
        validation_status="IMPORTED",
        target_object=object_name,
        version_id=version.id,
    )
    db.add_all([card, item])
    db.commit()
    monkeypatch.setenv("APP_GIT_COMMIT", "a" * 40)

    review_asset_version_v13(
        version.id,
        AssetReviewPayload(batch_id=batch_id, visual_qa_result="PASS", visual_qa_note="Visual evidence.", visual_qa_reviewer="qa-user"),
        db,
    )
    review = review_asset_version_v13(
        version.id,
        AssetReviewPayload(batch_id=batch_id, content_qa_result="PASS", content_qa_note="Content evidence.", content_qa_reviewer="qa-user"),
        db,
    )
    assert review["validation_result"] == "PASS"
    result = freeze_asset_batch_v13(batch_id, db)
    assert result["frozen"] == 1
    row = result["manifest"][0]
    assert row["source_sha256"] == raw_sha
    assert row["derived_media_sha256"] == derived_sha
    assert row["gcs_object_name"] == object_name
    assert row["gcs_generation"] == "17283940"
    assert row["asset_status"] == "DRAFT"
    assert row["binding_type"] == "FISH_CARD"
    assert row["binding_id"] == card.id
    assert row["binding_status"] == "DRAFT"
    assert row["cms_content_sha256"] == hashlib.sha256(
        json.dumps({"tag": "中上层鱼", "description": "体形修长"}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert db.get(FishKnowledgeAssetVersion, version.id).status == "DRAFT"
    assert db.scalar(select(FishKnowledgeAssetReview).where(FishKnowledgeAssetReview.batch_id == batch_id)).frozen_at is not None

    with pytest.raises(HTTPException) as blocked:
        review_asset_version_v13(
            version.id,
            AssetReviewPayload(
                batch_id=batch_id,
                visual_qa_result="PASS",
                content_qa_result="PASS",
            ),
            db,
        )
    assert blocked.value.status_code == 409
