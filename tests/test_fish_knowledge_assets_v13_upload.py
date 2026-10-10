import asyncio
import io
import json

import pytest
from fastapi import HTTPException
from PIL import Image
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app import models  # noqa: F401
from app.db import Base
from app.platform import models as platform_models  # noqa: F401
from app.fish_knowledge.cards import FishCard
from app.fish_knowledge.import_batch import (
    CreateLocalBatchPayload,
    ClientAcceptancePayload,
    ExecuteBatchPayload,
    FishAssetImportBatch,
    FishKnowledgeAssetQAAudit,
    FishKnowledgeAssetVersion,
    FishKnowledgePublicationAudit,
    AssetReviewPayload,
    PublicAPIReadbackPayload,
    activate_asset_version_v13,
    create_local_batch,
    execute_batch,
    record_public_api_readback_v14,
    record_client_acceptance_v14,
    upload_batch_file,
    upload_single_asset_v13,
    review_asset_version_v13,
    get_unified_card_workspace,
)
from app.fish_knowledge.publication import save_bound_card_content
from app.fish_knowledge.api import build_species_full_detail
from app.fish_knowledge.species import FishSpecies
from app.models import SpeciesCatalog


class MemoryUpload:
    def __init__(self, data: bytes, filename: str = "upload.png"):
        self._data = data
        self.filename = filename
        self.content_type = "image/png"

    async def read(self, size=-1):
        if size < 0:
            data, self._data = self._data, b""
            return data
        data, self._data = self._data[:size], self._data[size:]
        return data

    async def close(self):
        return None


class MemoryBlob:
    def __init__(self, name: str):
        self.name = name
        self.data = None
        self.size = 0
        self.generation = None
        self.metadata = None

    def exists(self, _client=None):
        return self.data is not None

    def download_as_bytes(self, **_kwargs):
        return self.data or b""

    def upload_from_string(self, data, *, content_type, **_kwargs):
        self.data = bytes(data)
        self.size = len(self.data)
        self.content_type = content_type
        self.generation = 17

    def reload(self, _client=None):
        return None


class MemoryBucket:
    def __init__(self):
        self.blobs = {}

    def blob(self, name):
        return self.blobs.setdefault(name, MemoryBlob(name))


class MemoryStorageClient:
    def __init__(self, bucket):
        self._bucket = bucket

    def bucket(self, name):
        assert name == "test-bucket"
        return self._bucket

    def list_blobs(self, bucket, prefix=""):
        return [blob for name, blob in bucket.blobs.items() if name.startswith(prefix) and blob.data is not None]


def _png(color, *, alpha=False, size=(1254, 1254)):
    output = io.BytesIO()
    mode = "RGBA" if alpha else "RGB"
    fill = color if alpha else color[:3]
    Image.new(mode, size, fill).save(output, format="PNG")
    return output.getvalue()


def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'v13-upload.db'}")
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
    db.commit()
    return db


def _review_both_stages(version_id, batch_id, db, *, visual_result="PASS", content_result="PASS", reviewer="test-reviewer"):
    review_asset_version_v13(version_id, AssetReviewPayload(
        batch_id=batch_id,
        visual_qa_result=visual_result,
        visual_qa_note="Visual stage evidence.",
        visual_qa_reviewer=reviewer,
    ), db)
    return review_asset_version_v13(version_id, AssetReviewPayload(
        batch_id=batch_id,
        content_qa_result=content_result,
        content_qa_note="Content stage evidence.",
        content_qa_reviewer=reviewer,
    ), db)


def test_single_upload_uses_batch_core_and_keeps_roles_independent(monkeypatch, tmp_path):
    db = _session(tmp_path)
    bucket = MemoryBucket()
    client = MemoryStorageClient(bucket)
    monkeypatch.setattr("app.fish_knowledge.import_batch.storage.Client", lambda: client)
    monkeypatch.setattr("app.fish_knowledge.publication.storage.Client", lambda: client)
    monkeypatch.setattr("app.fish_knowledge.import_batch.get_bucket_name", lambda: "test-bucket")
    monkeypatch.setattr("app.fish_knowledge.publication.get_bucket_name", lambda: "test-bucket")
    try:
        hero = asyncio.run(upload_single_asset_v13(
            species_id="sharpbelly",
            asset_role="HERO",
            file=MemoryUpload(_png((20, 120, 220)), "hero.png"),
            allow_warnings=False,
            db=db,
        ))
        cover = asyncio.run(upload_single_asset_v13(
            species_id="sharpbelly",
            asset_role="COVER_LIST",
            file=MemoryUpload(_png((220, 120, 20)), "cover.png"),
            allow_warnings=False,
            db=db,
        ))

        assert hero["validation_status"] == "IMPORTED"
        assert cover["validation_status"] == "IMPORTED"
        assert hero["next_action"] == "DRAFT_CREATED"
        assert hero["version"]["status"] == "DRAFT"
        assert hero["version"]["id"] == hero["item"]["version_id"]
        assert hero["version"]["preview_url"].endswith(f"/{hero['version']['id']}/preview")
        hero_version = db.get(FishKnowledgeAssetVersion, hero["version"]["id"])
        cover_version = db.get(FishKnowledgeAssetVersion, cover["version"]["id"])
        assert hero_version.status == cover_version.status == "DRAFT"
        assert hero_version.version == cover_version.version == 1
        assert hero_version.asset_role == "HERO"
        assert cover_version.asset_role == "COVER_LIST"
        assert hero_version.object_name.endswith("/hero/v1.webp")
        assert cover_version.object_name.endswith("/cover_list/v1.webp")
        assert db.scalar(select(FishCard).where(FishCard.species_id == "sharpbelly", FishCard.card_type == "HERO")).status == "DRAFT"
        assert len([name for name in bucket.blobs if "/hero/" in name or "/cover_list/" in name]) == 2
        assert not any(version.status == "ACTIVE" for version in db.scalars(select(FishKnowledgeAssetVersion)).all())
        hero_workspace = get_unified_card_workspace("sharpbelly", db=db)["roles"]["HERO"]
        assert hero_workspace["active_version"] is None
        assert hero_workspace["draft_version"]["id"] == hero_version.id
        assert hero_workspace["selected_version_id"] == hero_version.id
        assert hero_workspace["card_id"] == db.scalar(select(FishCard.id).where(FishCard.asset_version_id == hero_version.id))

        with pytest.raises(HTTPException) as qa_required:
            activate_asset_version_v13(hero_version.id, db)
        assert qa_required.value.status_code == 409
        assert db.get(FishKnowledgeAssetVersion, hero_version.id).status == "DRAFT"

        save_bound_card_content(
            db,
            species_id="sharpbelly",
            role="HERO",
            version_id=hero_version.id,
            title="白条识别卡",
            description='{"type":"HERO","description":"中上层小型鱼"}',
        )
        _review_both_stages(hero_version.id, hero["batch_id"], db)
        published = activate_asset_version_v13(hero_version.id, db)
        assert published["status"] == "ACTIVE"
        assert db.get(FishKnowledgeAssetVersion, hero_version.id).status == "ACTIVE"
        assert db.get(FishKnowledgeAssetVersion, cover_version.id).status == "DRAFT"
        assert published["validation"]["public_detail_readback"] == "PASS"

        repeated = activate_asset_version_v13(hero_version.id, db)
        assert repeated["idempotent"] is True
        assert db.get(FishKnowledgeAssetVersion, hero_version.id).status == "ACTIVE"

        hero_v2 = asyncio.run(upload_single_asset_v13(
            species_id="sharpbelly",
            asset_role="HERO",
            file=MemoryUpload(_png((180, 40, 90)), "hero-v2.png"),
            allow_warnings=False,
            db=db,
        ))
        next_version = db.get(FishKnowledgeAssetVersion, hero_v2["version"]["id"])
        next_card = db.scalar(select(FishCard).where(FishCard.asset_version_id == next_version.id))
        assert next_version.status == "DRAFT"
        assert next_card is not None and next_card.status == "DRAFT"
        save_bound_card_content(
            db,
            species_id="sharpbelly",
            role="HERO",
            version_id=next_version.id,
            title="白条识别卡 v2",
            description='{"type":"HERO","description":"新版中上层小型鱼"}',
        )
        _review_both_stages(next_version.id, hero_v2["batch_id"], db, content_result="BLOCKED_CONTENT_MISMATCH")
        with pytest.raises(HTTPException) as retryable_failure:
            activate_asset_version_v13(next_version.id, db)
        assert retryable_failure.value.status_code == 409
        assert db.get(FishKnowledgeAssetVersion, hero_version.id).status == "ACTIVE"
        assert db.get(FishKnowledgeAssetVersion, next_version.id).status == "DRAFT"
        assert db.get(FishCard, next_card.id).status == "DRAFT"

        _review_both_stages(next_version.id, hero_v2["batch_id"], db)

        published_v2 = activate_asset_version_v13(next_version.id, db)
        assert published_v2["publication_status"] == "ACTIVE"
        assert db.get(FishKnowledgeAssetVersion, hero_version.id).status == "ARCHIVED"
        assert db.get(FishKnowledgeAssetVersion, next_version.id).status == "ACTIVE"
        assert db.get(FishCard, next_card.id).status == "ACTIVE"
        assert db.get(FishCard, db.scalar(select(FishCard.id).where(FishCard.asset_version_id == hero_version.id))).status == "DRAFT"
        hero_workspace = get_unified_card_workspace("sharpbelly", db=db)["roles"]["HERO"]
        assert hero_workspace["active_version"]["id"] == next_version.id
        assert hero_workspace["draft_version"] is None
        assert {row["status"] for row in hero_workspace["history"]["versions"]} == {"ACTIVE", "ARCHIVED"}
        public_detail = build_species_full_detail(db.get(FishSpecies, "sharpbelly"), db)
        public_hero = next(card for card in public_detail.cards if card.card_type == "HERO")
        assert public_hero.image_url == public_detail.knowledge_assets["HERO"]["image_url"] == next_version.image_url
        assert public_detail.knowledge_assets["HERO"]["version_id"] == next_version.id
    finally:
        db.close()


def test_transparent_single_upload_rejects_missing_alpha_without_creating_version(monkeypatch, tmp_path):
    db = _session(tmp_path)
    bucket = MemoryBucket()
    client = MemoryStorageClient(bucket)
    monkeypatch.setattr("app.fish_knowledge.import_batch.storage.Client", lambda: client)
    monkeypatch.setattr("app.fish_knowledge.import_batch.get_bucket_name", lambda: "test-bucket")
    try:
        result = asyncio.run(upload_single_asset_v13(
            species_id="sharpbelly",
            asset_role="TRANSPARENT_MAIN",
            file=MemoryUpload(_png((20, 120, 220)), "fish.png"),
            allow_warnings=False,
            db=db,
        ))
        assert result["validation_status"] == "INVALID"
        assert result["next_action"] == "EDIT_OR_REJECT"
        assert any(x["code"] == "ALPHA_REQUIRED" for x in result["item"]["validation_errors"])
        assert db.scalars(select(FishKnowledgeAssetVersion)).all() == []
    finally:
        db.close()


def test_warning_preflight_requires_confirmation_then_records_each_qa_stage(monkeypatch, tmp_path):
    db = _session(tmp_path)
    bucket = MemoryBucket()
    client = MemoryStorageClient(bucket)
    monkeypatch.setattr("app.fish_knowledge.import_batch.storage.Client", lambda: client)
    monkeypatch.setattr("app.fish_knowledge.publication.storage.Client", lambda: client)
    monkeypatch.setattr("app.fish_knowledge.import_batch.get_bucket_name", lambda: "test-bucket")
    monkeypatch.setattr("app.fish_knowledge.publication.get_bucket_name", lambda: "test-bucket")
    try:
        staged = asyncio.run(upload_single_asset_v13(
            species_id="sharpbelly",
            asset_role="HERO",
            file=MemoryUpload(_png((30, 120, 220)), "hero.jpg"),
            preflight_only=True,
            db=db,
        ))
        assert staged["validation_status"] == "WARNING"
        assert staged["next_action"] == "CONFIRM_WARNING"
        assert staged["needs_warning_confirmation"] is True
        assert staged["item"]["source_preview_url"]
        assert any(value["code"] == "MIME_EXTENSION_MISMATCH" for value in staged["item"]["validation_warnings"])
        assert db.scalars(select(FishKnowledgeAssetVersion)).all() == []

        with pytest.raises(HTTPException) as acknowledgement_required:
            execute_batch(staged["batch_id"], ExecuteBatchPayload(allow_warnings=False), db)
        assert acknowledgement_required.value.status_code == 409
        assert db.scalars(select(FishKnowledgeAssetVersion)).all() == []

        imported = execute_batch(staged["batch_id"], ExecuteBatchPayload(allow_warnings=True), db)
        row = imported["items"][0]
        version = db.get(FishKnowledgeAssetVersion, row["version_id"])
        assert version.status == "DRAFT"
        with pytest.raises(HTTPException) as combined_qa:
            review_asset_version_v13(version.id, AssetReviewPayload(
                batch_id=staged["batch_id"],
                visual_qa_result="PASS",
                content_qa_result="PASS",
            ), db)
        assert combined_qa.value.status_code == 400
        assert combined_qa.value.detail["code"] == "QA_STAGES_MUST_BE_SEPARATE"
        review_asset_version_v13(version.id, AssetReviewPayload(
            batch_id=staged["batch_id"],
            visual_qa_result="PASS",
            visual_qa_note="对照源预览确认主体、清晰度和裁切。",
            visual_qa_reviewer="visual-reviewer",
        ), db)
        workspace = get_unified_card_workspace("sharpbelly", db=db)["roles"]["HERO"]
        assert workspace["visual_qa"] == "PASS"
        assert workspace["content_qa"] == "PENDING"
        saved_content = save_bound_card_content(
            db,
            species_id="sharpbelly",
            role="HERO",
            version_id=version.id,
            title="白条概览",
            description='{"tag":"银色快鱼","description":"体态细长"}',
        )
        assert saved_content["content_revision"] == 2
        assert saved_content["content_qa_result"] == "PENDING"
        review_asset_version_v13(version.id, AssetReviewPayload(
            batch_id=staged["batch_id"],
            content_qa_result="PASS",
            content_qa_note="已核对物种身份、HERO 角色和本 revision 字段。",
            content_qa_reviewer="content-reviewer",
        ), db)
        audits = db.scalars(select(FishKnowledgeAssetQAAudit).where(FishKnowledgeAssetQAAudit.version_id == version.id).order_by(FishKnowledgeAssetQAAudit.id)).all()
        assert [(audit.qa_stage, audit.result, audit.reviewer) for audit in audits] == [
            ("VISUAL", "PASS", "visual-reviewer"),
            ("CONTENT", "PASS", "content-reviewer"),
        ]
        assert audits[1].content_revision == 2
        assert audits[1].card_id is not None
        assert db.query(FishAssetImportBatch).filter_by(batch_id=staged["batch_id"]).one().warnings_acknowledged is True
    finally:
        db.close()


def test_publish_readback_failure_reports_database_commit_not_rollback(monkeypatch, tmp_path):
    db = _session(tmp_path)
    bucket = MemoryBucket()
    client = MemoryStorageClient(bucket)
    monkeypatch.setattr("app.fish_knowledge.import_batch.storage.Client", lambda: client)
    monkeypatch.setattr("app.fish_knowledge.publication.storage.Client", lambda: client)
    monkeypatch.setattr("app.fish_knowledge.import_batch.get_bucket_name", lambda: "test-bucket")
    monkeypatch.setattr("app.fish_knowledge.publication.get_bucket_name", lambda: "test-bucket")
    try:
        upload = asyncio.run(upload_single_asset_v13(
            species_id="sharpbelly",
            asset_role="COVER_HERO",
            file=MemoryUpload(_png((15, 90, 180), size=(673, 923)), "cover.png"),
            allow_warnings=True,
            db=db,
        ))
        assert upload.get("version"), upload.get("item", {}).get("validation_errors", upload)
        version_id = upload["version"]["id"]
        review_asset_version_v13(version_id, AssetReviewPayload(
            batch_id=upload["batch_id"],
            visual_qa_result="PASS",
            visual_qa_note="Hero image visually checked.",
            visual_qa_reviewer="visual-reviewer",
        ), db)
        review_asset_version_v13(version_id, AssetReviewPayload(
            batch_id=upload["batch_id"],
            content_qa_result="PASS",
            content_qa_note="Species and role metadata checked.",
            content_qa_reviewer="content-reviewer",
        ), db)

        def fail_public_readback(*_args, **_kwargs):
            raise RuntimeError("simulated public API serializer outage")

        monkeypatch.setattr("app.fish_knowledge.api.build_species_full_detail", fail_public_readback)
        with pytest.raises(HTTPException) as failed:
            activate_asset_version_v13(version_id, db)
        assert failed.value.status_code == 502
        assert failed.value.detail["code"] == "PUBLICATION_COMMITTED_READBACK_FAILED"
        assert failed.value.detail["publication_committed"] is True
        db.expire_all()
        assert db.get(FishKnowledgeAssetVersion, version_id).status == "ACTIVE"
        assert db.query(FishKnowledgeAssetQAAudit).filter_by(version_id=version_id).count() == 2
        audit = db.query(FishKnowledgePublicationAudit).filter_by(asset_version_id=version_id).one()
        assert json.loads(audit.validation_json)["public_detail_readback"] == "FAILED_AFTER_COMMIT"
    finally:
        db.close()


def test_public_api_and_image_readback_is_persisted_on_publication_audit(monkeypatch, tmp_path):
    db = _session(tmp_path)
    bucket = MemoryBucket()
    client = MemoryStorageClient(bucket)
    monkeypatch.setattr("app.fish_knowledge.import_batch.storage.Client", lambda: client)
    monkeypatch.setattr("app.fish_knowledge.publication.storage.Client", lambda: client)
    monkeypatch.setattr("app.fish_knowledge.import_batch.get_bucket_name", lambda: "test-bucket")
    monkeypatch.setattr("app.fish_knowledge.publication.get_bucket_name", lambda: "test-bucket")
    try:
        species = db.get(FishSpecies, "sharpbelly")
        species.status = "ACTIVE"
        db.commit()
        upload = asyncio.run(upload_single_asset_v13(
            species_id="sharpbelly",
            asset_role="COVER_HERO",
            file=MemoryUpload(_png((55, 135, 205), size=(673, 923)), "cover.png"),
            allow_warnings=True,
            db=db,
        ))
        version_id = upload["version"]["id"]
        review_asset_version_v13(version_id, AssetReviewPayload(
            batch_id=upload["batch_id"],
            visual_qa_result="PASS",
            visual_qa_note="Hero preview checked.",
            visual_qa_reviewer="visual-reviewer",
        ), db)
        review_asset_version_v13(version_id, AssetReviewPayload(
            batch_id=upload["batch_id"],
            content_qa_result="PASS",
            content_qa_note="Species identity and role checked.",
            content_qa_reviewer="content-reviewer",
        ), db)
        activate_asset_version_v13(version_id, db)
        version = db.get(FishKnowledgeAssetVersion, version_id)
        image_sha = json.loads(version.metadata_json)["derived_sha256"]
        with pytest.raises(HTTPException) as client_before_api:
            record_client_acceptance_v14(version_id, ClientAcceptancePayload(
                result="CLIENT_PASSED",
                reviewer="android-validator",
                observed_version_id=version_id,
                detail="Client shown.",
            ), db)
        assert client_before_api.value.status_code == 409
        assert client_before_api.value.detail["code"] == "PUBLIC_API_CHECK_REQUIRED"
        result = record_public_api_readback_v14(version_id, PublicAPIReadbackPayload(
            status="PUBLIC_API_OK",
            reviewer="api-reviewer",
            observed_version_id=version_id,
            public_image_sha256=image_sha,
            preview_image_sha256=image_sha,
        ), db)
        assert result["status"] == "PUBLIC_API_OK"
        assert result["version_id"] == version_id
        assert result["reviewer"] == "api-reviewer"
        workspace = get_unified_card_workspace("sharpbelly", db=db)
        publication = workspace["roles"]["COVER_HERO"]["history"]["publication_audits"][0]
        assert publication["validation"]["public_api_check"]["status"] == "PUBLIC_API_OK"
        client_result = record_client_acceptance_v14(version_id, ClientAcceptancePayload(
            result="CLIENT_PASSED",
            reviewer="android-validator",
            observed_version_id=version_id,
            detail="Authorized Android device showed the published hero image.",
        ), db)
        assert client_result["status"] == "CLIENT_PASSED"
        workspace = get_unified_card_workspace("sharpbelly", db=db)
        publication = workspace["roles"]["COVER_HERO"]["history"]["publication_audits"][0]
        assert publication["validation"]["client_acceptance"]["status"] == "CLIENT_PASSED"
        with pytest.raises(HTTPException) as wrong_version:
            record_public_api_readback_v14(version_id, PublicAPIReadbackPayload(
                status="PUBLIC_API_OK",
                reviewer="api-reviewer",
                observed_version_id=version_id + 1,
                public_image_sha256=image_sha,
                preview_image_sha256=image_sha,
            ), db)
        assert wrong_version.value.status_code == 400
        with pytest.raises(HTTPException) as wrong_client_version:
            record_client_acceptance_v14(version_id, ClientAcceptancePayload(
                result="CLIENT_PASSED",
                reviewer="android-validator",
                observed_version_id=version_id + 1,
            ), db)
        assert wrong_client_version.value.status_code == 400
    finally:
        db.close()


def test_warning_execute_requires_explicit_acknowledgement(monkeypatch, tmp_path):
    db = _session(tmp_path)
    batch_id = "FK_WARN_ACK_001"
    db.add(FishAssetImportBatch(
        batch_id=batch_id,
        source_gcs_uri=f"gs://test-bucket/fish-assets/imports/{batch_id}/",
        status="READY",
        created_by="qa-operator",
        warning_files=1,
    ))
    db.commit()

    with pytest.raises(HTTPException) as blocked:
        execute_batch(batch_id, ExecuteBatchPayload(allow_warnings=False), db)
    assert blocked.value.status_code == 409
    assert db.query(FishAssetImportBatch).filter_by(batch_id=batch_id).one().warnings_acknowledged is False

    monkeypatch.setattr("app.fish_knowledge.import_batch._run_import", lambda _db, batch: setattr(batch, "status", "COMPLETED") or {})
    execute_batch(batch_id, ExecuteBatchPayload(allow_warnings=True), db)
    batch = db.query(FishAssetImportBatch).filter_by(batch_id=batch_id).one()
    assert batch.warnings_acknowledged is True
    assert batch.warnings_acknowledged_by == "qa-operator"
    assert batch.warnings_acknowledged_at is not None
    db.close()


def test_batch_upload_path_safety_and_interrupted_upload_resume(monkeypatch, tmp_path):
    db = _session(tmp_path)
    bucket = MemoryBucket()
    client = MemoryStorageClient(bucket)
    monkeypatch.setattr("app.fish_knowledge.import_batch.storage.Client", lambda: client)
    monkeypatch.setattr("app.fish_knowledge.import_batch.get_bucket_name", lambda: "test-bucket")
    batch_id = "FK_BATCH_RESUME_001"
    create_local_batch(CreateLocalBatchPayload(batch_id=batch_id), db)
    path = "sharpbelly/01_hero.png"
    data = _png((40, 100, 180))

    first = asyncio.run(upload_batch_file(batch_id, path, MemoryUpload(data), db))
    resumed = asyncio.run(upload_batch_file(batch_id, path, MemoryUpload(data), db))
    assert first["idempotent"] is False
    assert resumed["idempotent"] is True

    batch = db.query(FishAssetImportBatch).filter_by(batch_id=batch_id).one()
    batch.status = "READY"
    db.commit()
    resumed_after_scan = asyncio.run(upload_batch_file(batch_id, path, MemoryUpload(data), db))
    assert resumed_after_scan["idempotent"] is True

    with pytest.raises(HTTPException) as path_error:
        asyncio.run(upload_batch_file(batch_id, "../escape.png", MemoryUpload(data), db))
    assert path_error.value.status_code == 400
    with pytest.raises(HTTPException) as zip_error:
        asyncio.run(upload_batch_file(batch_id, "assets.zip", MemoryUpload(data, "assets.zip"), db))
    assert zip_error.value.status_code == 400
    with pytest.raises(HTTPException) as conflict:
        asyncio.run(upload_batch_file(batch_id, path, MemoryUpload(_png((180, 50, 40))), db))
    assert conflict.value.status_code == 409
    db.close()
