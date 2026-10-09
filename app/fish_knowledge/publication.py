from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from fastapi import HTTPException
from google.cloud import storage
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.factory import get_bucket_name
from app.fish_knowledge.cards import CARD_TYPE_ORDER, FishCard, normalize_card_type
from app.fish_knowledge.content import parse_card_content
from app.fish_knowledge.cover import FishSpeciesCover
from app.fish_knowledge.import_batch import (
    ASSET_ROLES,
    ASSET_DIR,
    FishCardContentRevision,
    FishKnowledgeAssetReview,
    FishKnowledgeAssetVersion,
    FishKnowledgePublicationAudit,
    _asset_role_for_version,
    _read_json,
)
from app.fish_knowledge.species import FishSpecies
from app.models import utcnow


CARD_ROLES = frozenset(CARD_TYPE_ORDER)


def _blocked(status: int, code: str, message: str) -> None:
    raise HTTPException(status_code=status, detail={"code": code, "message": message})


def _verify_media(version: FishKnowledgeAssetVersion, review: FishKnowledgeAssetReview) -> dict[str, Any]:
    metadata = _read_json(version.metadata_json, {})
    metadata = metadata if isinstance(metadata, dict) else {}
    expected_derived_sha = str(metadata.get("derived_sha256") or "").lower()
    expected_source_sha = str(metadata.get("source_sha256") or version.sha256 or "").lower()
    expected_generation = str(metadata.get("gcs_generation") or "")
    role = _asset_role_for_version(version)
    expected_url = f"/api/v1/fish/knowledge-media/{version.species_id}/{role.lower()}/v{version.version}.webp"
    expected_object = f"fish-assets/fish-knowledge/{version.species_id}/{ASSET_DIR[role]}/v{version.version}.webp"

    if version.image_url != expected_url or version.object_name != expected_object:
        _blocked(409, "ASSET_VERSION_PATH_MISMATCH", "素材 URL 或 GCS 路径与角色版本号不一致")

    if not re.fullmatch(r"[0-9a-f]{64}", expected_derived_sha):
        _blocked(409, "ASSET_PROVENANCE_MISMATCH", "缺少有效的派生图片 SHA-256")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", str(version.sha256 or "")):
        _blocked(409, "ASSET_PROVENANCE_MISMATCH", "素材来源 SHA-256 无效")
    if expected_source_sha != str(version.sha256).lower() or str(review.source_sha256).lower() != str(version.sha256).lower():
        _blocked(409, "ASSET_PROVENANCE_MISMATCH", "来源 SHA-256 与审核快照不一致")
    if review.object_name != version.object_name or not expected_generation.isdigit():
        _blocked(409, "ASSET_PROVENANCE_MISMATCH", "GCS 对象路径或 generation 与审核快照不一致")
    if str(review.object_generation or "") != expected_generation:
        _blocked(409, "ASSET_PROVENANCE_MISMATCH", "GCS generation 与审核快照不一致")

    try:
        client = storage.Client()
        blob = client.bucket(get_bucket_name()).blob(version.object_name)
        if not blob.exists(client):
            _blocked(409, "ASSET_MEDIA_MISSING", "发布素材在 GCS 中不存在")
        blob.reload(client)
        current_generation = str(getattr(blob, "generation", "") or "")
        if current_generation and current_generation != expected_generation:
            _blocked(409, "ASSET_GENERATION_CHANGED", "GCS 对象已被替换，generation 与审核快照不一致")
        body = blob.download_as_bytes(timeout=120)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={"code": "GCS_READ_FAILED", "message": "无法验证 GCS 发布素材"},
        ) from exc
    actual_sha = hashlib.sha256(body).hexdigest()
    if actual_sha != expected_derived_sha:
        _blocked(409, "ASSET_SHA_MISMATCH", "GCS 图片 SHA-256 与审核快照不一致")
    return {
        "object_name": version.object_name,
        "generation": expected_generation,
        "source_sha256": str(version.sha256).lower(),
        "derived_sha256": actual_sha,
        "file_exists": True,
    }


def _find_bound_card(db: Session, version: FishKnowledgeAssetVersion, role: str) -> FishCard | None:
    matches = db.scalars(select(FishCard).where(FishCard.asset_version_id == version.id)).all()
    if len(matches) > 1:
        _blocked(409, "DUPLICATE_VERSION_BINDING", "同一素材版本绑定了多条 FishCard")
    if not matches:
        return None
    card = matches[0]
    if (
        card.species_id != version.species_id
        or normalize_card_type(card.card_type) != role
        or card.image_url != version.image_url
    ):
        _blocked(409, "VERSION_BINDING_MISMATCH", "FishCard 与素材版本的鱼种、角色或图片 URL 不一致")
    return card


def _validate_card_content(card: FishCard, version: FishKnowledgeAssetVersion) -> dict[str, Any]:
    content = parse_card_content(card.description)
    if not isinstance(content, dict) or not content:
        _blocked(409, "STRUCTURED_CONTENT_MISSING", "该图片版本缺少已绑定的结构化知识内容")
    return {
        "card_id": card.id,
        "content_revision": int(card.content_revision or 1),
        "content_version_id": version.id,
    }


def publish_asset_version(
    db: Session,
    version_id: int,
    *,
    expected_batch_id: str | None = None,
    actor: str = "admin",
) -> dict[str, Any]:
    """Atomically activate the exact image version and its bound content row."""

    try:
        version = db.scalar(
            select(FishKnowledgeAssetVersion)
            .where(FishKnowledgeAssetVersion.id == version_id)
            .with_for_update()
        )
        if version is None:
            _blocked(404, "VERSION_NOT_FOUND", "素材版本不存在")
        if expected_batch_id is not None and version.batch_id != expected_batch_id:
            _blocked(404, "VERSION_NOT_FOUND", "该导入批次未包含此素材版本")
        role = _asset_role_for_version(version)
        if role not in ASSET_ROLES:
            _blocked(409, "INVALID_ASSET_ROLE", "素材版本的角色无效")
        if version.status not in {"DRAFT", "ACTIVE"}:
            _blocked(409, "VERSION_NOT_PUBLISHABLE", "只有 DRAFT 或当前 ACTIVE 版本可参与发布校验")

        # Serialize all publication changes for a species. PostgreSQL's row
        # lock prevents parallel publishers from racing the active-role scan.
        species = db.scalar(
            select(FishSpecies)
            .where(FishSpecies.id == version.species_id)
            .with_for_update()
        )
        if species is None or species.status == "DELETED":
            _blocked(404, "SPECIES_NOT_FOUND", "鱼种不存在或已删除")

        active_versions = db.scalars(select(FishKnowledgeAssetVersion).where(
            FishKnowledgeAssetVersion.species_id == version.species_id,
            FishKnowledgeAssetVersion.status == "ACTIVE",
        )).all()
        active_versions = [item for item in active_versions if _asset_role_for_version(item) == role]
        if len(active_versions) > 1:
            _blocked(409, "ACTIVE_VERSION_CONFLICT", "同一鱼种和角色存在多个 ACTIVE 素材版本")
        old_version = active_versions[0] if active_versions else None

        review = db.scalar(select(FishKnowledgeAssetReview).where(
            FishKnowledgeAssetReview.version_id == version.id,
            FishKnowledgeAssetReview.species_id == version.species_id,
            FishKnowledgeAssetReview.asset_role == role,
        ))
        if review is None or review.validation_result != "PASS":
            _blocked(409, "ASSET_VALIDATION_NOT_PASSED", "素材完整性校验尚未通过")
        if review.visual_qa_result != "PASS":
            _blocked(409, "VISUAL_QA_NOT_PASSED", "视觉 QA 尚未通过")
        if review.content_qa_result != "PASS":
            _blocked(409, "CONTENT_QA_NOT_PASSED", "内容 QA 尚未通过")
        media_check = _verify_media(version, review)

        card: FishCard | None = None
        content_check: dict[str, Any] = {}
        if role in CARD_ROLES:
            card = _find_bound_card(db, version, role)
            if card is None:
                _blocked(409, "CONTENT_BINDING_MISSING", "该素材版本没有绑定对应角色的 FishCard 草稿")
            content_check = _validate_card_content(card, version)

        active_cards = db.scalars(select(FishCard).where(
            FishCard.species_id == version.species_id,
            FishCard.status == "ACTIVE",
        )).all()
        active_cards = [item for item in active_cards if normalize_card_type(item.card_type) == role]
        if len(active_cards) > 1:
            _blocked(409, "ACTIVE_CARD_CONFLICT", "同一鱼种和角色存在多个 ACTIVE FishCard")
        old_card = active_cards[0] if active_cards else None

        # A repeated request for the already-published exact pair is a no-op.
        if version.status == "ACTIVE":
            if old_version is None or old_version.id != version.id:
                _blocked(409, "ACTIVE_VERSION_CONFLICT", "素材版本状态与角色 ACTIVE 状态不一致")
            if card is not None and card.status != "ACTIVE":
                _blocked(409, "ACTIVE_BINDING_CONFLICT", "线上图片版本对应的 FishCard 不是 ACTIVE")
            if card is not None and old_card is not None and old_card.id != card.id:
                _blocked(409, "ACTIVE_BINDING_CONFLICT", "存在未绑定到该版本的 ACTIVE FishCard")
            return {
                "success": True,
                "species_id": version.species_id,
                "role": role,
                "version_id": version.id,
                "image_url": version.image_url,
                "card_id": card.id if card else None,
                "publication_status": "ACTIVE",
                "idempotent": True,
                "validation": {"media": media_check, "content": content_check, "visual_qa": "PASS", "content_qa": "PASS"},
            }

        # Change the old card first to satisfy the existing partial unique
        # active-card index while preserving its historical row.
        if old_card is not None and old_card.id != (card.id if card else None):
            old_card.status = "DRAFT"
        for prior in active_versions:
            if prior.id != version.id:
                prior.status = "ARCHIVED"

        if role == "COVER_LIST":
            cover = db.scalar(select(FishSpeciesCover).where(FishSpeciesCover.species_id == species.id))
            if cover is None:
                cover = FishSpeciesCover(
                    species_id=species.id,
                    image_url=version.image_url,
                    style="ANIME_CARD",
                    title=f"{species.name_cn}图鉴卡",
                    status="ACTIVE",
                )
                db.add(cover)
            else:
                cover.image_url = version.image_url
                cover.status = "ACTIVE"
        elif card is not None:
            card.image_url = version.image_url
            card.status = "ACTIVE"

        version.status = "ACTIVE"
        validation = {
            "species_id": version.species_id,
            "role": role,
            "version_id": version.id,
            "image_url": version.image_url,
            "card_id": card.id if card else None,
            "media": media_check,
            "content": content_check,
            "visual_qa": review.visual_qa_result,
            "content_qa": review.content_qa_result,
            "review_id": review.id,
        }
        db.add(FishKnowledgePublicationAudit(
            species_id=version.species_id,
            asset_role=role,
            asset_version_id=version.id,
            card_id=card.id if card else None,
            previous_version_id=old_version.id if old_version and old_version.id != version.id else None,
            previous_card_id=old_card.id if old_card and (card is None or old_card.id != card.id) else None,
            publication_status="ACTIVE",
            validation_json=json.dumps(validation, ensure_ascii=False, sort_keys=True),
            actor=(actor or "admin")[:256],
            created_at=utcnow(),
        ))
        db.flush()

        # Validate the public response projection while all changes are still
        # in this transaction. The public API is built from these same rows.
        if role in CARD_ROLES:
            from app.fish_knowledge.api import _published_role_projection

            projection = _published_role_projection(db, species.id, role)
            if (
                projection.get("asset_image_url") != version.image_url
                or projection.get("card_image_url") != version.image_url
                or projection.get("asset_version_id") != version.id
                or projection.get("card_id") != (card.id if card else None)
            ):
                _blocked(409, "PUBLICATION_PROJECTION_MISMATCH", "公共详情投影与待发布版本不一致")

        db.commit()
        # Re-read the same public serializer after commit and verify the
        # committed API contract. DRAFT species are not externally visible
        # yet, but the internal public projection still remains testable.
        db.expire_all()
        refreshed_species = db.get(FishSpecies, version.species_id)
        if refreshed_species is None:
            _blocked(500, "PUBLICATION_READBACK_FAILED", "发布后无法重新读取鱼种")
        from app.fish_knowledge.api import build_species_full_detail

        public_detail = build_species_full_detail(refreshed_species, db)
        if role in CARD_ROLES:
            public_asset = public_detail.knowledge_assets.get(role)
            public_card = next((item for item in public_detail.cards if item.card_type == role), None)
            verified = bool(
                public_asset
                and public_asset.get("version_id") == version.id
                and public_asset.get("image_url") == version.image_url
                and public_card
                and public_card.id == card.id
                and public_card.image_url == version.image_url
            )
        else:
            public_asset = public_detail.cover_assets.get(role)
            verified = bool(
                public_asset
                and public_asset.get("version_id") == version.id
                and public_asset.get("image_url") == version.image_url
            )
        if not verified:
            _blocked(500, "PUBLICATION_READBACK_MISMATCH", "提交后公共详情没有返回目标 ACTIVE 版本")
        return {
            "success": True,
            "species_id": version.species_id,
            "role": role,
            "version_id": version.id,
            "image_url": version.image_url,
            "card_id": card.id if card else None,
            "publication_status": "ACTIVE",
            "idempotent": False,
            "validation": {
                "media": media_check,
                "content": content_check,
                "visual_qa": "PASS",
                "content_qa": "PASS",
                "public_detail_readback": "PASS",
                "species_visibility": refreshed_species.status,
            },
        }
    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail={"code": "PUBLICATION_ROLLED_BACK", "message": "发布事务失败，数据库变更已回滚"},
        ) from exc


def save_bound_card_content(
    db: Session,
    *,
    species_id: str,
    role: str,
    version_id: int,
    title: str,
    description: str,
    actor: str = "admin",
) -> dict[str, Any]:
    role = role.strip().upper()
    if role not in CARD_ROLES:
        _blocked(400, "INVALID_CARD_ROLE", "只有五张知识卡角色支持结构化内容编辑")
    version = db.scalar(
        select(FishKnowledgeAssetVersion)
        .where(FishKnowledgeAssetVersion.id == version_id)
        .with_for_update()
    )
    if version is None or version.species_id != species_id or _asset_role_for_version(version) != role:
        _blocked(404, "VERSION_NOT_FOUND", "指定版本不属于该鱼种和角色")
    if version.status != "DRAFT":
        _blocked(409, "ACTIVE_VERSION_READ_ONLY", "ACTIVE 或已归档版本的内容不可覆盖，请创建新素材版本")
    review = db.scalar(select(FishKnowledgeAssetReview).where(FishKnowledgeAssetReview.version_id == version.id))
    if review is not None and review.frozen_at is not None:
        _blocked(409, "ASSET_FROZEN", "冻结版本为只读；更新必须创建新版本")
    card = _find_bound_card(db, version, role)
    if card is None:
        _blocked(409, "CONTENT_BINDING_MISSING", "此版本尚未绑定 FishCard")
    db.refresh(card, with_for_update=True)
    normalized_content = parse_card_content(description)
    if not normalized_content:
        _blocked(400, "STRUCTURED_CONTENT_MISSING", "结构化知识内容不能为空")

    card.title = (title or "").strip()
    card.description = description
    card.image_url = version.image_url
    card.content_revision = int(card.content_revision or 1) + 1
    db.add(FishCardContentRevision(
        card_id=card.id,
        asset_version_id=version.id,
        content_revision=card.content_revision,
        title=card.title,
        description=card.description,
        image_url=card.image_url,
        created_by=(actor or "admin")[:256],
    ))
    if review is not None:
        review.content_qa_result = "PENDING"
        review.reviewed_at = None
        review.review_note = "结构化内容已更新；需要重新完成内容 QA"
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail={"code": "CONTENT_SAVE_CONFLICT", "message": "内容保存冲突，未写入更改"}) from exc
    return {
        "species_id": species_id,
        "role": role,
        "version_id": version.id,
        "card_id": card.id,
        "content_revision": card.content_revision,
        "binding_status": "BOUND_DRAFT",
    }
