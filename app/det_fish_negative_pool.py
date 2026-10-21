"""DET_FISH-only negative-pool APIs with append-only membership auditing."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from google.cloud import storage
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import get_db
from app.factory import get_bucket_name
from app.models import BatchCropReview, DetectorNegativePoolEntry, DetectorNegativePoolEvent, ImageAsset, ReviewEvent
from app.presence import FishPresenceResult

router = APIRouter(prefix="/api/det-fish/negative-pool", tags=["det-fish-negative-pool"])
LEGACY_PREFIX = "detector-datasets/DET_DS_v0.1"


class CandidateRef(BaseModel):
    image_asset_id: int = Field(gt=0)
    source_record_id: str = Field(min_length=1, max_length=256)


class AddRequest(BaseModel):
    items: list[CandidateRef] = Field(min_length=1, max_length=100)
    source_kind: str = Field(pattern="^(PRESENCE_NO_FISH|DET_DS_V0_1)$")
    actor: str = Field(default="det-fish-console-ui", min_length=1, max_length=256)
    confirm_pending_as_no_fish: bool = False


class RemoveRequest(BaseModel):
    entry_ids: list[int] = Field(min_length=1, max_length=100)
    actor: str = Field(default="det-fish-console-ui", min_length=1, max_length=256)


def _json(value: str | None) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _media(image: ImageAsset) -> tuple[str, str]:
    root = f"/media/{image.batch_id}/{image.image_id}"
    return root + "?variant=thumbnail", root


def _no_fish_event(db: Session, image_id: int) -> ReviewEvent | None:
    events = db.scalars(select(ReviewEvent).where(
        ReviewEvent.image_asset_id == image_id,
        ReviewEvent.action == "presence_override_update",
    ).order_by(ReviewEvent.created_at.desc(), ReviewEvent.id.desc())).all()
    return next((event for event in events if _json(event.after_json).get("human_override") == "no_fish"), None)


def _human_presence(db: Session, presence: FishPresenceResult | None) -> bool:
    return bool(
        presence
        and presence.provider == "human_override"
        and presence.status == "no_fish"
        and _json(presence.evidence_json).get("human_override") == "no_fish"
        and _no_fish_event(db, presence.image_asset_id)
    )


def _confirmed_event_asset_ids(db: Session) -> set[int]:
    result: set[int] = set()
    events = db.scalars(select(ReviewEvent).where(ReviewEvent.action == "presence_override_update")).all()
    for event in events:
        if _json(event.after_json).get("human_override") == "no_fish":
            result.add(event.image_asset_id)
    return result


def _positive_conflict(db: Session, image: ImageAsset) -> bool:
    if image.review_status == "approved" and str(image.truth_species or "").strip():
        return True
    return db.scalar(select(BatchCropReview.id).where(
        BatchCropReview.image_asset_id == image.id,
        BatchCropReview.status.in_({"ACCEPTED", "TRAINING_READY"}),
        BatchCropReview.accepted_bbox_json.is_not(None),
    )) is not None


def _asset_blockers(db: Session, image: ImageAsset) -> list[str]:
    blockers = []
    if not str(image.gcs_uri or "").startswith("gs://"):
        blockers.append("原图 GCS 来源不可核验")
    if not image.training_eligible:
        blockers.append("图片当前不具备训练资格")
    if image.duplicate_of_image_asset_id:
        blockers.append("图片是已知重复项")
    if _positive_conflict(db, image):
        blockers.append("存在已确认鱼体标签冲突")
    return blockers


def _source_sha256(image: ImageAsset, client=None) -> str:
    uri = str(image.gcs_uri or "").strip()
    if not uri.startswith("gs://") or "/" not in uri[5:]:
        raise ValueError("source image URI is not a complete gs:// URI")
    bucket_name, object_name = uri[5:].split("/", 1)
    if not bucket_name or not object_name:
        raise ValueError("source image URI is not a complete gs:// URI")
    client = client or storage.Client()
    blob = client.bucket(bucket_name).blob(object_name)
    if not blob.exists(timeout=15):
        raise FileNotFoundError("source GCS object does not exist")
    digest = hashlib.sha256()
    with blob.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _legacy_rows(client=None):
    client = client or storage.Client()
    bucket = client.bucket(get_bucket_name())
    report_blob = bucket.blob(LEGACY_PREFIX + "/bootstrap_report.json")
    manifest_blob = bucket.blob(LEGACY_PREFIX + "/bootstrap_manifest.json")
    if not report_blob.exists(timeout=15) or not manifest_blob.exists(timeout=15):
        raise FileNotFoundError("DET_DS_v0.1 bootstrap report or manifest is missing")
    report_bytes, manifest_bytes = report_blob.download_as_bytes(timeout=60), manifest_blob.download_as_bytes(timeout=120)
    report, manifest = json.loads(report_bytes.decode("utf-8")), json.loads(manifest_bytes.decode("utf-8"))
    if not isinstance(report, dict) or not isinstance(manifest, list):
        raise ValueError("DET_DS_v0.1 report/manifest has an invalid shape")
    if report.get("source") != "FishPresenceResult/google-vision-presence-v0.4":
        raise ValueError("DET_DS_v0.1 source identity does not match its recorded Vision source")
    negatives = [row for row in manifest if isinstance(row, dict) and row.get("negative") is True and str(row.get("presence_status") or "").lower() == "no_fish"]
    if int(report.get("negatives", -1)) != len(negatives):
        raise ValueError("DET_DS_v0.1 negative count does not match its manifest")
    identity = {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "source_identity_matches_report": True,
        "manifest_negative_count_matches_report": True,
    }
    return negatives, report, identity


def _legacy_asset(db: Session, row: dict[str, Any]) -> ImageAsset | None:
    try:
        image = db.get(ImageAsset, int(row.get("image_asset_id")))
        if image:
            return image
    except (TypeError, ValueError):
        pass
    batch_id, image_id = str(row.get("batch_id") or ""), str(row.get("image_id") or "")
    if batch_id and image_id:
        return db.scalar(select(ImageAsset).where(ImageAsset.batch_id == batch_id, ImageAsset.image_id == image_id))
    return None


def _active(db: Session, image_id: int) -> DetectorNegativePoolEntry | None:
    return db.scalar(select(DetectorNegativePoolEntry).where(
        DetectorNegativePoolEntry.image_asset_id == image_id,
        DetectorNegativePoolEntry.state == "ACTIVE",
    ))


def _entry_dict(db: Session, entry: DetectorNegativePoolEntry, image: ImageAsset) -> dict[str, Any]:
    thumb, original = _media(image)
    snapshot = _json(entry.source_snapshot_json)
    blockers = _asset_blockers(db, image)
    events = db.scalars(select(DetectorNegativePoolEvent).where(
        DetectorNegativePoolEvent.pool_entry_id == entry.id,
    ).order_by(DetectorNegativePoolEvent.occurred_at, DetectorNegativePoolEvent.id)).all()
    return {
        "entry_id": entry.id, "image_asset_id": image.id, "image_id": image.image_id,
        "file_name": image.file_name, "batch_id": image.batch_id,
        "batch_source": image.batch.source if image.batch else None,
        "thumbnail_url": thumb, "original_url": original,
        "source_kind": entry.source_kind, "source_record_id": entry.source_record_id,
        "source_dataset_version": entry.source_dataset_version, "source_status": entry.source_status,
        "source_provider": entry.source_provider, "source_model_version": entry.source_model_version,
        "source_human_override": snapshot.get("human_override"),
        "source_machine_status": snapshot.get("machine_status"),
        "source_image_sha256": entry.source_image_sha256,
        "source_evidence_sha256": entry.source_evidence_sha256,
        "added_by": entry.added_by, "added_at": _iso(entry.added_at), "state": entry.state,
        "removed_by": entry.removed_by, "removed_at": _iso(entry.removed_at),
        "eligible_for_future_build": entry.state == "ACTIVE" and not blockers,
        "eligibility_blockers": blockers,
        "audit_events": [{"action": e.action, "actor": e.actor, "occurred_at": _iso(e.occurred_at),
                          "before_state": e.before_state, "after_state": e.after_state} for e in events],
    }


@router.get("/summary")
def summary(db: Session = Depends(get_db)) -> dict[str, Any]:
    active_count = int(db.scalar(select(func.count(DetectorNegativePoolEntry.id)).where(DetectorNegativePoolEntry.state == "ACTIVE")) or 0)
    removed_count = int(db.scalar(select(func.count(DetectorNegativePoolEntry.id)).where(DetectorNegativePoolEntry.state == "REMOVED")) or 0)
    return {"active_count": active_count, "removed_count": removed_count,
            "count_semantics": "DET_FISH-specific active membership rows; same query as the list",
            "dataset_version": "DET_DS_v0.2", "frozen_dataset_count_is_separate": True}


@router.get("")
def list_pool(state: str = Query(default="ACTIVE", pattern="^(ACTIVE|REMOVED|ALL)$"),
              limit: int = Query(default=25, ge=1, le=100), offset: int = Query(default=0, ge=0),
              batch_id: str | None = None, db: Session = Depends(get_db)) -> dict[str, Any]:
    query = select(DetectorNegativePoolEntry, ImageAsset).join(ImageAsset, ImageAsset.id == DetectorNegativePoolEntry.image_asset_id)
    count_query = select(func.count(DetectorNegativePoolEntry.id)).join(ImageAsset, ImageAsset.id == DetectorNegativePoolEntry.image_asset_id)
    if state != "ALL":
        query, count_query = query.where(DetectorNegativePoolEntry.state == state), count_query.where(DetectorNegativePoolEntry.state == state)
    if batch_id:
        query, count_query = query.where(ImageAsset.batch_id == batch_id), count_query.where(ImageAsset.batch_id == batch_id)
    total = int(db.scalar(count_query) or 0)
    pairs = db.execute(query.order_by(DetectorNegativePoolEntry.added_at.desc(), DetectorNegativePoolEntry.id.desc()).limit(limit).offset(offset)).all()
    return {"total": total, "limit": limit, "offset": offset,
            "items": [_entry_dict(db, entry, image) for entry, image in pairs]}


@router.get("/candidates")
def candidates(source_kind: str = Query(default="PRESENCE_NO_FISH", pattern="^(PRESENCE_NO_FISH|DET_DS_V0_1)$"),
               limit: int = Query(default=25, ge=1, le=100), offset: int = Query(default=0, ge=0),
               batch_id: str | None = None,
               verification: str = Query(default="ALL", pattern="^(ALL|HUMAN|PENDING)$"),
               db: Session = Depends(get_db)) -> dict[str, Any]:
    if source_kind == "PRESENCE_NO_FISH":
        query = select(ImageAsset, FishPresenceResult).join(FishPresenceResult, FishPresenceResult.image_asset_id == ImageAsset.id).where(FishPresenceResult.status == "no_fish")
        count_query = select(func.count(FishPresenceResult.id)).join(ImageAsset, ImageAsset.id == FishPresenceResult.image_asset_id).where(FishPresenceResult.status == "no_fish")
        if batch_id:
            query, count_query = query.where(ImageAsset.batch_id == batch_id), count_query.where(ImageAsset.batch_id == batch_id)
        pairs_all = db.execute(query.order_by(ImageAsset.id)).all()
        confirmed_events = _confirmed_event_asset_ids(db)
        pairs = []
        for image, presence in pairs_all:
            manual = bool(presence.provider == "human_override" and presence.status == "no_fish"
                          and _json(presence.evidence_json).get("human_override") == "no_fish"
                          and image.id in confirmed_events)
            entry = _active(db, image.id)
            verified = manual or bool(entry)
            if verification == "HUMAN" and not verified:
                continue
            if verification == "PENDING" and verified:
                continue
            pairs.append((image, presence))
        total = len(pairs)
        pairs = pairs[offset:offset + limit]
        assets = [image for image, _ in pairs]
        pool_entries = db.scalars(select(DetectorNegativePoolEntry).where(DetectorNegativePoolEntry.image_asset_id.in_([x.id for x in assets] or [-1]))).all()
        pool_by_asset = {x.image_asset_id: x for x in pool_entries}
        items = []
        for image, presence in pairs:
            thumb, original = _media(image)
            manual = bool(presence.provider == "human_override" and presence.status == "no_fish"
                          and _json(presence.evidence_json).get("human_override") == "no_fish"
                          and image.id in confirmed_events)
            blockers = _asset_blockers(db, image)
            entry = pool_by_asset.get(image.id)
            items.append({
                "image_asset_id": image.id, "source_record_id": str(presence.id), "image_id": image.image_id,
                "file_name": image.file_name, "batch_id": image.batch_id,
                "batch_source": image.batch.source if image.batch else None,
                "thumbnail_url": thumb, "original_url": original, "source_kind": source_kind,
                "source_status": presence.status, "source_provider": presence.provider,
                "source_model_version": presence.model_version, "human_confirmed_source": manual,
                "human_confirmation_status": "已核验人工无鱼" if manual else ("已人工加入 DET_FISH 池" if entry and entry.state == "ACTIVE" else "待人工确认"),
                "machine_status": _json(presence.evidence_json).get("machine_status", presence.status),
                "already_added": bool(entry and entry.state == "ACTIVE"), "pool_state": entry.state if entry else None,
                "eligible_to_add": not blockers and not (entry and entry.state == "ACTIVE"),
                "eligibility_blockers": blockers,
            })
        return {"source_kind": source_kind, "total": total, "limit": limit, "offset": offset, "items": items}

    try:
        rows, report, identity = _legacy_rows()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"无法核验 DET_DS_v0.1 历史候选：{type(exc).__name__}: {exc}") from exc
    mapped_all = [(row, _legacy_asset(db, row)) for row in rows]
    all_asset_ids = [image.id for _, image in mapped_all if image]
    all_pool_entries = db.scalars(select(DetectorNegativePoolEntry).where(
        DetectorNegativePoolEntry.image_asset_id.in_(all_asset_ids or [-1])
    )).all()
    all_pool_by_asset = {entry.image_asset_id: entry for entry in all_pool_entries}
    if verification != "ALL":
        filtered = []
        for row, image in mapped_all:
            entry = all_pool_by_asset.get(image.id) if image else None
            is_confirmed = bool(entry and entry.state == "ACTIVE")
            if verification == "HUMAN" and is_confirmed:
                filtered.append((row, image))
            elif verification == "PENDING" and not is_confirmed:
                filtered.append((row, image))
        mapped_all = filtered
    total = len(mapped_all)
    mapped = mapped_all[offset:offset + limit]
    asset_ids = [image.id for _, image in mapped if image]
    pool_entries = db.scalars(select(DetectorNegativePoolEntry).where(DetectorNegativePoolEntry.image_asset_id.in_(asset_ids or [-1]))).all()
    pool_by_asset = {x.image_asset_id: x for x in pool_entries}
    items = []
    for row, image in mapped:
        if image is None:
            items.append({"image_asset_id": None, "source_record_id": str(row.get("image_asset_id") or row.get("image_id") or "unknown"),
                          "image_id": row.get("image_id"), "file_name": row.get("image_id"), "batch_id": row.get("batch_id"),
                          "source_kind": source_kind, "source_status": "no_fish", "source_provider": "google_vision",
                          "source_model_version": "google-vision-presence-v0.4", "human_confirmation_status": "待人工确认",
                          "already_added": False, "eligible_to_add": False,
                          "eligibility_blockers": ["原始图片记录不在当前数据库中"]})
            continue
        thumb, original = _media(image)
        blockers = _asset_blockers(db, image)
        entry = pool_by_asset.get(image.id)
        items.append({"image_asset_id": image.id, "source_record_id": str(row.get("image_asset_id") or image.id),
                      "image_id": image.image_id, "file_name": image.file_name, "batch_id": image.batch_id,
                      "batch_source": image.batch.source if image.batch else None, "thumbnail_url": thumb,
                      "original_url": original, "source_kind": source_kind, "source_dataset_version": "DET_DS_v0.1",
                      "source_status": "no_fish", "source_provider": "google_vision",
                      "source_model_version": "google-vision-presence-v0.4",
                      "human_confirmation_status": "已人工加入 DET_FISH 池" if entry and entry.state == "ACTIVE" else "待人工确认",
                      "already_added": bool(entry and entry.state == "ACTIVE"), "pool_state": entry.state if entry else None,
                      "eligible_to_add": not blockers and not (entry and entry.state == "ACTIVE"),
                      "eligibility_blockers": blockers})
    return {"source_kind": source_kind, "total": total, "limit": limit, "offset": offset, "items": items,
            "source_artifact": {**identity, "report_negative_count": int(report["negatives"])}}


@router.get("/historical-v01-audit")
def historical_v01_audit(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Reconcile every v0.1 no-fish candidate to its DB row and source bytes."""
    try:
        rows, report, identity = _legacy_rows()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"历史数据源不可核验：{type(exc).__name__}: {exc}") from exc
    client = None
    counts = {"valid": 0, "active": 0, "pending": 0, "duplicate": 0, "missing": 0, "excluded": 0}
    seen_sha: set[str] = set()
    details = []
    for row in rows:
        image = _legacy_asset(db, row)
        if image is None:
            counts["missing"] += 1
            details.append({"source_record_id": str(row.get("image_asset_id") or "unknown"), "status": "MISSING_IMAGE_ASSET"})
            continue
        entry = db.scalar(select(DetectorNegativePoolEntry).where(DetectorNegativePoolEntry.image_asset_id == image.id))
        if entry and entry.state == "ACTIVE":
            counts["active"] += 1
        else:
            counts["pending"] += 1
        blockers = _asset_blockers(db, image)
        try:
            if client is None:
                client = storage.Client()
            source_sha = _source_sha256(image, client=client)
        except Exception as exc:
            counts["missing"] += 1
            details.append({"source_record_id": str(row.get("image_asset_id") or image.id), "image_asset_id": image.id,
                            "status": "SOURCE_OBJECT_UNAVAILABLE", "error_type": type(exc).__name__})
            continue
        duplicate = source_sha in seen_sha
        seen_sha.add(source_sha)
        if duplicate:
            counts["duplicate"] += 1
        if blockers or duplicate:
            counts["excluded"] += 1
        else:
            counts["valid"] += 1
        thumb, original = _media(image)
        details.append({"source_record_id": str(row.get("image_asset_id") or image.id), "image_asset_id": image.id,
                        "image_id": image.image_id, "batch_id": image.batch_id, "source_image_sha256": source_sha,
                        "source_object_verified": True, "pool_state": entry.state if entry else None,
                        "status": "EXCLUDED" if blockers or duplicate else "VALID_PENDING_CONFIRMATION",
                        "eligibility_blockers": blockers + (["原图 SHA 与历史候选重复"] if duplicate else []),
                        "thumbnail_url": thumb, "original_url": original})
    return {"status": "AUDITED", "source_dataset_version": "DET_DS_v0.1",
            "source_report_negative_count": int(report["negatives"]), "manifest_negative_rows": len(rows),
            "source_artifact": identity, "historical_records_total": len(rows), "valid_records": counts["valid"],
            "already_in_pool": counts["active"], "pending_confirmation": counts["pending"],
            "duplicate_records": counts["duplicate"], "missing_records": counts["missing"],
            "excluded_records": counts["excluded"], "records": details}


@router.post("/add")
def add_members(payload: AddRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    actor = payload.actor.strip() or "det-fish-console-ui"
    candidate_ids = [item.image_asset_id for item in payload.items]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise HTTPException(status_code=400, detail="同一请求包含重复图片")
    legacy_identity: set[tuple[int, str]] = set()
    if payload.source_kind == "DET_DS_V0_1":
        try:
            legacy_rows, _report, _identity = _legacy_rows()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"无法核验历史候选：{type(exc).__name__}: {exc}") from exc
        legacy_identity = {
            (int(row["image_asset_id"]), str(row["image_asset_id"]))
            for row in legacy_rows
            if str(row.get("image_asset_id") or "").isdigit()
        }

    client = None
    now = datetime.now(timezone.utc)
    added, already, rejected = [], [], []
    for item in payload.items:
        image = db.get(ImageAsset, item.image_asset_id)
        if image is None:
            rejected.append({"image_asset_id": item.image_asset_id, "reason": "原始图片记录不存在"})
            continue
        entry = db.scalar(select(DetectorNegativePoolEntry).where(
            DetectorNegativePoolEntry.image_asset_id == image.id,
        ).with_for_update())
        if entry and entry.state == "ACTIVE":
            already.append({"image_asset_id": image.id, "entry_id": entry.id})
            continue
        presence = db.scalar(select(FishPresenceResult).where(FishPresenceResult.image_asset_id == image.id))
        snapshot: dict[str, Any]
        source_status, source_provider, source_model = "no_fish", None, None
        source_evidence, dataset_version = None, None
        if payload.source_kind == "PRESENCE_NO_FISH":
            if not presence or presence.status != "no_fish" or str(presence.id) != item.source_record_id:
                rejected.append({"image_asset_id": image.id, "reason": "当前无鱼来源记录与请求不一致"})
                continue
            evidence = _json(presence.evidence_json)
            previous_human_review = _no_fish_event(db, image.id) if _human_presence(db, presence) else None
            if previous_human_review is None and not payload.confirm_pending_as_no_fish:
                rejected.append({"image_asset_id": image.id, "reason": "该记录只有 AI 无鱼判断；请确认所选图片确实无鱼"})
                continue
            snapshot = {
                "human_override": evidence.get("human_override"),
                "machine_status": evidence.get("machine_status", presence.status),
                "original_status": presence.status,
                "provider": presence.provider,
                "model_version": presence.model_version,
                "evidence_sha256": hashlib.sha256((presence.evidence_json or "").encode("utf-8")).hexdigest(),
                "source_human_review_event_id": previous_human_review.id if previous_human_review else None,
                # Every pool insertion is a DET_FISH-scoped human action. When
                # the source already has a human presence override, retain that
                # event as separate provenance instead of treating the pool
                # operation as unconfirmed.
                "det_fish_manual_confirmation": True,
                "det_fish_confirmation_basis": (
                    "existing_human_no_fish_review" if previous_human_review else "explicit_pool_add_confirmation"
                ),
                "det_fish_confirmation_actor": actor,
                "det_fish_confirmation_at": now.isoformat(),
            }
            source_provider, source_model = presence.provider, presence.model_version
            source_evidence = snapshot["evidence_sha256"]
            source_record_id = str(presence.id)
        else:
            source_record_id = item.source_record_id
            if (image.id, source_record_id) not in legacy_identity:
                rejected.append({"image_asset_id": image.id, "reason": "图片不属于已核验的 DET_DS_v0.1 无鱼候选"})
                continue
            if not payload.confirm_pending_as_no_fish:
                rejected.append({"image_asset_id": image.id, "reason": "历史 v0.1 标签来自 AI；请确认所选图片确实无鱼"})
                continue
            snapshot = {"original_status": "no_fish", "provider": "google_vision",
                        "model_version": "google-vision-presence-v0.4", "det_fish_manual_confirmation": True,
                        "det_fish_confirmation_actor": actor, "det_fish_confirmation_at": now.isoformat()}
            source_provider, source_model, dataset_version = "google_vision", "google-vision-presence-v0.4", "DET_DS_v0.1"

        blockers = _asset_blockers(db, image)
        if blockers:
            rejected.append({"image_asset_id": image.id, "reason": "; ".join(blockers)})
            continue
        before_state = entry.state if entry else None
        try:
            if client is None:
                client = storage.Client()
            source_sha = _source_sha256(image, client=client)
        except Exception as exc:
            rejected.append({"image_asset_id": image.id, "reason": f"无法读取原始场景图：{type(exc).__name__}"})
            continue
        duplicate = db.scalar(select(DetectorNegativePoolEntry.id).where(
            DetectorNegativePoolEntry.source_image_sha256 == source_sha,
            DetectorNegativePoolEntry.state == "ACTIVE",
            DetectorNegativePoolEntry.image_asset_id != image.id,
        ))
        if duplicate:
            rejected.append({"image_asset_id": image.id, "reason": "原图 SHA 已被另一条活动负样本记录占用"})
            continue
        if entry is None:
            entry = DetectorNegativePoolEntry(image_asset_id=image.id, source_kind=payload.source_kind,
                source_record_id=source_record_id, source_dataset_version=dataset_version, source_uri=image.gcs_uri,
                source_status=source_status, source_provider=source_provider, source_model_version=source_model,
                source_evidence_sha256=source_evidence, source_snapshot_json=json.dumps(snapshot, ensure_ascii=False, sort_keys=True),
                source_image_sha256=source_sha, state="ACTIVE", added_by=actor, added_at=now)
            db.add(entry)
            db.flush()
        else:
            entry.source_kind, entry.source_record_id = payload.source_kind, source_record_id
            entry.source_dataset_version, entry.source_uri = dataset_version, image.gcs_uri
            entry.source_status, entry.source_provider, entry.source_model_version = source_status, source_provider, source_model
            entry.source_evidence_sha256 = source_evidence
            entry.source_snapshot_json, entry.source_image_sha256 = json.dumps(snapshot, ensure_ascii=False, sort_keys=True), source_sha
            entry.state, entry.added_by, entry.added_at = "ACTIVE", actor, now
            entry.removed_by, entry.removed_at = None, None
        db.add(DetectorNegativePoolEvent(pool_entry_id=entry.id, image_asset_id=image.id,
            action="ADD" if before_state is None else "RE_ADD", actor=actor, occurred_at=now,
            before_state=before_state, after_state="ACTIVE", source_snapshot_json=entry.source_snapshot_json))
        added.append({"image_asset_id": image.id, "entry_id": entry.id, "source_image_sha256": source_sha})
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="负样本成员或原图 SHA 已被并发操作占用；刷新后重试") from exc
    active_count = int(db.scalar(select(func.count(DetectorNegativePoolEntry.id)).where(DetectorNegativePoolEntry.state == "ACTIVE")) or 0)
    return {"added": added, "already_added": already, "rejected": rejected, "active_count": active_count}


@router.post("/remove")
def remove_members(payload: RemoveRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    actor = payload.actor.strip() or "det-fish-console-ui"
    if len(set(payload.entry_ids)) != len(payload.entry_ids):
        raise HTTPException(status_code=400, detail="同一请求包含重复负样本记录")
    removed, already_removed, missing = [], [], []
    now = datetime.now(timezone.utc)
    for entry_id in payload.entry_ids:
        entry = db.scalar(select(DetectorNegativePoolEntry).where(
            DetectorNegativePoolEntry.id == entry_id,
        ).with_for_update())
        if entry is None:
            missing.append(entry_id)
            continue
        if entry.state == "REMOVED":
            already_removed.append(entry_id)
            continue
        before_state = entry.state
        entry.state, entry.removed_by, entry.removed_at = "REMOVED", actor, now
        db.add(DetectorNegativePoolEvent(pool_entry_id=entry.id, image_asset_id=entry.image_asset_id,
            action="REMOVE", actor=actor, occurred_at=now, before_state=before_state,
            after_state="REMOVED", source_snapshot_json=entry.source_snapshot_json))
        removed.append(entry.id)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="负样本状态已被并发操作改变；刷新后重试") from exc
    active_count = int(db.scalar(select(func.count(DetectorNegativePoolEntry.id)).where(DetectorNegativePoolEntry.state == "ACTIVE")) or 0)
    return {"removed": removed, "already_removed": already_removed, "missing": missing, "active_count": active_count}
