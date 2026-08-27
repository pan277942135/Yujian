from __future__ import annotations

import hashlib
import io
import json
import math
from datetime import datetime, timezone

import imagehash
from fastapi import APIRouter, Depends, HTTPException
from google.cloud import storage
from PIL import Image
from pydantic import BaseModel, Field
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func, select
from sqlalchemy.orm import Session

from app.db import Base, get_db
from app.factory import DOWNLOAD_RETRY, get_bucket_name
from app.models import Batch, ImageAsset, ReviewEvent

router = APIRouter(prefix="/api/dedupe", tags=["image-dedupe"])

FINGERPRINT_VERSION = "phash-dhash-crophash-hist-v0.2"
FILTERABLE_REVIEW_STATUSES = {"pending", "needs_review", "hard_case"}
CENTER_CROP_RATIOS = (1.0, 0.9, 0.8, 0.7)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ImageFingerprint(Base):
    __tablename__ = "image_fingerprints"
    __table_args__ = (UniqueConstraint("image_asset_id", name="uq_fingerprint_image_asset"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_asset_id = Column(Integer, ForeignKey("image_assets.id"), nullable=False, index=True)
    batch_id = Column(String(128), nullable=False, index=True)
    sha256 = Column(String(64), nullable=False, index=True)
    phash_json = Column(Text, nullable=False)
    dhash = Column(String(64), nullable=False)
    crop_hash = Column(Text, nullable=False)
    histogram_json = Column(Text, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    fingerprint_version = Column(String(128), nullable=False, default=FINGERPRINT_VERSION)
    duplicate_group = Column(String(128), index=True)
    is_representative = Column(Boolean, nullable=False, default=True, index=True)
    duplicate_kind = Column(String(32))  # exact / near
    distance_to_representative = Column(Float)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)


class DedupeScanRequest(BaseModel):
    batch_id: str
    limit: int = Field(default=100, ge=1, le=250)
    rescan: bool = False


class DedupeFilterRequest(BaseModel):
    batch_id: str


def _center_crop(image: Image.Image, ratio: float) -> Image.Image:
    if ratio >= 0.999:
        return image
    width, height = image.size
    crop_w = max(1, int(width * ratio))
    crop_h = max(1, int(height * ratio))
    left = max(0, (width - crop_w) // 2)
    top = max(0, (height - crop_h) // 2)
    return image.crop((left, top, left + crop_w, top + crop_h))


def _histogram(image: Image.Image) -> list[float]:
    resized = image.resize((64, 64)).convert("RGB")
    raw = resized.histogram()
    result: list[float] = []
    for channel in range(3):
        values = raw[channel * 256 : (channel + 1) * 256]
        for bucket in range(16):
            result.append(float(sum(values[bucket * 16 : (bucket + 1) * 16])))
    norm = math.sqrt(sum(x * x for x in result)) or 1.0
    return [round(x / norm, 8) for x in result]


def fingerprint_bytes(content: bytes) -> dict:
    with Image.open(io.BytesIO(content)) as source:
        image = source.convert("RGB")
        width, height = image.size
        phashes = [str(imagehash.phash(_center_crop(image, ratio), hash_size=8)) for ratio in CENTER_CROP_RATIOS]
        dhash = str(imagehash.dhash(image, hash_size=8))
        crop_hash = str(
            imagehash.crop_resistant_hash(
                image,
                min_segment_size=250,
                segmentation_image_size=300,
            )
        )
        hist = _histogram(image)
    return {
        "sha256": hashlib.sha256(content).hexdigest(),
        "phashes": phashes,
        "dhash": dhash,
        "crop_hash": crop_hash,
        "histogram": hist,
        "width": width,
        "height": height,
    }


def _hamming_hex(left: str, right: str) -> int:
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except Exception:
        return 64


def _min_phash_distance(a: list[str], b: list[str]) -> int:
    return min((_hamming_hex(x, y) for x in a for y in b), default=64)


def _hist_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return max(0.0, min(1.0, sum(x * y for x, y in zip(a, b))))


def _crop_match(left: str, right: str) -> tuple[float, float]:
    """Return (coverage, avg_hamming) for the smaller crop-resistant hash."""
    try:
        left_hash = imagehash.hex_to_multihash(left)
        right_hash = imagehash.hex_to_multihash(right)
        if not left_hash.segment_hashes or not right_hash.segment_hashes:
            return 0.0, 64.0
        smaller, larger = (
            (left_hash, right_hash)
            if len(left_hash.segment_hashes) <= len(right_hash.segment_hashes)
            else (right_hash, left_hash)
        )
        matches, distance = smaller.hash_diff(larger, bit_error_rate=0.18)
        coverage = matches / max(1, len(smaller.segment_hashes))
        avg_distance = distance / max(1, matches)
        return round(float(coverage), 6), round(float(avg_distance), 6)
    except Exception:
        return 0.0, 64.0


def _row_features(row: ImageFingerprint) -> dict:
    return {
        "sha256": row.sha256,
        "phashes": json.loads(row.phash_json or "[]"),
        "dhash": row.dhash or "",
        "crop_hash": row.crop_hash or "",
        "histogram": json.loads(row.histogram_json or "[]"),
    }


def _compare_features(left: dict, right: dict) -> tuple[str | None, float | None]:
    if left["sha256"] == right["sha256"]:
        return "exact", 0.0

    p_dist = _min_phash_distance(left["phashes"], right["phashes"])
    d_dist = _hamming_hex(left["dhash"], right["dhash"])
    hist_sim = _hist_similarity(left["histogram"], right["histogram"])

    # Fast path for same framing under re-encoding/resizing.
    reencode_near = (p_dist <= 3 and hist_sim >= 0.965) or (p_dist <= 5 and d_dist <= 7 and hist_sim >= 0.985)
    if reencode_near:
        score = float(p_dist) + float(d_dist) * 0.20 + (1.0 - hist_sim) * 16.0
        return "near", round(score, 4)

    # Only invoke the more expensive crop-resistant comparison when cheap
    # structural + color evidence already says the pair is plausible.
    if p_dist > 12 or hist_sim < 0.94:
        return None, None

    crop_coverage, crop_hamming = _crop_match(left["crop_hash"], right["crop_hash"])
    crop_near = crop_coverage >= 0.50 and crop_hamming <= 6.0
    if not crop_near:
        return None, None

    crop_penalty = (1.0 - crop_coverage) * 4.0 + min(crop_hamming, 16.0) * 0.15
    score = float(p_dist) + float(d_dist) * 0.20 + (1.0 - hist_sim) * 16.0 + crop_penalty
    return "near", round(score, 4)


def duplicate_distance(left: ImageFingerprint, right: ImageFingerprint) -> tuple[str | None, float | None]:
    return _compare_features(_row_features(left), _row_features(right))


def _rebuild_groups(db: Session, batch_id: str) -> None:
    rows = db.scalars(
        select(ImageFingerprint).where(ImageFingerprint.batch_id == batch_id).order_by(ImageFingerprint.image_asset_id)
    ).all()
    if not rows:
        return

    features = [_row_features(row) for row in rows]
    parent = list(range(len(rows)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            kind, _ = _compare_features(features[i], features[j])
            if kind is not None:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for idx in range(len(rows)):
        groups.setdefault(find(idx), []).append(idx)

    for row in rows:
        row.duplicate_group = None
        row.is_representative = True
        row.duplicate_kind = None
        row.distance_to_representative = None

    for members in groups.values():
        if len(members) < 2:
            continue
        representative_idx = max(
            members,
            key=lambda idx: (rows[idx].width * rows[idx].height, -rows[idx].image_asset_id),
        )
        representative = rows[representative_idx]
        group_id = f"DUP_{batch_id}_{representative.image_asset_id}"
        representative.duplicate_group = group_id
        representative.is_representative = True

        representative_image = db.get(ImageAsset, representative.image_asset_id)
        if representative_image and not representative_image.group_id:
            representative_image.group_id = group_id

        for idx in members:
            if idx == representative_idx:
                continue
            row = rows[idx]
            kind, distance = _compare_features(features[representative_idx], features[idx])
            if kind is None:
                # Transitive clusters can contain a member that matched through an
                # intermediate transform. Keep the cluster intact and flag distance.
                kind, distance = "near", 999.0
            row.duplicate_group = group_id
            row.is_representative = False
            row.duplicate_kind = kind
            row.distance_to_representative = distance
            image = db.get(ImageAsset, row.image_asset_id)
            if image and not image.group_id:
                image.group_id = group_id

    db.commit()


def dedupe_summary(db: Session, batch_id: str) -> dict:
    batch = db.get(Batch, batch_id)
    if not batch:
        raise ValueError("batch not found")
    total = db.scalar(select(func.count()).select_from(ImageAsset).where(ImageAsset.batch_id == batch_id)) or 0
    scanned = db.scalar(select(func.count()).select_from(ImageFingerprint).where(ImageFingerprint.batch_id == batch_id)) or 0
    groups = db.scalar(
        select(func.count(func.distinct(ImageFingerprint.duplicate_group))).where(
            ImageFingerprint.batch_id == batch_id,
            ImageFingerprint.duplicate_group.is_not(None),
        )
    ) or 0
    duplicates = db.scalar(
        select(func.count()).select_from(ImageFingerprint).where(
            ImageFingerprint.batch_id == batch_id,
            ImageFingerprint.duplicate_group.is_not(None),
            ImageFingerprint.is_representative.is_(False),
        )
    ) or 0
    exact = db.scalar(
        select(func.count()).select_from(ImageFingerprint).where(
            ImageFingerprint.batch_id == batch_id,
            ImageFingerprint.is_representative.is_(False),
            ImageFingerprint.duplicate_kind == "exact",
        )
    ) or 0
    near = db.scalar(
        select(func.count()).select_from(ImageFingerprint).where(
            ImageFingerprint.batch_id == batch_id,
            ImageFingerprint.is_representative.is_(False),
            ImageFingerprint.duplicate_kind == "near",
        )
    ) or 0
    return {
        "batch_id": batch_id,
        "total": total,
        "scanned": scanned,
        "groups": groups,
        "duplicate_images": duplicates,
        "exact_duplicates": exact,
        "near_duplicates": near,
        "remaining": max(0, total - scanned),
    }


def scan_batch(db: Session, batch_id: str, limit: int = 100, rescan: bool = False) -> dict:
    if not db.get(Batch, batch_id):
        raise ValueError("batch not found")
    stmt = select(ImageAsset).where(ImageAsset.batch_id == batch_id).order_by(ImageAsset.id)
    if not rescan:
        stmt = stmt.outerjoin(ImageFingerprint, ImageFingerprint.image_asset_id == ImageAsset.id).where(ImageFingerprint.id.is_(None))
    images = db.scalars(stmt.limit(limit)).all()
    if not images:
        _rebuild_groups(db, batch_id)
        summary = dedupe_summary(db, batch_id)
        summary["processed"] = 0
        return summary

    bucket = storage.Client().bucket(get_bucket_name())
    processed = 0
    for image in images:
        content = bucket.blob(image.object_name).download_as_bytes(timeout=120, retry=DOWNLOAD_RETRY)
        fp = fingerprint_bytes(content)
        row = db.scalar(select(ImageFingerprint).where(ImageFingerprint.image_asset_id == image.id))
        if not row:
            row = ImageFingerprint(
                image_asset_id=image.id,
                batch_id=batch_id,
                sha256=fp["sha256"],
                phash_json="[]",
                dhash="",
                crop_hash=fp["crop_hash"],
                histogram_json="[]",
                width=fp["width"],
                height=fp["height"],
            )
            db.add(row)
        row.sha256 = fp["sha256"]
        row.phash_json = json.dumps(fp["phashes"])
        row.dhash = fp["dhash"]
        row.crop_hash = fp["crop_hash"]
        row.histogram_json = json.dumps(fp["histogram"])
        row.width = fp["width"]
        row.height = fp["height"]
        row.fingerprint_version = FINGERPRINT_VERSION
        row.updated_at = utcnow()
        db.commit()
        processed += 1

    # Fingerprinting is chunked for browser/Cloud Run resilience. The O(N²)
    # clustering pass is deferred until the whole batch is fingerprinted.
    summary = dedupe_summary(db, batch_id)
    if summary["remaining"] == 0:
        _rebuild_groups(db, batch_id)
        summary = dedupe_summary(db, batch_id)
    summary["processed"] = processed
    return summary


def reject_duplicates(db: Session, batch_id: str) -> dict:
    if not db.get(Batch, batch_id):
        raise ValueError("batch not found")
    images = db.scalars(
        select(ImageAsset)
        .join(ImageFingerprint, ImageFingerprint.image_asset_id == ImageAsset.id)
        .where(
            ImageAsset.batch_id == batch_id,
            ImageAsset.review_status.in_(FILTERABLE_REVIEW_STATUSES),
            ImageFingerprint.duplicate_group.is_not(None),
            ImageFingerprint.is_representative.is_(False),
        )
        .order_by(ImageAsset.id)
    ).all()
    changed = 0
    for image in images:
        before = {"review_status": image.review_status, "notes": image.notes}
        image.review_status = "rejected"
        note = "[近重复检测] 与同组代表图高度相似，已批量标记为不通过。"
        if note not in (image.notes or ""):
            image.notes = f"{image.notes or ''}\n{note}".strip()
        image.reviewed_by = "近重复检测"
        image.reviewed_at = utcnow()
        db.add(
            ReviewEvent(
                image_asset_id=image.id,
                action="dedupe_reject",
                reviewer="近重复检测",
                before_json=json.dumps(before, ensure_ascii=False),
                after_json=json.dumps({"review_status": image.review_status, "notes": image.notes}, ensure_ascii=False),
            )
        )
        changed += 1
    db.commit()
    return {"batch_id": batch_id, "rejected": changed, "summary": dedupe_summary(db, batch_id)}


def fingerprint_for_image(db: Session, image_asset_id: int) -> dict | None:
    row = db.scalar(select(ImageFingerprint).where(ImageFingerprint.image_asset_id == image_asset_id))
    if not row:
        return None
    return {
        "duplicate_group": row.duplicate_group,
        "is_representative": row.is_representative,
        "duplicate_kind": row.duplicate_kind,
        "distance": row.distance_to_representative,
    }


@router.get("/batches")
def api_dedupe_batches(db: Session = Depends(get_db)):
    ids = db.scalars(select(Batch.batch_id).order_by(Batch.created_at.desc())).all()
    return [dedupe_summary(db, batch_id) for batch_id in ids]


@router.get("/batch/{batch_id}")
def api_dedupe_batch(batch_id: str, db: Session = Depends(get_db)):
    try:
        return dedupe_summary(db, batch_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/scan")
def api_dedupe_scan(payload: DedupeScanRequest, db: Session = Depends(get_db)):
    try:
        return scan_batch(db, payload.batch_id, payload.limit, payload.rescan)
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/reject-duplicates")
def api_reject_duplicates(payload: DedupeFilterRequest, db: Session = Depends(get_db)):
    try:
        return reject_duplicates(db, payload.batch_id)
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/groups/{batch_id}")
def api_duplicate_groups(batch_id: str, db: Session = Depends(get_db)):
    rows = db.execute(
        select(ImageAsset, ImageFingerprint)
        .join(ImageFingerprint, ImageFingerprint.image_asset_id == ImageAsset.id)
        .where(ImageAsset.batch_id == batch_id, ImageFingerprint.duplicate_group.is_not(None))
        .order_by(ImageFingerprint.duplicate_group, ImageFingerprint.is_representative.desc(), ImageAsset.id)
    ).all()
    groups: dict[str, list[dict]] = {}
    for image, fp in rows:
        groups.setdefault(fp.duplicate_group, []).append(
            {
                "image_id": image.image_id,
                "media_url": f"/media/{image.batch_id}/{image.image_id}",
                "is_representative": fp.is_representative,
                "kind": fp.duplicate_kind,
                "distance": fp.distance_to_representative,
                "review_status": image.review_status,
            }
        )
    return [{"group_id": group_id, "items": items} for group_id, items in groups.items()]
