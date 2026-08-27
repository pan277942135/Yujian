from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter

from fastapi import APIRouter, Depends, HTTPException, Query
from google.cloud import storage
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.dataset_models import DatasetItem
from app.db import get_db
from app.dedupe import ImageFingerprint
from app.factory import get_bucket_name
from app.flywheel import ensure_species_catalog
from app.models import DatasetVersion, ImageAsset, SpeciesCatalog
from app.presence import FishPresenceResult, effective_status

router = APIRouter(prefix="/api/dataset-freeze", tags=["dataset-freeze"])


class DatasetFreezePreviewRequest(BaseModel):
    dataset_version: str = Field(default="DS_M1_v0.1", min_length=4, max_length=128)
    parent_version: str | None = None
    seed: int = 20260826
    train: float = 0.70
    val: float = 0.15


def _validate(payload: DatasetFreezePreviewRequest) -> None:
    if not payload.dataset_version.startswith("DS_"):
        raise ValueError("数据集版本必须以 DS_ 开头")
    if not (0 < payload.train < 1 and 0 <= payload.val < 1 and payload.train + payload.val < 1):
        raise ValueError("训练集/验证集比例不合法")


def _stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _choose_split(key: str, seed: int, train: float, val: float) -> str:
    p = _stable_fraction(key, seed)
    if p < train:
        return "train"
    if p < train + val:
        return "val"
    return "test"


def build_preview(db: Session, payload: DatasetFreezePreviewRequest) -> dict:
    """Preview the same P0 quality policy used by freeze_cumulative_dataset, without GCS writes."""
    _validate(payload)
    ensure_species_catalog(db)

    if payload.parent_version and not db.get(DatasetVersion, payload.parent_version):
        raise ValueError(f"父版本不存在：{payload.parent_version}")

    catalog_rows = db.scalars(select(SpeciesCatalog).order_by(SpeciesCatalog.catalog_order)).all()
    active_by_name = {row.common_name_zh: row for row in catalog_rows if row.status == "active"}
    images = db.scalars(
        select(ImageAsset)
        .where(ImageAsset.review_status == "approved")
        .order_by(ImageAsset.batch_id, ImageAsset.id)
    ).all()

    if not images:
        return {
            "dataset_version": payload.dataset_version,
            "approved_master_pool_count": 0,
            "image_count": 0,
            "species_count": 0,
            "species_counts": {},
            "split_counts": {"train": 0, "val": 0, "test": 0},
            "excluded_quality_counts": {},
            "excluded_species_counts": {},
            "selection_mode": "ALL_APPROVED",
        }

    image_ids = [image.id for image in images]
    fingerprints = {
        row.image_asset_id: row
        for row in db.scalars(select(ImageFingerprint).where(ImageFingerprint.image_asset_id.in_(image_ids))).all()
    }
    presences = {
        row.image_asset_id: row
        for row in db.scalars(select(FishPresenceResult).where(FishPresenceResult.image_asset_id.in_(image_ids))).all()
    }

    seen: set[str] = set()
    species_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    excluded_quality: Counter[str] = Counter()
    excluded_species: Counter[str] = Counter()

    for image in images:
        fp = fingerprints.get(image.id)
        presence = presences.get(image.id)
        presence_status = effective_status(presence)

        if fp and fp.duplicate_group and not fp.is_representative:
            excluded_quality["near_duplicate"] += 1
            continue
        if presence_status == "multi_fish":
            excluded_quality["multi_fish"] += 1
            continue
        if presence_status == "no_fish":
            excluded_quality["no_fish"] += 1
            continue

        unique_key = fp.sha256 if fp else image.gcs_uri
        if unique_key in seen:
            excluded_quality["exact_duplicate"] += 1
            continue
        seen.add(unique_key)

        truth_name = (image.truth_species or image.claimed_species or "").strip()
        catalog = active_by_name.get(truth_name)
        if not catalog:
            excluded_species[truth_name or "unknown"] += 1
            continue

        duplicate_group = fp.duplicate_group if fp and fp.duplicate_group else ""
        group = image.group_id or duplicate_group or f"{image.batch_id}:{image.image_id}"
        split = _choose_split(group, payload.seed, payload.train, payload.val)
        species_counts[catalog.common_name_zh] += 1
        split_counts[split] += 1

    image_count = sum(split_counts.values())
    return {
        "dataset_version": payload.dataset_version,
        "approved_master_pool_count": len(images),
        "image_count": image_count,
        "species_count": len(species_counts),
        "species_counts": dict(species_counts),
        "split_counts": {name: split_counts.get(name, 0) for name in ("train", "val", "test")},
        "excluded_quality_counts": dict(excluded_quality),
        "excluded_species_counts": dict(excluded_species),
        "selection_mode": "ALL_APPROVED",
        "train_ratio": payload.train,
        "val_ratio": payload.val,
        "test_ratio": round(1.0 - payload.train - payload.val, 6),
    }


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"不是有效的 GCS URI：{uri}")
    body = uri[5:]
    if "/" not in body:
        raise ValueError(f"不是有效的 GCS URI：{uri}")
    return tuple(body.split("/", 1))  # type: ignore[return-value]


def finalize_dataset_lineage(db: Session, dataset_version: str) -> dict:
    """Idempotently materialize DatasetItem lineage and freeze_report.json after a freeze."""
    dataset = db.get(DatasetVersion, dataset_version)
    if not dataset:
        raise ValueError("数据集版本不存在，请先完成冻结")
    if dataset.status != "FROZEN":
        raise ValueError(f"数据集尚未冻结：{dataset.status}")

    bucket_name, object_name = _parse_gs_uri(dataset.manifest_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    manifest_blob = bucket.blob(object_name)
    if not manifest_blob.exists(client):
        raise ValueError(f"数据清单不存在：{dataset.manifest_uri}")

    manifest_text = manifest_blob.download_as_text(encoding="utf-8")
    rows = list(csv.DictReader(io.StringIO(manifest_text)))
    synced = 0
    missing = 0

    for item in rows:
        batch_id = (item.get("batch_id") or "").strip()
        image_id = (item.get("image_id") or "").strip()
        if not batch_id or not image_id:
            missing += 1
            continue
        image = db.scalar(
            select(ImageAsset).where(ImageAsset.batch_id == batch_id, ImageAsset.image_id == image_id)
        )
        if not image:
            missing += 1
            continue

        existing = db.scalar(
            select(DatasetItem).where(
                DatasetItem.dataset_version == dataset_version,
                DatasetItem.image_asset_id == image.id,
            )
        )
        if not existing:
            existing = DatasetItem(dataset_version=dataset_version, image_asset_id=image.id)
            db.add(existing)

        existing.batch_id = batch_id
        existing.image_id = image_id
        existing.gcs_uri = image.gcs_uri
        existing.species_key = (item.get("species_key") or "unknown").strip() or "unknown"
        existing.species_name = (item.get("species") or item.get("truth_species") or image.truth_species or image.claimed_species or "unknown").strip()
        existing.class_index = int(item.get("class_index") or 0)
        existing.split = (item.get("split") or "train").strip()
        existing.presence_status = (item.get("presence_status") or "").strip() or None
        existing.duplicate_group = (item.get("duplicate_group") or "").strip() or None
        existing.group_id = (item.get("group_id") or image.group_id or "").strip() or None
        synced += 1

    db.commit()

    prefix = object_name.rsplit("/", 1)[0] + "/"
    dataset_meta_blob = bucket.blob(prefix + "dataset.json")
    report_blob = bucket.blob(prefix + "freeze_report.json")
    if dataset_meta_blob.exists(client) and not report_blob.exists(client):
        report_blob.upload_from_string(
            dataset_meta_blob.download_as_text(encoding="utf-8"),
            content_type="application/json",
            if_generation_match=0,
        )

    return {
        "dataset_version": dataset_version,
        "status": dataset.status,
        "item_count": db.scalar(
            select(func.count()).select_from(DatasetItem).where(DatasetItem.dataset_version == dataset_version)
        ) or 0,
        "synced": synced,
        "missing": missing,
        "manifest_uri": dataset.manifest_uri,
        "class_map_uri": dataset.class_map_uri,
        "report_uri": f"gs://{bucket_name}/{prefix}freeze_report.json",
    }


@router.post("/preview")
def preview(payload: DatasetFreezePreviewRequest, db: Session = Depends(get_db)):
    try:
        return build_preview(db, payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{dataset_version}/finalize")
def finalize(dataset_version: str, db: Session = Depends(get_db)):
    try:
        return finalize_dataset_lineage(db, dataset_version)
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{dataset_version}")
def detail(dataset_version: str, db: Session = Depends(get_db)):
    dataset = db.get(DatasetVersion, dataset_version)
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集版本不存在")
    item_count = db.scalar(
        select(func.count()).select_from(DatasetItem).where(DatasetItem.dataset_version == dataset_version)
    ) or 0
    return {
        "dataset_version": dataset.dataset_version,
        "parent_version": dataset.parent_version,
        "status": dataset.status,
        "image_count": dataset.train_count + dataset.val_count + dataset.test_count,
        "train_count": dataset.train_count,
        "val_count": dataset.val_count,
        "test_count": dataset.test_count,
        "species_count": dataset.species_count,
        "manifest_uri": dataset.manifest_uri,
        "class_map_uri": dataset.class_map_uri,
        "lineage_item_count": item_count,
    }


@router.get("/{dataset_version}/items")
def items(
    dataset_version: str,
    split: str | None = None,
    species: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    stmt = select(DatasetItem).where(DatasetItem.dataset_version == dataset_version)
    if split:
        stmt = stmt.where(DatasetItem.split == split)
    if species:
        stmt = stmt.where(DatasetItem.species_name == species)
    rows = db.scalars(stmt.order_by(DatasetItem.id).offset(offset).limit(limit)).all()
    return [
        {
            "batch_id": row.batch_id,
            "image_id": row.image_id,
            "species": row.species_name,
            "species_key": row.species_key,
            "class_index": row.class_index,
            "split": row.split,
            "gcs_uri": row.gcs_uri,
            "presence_status": row.presence_status,
            "duplicate_group": row.duplicate_group,
            "group_id": row.group_id,
        }
        for row in rows
    ]
