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
from app.data_policy import UNCONFIRMED_TRUTH
from app.dedupe import ImageFingerprint
from app.freeze_policy import select_freeze_candidates
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
    """Preview the exact canonical selection used by formal Freeze."""
    _validate(payload)
    ensure_species_catalog(db)

    if payload.parent_version:
        parent = db.get(DatasetVersion, payload.parent_version)
        if not parent:
            raise ValueError(f"父版本不存在：{payload.parent_version}")
    else:
        parent = db.scalar(select(DatasetVersion).order_by(DatasetVersion.created_at.desc()).limit(1))
    parent_version = parent.dataset_version if parent else None

    # Preview must expose split blockers rather than raising so operators can repair
    # the Master Pool before formal Freeze.
    policy = select_freeze_candidates(
        db,
        seed=payload.seed,
        train=payload.train,
        val=payload.val,
        allow_split_blockers=True,
    )
    selected = policy["selected"]
    species_counts: Counter[str] = Counter(item["catalog"].common_name_zh for item in selected)
    split_counts: Counter[str] = Counter(item["split"] for item in selected)
    active_keys = sorted(row.species_key for row in policy["catalog_rows"] if row.status == "active")
    snapshot = {
        "dataset_version": payload.dataset_version,
        "parent_version": parent_version,
        "seed": payload.seed,
        "train": payload.train,
        "val": payload.val,
        "split_strategy": policy.get("split_strategy"),
        "active_species_keys": active_keys,
        "items": sorted(
            [
                [
                    item["image"].batch_id,
                    item["image"].image_id,
                    item["catalog"].species_key,
                    item["split"],
                ]
                for item in selected
            ]
        ),
    }
    selection_hash = hashlib.sha256(
        json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return {
        "dataset_version": payload.dataset_version,
        "parent_version": parent_version,
        "approved_master_pool_count": policy["approved_master_pool_count"],
        "image_count": len(selected),
        "species_count": len(species_counts),
        "species_counts": dict(species_counts),
        "split_counts": {name: split_counts.get(name, 0) for name in ("train", "val", "test")},
        "split_strategy": policy.get("split_strategy"),
        "split_group_count": policy.get("split_group_count", 0),
        "per_species_split_counts": policy.get("per_species_split_counts", {}),
        "split_warnings": policy.get("split_warnings", []),
        "split_blockers": policy.get("split_blockers", []),
        "freeze_ready": not bool(policy.get("split_blockers")),
        "excluded_quality_counts": dict(policy["excluded_quality"]),
        "excluded_species_counts": dict(policy["excluded_species"]),
        "selection_mode": "ALL_APPROVED_VERIFIED_TRUTH",
        "selection_hash": selection_hash,
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
    """Idempotently materialize immutable lineage strictly from the frozen manifest."""
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
        frozen_gcs_uri = (item.get("gcs_uri") or "").strip()
        species_key = (item.get("species_key") or "").strip()
        species_name = (item.get("species") or "").strip()
        split = (item.get("split") or "").strip()
        if not batch_id or not image_id or not frozen_gcs_uri or not species_key or not species_name or not split:
            missing += 1
            continue
        image = db.scalar(select(ImageAsset).where(ImageAsset.batch_id == batch_id, ImageAsset.image_id == image_id))
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
        existing.gcs_uri = frozen_gcs_uri
        existing.species_key = species_key
        existing.species_name = species_name
        existing.class_index = int(item.get("class_index") or 0)
        existing.split = split
        existing.presence_status = (item.get("presence_status") or "").strip() or None
        existing.duplicate_group = (item.get("duplicate_group") or "").strip() or None
        existing.group_id = (item.get("group_id") or "").strip() or None
        synced += 1

    if missing:
        raise ValueError(f"追溯物化失败：manifest 有 {missing} 行缺失必要字段或源 ImageAsset")
    db.flush()
    item_count = db.scalar(
        select(func.count()).select_from(DatasetItem).where(DatasetItem.dataset_version == dataset_version)
    ) or 0
    if item_count != len(rows):
        raise ValueError(f"追溯数量不一致：manifest={len(rows)}, dataset_items={item_count}")
    db.commit()

    prefix = object_name.rsplit("/", 1)[0] + "/"
    dataset_meta_blob = bucket.blob(prefix + "dataset.json")
    report_blob = bucket.blob(prefix + "freeze_report.json")
    if not dataset_meta_blob.exists(client):
        raise ValueError("dataset.json 缺失")
    if not report_blob.exists(client):
        report_blob.upload_from_string(
            dataset_meta_blob.download_as_text(encoding="utf-8"),
            content_type="application/json",
            if_generation_match=0,
        )

    return {
        "dataset_version": dataset_version,
        "status": dataset.status,
        "item_count": item_count,
        "synced": synced,
        "missing": 0,
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


@router.get("/{dataset_version}/audit")
def audit_dataset(dataset_version: str, db: Session = Depends(get_db)):
    dataset = db.get(DatasetVersion, dataset_version)
    if not dataset:
        raise HTTPException(status_code=404, detail="数据集版本不存在")
    bucket_name, object_name = _parse_gs_uri(dataset.manifest_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    if not blob.exists(client):
        raise HTTPException(status_code=400, detail="dataset manifest 不存在")
    rows = list(csv.DictReader(io.StringIO(blob.download_as_text(encoding="utf-8"))))

    if getattr(dataset, "pipeline_type", "WHOLE_IMAGE_V1") == "CROP_CLASSIFIER_V1":
        from trainer.crop_dataset_validator import validate_crop_rows

        def image_exists(uri: str) -> bool:
            try:
                source_bucket, source_object = _parse_gs_uri(uri)
                return bool(client.bucket(source_bucket).blob(source_object).exists(client))
            except Exception:
                return False

        validation = validate_crop_rows(
            rows,
            require_bbox=True,
            require_metadata=True,
            check_source_image=True,
            image_exists=image_exists,
        )
        return {
            "dataset_version": dataset_version,
            "passed": bool(validation.get("valid")),
            "checks": validation.get("checks", {}),
            "validation": validation,
            "lineage_item_count": None,
        }

    truth_empty = 0
    species_truth_mismatch = 0
    bad_presence = 0
    inactive_species = 0
    missing_source = 0
    current_nonrepresentative_duplicates = 0
    split_zero_coverage = 0
    active_names = {
        row.common_name_zh
        for row in db.scalars(select(SpeciesCatalog).where(SpeciesCatalog.status == "active")).all()
    }
    species_splits: dict[str, Counter[str]] = {}
    for item in rows:
        truth = (item.get("truth_species") or "").strip()
        species = (item.get("species") or "").strip()
        split = (item.get("split") or "").strip()
        if not truth:
            truth_empty += 1
        if species != truth:
            species_truth_mismatch += 1
        if species:
            species_splits.setdefault(species, Counter())[split] += 1
        if (item.get("presence_status") or "").strip() in {"no_fish", "multi_fish"}:
            bad_presence += 1
        if species not in active_names:
            inactive_species += 1
        image = db.scalar(
            select(ImageAsset).where(
                ImageAsset.batch_id == (item.get("batch_id") or "").strip(),
                ImageAsset.image_id == (item.get("image_id") or "").strip(),
            )
        )
        if not image:
            missing_source += 1
            continue
        fp = db.scalar(select(ImageFingerprint).where(ImageFingerprint.image_asset_id == image.id))
        if fp and fp.duplicate_group and not fp.is_representative:
            current_nonrepresentative_duplicates += 1

    for _species, counts in species_splits.items():
        if any(counts.get(split, 0) == 0 for split in ("train", "val", "test")):
            split_zero_coverage += 1

    lineage_item_count = db.scalar(
        select(func.count()).select_from(DatasetItem).where(DatasetItem.dataset_version == dataset_version)
    ) or 0
    checks = {
        "manifest_rows": len(rows),
        "lineage_item_count": lineage_item_count,
        "truth_species_empty": truth_empty,
        "species_truth_mismatch": species_truth_mismatch,
        "bad_presence": bad_presence,
        "inactive_species": inactive_species,
        "missing_source": missing_source,
        "current_nonrepresentative_duplicates": current_nonrepresentative_duplicates,
        # Advisory for historical datasets; new Freeze Gate prevents this for v0.3+.
        "species_with_zero_split_coverage": split_zero_coverage,
    }
    passed = (
        lineage_item_count == len(rows)
        and truth_empty == 0
        and species_truth_mismatch == 0
        and bad_presence == 0
        and inactive_species == 0
        and missing_source == 0
        and current_nonrepresentative_duplicates == 0
    )
    return {"dataset_version": dataset_version, "passed": passed, "checks": checks}


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
        "pipeline_type": getattr(dataset, "pipeline_type", "WHOLE_IMAGE_V1"),
        "metadata": json.loads(dataset.metadata_json) if getattr(dataset, "metadata_json", None) else None,
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
