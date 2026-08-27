from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter
from datetime import datetime, timezone

from google.cloud import storage
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.factory import get_bucket_name
from app.models import DatasetVersion, FeedbackEvent, ImageAsset, SpeciesCatalog

INITIAL_SPECIES = [
    {"species_key": "grass_carp", "catalog_order": 0, "common_name_zh": "草鱼", "common_name_en": "Grass carp", "status": "active", "is_other": False},
    {"species_key": "bighead_carp", "catalog_order": 1, "common_name_zh": "鳙鱼", "common_name_en": "Bighead carp", "status": "active", "is_other": False},
    {"species_key": "silver_carp", "catalog_order": 2, "common_name_zh": "白鲢", "common_name_en": "Silver carp", "status": "active", "is_other": False},
    {"species_key": "common_carp", "catalog_order": 3, "common_name_zh": "鲤鱼", "common_name_en": "Common carp", "status": "active", "is_other": False},
    {"species_key": "crucian_carp", "catalog_order": 4, "common_name_zh": "鲫鱼", "common_name_en": "Crucian carp", "status": "active", "is_other": False},
    {"species_key": "largemouth_bass", "catalog_order": 5, "common_name_zh": "加州鲈", "common_name_en": "Largemouth bass", "status": "active", "is_other": False},
    {"species_key": "snakehead", "catalog_order": 6, "common_name_zh": "黑鱼", "common_name_en": "Snakehead", "status": "active", "is_other": False},
    {"species_key": "yellow_catfish", "catalog_order": 7, "common_name_zh": "黄骨鱼", "common_name_en": "Yellow catfish", "status": "active", "is_other": False},
    {"species_key": "black_carp", "catalog_order": 8, "common_name_zh": "青鱼", "common_name_en": "Black carp", "status": "active", "is_other": False},
    {"species_key": "other_freshwater_fish", "catalog_order": 9, "common_name_zh": "其他淡水鱼", "common_name_en": "Other freshwater fish", "status": "active", "is_other": True},
]

VALID_SPECIES_STATUS = {"candidate", "active", "retired"}
VALID_FEEDBACK_TYPES = {"confirmed", "corrected", "unknown", "new_species_candidate"}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def ensure_species_catalog(db: Session) -> None:
    existing = {row.species_key: row for row in db.scalars(select(SpeciesCatalog)).all()}
    used_orders = {row.catalog_order for row in existing.values()}
    next_order = max(used_orders, default=-1) + 1
    changed = False
    for seed in INITIAL_SPECIES:
        if seed["species_key"] in existing:
            continue
        order = seed["catalog_order"]
        if order in used_orders:
            order = next_order
            next_order += 1
        used_orders.add(order)
        db.add(SpeciesCatalog(**{**seed, "catalog_order": order}))
        changed = True
    if changed:
        db.commit()


def species_dict(row: SpeciesCatalog) -> dict:
    return {
        "species_key": row.species_key,
        "catalog_order": row.catalog_order,
        "common_name_zh": row.common_name_zh,
        "common_name_en": row.common_name_en,
        "scientific_name": row.scientific_name,
        "status": row.status,
        "is_other": row.is_other,
        "notes": row.notes,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def list_species(db: Session, status: str | None = None) -> list[dict]:
    ensure_species_catalog(db)
    stmt = select(SpeciesCatalog)
    if status:
        stmt = stmt.where(SpeciesCatalog.status == status)
    rows = db.scalars(stmt.order_by(SpeciesCatalog.catalog_order)).all()
    return [species_dict(row) for row in rows]


def species_names(db: Session, include_candidates: bool = True) -> list[str]:
    ensure_species_catalog(db)
    statuses = ["active", "candidate"] if include_candidates else ["active"]
    rows = db.scalars(
        select(SpeciesCatalog).where(SpeciesCatalog.status.in_(statuses)).order_by(SpeciesCatalog.catalog_order)
    ).all()
    return [row.common_name_zh for row in rows]


def _candidate_key(common_name_zh: str) -> str:
    digest = hashlib.sha1(common_name_zh.strip().encode("utf-8")).hexdigest()[:12]
    return f"species_{digest}"


def create_species_candidate(
    db: Session,
    common_name_zh: str,
    species_key: str | None = None,
    common_name_en: str | None = None,
    scientific_name: str | None = None,
    notes: str | None = None,
) -> dict:
    ensure_species_catalog(db)
    name = common_name_zh.strip()
    if not name:
        raise ValueError("common_name_zh is required")
    existing = db.scalar(select(SpeciesCatalog).where(SpeciesCatalog.common_name_zh == name))
    if existing:
        return species_dict(existing)
    key = (species_key or _candidate_key(name)).strip()
    if db.get(SpeciesCatalog, key):
        raise ValueError(f"species_key already exists: {key}")
    max_order = db.scalar(select(func.max(SpeciesCatalog.catalog_order)))
    row = SpeciesCatalog(
        species_key=key,
        catalog_order=(max_order if max_order is not None else -1) + 1,
        common_name_zh=name,
        common_name_en=(common_name_en or "").strip() or None,
        scientific_name=(scientific_name or "").strip() or None,
        status="candidate",
        is_other=False,
        notes=(notes or "").strip() or None,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return species_dict(row)


def set_species_status(db: Session, species_key: str, status: str) -> dict:
    if status not in VALID_SPECIES_STATUS:
        raise ValueError(f"invalid species status: {status}")
    row = db.get(SpeciesCatalog, species_key)
    if not row:
        raise ValueError("species not found")
    row.status = status
    row.updated_at = utcnow()
    db.commit()
    db.refresh(row)
    return species_dict(row)


def record_feedback(
    db: Session,
    *,
    source_event_id: str,
    feedback_type: str,
    source: str = "app",
    image_gcs_uri: str | None = None,
    model_version: str | None = None,
    predicted_species: str | None = None,
    confidence: float | None = None,
    corrected_species: str | None = None,
    user_note: str | None = None,
) -> dict:
    ensure_species_catalog(db)
    if feedback_type not in VALID_FEEDBACK_TYPES:
        raise ValueError(f"invalid feedback_type: {feedback_type}")
    event_id = source_event_id.strip()
    if not event_id:
        raise ValueError("source_event_id is required")
    existing = db.scalar(select(FeedbackEvent).where(FeedbackEvent.source_event_id == event_id))
    if existing:
        return feedback_dict(existing)

    corrected = (corrected_species or "").strip() or None
    if corrected:
        known = db.scalar(select(SpeciesCatalog).where(SpeciesCatalog.common_name_zh == corrected))
        if not known:
            create_species_candidate(
                db,
                common_name_zh=corrected,
                notes=f"Auto-created from feedback event {event_id}",
            )
            if feedback_type == "corrected":
                feedback_type = "new_species_candidate"

    row = FeedbackEvent(
        source_event_id=event_id,
        source=(source or "app").strip() or "app",
        image_gcs_uri=(image_gcs_uri or "").strip() or None,
        model_version=(model_version or "").strip() or None,
        predicted_species=(predicted_species or "").strip() or None,
        confidence=confidence,
        feedback_type=feedback_type,
        corrected_species=corrected,
        user_note=(user_note or "").strip() or None,
        pipeline_status="NEW",
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return feedback_dict(row)


def feedback_dict(row: FeedbackEvent) -> dict:
    return {
        "id": row.id,
        "source_event_id": row.source_event_id,
        "source": row.source,
        "image_gcs_uri": row.image_gcs_uri,
        "model_version": row.model_version,
        "predicted_species": row.predicted_species,
        "confidence": row.confidence,
        "feedback_type": row.feedback_type,
        "corrected_species": row.corrected_species,
        "user_note": row.user_note,
        "pipeline_status": row.pipeline_status,
        "materialized_batch_id": row.materialized_batch_id,
        "materialized_image_id": row.materialized_image_id,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def list_feedback(db: Session, status: str | None = None, limit: int = 100) -> list[dict]:
    stmt = select(FeedbackEvent)
    if status:
        stmt = stmt.where(FeedbackEvent.pipeline_status == status)
    rows = db.scalars(stmt.order_by(FeedbackEvent.created_at.desc()).limit(limit)).all()
    return [feedback_dict(row) for row in rows]


def flywheel_summary(db: Session) -> dict:
    ensure_species_catalog(db)
    latest = db.scalar(select(DatasetVersion).order_by(DatasetVersion.created_at.desc()).limit(1))
    cutoff = latest.source_cutoff_at if latest else None

    approved_total = db.scalar(
        select(func.count()).select_from(ImageAsset).where(ImageAsset.review_status == "approved")
    ) or 0
    new_approved_stmt = select(func.count()).select_from(ImageAsset).where(ImageAsset.review_status == "approved")
    if cutoff:
        new_approved_stmt = new_approved_stmt.where(ImageAsset.updated_at > cutoff)
    new_approved = db.scalar(new_approved_stmt) or 0

    active_species = db.scalar(
        select(func.count()).select_from(SpeciesCatalog).where(SpeciesCatalog.status == "active")
    ) or 0
    candidate_species = db.scalar(
        select(func.count()).select_from(SpeciesCatalog).where(SpeciesCatalog.status == "candidate")
    ) or 0
    feedback_new = db.scalar(
        select(func.count()).select_from(FeedbackEvent).where(FeedbackEvent.pipeline_status == "NEW")
    ) or 0

    species_key = func.coalesce(ImageAsset.truth_species, ImageAsset.claimed_species, "unknown")
    distribution = db.execute(
        select(species_key, func.count())
        .where(ImageAsset.review_status == "approved")
        .group_by(species_key)
        .order_by(func.count().desc())
    ).all()
    return {
        "approved_master_pool": approved_total,
        "new_approved_since_latest_dataset": new_approved,
        "active_species": active_species,
        "candidate_species": candidate_species,
        "new_feedback": feedback_new,
        "latest_dataset": latest.dataset_version if latest else None,
        "approved_species": [{"species": name, "count": count} for name, count in distribution],
    }


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


def _load_parent_class_map(bucket: storage.Bucket, parent: DatasetVersion | None) -> list[dict]:
    if not parent or not parent.class_map_uri:
        return []
    uri = parent.class_map_uri
    body = uri[5:] if uri.startswith("gs://") else ""
    if not body or "/" not in body:
        return []
    bucket_name, object_name = body.split("/", 1)
    if bucket_name != bucket.name:
        return []
    blob = bucket.blob(object_name)
    if not blob.exists():
        return []
    doc = json.loads(blob.download_as_text(encoding="utf-8"))
    return list(doc.get("classes") or [])


def freeze_cumulative_dataset(
    db: Session,
    *,
    dataset_version: str,
    git_commit: str,
    parent_version: str | None = None,
    seed: int = 20260826,
    train: float = 0.70,
    val: float = 0.15,
    bucket_name: str | None = None,
) -> dict:
    """Freeze the complete Approved Master Pool as an immutable model-training snapshot.

    Canonical datasets are cumulative: every currently-approved image from every
    batch is considered. Candidate/retired species stay in the master pool but are
    excluded from model training until activated in Species Catalog.
    """
    ensure_species_catalog(db)
    bucket_name = bucket_name or get_bucket_name()
    if not dataset_version.startswith("DS_"):
        raise ValueError("dataset_version must start with DS_")
    if not (0 < train < 1 and 0 <= val < 1 and train + val < 1):
        raise ValueError("invalid split ratios")
    if db.get(DatasetVersion, dataset_version):
        raise ValueError(f"dataset already registered: {dataset_version}")

    if parent_version:
        parent = db.get(DatasetVersion, parent_version)
        if not parent:
            raise ValueError(f"parent dataset not found: {parent_version}")
    else:
        parent = db.scalar(select(DatasetVersion).order_by(DatasetVersion.created_at.desc()).limit(1))
        parent_version = parent.dataset_version if parent else None

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    out_prefix = f"datasets/{dataset_version}/"
    marker = bucket.blob(out_prefix + "dataset.json")
    if marker.exists(client):
        raise ValueError(f"dataset already exists in GCS: gs://{bucket_name}/{out_prefix}")

    catalog_rows = db.scalars(select(SpeciesCatalog).order_by(SpeciesCatalog.catalog_order)).all()
    active_by_name = {row.common_name_zh: row for row in catalog_rows if row.status == "active"}
    images = db.scalars(
        select(ImageAsset).where(ImageAsset.review_status == "approved").order_by(ImageAsset.batch_id, ImageAsset.id)
    ).all()
    if not images:
        raise ValueError("Approved Master Pool is empty")

    cutoff = utcnow()
    seen: set[str] = set()
    eligible: list[tuple[ImageAsset, SpeciesCatalog]] = []
    excluded_species = Counter()
    for image in images:
        unique_key = image.gcs_uri
        if unique_key in seen:
            continue
        seen.add(unique_key)
        truth_name = (image.truth_species or image.claimed_species or "").strip()
        catalog = active_by_name.get(truth_name)
        if not catalog:
            excluded_species[truth_name or "unknown"] += 1
            continue
        eligible.append((image, catalog))
    if not eligible:
        raise ValueError("no approved images belong to active species")

    used_keys = {catalog.species_key for _, catalog in eligible}
    parent_classes = _load_parent_class_map(bucket, parent)
    class_rows: list[dict] = []
    used_parent_keys: set[str] = set()
    for item in sorted(parent_classes, key=lambda x: int(x.get("class_index", 0))):
        key = item.get("species_key")
        if not key:
            continue
        row = next((r for r in catalog_rows if r.species_key == key), None)
        if not row:
            continue
        class_rows.append(
            {
                "class_index": len(class_rows),
                "species_key": row.species_key,
                "common_name_zh": row.common_name_zh,
                "common_name_en": row.common_name_en,
                "catalog_order": row.catalog_order,
                "status": row.status,
            }
        )
        used_parent_keys.add(key)

    for row in catalog_rows:
        if row.species_key not in used_keys or row.species_key in used_parent_keys:
            continue
        class_rows.append(
            {
                "class_index": len(class_rows),
                "species_key": row.species_key,
                "common_name_zh": row.common_name_zh,
                "common_name_en": row.common_name_en,
                "catalog_order": row.catalog_order,
                "status": row.status,
            }
        )
    class_index = {row["species_key"]: row["class_index"] for row in class_rows}

    frozen: list[dict] = []
    for image, catalog in eligible:
        group = image.group_id or f"{image.batch_id}:{image.image_id}"
        frozen.append(
            {
                "dataset_version": dataset_version,
                "batch_id": image.batch_id,
                "image_id": image.image_id,
                "file_name": image.file_name,
                "gcs_uri": image.gcs_uri,
                "object_name": image.object_name,
                "source_url": image.source_url or "",
                "source_platform": image.source_platform or "",
                "claimed_species": image.claimed_species or "",
                "truth_species": image.truth_species or "",
                "species_key": catalog.species_key,
                "species": catalog.common_name_zh,
                "class_index": class_index[catalog.species_key],
                "truth_status": image.truth_status,
                "review_status": image.review_status,
                "group_id": image.group_id or "",
                "split": _choose_split(group, seed, train, val),
            }
        )

    csv_buf = io.StringIO()
    writer = csv.DictWriter(csv_buf, fieldnames=list(frozen[0].keys()))
    writer.writeheader()
    writer.writerows(frozen)
    manifest_uri = f"gs://{bucket_name}/{out_prefix}dataset_manifest.csv"
    bucket.blob(out_prefix + "dataset_manifest.csv").upload_from_string(
        csv_buf.getvalue(), content_type="text/csv", if_generation_match=0
    )

    class_map_doc = {
        "dataset_version": dataset_version,
        "parent_version": parent_version,
        "created_at": cutoff.isoformat(),
        "classes": class_rows,
    }
    class_map_uri = f"gs://{bucket_name}/{out_prefix}class_map.json"
    bucket.blob(out_prefix + "class_map.json").upload_from_string(
        json.dumps(class_map_doc, ensure_ascii=False, indent=2),
        content_type="application/json",
        if_generation_match=0,
    )

    split_counts = Counter(row["split"] for row in frozen)
    species_counts = Counter(row["species"] for row in frozen)
    meta = {
        "dataset_version": dataset_version,
        "parent_version": parent_version,
        "created_at": cutoff.isoformat(),
        "source_cutoff_at": cutoff.isoformat(),
        "selection_mode": "ALL_APPROVED",
        "git_commit": git_commit or "unknown",
        "seed": seed,
        "approved_master_pool_count": len(images),
        "image_count": len(frozen),
        "excluded_non_active_species_count": sum(excluded_species.values()),
        "excluded_species_counts": dict(excluded_species),
        "split_counts": dict(split_counts),
        "species_counts": dict(species_counts),
        "species_count": len(class_rows),
        "manifest_uri": manifest_uri,
        "class_map_uri": class_map_uri,
        "immutable": True,
    }
    marker.upload_from_string(
        json.dumps(meta, ensure_ascii=False, indent=2), content_type="application/json", if_generation_match=0
    )

    db.add(
        DatasetVersion(
            dataset_version=dataset_version,
            parent_version=parent_version,
            manifest_uri=manifest_uri,
            class_map_uri=class_map_uri,
            train_count=split_counts.get("train", 0),
            val_count=split_counts.get("val", 0),
            test_count=split_counts.get("test", 0),
            species_count=len(class_rows),
            git_commit=git_commit or "unknown",
            selection_mode="ALL_APPROVED",
            source_cutoff_at=cutoff,
            status="FROZEN",
        )
    )
    db.commit()
    return meta
