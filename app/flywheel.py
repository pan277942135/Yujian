from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter
from datetime import datetime, timezone
from dataclasses import dataclass

from google.cloud import storage
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.dedupe import ImageFingerprint
from app.factory import get_bucket_name
from app.models import DatasetVersion, FeedbackEvent, ImageAsset, SpeciesCatalog
from app.presence import FishPresenceResult, effective_status

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
DATASET_SPLITS = ("train", "val", "test")
ELIGIBLE_PRESENCE_STATUSES = {"single_fish", "uncertain", "not_scanned"}
MIN_SAMPLES_PER_SPECIES_WARNING = 5


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
            create_species_candidate(db, common_name_zh=corrected, notes=f"Auto-created from feedback event {event_id}")
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

    approved_total = db.scalar(select(func.count()).select_from(ImageAsset).where(ImageAsset.review_status == "approved")) or 0
    new_approved_stmt = select(func.count()).select_from(ImageAsset).where(ImageAsset.review_status == "approved")
    if cutoff:
        new_approved_stmt = new_approved_stmt.where(ImageAsset.updated_at > cutoff)
    new_approved = db.scalar(new_approved_stmt) or 0

    active_species = db.scalar(select(func.count()).select_from(SpeciesCatalog).where(SpeciesCatalog.status == "active")) or 0
    candidate_species = db.scalar(select(func.count()).select_from(SpeciesCatalog).where(SpeciesCatalog.status == "candidate")) or 0
    feedback_new = db.scalar(select(func.count()).select_from(FeedbackEvent).where(FeedbackEvent.pipeline_status == "NEW")) or 0

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
        raise ValueError(f"invalid parent class map URI: {uri}")
    bucket_name, object_name = body.split("/", 1)
    if bucket_name != bucket.name:
        raise ValueError(f"parent class map is in a different bucket: {uri}")
    blob = bucket.blob(object_name)
    if not blob.exists():
        raise ValueError(f"parent class map not found: {uri}")
    doc = json.loads(blob.download_as_text(encoding="utf-8"))
    return list(doc.get("classes") or [])


@dataclass(frozen=True)
class DatasetFreezePlan:
    """The complete immutable snapshot plan shared by preview and freeze."""

    dataset_version: str
    parent_version: str | None
    git_commit: str
    seed: int
    train: float
    val: float
    created_at: datetime
    approved_master_pool_count: int
    rows: tuple[dict, ...]
    class_map: dict
    species_counts: dict[str, int]
    split_counts: dict[str, int]
    excluded: dict[str, int]
    excluded_species_counts: dict[str, int]
    warnings: tuple[dict, ...]


def _validate_freeze_inputs(dataset_version: str, train: float, val: float) -> None:
    if not dataset_version.startswith("DS_"):
        raise ValueError("dataset_version must start with DS_")
    if not (0 < train < 1 and 0 <= val < 1 and train + val < 1):
        raise ValueError("invalid split ratios")


def _resolve_parent(db: Session, parent_version: str | None) -> tuple[DatasetVersion | None, str | None]:
    if parent_version:
        parent = db.get(DatasetVersion, parent_version)
        if not parent:
            raise ValueError(f"parent dataset not found: {parent_version}")
        return parent, parent_version
    parent = db.scalar(select(DatasetVersion).order_by(DatasetVersion.created_at.desc()).limit(1))
    return parent, parent.dataset_version if parent else None


def _catalog_class_row(row: SpeciesCatalog, class_index: int) -> dict:
    return {
        "class_index": class_index,
        "species_key": row.species_key,
        "common_name_zh": row.common_name_zh,
        "common_name_en": row.common_name_en,
        "catalog_order": row.catalog_order,
        "status": row.status,
    }


def _build_class_map(
    catalog_rows: list[SpeciesCatalog],
    used_species_keys: set[str],
    parent_classes: list[dict],
    *,
    dataset_version: str,
    parent_version: str | None,
) -> dict:
    """Preserve every parent index and append newly used active species."""

    catalog_by_key = {row.species_key: row for row in catalog_rows}
    classes: list[dict] = []
    seen_keys: set[str] = set()
    seen_indices: set[int] = set()

    for item in sorted(parent_classes, key=lambda x: int(x.get("class_index", 0))):
        key = str(item.get("species_key") or "").strip()
        if not key or key in seen_keys:
            continue
        try:
            class_index = int(item.get("class_index"))
        except (TypeError, ValueError):
            class_index = max(seen_indices, default=-1) + 1
        if class_index in seen_indices:
            class_index = max(seen_indices, default=-1) + 1
        catalog = catalog_by_key.get(key)
        if catalog:
            classes.append(_catalog_class_row(catalog, class_index))
        else:
            # Keep a historical class even if its catalog row was retired or migrated.
            preserved = dict(item)
            preserved["class_index"] = class_index
            preserved["species_key"] = key
            classes.append(preserved)
        seen_keys.add(key)
        seen_indices.add(class_index)

    next_index = max(seen_indices, default=-1) + 1
    for row in catalog_rows:
        if row.status != "active" or row.species_key not in used_species_keys or row.species_key in seen_keys:
            continue
        classes.append(_catalog_class_row(row, next_index))
        seen_keys.add(row.species_key)
        seen_indices.add(next_index)
        next_index += 1

    classes.sort(key=lambda x: int(x.get("class_index", 0)))
    return {
        "dataset_version": dataset_version,
        "parent_version": parent_version,
        "classes": classes,
    }


def _dataset_group_key(image: ImageAsset, duplicate_group: str) -> str:
    group = (image.group_id or "").strip()
    if group:
        return f"group:{group}"
    if duplicate_group:
        return f"duplicate:{duplicate_group}"
    return f"image:{image.batch_id}:{image.image_id}"


def _empty_exclusions() -> dict[str, int]:
    return {
        "no_fish": 0,
        "multi_fish": 0,
        "near_duplicate": 0,
        "exact_duplicate": 0,
        "inactive_species": 0,
        "presence_error": 0,
    }


def _prepare_dataset_freeze(
    db: Session,
    *,
    dataset_version: str,
    git_commit: str,
    parent_version: str | None = None,
    seed: int = 20260826,
    train: float = 0.70,
    val: float = 0.15,
    bucket: storage.Bucket | None = None,
) -> DatasetFreezePlan:
    """Build the one canonical snapshot plan used by both API operations."""

    ensure_species_catalog(db)
    _validate_freeze_inputs(dataset_version, train, val)
    if db.get(DatasetVersion, dataset_version):
        raise ValueError(f"dataset already registered: {dataset_version}")

    parent, resolved_parent_version = _resolve_parent(db, parent_version)
    if parent and parent.class_map_uri and bucket is None:
        raise ValueError("a GCS bucket is required to read the parent class map")

    catalog_rows = db.scalars(select(SpeciesCatalog).order_by(SpeciesCatalog.catalog_order)).all()
    catalog_by_identity = {
        identity: row
        for row in catalog_rows
        for identity in (row.species_key, row.common_name_zh)
    }
    images = db.scalars(
        select(ImageAsset).where(ImageAsset.review_status == "approved").order_by(ImageAsset.batch_id, ImageAsset.id)
    ).all()
    image_ids = [image.id for image in images]
    fingerprints = {
        row.image_asset_id: row
        for row in db.scalars(select(ImageFingerprint).where(ImageFingerprint.image_asset_id.in_(image_ids))).all()
    } if image_ids else {}
    presences = {
        row.image_asset_id: row
        for row in db.scalars(select(FishPresenceResult).where(FishPresenceResult.image_asset_id.in_(image_ids))).all()
    } if image_ids else {}

    created_at = utcnow()
    excluded = _empty_exclusions()
    excluded_species: Counter[str] = Counter()
    seen: set[str] = set()
    eligible: list[tuple[ImageAsset, SpeciesCatalog, str, str]] = []
    for image in images:
        fingerprint = fingerprints.get(image.id)
        presence_status = effective_status(presences.get(image.id))
        if fingerprint and fingerprint.duplicate_group and not fingerprint.is_representative:
            excluded["near_duplicate"] += 1
            continue
        if presence_status == "multi_fish":
            excluded["multi_fish"] += 1
            continue
        if presence_status == "no_fish":
            excluded["no_fish"] += 1
            continue
        if presence_status not in ELIGIBLE_PRESENCE_STATUSES:
            excluded["presence_error"] += 1
            continue

        unique_key = (fingerprint.sha256 if fingerprint else "") or image.gcs_uri
        if unique_key in seen:
            excluded["exact_duplicate"] += 1
            continue
        seen.add(unique_key)

        truth_name = (image.truth_species or image.claimed_species or "").strip()
        catalog = catalog_by_identity.get(truth_name)
        if not catalog or catalog.status != "active":
            excluded["inactive_species"] += 1
            excluded_species[truth_name or "unknown"] += 1
            continue
        duplicate_group = fingerprint.duplicate_group if fingerprint and fingerprint.duplicate_group else ""
        eligible.append((image, catalog, presence_status, duplicate_group))

    used_species_keys = {catalog.species_key for _, catalog, _, _ in eligible}
    parent_classes = _load_parent_class_map(bucket, parent) if parent and parent.class_map_uri else []
    class_map = _build_class_map(
        catalog_rows,
        used_species_keys,
        parent_classes,
        dataset_version=dataset_version,
        parent_version=resolved_parent_version,
    )
    class_index = {item["species_key"]: item["class_index"] for item in class_map["classes"]}

    frozen: list[dict] = []
    for image, catalog, presence_status, duplicate_group in eligible:
        group_id = (image.group_id or "").strip()
        frozen.append({
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
            "presence_status": presence_status,
            "duplicate_group": duplicate_group,
            "group_id": group_id,
            "split": _choose_split(_dataset_group_key(image, duplicate_group), seed, train, val),
        })

    species_counts = Counter(row["species"] for row in frozen)
    split_counts = Counter(row["split"] for row in frozen)
    warnings = tuple(
        {
            "code": "few_samples",
            "species_key": item.get("species_key"),
            "species": item.get("common_name_zh") or item.get("species_key"),
            "sample_count": species_counts.get(item.get("common_name_zh"), 0),
            "threshold": MIN_SAMPLES_PER_SPECIES_WARNING,
            "message": f"{item.get('common_name_zh') or item.get('species_key')} 样本数较少，当前为 {species_counts.get(item.get('common_name_zh'), 0)} 张。",
        }
        for item in class_map["classes"]
        if species_counts.get(item.get("common_name_zh"), 0) < MIN_SAMPLES_PER_SPECIES_WARNING
    )
    return DatasetFreezePlan(
        dataset_version=dataset_version,
        parent_version=resolved_parent_version,
        git_commit=git_commit or "unknown",
        seed=seed,
        train=train,
        val=val,
        created_at=created_at,
        approved_master_pool_count=len(images),
        rows=tuple(frozen),
        class_map=class_map,
        species_counts=dict(species_counts),
        split_counts={name: split_counts.get(name, 0) for name in DATASET_SPLITS},
        excluded=excluded,
        excluded_species_counts=dict(excluded_species),
        warnings=warnings,
    )


def dataset_freeze_preview(plan: DatasetFreezePlan) -> dict:
    """Serialize a freeze plan for the preview API without writing GCS or DB."""

    distribution = [
        {
            "species_key": item.get("species_key"),
            "species": item.get("common_name_zh"),
            "count": plan.species_counts.get(item.get("common_name_zh"), 0),
        }
        for item in plan.class_map["classes"]
        if plan.species_counts.get(item.get("common_name_zh"), 0)
    ]
    return {
        "dataset_version": plan.dataset_version,
        "parent_version": plan.parent_version,
        "eligible_images": len(plan.rows),
        "image_count": len(plan.rows),
        "approved_master_pool_count": plan.approved_master_pool_count,
        "species_count": len(plan.class_map["classes"]),
        "species_distribution": distribution,
        "species_counts": plan.species_counts,
        "train_count": plan.split_counts["train"],
        "val_count": plan.split_counts["val"],
        "test_count": plan.split_counts["test"],
        "split_counts": plan.split_counts,
        "excluded": plan.excluded,
        "excluded_quality_counts": {
            key: plan.excluded[key]
            for key in ("no_fish", "multi_fish", "near_duplicate", "exact_duplicate", "presence_error")
        },
        "excluded_species_counts": plan.excluded_species_counts,
        "class_map": plan.class_map,
        "warnings": list(plan.warnings),
        "selection_mode": "ALL_APPROVED_ACTIVE_SPECIES",
        "train_ratio": plan.train,
        "val_ratio": plan.val,
        "test_ratio": round(1.0 - plan.train - plan.val, 6),
    }


def preview_cumulative_dataset(
    db: Session,
    *,
    dataset_version: str,
    git_commit: str = "unknown",
    parent_version: str | None = None,
    seed: int = 20260826,
    train: float = 0.70,
    val: float = 0.15,
    bucket_name: str | None = None,
) -> dict:
    """Build the canonical preview without writing production GCS or DB state."""

    ensure_species_catalog(db)
    parent, _ = _resolve_parent(db, parent_version)
    bucket = None
    if parent and parent.class_map_uri:
        bucket = storage.Client().bucket(bucket_name or get_bucket_name())
    plan = _prepare_dataset_freeze(
        db,
        dataset_version=dataset_version,
        parent_version=parent_version,
        git_commit=git_commit,
        seed=seed,
        train=train,
        val=val,
        bucket=bucket,
    )
    return dataset_freeze_preview(plan)


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
    """Freeze cumulative approved data with P0 quality guards.

    Approved Master Pool remains permanent. For the current single-fish classifier,
    explicit multi-fish/no-fish detections and non-representative near duplicates
    are excluded from the immutable Dataset snapshot.
    """
    ensure_species_catalog(db)
    if db.get(DatasetVersion, dataset_version):
        raise ValueError(f"dataset already registered: {dataset_version}")
    bucket_name = bucket_name or get_bucket_name()
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    out_prefix = f"datasets/{dataset_version}/"
    marker = bucket.blob(out_prefix + "dataset.json")
    if marker.exists(client):
        raise ValueError(f"dataset already exists in GCS: gs://{bucket_name}/{out_prefix}")

    plan = _prepare_dataset_freeze(
        db,
        dataset_version=dataset_version,
        parent_version=parent_version,
        git_commit=git_commit,
        seed=seed,
        train=train,
        val=val,
        bucket=bucket,
    )
    if not plan.rows:
        if not plan.approved_master_pool_count:
            raise ValueError("Approved Master Pool is empty")
        raise ValueError("no approved images remain after active-species and P0 quality filters")

    csv_buf = io.StringIO()
    writer = csv.DictWriter(csv_buf, fieldnames=list(plan.rows[0].keys()))
    writer.writeheader()
    writer.writerows(plan.rows)
    manifest_uri = f"gs://{bucket_name}/{out_prefix}dataset_manifest.csv"
    bucket.blob(out_prefix + "dataset_manifest.csv").upload_from_string(
        csv_buf.getvalue(), content_type="text/csv", if_generation_match=0
    )

    class_map_uri = f"gs://{bucket_name}/{out_prefix}class_map.json"
    bucket.blob(out_prefix + "class_map.json").upload_from_string(
        json.dumps(plan.class_map, ensure_ascii=False, indent=2),
        content_type="application/json",
        if_generation_match=0,
    )

    meta = {
        **dataset_freeze_preview(plan),
        "created_at": plan.created_at.isoformat(),
        "source_cutoff_at": plan.created_at.isoformat(),
        "git_commit": plan.git_commit,
        "seed": plan.seed,
        "manifest_uri": manifest_uri,
        "class_map_uri": class_map_uri,
        "immutable": True,
    }
    marker.upload_from_string(json.dumps(meta, ensure_ascii=False, indent=2), content_type="application/json", if_generation_match=0)

    db.add(DatasetVersion(
        dataset_version=dataset_version,
        parent_version=plan.parent_version,
        manifest_uri=manifest_uri,
        class_map_uri=class_map_uri,
        train_count=plan.split_counts["train"],
        val_count=plan.split_counts["val"],
        test_count=plan.split_counts["test"],
        species_count=len(plan.class_map["classes"]),
        git_commit=plan.git_commit,
        selection_mode="ALL_APPROVED_ACTIVE_SPECIES",
        source_cutoff_at=plan.created_at,
        status="FROZEN",
    ))
    db.commit()
    return meta
