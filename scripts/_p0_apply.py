#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    (ROOT / path).write_text(text, encoding="utf-8")
    print("patched", path)


def replace_once(path: str, old: str, new: str) -> None:
    text = read(path)
    if old not in text:
        raise RuntimeError(f"target not found in {path}: {old[:160]!r}")
    write(path, text.replace(old, new, 1))


def replace_between(path: str, start: str, end: str, new: str) -> None:
    text = read(path)
    i = text.find(start)
    if i < 0:
        raise RuntimeError(f"start marker not found in {path}: {start!r}")
    j = text.find(end, i + len(start))
    if j < 0:
        raise RuntimeError(f"end marker not found in {path}: {end!r}")
    write(path, text[:i] + new + text[j:])


def replace_from(path: str, start: str, new: str) -> None:
    text = read(path)
    i = text.find(start)
    if i < 0:
        raise RuntimeError(f"start marker not found in {path}: {start!r}")
    write(path, text[:i] + new)


# ---------------------------------------------------------------------------
# Shared data policy: claimed != truth; human review may override machine QA.
# ---------------------------------------------------------------------------
write(
    "app/data_policy.py",
    '''from __future__ import annotations

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from app.models import FeedbackEvent, ImageAsset, SpeciesCatalog

UNCONFIRMED_TRUTH = "未确认真实鱼种"
AUTO_REVIEWERS = {"鱼体检测", "近重复检测"}


def normalized_truth(image: ImageAsset) -> str:
    return (image.truth_species or "").strip()


def normalized_claimed(image: ImageAsset) -> str:
    return (image.claimed_species or "").strip()


def truth_sql_expr():
    return func.nullif(func.trim(ImageAsset.truth_species), "")


def claimed_sql_expr():
    return func.nullif(func.trim(ImageAsset.claimed_species), "")


def truth_filter_clause(species: str):
    truth = truth_sql_expr()
    if species == UNCONFIRMED_TRUTH:
        return truth.is_(None)
    return truth == species


def review_group_name(image: ImageAsset) -> str:
    return normalized_truth(image) or normalized_claimed(image) or "未标注"


def review_group_clause(species: str):
    truth = truth_sql_expr()
    claimed = claimed_sql_expr()
    if species == "未标注":
        return and_(truth.is_(None), claimed.is_(None))
    return or_(truth == species, and_(truth.is_(None), claimed == species))


def truth_distribution(db: Session, *, review_status: str | None = None) -> tuple[list[tuple[str, int]], int]:
    truth = truth_sql_expr()
    stmt = select(truth, func.count()).where(truth.is_not(None))
    unconfirmed_stmt = select(func.count()).select_from(ImageAsset).where(truth.is_(None))
    if review_status:
        stmt = stmt.where(ImageAsset.review_status == review_status)
        unconfirmed_stmt = unconfirmed_stmt.where(ImageAsset.review_status == review_status)
    rows = db.execute(stmt.group_by(truth).order_by(func.count().desc())).all()
    unconfirmed = db.scalar(unconfirmed_stmt) or 0
    return [(str(name), int(count)) for name, count in rows], int(unconfirmed)


def valid_truth_for_image(db: Session, image: ImageAsset, proposed: str) -> bool:
    proposed = proposed.strip()
    if not proposed:
        return True
    # Historical retired truth may be preserved, but retired species cannot be newly assigned.
    if proposed == normalized_truth(image):
        return True
    row = db.scalar(select(SpeciesCatalog).where(SpeciesCatalog.common_name_zh == proposed))
    return bool(row and row.status in {"active", "candidate"})


def human_approval_overrides(image: ImageAsset, machine_updated_at) -> bool:
    if image.review_status != "approved" or not image.reviewed_at or not machine_updated_at:
        return False
    if image.reviewed_by in AUTO_REVIEWERS:
        return False
    return image.reviewed_at >= machine_updated_at


def mark_feedback_reviewed(db: Session, image: ImageAsset) -> None:
    if image.review_status not in {"approved", "rejected"}:
        return
    row = db.scalar(
        select(FeedbackEvent).where(
            FeedbackEvent.materialized_batch_id == image.batch_id,
            FeedbackEvent.materialized_image_id == image.image_id,
        )
    )
    if row and row.pipeline_status == "BATCHED":
        row.pipeline_status = "REVIEWED"
''',
)

# ---------------------------------------------------------------------------
# Canonical Freeze selection shared by Preview and Freeze.
# ---------------------------------------------------------------------------
write(
    "app/freeze_policy.py",
    '''from __future__ import annotations

import hashlib
from collections import Counter

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.data_policy import UNCONFIRMED_TRUTH, human_approval_overrides, normalized_truth
from app.dedupe import ImageFingerprint
from app.models import ImageAsset, SpeciesCatalog
from app.presence import FishPresenceResult, effective_status


def stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def choose_split(key: str, seed: int, train: float, val: float) -> str:
    p = stable_fraction(key, seed)
    if p < train:
        return "train"
    if p < train + val:
        return "val"
    return "test"


def select_freeze_candidates(db: Session, *, seed: int, train: float, val: float) -> dict:
    catalog_rows = db.scalars(select(SpeciesCatalog).order_by(SpeciesCatalog.catalog_order)).all()
    active_by_name = {row.common_name_zh: row for row in catalog_rows if row.status == "active"}
    images = db.scalars(
        select(ImageAsset)
        .where(ImageAsset.review_status == "approved")
        .order_by(ImageAsset.batch_id, ImageAsset.id)
    ).all()
    if not images:
        return {
            "approved_master_pool_count": 0,
            "selected": [],
            "catalog_rows": catalog_rows,
            "excluded_quality": Counter(),
            "excluded_species": Counter(),
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

    selected = []
    seen: set[str] = set()
    excluded_quality: Counter[str] = Counter()
    excluded_species: Counter[str] = Counter()

    for image in images:
        fp = fingerprints.get(image.id)
        presence = presences.get(image.id)

        # A Dataset Freeze is only allowed after both automated QA stages ran.
        if fp is None:
            excluded_quality["dedupe_not_scanned"] += 1
            continue
        if presence is None:
            excluded_quality["presence_not_scanned"] += 1
            continue

        presence_status = effective_status(presence)
        if fp.duplicate_group and not fp.is_representative and not human_approval_overrides(image, fp.updated_at):
            excluded_quality["exact_duplicate" if fp.duplicate_kind == "exact" else "near_duplicate"] += 1
            continue
        if presence_status == "multi_fish" and not human_approval_overrides(image, presence.updated_at):
            excluded_quality["multi_fish"] += 1
            continue
        if presence_status == "no_fish" and not human_approval_overrides(image, presence.updated_at):
            excluded_quality["no_fish"] += 1
            continue

        unique_key = fp.sha256 or image.gcs_uri
        if unique_key in seen:
            excluded_quality["exact_duplicate"] += 1
            continue
        seen.add(unique_key)

        truth_name = normalized_truth(image)
        if not truth_name:
            excluded_species[UNCONFIRMED_TRUTH] += 1
            continue
        catalog = active_by_name.get(truth_name)
        if not catalog:
            excluded_species[truth_name] += 1
            continue

        duplicate_group = fp.duplicate_group or ""
        group = image.group_id or duplicate_group or f"{image.batch_id}:{image.image_id}"
        selected.append(
            {
                "image": image,
                "catalog": catalog,
                "presence_status": presence_status,
                "duplicate_group": duplicate_group,
                "split": choose_split(group, seed, train, val),
            }
        )

    return {
        "approved_master_pool_count": len(images),
        "selected": selected,
        "catalog_rows": catalog_rows,
        "excluded_quality": excluded_quality,
        "excluded_species": excluded_species,
    }
''',
)

# ---------------------------------------------------------------------------
# main.py: truth-only filters/stats, approval invariant, preview hash gate.
# ---------------------------------------------------------------------------
replace_once(
    "app/main.py",
    "from app.models import Batch, DatasetVersion, ImageAsset, ReviewEvent\n",
    "from app.data_policy import (\n    UNCONFIRMED_TRUTH,\n    mark_feedback_reviewed,\n    normalized_truth,\n    truth_distribution,\n    truth_filter_clause,\n    valid_truth_for_image,\n)\nfrom app.models import Batch, DatasetVersion, ImageAsset, ReviewEvent\n",
)
replace_once(
    "app/main.py",
    '''class DatasetFreeze(BaseModel):
    dataset_version: str
    parent_version: str | None = None
    git_commit: str = Field(default_factory=lambda: os.getenv("APP_GIT_COMMIT", "unknown"))
    seed: int = 20260826
    train: float = 0.70
    val: float = 0.15
''',
    '''class DatasetFreeze(BaseModel):
    dataset_version: str
    parent_version: str | None = None
    git_commit: str | None = None
    preview_hash: str | None = None
    seed: int = 20260826
    train: float = 0.70
    val: float = 0.15
''',
)
replace_between(
    "app/main.py",
    "def apply_review_filters(stmt, status=None, batch_id=None, species=None, q=None):\n",
    '\n\n@app.get("/health")',
    '''def apply_review_filters(stmt, status=None, batch_id=None, species=None, q=None):
    if status:
        stmt = stmt.where(ImageAsset.review_status == status)
    if batch_id:
        stmt = stmt.where(ImageAsset.batch_id == batch_id)
    if species:
        stmt = stmt.where(truth_filter_clause(species))
    if q:
        like = f"%{q}%"
        stmt = stmt.where(
            or_(
                ImageAsset.image_id.ilike(like),
                ImageAsset.file_name.ilike(like),
                ImageAsset.source_url.ilike(like),
            )
        )
    return stmt
''',
)
replace_between(
    "app/main.py",
    '@app.get("/api/overview")\n',
    '\n\n@app.get("/api/flywheel/summary")',
    '''@app.get("/api/overview")
def overview(db: Session = Depends(get_db)):
    total = db.scalar(select(func.count()).select_from(ImageAsset)) or 0
    status_rows = db.execute(select(ImageAsset.review_status, func.count()).group_by(ImageAsset.review_status)).all()
    status_counts = {status: count for status, count in status_rows}
    species_rows, unconfirmed_truth = truth_distribution(db)
    species = [{"species": name, "count": count} for name, count in species_rows]
    if unconfirmed_truth:
        species.append({"species": UNCONFIRMED_TRUTH, "count": unconfirmed_truth})
    result = {
        "total_images": total,
        "batch_count": db.scalar(select(func.count()).select_from(Batch)) or 0,
        "dataset_count": db.scalar(select(func.count()).select_from(DatasetVersion)) or 0,
        "review": {name: status_counts.get(name, 0) for name in REVIEW_VALUES},
        "species": species,
        "unconfirmed_truth_count": unconfirmed_truth,
    }
    result["flywheel"] = flywheel_summary(db)
    return result
''',
)
replace_once(
    "app/main.py",
    '''                "image_count": batch.image_count,
                "status": batch.status,
''',
    '''                "image_count": db.scalar(select(func.count()).select_from(ImageAsset).where(ImageAsset.batch_id == batch.batch_id)) or 0,
                "raw_image_count": batch.image_count,
                "status": batch.status,
''',
)
replace_between(
    "app/main.py",
    '@app.get("/api/review/stats")\n',
    '\n\n@app.patch("/api/review/{batch_id}/{image_id}")',
    '''@app.get("/api/review/stats")
def review_stats(
    status: str | None = Query(default=None),
    batch_id: str | None = None,
    species: str | None = None,
    q: str | None = None,
    db: Session = Depends(get_db),
):
    filtered = apply_review_filters(select(ImageAsset.id), status, batch_id, species, q).subquery()
    count = db.scalar(select(func.count()).select_from(filtered)) or 0
    status_stmt = apply_review_filters(
        select(ImageAsset.review_status, func.count()),
        None,
        batch_id,
        species,
        q,
    ).group_by(ImageAsset.review_status)
    all_status = db.execute(status_stmt).all()
    return {"filtered": count, "status": {key: value for key, value in all_status}}
''',
)
replace_between(
    "app/main.py",
    '@app.patch("/api/review/{batch_id}/{image_id}")\n',
    '\n\n@app.get("/media/{batch_id}/{image_id}")',
    '''@app.patch("/api/review/{batch_id}/{image_id}")
def update_review(batch_id: str, image_id: str, payload: ReviewUpdate, db: Session = Depends(get_db)):
    image = db.scalar(select(ImageAsset).where(ImageAsset.batch_id == batch_id, ImageAsset.image_id == image_id))
    if not image:
        raise HTTPException(status_code=404, detail="image not found")
    if payload.review_status is not None and payload.review_status not in REVIEW_VALUES:
        raise HTTPException(status_code=400, detail=f"invalid review_status: {payload.review_status}")
    if payload.truth_status is not None and payload.truth_status not in TRUTH_VALUES:
        raise HTTPException(status_code=400, detail=f"invalid truth_status: {payload.truth_status}")

    proposed_truth = normalized_truth(image)
    if "truth_species" in payload.model_fields_set:
        proposed_truth = (payload.truth_species or "").strip()
    if proposed_truth and not valid_truth_for_image(db, image, proposed_truth):
        raise HTTPException(status_code=400, detail="真实鱼种不是可用鱼种；已停用鱼种只能保留历史值，不能新分配")

    proposed_status = payload.review_status if payload.review_status is not None else image.review_status
    if proposed_status == "approved" and not proposed_truth:
        raise HTTPException(status_code=400, detail="通过前必须确认真实鱼种；采集标注不能自动作为 Ground Truth")

    before = image_dict(image)
    image.review_status = proposed_status
    image.truth_species = proposed_truth or None
    if payload.truth_status is not None:
        image.truth_status = payload.truth_status
    elif not proposed_truth:
        image.truth_status = "UNCERTAIN"
    elif proposed_status == "approved":
        image.truth_status = "LIKELY_CORRECT"
    if payload.notes is not None:
        image.notes = payload.notes
    image.reviewed_by = payload.reviewer
    image.reviewed_at = datetime.now(timezone.utc)
    mark_feedback_reviewed(db, image)
    after = image_dict(image)
    db.add(
        ReviewEvent(
            image_asset_id=image.id,
            action="review_update",
            reviewer=payload.reviewer,
            before_json=json.dumps(before, ensure_ascii=False),
            after_json=json.dumps(after, ensure_ascii=False),
        )
    )
    db.commit()
    db.refresh(image)
    return image_dict(image)
''',
)
replace_from(
    "app/main.py",
    '@app.post("/api/datasets/freeze")\n',
    '''@app.post("/api/datasets/freeze")
def dataset_freeze(payload: DatasetFreeze, db: Session = Depends(get_db)):
    try:
        from app.dataset_api import DatasetFreezePreviewRequest, build_preview

        if not payload.preview_hash:
            raise ValueError("请先生成冻结预览；Freeze 必须携带 preview_hash")
        preview = build_preview(
            db,
            DatasetFreezePreviewRequest(
                dataset_version=payload.dataset_version,
                parent_version=payload.parent_version,
                seed=payload.seed,
                train=payload.train,
                val=payload.val,
            ),
        )
        if preview.get("selection_hash") != payload.preview_hash:
            raise ValueError("冻结预览已失效：数据、鱼种状态或父版本发生变化，请重新预览")
        deployed_git = (os.getenv("APP_GIT_COMMIT") or "unknown").strip() or "unknown"
        return freeze_cumulative_dataset(
            db,
            dataset_version=payload.dataset_version,
            parent_version=preview.get("parent_version"),
            git_commit=deployed_git,
            seed=payload.seed,
            train=payload.train,
            val=payload.val,
        )
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
''',
)

# ---------------------------------------------------------------------------
# flywheel.py: truth-only summary and canonical shared Freeze selection.
# ---------------------------------------------------------------------------
replace_once(
    "app/flywheel.py",
    "from app.factory import get_bucket_name\nfrom app.models import DatasetVersion, FeedbackEvent, ImageAsset, SpeciesCatalog\n",
    "from app.data_policy import UNCONFIRMED_TRUTH, truth_distribution\nfrom app.factory import get_bucket_name\nfrom app.freeze_policy import select_freeze_candidates\nfrom app.models import DatasetVersion, FeedbackEvent, ImageAsset, SpeciesCatalog\n",
)
replace_between(
    "app/flywheel.py",
    "def flywheel_summary(db: Session) -> dict:\n",
    "\n\ndef _stable_fraction",
    '''def flywheel_summary(db: Session) -> dict:
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

    distribution, unconfirmed_truth = truth_distribution(db, review_status="approved")
    approved_species = [{"species": name, "count": count} for name, count in distribution]
    if unconfirmed_truth:
        approved_species.append({"species": UNCONFIRMED_TRUTH, "count": unconfirmed_truth})
    return {
        "approved_master_pool": approved_total,
        "approved_truth_unconfirmed": unconfirmed_truth,
        "new_approved_since_latest_dataset": new_approved,
        "active_species": active_species,
        "candidate_species": candidate_species,
        "new_feedback": feedback_new,
        "latest_dataset": latest.dataset_version if latest else None,
        "approved_species": approved_species,
    }
''',
)
# Replace the whole freeze implementation; retain helper functions above it.
replace_from(
    "app/flywheel.py",
    "def freeze_cumulative_dataset(\n",
    '''def freeze_cumulative_dataset(
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
    """Freeze the canonical approved + verified-truth + machine-QA snapshot."""
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

    policy = select_freeze_candidates(db, seed=seed, train=train, val=val)
    selected = policy["selected"]
    if not selected:
        raise ValueError("no images remain after verified-truth, machine-QA and active-species gates")

    catalog_rows = policy["catalog_rows"]
    used_keys = {item["catalog"].species_key for item in selected}
    parent_classes = _load_parent_class_map(bucket, parent)
    class_rows: list[dict] = []
    used_parent_keys: set[str] = set()

    # Preserve prior ordering only for classes that are active and actually have samples.
    for item in sorted(parent_classes, key=lambda x: int(x.get("class_index", 0))):
        key = item.get("species_key")
        if not key or key not in used_keys:
            continue
        row = next((r for r in catalog_rows if r.species_key == key and r.status == "active"), None)
        if not row:
            continue
        class_rows.append({
            "class_index": len(class_rows),
            "species_key": row.species_key,
            "common_name_zh": row.common_name_zh,
            "common_name_en": row.common_name_en,
            "catalog_order": row.catalog_order,
            "status": row.status,
        })
        used_parent_keys.add(key)

    for row in catalog_rows:
        if row.status != "active" or row.species_key not in used_keys or row.species_key in used_parent_keys:
            continue
        class_rows.append({
            "class_index": len(class_rows),
            "species_key": row.species_key,
            "common_name_zh": row.common_name_zh,
            "common_name_en": row.common_name_en,
            "catalog_order": row.catalog_order,
            "status": row.status,
        })
    class_index = {row["species_key"]: row["class_index"] for row in class_rows}

    cutoff = utcnow()
    frozen: list[dict] = []
    for item in selected:
        image = item["image"]
        catalog = item["catalog"]
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
            "presence_status": item["presence_status"],
            "duplicate_group": item["duplicate_group"],
            "group_id": image.group_id or "",
            "split": item["split"],
        })

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
        "selection_mode": "ALL_APPROVED_VERIFIED_TRUTH",
        "quality_filters": [
            "require_truth_species",
            "require_presence_scan",
            "require_dedupe_scan",
            "exclude_nonrepresentative_duplicates_unless_human_reapproved",
            "exclude_multi_fish_unless_human_reapproved",
            "exclude_no_fish_unless_human_reapproved",
            "active_species_only",
        ],
        "git_commit": git_commit or "unknown",
        "seed": seed,
        "approved_master_pool_count": policy["approved_master_pool_count"],
        "image_count": len(frozen),
        "excluded_non_active_species_count": sum(policy["excluded_species"].values()),
        "excluded_species_counts": dict(policy["excluded_species"]),
        "excluded_quality_counts": dict(policy["excluded_quality"]),
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

    db.add(DatasetVersion(
        dataset_version=dataset_version,
        parent_version=parent_version,
        manifest_uri=manifest_uri,
        class_map_uri=class_map_uri,
        train_count=split_counts.get("train", 0),
        val_count=split_counts.get("val", 0),
        test_count=split_counts.get("test", 0),
        species_count=len(class_rows),
        git_commit=git_commit or "unknown",
        selection_mode="ALL_APPROVED_VERIFIED_TRUTH",
        source_cutoff_at=cutoff,
        status="FROZEN",
    ))
    db.commit()
    return meta
''',
)

# ---------------------------------------------------------------------------
# dataset_api.py: same selection engine, preview hash, strict lineage + audit.
# ---------------------------------------------------------------------------
replace_once(
    "app/dataset_api.py",
    "from app.dedupe import ImageFingerprint\n",
    "from app.data_policy import UNCONFIRMED_TRUTH\nfrom app.dedupe import ImageFingerprint\nfrom app.freeze_policy import select_freeze_candidates\n",
)
replace_between(
    "app/dataset_api.py",
    "def build_preview(db: Session, payload: DatasetFreezePreviewRequest) -> dict:\n",
    "\n\ndef _parse_gs_uri",
    '''def build_preview(db: Session, payload: DatasetFreezePreviewRequest) -> dict:
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

    policy = select_freeze_candidates(db, seed=payload.seed, train=payload.train, val=payload.val)
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
        "excluded_quality_counts": dict(policy["excluded_quality"]),
        "excluded_species_counts": dict(policy["excluded_species"]),
        "selection_mode": "ALL_APPROVED_VERIFIED_TRUTH",
        "selection_hash": selection_hash,
        "train_ratio": payload.train,
        "val_ratio": payload.val,
        "test_ratio": round(1.0 - payload.train - payload.val, 6),
    }
''',
)
replace_between(
    "app/dataset_api.py",
    "def finalize_dataset_lineage(db: Session, dataset_version: str) -> dict:\n",
    '\n\n@router.post("/preview")',
    '''def finalize_dataset_lineage(db: Session, dataset_version: str) -> dict:
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
''',
)
# Insert immutable Dataset audit endpoint before detail.
replace_once(
    "app/dataset_api.py",
    '''@router.get("/{dataset_version}")
def detail(dataset_version: str, db: Session = Depends(get_db)):
''',
    '''@router.get("/{dataset_version}/audit")
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

    truth_empty = 0
    species_truth_mismatch = 0
    bad_presence = 0
    inactive_species = 0
    missing_source = 0
    current_nonrepresentative_duplicates = 0
    active_names = {
        row.common_name_zh
        for row in db.scalars(select(SpeciesCatalog).where(SpeciesCatalog.status == "active")).all()
    }
    for item in rows:
        truth = (item.get("truth_species") or "").strip()
        species = (item.get("species") or "").strip()
        if not truth:
            truth_empty += 1
        if species != truth:
            species_truth_mismatch += 1
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
''',
)

# ---------------------------------------------------------------------------
# Ingestion: source labels never become approved truth implicitly.
# ---------------------------------------------------------------------------
replace_once(
    "app/factory.py",
    '''        claimed_species = (row.get("claimed_species") or row.get("species") or row.get("class_name") or "").strip() or None
        notes = row.get("notes") or ""
''',
    '''        claimed_species = (row.get("claimed_species") or row.get("species") or row.get("class_name") or "").strip() or None
        truth_species = (row.get("truth_species") or row.get("species_truth") or "").strip() or None
        # An external manifest may claim a review result, but approved is only valid with explicit Ground Truth.
        if manifest_review == "approved" and not truth_species:
            manifest_review = "pending"
        notes = row.get("notes") or ""
''',
)
replace_once(
    "app/factory.py",
    '''            "claimed_species": claimed_species,
            "scene": row.get("scene"),
''',
    '''            "claimed_species": claimed_species,
            "truth_species": truth_species,
            "scene": row.get("scene"),
''',
)
replace_once(
    "app/factory.py",
    '''                    review_status=manifest_review,
                    truth_status="LIKELY_CORRECT" if manifest_review == "approved" else "UNCERTAIN",
                    **values,
''',
    '''                    review_status=manifest_review,
                    truth_status="LIKELY_CORRECT" if manifest_review == "approved" and truth_species else "UNCERTAIN",
                    **values,
''',
)
replace_between(
    "app/factory.py",
    "def approved_summary(db: Session) -> dict:\n",
    "\n\ndef list_datasets",
    '''def approved_summary(db: Session) -> dict:
    truth = func.nullif(func.trim(ImageAsset.truth_species), "")
    rows = db.execute(
        select(ImageAsset.batch_id, truth, func.count())
        .where(ImageAsset.review_status == "approved")
        .group_by(ImageAsset.batch_id, truth)
    ).all()
    batches: dict[str, dict] = {}
    total = 0
    for batch_id, species, count in rows:
        name = species or "未确认真实鱼种"
        entry = batches.setdefault(batch_id, {"batch_id": batch_id, "approved": 0, "species": {}})
        entry["approved"] += count
        entry["species"][name] = count
        total += count
    return {"total_approved": total, "batches": list(batches.values())}
''',
)
replace_once(
    "app/factory.py",
    ''') -> dict:
    bucket_name = bucket_name or get_bucket_name()
    if not dataset_version.startswith("DS_"):
''',
    ''') -> dict:
    raise RuntimeError("Legacy factory.freeze_dataset is disabled; use POST /api/dataset-freeze/preview then POST /api/datasets/freeze")
    bucket_name = bucket_name or get_bucket_name()
    if not dataset_version.startswith("DS_"):
''',
)
write(
    "scripts/freeze_dataset.py",
    '''#!/usr/bin/env python3
raise SystemExit(
    "Legacy offline Freeze is disabled. Use Model Factory Console: "
    "POST /api/dataset-freeze/preview -> POST /api/datasets/freeze -> finalize."
)
''',
)

# ---------------------------------------------------------------------------
# Bulk review: grouping and drilldown identical; no claimed->truth promotion.
# ---------------------------------------------------------------------------
replace_once(
    "app/bulk_review.py",
    "from app.db import get_db\n",
    "from app.data_policy import mark_feedback_reviewed, review_group_clause, review_group_name, valid_truth_for_image\nfrom app.db import get_db\n",
)
replace_once(
    "app/bulk_review.py",
    'PUBLIC_REVIEW_STATUSES = {"approved", "rejected", "pending"}\n',
    'PUBLIC_REVIEW_STATUSES = {"approved", "rejected", "pending", "needs_review", "hard_case"}\n',
)
replace_once(
    "app/bulk_review.py",
    '''def _image_species(image: ImageAsset) -> str:
    return (image.truth_species or image.claimed_species or "未标注").strip() or "未标注"
''',
    '''def _image_species(image: ImageAsset) -> str:
    return review_group_name(image)
''',
)
replace_once(
    "app/bulk_review.py",
    '''    if species:
        stmt = stmt.where(or_(ImageAsset.truth_species == species, ImageAsset.claimed_species == species))
''',
    '''    if species:
        stmt = stmt.where(review_group_clause(species))
''',
)
replace_once(
    "app/bulk_review.py",
    '                "truth_species": image.truth_species or image.claimed_species,\n',
    '                "truth_species": image.truth_species,\n',
)
replace_from(
    "app/bulk_review.py",
    '@router.post("/api/bulk-review/apply")\n',
    '''@router.post("/api/bulk-review/apply")
def api_bulk_apply(payload: BulkReviewApply, db: Session = Depends(get_db)):
    if not db.get(Batch, payload.batch_id):
        raise HTTPException(status_code=404, detail="batch not found")
    changed = 0
    for item in payload.items:
        if item.review_status not in PUBLIC_REVIEW_STATUSES:
            raise HTTPException(status_code=400, detail=f"invalid review_status: {item.review_status}")
        image = db.scalar(
            select(ImageAsset).where(ImageAsset.batch_id == payload.batch_id, ImageAsset.image_id == item.image_id)
        )
        if not image:
            raise HTTPException(status_code=404, detail=f"image not found: {item.image_id}")

        if "truth_species" in item.model_fields_set:
            truth = (item.truth_species or "").strip()
        else:
            truth = (image.truth_species or "").strip()
        if truth and not valid_truth_for_image(db, image, truth):
            raise HTTPException(status_code=400, detail=f"不可分配真实鱼种: {truth}")
        if item.review_status == "approved" and not truth:
            raise HTTPException(status_code=400, detail=f"{item.image_id}: 通过前必须确认真实鱼种")

        before = {
            "review_status": image.review_status,
            "truth_species": image.truth_species,
            "truth_status": image.truth_status,
            "notes": image.notes,
        }
        image.review_status = item.review_status
        image.truth_species = truth or None
        image.truth_status = "LIKELY_CORRECT" if item.review_status == "approved" and truth else ("UNCERTAIN" if not truth else image.truth_status)
        if item.notes is not None:
            image.notes = item.notes
        image.reviewed_by = "批量审核"
        image.reviewed_at = utcnow()
        mark_feedback_reviewed(db, image)
        db.add(
            ReviewEvent(
                image_asset_id=image.id,
                action="bulk_review_update",
                reviewer="批量审核",
                before_json=json.dumps(before, ensure_ascii=False),
                after_json=json.dumps(
                    {
                        "review_status": image.review_status,
                        "truth_species": image.truth_species,
                        "truth_status": image.truth_status,
                        "notes": image.notes,
                    },
                    ensure_ascii=False,
                ),
            )
        )
        changed += 1
    db.commit()
    return {"batch_id": payload.batch_id, "updated": changed}
''',
)

# ---------------------------------------------------------------------------
# Inspect drilldown: truth-only filters, raw truth, duplicate visibility.
# ---------------------------------------------------------------------------
replace_once(
    "app/inspect.py",
    "from app.db import get_db\n",
    "from app.data_policy import truth_filter_clause\nfrom app.db import get_db\nfrom app.dedupe import ImageFingerprint\n",
)
replace_once(
    "app/inspect.py",
    '''    if species:
        stmt = stmt.where(or_(ImageAsset.truth_species == species, ImageAsset.claimed_species == species))
''',
    '''    if species:
        stmt = stmt.where(truth_filter_clause(species))
''',
)
replace_once(
    "app/inspect.py",
    '''    presence_map = {}
    if image_ids:
        presence_map = {
            row.image_asset_id: row
            for row in db.scalars(select(FishPresenceResult).where(FishPresenceResult.image_asset_id.in_(image_ids))).all()
        }

    filtered = []
''',
    '''    presence_map = {}
    duplicate_map = {}
    if image_ids:
        presence_map = {
            row.image_asset_id: row
            for row in db.scalars(select(FishPresenceResult).where(FishPresenceResult.image_asset_id.in_(image_ids))).all()
        }
        duplicate_map = {
            row.image_asset_id: row
            for row in db.scalars(select(ImageFingerprint).where(ImageFingerprint.image_asset_id.in_(image_ids))).all()
        }

    filtered = []
''',
)
replace_once(
    "app/inspect.py",
    '''        filtered.append(
            {
                "batch_id": image.batch_id,
                "image_id": image.image_id,
                "media_url": f"/media/{image.batch_id}/{image.image_id}",
                "claimed_species": image.claimed_species,
                "truth_species": image.truth_species or image.claimed_species,
                "review_status": image.review_status,
                "notes": image.notes or "",
                "presence": p,
            }
        )
''',
    '''        fp = duplicate_map.get(image.id)
        filtered.append(
            {
                "batch_id": image.batch_id,
                "image_id": image.image_id,
                "media_url": f"/media/{image.batch_id}/{image.image_id}",
                "claimed_species": image.claimed_species,
                "truth_species": image.truth_species,
                "review_status": image.review_status,
                "notes": image.notes or "",
                "presence": p,
                "duplicate": {
                    "group": fp.duplicate_group if fp else None,
                    "is_duplicate": bool(fp and fp.duplicate_group and not fp.is_representative),
                    "kind": fp.duplicate_kind if fp else None,
                },
            }
        )
''',
)

# ---------------------------------------------------------------------------
# Presence / Dedupe summaries: displayed filter count must equal mutation count.
# ---------------------------------------------------------------------------
replace_once(
    "app/presence.py",
    "from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func, select\n",
    "from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func, select\n",
)
replace_between(
    "app/presence.py",
    "def presence_summary(db: Session, batch_id: str) -> dict:\n",
    "\n\ndef _reclassify_saved_evidence",
    '''def presence_summary(db: Session, batch_id: str) -> dict:
    batch = db.get(Batch, batch_id)
    if not batch:
        raise ValueError("batch not found")
    eligible = db.scalar(
        select(func.count()).select_from(ImageAsset).where(
            ImageAsset.batch_id == batch_id,
            ImageAsset.review_status.in_(SCANNABLE_REVIEW_STATUSES),
        )
    ) or 0
    pairs = db.execute(
        select(ImageAsset, FishPresenceResult)
        .join(FishPresenceResult, FishPresenceResult.image_asset_id == ImageAsset.id)
        .where(ImageAsset.batch_id == batch_id)
    ).all()
    counts = {"single_fish": 0, "multi_fish": 0, "no_fish": 0, "uncertain": 0, "error": 0}
    filterable_no_fish = 0
    for image, row in pairs:
        status = effective_status(row)
        counts[status] = counts.get(status, 0) + 1
        if status == "no_fish" and image.review_status in FILTERABLE_REVIEW_STATUSES:
            filterable_no_fish += 1
    remaining = db.scalar(
        select(func.count())
        .select_from(ImageAsset)
        .outerjoin(FishPresenceResult, FishPresenceResult.image_asset_id == ImageAsset.id)
        .where(
            ImageAsset.batch_id == batch_id,
            ImageAsset.review_status.in_(SCANNABLE_REVIEW_STATUSES),
            FishPresenceResult.id.is_(None),
        )
    ) or 0
    return {
        "batch_id": batch_id,
        "eligible": eligible,
        "scanned": len(pairs),
        "single_fish": counts.get("single_fish", 0),
        "multi_fish": counts.get("multi_fish", 0),
        "no_fish": counts.get("no_fish", 0),
        "filterable_no_fish": filterable_no_fish,
        "uncertain": counts.get("uncertain", 0),
        "error": counts.get("error", 0),
        "remaining": remaining,
    }
''',
)
replace_once(
    "app/dedupe.py",
    '''    return {
        "batch_id": batch_id,
        "total": total,
        "scanned": scanned,
        "groups": groups,
        "duplicate_images": duplicates,
        "exact_duplicates": exact,
        "near_duplicates": near,
        "remaining": max(0, total - scanned),
    }
''',
    '''    filterable_duplicates = db.scalar(
        select(func.count()).select_from(ImageFingerprint)
        .join(ImageAsset, ImageAsset.id == ImageFingerprint.image_asset_id)
        .where(
            ImageFingerprint.batch_id == batch_id,
            ImageFingerprint.duplicate_group.is_not(None),
            ImageFingerprint.is_representative.is_(False),
            ImageAsset.review_status.in_(FILTERABLE_REVIEW_STATUSES),
        )
    ) or 0
    return {
        "batch_id": batch_id,
        "total": total,
        "scanned": scanned,
        "groups": groups,
        "duplicate_images": duplicates,
        "filterable_duplicate_images": filterable_duplicates,
        "exact_duplicates": exact,
        "near_duplicates": near,
        "remaining": max(0, total - scanned),
    }
''',
)

# ---------------------------------------------------------------------------
# UI: truthful terminology, no hidden truth promotion, matching drilldowns.
# ---------------------------------------------------------------------------
replace_once(
    "app/templates/overview.html",
    "<div class=\"section\"><h2>可用数据总池 · 鱼种分布</h2>",
    "<div class=\"section\"><h2>人工通过池 · 真实鱼种分布</h2>",
)
replace_once(
    "app/templates/overview.html",
    "['累计可用图片',f.approved_master_pool||0,()=>openInspect({review_status:'approved'})]",
    "['人工通过图片',f.approved_master_pool||0,()=>openInspect({review_status:'approved'})]",
)

# review.html
replace_once(
    "app/templates/review.html",
    '<select id="species"><option value="">全部鱼种</option></select>',
    '<select id="species"><option value="">全部真实鱼种</option><option value="未确认真实鱼种">未确认真实鱼种</option></select>',
)
replace_once(
    "app/templates/review.html",
    '''function speciesOptions(selected){return catalog.filter(x=>x.status!=='retired').map(x=>`<option value="${esc(x.common_name_zh)}" ${x.common_name_zh===selected?'selected':''}>${esc(x.common_name_zh)}${x.status==='candidate'?' · 候选鱼种':''}</option>`).join('')}''',
    '''function speciesOptions(selected){const current=catalog.find(x=>x.common_name_zh===selected);const historical=current?.status==='retired'?`<option value="${esc(selected)}" selected>${esc(selected)} · 已停用（历史保留）</option>`:'';return `<option value="" ${!selected?'selected':''}>未确认真实鱼种</option>`+historical+catalog.filter(x=>x.status!=='retired').map(x=>`<option value="${esc(x.common_name_zh)}" ${x.common_name_zh===selected?'selected':''}>${esc(x.common_name_zh)}${x.status==='candidate'?' · 候选鱼种':''}</option>`).join('')}''',
)
replace_once(
    "app/templates/review.html",
    "const selected=x.truth_species||x.claimed_species||'';",
    "const selected=x.truth_species||'';",
)
replace_once(
    "app/templates/review.html",
    "真实鱼种默认沿用采集标注，只在发现错误时修改。多鱼会保留，但不会进入当前单鱼分类数据集。",
    "采集标注仅供参考，不会自动成为真实鱼种。点击“通过”前必须明确确认真实鱼种。",
)
replace_once(
    "app/templates/review.html",
    '''async function save(status){if(!current)return;const selected=document.getElementById('status').value;const payload={review_status:status,truth_species:document.getElementById('truthSpecies').value,notes:document.getElementById('notes').value,reviewer:'网页审核'};''',
    '''async function save(status){if(!current)return;const selected=document.getElementById('status').value;const truth=document.getElementById('truthSpecies').value;if(status==='approved'&&!truth){alert('通过前必须确认真实鱼种');return}const payload={review_status:status,truth_species:truth,notes:document.getElementById('notes').value,reviewer:'网页审核'};''',
)

# bulk_review.html
replace_once(
    "app/templates/bulk_review.html",
    "按鱼种连续看图；默认真实鱼种沿用采集标注，只处理异常图片。",
    "按审核分组连续看图：优先真实鱼种，未确认时仅用采集标注分组；采集标注不会自动成为 Ground Truth。",
)
replace_once(
    "app/templates/bulk_review.html",
    '''function speciesOptions(selected){return catalog.filter(x=>x.status!=='retired').map(x=>`<option value="${esc(x.common_name_zh)}" ${x.common_name_zh===selected?'selected':''}>${esc(x.common_name_zh)}</option>`).join('')}''',
    '''function speciesOptions(selected){const current=catalog.find(x=>x.common_name_zh===selected);const historical=current?.status==='retired'?`<option value="${esc(selected)}" selected>${esc(selected)} · 已停用（历史保留）</option>`:'';return `<option value="" ${!selected?'selected':''}>未确认真实鱼种</option>`+historical+catalog.filter(x=>x.status!=='retired').map(x=>`<option value="${esc(x.common_name_zh)}" ${x.common_name_zh===selected?'selected':''}>${esc(x.common_name_zh)}</option>`).join('')}''',
)
replace_once(
    "app/templates/bulk_review.html",
    "${speciesOptions(x.truth_species||x.claimed_species||currentSpecies)}",
    "${speciesOptions(x.truth_species||'')}",
)
replace_once(
    "app/templates/bulk_review.html",
    "items.forEach((x,i)=>{x._state=(x.review_status==='approved'?'approved':x.review_status==='rejected'?'rejected':'pending')})",
    "items.forEach((x,i)=>{x._state=x.review_status||'pending'})",
)
replace_once(
    "app/templates/bulk_review.html",
    '''async function submitPage(){if(!items.length)return;const batch=document.getElementById('batch').value;const payload={batch_id:batch,items:items.map((x,i)=>({image_id:x.image_id,review_status:x._state||'pending',truth_species:document.getElementById('species-'+i).value,notes:x.notes||''}))};''',
    '''async function submitPage(){if(!items.length)return;for(let i=0;i<items.length;i++){if(items[i]._state==='approved'&&!document.getElementById('species-'+i).value){msg(`${items[i].image_id} 通过前必须确认真实鱼种`,true);return}}const batch=document.getElementById('batch').value;const payload={batch_id:batch,items:items.map((x,i)=>({image_id:x.image_id,review_status:x._state||'pending',truth_species:document.getElementById('species-'+i).value,notes:x.notes||''}))};''',
)

# inspect.html
replace_once(
    "app/templates/inspect.html",
    '<select id="species"><option value="">全部鱼种</option></select>',
    '<select id="species"><option value="">全部真实鱼种</option><option value="未确认真实鱼种">未确认真实鱼种</option></select>',
)
replace_once(
    "app/templates/inspect.html",
    '''function speciesOptions(selected){return catalog.filter(x=>x.status!=='retired').map(x=>`<option value="${esc(x.common_name_zh)}" ${x.common_name_zh===selected?'selected':''}>${esc(x.common_name_zh)}</option>`).join('')}''',
    '''function speciesOptions(selected){const current=catalog.find(x=>x.common_name_zh===selected);const historical=current?.status==='retired'?`<option value="${esc(selected)}" selected>${esc(selected)} · 已停用（历史保留）</option>`:'';return `<option value="" ${!selected?'selected':''}>未确认真实鱼种</option>`+historical+catalog.filter(x=>x.status!=='retired').map(x=>`<option value="${esc(x.common_name_zh)}" ${x.common_name_zh===selected?'selected':''}>${esc(x.common_name_zh)}</option>`).join('')}''',
)
replace_once(
    "app/templates/inspect.html",
    '''const state=normalizeStatus(x.review_status),p=x.presence||{},overridden=!!p.human_override;return `<div class="card"''',
    '''const state=normalizeStatus(x.review_status),p=x.presence||{},d=x.duplicate||{},overridden=!!p.human_override;return `<div class="card"''',
)
replace_once(
    "app/templates/inspect.html",
    '''<div class="muted">采集标注：${esc(x.claimed_species||'-')}</div><label>真实鱼种</label><select class="speciesSelect" id="species-${i}">${speciesOptions(x.truth_species||x.claimed_species||'')}</select>''',
    '''<div class="muted">采集标注：${esc(x.claimed_species||'-')}</div>${d.is_duplicate?`<div class="row"><span class="pill warn">机器重复：${esc(d.kind||'near')}</span><span class="overrideHint">若误判，人工重新“通过”并提交后 Freeze 以人工结果为准</span></div>`:''}<label>真实鱼种</label><select class="speciesSelect" id="species-${i}">${speciesOptions(x.truth_species||'')}</select>''',
)
replace_once(
    "app/templates/inspect.html",
    "items.forEach(x=>{x._state=normalizeStatus(x.review_status);x._override=x.presence?.human_override||''})",
    "items.forEach(x=>{x._state=x.review_status||'pending';x._override=x.presence?.human_override||''})",
)
replace_once(
    "app/templates/inspect.html",
    '''async function submitPage(){if(!items.length)return;document.body.classList.add('saving');let done=0;try{for(let i=0;i<items.length;i++){const x=items[i];''',
    '''async function submitPage(){if(!items.length)return;for(let i=0;i<items.length;i++){if(items[i]._state==='approved'&&!document.getElementById('species-'+i).value){msg(`${items[i].image_id} 通过前必须确认真实鱼种`,true);return}}document.body.classList.add('saving');let done=0;try{for(let i=0;i<items.length;i++){const x=items[i];''',
)

# datasets.html
replace_once("app/templates/datasets.html", "<h2>累计可用数据总池</h2>", "<h2>累计人工通过池</h2>")
replace_once(
    "app/templates/datasets.html",
    "从全部“通过”图片中，排除明确无鱼、多鱼、近重复和完全重复；只训练已启用鱼种。",
    "从全部“通过”图片中，仅选择已确认真实鱼种、已完成鱼体检测和去重检查的数据；排除无鱼、多鱼、重复及非启用鱼种。人工在机器结果之后重新通过可作为最终裁决。",
)
replace_once(
    "app/templates/datasets.html",
    "let previewSignature='';",
    "let previewSignature='',previewHash='';",
)
replace_once(
    "app/templates/datasets.html",
    "function invalidate(){previewSignature='';document.getElementById('freezeBtn').disabled=true;document.getElementById('preview').style.display='none'}",
    "function invalidate(){previewSignature='';previewHash='';document.getElementById('freezeBtn').disabled=true;document.getElementById('preview').style.display='none'}",
)
replace_once(
    "app/templates/datasets.html",
    "previewSignature=signature(p);document.getElementById('freezeBtn').disabled=d.image_count<=0;",
    "previewSignature=signature(p);previewHash=d.selection_hash||'';document.getElementById('freezeBtn').disabled=d.image_count<=0||!previewHash;",
)
replace_once(
    "app/templates/datasets.html",
    "if(previewSignature!==signature(p)){msg('配置已变化，请重新生成预览。',true);return}",
    "if(previewSignature!==signature(p)||!previewHash){msg('配置或数据已变化，请重新生成预览。',true);return}p.preview_hash=previewHash",
)
replace_once(
    "app/templates/datasets.html",
    "const q=d.excluded_quality_counts||{};const sp=d.excluded_species_counts||{};document.getElementById('pvExcluded').textContent=`通过池 ${d.approved_master_pool_count} 张；排除：无鱼 ${q.no_fish||0}、多鱼 ${q.multi_fish||0}、近重复 ${q.near_duplicate||0}、完全重复 ${q.exact_duplicate||0}、非启用鱼种 ${Object.values(sp).reduce((a,b)=>a+b,0)}。`;",
    "const q=d.excluded_quality_counts||{};const sp=d.excluded_species_counts||{};document.getElementById('pvExcluded').textContent=`通过池 ${d.approved_master_pool_count} 张；排除：未确认真实鱼种 ${sp['未确认真实鱼种']||0}、未鱼体检测 ${q.presence_not_scanned||0}、未去重检测 ${q.dedupe_not_scanned||0}、无鱼 ${q.no_fish||0}、多鱼 ${q.multi_fish||0}、近重复 ${q.near_duplicate||0}、完全重复 ${q.exact_duplicate||0}、其他非启用鱼种 ${Object.entries(sp).filter(([k])=>k!=='未确认真实鱼种').reduce((a,[,v])=>a+v,0)}。`;",
)

# batches.html use actual filterable counts for mutation prompts/buttons.
replace_once(
    "app/templates/batches.html",
    "${d.duplicate_images?`<button class=\"filterBtn\" onclick=\"filterDuplicates('${esc(x.batch_id)}',${d.duplicate_images})\">过滤重复项</button>`:''}",
    "${d.filterable_duplicate_images?`<button class=\"filterBtn\" onclick=\"filterDuplicates('${esc(x.batch_id)}',${d.filterable_duplicate_images})\">过滤重复项</button>`:''}",
)
replace_once(
    "app/templates/batches.html",
    "${p.no_fish?`<button class=\"filterBtn\" onclick=\"filterNoFish('${esc(x.batch_id)}',${p.no_fish})\">过滤无鱼</button>`:''}",
    "${p.filterable_no_fish?`<button class=\"filterBtn\" onclick=\"filterNoFish('${esc(x.batch_id)}',${p.filterable_no_fish})\">过滤无鱼</button>`:''}",
)

# ---------------------------------------------------------------------------
# Business-state regression test.
# ---------------------------------------------------------------------------
write(
    "scripts/smoke_business_state.py",
    '''#!/usr/bin/env python3
import os
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("REGISTRY_DB_URL", "sqlite:///:memory:")

from fastapi import HTTPException
from sqlalchemy import select

from app.entry import app
from app.bulk_review import api_bulk_apply, api_bulk_images, api_bulk_species, BulkReviewApply, BulkReviewItem
from app.dataset_api import DatasetFreezePreviewRequest, build_preview
from app.db import SessionLocal, init_db
from app.dedupe import ImageFingerprint
from app.flywheel import ensure_species_catalog, flywheel_summary
from app.main import ReviewUpdate, apply_review_filters, update_review
from app.models import Batch, ImageAsset, SpeciesCatalog
from app.presence import FishPresenceResult


def add_image(db, batch, image_id, claimed, truth=None, status="pending"):
    row = ImageAsset(
        batch_id=batch,
        image_id=image_id,
        file_name=f"{image_id}.jpg",
        object_name=f"raw/{image_id}.jpg",
        gcs_uri=f"gs://test/{image_id}.jpg",
        claimed_species=claimed,
        truth_species=truth,
        review_status=status,
    )
    db.add(row)
    db.flush()
    return row


def add_qa(db, image, *, presence="single_fish", duplicate_kind=None):
    p = FishPresenceResult(
        image_asset_id=image.id,
        batch_id=image.batch_id,
        status=presence,
        fish_count=1 if presence == "single_fish" else 0,
    )
    db.add(p)
    fp = ImageFingerprint(
        image_asset_id=image.id,
        batch_id=image.batch_id,
        sha256=(f"{image.id:064x}")[-64:],
        phash_json="[]",
        dhash="0" * 16,
        crop_hash="",
        histogram_json="[]",
        width=100,
        height=100,
        duplicate_group=f"DUP_{image.id}" if duplicate_kind else None,
        is_representative=False if duplicate_kind else True,
        duplicate_kind=duplicate_kind,
    )
    db.add(fp)
    db.flush()
    return p, fp


def main():
    init_db()
    db = SessionLocal()
    try:
        ensure_species_catalog(db)
        db.add(Batch(batch_id="BATCH_P0", source="smoke", image_count=8, manifest_uri="gs://test/m.csv", raw_uri="gs://test/raw", status="REGISTERED"))
        db.flush()

        wrong_claim = add_image(db, "BATCH_P0", "I1", "黄骨鱼", "黑鱼", "approved")
        yellow = add_image(db, "BATCH_P0", "I2", "黄骨鱼", "黄骨鱼", "approved")
        legacy_unconfirmed = add_image(db, "BATCH_P0", "I3", "黄骨鱼", None, "approved")
        pending = add_image(db, "BATCH_P0", "I4", "黄骨鱼", None, "needs_review")
        nofish = add_image(db, "BATCH_P0", "I5", "鲫鱼", "鲫鱼", "approved")
        duplicate = add_image(db, "BATCH_P0", "I6", "鲤鱼", "鲤鱼", "approved")
        unscanned = add_image(db, "BATCH_P0", "I7", "草鱼", "草鱼", "approved")
        db.commit()

        for image in [wrong_claim, yellow, legacy_unconfirmed, pending]:
            add_qa(db, image)
        p_nofish, _ = add_qa(db, nofish, presence="no_fish")
        _, fp_dup = add_qa(db, duplicate, duplicate_kind="near")
        db.commit()

        # Statistics are Ground Truth only: I1 is black, not yellow; legacy null is explicit unconfirmed.
        summary = flywheel_summary(db)
        dist = {x["species"]: x["count"] for x in summary["approved_species"]}
        assert dist["黑鱼"] == 1, dist
        assert dist["黄骨鱼"] == 1, dist
        assert dist["未确认真实鱼种"] == 1, dist

        yellow_rows = db.scalars(apply_review_filters(select(ImageAsset), None, "BATCH_P0", "黄骨鱼", None)).all()
        assert [x.image_id for x in yellow_rows] == ["I2"], [x.image_id for x in yellow_rows]

        # Approval cannot promote claimed -> truth implicitly.
        try:
            update_review("BATCH_P0", "I4", ReviewUpdate(review_status="approved", reviewer="smoke"), db)
            raise AssertionError("approval without truth should fail")
        except HTTPException as exc:
            assert exc.status_code == 400
            db.rollback()

        update_review("BATCH_P0", "I4", ReviewUpdate(review_status="approved", truth_species="黄骨鱼", reviewer="smoke"), db)
        db.refresh(pending)
        assert pending.truth_species == "黄骨鱼" and pending.review_status == "approved"

        # Bulk work queue groups by truth first, claimed only while truth is empty; drilldown matches count.
        pending2 = add_image(db, "BATCH_P0", "I8", "黄骨鱼", None, "hard_case")
        add_qa(db, pending2)
        db.commit()
        groups = api_bulk_species("BATCH_P0", "pending", db)
        yellow_group = next(x for x in groups if x["species"] == "黄骨鱼")
        drilled = api_bulk_images("BATCH_P0", "黄骨鱼", "pending", None, 60, 0, db)
        assert yellow_group["count"] == drilled["total"] == 1, (groups, drilled)
        assert drilled["items"][0]["truth_species"] is None

        # Pending save with blank truth does not upgrade claimed label.
        api_bulk_apply(BulkReviewApply(batch_id="BATCH_P0", items=[BulkReviewItem(image_id="I8", review_status="hard_case", truth_species=None)]), db)
        db.refresh(pending2)
        assert pending2.truth_species is None and pending2.review_status == "hard_case"

        # Freeze Preview requires QA scans and verified truth.
        preview1 = build_preview(db, DatasetFreezePreviewRequest(dataset_version="DS_P0", seed=7, train=0.7, val=0.15))
        assert preview1["excluded_species_counts"].get("未确认真实鱼种") == 1, preview1
        assert preview1["excluded_quality_counts"].get("presence_not_scanned") == 1, preview1
        assert preview1["excluded_quality_counts"].get("no_fish") == 1, preview1
        assert preview1["excluded_quality_counts"].get("near_duplicate") == 1, preview1

        # Human re-approval after machine result overrides no-fish / near-duplicate machine gate.
        nofish.reviewed_at = p_nofish.updated_at + timedelta(seconds=1)
        nofish.reviewed_by = "人工复核"
        duplicate.reviewed_at = fp_dup.updated_at + timedelta(seconds=1)
        duplicate.reviewed_by = "人工复核"
        db.commit()
        preview2 = build_preview(db, DatasetFreezePreviewRequest(dataset_version="DS_P0", seed=7, train=0.7, val=0.15))
        assert preview2["image_count"] == preview1["image_count"] + 2, (preview1, preview2)

        # Snapshot hash changes when truth changes.
        old_hash = preview2["selection_hash"]
        yellow.truth_species = "黑鱼"
        db.commit()
        preview3 = build_preview(db, DatasetFreezePreviewRequest(dataset_version="DS_P0", seed=7, train=0.7, val=0.15))
        assert preview3["selection_hash"] != old_hash

        # Retired historical truth may be preserved but not newly assigned.
        other = db.get(SpeciesCatalog, "other_freshwater_fish")
        other.status = "retired"
        hist = add_image(db, "BATCH_P0", "I9", "其他淡水鱼", "其他淡水鱼", "pending")
        db.commit()
        update_review("BATCH_P0", "I9", ReviewUpdate(review_status="pending", truth_species="其他淡水鱼", reviewer="smoke"), db)
        try:
            update_review("BATCH_P0", "I8", ReviewUpdate(review_status="pending", truth_species="其他淡水鱼", reviewer="smoke"), db)
            raise AssertionError("new retired truth assignment should fail")
        except HTTPException:
            db.rollback()

        paths = app.openapi()["paths"]
        assert "/api/dataset-freeze/{dataset_version}/audit" in paths
        print("P0 business state smoke OK", preview3)
    finally:
        db.close()


if __name__ == "__main__":
    main()
''',
)
replace_once(
    ".github/workflows/ci.yml",
    '''      - name: Run Dataset Freeze smoke test
        env:
          REGISTRY_DB_URL: 'sqlite:///:memory:'
        run: python scripts/smoke_dataset_freeze.py
''',
    '''      - name: Run Dataset Freeze smoke test
        env:
          REGISTRY_DB_URL: 'sqlite:///:memory:'
        run: python scripts/smoke_dataset_freeze.py
      - name: Run P0 business state consistency smoke test
        env:
          REGISTRY_DB_URL: 'sqlite:///:memory:'
        run: python scripts/smoke_business_state.py
''',
)

# Existing Dataset Freeze smoke must provide QA for all eligible rows and assert truth-only gate.
replace_once(
    "scripts/smoke_dataset_freeze.py",
    '''        add_image(db, batch_id, "IMG_UNSCANNED", "鲤鱼")
''',
    '''        unscanned = add_image(db, batch_id, "IMG_UNSCANNED", "鲤鱼")
''',
)
replace_once(
    "scripts/smoke_dataset_freeze.py",
    '''                FishPresenceResult(image_asset_id=single.id, batch_id=batch_id, status="single_fish", fish_count=1),
''',
    '''                FishPresenceResult(image_asset_id=single.id, batch_id=batch_id, status="single_fish", fish_count=1),
                FishPresenceResult(image_asset_id=unscanned.id, batch_id=batch_id, status="single_fish", fish_count=1),
''',
)
# Give the two expected eligible samples dedupe fingerprints; preserve the existing duplicate fixture.
replace_once(
    "scripts/smoke_dataset_freeze.py",
    '''                ImageFingerprint(
                    image_asset_id=duplicate.id,
''',
    '''                ImageFingerprint(
                    image_asset_id=single.id,
                    batch_id=batch_id,
                    sha256="a" * 64,
                    phash_json="[]",
                    dhash="0" * 16,
                    crop_hash="",
                    histogram_json="[]",
                    width=100,
                    height=100,
                ),
                ImageFingerprint(
                    image_asset_id=unscanned.id,
                    batch_id=batch_id,
                    sha256="b" * 64,
                    phash_json="[]",
                    dhash="0" * 16,
                    crop_hash="",
                    histogram_json="[]",
                    width=100,
                    height=100,
                ),
                ImageFingerprint(
                    image_asset_id=multi.id,
                    batch_id=batch_id,
                    sha256="c" * 64,
                    phash_json="[]",
                    dhash="0" * 16,
                    crop_hash="",
                    histogram_json="[]",
                    width=100,
                    height=100,
                ),
                ImageFingerprint(
                    image_asset_id=no_fish.id,
                    batch_id=batch_id,
                    sha256="e" * 64,
                    phash_json="[]",
                    dhash="0" * 16,
                    crop_hash="",
                    histogram_json="[]",
                    width=100,
                    height=100,
                ),
                ImageFingerprint(
                    image_asset_id=duplicate.id,
''',
)

# Remove transient patcher/workflow from the final diff. Workflow file is created separately.
for transient in [ROOT / "scripts/_p0_apply.py", ROOT / ".github/workflows/p0-self-patch.yml"]:
    if transient.exists():
        transient.unlink()

print("P0 patch applied")
