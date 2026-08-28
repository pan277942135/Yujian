from __future__ import annotations

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
