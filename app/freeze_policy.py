from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.data_policy import UNCONFIRMED_TRUTH, human_approval_overrides, normalized_truth
from app.dedupe import ImageFingerprint
from app.models import ImageAsset, SpeciesCatalog
from app.presence import FishPresenceResult, effective_status
from app.species_policy import ensure_target_species, training_eligibility, training_thresholds

SPLITS = ("train", "val", "test")
SPLIT_STRATEGY = "DETERMINISTIC_STRATIFIED_GROUP_V1"


def stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def choose_split(key: str, seed: int, train: float, val: float) -> str:
    """Legacy deterministic hash split helper retained for compatibility/tests."""
    p = stable_fraction(key, seed)
    if p < train:
        return "train"
    if p < train + val:
        return "val"
    return "test"


def _target_split_counts(total: int, train: float, val: float) -> dict[str, int]:
    """Largest-remainder targets with one sample per split whenever total >= 3."""
    ratios = {"train": train, "val": val, "test": 1.0 - train - val}
    raw = {name: total * ratio for name, ratio in ratios.items()}
    counts = {name: int(math.floor(value)) for name, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(SPLITS, key=lambda name: (-(raw[name] - counts[name]), SPLITS.index(name)))
    for name in order[:remaining]:
        counts[name] += 1

    if total >= 3:
        for missing in [name for name in SPLITS if counts[name] == 0]:
            donors = [name for name in SPLITS if counts[name] > 1]
            if not donors:
                break
            donor = max(donors, key=lambda name: (counts[name], ratios[name], -SPLITS.index(name)))
            counts[donor] -= 1
            counts[missing] += 1
    return counts


def _group_composition(items: list[dict]) -> Counter[str]:
    return Counter(item["catalog"].species_key for item in items)


def _deviation_cost(current: dict[str, Counter[str]], targets: dict[str, dict[str, int]]) -> float:
    cost = 0.0
    for species_key, target in targets.items():
        for split in SPLITS:
            denom = max(1, target[split])
            diff = current[species_key][split] - target[split]
            cost += (diff * diff) / denom
    return cost


def _assign_stratified_group_splits(selected: list[dict], *, seed: int, train: float, val: float) -> dict:
    """Assign whole groups to splits while balancing every species deterministically."""
    groups: dict[str, list[dict]] = defaultdict(list)
    species_totals: Counter[str] = Counter()
    species_group_keys: dict[str, set[str]] = defaultdict(set)
    species_names: dict[str, str] = {}

    for item in selected:
        group_key = item["group_key"]
        groups[group_key].append(item)
        species_key = item["catalog"].species_key
        species_totals[species_key] += 1
        species_group_keys[species_key].add(group_key)
        species_names[species_key] = item["catalog"].common_name_zh

    targets = {species_key: _target_split_counts(total, train, val) for species_key, total in species_totals.items()}
    current: dict[str, Counter[str]] = {key: Counter() for key in species_totals}
    assignment: dict[str, str] = {}

    group_order = sorted(
        groups,
        key=lambda key: (
            -len(groups[key]),
            -len(_group_composition(groups[key])),
            stable_fraction(f"group-order:{key}", seed),
            key,
        ),
    )

    for group_key in group_order:
        composition = _group_composition(groups[group_key])
        candidates = []
        for split in SPLITS:
            need_score = 0.0
            overflow = 0.0
            for species_key, count in composition.items():
                target = targets[species_key][split]
                before = current[species_key][split]
                need_score += count * (target - before) / max(1, target)
                overflow += max(0, before + count - target) / max(1, target)
            score = need_score - 0.35 * overflow
            tie = stable_fraction(f"place:{group_key}:{split}", seed)
            candidates.append((score, -tie, -SPLITS.index(split), split))
        chosen = max(candidates)[-1]
        assignment[group_key] = chosen
        for species_key, count in composition.items():
            current[species_key][chosen] += count

    for _ in range(max(1, len(groups) * 3)):
        missing = [
            (species_key, split)
            for species_key in sorted(species_totals)
            for split in SPLITS
            if current[species_key][split] == 0
        ]
        if not missing:
            break
        moved = False
        baseline_cost = _deviation_cost(current, targets)
        for species_key, wanted_split in missing:
            options = []
            for group_key in sorted(species_group_keys[species_key]):
                donor = assignment[group_key]
                if donor == wanted_split:
                    continue
                composition = _group_composition(groups[group_key])
                if any(current[key][donor] - count <= 0 for key, count in composition.items()):
                    continue

                for key, count in composition.items():
                    current[key][donor] -= count
                    current[key][wanted_split] += count
                new_cost = _deviation_cost(current, targets)
                for key, count in composition.items():
                    current[key][wanted_split] -= count
                    current[key][donor] += count

                options.append(
                    (
                        new_cost - baseline_cost,
                        stable_fraction(f"repair:{species_key}:{wanted_split}:{group_key}", seed),
                        group_key,
                        donor,
                    )
                )
            if not options:
                continue
            _delta, _tie, group_key, donor = min(options)
            composition = _group_composition(groups[group_key])
            assignment[group_key] = wanted_split
            for key, count in composition.items():
                current[key][donor] -= count
                current[key][wanted_split] += count
            moved = True
            break
        if not moved:
            break

    blockers: list[dict] = []
    warnings: list[dict] = []
    per_species: dict[str, dict] = {}
    for species_key in sorted(species_totals, key=lambda key: species_names[key]):
        name = species_names[species_key]
        counts = {split: int(current[species_key][split]) for split in SPLITS}
        total = int(species_totals[species_key])
        group_count = len(species_group_keys[species_key])
        per_species[name] = {
            "species_key": species_key,
            "total": total,
            "group_count": group_count,
            **counts,
            "target": targets[species_key],
        }

        if total < 3:
            blockers.append({"species": name, "code": "TOO_FEW_SAMPLES_FOR_THREE_WAY_SPLIT", "message": f"{name} 仅 {total} 张，无法同时覆盖 Train/Val/Test"})
        elif group_count < 3:
            blockers.append({"species": name, "code": "TOO_FEW_GROUPS_FOR_THREE_WAY_SPLIT", "message": f"{name} 仅 {group_count} 个独立 group，保持 group 隔离时无法稳定三路切分"})

        for split in SPLITS:
            if counts[split] == 0:
                blockers.append({"species": name, "code": f"ZERO_{split.upper()}_COVERAGE", "message": f"{name} 的 {split} 样本为 0，禁止 Freeze"})

        if 0 < counts["train"] < 10:
            warnings.append({"species": name, "code": "LOW_TRAIN", "message": f"{name} Train 仅 {counts['train']} 张（<10）"})
        if 0 < counts["val"] < 3:
            warnings.append({"species": name, "code": "LOW_VAL", "message": f"{name} Val 仅 {counts['val']} 张（<3）"})
        if 0 < counts["test"] < 3:
            warnings.append({"species": name, "code": "LOW_TEST", "message": f"{name} Test 仅 {counts['test']} 张（<3）"})

    for item in selected:
        item["split"] = assignment[item["group_key"]]

    return {
        "strategy": SPLIT_STRATEGY,
        "targets": targets,
        "per_species": per_species,
        "warnings": warnings,
        "blockers": blockers,
        "group_count": len(groups),
    }


def _empty_split() -> dict:
    return {
        "strategy": SPLIT_STRATEGY,
        "targets": {},
        "per_species": {},
        "warnings": [],
        "blockers": [],
        "group_count": 0,
    }


def _training_gate(
    selected: list[dict],
    catalog_rows: list[SpeciesCatalog],
    *,
    seed: int,
    train: float,
    val: float,
) -> tuple[list[dict], dict, list[dict], list[dict]]:
    """Remove low-data active species without blocking mature classes.

    Eligibility is recalculated after each removal because group-aware stratified
    assignment can shift when a class disappears. The loop stops only when every
    remaining class satisfies the default training thresholds.
    """
    active_rows = [row for row in catalog_rows if row.status == "active"]
    active_by_key = {row.species_key: row for row in active_rows}
    remaining = list(selected)
    disabled_by_key: dict[str, dict] = {}

    while remaining:
        split = _assign_stratified_group_splits(remaining, seed=seed, train=train, val=val)
        per_by_key = {row["species_key"]: row for row in split["per_species"].values()}
        eligible_keys: set[str] = set()
        newly_disabled: list[str] = []

        for row in active_rows:
            counts = per_by_key.get(
                row.species_key,
                {
                    "species_key": row.species_key,
                    "total": 0,
                    "group_count": 0,
                    "train": 0,
                    "val": 0,
                    "test": 0,
                    "target": {"train": 0, "val": 0, "test": 0},
                },
            )
            enabled, reasons = training_eligibility(counts, is_other=bool(row.is_other))
            if enabled:
                eligible_keys.add(row.species_key)
            else:
                if row.species_key not in disabled_by_key:
                    disabled_by_key[row.species_key] = {
                        "species": row.common_name_zh,
                        "species_key": row.species_key,
                        "training_enabled": False,
                        "reasons": reasons,
                        **counts,
                    }
                if row.species_key in {item["catalog"].species_key for item in remaining}:
                    newly_disabled.append(row.species_key)

        if not newly_disabled:
            enabled_rows = [
                {
                    "species": active_by_key[key].common_name_zh,
                    "species_key": key,
                    "training_enabled": True,
                    **per_by_key[key],
                }
                for key in sorted(eligible_keys)
                if key in per_by_key
            ]
            return remaining, split, enabled_rows, sorted(disabled_by_key.values(), key=lambda x: x["species"])

        remaining = [item for item in remaining if item["catalog"].species_key not in set(newly_disabled)]

    # Report all active species as disabled when none can satisfy the gate.
    for row in active_rows:
        if row.species_key not in disabled_by_key:
            counts = {
                "species_key": row.species_key,
                "total": 0,
                "group_count": 0,
                "train": 0,
                "val": 0,
                "test": 0,
                "target": {"train": 0, "val": 0, "test": 0},
            }
            _enabled, reasons = training_eligibility(counts, is_other=bool(row.is_other))
            disabled_by_key[row.species_key] = {
                "species": row.common_name_zh,
                "species_key": row.species_key,
                "training_enabled": False,
                "reasons": reasons,
                **counts,
            }
    return [], _empty_split(), [], sorted(disabled_by_key.values(), key=lambda x: x["species"])


def select_freeze_candidates(
    db: Session,
    *,
    seed: int,
    train: float,
    val: float,
    allow_split_blockers: bool = False,
) -> dict:
    ensure_target_species(db)
    catalog_rows = db.scalars(select(SpeciesCatalog).order_by(SpeciesCatalog.catalog_order)).all()
    active_by_name = {row.common_name_zh: row for row in catalog_rows if row.status == "active"}
    images = db.scalars(
        select(ImageAsset)
        .where(ImageAsset.review_status == "approved")
        .order_by(ImageAsset.batch_id, ImageAsset.id)
    ).all()
    if not images:
        disabled = []
        for row in catalog_rows:
            if row.status != "active":
                continue
            counts = {"total": 0, "group_count": 0, "train": 0, "val": 0, "test": 0}
            _enabled, reasons = training_eligibility(counts, is_other=bool(row.is_other))
            disabled.append({"species": row.common_name_zh, "species_key": row.species_key, "training_enabled": False, "reasons": reasons, **counts})
        return {
            "approved_master_pool_count": 0,
            "selected": [],
            "catalog_rows": catalog_rows,
            "excluded_quality": Counter(),
            "excluded_species": Counter(),
            "split_strategy": SPLIT_STRATEGY,
            "per_species_split_counts": {},
            "split_warnings": [],
            "split_blockers": [{"species": "系统", "code": "NO_TRAINING_ELIGIBLE_SPECIES", "message": "没有满足默认训练门槛的鱼种"}],
            "split_group_count": 0,
            "training_thresholds": training_thresholds(),
            "training_enabled_species": [],
            "training_disabled_species": disabled,
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
        group_key = image.group_id or duplicate_group or f"{image.batch_id}:{image.image_id}"
        selected.append(
            {
                "image": image,
                "catalog": catalog,
                "presence_status": presence_status,
                "duplicate_group": duplicate_group,
                "group_key": group_key,
            }
        )

    selected, split, training_enabled, training_disabled = _training_gate(
        selected,
        catalog_rows,
        seed=seed,
        train=train,
        val=val,
    )

    blockers = list(split["blockers"])
    if not selected:
        blockers.append({"species": "系统", "code": "NO_TRAINING_ELIGIBLE_SPECIES", "message": "没有满足默认训练门槛的鱼种"})

    if blockers and not allow_split_blockers:
        messages = "; ".join(item["message"] for item in blockers[:8])
        if len(blockers) > 8:
            messages += f"; 另有 {len(blockers) - 8} 项"
        raise ValueError(f"Dataset Split Gate 未通过：{messages}")

    return {
        "approved_master_pool_count": len(images),
        "selected": selected,
        "catalog_rows": catalog_rows,
        "excluded_quality": excluded_quality,
        "excluded_species": excluded_species,
        "split_strategy": split["strategy"],
        "per_species_split_counts": split["per_species"],
        "split_warnings": split["warnings"],
        "split_blockers": blockers,
        "split_group_count": split["group_count"],
        "training_thresholds": training_thresholds(),
        "training_enabled_species": training_enabled,
        "training_disabled_species": training_disabled,
    }
