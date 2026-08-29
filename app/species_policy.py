from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import SpeciesCatalog


TRAINING_MIN_TOTAL = 20
TRAINING_MIN_TRAIN = 10
TRAINING_MIN_VAL = 3
TRAINING_MIN_TEST = 3
TRAINING_MIN_GROUPS = 3

# Collection Contract V1. species_key is the stable machine identity; common_name_zh
# is the label shown in review / Dataset class maps. Search aliases are retained as
# catalog notes for collectors and future alias-normalization work.
TARGET_SPECIES_PRESETS = [
    {"catalog_order": 0, "species_key": "grass_carp", "common_name_zh": "草鱼", "common_name_en": "Grass carp", "aliases": ["草鱼", "鲩鱼"]},
    {"catalog_order": 1, "species_key": "bighead_carp", "common_name_zh": "鳙鱼", "common_name_en": "Bighead carp", "aliases": ["鳙鱼", "花鲢", "胖头鱼"]},
    {"catalog_order": 2, "species_key": "silver_carp", "common_name_zh": "白鲢", "common_name_en": "Silver carp", "aliases": ["白鲢", "鲢鱼"]},
    {"catalog_order": 3, "species_key": "common_carp", "common_name_zh": "鲤鱼", "common_name_en": "Common carp", "aliases": ["鲤鱼"]},
    {"catalog_order": 4, "species_key": "crucian_carp", "common_name_zh": "鲫鱼", "common_name_en": "Crucian carp", "aliases": ["鲫鱼", "野鲫"]},
    {"catalog_order": 5, "species_key": "largemouth_bass", "common_name_zh": "加州鲈", "common_name_en": "Largemouth bass", "aliases": ["加州鲈", "大口黑鲈"]},
    {"catalog_order": 6, "species_key": "snakehead", "common_name_zh": "黑鱼", "common_name_en": "Snakehead", "aliases": ["黑鱼", "乌鳢", "生鱼"]},
    {"catalog_order": 7, "species_key": "yellow_catfish", "common_name_zh": "黄骨鱼", "common_name_en": "Yellow catfish", "aliases": ["黄骨鱼", "黄颡鱼", "黄辣丁"]},
    {"catalog_order": 8, "species_key": "black_carp", "common_name_zh": "青鱼", "common_name_en": "Black carp", "aliases": ["青鱼", "螺蛳青"]},
    {"catalog_order": 9, "species_key": "tilapia", "common_name_zh": "罗非鱼", "common_name_en": "Tilapia", "aliases": ["罗非鱼", "非洲鲫"]},
    {"catalog_order": 10, "species_key": "mandarin_fish", "common_name_zh": "鳜鱼", "common_name_en": "Mandarin fish", "aliases": ["鳜鱼", "桂鱼", "桂花鱼"]},
    {"catalog_order": 11, "species_key": "topmouth_culter", "common_name_zh": "翘嘴鲌", "common_name_en": "Topmouth culter", "aliases": ["翘嘴", "翘嘴鲌", "翘嘴红鲌"]},
    {"catalog_order": 12, "species_key": "blunt_snout_bream", "common_name_zh": "鳊鱼 / 武昌鱼", "common_name_en": "Blunt snout bream", "aliases": ["鳊鱼", "武昌鱼", "团头鲂"]},
    {"catalog_order": 13, "species_key": "chinese_catfish", "common_name_zh": "鲶鱼", "common_name_en": "Chinese catfish", "aliases": ["鲶鱼", "土鲶"], "notes": "采集时不要混入塘鲺 / 胡子鲶"},
    {"catalog_order": 14, "species_key": "mud_carp", "common_name_zh": "鲮鱼", "common_name_en": "Mud carp", "aliases": ["鲮鱼", "土鲮"]},
    {"catalog_order": 15, "species_key": "sharpbelly", "common_name_zh": "白条", "common_name_en": "Sharpbelly", "aliases": ["白条", "餐条", "白鲦"]},
    {"catalog_order": 16, "species_key": "chinese_hooksnout_carp", "common_name_zh": "马口鱼", "common_name_en": "Chinese hooksnout carp", "aliases": ["马口", "马口鱼"]},
    {"catalog_order": 17, "species_key": "yellowcheek", "common_name_zh": "鳡鱼", "common_name_en": "Yellowcheek", "aliases": ["鳡鱼", "鳡"]},
    {"catalog_order": 18, "species_key": "yellowfin_culter", "common_name_zh": "黄尾鲴", "common_name_en": "Yellowfin culter", "aliases": ["黄尾鲴", "黄尾"]},
    {"catalog_order": 19, "species_key": "redfin_culter", "common_name_zh": "红眼鳟", "common_name_en": "Redfin culter", "aliases": ["红眼鳟", "赤眼鳟"]},
]


def _preset_notes(preset: dict) -> str:
    parts = ["系统预置鱼种（20 类采集合同 V1）"]
    aliases = preset.get("aliases") or []
    if aliases:
        parts.append("搜索别名：" + "、".join(aliases))
    if preset.get("notes"):
        parts.append(str(preset["notes"]))
    return "；".join(parts)


def ensure_target_species(db: Session) -> None:
    """Idempotently seed the 20 target species without overriding later operator status changes.

    If the same Chinese canonical name was manually created with an auto-generated
    `species_*` key, adopt it into the stable collection-contract key once. This is
    safe because image Ground Truth is stored by canonical Chinese name, not by a
    SpeciesCatalog foreign key. Explicit non-generated keys are never rewritten.
    """

    rows = db.scalars(select(SpeciesCatalog)).all()
    by_key = {row.species_key: row for row in rows}
    by_name = {row.common_name_zh: row for row in rows}
    used_orders = {row.catalog_order for row in rows}
    changed = False

    for preset in TARGET_SPECIES_PRESETS:
        key = preset["species_key"]
        name = preset["common_name_zh"]
        row = by_key.get(key)

        if row is None:
            same_name = by_name.get(name)
            if same_name is not None and same_name.species_key.startswith("species_"):
                old_key = same_name.species_key
                same_name.species_key = key
                same_name.status = "active"
                same_name.common_name_en = same_name.common_name_en or preset.get("common_name_en")
                same_name.notes = same_name.notes or _preset_notes(preset)
                by_key.pop(old_key, None)
                by_key[key] = same_name
                row = same_name
                changed = True
            elif same_name is not None:
                # A deliberate explicit key already owns the canonical name. Keep it
                # rather than creating a duplicate name or rewriting operator intent.
                continue

        if row is None:
            order = int(preset["catalog_order"])
            if order in used_orders:
                order = max(used_orders, default=-1) + 1
            used_orders.add(order)
            row = SpeciesCatalog(
                species_key=key,
                catalog_order=order,
                common_name_zh=name,
                common_name_en=preset.get("common_name_en"),
                status="active",
                is_other=False,
                notes=_preset_notes(preset),
            )
            db.add(row)
            by_key[key] = row
            by_name[name] = row
            changed = True
        else:
            # Do not force status back to active on every startup: operators may
            # intentionally move a preset to candidate/retired later.
            if not row.common_name_en and preset.get("common_name_en"):
                row.common_name_en = preset["common_name_en"]
                changed = True
            if not row.notes:
                row.notes = _preset_notes(preset)
                changed = True

    if changed:
        db.commit()


def training_thresholds() -> dict[str, int]:
    return {
        "total": TRAINING_MIN_TOTAL,
        "train": TRAINING_MIN_TRAIN,
        "val": TRAINING_MIN_VAL,
        "test": TRAINING_MIN_TEST,
        "group_count": TRAINING_MIN_GROUPS,
    }


def training_eligibility(counts: dict, *, is_other: bool = False) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if is_other:
        reasons.append("其他淡水鱼为兜底标签，默认不作为训练类别")
    if int(counts.get("total", 0) or 0) < TRAINING_MIN_TOTAL:
        reasons.append(f"总数 {int(counts.get('total', 0) or 0)} < {TRAINING_MIN_TOTAL}")
    if int(counts.get("group_count", 0) or 0) < TRAINING_MIN_GROUPS:
        reasons.append(f"独立 group {int(counts.get('group_count', 0) or 0)} < {TRAINING_MIN_GROUPS}")
    if int(counts.get("train", 0) or 0) < TRAINING_MIN_TRAIN:
        reasons.append(f"Train {int(counts.get('train', 0) or 0)} < {TRAINING_MIN_TRAIN}")
    if int(counts.get("val", 0) or 0) < TRAINING_MIN_VAL:
        reasons.append(f"Val {int(counts.get('val', 0) or 0)} < {TRAINING_MIN_VAL}")
    if int(counts.get("test", 0) or 0) < TRAINING_MIN_TEST:
        reasons.append(f"Test {int(counts.get('test', 0) or 0)} < {TRAINING_MIN_TEST}")
    return (not reasons, reasons)
