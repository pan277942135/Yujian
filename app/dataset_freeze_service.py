"""Dataset Freeze V0.1 service layer.

Creates immutable dataset preparation metadata from reviewed images.
The API layer will call this service in the next step.
"""

import hashlib
import json
from dataclasses import dataclass


TARGET_SPECIES = [
    "鲫鱼",
    "鲤鱼",
    "草鱼",
    "青鱼",
    "白鲢",
    "鳙鱼",
    "黑鱼",
    "黄骨鱼",
    "加州鲈",
]


@dataclass
class FreezeConfig:
    dataset_version: str
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15



def stable_split(image_id: str, config: FreezeConfig) -> str:
    """Stable split. Same image always goes to same bucket."""
    value = int(hashlib.sha256(image_id.encode("utf-8")).hexdigest()[:8], 16)
    score = (value % 10000) / 10000

    if score < config.train_ratio:
        return "train"
    if score < config.train_ratio + config.val_ratio:
        return "val"
    return "test"



def should_include(image) -> bool:
    """First version dataset inclusion policy."""
    return (
        image.review_status == "approved"
        and image.truth_species in TARGET_SPECIES
        and getattr(image, "presence_status", "single_fish") == "single_fish"
        and not getattr(image, "is_duplicate", False)
    )



def build_manifest(images, config: FreezeConfig):
    items = []

    for image in images:
        if not should_include(image):
            continue

        items.append(
            {
                "image_id": image.image_id,
                "batch_id": image.batch_id,
                "gcs_uri": image.gcs_uri,
                "species": image.truth_species,
                "split": stable_split(image.image_id, config),
            }
        )

    return items



def build_class_mapping(items):
    species = []
    for item in items:
        if item["species"] not in species:
            species.append(item["species"])

    species = [x for x in TARGET_SPECIES if x in species]
    return {str(index): name for index, name in enumerate(species)}



def build_freeze_report(items, config: FreezeConfig):
    counts = {}
    split_counts = {}

    for item in items:
        counts[item["species"]] = counts.get(item["species"], 0) + 1
        split_counts[item["split"]] = split_counts.get(item["split"], 0) + 1

    return {
        "dataset_version": config.dataset_version,
        "total_images": len(items),
        "species_count": len(counts),
        "species_distribution": counts,
        "split_distribution": split_counts,
        "status": "FROZEN",
    }



def serialize_metadata(items, config: FreezeConfig):
    return {
        "manifest": items,
        "class_mapping": build_class_mapping(items),
        "report": build_freeze_report(items, config),
    }
