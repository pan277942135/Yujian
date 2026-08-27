"""
YuJian Dataset Freeze V0.1

Creates immutable training dataset manifests from reviewed image assets.
This module is intentionally independent from training code.
"""

import csv
import json
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class FreezeRule:
    review_status: str = "approved"
    presence_status: str = "single_fish"
    exclude_duplicates: bool = True


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


class DatasetFreezeService:
    """Build Dataset Version artifacts from approved master data."""

    def __init__(self, dataset_version: str):
        self.dataset_version = dataset_version

    def split(self, image_id: str):
        """Stable split so reruns produce identical datasets."""
        value = int(hashlib.md5(image_id.encode()).hexdigest()[:8], 16) % 100
        if value < 70:
            return "train"
        if value < 90:
            return "val"
        return "test"

    def build_manifest(self, rows, output_dir: str):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        manifest = []
        for row in rows:
            species = row.get("truth_species") or row.get("claimed_species")
            if species not in TARGET_SPECIES:
                continue

            manifest.append({
                "image_id": row["image_id"],
                "gcs_uri": row["gcs_uri"],
                "species": species,
                "split": self.split(row["image_id"]),
                "batch_id": row.get("batch_id"),
            })

        with open(output / "dataset_manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=manifest[0].keys() if manifest else [])
            writer.writeheader()
            writer.writerows(manifest)

        classes = {str(i): name for i, name in enumerate(TARGET_SPECIES)}
        (output / "class_mapping.json").write_text(
            json.dumps(classes, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        report = {
            "dataset_version": self.dataset_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_images": len(manifest),
            "classes": len(TARGET_SPECIES),
        }
        (output / "freeze_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return report
