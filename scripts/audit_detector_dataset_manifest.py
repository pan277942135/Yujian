#!/usr/bin/env python3
"""Summarize a DET_DS bootstrap manifest without opening or publishing images."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def audit_manifest(entries: list[dict]) -> dict:
    split_images: Counter[str] = Counter()
    split_boxes: Counter[str] = Counter()
    split_negatives: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    batch_splits: dict[str, set[str]] = defaultdict(set)
    image_ids: Counter[str] = Counter()
    source_shas: Counter[str] = Counter()
    negative_with_boxes = 0
    positive_without_boxes = 0
    missing_core_fields = Counter()
    audited_entries = []
    for entry in entries:
        split = str(entry.get("split") or "UNKNOWN")
        status = str(entry.get("presence_status") or entry.get("status") or "UNKNOWN")
        boxes = entry.get("boxes") or []
        is_negative = bool(entry.get("negative", status == "no_fish"))
        split_images[split] += int(not entry.get("skipped", False))
        split_boxes[split] += len(boxes)
        split_negatives[split] += int(is_negative and not entry.get("skipped", False))
        statuses[status] += 1
        if not entry.get("skipped", False):
            audited_entries.append(entry)
        if not entry.get("skipped", False) and entry.get("batch_id") is not None:
            batch_splits[str(entry["batch_id"])].add(split)
        if entry.get("image_id") is not None:
            image_ids[str(entry["image_id"])] += 1
        if entry.get("source_sha256"):
            source_shas[str(entry["source_sha256"]).lower()] += 1
        else:
            missing_core_fields["source_sha256"] += 1
        for field in ("reviewer_id", "review_status", "license", "permission_basis", "source_group_id", "hard_case_tags"):
            if entry.get(field) in (None, "", []):
                missing_core_fields[field] += 1
        negative_with_boxes += int(is_negative and bool(boxes))
        positive_without_boxes += int(not is_negative and not boxes and not entry.get("skipped", False))

    cross_split_batches = sorted(batch for batch, splits in batch_splits.items() if len(splits) > 1)
    duplicate_image_ids = sorted(image_id for image_id, count in image_ids.items() if count > 1)
    duplicate_source_shas = sorted(sha for sha, count in source_shas.items() if count > 1)
    return {
        "entry_count": len(entries),
        "split_images": dict(sorted(split_images.items())),
        "split_annotations": dict(sorted(split_boxes.items())),
        "split_negative_images": dict(sorted(split_negatives.items())),
        "presence_status_counts": dict(sorted(statuses.items())),
        "negative_rows_with_boxes": negative_with_boxes,
        "positive_rows_without_boxes": positive_without_boxes,
        "cross_split_batch_ids": cross_split_batches,
        "duplicate_image_ids": duplicate_image_ids,
        "duplicate_source_sha256": duplicate_source_shas,
        "missing_provenance_fields_count": dict(sorted(missing_core_fields.items())),
        "source_identity_audit_possible": bool(audited_entries)
        and all(entry.get("source_sha256") for entry in audited_entries),
        "hard_scene_coverage_audit_possible": bool(audited_entries)
        and all(entry.get("hard_case_tags") for entry in audited_entries),
        "manual_bbox_provenance_audit_possible": bool(audited_entries)
        and all(entry.get("reviewer_id") for entry in audited_entries),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    entries = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise SystemExit("bootstrap manifest must be a JSON list")
    result = {
        "schema_version": "DETECTOR_DATASET_MANIFEST_AUDIT_V1",
        "status": "AUDITED",
        "manifest_path_disclosed": False,
        "audit": audit_manifest(entries),
    }
    if args.report:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        result["bootstrap_report"] = report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
