import hashlib
import json
from pathlib import Path

from trainer.detector_dataset_gate import validate_v02_dataset, validate_v02_materialized_dataset


def ready_dataset():
    rows = []
    split_image_counts = {"train": 41, "val": 3, "test": 1}
    for split, image_count in split_image_counts.items():
        for index in range(image_count):
            negative = split == "train" and index == image_count - 1
            image_sha = hashlib.sha256(f"source-{split}-{index}".encode()).hexdigest()
            review_id = f"det_fish_negative_pool_entry:{index + 1}:event:{index + 1}" if negative else f"review-{split}-{index}"
            box = {"bbox_xyxy": [0.1, 0.1, 0.6, 0.8], "review_record_ids": [review_id]}
            row_provenance = [{
                "review_record_id": review_id,
                "reviewer_id": "reviewer-1",
                "reviewed_at": "2026-10-20T00:00:00Z",
                "review_status": "NO_FISH_CONFIRMED" if negative else "ACCEPTED",
                "accepted_bbox_xywh": [0.1, 0.1, 0.5, 0.7],
            }]
            if negative:
                row_provenance[0].update({
                    "pool_entry_id": 1,
                    "pool_event_id": 1,
                    "pool_action": "ADD",
                    "pool_source_kind": "PRESENCE_NO_FISH",
                    "pool_source_record_id": "presence-1",
                    "pool_source_image_sha256": image_sha,
                })
            rows.append({
                "image_id": f"{split}-{index}",
                "image_sha256": image_sha,
                "materialized_image_sha256": hashlib.sha256(b"fixture image").hexdigest(),
                "source_group_id": f"group-{split}-{index}",
                "split": split,
                "file_name": f"{split}-{index}.jpg",
                "image_kind": "complete_scene",
                "width": 100,
                "height": 80,
                "reviewer_id": "reviewer-1",
                "reviewed_at": "2026-10-20T00:00:00Z",
                "source_uri": f"gs://source/{split}-{index}.jpg",
                "training_eligibility_basis": "database eligibility fixture",
                "review_status": "NO_FISH_CONFIRMED" if negative else "ACCEPTED",
                "fish_present": not negative,
                "label_source": "accepted_bbox_human_review" if not negative else "det_fish_negative_pool_human_confirmation",
                "review_record_ids": [review_id],
                "review_provenance": row_provenance,
                "boxes": [] if negative else [box],
            })
    manifest_bytes = json.dumps(rows, separators=(",", ":"), sort_keys=True).encode()
    archive_bytes = b"immutable frozen COCO archive fixture"
    source_image_audit = [
        {
            "review_record_id": review_id,
            "kind": "negative" if not row["fish_present"] else "positive",
            "source_uri": row["source_uri"],
            "status": "READ_OK",
            "source_image_sha256": row["image_sha256"],
            "width": row["width"],
            "height": row["height"],
            "exif_orientation": 1,
        }
        for row in rows
        for review_id in row["review_record_ids"]
    ]
    counts = {
        "train": {"images": 41, "annotations": 40, "negative_images": 1},
        "val": {"images": 3, "annotations": 3, "negative_images": 0},
        "test": {"images": 1, "annotations": 1, "negative_images": 0},
    }
    report = {
        "dataset_version": "DET_DS_v0.2",
        "status": "FROZEN_READY_FOR_TRAINING",
        "source": "accepted_bbox_human_review",
        "source_image_audit_verified": True,
        "source_image_audit": source_image_audit,
        "human_review_verified": True,
        "held_out_exclusions_verified": True,
        "split_integrity": "PASS",
        "negative_coverage_verified": True,
        "negative_pool_provenance_verified": True,
        "source_reconciliation": {
            "status": "PASS",
            "manifest_sha256": "1" * 64,
            "metadata_sha256": "2" * 64,
            "manifest_uri": "gs://test-bucket/datasets/accepted_pool/manifest.csv",
            "metadata_uri": "gs://test-bucket/datasets/accepted_pool/metadata.json",
            "accepted_pool_rows": 44,
        },
        "source_accepted_bbox_record_count": 44,
        "source_human_negative_record_count": 1,
        "source_negative_pool_active_member_count": 1,
        "negative_pool_member_errors": [],
        "excluded_held_out_accepted_bbox_record_count": 0,
        "excluded_held_out_negative_record_count": 0,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "negatives": 1,
        "splits": counts,
    }
    return report, manifest_bytes, archive_bytes, rows


def test_gate_derives_verified_counts_from_manifest_and_hashes():
    report, manifest_bytes, archive_bytes, _rows = ready_dataset()
    result = validate_v02_dataset(report, manifest_bytes, hashlib.sha256(archive_bytes).hexdigest())
    assert result["ready"] is True
    assert result["counts_verified"] is True
    assert result["counts"]["train_images"] == 41
    assert result["counts"]["train_boxes"] == 40
    assert result["counts"]["human_reviewed_negatives"] == 1


def test_gate_does_not_mark_report_count_mismatch_as_verified():
    report, manifest_bytes, archive_bytes, _rows = ready_dataset()
    report["splits"]["train"]["images"] += 1

    result = validate_v02_dataset(report, manifest_bytes, hashlib.sha256(archive_bytes).hexdigest())

    assert result["counts_verified"] is False
    assert any("train 图片数或 BBox 数" in item for item in result["blockers"])


def test_gate_rejects_manifest_archive_mismatch_and_cross_split_leakage():
    report, manifest_bytes, archive_bytes, rows = ready_dataset()
    report["archive_sha256"] = "0" * 64
    result = validate_v02_dataset(report, manifest_bytes, hashlib.sha256(archive_bytes).hexdigest())
    assert not result["ready"]
    assert any("实际 SHA-256" in item for item in result["blockers"])

    rows[0]["source_group_id"] = rows[-1]["source_group_id"]
    manifest_bytes = json.dumps(rows, separators=(",", ":"), sort_keys=True).encode()
    report["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    report["archive_sha256"] = hashlib.sha256(archive_bytes).hexdigest()
    result = validate_v02_dataset(report, manifest_bytes, hashlib.sha256(archive_bytes).hexdigest())
    assert any("跨集合泄漏" in item for item in result["blockers"])


def test_gate_rejects_held_out_case_and_nonhuman_positive_label():
    report, manifest_bytes, archive_bytes, rows = ready_dataset()
    rows[0]["image_sha256"] = "57dd3d39f832dfa1cc56b8af74a00b274f321714228adf08097ed131e19ecab4"
    rows[1]["label_source"] = "FishPresenceResult/google-vision-presence-v0.4"
    manifest_bytes = json.dumps(rows, separators=(",", ":"), sort_keys=True).encode()
    report["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    result = validate_v02_dataset(report, manifest_bytes, hashlib.sha256(archive_bytes).hexdigest())
    assert any("独立留出图片" in item for item in result["blockers"])
    assert any("无效或非人工确认" in item for item in result["blockers"])


def test_materialized_coco_package_must_match_reviewed_boxes(tmp_path: Path):
    report, manifest_bytes, archive_bytes, rows = ready_dataset()
    gate = validate_v02_dataset(report, manifest_bytes, hashlib.sha256(archive_bytes).hexdigest())
    root = tmp_path / "DET_DS_v0.2"
    (root / "annotations").mkdir(parents=True)
    (root / "bootstrap_manifest.json").write_bytes(manifest_bytes)
    for split in ("train", "val", "test"):
        (root / f"{split}2017").mkdir()
        selected = [row for row in rows if row["split"] == split]
        images = []
        annotations = []
        annotation_id = 1
        for image_id, row in enumerate(selected, start=1):
            (root / f"{split}2017" / row["file_name"]).write_bytes(b"fixture image")
            images.append({"id": image_id, "file_name": row["file_name"], "width": 100, "height": 80})
            for box in row["boxes"]:
                x1, y1, x2, y2 = box["bbox_xyxy"]
                width, height = (x2 - x1) * 100, (y2 - y1) * 80
                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x1 * 100, y1 * 80, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                })
                annotation_id += 1
        payload = {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "fish"}]}
        (root / "annotations" / f"instances_{split}2017.json").write_text(json.dumps(payload))

    assert validate_v02_materialized_dataset(root, manifest_bytes, gate["counts"]) == []

    coco_path = root / "annotations" / "instances_train2017.json"
    coco = json.loads(coco_path.read_text())
    coco["annotations"][0]["bbox"][0] += 3
    coco_path.write_text(json.dumps(coco))
    blockers = validate_v02_materialized_dataset(root, manifest_bytes, gate["counts"])
    assert any("BBox 坐标与人工审核 Manifest 不一致" in item for item in blockers)
