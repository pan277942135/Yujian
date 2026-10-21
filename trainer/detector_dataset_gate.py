"""Strict, offline-checkable gate for the reviewed DET_DS_v0.2 artifact.

This module is intentionally stdlib-only so the console and GPU worker can use
the same dataset contract without importing either runtime's heavy dependencies.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


DATASET_VERSION = "DET_DS_v0.2"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HELD_OUT_SHA256 = {
    # Case A, Case B, and the confirmed 04-181200 incident image.
    "2dd703a46b378093604e10f0add454606393138f7501774bd3bbcca75352c9fe",
    "a7ed3bb0410c191364f6d853b4078aebdc2f3b9baa4da34b7b3f6e8e0e93a56b",
    "57dd3d39f832dfa1cc56b8af74a00b274f321714228adf08097ed131e19ecab4",
    # Seven separately reviewed hard-case regression images.
    "a59fbac1d7d5b36207ccf1f0bf3930ba034124c448775e624e009b0dbb3918af",
    "0fc341627e4345131a15fbf3514454f053dcb349db17bf1171c8f2f8756226cf",
    "e74132cd01500b13515262c16a07e1be5d2db6a8401e5aed7222740b184cc649",
    "ad31bb8c2c545005b445917de0e888bbceb5bcd77a034c63d5bddabfb13175d3",
    "3ed483f0401201b76b52ca27e56091c21f712603922e7c04e75fec7ba98a1f41",
    "77bf1fe6f368b6bcc3bd25a274ba8ed378f3a46519dedb0bdd19a098d4a30a98",
}
_HUMAN_BOX_SOURCES = {
    "accepted_review",
    "accepted_bbox_human_review",
    "human_reviewed_accepted_bbox",
}


def _sha(value: Any) -> bool:
    return bool(_SHA256_RE.fullmatch(str(value or "")))


def _count(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed >= 0 else None


def _row_boxes(row: dict[str, Any]) -> list[dict[str, Any]] | None:
    boxes = row.get("boxes")
    if not isinstance(boxes, list):
        return None
    if any(not isinstance(box, dict) for box in boxes):
        return None
    return boxes


def _valid_box(box: dict[str, Any]) -> bool:
    coords = box.get("bbox_xyxy")
    if not isinstance(coords, (list, tuple)) or len(coords) != 4:
        return False
    try:
        x1, y1, x2, y2 = (float(item) for item in coords)
    except (TypeError, ValueError):
        return False
    return (
        all(0.0 <= item <= 1.0 for item in (x1, y1, x2, y2))
        and x1 < x2
        and y1 < y2
    )


def validate_v02_dataset(
    report: dict[str, Any] | None,
    manifest_bytes: bytes | None,
    archive_sha256: str | None,
) -> dict[str, Any]:
    """Return derived split counts and precise blockers for a frozen v0.2 set.

    Boolean flags in the report are necessary but not sufficient: row-level
    human review, source hashes, group splits, held-out incident hashes, and
    file digests are checked from the actual manifest/archive bytes as well.
    """

    blockers: list[str] = []
    counts = {
        "train_images": 0,
        "train_boxes": 0,
        "val_images": 0,
        "val_boxes": 0,
        "test_images": 0,
        "test_boxes": 0,
        "human_reviewed_negatives": 0,
        "train_negative_images": 0,
        "val_negative_images": 0,
        "test_negative_images": 0,
        "human_reviewed_complete_scene_images": 0,
        "human_reviewed_bbox_images": 0,
        "human_reviewed_bbox_records": 0,
    }
    if not isinstance(report, dict):
        return {
            "ready": False,
            "blockers": ["尚未找到 DET_DS_v0.2/bootstrap_report.json"],
            "counts": counts,
            "counts_verified": False,
        }

    if report.get("dataset_version") != DATASET_VERSION:
        blockers.append("数据集版本不是 DET_DS_v0.2")
    if report.get("status") != "FROZEN_READY_FOR_TRAINING":
        blockers.append("数据集尚未标记为 FROZEN_READY_FOR_TRAINING")
    source_reconciliation = report.get("source_reconciliation")
    if not isinstance(source_reconciliation, dict) or source_reconciliation.get("status") != "PASS":
        blockers.append("Accepted Pool Manifest 与审核数据库的逐记录核对未通过")
        source_reconciliation_verified = False
    else:
        source_reconciliation_verified = True
        for field, label in (("manifest_sha256", "Accepted Pool Manifest"), ("metadata_sha256", "Accepted Pool metadata")):
            if not _sha(source_reconciliation.get(field)):
                source_reconciliation_verified = False
                blockers.append(f"{label} 缺少有效 SHA-256 记录")
        for field, label in (("manifest_uri", "Accepted Pool Manifest"), ("metadata_uri", "Accepted Pool metadata")):
            if not str(source_reconciliation.get(field) or "").startswith("gs://"):
                source_reconciliation_verified = False
                blockers.append(f"{label} 缺少 GCS 来源 URI")
    if report.get("source_record_errors"):
        blockers.append("存在未解决的原图、权限、审核或标签来源错误")
    if report.get("source_image_audit_verified") is not True:
        blockers.append("Accepted Pool 原图 URI、实际字节 SHA、EXIF 方向或归一化尺寸审计未通过")
    if str(report.get("source") or "").strip().lower() not in _HUMAN_BOX_SOURCES:
        blockers.append("数据集来源没有声明为人工 accepted_bbox 审核")
    for field, message in (
        ("human_review_verified", "缺少人工标签核验通过证据"),
        ("held_out_exclusions_verified", "缺少独立事故留出集排除证据"),
        ("negative_coverage_verified", "缺少人工确认无鱼负样本覆盖证据"),
        ("negative_pool_provenance_verified", "缺少 DET_FISH 负样本池人工操作来源核验"),
    ):
        if report.get(field) is not True:
            blockers.append(message)
    if report.get("split_integrity") != "PASS":
        blockers.append("缺少按来源组划分且无数据泄漏的 PASS 记录")

    if manifest_bytes is None:
        blockers.append("缺少 DET_DS_v0.2/bootstrap_manifest.json")
        manifest = None
        manifest_counts_verified = False
    else:
        actual_manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        expected_manifest_sha = str(report.get("manifest_sha256") or "")
        manifest_counts_verified = _sha(expected_manifest_sha) and actual_manifest_sha == expected_manifest_sha
        if not manifest_counts_verified:
            blockers.append("bootstrap_manifest.json SHA-256 与冻结报告不一致")
        try:
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            manifest = None
            blockers.append("bootstrap_manifest.json 不是有效 UTF-8 JSON")
        if not isinstance(manifest, list) or not manifest:
            manifest = None
            blockers.append("bootstrap_manifest.json 必须是非空的逐图记录数组")

    expected_archive_sha = str(report.get("archive_sha256") or "")
    if not _sha(expected_archive_sha):
        blockers.append("冻结报告缺少有效的 archive_sha256")
    elif not _sha(archive_sha256) or archive_sha256 != expected_archive_sha:
        blockers.append("DET_DS_v0.2.tar.gz 实际 SHA-256 与冻结报告不一致")

    split_images: dict[str, int] = defaultdict(int)
    split_boxes: dict[str, int] = defaultdict(int)
    source_group_splits: dict[str, set[str]] = defaultdict(set)
    image_sha_splits: dict[str, set[str]] = defaultdict(set)
    image_ids: set[str] = set()
    held_out: set[str] = set()
    row_blockers = 0
    all_review_ids: set[str] = set()
    all_negative_pool_entry_ids: set[int] = set()
    all_negative_pool_event_ids: set[int] = set()

    if manifest is not None:
        for row_number, row in enumerate(manifest, start=1):
            if not isinstance(row, dict):
                row_blockers += 1
                continue
            image_id = str(row.get("image_id") or row.get("stable_image_id") or "").strip()
            image_sha = str(row.get("image_sha256") or row.get("source_image_sha256") or "").strip()
            source_group = str(row.get("source_group_id") or "").strip()
            split = str(row.get("split") or "").strip().lower()
            reviewer = str(row.get("reviewer_id") or "").strip()
            reviewed_at = str(row.get("reviewed_at") or "").strip()
            source_uri = str(row.get("source_uri") or "").strip()
            training_eligibility_basis = str(row.get("training_eligibility_basis") or "").strip()
            materialized_sha = str(row.get("materialized_image_sha256") or "").strip()
            review_record_ids = row.get("review_record_ids")
            review_provenance = row.get("review_provenance")
            file_name = str(row.get("file_name") or "").strip()
            image_kind = str(row.get("image_kind") or "").strip().lower()
            review_status = str(row.get("review_status") or "").strip().upper()
            fish_present = row.get("fish_present")
            boxes = _row_boxes(row)
            try:
                width = int(row.get("width") or 0)
                height = int(row.get("height") or 0)
            except (TypeError, ValueError):
                width = height = 0
            label_source = str(row.get("label_source") or row.get("bbox_source") or "").strip().lower()

            valid = (
                bool(image_id)
                and image_id not in image_ids
                and _sha(image_sha)
                and _sha(materialized_sha)
                and bool(source_group)
                and split in {"train", "val", "test"}
                and bool(file_name)
                and Path(file_name).name == file_name
                and image_kind == "complete_scene"
                and width > 0
                and height > 0
                and bool(reviewer)
                and bool(reviewed_at)
                and bool(source_uri)
                and bool(training_eligibility_basis)
                and isinstance(review_record_ids, list)
                and all(str(item or "").strip() for item in review_record_ids)
                and isinstance(review_provenance, list)
                and isinstance(fish_present, bool)
                and boxes is not None
            )
            if not valid:
                row_blockers += 1
                continue
            image_ids.add(image_id)
            source_group_splits[source_group].add(split)
            image_sha_splits[image_sha].add(split)
            if image_sha in _HELD_OUT_SHA256:
                held_out.add(image_sha)

            if fish_present:
                if not boxes or review_status not in {"ACCEPTED", "TRAINING_READY"} or label_source not in _HUMAN_BOX_SOURCES:
                    row_blockers += 1
                    continue
                if any(not _valid_box(box) for box in boxes):
                    row_blockers += 1
                    continue
                row_review_ids = {str(item).strip() for item in review_record_ids}
                if len(row_review_ids) != len(review_record_ids):
                    row_blockers += 1
                if row_review_ids & all_review_ids:
                    row_blockers += 1
                all_review_ids.update(row_review_ids)
                provenance_by_id: dict[str, dict[str, Any]] = {}
                for provenance in review_provenance:
                    if not isinstance(provenance, dict):
                        row_blockers += 1
                        continue
                    record_id = str(provenance.get("review_record_id") or "").strip()
                    bbox_xywh = provenance.get("accepted_bbox_xywh")
                    try:
                        x, y, box_width, box_height = (float(value) for value in bbox_xywh)
                    except (TypeError, ValueError):
                        row_blockers += 1
                        continue
                    valid_provenance = (
                        bool(record_id)
                        and bool(str(provenance.get("reviewer_id") or "").strip())
                        and bool(str(provenance.get("reviewed_at") or "").strip())
                        and str(provenance.get("review_status") or "").upper() in {"ACCEPTED", "TRAINING_READY"}
                        and len(bbox_xywh) == 4
                        and all(0.0 <= value <= 1.0 for value in (x, y, box_width, box_height))
                        and box_width > 0.0
                        and box_height > 0.0
                        and x + box_width <= 1.00001
                        and y + box_height <= 1.00001
                    )
                    if not valid_provenance or record_id in provenance_by_id:
                        row_blockers += 1
                        continue
                    provenance_by_id[record_id] = {
                        "bbox_xyxy": [x, y, x + box_width, y + box_height],
                    }
                if set(provenance_by_id) != row_review_ids:
                    row_blockers += 1
                box_review_ids: set[str] = set()
                for box in boxes:
                    raw_box_review_ids = box.get("review_record_ids")
                    if not isinstance(raw_box_review_ids, list) or not raw_box_review_ids:
                        row_blockers += 1
                        break
                    current_ids = {str(item or "").strip() for item in raw_box_review_ids}
                    if "" in current_ids or not current_ids.issubset(row_review_ids):
                        row_blockers += 1
                        break
                    box_review_ids.update(current_ids)
                    for record_id in current_ids:
                        provenance_box = provenance_by_id.get(record_id, {}).get("bbox_xyxy")
                        if provenance_box is None or any(
                            not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-6)
                            for actual, expected in zip(box.get("bbox_xyxy") or [], provenance_box)
                        ):
                            row_blockers += 1
                            break
                if box_review_ids != row_review_ids:
                    row_blockers += 1
                    continue
                counts["human_reviewed_bbox_records"] += len(row_review_ids)
                counts["human_reviewed_bbox_images"] += 1
                split_boxes[split] += len(boxes)
            else:
                if (boxes or review_status != "NO_FISH_CONFIRMED" or not review_provenance
                        or label_source != "det_fish_negative_pool_human_confirmation"):
                    row_blockers += 1
                    continue
                negative_ids = {str(item).strip() for item in review_record_ids}
                valid_negative_ids = set()
                for provenance in review_provenance:
                    if not isinstance(provenance, dict):
                        row_blockers += 1
                        continue
                    record_id = str(provenance.get("review_record_id") or "").strip()
                    try:
                        pool_entry_id = int(provenance.get("pool_entry_id") or 0)
                        pool_event_id = int(provenance.get("pool_event_id") or 0)
                    except (TypeError, ValueError):
                        pool_entry_id = pool_event_id = 0
                    provenance_sha = str(provenance.get("pool_source_image_sha256") or "").strip()
                    if (
                        record_id
                        and str(provenance.get("reviewer_id") or "").strip()
                        and str(provenance.get("reviewed_at") or "").strip()
                        and str(provenance.get("review_status") or "").upper() == "NO_FISH_CONFIRMED"
                        and pool_entry_id > 0
                        and pool_event_id > 0
                        and str(provenance.get("pool_action") or "").upper() in {"ADD", "RE_ADD"}
                        and str(provenance.get("pool_source_kind") or "") in {"PRESENCE_NO_FISH", "DET_DS_V0_1"}
                        and str(provenance.get("pool_source_record_id") or "").strip()
                        and provenance_sha == image_sha
                    ):
                        valid_negative_ids.add(record_id)
                    else:
                        row_blockers += 1
                        continue
                    if pool_entry_id in all_negative_pool_entry_ids or pool_event_id in all_negative_pool_event_ids:
                        row_blockers += 1
                    all_negative_pool_entry_ids.add(pool_entry_id)
                    all_negative_pool_event_ids.add(pool_event_id)
                if valid_negative_ids != negative_ids:
                    row_blockers += 1
                if negative_ids & all_review_ids:
                    row_blockers += 1
                all_review_ids.update(negative_ids)
                counts["human_reviewed_negatives"] += 1
                counts[f"{split}_negative_images"] += 1

            split_images[split] += 1
            counts["human_reviewed_complete_scene_images"] += 1

    if row_blockers:
        blockers.append(f"逐图 Manifest 有 {row_blockers} 条无效或非人工确认记录")
    leaked_groups = [group for group, splits in source_group_splits.items() if len(splits) > 1]
    leaked_hashes = [digest for digest, splits in image_sha_splits.items() if len(splits) > 1]
    if leaked_groups or leaked_hashes:
        blockers.append(
            f"Manifest 存在来源组或重复原图跨集合泄漏（source_groups={len(leaked_groups)}, image_hashes={len(leaked_hashes)}）"
        )
    if held_out:
        blockers.append("Manifest 包含已冻结的 Case A、Case B、04-181200 或七图困难样本独立留出图片")

    manifest_review_ids: set[str] = set()
    if manifest is not None:
        for row in manifest:
            if not isinstance(row, dict) or row.get("fish_present") is not True:
                continue
            for record_id in row.get("review_record_ids") or []:
                record_id = str(record_id or "").strip()
                if record_id in manifest_review_ids:
                    blockers.append("人工 accepted_bbox review record 被重复计入多个 Manifest 图像")
                manifest_review_ids.add(record_id)

    source_record_count = _count(report.get("source_accepted_bbox_record_count"))
    excluded_record_count = _count(report.get("excluded_held_out_accepted_bbox_record_count"))
    source_negative_count = _count(report.get("source_human_negative_record_count"))
    active_pool_count = _count(report.get("source_negative_pool_active_member_count"))
    if active_pool_count is None or source_negative_count is None or active_pool_count != source_negative_count:
        blockers.append("活动 DET_FISH 负样本池成员数与构建读取数不一致")
    if report.get("negative_pool_member_errors"):
        blockers.append("DET_FISH 负样本池存在无效或冲突成员")
    audit_rows = report.get("source_image_audit")
    audit_by_review_id: dict[str, str] = {}
    audit_rows_valid = isinstance(audit_rows, list) and bool(audit_rows)
    if not audit_rows_valid:
        blockers.append("缺少逐条 Accepted Pool 原图字节审计记录")
    else:
        for row in audit_rows:
            if not isinstance(row, dict):
                audit_rows_valid = False
                continue
            record_id = str(row.get("review_record_id") or "").strip()
            source_sha = str(row.get("source_image_sha256") or "").strip()
            try:
                width, height = int(row.get("width") or 0), int(row.get("height") or 0)
                orientation = int(row.get("exif_orientation") or 0)
            except (TypeError, ValueError):
                width = height = orientation = 0
            valid_audit_row = (
                bool(record_id)
                and record_id not in audit_by_review_id
                and row.get("status") == "READ_OK"
                and _sha(source_sha)
                and width > 0
                and height > 0
                and 1 <= orientation <= 8
            )
            if not valid_audit_row:
                audit_rows_valid = False
                continue
            audit_by_review_id[record_id] = source_sha
    if source_record_count is not None and source_negative_count is not None and isinstance(audit_rows, list):
        expected_audit_rows = source_record_count + source_negative_count
        if len(audit_rows) != expected_audit_rows:
            audit_rows_valid = False
            blockers.append("原图字节审计行数与 Accepted Pool 正样本和人工无鱼来源记录数不一致")
    if manifest is not None:
        for row in manifest:
            if not isinstance(row, dict):
                continue
            image_sha = str(row.get("image_sha256") or row.get("source_image_sha256") or "")
            for record_id in row.get("review_record_ids") or []:
                record_id = str(record_id or "").strip()
                if audit_by_review_id.get(record_id) != image_sha:
                    audit_rows_valid = False
                    blockers.append("Manifest 中的原图 SHA 与对应审核记录的实际字节审计不一致")
                    break
    if not audit_rows_valid:
        blockers.append("逐条原图字节审计记录无效、重复或与 Manifest 不一致")
    if source_record_count is None or excluded_record_count is None:
        blockers.append("冻结报告缺少 Accepted Pool 原始记录数或留出记录排除数")
        source_records_match = False
    else:
        source_records_match = source_record_count == len(manifest_review_ids) + excluded_record_count
        if isinstance(source_reconciliation, dict):
            source_records_match = source_records_match and _count(source_reconciliation.get("accepted_pool_rows")) == source_record_count
        if not source_records_match:
            blockers.append("Accepted Pool 原始记录数与 Manifest/明确留出排除记录数不一致")
    excluded_negative_count = _count(report.get("excluded_held_out_negative_record_count"))
    if source_negative_count is None or excluded_negative_count is None or source_negative_count != counts["human_reviewed_negatives"] + excluded_negative_count:
        blockers.append("人工无鱼来源记录数与 Manifest/留出排除记录数不一致")

    for split in ("train", "val", "test"):
        counts[f"{split}_images"] = split_images[split]
        counts[f"{split}_boxes"] = split_boxes[split]
    reported_splits = report.get("splits") or {}
    if not isinstance(reported_splits, dict):
        reported_splits = {}
    report_splits_match = True
    for split in ("train", "val", "test"):
        reported = reported_splits.get(split) or {}
        if not isinstance(reported, dict):
            reported = {}
        reported_images = _count(reported.get("images"))
        reported_boxes = _count(reported.get("annotations", reported.get("boxes")))
        if reported_images != counts[f"{split}_images"] or reported_boxes != counts[f"{split}_boxes"]:
            report_splits_match = False
            blockers.append(f"冻结报告中的 {split} 图片数或 BBox 数与逐图 Manifest 不一致")
        reported_negatives = _count(reported.get("negative_images"))
        if reported_negatives != counts[f"{split}_negative_images"]:
            report_splits_match = False
            blockers.append(f"冻结报告中的 {split} 无鱼图片数与逐图 Manifest 不一致")

    negatives_match = _count(report.get("negatives")) == counts["human_reviewed_negatives"]
    if not negatives_match:
        blockers.append("冻结报告无鱼负样本数量与逐图人工审核记录不一致")
    if counts["train_images"] < 40:
        blockers.append("训练集完整场景图片不足 40 张")
    if counts["train_boxes"] < 30:
        blockers.append("训练集人工鱼体框不足 30 个")
    if counts["val_boxes"] < 3:
        blockers.append("验证集人工鱼体框不足 3 个")
    if counts["test_images"] < 1:
        blockers.append("独立测试集为空")
    if counts["human_reviewed_negatives"] < 1:
        blockers.append("缺少人工确认的无鱼负样本")

    # Preserve order while removing duplicate diagnostics.
    blockers = list(dict.fromkeys(blockers))
    return {
        "ready": not blockers,
        "blockers": blockers,
        "counts": counts,
        "counts_verified": bool(
            manifest is not None
            and manifest_counts_verified
            and row_blockers == 0
            and report_splits_match
            and negatives_match
            and source_records_match
            and audit_rows_valid
            and source_reconciliation_verified
        ),
    }


def validate_v02_materialized_dataset(
    dataset_root: str | Path,
    manifest_bytes: bytes,
    counts: dict[str, int],
) -> list[str]:
    """Verify the extracted YOLOX COCO package matches its reviewed manifest.

    The freeze manifest stores normalized accepted ``bbox_xyxy`` coordinates and
    each packaged image's ``file_name``/dimensions. This check prevents a
    correctly hashed report/manifest from blessing a different COCO annotation
    package inside the archive.
    """

    root = Path(dataset_root)
    blockers: list[str] = []
    try:
        internal_manifest = (root / "bootstrap_manifest.json").read_bytes()
    except OSError:
        return ["冻结数据包内部缺少 bootstrap_manifest.json"]
    if hashlib.sha256(internal_manifest).digest() != hashlib.sha256(manifest_bytes).digest():
        blockers.append("冻结数据包内部 Manifest 与外部 bootstrap_manifest.json 不一致")
    try:
        manifest = json.loads(internal_manifest.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return blockers + ["冻结数据包内部 Manifest 不是有效 JSON"]
    if not isinstance(manifest, list):
        return blockers + ["冻结数据包内部 Manifest 不是逐图记录数组"]

    rows_by_split: dict[str, dict[str, dict[str, Any]]] = {
        split: {} for split in ("train", "val", "test")
    }
    for row in manifest:
        if not isinstance(row, dict):
            blockers.append("冻结数据包 Manifest 含无效记录")
            continue
        split = str(row.get("split") or "").lower()
        filename = str(row.get("file_name") or "")
        if split not in rows_by_split or not filename or Path(filename).name != filename:
            blockers.append("冻结数据包 Manifest 含无效 split 或 file_name")
            continue
        if filename in rows_by_split[split]:
            blockers.append(f"冻结数据包 Manifest 的 {split} 包含重复 file_name")
            continue
        rows_by_split[split][filename] = row

    for split in ("train", "val", "test"):
        annotations_path = root / "annotations" / f"instances_{split}2017.json"
        try:
            coco = json.loads(annotations_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            blockers.append(f"冻结数据包缺少有效 COCO {split} 标注文件")
            continue
        images = coco.get("images") if isinstance(coco, dict) else None
        annotations = coco.get("annotations") if isinstance(coco, dict) else None
        if not isinstance(images, list) or not isinstance(annotations, list):
            blockers.append(f"COCO {split} 标注文件缺少 images/annotations 数组")
            continue
        categories = coco.get("categories") or []
        if not any(
            isinstance(category, dict)
            and str(category.get("id") or "") == "1"
            and str(category.get("name") or "").lower() == "fish"
            for category in categories
        ):
            blockers.append(f"COCO {split} 类别表缺少 id=1 的 fish 类")
        images_by_name = {
            str(item.get("file_name") or ""): item
            for item in images
            if isinstance(item, dict) and item.get("file_name")
        }
        expected_rows = rows_by_split[split]
        if len(images_by_name) != len(images) or set(images_by_name) != set(expected_rows):
            blockers.append(f"COCO {split} 图像清单与人工审核 Manifest 不一致")
        expected_box_count = int(counts.get(f"{split}_boxes") or 0)
        if len(annotations) != expected_box_count:
            blockers.append(f"COCO {split} BBox 数与人工审核 Manifest 不一致")

        image_ids: dict[int, dict[str, Any]] = {}
        for name, image in images_by_name.items():
            try:
                image_id = int(image["id"])
                width = int(image["width"])
                height = int(image["height"])
                if width <= 0 or height <= 0:
                    raise ValueError
                if image_id in image_ids:
                    blockers.append(f"COCO {split} 含重复 image id")
                image_ids[image_id] = {"name": name, "width": width, "height": height}
                if not (root / f"{split}2017" / name).is_file():
                    blockers.append(f"COCO {split} 图像文件缺失：{name}")
                else:
                    image_path = root / f"{split}2017" / name
                    expected_sha = str((expected_rows.get(name) or {}).get("materialized_image_sha256") or "")
                    if not _sha(expected_sha) or hashlib.sha256(image_path.read_bytes()).hexdigest() != expected_sha:
                        blockers.append(f"COCO {split} 图像文件 SHA-256 与 Manifest 不一致：{name}")
                row = expected_rows.get(name)
                if row and (int(row.get("width") or 0) != width or int(row.get("height") or 0) != height):
                    blockers.append(f"COCO {split} 图像尺寸与 Manifest 不一致：{name}")
            except (KeyError, TypeError, ValueError):
                blockers.append(f"COCO {split} 图像记录无效：{name}")

        expected_boxes: dict[str, list[list[float]]] = {}
        for name, row in expected_rows.items():
            raw_boxes = row.get("boxes")
            if not isinstance(raw_boxes, list):
                blockers.append(f"Manifest 图像缺少 boxes 数组：{name}")
                continue
            expected_boxes[name] = []
            for box in raw_boxes:
                coords = box.get("bbox_xyxy") if isinstance(box, dict) else None
                try:
                    xyxy = [float(value) for value in coords]
                    if len(xyxy) != 4 or not all(math.isfinite(value) for value in xyxy):
                        raise ValueError
                    expected_boxes[name].append(xyxy)
                except (TypeError, ValueError):
                    blockers.append(f"Manifest BBox 无效：{name}")

        observed_by_name: dict[str, list[list[float]]] = defaultdict(list)
        for annotation in annotations:
            try:
                image_id = int(annotation["image_id"])
                image = image_ids[image_id]
                bbox = [float(value) for value in annotation["bbox"]]
                if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
                    raise ValueError
                if int(annotation.get("category_id", -1)) != 1 or int(annotation.get("iscrowd", 0)) != 0:
                    raise ValueError
                x, y, width, height = bbox
                if width <= 0 or height <= 0:
                    raise ValueError
                observed_by_name[image["name"]].append(
                    [x / image["width"], y / image["height"], (x + width) / image["width"], (y + height) / image["height"]]
                )
            except (KeyError, TypeError, ValueError):
                blockers.append(f"COCO {split} 含无效或孤立的鱼体标注")

        for name, boxes in expected_boxes.items():
            observed = observed_by_name.get(name, [])
            if len(observed) != len(boxes):
                blockers.append(f"COCO {split} 图像 BBox 数与人工审核 Manifest 不一致：{name}")
                continue
            unmatched = list(observed)
            for expected in boxes:
                match = next(
                    (index for index, actual in enumerate(unmatched)
                     if all(math.isclose(a, b, rel_tol=0.0, abs_tol=0.001) for a, b in zip(expected, actual))),
                    None,
                )
                if match is None:
                    blockers.append(f"COCO {split} BBox 坐标与人工审核 Manifest 不一致：{name}")
                    break
                unmatched.pop(match)

    return list(dict.fromkeys(blockers))
