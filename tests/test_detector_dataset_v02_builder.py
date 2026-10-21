import hashlib
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from trainer import build_detector_dataset_v02 as builder
from trainer.detector_dataset_gate import validate_v02_dataset


def png_bytes(index: int) -> bytes:
    image = Image.new("RGB", (64, 48), (index % 256, (index * 3) % 256, (index * 7) % 256))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def fixture_records():
    sources = {}
    positives = []
    negatives = []
    for index in range(60):
        uri = f"memory://scene/{index}"
        sources[uri] = png_bytes(index)
        common = {
            "review_record_id": f"review-{index}",
            "batch_id": f"batch-{index}",
            "image_id": f"image-{index}",
            "source_uri": uri,
            "source_group_id": "capture-group-a" if index in {0, 1} else f"capture-group-{index}",
            "source_group_basis": "test",
            "reviewer_id": "fixture-reviewer",
            "reviewed_at": "2026-10-20T00:00:00Z",
            "training_eligibility_basis": "synthetic-test-fixture-only",
        }
        if index < 54:
            positives.append({
                **common,
                "kind": "positive",
                "bbox_xywh": [0.1, 0.15, 0.4, 0.5],
                "review_status": "ACCEPTED",
            })
        else:
            pool_source_sha = hashlib.sha256(png_bytes(index)).hexdigest()
            negatives.append({
                **common,
                "review_record_id": f"det_fish_negative_pool_entry:{index}:event:{index}",
                "kind": "negative",
                "review_status": "NO_FISH_CONFIRMED",
                "pool_entry_id": index,
                "pool_event_id": index,
                "pool_action": "ADD",
                "pool_source_kind": "PRESENCE_NO_FISH",
                "pool_source_record_id": f"presence-{index}",
                "pool_source_image_sha256": pool_source_sha,
                "pool_source_provider": "fixture",
                "pool_source_model_version": "fixture-v0",
                "pool_source_status": "no_fish",
                "pool_confirmation_snapshot": {"det_fish_manual_confirmation": True},
            })
    records = {
        "positives": positives,
        "negatives": negatives,
        "source_reconciliation": {
            "status": "PASS",
            "problems": [],
            "accepted_pool_rows": 54,
            "manifest_sha256": "1" * 64,
            "metadata_sha256": "2" * 64,
            "manifest_uri": "gs://test-bucket/datasets/accepted_pool/manifest.csv",
            "metadata_uri": "gs://test-bucket/datasets/accepted_pool/metadata.json",
        },
        "accepted_pool_metadata_summary": {"source": "synthetic fixture"},
    }
    return records, sources


def test_source_group_fallback_uses_asset_id_not_import_batch():
    first = SimpleNamespace(id=101, group_id=None, batch_id="bulk-import-20261007")
    second = SimpleNamespace(id=102, group_id=None, batch_id="bulk-import-20261007")

    assert builder._source_group(first) == ("asset:101", "ImageAsset.id_fallback")
    assert builder._source_group(second) == ("asset:102", "ImageAsset.id_fallback")
    assert builder._source_group(first) != builder._source_group(second)


def test_stable_split_balances_positive_and_negative_classes_without_splitting_groups():
    groups = {f"component-{index}": {f"sha-{index}"} for index in range(1000)}
    fish_presence = {f"sha-{index}": index < 800 for index in range(1000)}

    assignments = builder._stable_split(groups, fish_presence)

    for fish_present, total in ((True, 800), (False, 200)):
        counts = {
            split: sum(
                1 for component, split_value in assignments.items()
                if split_value == split
                and fish_presence[next(iter(groups[component]))] is fish_present
            )
            for split in ("train", "val", "test")
        }
        assert abs(counts["train"] / total - 0.80) <= 0.03, counts
        assert abs(counts["val"] / total - 0.10) <= 0.03, counts
        assert abs(counts["test"] / total - 0.10) <= 0.03, counts


def test_builder_freezes_full_scenes_with_provenance_and_deterministic_archive(tmp_path: Path):
    records, sources = fixture_records()
    first = tmp_path / "first"
    second = tmp_path / "second"

    result = builder.build_dataset(records, first, fetch_source=sources.__getitem__)
    repeated = builder.build_dataset(records, second, fetch_source=sources.__getitem__)

    assert result["ready"] is True, result["blockers"]
    assert repeated["ready"] is True
    report = result["report"]
    assert report["status"] == "FROZEN_READY_FOR_TRAINING"
    assert report["source_accepted_bbox_record_count"] == 54
    assert report["unique_positive_images"] == 54
    assert report["unique_bbox_count"] == 54
    assert report["negatives"] == 6
    assert report["actual_source_images_read"] == 60
    assert report["source_uri_failures"] == 0
    assert report["source_image_audit_verified"] is True
    assert report["source_image_audit_summary"]["source_record_rows"] == 60
    assert all(item["status"] == "READ_OK" for item in report["source_image_audit"])
    assert hashlib.sha256((first / "DET_DS_v0.2.tar.gz").read_bytes()).hexdigest() == report["archive_sha256"]
    assert report["archive_sha256"] == repeated["report"]["archive_sha256"]

    manifest_bytes = (first / "bootstrap_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    assert len(manifest) == 60
    assert all(row["image_kind"] == "complete_scene" for row in manifest)
    assert all(row["file_name"].endswith(".png") for row in manifest)
    positive_rows = [row for row in manifest if row["fish_present"]]
    negative_rows = [row for row in manifest if not row["fish_present"]]
    assert len({record for row in positive_rows for record in row["review_record_ids"]}) == 54
    assert len(negative_rows) == 6
    negative_files_by_split = {
        split: {row["file_name"] for row in negative_rows if row["split"] == split}
        for split in ("train", "val", "test")
    }
    with tarfile.open(first / "DET_DS_v0.2.tar.gz", "r:gz") as archive:
        for split in ("train", "val", "test"):
            annotation_file = archive.extractfile(f"DET_DS_v0.2/annotations/instances_{split}2017.json")
            assert annotation_file is not None
            coco = json.loads(annotation_file.read())
            image_ids_by_name = {row["file_name"]: row["id"] for row in coco["images"]}
            assert negative_files_by_split[split].issubset(image_ids_by_name)
            negative_image_ids = {image_ids_by_name[name] for name in negative_files_by_split[split]}
            assert not any(annotation["image_id"] in negative_image_ids for annotation in coco["annotations"])
    capture_splits = {row["split"] for row in positive_rows if "capture-group-a" in row["source_group_ids"]}
    assert len(capture_splits) == 1

    gate = validate_v02_dataset(report, manifest_bytes, report["archive_sha256"])
    assert gate["ready"] is True, gate["blockers"]

    with tarfile.open(first / "DET_DS_v0.2.tar.gz", "r:gz") as archive:
        names = archive.getnames()
    assert any(name.startswith("DET_DS_v0.2/train2017/") for name in names)
    assert any(name.endswith("annotations/instances_val2017.json") for name in names)
    assert not any("crop" in Path(name).name for name in names)


def test_builder_blocks_without_human_negative_or_on_missing_scene(tmp_path: Path):
    records, sources = fixture_records()
    records["negatives"] = []
    no_negative = builder.build_dataset(records, tmp_path / "no-negative", fetch_source=sources.__getitem__)
    assert no_negative["ready"] is False
    assert not (tmp_path / "no-negative" / "DET_DS_v0.2.tar.gz").exists()
    assert any("缺少人工确认的无鱼负样本" in item for item in no_negative["blockers"])

    records, sources = fixture_records()
    sources.pop("memory://scene/7")
    missing = builder.build_dataset(records, tmp_path / "missing-source", fetch_source=sources.__getitem__)
    assert missing["ready"] is False
    assert missing["report"]["source_uri_failures"] >= 1
    assert not (tmp_path / "missing-source" / "bootstrap_manifest.json").exists()


def test_builder_audits_all_source_objects_even_when_review_reconciliation_blocks(tmp_path: Path):
    records, sources = fixture_records()
    records["source_reconciliation"]["status"] = "BLOCKED"
    records["source_reconciliation"]["problems"] = ["synthetic mismatch"]

    result = builder.build_dataset(records, tmp_path / "blocked-review", fetch_source=sources.__getitem__)

    assert result["ready"] is False
    assert result["report"]["actual_source_images_read"] == 60
    assert result["report"]["source_image_audit_verified"] is True
    assert not (tmp_path / "blocked-review" / "DET_DS_v0.2.tar.gz").exists()
    assert not (tmp_path / "blocked-review" / "bootstrap_manifest.json").exists()


def test_builder_does_not_overwrite_existing_output(tmp_path: Path):
    records, sources = fixture_records()
    target = tmp_path / "frozen"
    result = builder.build_dataset(records, target, fetch_source=sources.__getitem__)
    assert result["ready"]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        builder.build_dataset(records, target, fetch_source=sources.__getitem__)


def test_accepted_pool_reconciliation_uses_source_scene_bbox_and_ignores_detector_bbox():
    record = {
        "review_record_id": "10",
        "source_uri": "gs://source/full.jpg",
        "bbox_xywh": [0.1, 0.1, 0.5, 0.5],
        "review_status": "ACCEPTED",
        "reviewer_id": "human-1",
        "reviewed_at": "2026-10-20T00:00:00Z",
        "training_eligibility_basis": "database eligibility fixture",
        "source_group_id": "capture-1",
        "source_group_basis": "ImageAsset.group_id",
    }
    row = {
        "source_review_id": "10",
        "review_status": "ACCEPTED",
        "bbox_source": "accepted_bbox",
        "source_image_gcs_uri": "gs://source/full.jpg",
        "source_image_exists": "true",
        "accepted_bbox": "[0.1,0.1,0.5,0.5]",
        "bbox": "[0.1,0.1,0.5,0.5]",
        "detector_bbox": "[0.2,0.2,0.2,0.2]",
        "pixel_bbox": "[10,8,60,48]",
        "source_size": "[100,80]",
        "crop_left": "10",
        "crop_top": "8",
        "crop_right": "60",
        "crop_bottom": "48",
    }
    source_files = {"metadata_current_accepted_bbox_count": 1}
    result = builder.reconcile_accepted_pool_export([row], {"positives": [record], "negatives": []}, source_files)
    assert result["source_reconciliation"]["status"] == "PASS", result["source_reconciliation"]["problems"]
    assert result["source_reconciliation"]["detector_bbox_nonempty_ignored_count"] == 1
    assert result["positives"][0]["bbox_xywh"] == [0.1, 0.1, 0.5, 0.5]
    assert result["positives"][0]["expected_pixel_bbox"] == [10, 8, 60, 48]


def test_accepted_pool_reconciliation_blocks_bbox_or_source_mismatch():
    record = {
        "review_record_id": "10", "source_uri": "gs://source/full.jpg",
        "bbox_xywh": [0.1, 0.1, 0.5, 0.5], "review_status": "ACCEPTED",
        "reviewer_id": "human-1", "reviewed_at": "2026-10-20T00:00:00Z",
        "training_eligibility_basis": "database eligibility fixture", "source_group_id": "g",
    }
    row = {
        "source_review_id": "10", "review_status": "ACCEPTED", "bbox_source": "accepted_bbox",
        "source_image_gcs_uri": "gs://other/wrong.jpg", "source_image_exists": "true",
        "accepted_bbox": "[0.2,0.1,0.5,0.5]", "bbox": "[0.2,0.1,0.5,0.5]",
        "pixel_bbox": "[20,8,70,48]", "source_size": "[100,80]",
        "crop_left": "20", "crop_top": "8", "crop_right": "70", "crop_bottom": "48",
    }
    result = builder.reconcile_accepted_pool_export([row], {"positives": [record]}, {})
    assert result["source_reconciliation"]["status"] == "BLOCKED"
    assert not result["positives"]


def test_accepted_pool_reconciliation_reports_invalid_accepted_bbox_count():
    record = {
        "review_record_id": "10", "source_uri": "gs://source/full.jpg",
        "bbox_xywh": [0.1, 0.1, 0.5, 0.5], "review_status": "ACCEPTED",
        "reviewer_id": "human-1", "reviewed_at": "2026-10-20T00:00:00Z",
        "training_eligibility_basis": "database eligibility fixture", "source_group_id": "g",
    }
    row = {
        "source_review_id": "10", "review_status": "ACCEPTED", "bbox_source": "accepted_bbox",
        "source_image_gcs_uri": "gs://source/full.jpg", "source_image_exists": "true",
        "accepted_bbox": "[0.8,0.8,0.5,0.5]", "bbox": "[0.8,0.8,0.5,0.5]",
        "pixel_bbox": "[80,64,130,104]", "source_size": "[100,80]",
        "crop_left": "80", "crop_top": "64", "crop_right": "130", "crop_bottom": "104",
    }
    result = builder.reconcile_accepted_pool_export([row], {"positives": [record]}, {})

    assert result["source_reconciliation"]["status"] == "BLOCKED"
    assert result["source_reconciliation"]["invalid_accepted_bbox_count"] == 1
    assert not result["positives"]


class FakeBlob:
    def __init__(self, bucket, name):
        self.bucket = bucket
        self.name = name

    def exists(self, _client=None):
        return self.name in self.bucket.objects

    def download_as_bytes(self):
        return self.bucket.objects[self.name]

    def upload_from_filename(self, filename, content_type=None, if_generation_match=None):
        assert if_generation_match == 0
        self.bucket.upload_order.append(self.name)
        self.bucket.objects[self.name] = Path(filename).read_bytes()


class FakeBucket:
    def __init__(self):
        self.objects = {}
        self.upload_order = []

    def blob(self, name):
        return FakeBlob(self, name)


class FakeStorageClient:
    def __init__(self):
        self._bucket = FakeBucket()

    def bucket(self, _name):
        return self._bucket


def test_gcs_publish_is_conditional_idempotent_and_report_last(tmp_path: Path):
    records, sources = fixture_records()
    result = builder.build_dataset(records, tmp_path / "freeze", fetch_source=sources.__getitem__)
    assert result["ready"]
    client = FakeStorageClient()
    first = builder.publish_gcs(tmp_path / "freeze", "fake-bucket", client=client)
    assert set(first.values()) == {"CREATED"}
    assert client._bucket.upload_order[-1].endswith("bootstrap_report.json")
    second = builder.publish_gcs(tmp_path / "freeze", "fake-bucket", client=client)
    assert set(second.values()) == {"UNCHANGED"}
    existing = next(iter(client._bucket.objects))
    client._bucket.objects[existing] = b"different frozen bytes"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        builder.publish_gcs(tmp_path / "freeze", "fake-bucket", client=client)


def test_gcs_publish_revalidates_local_manifest_and_archive_hashes(tmp_path: Path):
    records, sources = fixture_records()
    output = tmp_path / "freeze"
    result = builder.build_dataset(records, output, fetch_source=sources.__getitem__)
    assert result["ready"]
    with (output / "DET_DS_v0.2.tar.gz").open("ab") as handle:
        handle.write(b"altered")
    client = FakeStorageClient()
    with pytest.raises(ValueError, match="local frozen evidence failed validation"):
        builder.publish_gcs(output, "fake-bucket", client=client)
    assert client._bucket.objects == {}


def test_gcs_publish_treats_same_frozen_content_with_new_timestamps_as_unchanged(tmp_path: Path):
    records, sources = fixture_records()
    output = tmp_path / "freeze"
    first = builder.build_dataset(records, output, fetch_source=sources.__getitem__)
    assert first["ready"]
    client = FakeStorageClient()
    builder.publish_gcs(output, "fake-bucket", client=client)

    report_name = "detector-datasets/DET_DS_v0.2/bootstrap_report.json"
    frozen_report = json.loads(client._bucket.objects[report_name])
    frozen_report["created_at"] = "2026-10-19T00:00:00+00:00"
    client._bucket.objects[report_name] = json.dumps(frozen_report, sort_keys=True).encode()

    assert set(builder.publish_gcs(output, "fake-bucket", client=client).values()) == {"UNCHANGED"}
