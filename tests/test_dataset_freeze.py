import json
from collections import defaultdict

import pytest
from sqlalchemy import select

from app import flywheel
from app.db import Base
from app.dedupe import ImageFingerprint
from app.main import app
from app.models import Batch, DatasetVersion, ImageAsset
from app.presence import FishPresenceResult, effective_status


class FakeBlob:
    def __init__(self, bucket, name):
        self.bucket = bucket
        self.name = name

    def exists(self, *args, **kwargs):
        return self.name in self.bucket.objects

    def upload_from_string(self, data, **kwargs):
        if kwargs.get("if_generation_match") == 0 and self.exists():
            raise RuntimeError(f"object already exists: {self.name}")
        self.bucket.objects[self.name] = data

    def download_as_text(self, encoding="utf-8"):
        value = self.bucket.objects[self.name]
        return value.decode(encoding) if isinstance(value, bytes) else value


class FakeBucket:
    def __init__(self, name):
        self.name = name
        self.objects = {}

    def blob(self, name):
        return FakeBlob(self, name)


class FakeStorageClient:
    def __init__(self, bucket):
        self._bucket = bucket
        self.calls = 0

    def bucket(self, name):
        assert name == self._bucket.name
        self.calls += 1
        return self._bucket


def add_batch(db, batch_id="BATCH_TEST"):
    db.add(
        Batch(
            batch_id=batch_id,
            source="pytest",
            image_count=0,
            manifest_uri=f"gs://test-bucket/raw/{batch_id}/manifest.csv",
            raw_uri=f"gs://test-bucket/raw/{batch_id}/",
            status="INGESTED",
        )
    )
    db.flush()


def add_image(db, image_id, species, *, batch_id="BATCH_TEST", group_id=None, review_status="approved"):
    row = ImageAsset(
        batch_id=batch_id,
        image_id=image_id,
        file_name=f"{image_id}.jpg",
        object_name=f"raw/{batch_id}/{image_id}.jpg",
        gcs_uri=f"gs://test-bucket/raw/{batch_id}/{image_id}.jpg",
        claimed_species=species,
        truth_species=species,
        review_status=review_status,
        group_id=group_id,
    )
    db.add(row)
    db.flush()
    return row


def add_presence(db, image, status, *, fish_count=0, evidence=None):
    db.add(
        FishPresenceResult(
            image_asset_id=image.id,
            batch_id=image.batch_id,
            status=status,
            fish_count=fish_count,
            evidence_json=json.dumps(evidence, ensure_ascii=False) if evidence else None,
        )
    )


def patch_storage(monkeypatch, fake_client):
    monkeypatch.setattr(flywheel.storage, "Client", lambda: fake_client)


def comparable_preview(data):
    keys = ("eligible_images", "species_count", "species_counts", "split_counts", "excluded", "class_map", "parent_version")
    return {key: data[key] for key in keys}


def test_preview_and_freeze_share_eligibility_and_do_not_write_on_preview(db, monkeypatch):
    add_batch(db)
    active = add_image(db, "active", "鲫鱼", group_id="EVENT_A")
    uncertain = add_image(db, "uncertain", "鲤鱼", group_id="EVENT_A")
    no_fish = add_image(db, "no-fish", "草鱼")
    multi = add_image(db, "multi", "草鱼")
    candidate = add_image(db, "candidate", "候选鱼")
    duplicate = add_image(db, "duplicate", "黑鱼")
    override = add_image(db, "override", "黄骨鱼")
    add_presence(db, active, "single_fish", fish_count=1)
    add_presence(db, uncertain, "uncertain")
    add_presence(db, no_fish, "no_fish")
    add_presence(db, multi, "multi_fish", fish_count=2)
    add_presence(db, override, "no_fish", evidence={"machine_status": "no_fish", "human_override": "single_fish"})
    db.add(
        ImageFingerprint(
            image_asset_id=duplicate.id,
            batch_id=duplicate.batch_id,
            sha256="a" * 64,
            phash_json="[]",
            dhash="0" * 16,
            crop_hash="",
            histogram_json="[]",
            width=100,
            height=100,
            duplicate_group="DUP_1",
            is_representative=False,
            duplicate_kind="near",
        )
    )
    db.commit()
    candidate_catalog = flywheel.create_species_candidate(db, "候选鱼")
    assert candidate_catalog["status"] == "candidate"

    fake_bucket = FakeBucket("test-bucket")
    fake_client = FakeStorageClient(fake_bucket)
    patch_storage(monkeypatch, fake_client)
    preview_client_calls_before = fake_client.calls
    preview = flywheel.preview_cumulative_dataset(db, dataset_version="DS_TEST_v0.1", seed=7)
    assert fake_client.calls == preview_client_calls_before, "preview without a parent must not touch GCS"
    assert db.query(DatasetVersion).count() == 0
    assert preview["eligible_images"] == 3
    assert preview["species_counts"] == {"鲫鱼": 1, "鲤鱼": 1, "黄骨鱼": 1}
    assert preview["excluded"]["no_fish"] == 1
    assert preview["excluded"]["multi_fish"] == 1
    assert preview["excluded"]["near_duplicate"] == 1
    assert preview["excluded"]["inactive_species"] == 1
    override_presence = db.scalar(select(FishPresenceResult).where(FishPresenceResult.image_asset_id == override.id))
    assert effective_status(override_presence) == "single_fish"

    frozen = flywheel.freeze_cumulative_dataset(
        db,
        dataset_version="DS_TEST_v0.1",
        git_commit="test-sha",
        seed=7,
        bucket_name="test-bucket",
    )
    assert comparable_preview(frozen) == comparable_preview(preview)
    assert db.query(DatasetVersion).count() == 1
    assert set(fake_bucket.objects) == {
        "datasets/DS_TEST_v0.1/dataset_manifest.csv",
        "datasets/DS_TEST_v0.1/class_map.json",
        "datasets/DS_TEST_v0.1/dataset.json",
    }


def test_same_group_is_assigned_to_one_split(db):
    add_batch(db)
    add_image(db, "a", "鲫鱼", group_id="CATCH_1")
    add_image(db, "b", "鲫鱼", group_id="CATCH_1")
    add_image(db, "c", "鲤鱼", group_id="CATCH_2")
    db.commit()

    plan = flywheel._prepare_dataset_freeze(db, dataset_version="DS_GROUP_v0.1", git_commit="test", seed=11)
    splits = defaultdict(set)
    for row in plan.rows:
        splits[row["group_id"]].add(row["split"])
    assert splits["CATCH_1"] == {next(row["split"] for row in plan.rows if row["group_id"] == "CATCH_1")}
    assert all(len(values) == 1 for values in splits.values())


def test_parent_indices_are_inherited_and_new_active_species_append(db, monkeypatch):
    add_batch(db)
    add_image(db, "old", "鲫鱼")
    new_catalog = flywheel.create_species_candidate(db, "新鱼")
    flywheel.set_species_status(db, new_catalog["species_key"], "active")
    add_image(db, "new", "新鱼")
    parent = DatasetVersion(
        dataset_version="DS_PARENT_v0.1",
        manifest_uri="gs://test-bucket/datasets/DS_PARENT_v0.1/dataset_manifest.csv",
        class_map_uri="gs://test-bucket/datasets/DS_PARENT_v0.1/class_map.json",
        git_commit="parent",
        status="FROZEN",
    )
    db.add(parent)
    db.commit()

    fake_bucket = FakeBucket("test-bucket")
    fake_bucket.objects["datasets/DS_PARENT_v0.1/class_map.json"] = json.dumps({
        "dataset_version": "DS_PARENT_v0.1",
        "classes": [
            {"class_index": 3, "species_key": "crucian_carp", "common_name_zh": "旧名"},
        ],
    }, ensure_ascii=False)
    patch_storage(monkeypatch, FakeStorageClient(fake_bucket))
    preview = flywheel.preview_cumulative_dataset(
        db,
        dataset_version="DS_CHILD_v0.1",
        parent_version="DS_PARENT_v0.1",
        bucket_name="test-bucket",
    )
    classes = preview["class_map"]["classes"]
    by_key = {item["species_key"]: item["class_index"] for item in classes}
    assert by_key["crucian_carp"] == 3
    assert by_key[new_catalog["species_key"]] == 4


def test_dataset_routes_are_canonical_and_no_dataset_items_table():
    paths = app.openapi()["paths"]
    assert "/datasets" in paths
    assert "/api/datasets" in paths
    assert "/api/datasets/summary" in paths
    assert "/api/datasets/freeze/preview" in paths
    assert "/api/datasets/freeze" in paths
    assert "/api/dataset-freeze/preview" not in paths
    assert "dataset_items" not in Base.metadata.tables


def test_existing_dataset_version_cannot_be_overwritten(db, monkeypatch):
    add_batch(db)
    add_image(db, "only", "鲫鱼")
    db.commit()
    fake_bucket = FakeBucket("test-bucket")
    patch_storage(monkeypatch, FakeStorageClient(fake_bucket))
    flywheel.freeze_cumulative_dataset(db, dataset_version="DS_IMMUTABLE_v0.1", git_commit="one", bucket_name="test-bucket")
    object_snapshot = dict(fake_bucket.objects)
    with pytest.raises(ValueError, match="already registered"):
        flywheel.freeze_cumulative_dataset(db, dataset_version="DS_IMMUTABLE_v0.1", git_commit="two", bucket_name="test-bucket")
    assert fake_bucket.objects == object_snapshot
