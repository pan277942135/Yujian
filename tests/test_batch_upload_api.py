import csv
import base64
import hashlib
import io

import pytest

from app.batch_upload_api import (
    _build_fish_manifest,
    _safe_relative_path,
    _upload_resumable_blob,
    _validate_batch_id,
)


class FakeBlob:
    def __init__(self, data=None):
        self.data = data
        self.size = len(data) if data is not None else None
        self.md5_hash = (
            base64.b64encode(hashlib.md5(data).digest()).decode("ascii") if data is not None else None
        )
        self.uploads = 0

    def exists(self, _client=None):
        return self.data is not None

    def reload(self, _client=None):
        return None

    def download_as_bytes(self):
        return self.data

    def upload_from_string(self, data, **_kwargs):
        self.data = data
        self.size = len(data)
        self.md5_hash = base64.b64encode(hashlib.md5(data).digest()).decode("ascii")
        self.uploads += 1


class FakeBucket:
    def __init__(self, blob):
        self._blob = blob

    def blob(self, _name):
        return self._blob


def test_collection_manifest_is_adapted_to_factory_contract():
    source = """image_id,file_name,species_key,species_name,source_url,source_platform,notes\nP5_00001,bighead_carp_0001.jpg,bighead_carp,鳙鱼,https://example.com/1,douyin,人工审核通过\nP5_00002,grass_carp_0001.jpg,grass_carp,草鱼,https://example.com/2,douyin,人工审核通过\n"""
    text, count = _build_fish_manifest(source)
    rows = list(csv.DictReader(io.StringIO(text)))

    assert count == 2
    assert rows[0]["image_id"] == "P5_00001"
    assert rows[0]["file_name"] == "bighead_carp_0001.jpg"
    assert rows[0]["claimed_species"] == "鳙鱼"
    assert rows[1]["claimed_species"] == "草鱼"
    assert rows[0]["species_key"] == "bighead_carp"
    assert rows[0]["source_url"] == "https://example.com/1"


def test_existing_factory_manifest_remains_valid():
    source = "image_id,file_name,claimed_species\nA001,a.jpg,鲫鱼\n"
    text, count = _build_fish_manifest(source)
    row = next(csv.DictReader(io.StringIO(text)))
    assert count == 1
    assert row["claimed_species"] == "鲫鱼"


def test_manifest_requires_species_truth_claim_column():
    with pytest.raises(ValueError, match="species_name"):
        _build_fish_manifest("image_id,file_name\nA001,a.jpg\n")


def test_upload_paths_reject_parent_traversal():
    with pytest.raises(ValueError):
        _safe_relative_path("../secret.txt")
    assert _safe_relative_path("metadata/manifest.csv") == "metadata/manifest.csv"


def test_batch_id_contract():
    assert _validate_batch_id("BATCH_20260901_P5_001") == "BATCH_20260901_P5_001"
    with pytest.raises(ValueError):
        _validate_batch_id("P5_001")


def test_resumable_upload_skips_existing_object_with_same_hash():
    blob = FakeBlob(b"same bytes")
    result = _upload_resumable_blob(FakeBucket(blob), object(), "incoming/BATCH/x.jpg", b"same bytes", content_type="image/jpeg")

    assert result["status"] == "SKIP"
    assert result["skipped"] is True
    assert blob.uploads == 0


def test_resumable_upload_reports_conflict_without_overwrite():
    blob = FakeBlob(b"old bytes")
    result = _upload_resumable_blob(FakeBucket(blob), object(), "incoming/BATCH/x.jpg", b"new bytes", content_type="image/jpeg")

    assert result["status"] == "CONFLICT"
    assert result["conflict"] is True
    assert blob.data == b"old bytes"
    assert blob.uploads == 0
