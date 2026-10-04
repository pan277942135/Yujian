import asyncio
import base64
import csv
import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone

from fastapi import UploadFile
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import app.batch_upload_api as upload_api
import app.exact_dedupe as exact_dedupe
import app.factory as factory
from app.db import Base
from app.models import Batch, GlobalDuplicateAudit, GlobalImageContent, ImageAsset


class RecoveryBlob:
    def __init__(self, name: str, data: bytes | None = None):
        self.name = name
        self.data = data
        self.generation = 1 if data is not None else None
        self.uploads = 0

    @property
    def size(self):
        return len(self.data) if self.data is not None else None

    @property
    def md5_hash(self):
        if self.data is None:
            return None
        return base64.b64encode(hashlib.md5(self.data).digest()).decode("ascii")

    def exists(self, _client=None):
        return self.data is not None

    def reload(self, _client=None):
        return None

    def download_as_bytes(self, **_kwargs):
        if self.data is None:
            raise FileNotFoundError(self.name)
        return self.data

    def download_as_text(self, encoding="utf-8", **_kwargs):
        return self.download_as_bytes().decode(encoding)

    def upload_from_string(self, data, **kwargs):
        if kwargs.get("if_generation_match") == 0 and self.data is not None:
            raise RuntimeError("already exists")
        self.data = data.encode("utf-8") if isinstance(data, str) else bytes(data)
        self.generation = (self.generation or 0) + 1
        self.uploads += 1


class RecoveryBucket:
    def __init__(self):
        self.blobs: dict[str, RecoveryBlob] = {}

    def blob(self, name):
        return self.blobs.setdefault(name, RecoveryBlob(name))

    def put(self, name: str, data: bytes):
        blob = self.blob(name)
        blob.data = data
        blob.generation = 1
        return blob

    def object_names(self):
        return {name for name, blob in self.blobs.items() if blob.data is not None}


class RecoveryClient:
    def __init__(self, bucket: RecoveryBucket):
        self._bucket = bucket

    def bucket(self, _name):
        return self._bucket

    def list_blobs(self, _bucket_name, prefix="", max_results=None, delimiter=None):
        rows = [
            blob
            for name, blob in sorted(self._bucket.blobs.items())
            if name.startswith(prefix) and blob.data is not None
        ]
        if max_results is not None:
            rows = rows[:max_results]
        return rows


@pytest.fixture
def recovery_db(monkeypatch):
    import app.platform.models  # noqa: F401

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    monkeypatch.setattr(upload_api, "SessionLocal", session_factory)
    monkeypatch.setattr(exact_dedupe, "SessionLocal", session_factory)
    return session_factory


def _upload(bucket, db_factory, *, batch_id="BATCH_RECOVERY_001", path="images/a.jpg", payload=b"a"):
    client = RecoveryClient(bucket)
    return upload_api._guarded_image_upload(
        bucket,
        client,
        batch_id=batch_id,
        relative_path=path,
        object_name=f"incoming/{batch_id}/{path}",
        data=payload,
        content_type="image/jpeg",
        source="test",
    )


def test_U1_fresh_upload_creates_one_object_and_one_registry_record(recovery_db):
    bucket = RecoveryBucket()

    result = _upload(bucket, recovery_db)

    assert result["status"] == "UPLOADED"
    assert bucket.object_names() == {"incoming/BATCH_RECOVERY_001/images/a.jpg"}
    with recovery_db() as db:
        rows = db.scalars(select(GlobalImageContent)).all()
        assert len(rows) == 1
        assert rows[0].lifecycle_status == "ACTIVE"


def test_U2_completed_duplicate_skips_without_duplicate_object_or_record(recovery_db):
    bucket = RecoveryBucket()

    first = _upload(bucket, recovery_db)
    retry = _upload(bucket, recovery_db)

    assert first["status"] == "UPLOADED"
    assert retry["status"] == "SKIP"
    assert bucket.blob("incoming/BATCH_RECOVERY_001/images/a.jpg").uploads == 1
    with recovery_db() as db:
        assert len(db.scalars(select(GlobalImageContent)).all()) == 1


def test_U3_gcs_existing_db_missing_reconciles_registry_without_reupload(recovery_db):
    bucket = RecoveryBucket()
    object_name = "incoming/BATCH_RECOVERY_001/images/a.jpg"
    blob = bucket.put(object_name, b"a")

    result = _upload(bucket, recovery_db)

    assert result["status"] == "SKIP"
    assert blob.uploads == 0
    with recovery_db() as db:
        row = db.scalar(select(GlobalImageContent))
        assert row is not None
        assert row.lifecycle_status == "ACTIVE"
        assert row.canonical_object_name == object_name


def test_U4_partial_registry_record_converges_to_one_active_record(recovery_db):
    bucket = RecoveryBucket()
    bucket.put("incoming/BATCH_RECOVERY_001/images/a.jpg", b"a")
    digest = hashlib.sha256(b"a").hexdigest()
    with recovery_db() as db:
        db.add(
            GlobalImageContent(
                sha256=digest,
                lifecycle_status="RESERVED",
                canonical_batch_id="BATCH_RECOVERY_001",
                canonical_object_name="incoming/BATCH_RECOVERY_001/images/a.jpg",
                incoming_batch_id="BATCH_RECOVERY_001",
                incoming_path="images/a.jpg",
                source="test",
            )
        )
        db.commit()

    result = _upload(bucket, recovery_db)

    assert result["status"] == "SKIP"
    with recovery_db() as db:
        rows = db.scalars(select(GlobalImageContent)).all()
        assert len(rows) == 1
        assert rows[0].lifecycle_status == "ACTIVE"


def test_U4_partial_registry_without_object_resumes_upload(recovery_db):
    bucket = RecoveryBucket()
    digest = hashlib.sha256(b"a").hexdigest()
    with recovery_db() as db:
        db.add(
            GlobalImageContent(
                sha256=digest,
                lifecycle_status="RESERVED",
                canonical_batch_id="BATCH_RECOVERY_001",
                canonical_object_name="incoming/BATCH_RECOVERY_001/images/a.jpg",
                incoming_batch_id="BATCH_RECOVERY_001",
                incoming_path="images/a.jpg",
                source="test",
            )
        )
        db.commit()

    result = _upload(bucket, recovery_db)

    assert result["status"] == "UPLOADED"
    assert bucket.object_names() == {"incoming/BATCH_RECOVERY_001/images/a.jpg"}
    with recovery_db() as db:
        row = db.scalar(select(GlobalImageContent))
        assert row.lifecycle_status == "ACTIVE"


def test_U5_repeating_same_package_is_deterministic(recovery_db):
    bucket = RecoveryBucket()
    objects = {"images/a.jpg": b"a", "images/b.jpg": b"b"}

    first = [_upload(bucket, recovery_db, path=path, payload=data) for path, data in objects.items()]
    second = [_upload(bucket, recovery_db, path=path, payload=data) for path, data in objects.items()]

    assert [row["status"] for row in first] == ["UPLOADED", "UPLOADED"]
    assert [row["status"] for row in second] == ["SKIP", "SKIP"]
    assert len(bucket.object_names()) == 2
    with recovery_db() as db:
        assert len(db.scalars(select(GlobalImageContent)).all()) == 2


def test_U5_repeat_zip_twice_preserves_object_count_and_upload_counters(recovery_db, monkeypatch):
    bucket = RecoveryBucket()
    _prepare_fakes(monkeypatch, bucket)
    monkeypatch.setattr(upload_api, "get_bucket_name", lambda: "test-bucket")
    package = io.BytesIO()
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr(
            "metadata/manifest.csv",
            "image_id,file_name,claimed_species\nA,a.jpg,草鱼\n".encode("utf-8"),
        )
        archive.writestr("images/a.jpg", b"a")

    def run_once():
        return asyncio.run(
            upload_api.upload_batch_dataset(
                file=UploadFile(file=io.BytesIO(package.getvalue()), filename="batch.zip"),
                batch_id="BATCH_ZIP_RECOVERY_001",
                source="test",
                batch_name="zip recovery",
            )
        )

    first = run_once()
    object_names_after_first = set(bucket.object_names())
    second = run_once()

    assert first["status"] == "READY_FOR_AUDIT"
    assert second["status"] == "READY_FOR_AUDIT"
    assert first["upload_summary"]["uploaded"] == 2
    assert second["upload_summary"]["skipped"] == 2
    assert set(bucket.object_names()) == object_names_after_first
    with recovery_db() as db:
        assert len(db.scalars(select(GlobalImageContent)).all()) == 1


def test_U6_duplicate_query_executes_select_without_nameerror(recovery_db):
    with recovery_db() as db:
        db.add(
            GlobalDuplicateAudit(
                sha256="a" * 64,
                incoming_batch_id="BATCH_RECOVERY_001",
                incoming_path="images/a.jpg",
                source="test",
                reason="GLOBAL_EXACT_DUPLICATE",
            )
        )
        db.commit()

    assert upload_api._duplicate_paths("BATCH_RECOVERY_001") == {"images/a.jpg"}


def _seed_incoming(bucket: RecoveryBucket, batch_id="BATCH_PREPARE_001"):
    prefix = f"incoming/{batch_id}/"
    bucket.put(
        prefix + "metadata/manifest.csv",
        "image_id,file_name,claimed_species\nA,a.jpg,草鱼\nB,b.jpg,鲤鱼\n".encode("utf-8"),
    )
    bucket.put(prefix + "images/a.jpg", b"a")
    bucket.put(prefix + "images/b.jpg", b"b")
    return prefix


def _prepare_fakes(monkeypatch, bucket):
    client = RecoveryClient(bucket)
    monkeypatch.setattr(upload_api.storage, "Client", lambda: client)
    monkeypatch.setattr(factory.storage, "Client", lambda: client)
    return client


def test_P1_fresh_prepare_audit_reaches_canonical_manifest_and_report(recovery_db, monkeypatch):
    bucket = RecoveryBucket()
    prefix = _seed_incoming(bucket)
    _prepare_fakes(monkeypatch, bucket)

    manifest = upload_api.ensure_incoming_manifest(prefix, "test-bucket")
    report = factory.audit_incoming_batch(prefix, "BATCH_PREPARE_001", "test", "test-bucket")

    assert manifest["status"] == "MANIFEST_READY"
    assert manifest["manifest_rows"] == 2
    assert report["linked_unique_images"] == 2
    assert "cleaning/BATCH_PREPARE_001/auto_v1/audit_report.json" in bucket.object_names()


def test_P2_interrupted_prepare_retry_retains_completed_manifest_step(recovery_db, monkeypatch):
    bucket = RecoveryBucket()
    prefix = _seed_incoming(bucket)
    _prepare_fakes(monkeypatch, bucket)

    first = upload_api.ensure_incoming_manifest(prefix, "test-bucket")
    before = set(bucket.object_names())
    second = upload_api.ensure_incoming_manifest(prefix, "test-bucket")

    assert first["generated"] is True
    assert second["generated"] is False
    assert second["manifest_rows"] == 2
    assert set(bucket.object_names()) == before


def test_P3_partial_prepare_output_is_rebuilt_once_per_logical_item(recovery_db, monkeypatch):
    bucket = RecoveryBucket()
    prefix = _seed_incoming(bucket)
    _prepare_fakes(monkeypatch, bucket)
    bucket.put(
        prefix + "metadata/fish_manifest.csv",
        "image_path,image_id,claimed_species\nimages/a.jpg,A,草鱼\n".encode("utf-8"),
    )
    bucket.put(
        "cleaning/BATCH_PREPARE_001/auto_v1/review_queue.csv",
        b"image_id,auto_status\nA,CANDIDATE\nA,CANDIDATE\n",
    )

    report = factory.audit_incoming_batch(prefix, "BATCH_PREPARE_001", "test", "test-bucket")
    rows = list(
        csv.DictReader(
            io.StringIO(
                bucket.blob("cleaning/BATCH_PREPARE_001/auto_v1/review_queue.csv").download_as_text()
            )
        )
    )

    assert report["linked_unique_images"] == 1
    assert len(rows) == 1
    assert len({row["image_id"] for row in rows}) == 1


def test_P4_successful_prepare_rerun_is_idempotent(recovery_db, monkeypatch):
    bucket = RecoveryBucket()
    prefix = _seed_incoming(bucket)
    _prepare_fakes(monkeypatch, bucket)

    upload_api.ensure_incoming_manifest(prefix, "test-bucket")
    first = factory.audit_incoming_batch(prefix, "BATCH_PREPARE_001", "test", "test-bucket")
    after_first = {name: bucket.blob(name).data for name in bucket.object_names()}
    second = factory.audit_incoming_batch(prefix, "BATCH_PREPARE_001", "test", "test-bucket")
    after_second = {name: bucket.blob(name).data for name in bucket.object_names()}

    assert second["status_counts"] == first["status_counts"]
    assert after_second.keys() == after_first.keys()
    assert after_second["cleaning/BATCH_PREPARE_001/auto_v1/review_queue.csv"] == after_first[
        "cleaning/BATCH_PREPARE_001/auto_v1/review_queue.csv"
    ]


def test_P5_prepare_manifest_query_path_executes_fixed_select(recovery_db, monkeypatch):
    bucket = RecoveryBucket()
    prefix = _seed_incoming(bucket)
    _prepare_fakes(monkeypatch, bucket)

    result = upload_api.ensure_incoming_manifest(prefix, "test-bucket")

    assert result["manifest_rows"] == 2


def test_P6_upload_to_prepare_boundary_accepts_ingested_items(recovery_db, monkeypatch):
    bucket = RecoveryBucket()
    prefix = _seed_incoming(bucket)
    _prepare_fakes(monkeypatch, bucket)

    for path, payload in (("images/a.jpg", b"a"), ("images/b.jpg", b"b")):
        _upload(bucket, recovery_db, batch_id="BATCH_PREPARE_001", path=path, payload=payload)
    upload_api.ensure_incoming_manifest(prefix, "test-bucket")
    result = factory.audit_incoming_batch(prefix, "BATCH_PREPARE_001", "test", "test-bucket")

    assert result["manifest_rows"] == 2
    assert result["linked_unique_images"] == 2



def _seed_registry_batch(
    bucket: RecoveryBucket,
    *,
    batch_id: str,
    image_id: str,
    source_platform: str,
    source_url: str = "",
    truth_species: str = "",
    notes: str = "",
):
    prefix = f"raw/batches/{batch_id}"
    manifest_uri = f"gs://test-bucket/{prefix}/metadata/fish_manifest.csv"
    raw_uri = f"gs://test-bucket/{prefix}/"
    bucket.put(
        f"{prefix}/batch.json",
        json.dumps(
            {
                "batch_id": batch_id,
                "source": "other",
                "image_count": 1,
                "manifest_uri": manifest_uri,
                "raw_uri": raw_uri,
            }
        ).encode("utf-8"),
    )
    manifest = io.StringIO(newline="")
    writer = csv.DictWriter(
        manifest,
        fieldnames=[
            "image_path",
            "image_id",
            "claimed_species",
            "truth_species",
            "source_platform",
            "source_url",
            "notes",
        ],
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerow(
        {
            "image_path": "images/grass_carp/草鱼_001.jpg",
            "image_id": image_id,
            "claimed_species": "草鱼",
            "truth_species": truth_species,
            "source_platform": source_platform,
            "source_url": source_url,
            "notes": notes,
        }
    )
    bucket.put(f"{prefix}/metadata/fish_manifest.csv", manifest.getvalue().encode("utf-8"))
    bucket.put(f"{prefix}/images/grass_carp/草鱼_001.jpg", b"fish-image-bytes")


def test_P7_production_shaped_google_url_in_platform_inserts_with_recovered_metadata(
    recovery_db, monkeypatch
):
    batch_id = "BATCH_20261004_DB_XP_001"
    image_id = "BATCH_EDP_M1_R01_grass_carp_001"
    source_url = "https://www.google.com.hk/imgres?q=草鱼&imgurl=source-image&" + ("query=" + "x" * 500)
    bucket = RecoveryBucket()
    _seed_registry_batch(
        bucket,
        batch_id=batch_id,
        image_id=image_id,
        source_platform=source_url,
    )
    client = RecoveryClient(bucket)
    monkeypatch.setattr(factory.storage, "Client", lambda: client)

    with recovery_db() as db:
        result = factory.sync_batch_registry(db, batch_id, "test-bucket")

    assert result["inserted"] == 1
    with recovery_db() as db:
        image = db.scalar(
            select(ImageAsset).where(
                ImageAsset.batch_id == batch_id,
                ImageAsset.image_id == image_id,
            )
        )
        assert image is not None
        assert image.file_name == "草鱼_001.jpg"
        assert image.object_name == f"raw/batches/{batch_id}/images/grass_carp/草鱼_001.jpg"
        assert image.source_platform == "google_images"
        assert image.source_url == source_url


def test_P8_registry_retry_reuses_asset_and_preserves_manual_review_state(
    recovery_db, monkeypatch
):
    batch_id = "BATCH_20261004_DB_XP_001"
    image_id = "BATCH_EDP_M1_R01_grass_carp_001"
    source_url = "https://www.google.com.hk/imgres?q=草鱼&imgurl=source-image"
    bucket = RecoveryBucket()
    _seed_registry_batch(
        bucket,
        batch_id=batch_id,
        image_id=image_id,
        source_platform="google_images",
        source_url=source_url,
        truth_species="鲤鱼",
        notes="source note",
    )
    client = RecoveryClient(bucket)
    monkeypatch.setattr(factory.storage, "Client", lambda: client)

    with recovery_db() as db:
        first = factory.sync_batch_registry(db, batch_id, "test-bucket")
        image = db.scalar(select(ImageAsset).where(ImageAsset.batch_id == batch_id))
        image.review_status = "approved"
        image.truth_species = "草鱼"
        image.truth_status = "LIKELY_CORRECT"
        image.reviewed_by = "manual-reviewer"
        image.reviewed_at = datetime.now(timezone.utc)
        image.notes = "manual review note"
        db.get(Batch, batch_id).status = "REVIEWED"
        db.commit()

    with recovery_db() as db:
        second = factory.sync_batch_registry(db, batch_id, "test-bucket")

    with recovery_db() as db:
        images = db.scalars(
            select(ImageAsset).where(
                ImageAsset.batch_id == batch_id,
                ImageAsset.image_id == image_id,
            )
        ).all()
        batch = db.get(Batch, batch_id)

    assert first["inserted"] == 1
    assert second["inserted"] == 0
    assert second["updated"] == 1
    assert len(images) == 1
    assert images[0].review_status == "approved"
    assert images[0].truth_species == "草鱼"
    assert images[0].truth_status == "LIKELY_CORRECT"
    assert images[0].reviewed_by == "manual-reviewer"
    assert images[0].notes == "manual review note"
    assert batch.status == "REVIEWED"
