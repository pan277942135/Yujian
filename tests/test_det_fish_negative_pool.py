import hashlib
import io
import json

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.db import Base
from app.det_fish_negative_pool import AddRequest, CandidateRef, RemoveRequest, add_members, candidates, list_pool, remove_members, summary


from app.models import Batch, DetectorNegativePoolEntry, DetectorNegativePoolEvent, ImageAsset, ReviewEvent
from app.platform import models as _platform_models  # registers related platform tables on the shared metadata
from app.presence import FishPresenceResult
from trainer.build_detector_dataset_v02 import collect_source_records


def test_operator_name_is_optional_for_negative_pool_actions():
    add_request = AddRequest(
        items=[CandidateRef(image_asset_id=1, source_record_id="legacy-1")],
        source_kind="DET_DS_V0_1",
        confirm_pending_as_no_fish=True,
    )
    remove_request = RemoveRequest(entry_ids=[1])
    assert add_request.actor == "det-fish-console-ui"
    assert remove_request.actor == "det-fish-console-ui"


class FakeBlob:
    def __init__(self, data):
        self.data = data

    def exists(self, timeout=None):
        return True

    def download_as_bytes(self, timeout=None):
        return self.data

    def open(self, mode):
        assert mode == "rb"
        return io.BytesIO(self.data)


class FakeBucket:
    def __init__(self, data):
        self.data = data

    def blob(self, _name):
        return FakeBlob(self.data)


class FakeStorageClient:
    data = b"full-scene-original-image-bytes"

    def bucket(self, _name):
        return FakeBucket(self.data)


def make_session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session = Session(engine)
    batch = Batch(batch_id="batch-1", source="fixture", image_count=1, manifest_uri="gs://bucket/m.csv", raw_uri="gs://bucket/raw/")
    session.add(batch)
    session.flush()
    image = ImageAsset(batch_id=batch.batch_id, image_id="image-1", file_name="scene.jpg", object_name="raw/scene.jpg",
                       gcs_uri="gs://bucket/raw/scene.jpg", review_status="pending", training_eligible=True, group_id="capture-1")
    session.add(image)
    session.flush()
    presence = FishPresenceResult(image_asset_id=image.id, batch_id=batch.batch_id, status="no_fish", fish_score=0.1,
                                  provider="google_vision", model_version="fixture-v0",
                                  evidence_json=json.dumps({"status": "no_fish", "machine_status": "no_fish"}))
    session.add(presence)
    session.commit()
    return session, image, presence


def test_pool_membership_is_idempotent_audited_and_does_not_mutate_source(monkeypatch):
    monkeypatch.setattr("app.det_fish_negative_pool.storage.Client", FakeStorageClient)
    db, image, presence = make_session()
    pending_candidate = candidates(source_kind="PRESENCE_NO_FISH", limit=25, offset=0, batch_id=None,
                                   verification="ALL", db=db)
    assert pending_candidate["items"][0]["human_confirmation_status"] == "待人工确认"
    assert pending_candidate["items"][0]["eligible_to_add"] is True
    payload = AddRequest(items=[CandidateRef(image_asset_id=image.id, source_record_id=str(presence.id))],
                         source_kind="PRESENCE_NO_FISH", actor="reviewer-1")

    rejected = add_members(payload, db)
    assert not rejected["added"]
    assert "只有 AI 无鱼判断" in rejected["rejected"][0]["reason"]

    confirmed_payload = payload.model_copy(update={"confirm_pending_as_no_fish": True})
    first = add_members(confirmed_payload, db)
    retry = add_members(confirmed_payload, db)
    assert len(first["added"]) == 1
    assert len(retry["already_added"]) == 1
    assert retry["active_count"] == 1
    listed_candidate = candidates(source_kind="PRESENCE_NO_FISH", limit=25, offset=0, batch_id=None,
                                  verification="HUMAN", db=db)
    assert listed_candidate["total"] == 1
    assert listed_candidate["items"][0]["human_confirmation_status"] == "已人工加入 DET_FISH 池"
    pending_after_add = candidates(source_kind="PRESENCE_NO_FISH", limit=25, offset=0, batch_id=None,
                                   verification="PENDING", db=db)
    assert pending_after_add["total"] == 0
    assert pending_after_add["items"] == []
    entry = db.scalar(select(DetectorNegativePoolEntry))
    events = db.scalars(select(DetectorNegativePoolEvent).where(DetectorNegativePoolEvent.pool_entry_id == entry.id)).all()
    assert entry.source_image_sha256 == hashlib.sha256(FakeStorageClient.data).hexdigest()
    assert len(events) == 1
    assert events[0].action == "ADD" and events[0].actor == "reviewer-1"
    assert presence.provider == "google_vision" and presence.status == "no_fish"
    assert json.loads(presence.evidence_json).get("human_override") is None

    listed = list_pool(state="ACTIVE", limit=25, offset=0, batch_id=None, db=db)
    assert listed["total"] == summary(db)["active_count"] == 1
    assert listed["items"][0]["eligible_for_future_build"] is True

    removed = remove_members(RemoveRequest(entry_ids=[entry.id], actor="reviewer-1"), db)
    repeated_remove = remove_members(RemoveRequest(entry_ids=[entry.id], actor="reviewer-1"), db)
    assert removed["removed"] == [entry.id]
    assert repeated_remove["already_removed"] == [entry.id]
    assert summary(db)["active_count"] == 0
    db.refresh(image)
    assert image.gcs_uri == "gs://bucket/raw/scene.jpg"
    assert db.scalar(select(DetectorNegativePoolEntry)).state == "REMOVED"
    assert len(db.scalars(select(DetectorNegativePoolEvent)).all()) == 2

    readded = add_members(confirmed_payload, db)
    assert len(readded["added"]) == 1
    assert db.scalar(select(DetectorNegativePoolEntry)).state == "ACTIVE"
    assert len(db.scalars(select(DetectorNegativePoolEvent)).all()) == 3

    records = collect_source_records(db)
    assert len(records["negatives"]) == 1
    assert records["negatives"][0]["pool_action"] == "RE_ADD"
    assert records["negatives"][0]["pool_source_image_sha256"] == hashlib.sha256(FakeStorageClient.data).hexdigest()
    assert records["negative_pool_active_member_count"] == 1
    db.close()


def test_negative_pool_builder_does_not_read_removed_members():
    db, image, _presence = make_session()
    entry = DetectorNegativePoolEntry(image_asset_id=image.id, source_kind="PRESENCE_NO_FISH", source_record_id="1",
        source_uri=image.gcs_uri, source_status="no_fish", source_snapshot_json="{}", source_image_sha256="a" * 64,
        state="REMOVED", added_by="reviewer-1")
    db.add(entry)
    db.commit()
    records = collect_source_records(db)
    assert records["negatives"] == []
    assert records["negative_pool_active_member_count"] == 0
    db.close()


def test_existing_human_no_fish_override_remains_valid_pool_provenance(monkeypatch):
    monkeypatch.setattr("app.det_fish_negative_pool.storage.Client", FakeStorageClient)
    db, image, presence = make_session()
    presence.provider = "human_override"
    presence.evidence_json = json.dumps({"status": "no_fish", "machine_status": "uncertain", "human_override": "no_fish"})
    db.add(ReviewEvent(image_asset_id=image.id, action="presence_override_update", reviewer="reviewer-0",
                       before_json="{}", after_json=json.dumps({"human_override": "no_fish"})))
    db.commit()

    pending_candidate = candidates(source_kind="PRESENCE_NO_FISH", limit=25, offset=0, batch_id=None,
                                   verification="HUMAN", db=db)
    assert pending_candidate["total"] == 1
    payload = AddRequest(items=[CandidateRef(image_asset_id=image.id, source_record_id=str(presence.id))],
                         source_kind="PRESENCE_NO_FISH", actor="reviewer-1")
    result = add_members(payload, db)
    assert len(result["added"]) == 1

    records = collect_source_records(db)
    assert len(records["negatives"]) == 1
    assert records["negative_pool_member_errors"] == []
    assert records["negatives"][0]["pool_confirmation_snapshot"]["det_fish_manual_confirmation"] is True
    assert records["negatives"][0]["pool_confirmation_snapshot"]["det_fish_confirmation_basis"] == "existing_human_no_fish_review"
    db.close()
