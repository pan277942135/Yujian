from app.accepted_bbox_review import AcceptedBBoxBulk, AcceptedBBoxBulkItem, _box, _reviewed
from app.models import BatchCropReview


def test_accepted_bbox_requires_normalized_positive_box():
    assert _box([0.1, 0.2, 0.4, 0.5]) == [0.1, 0.2, 0.4, 0.5]
    assert _box([0.8, 0.2, 0.4, 0.2]) is None
    assert _box([0, 0, 0, 0]) is None


def test_confirmed_pool_requires_status_and_bbox():
    accepted = BatchCropReview(status="ACCEPTED", accepted_bbox_json="[0.1,0.1,0.4,0.4]")
    pending = BatchCropReview(status="REVIEW_REQUIRED", accepted_bbox_json="[0.1,0.1,0.4,0.4]")
    missing = BatchCropReview(status="ACCEPTED", accepted_bbox_json=None)
    assert _reviewed(accepted)
    assert not _reviewed(pending)
    assert not _reviewed(missing)


def test_bulk_backfill_contract_is_bounded_and_explicit():
    payload = AcceptedBBoxBulk(
        items=[
            AcceptedBBoxBulkItem(
                batch_id="B1",
                image_id="I1",
                decision="ACCEPTED",
                accepted_bbox=[0.1, 0.1, 0.4, 0.4],
            )
        ]
    )
    assert payload.items[0].decision == "ACCEPTED"
    assert len(payload.items) == 1
