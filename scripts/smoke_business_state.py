#!/usr/bin/env python3
import os
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("REGISTRY_DB_URL", "sqlite:///:memory:")

from fastapi import HTTPException
from sqlalchemy import select

from app.entry import app
from app.bulk_review import api_bulk_apply, api_bulk_images, api_bulk_species, BulkReviewApply, BulkReviewItem
from app.dataset_api import DatasetFreezePreviewRequest, build_preview
from app.db import SessionLocal, init_db
from app.dedupe import ImageFingerprint
from app.flywheel import ensure_species_catalog, flywheel_summary
from app.main import ReviewUpdate, apply_review_filters, update_review
from app.models import Batch, ImageAsset, SpeciesCatalog
from app.presence import FishPresenceResult


def add_image(db, batch, image_id, claimed, truth=None, status="pending"):
    row = ImageAsset(
        batch_id=batch,
        image_id=image_id,
        file_name=f"{image_id}.jpg",
        object_name=f"raw/{image_id}.jpg",
        gcs_uri=f"gs://test/{image_id}.jpg",
        claimed_species=claimed,
        truth_species=truth,
        review_status=status,
    )
    db.add(row)
    db.flush()
    return row


def add_qa(db, image, *, presence="single_fish", duplicate_kind=None):
    p = FishPresenceResult(
        image_asset_id=image.id,
        batch_id=image.batch_id,
        status=presence,
        fish_count=1 if presence == "single_fish" else 0,
    )
    db.add(p)
    fp = ImageFingerprint(
        image_asset_id=image.id,
        batch_id=image.batch_id,
        sha256=(f"{image.id:064x}")[-64:],
        phash_json="[]",
        dhash="0" * 16,
        crop_hash="",
        histogram_json="[]",
        width=100,
        height=100,
        duplicate_group=f"DUP_{image.id}" if duplicate_kind else None,
        is_representative=False if duplicate_kind else True,
        duplicate_kind=duplicate_kind,
    )
    db.add(fp)
    db.flush()
    return p, fp


def main():
    init_db()
    db = SessionLocal()
    try:
        ensure_species_catalog(db)
        db.add(Batch(batch_id="BATCH_P0", source="smoke", image_count=8, manifest_uri="gs://test/m.csv", raw_uri="gs://test/raw", status="REGISTERED"))
        db.flush()

        wrong_claim = add_image(db, "BATCH_P0", "I1", "黄骨鱼", "黑鱼", "approved")
        yellow = add_image(db, "BATCH_P0", "I2", "黄骨鱼", "黄骨鱼", "approved")
        legacy_unconfirmed = add_image(db, "BATCH_P0", "I3", "黄骨鱼", None, "approved")
        pending = add_image(db, "BATCH_P0", "I4", "黄骨鱼", None, "needs_review")
        nofish = add_image(db, "BATCH_P0", "I5", "鲫鱼", "鲫鱼", "approved")
        duplicate = add_image(db, "BATCH_P0", "I6", "鲤鱼", "鲤鱼", "approved")
        unscanned = add_image(db, "BATCH_P0", "I7", "草鱼", "草鱼", "approved")
        db.commit()

        for image in [wrong_claim, yellow, legacy_unconfirmed, pending]:
            add_qa(db, image)
        p_nofish, _ = add_qa(db, nofish, presence="no_fish")
        _, fp_dup = add_qa(db, duplicate, duplicate_kind="near")
        db.commit()

        # Statistics are Ground Truth only: I1 is black, not yellow; legacy null is explicit unconfirmed.
        summary = flywheel_summary(db)
        dist = {x["species"]: x["count"] for x in summary["approved_species"]}
        assert dist["黑鱼"] == 1, dist
        assert dist["黄骨鱼"] == 1, dist
        assert dist["未确认真实鱼种"] == 1, dist

        yellow_rows = db.scalars(apply_review_filters(select(ImageAsset), None, "BATCH_P0", "黄骨鱼", None)).all()
        assert [x.image_id for x in yellow_rows] == ["I2"], [x.image_id for x in yellow_rows]

        # Approval cannot promote claimed -> truth implicitly.
        try:
            update_review("BATCH_P0", "I4", ReviewUpdate(review_status="approved", reviewer="smoke"), db)
            raise AssertionError("approval without truth should fail")
        except HTTPException as exc:
            assert exc.status_code == 400
            db.rollback()

        update_review("BATCH_P0", "I4", ReviewUpdate(review_status="approved", truth_species="黄骨鱼", reviewer="smoke"), db)
        db.refresh(pending)
        assert pending.truth_species == "黄骨鱼" and pending.review_status == "approved"

        # Bulk work queue groups by truth first, claimed only while truth is empty; drilldown matches count.
        pending2 = add_image(db, "BATCH_P0", "I8", "黄骨鱼", None, "hard_case")
        add_qa(db, pending2)
        db.commit()
        groups = api_bulk_species("BATCH_P0", "pending", db)
        yellow_group = next(x for x in groups if x["species"] == "黄骨鱼")
        drilled = api_bulk_images("BATCH_P0", "黄骨鱼", "pending", None, 60, 0, db)
        assert yellow_group["count"] == drilled["total"] == 1, (groups, drilled)
        assert drilled["items"][0]["truth_species"] is None

        # Pending save with blank truth does not upgrade claimed label.
        api_bulk_apply(BulkReviewApply(batch_id="BATCH_P0", items=[BulkReviewItem(image_id="I8", review_status="hard_case", truth_species=None)]), db)
        db.refresh(pending2)
        assert pending2.truth_species is None and pending2.review_status == "hard_case"

        # Freeze Preview requires QA scans and verified truth. This tiny fixture is
        # intentionally below the default training thresholds, so no class is enabled.
        preview1 = build_preview(db, DatasetFreezePreviewRequest(dataset_version="DS_P0", seed=7, train=0.7, val=0.15))
        assert preview1["excluded_species_counts"].get("未确认真实鱼种") == 1, preview1
        assert preview1["excluded_quality_counts"].get("dedupe_not_scanned") == 1, preview1
        assert preview1["excluded_quality_counts"].get("no_fish") == 1, preview1
        assert preview1["excluded_quality_counts"].get("near_duplicate") == 1, preview1
        assert preview1["image_count"] == 0 and not preview1["freeze_ready"], preview1

        # Human re-approval after machine result overrides no-fish / near-duplicate
        # quality gates. The images become eligible for counting, but the classes still
        # remain default-disabled because this tiny fixture is below training thresholds.
        nofish.reviewed_at = p_nofish.updated_at + timedelta(seconds=1)
        nofish.reviewed_by = "人工复核"
        duplicate.reviewed_at = fp_dup.updated_at + timedelta(seconds=1)
        duplicate.reviewed_by = "人工复核"
        db.commit()
        preview2 = build_preview(db, DatasetFreezePreviewRequest(dataset_version="DS_P0", seed=7, train=0.7, val=0.15))
        assert preview2["image_count"] == 0, preview2
        assert preview2["excluded_quality_counts"].get("no_fish", 0) == 0, preview2
        assert preview2["excluded_quality_counts"].get("near_duplicate", 0) == 0, preview2
        assert preview2["excluded_quality_counts"].get("dedupe_not_scanned") == 1, preview2

        # A truth edit that does not change the trainable class set is allowed to keep
        # the same selection hash: preview_hash protects the exact frozen selection,
        # not unrelated low-data rows that are excluded from this training snapshot.
        old_hash = preview2["selection_hash"]
        yellow.truth_species = "黑鱼"
        db.commit()
        preview3 = build_preview(db, DatasetFreezePreviewRequest(dataset_version="DS_P0", seed=7, train=0.7, val=0.15))
        assert preview3["selection_hash"] == old_hash, (preview2, preview3)
        assert preview3["image_count"] == 0 and not preview3["freeze_ready"], preview3

        # Retired historical truth may be preserved but not newly assigned.
        other = db.get(SpeciesCatalog, "other_freshwater_fish")
        other.status = "retired"
        hist = add_image(db, "BATCH_P0", "I9", "其他淡水鱼", "其他淡水鱼", "pending")
        db.commit()
        update_review("BATCH_P0", "I9", ReviewUpdate(review_status="pending", truth_species="其他淡水鱼", reviewer="smoke"), db)
        try:
            update_review("BATCH_P0", "I8", ReviewUpdate(review_status="pending", truth_species="其他淡水鱼", reviewer="smoke"), db)
            raise AssertionError("new retired truth assignment should fail")
        except HTTPException:
            db.rollback()

        paths = app.openapi()["paths"]
        assert "/api/dataset-freeze/{dataset_version}/audit" in paths
        print("P0 business state smoke OK", preview3)
    finally:
        db.close()


if __name__ == "__main__":
    main()
