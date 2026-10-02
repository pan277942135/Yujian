from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.data_policy import review_group_name, valid_truth_for_image
from app.dedupe import ImageFingerprint
from app.models import (
    AcceptedPoolSyncRequest,
    Batch,
    BatchCropReview,
    FeedbackEvent,
    ImageAsset,
    ReviewEvent,
    SpeciesCatalog,
)
from app.presence import FishPresenceResult
from app.species_alias import SEARCH_ONLY_ALIASES, alias_resolution, normalize_species_name


TABLES = [
    Batch,
    SpeciesCatalog,
    ImageAsset,
    BatchCropReview,
    FeedbackEvent,
    FishPresenceResult,
    ImageFingerprint,
    ReviewEvent,
    AcceptedPoolSyncRequest,
]


def _session():
    engine = create_engine("sqlite:///:memory:")
    for model in TABLES:
        model.__table__.create(engine)
    return engine, Session(engine)


def test_search_only_aliases_are_not_auto_mapped():
    assert SEARCH_ONLY_ALIASES == {"鲢鱼", "鳊鱼"}
    assert normalize_species_name("鳊鱼") == "鳊鱼"
    assert normalize_species_name("鲢鱼") == "鲢鱼"
    assert normalize_species_name("花鲢") == "鳙鱼"
    assert normalize_species_name("桂鱼") == "鳜鱼"
    assert normalize_species_name("团头鲂") == "鳊鱼 / 武昌鱼"
    assert normalize_species_name("土鲮") == "鲮鱼"
    assert normalize_species_name("餐条") == "白条"
    assert alias_resolution("鳊鱼")["search_only"] is True


def test_review_group_keeps_claimed_label_separate_from_ground_truth():
    image = ImageAsset(claimed_species="鳊鱼", truth_species=None)
    assert review_group_name(image) == "鳊鱼"


def test_backend_accepts_only_canonical_active_or_candidate_truth():
    _engine, db = _session()
    try:
        active = SpeciesCatalog(
            species_key="blunt_snout_bream",
            catalog_order=1,
            common_name_zh="鳊鱼 / 武昌鱼",
            status="active",
        )
        candidate = SpeciesCatalog(
            species_key="candidate_fish",
            catalog_order=2,
            common_name_zh="候选鱼",
            status="candidate",
        )
        db.add_all([active, candidate])
        db.flush()
        catalog = {row.common_name_zh: row for row in (active, candidate)}
        image = ImageAsset(claimed_species="鳊鱼", truth_species=None)

        assert valid_truth_for_image(db, image, "鳊鱼 / 武昌鱼", catalog_by_name=catalog)
        assert valid_truth_for_image(db, image, "候选鱼", catalog_by_name=catalog)
        assert not valid_truth_for_image(db, image, "鳊鱼", catalog_by_name=catalog)
        assert not valid_truth_for_image(db, image, "鲢鱼", catalog_by_name=catalog)
    finally:
        db.close()


def test_retired_truth_is_displayable_but_not_assignable():
    _engine, db = _session()
    try:
        retired = SpeciesCatalog(
            species_key="retired_fish",
            catalog_order=1,
            common_name_zh="历史鱼种",
            status="retired",
        )
        db.add(retired)
        db.flush()
        catalog = {retired.common_name_zh: retired}
        existing = ImageAsset(truth_species="历史鱼种")
        new_image = ImageAsset(truth_species=None)

        assert valid_truth_for_image(db, existing, "历史鱼种", catalog_by_name=catalog)
        assert not valid_truth_for_image(db, new_image, "历史鱼种", catalog_by_name=catalog)
    finally:
        db.close()


def test_24_claimed_bianyu_cards_commit_canonical_ground_truth():
    from app.bulk_review import BulkReviewApply, BulkReviewItem, api_bulk_apply

    _engine, db = _session()
    try:
        db.add(
            SpeciesCatalog(
                species_key="blunt_snout_bream",
                catalog_order=1,
                common_name_zh="鳊鱼 / 武昌鱼",
                status="active",
            )
        )
        db.add(Batch(batch_id="batch-bianyu", source="test", manifest_uri="m", raw_uri="r"))
        for index in range(24):
            image_id = f"image-{index}"
            image = ImageAsset(
                batch_id="batch-bianyu",
                image_id=image_id,
                file_name=f"{image_id}.jpg",
                object_name=image_id,
                gcs_uri=f"gs://test/{image_id}",
                claimed_species="鳊鱼",
            )
            db.add(image)
        db.commit()

        result = api_bulk_apply(
            BulkReviewApply(
                batch_id="batch-bianyu",
                items=[
                    BulkReviewItem(
                        image_id=f"image-{index}",
                        review_status="approved",
                        truth_species="鳊鱼 / 武昌鱼",
                        accepted_bbox=[0, 0, 1, 1],
                    )
                    for index in range(24)
                ],
            ),
            db,
        )

        assert result["updated"] == 24
        assert db.scalars(select(ImageAsset.truth_species)).all() == ["鳊鱼 / 武昌鱼"] * 24
    finally:
        db.close()


def test_bulk_review_ui_uses_catalog_authority_and_atomic_validation():
    source = Path("app/templates/bulk_review.html").read_text(encoding="utf-8")

    assert "assignableSpeciesRows().map" in source
    assert "speciesRows.map(x=>`<option" not in source
    assert "option.value===species" in source
    assert "未修改任何卡片" in source
    assert "const assignableNames=assignableSpeciesNames()" in source
    assert "不能提交别名或审核分组标签" in source
    assert "fetchAllImages" not in source
