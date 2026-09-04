from __future__ import annotations

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from app.models import FeedbackEvent, ImageAsset, SpeciesCatalog

UNCONFIRMED_TRUTH = "未确认真实鱼种"
AUTO_REVIEWERS = {"鱼体检测", "近重复检测"}


def normalized_truth(image: ImageAsset) -> str:
    return (image.truth_species or "").strip()


def normalized_claimed(image: ImageAsset) -> str:
    return (image.claimed_species or "").strip()


def truth_sql_expr():
    return func.nullif(func.trim(ImageAsset.truth_species), "")


def claimed_sql_expr():
    return func.nullif(func.trim(ImageAsset.claimed_species), "")


def truth_filter_clause(species: str):
    truth = truth_sql_expr()
    if species == UNCONFIRMED_TRUTH:
        return truth.is_(None)
    return truth == species


def review_group_name(image: ImageAsset) -> str:
    return normalized_truth(image) or normalized_claimed(image) or "未标注"


def review_group_clause(species: str):
    truth = truth_sql_expr()
    claimed = claimed_sql_expr()
    if species == "未标注":
        return and_(truth.is_(None), claimed.is_(None))
    return or_(truth == species, and_(truth.is_(None), claimed == species))


def truth_distribution(db: Session, *, review_status: str | None = None) -> tuple[list[tuple[str, int]], int]:
    truth = truth_sql_expr()
    stmt = select(truth, func.count()).where(truth.is_not(None))
    unconfirmed_stmt = select(func.count()).select_from(ImageAsset).where(truth.is_(None))
    if review_status:
        stmt = stmt.where(ImageAsset.review_status == review_status)
        unconfirmed_stmt = unconfirmed_stmt.where(ImageAsset.review_status == review_status)
    rows = db.execute(stmt.group_by(truth).order_by(func.count().desc())).all()
    unconfirmed = db.scalar(unconfirmed_stmt) or 0
    return [(str(name), int(count)) for name, count in rows], int(unconfirmed)


def valid_truth_for_image(db: Session, image: ImageAsset, proposed: str) -> bool:
    proposed = proposed.strip()
    if not proposed:
        return True
    # Historical retired truth may be preserved, but retired species cannot be newly assigned.
    if proposed == normalized_truth(image):
        return True
    row = db.scalar(select(SpeciesCatalog).where(SpeciesCatalog.common_name_zh == proposed))
    return bool(row and row.status in {"active", "candidate"})


def human_approval_overrides(image: ImageAsset, machine_updated_at) -> bool:
    if image.review_status != "approved" or not image.reviewed_at or not machine_updated_at:
        return False
    if image.reviewed_by in AUTO_REVIEWERS:
        return False
    return image.reviewed_at >= machine_updated_at


def mark_feedback_reviewed(db: Session, image: ImageAsset) -> None:
    if image.review_status not in {"approved", "rejected"}:
        return
    row = db.scalar(
        select(FeedbackEvent).where(
            FeedbackEvent.materialized_batch_id == image.batch_id,
            FeedbackEvent.materialized_image_id == image.image_id,
        )
    )
    if row and row.pipeline_status == "BATCHED":
        row.pipeline_status = "REVIEWED"
