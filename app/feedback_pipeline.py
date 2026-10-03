from __future__ import annotations

import csv
import io
from pathlib import PurePosixPath

from google.cloud import storage
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.exact_dedupe import claim_global_image, mark_global_image_active, mark_global_image_failed, sha256_bytes
from app.factory import get_bucket_name
from app.historical_duplicate_closure import assert_training_authority_writable
from app.models import FeedbackEvent


def parse_gs(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"feedback image must be a GCS URI: {uri}")
    body = uri[5:]
    if "/" not in body:
        raise ValueError(f"invalid GCS URI: {uri}")
    bucket_name, object_name = body.split("/", 1)
    return bucket_name, object_name


def materialize_feedback_batch(
    db: Session,
    *,
    batch_id: str,
    bucket_name: str | None = None,
    limit: int = 500,
) -> dict:
    """Turn NEW online feedback events into a normal incoming batch.

    No feedback is trusted as training truth here. Confirmations and corrections are
    both routed back through the normal Review stage.
    """
    assert_training_authority_writable()
    if not batch_id.startswith("BATCH_"):
        raise ValueError("batch_id must start with BATCH_")
    bucket_name = bucket_name or get_bucket_name()
    client = storage.Client()
    target_bucket = client.bucket(bucket_name)
    prefix = f"incoming/{batch_id}/"
    marker = target_bucket.blob(prefix + "metadata/fish_manifest.csv")
    if marker.exists(client):
        raise ValueError(f"incoming feedback batch already exists: gs://{bucket_name}/{prefix}")

    events = db.scalars(
        select(FeedbackEvent)
        .where(FeedbackEvent.pipeline_status == "NEW", FeedbackEvent.image_gcs_uri.is_not(None))
        .order_by(FeedbackEvent.created_at)
        .limit(limit)
    ).all()
    if not events:
        raise ValueError("no NEW feedback events with image_gcs_uri")

    rows: list[dict[str, str]] = []
    copied = 0
    skipped_missing = 0
    duplicates = 0
    for event in events:
        assert event.image_gcs_uri
        source_bucket_name, source_object = parse_gs(event.image_gcs_uri)
        source_bucket = client.bucket(source_bucket_name)
        source_blob = source_bucket.blob(source_object)
        if not source_blob.exists(client):
            skipped_missing += 1
            continue

        suffix = PurePosixPath(source_object).suffix.lower() or ".jpg"
        safe_name = f"FB{event.id:08d}{suffix}"
        target_object = prefix + "images/feedback/" + safe_name
        target_blob = target_bucket.blob(target_object)
        image_bytes = source_blob.download_as_bytes()
        digest = sha256_bytes(image_bytes)
        relative_path = "images/feedback/" + safe_name
        claim = claim_global_image(
            db,
            sha256=digest,
            batch_id=batch_id,
            incoming_path=relative_path,
            object_name=target_object,
            source="yujian_app_feedback",
            image_id=f"FB{event.id:08d}",
        )
        if claim.blocked:
            event.pipeline_status = "DUPLICATE_BLOCKED"
            event.materialized_batch_id = None
            event.materialized_image_id = None
            event.user_note = "\n".join(
                value for value in (event.user_note, f"[global_exact_duplicate] sha256={digest}") if value
            )
            duplicates += 1
            continue
        try:
            if target_blob.exists(client):
                existing_bytes = target_blob.download_as_bytes()
                if sha256_bytes(existing_bytes) != digest:
                    raise RuntimeError(f"feedback target differs from source: gs://{bucket_name}/{target_object}")
                skipped_missing += 0
            else:
                content_type = {
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".png": "image/png",
                    ".webp": "image/webp",
                }.get(suffix, "application/octet-stream")
                target_blob.upload_from_string(image_bytes, content_type=content_type, if_generation_match=0)
                copied += 1
            mark_global_image_active(
                db,
                sha256=digest,
                batch_id=batch_id,
                image_id=f"FB{event.id:08d}",
                object_name=target_object,
            )
        except Exception as exc:
            mark_global_image_failed(db, sha256=digest, error=str(exc))
            raise

        claimed = (event.corrected_species or event.predicted_species or "").strip()
        rows.append(
            {
                "image_id": f"FB{event.id:08d}",
                "file_name": f"images/feedback/{safe_name}",
                "source_url": "",
                "source_platform": "yujian_app_feedback",
                "crawl_date": event.created_at.isoformat() if event.created_at else "",
                "claimed_species": claimed,
                "scene": "user_catch",
                "image_quality": "unknown",
                "license_status": "user_contributed",
                "review_status": "needs_review" if event.feedback_type != "confirmed" else "pending",
                "group_id": event.source_event_id,
                "notes": (
                    f"feedback_type={event.feedback_type};model={event.model_version or ''};"
                    f"predicted={event.predicted_species or ''};confidence={event.confidence if event.confidence is not None else ''};"
                    f"user_corrected={event.corrected_species or ''}"
                ),
            }
        )
        event.pipeline_status = "BATCHED"
        event.materialized_batch_id = batch_id
        event.materialized_image_id = f"FB{event.id:08d}"

    if not rows:
        db.commit()
        return {
            "batch_id": batch_id,
            "status": "NO_NEW_IMAGES",
            "incoming_uri": None,
            "feedback_events": 0,
            "copied_images": copied,
            "duplicates": duplicates,
            "skipped_missing_images": skipped_missing,
            "next_step": "没有新的图片进入审核",
        }

    fields = [
        "image_id",
        "file_name",
        "source_url",
        "source_platform",
        "crawl_date",
        "claimed_species",
        "scene",
        "image_quality",
        "license_status",
        "review_status",
        "group_id",
        "notes",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    marker.upload_from_string(buf.getvalue(), content_type="text/csv", if_generation_match=0)
    db.commit()
    return {
        "batch_id": batch_id,
        "status": "READY_FOR_AUDIT",
        "incoming_uri": f"gs://{bucket_name}/{prefix}",
        "feedback_events": len(rows),
        "copied_images": copied,
        "duplicates": duplicates,
        "skipped_missing_images": skipped_missing,
        "next_step": "Open Batches and click 准备审核",
    }
