"""Global exact-image identity and ingestion guard.

The guard is intentionally content-addressed and exact: only the SHA-256 of
the original image bytes participates in the decision.  Per-batch perceptual
dedupe remains a separate review aid and is not consulted here.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PurePosixPath

from google.cloud import storage
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models import (
    GlobalDuplicateAudit,
    GlobalImageContent,
    GlobalImageDuplicateMember,
    ImageAsset,
)

LOGGER = logging.getLogger(__name__)
GLOBAL_EXACT_DUPLICATE = "GLOBAL_EXACT_DUPLICATE"
RESERVED = "RESERVED"
ACTIVE = "ACTIVE"
FAILED = "FAILED"
_LIFECYCLE = {RESERVED, ACTIVE, FAILED}


class GlobalExactGuardUnavailable(RuntimeError):
    """The authoritative global exact-content registry could not be consulted."""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def sha256_bytes(data: bytes) -> str:
    """Hash an image payload once at the ingestion boundary."""

    return hashlib.sha256(data).hexdigest()


def normalize_sha256(value: str) -> str:
    digest = (value or "").strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError("sha256 must be a 64-character hexadecimal digest")
    return digest


def normalize_path(value: str | None) -> str:
    return str(PurePosixPath((value or "").replace("\\", "/").lstrip("/"))) if value else ""


@dataclass(frozen=True)
class GlobalClaim:
    status: str
    sha256: str
    registry_id: int | None = None
    canonical_batch_id: str | None = None
    canonical_image_id: str | None = None
    canonical_image_asset_id: int | None = None
    canonical_object_name: str | None = None

    @property
    def blocked(self) -> bool:
        return self.status == "DUPLICATE_BLOCKED"

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "sha256": self.sha256,
            "canonical": {
                "batch_id": self.canonical_batch_id,
                "image_id": self.canonical_image_id,
                "image_asset_id": self.canonical_image_asset_id,
                "object_name": self.canonical_object_name,
            },
        }


def _same_logical_asset(
    row: GlobalImageContent,
    *,
    batch_id: str,
    incoming_path: str,
    object_name: str | None,
) -> bool:
    incoming_path = normalize_path(incoming_path)
    object_name = normalize_path(object_name)
    if row.incoming_batch_id != batch_id and row.canonical_batch_id != batch_id:
        return False
    known_paths = {
        normalize_path(row.incoming_path),
        normalize_path(row.canonical_object_name),
    }
    return incoming_path in known_paths or object_name in known_paths


def _claim_insert(db: Session, values: dict) -> bool:
    """Insert once using the database's unique index as the race arbiter."""

    dialect = db.get_bind().dialect.name
    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert

        result = db.execute(
            insert(GlobalImageContent).values(**values).on_conflict_do_nothing(index_elements=["sha256"])
        )
        return bool(result.rowcount)
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert

        result = db.execute(
            insert(GlobalImageContent).values(**values).on_conflict_do_nothing(index_elements=["sha256"])
        )
        return bool(result.rowcount)
    db.add(GlobalImageContent(**values))
    try:
        db.flush()
        return True
    except IntegrityError:
        db.rollback()
        return False


def _audit_duplicate(
    db: Session,
    row: GlobalImageContent,
    *,
    sha256: str,
    batch_id: str,
    incoming_path: str,
    source: str,
) -> None:
    db.add(
        GlobalDuplicateAudit(
            sha256=sha256,
            incoming_batch_id=batch_id,
            incoming_path=normalize_path(incoming_path),
            source=(source or "unknown")[:128],
            canonical_batch_id=row.canonical_batch_id,
            canonical_image_id=row.canonical_image_id,
            canonical_image_asset_id=row.canonical_image_asset_id,
            canonical_object_name=row.canonical_object_name,
            reason=GLOBAL_EXACT_DUPLICATE,
        )
    )


def claim_global_image(
    db: Session,
    *,
    sha256: str,
    batch_id: str,
    incoming_path: str,
    source: str,
    object_name: str | None = None,
    image_id: str | None = None,
) -> GlobalClaim:
    """Atomically reserve an exact image identity.

    A retry of the same batch/path is a resumable ``SKIP``.  Every other
    existing identity is a structured ``DUPLICATE_BLOCKED`` decision and is
    recorded append-only.  ``FAILED`` claims are reclaimable because no
    canonical asset was completed for them.
    """

    digest = normalize_sha256(sha256)
    path = normalize_path(incoming_path)
    object_name = normalize_path(object_name) or None
    now = utcnow()
    values = {
        "sha256": digest,
        "lifecycle_status": RESERVED,
        "canonical_batch_id": batch_id,
        "canonical_image_id": image_id,
        "canonical_object_name": object_name or path,
        "incoming_batch_id": batch_id,
        "incoming_path": path,
        "source": (source or "unknown")[:128],
        "first_seen_at": now,
        "created_at": now,
        "updated_at": now,
    }
    try:
        inserted = _claim_insert(db, values)
        row = db.scalar(select(GlobalImageContent).where(GlobalImageContent.sha256 == digest).with_for_update())
    except Exception as exc:
        db.rollback()
        LOGGER.exception("ingestion_guard_error sha256_prefix=%s batch_id=%s source=%s", digest[:16], batch_id, source)
        raise GlobalExactGuardUnavailable("global exact-content authority is unavailable; retry the ingestion") from exc
    if row is None:
        raise GlobalExactGuardUnavailable("global exact-content registry is unavailable; retry the ingestion")

    if inserted:
        LOGGER.info("ingestion_exact_new sha256_prefix=%s batch_id=%s source=%s", digest[:16], batch_id, source)
        return GlobalClaim("CLAIMED", digest, row.id, row.canonical_batch_id, row.canonical_image_id, row.canonical_image_asset_id, row.canonical_object_name)

    if row.lifecycle_status == FAILED and row.canonical_image_asset_id is None:
        row.lifecycle_status = RESERVED
        row.canonical_batch_id = batch_id
        row.canonical_image_id = image_id
        row.canonical_object_name = object_name or path
        row.incoming_batch_id = batch_id
        row.incoming_path = path
        row.source = (source or "unknown")[:128]
        row.last_error = None
        row.updated_at = now
        LOGGER.info("ingestion_exact_new sha256_prefix=%s batch_id=%s source=%s", digest[:16], batch_id, source)
        return GlobalClaim("CLAIMED", digest, row.id, row.canonical_batch_id, row.canonical_image_id, row.canonical_image_asset_id, row.canonical_object_name)

    if row.lifecycle_status not in _LIFECYCLE:
        raise RuntimeError(f"invalid global exact-content lifecycle: {row.lifecycle_status}")
    if row.lifecycle_status in {RESERVED, ACTIVE} and _same_logical_asset(
        row, batch_id=batch_id, incoming_path=path, object_name=object_name
    ):
        LOGGER.info("ingestion_exact_idempotent_skip sha256_prefix=%s batch_id=%s source=%s", digest[:16], batch_id, source)
        return GlobalClaim("SKIP", digest, row.id, row.canonical_batch_id, row.canonical_image_id, row.canonical_image_asset_id, row.canonical_object_name)

    _audit_duplicate(db, row, sha256=digest, batch_id=batch_id, incoming_path=path, source=source)
    LOGGER.info(
        "ingestion_exact_duplicate_blocked sha256_prefix=%s batch_id=%s source=%s canonical_batch=%s",
        digest[:16],
        batch_id,
        source,
        row.canonical_batch_id,
    )
    return GlobalClaim(
        "DUPLICATE_BLOCKED",
        digest,
        row.id,
        row.canonical_batch_id,
        row.canonical_image_id,
        row.canonical_image_asset_id,
        row.canonical_object_name,
    )


def mark_global_image_active(
    db: Session,
    *,
    sha256: str,
    batch_id: str | None = None,
    image_id: str | None = None,
    image_asset_id: int | None = None,
    object_name: str | None = None,
) -> GlobalImageContent:
    digest = normalize_sha256(sha256)
    row = db.scalar(select(GlobalImageContent).where(GlobalImageContent.sha256 == digest).with_for_update())
    if row is None:
        raise RuntimeError(f"global exact-content claim not found for {digest}")
    row.lifecycle_status = ACTIVE
    if batch_id:
        row.canonical_batch_id = batch_id
    if image_id:
        row.canonical_image_id = image_id
    if image_asset_id is not None:
        row.canonical_image_asset_id = image_asset_id
    if object_name:
        row.canonical_object_name = object_name
    row.updated_at = utcnow()
    row.last_error = None
    LOGGER.info("ingestion_exact_active sha256_prefix=%s batch_id=%s image_id=%s", digest[:16], batch_id, image_id)
    return row


def mark_global_image_failed(db: Session, *, sha256: str, error: str) -> None:
    digest = normalize_sha256(sha256)
    row = db.scalar(select(GlobalImageContent).where(GlobalImageContent.sha256 == digest).with_for_update())
    if row is None:
        return
    row.lifecycle_status = FAILED
    row.last_error = str(error)[:4000]
    row.updated_at = utcnow()
    LOGGER.error("ingestion_guard_error sha256_prefix=%s error=%s", digest[:16], row.last_error)


def record_historical_member(db: Session, *, sha256: str, image: ImageAsset) -> bool:
    """Record an existing ImageAsset in the historical exact-content group."""

    digest = normalize_sha256(sha256)
    existing = db.scalar(
        select(GlobalImageDuplicateMember).where(
            GlobalImageDuplicateMember.sha256 == digest,
            GlobalImageDuplicateMember.image_asset_id == image.id,
        )
    )
    if existing:
        return False
    db.add(
        GlobalImageDuplicateMember(
            sha256=digest,
            image_asset_id=image.id,
            batch_id=image.batch_id,
            image_id=image.image_id,
            object_name=image.object_name,
        )
    )
    return True


def _gcs_parts(uri: str | None, bucket_name: str | None) -> tuple[str, str]:
    value = (uri or "").strip()
    if value.startswith("gs://"):
        body = value[5:]
        if "/" not in body:
            raise ValueError(f"invalid GCS URI: {value}")
        return body.split("/", 1)
    if not bucket_name or not value:
        raise ValueError("image asset has no usable GCS object reference")
    return bucket_name, value


def bootstrap_global_registry(
    db: Session,
    *,
    bucket_name: str | None = None,
    limit: int | None = None,
) -> dict:
    """Populate the registry for every live ImageAsset without deleting history."""

    # ``app.dedupe`` imports factory helpers, so keep this legacy-fingerprint
    # dependency lazy and leave the ingestion guard importable by factory.
    from app.dedupe import ImageFingerprint

    assets = db.scalars(select(ImageAsset).order_by(ImageAsset.created_at, ImageAsset.id)).all()
    if limit is not None:
        assets = assets[:limit]
    fingerprints = {
        row.image_asset_id: row
        for row in db.scalars(select(ImageFingerprint).where(ImageFingerprint.image_asset_id.in_([a.id for a in assets]))).all()
    } if assets else {}
    client = None
    processed = missing = created = historical_duplicates = 0
    missing_details = []
    for image in assets:
        fingerprint = fingerprints.get(image.id)
        digest = fingerprint.sha256 if fingerprint and fingerprint.sha256 else None
        if not digest:
            try:
                if client is None:
                    client = storage.Client()
                bucket, object_name = _gcs_parts(image.gcs_uri, bucket_name)
                digest = sha256_bytes(client.bucket(bucket).blob(object_name).download_as_bytes())
                LOGGER.info("global_sha_backfill_processed image_asset_id=%s sha256_prefix=%s", image.id, digest[:16])
            except Exception as exc:
                missing += 1
                missing_details.append(
                    {
                        "image_asset_id": image.id,
                        "batch_id": image.batch_id,
                        "image_id": image.image_id,
                        "gcs_uri": image.gcs_uri,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                LOGGER.error("global_sha_backfill_missing image_asset_id=%s error=%s", image.id, exc)
                continue
        digest = normalize_sha256(digest)
        row = db.scalar(select(GlobalImageContent).where(GlobalImageContent.sha256 == digest).with_for_update())
        if row is None:
            now = utcnow()
            row = GlobalImageContent(
                sha256=digest,
                lifecycle_status=ACTIVE,
                canonical_batch_id=image.batch_id,
                canonical_image_id=image.image_id,
                canonical_image_asset_id=image.id,
                canonical_object_name=image.object_name,
                source="historical_bootstrap",
                first_seen_at=image.created_at or now,
                created_at=now,
                updated_at=now,
            )
            db.add(row)
            db.flush()
            created += 1
        else:
            current_key = (row.canonical_batch_id or "", row.canonical_image_id or "", row.canonical_image_asset_id or 0)
            image_key = (image.batch_id, image.image_id, image.id or 0)
            if not row.canonical_image_asset_id or image_key < current_key:
                row.canonical_batch_id = image.batch_id
                row.canonical_image_id = image.image_id
                row.canonical_image_asset_id = image.id
                row.canonical_object_name = image.object_name
                row.lifecycle_status = ACTIVE
            historical_duplicates += 1
        record_historical_member(db, sha256=digest, image=image)
        processed += 1
    db.commit()
    coverage = (processed / len(assets)) if assets else 1.0
    result = {
        "processed": processed,
        "missing": missing,
        "created": created,
        "historical_duplicate_members": historical_duplicates,
        "missing_details": missing_details,
        "coverage": coverage,
        "coverage_complete": missing == 0,
        "status": "COMPLETE" if missing == 0 else "BLOCKED_DEPENDENCY",
    }
    LOGGER.info("global_sha_backfill_summary result=%s", result)
    return result


def open_registry_session() -> Session:
    """Return a short-lived session for upload requests that lack DI context."""

    return SessionLocal()
