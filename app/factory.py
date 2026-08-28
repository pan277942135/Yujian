from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import PurePosixPath

from google.api_core.retry import Retry
from google.cloud import storage
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Batch, DatasetVersion, ImageAsset

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
TARGET_SPECIES = ["草鱼", "鳙鱼", "白鲢", "鲤鱼", "鲫鱼", "加州鲈", "黑鱼", "黄骨鱼", "青鱼"]
TARGET_SPECIES_SET = set(TARGET_SPECIES)
VALID_REVIEW = {"approved", "needs_review", "rejected", "hard_case", "pending"}
REQUIRED_COLUMNS = {"image_id", "file_name", "claimed_species"}
DOWNLOAD_RETRY = Retry(initial=1.0, maximum=20.0, multiplier=2.0, deadline=600.0)


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_bucket_name() -> str:
    value = os.getenv("GCS_BUCKET", "").strip()
    if not value:
        raise RuntimeError("GCS_BUCKET is not configured")
    return value


def norm_path(value: str | None) -> str:
    value = (value or "").strip().replace("\\", "/").lstrip("/")
    return str(PurePosixPath(value)) if value else ""


def _download_text(blob: storage.Blob, encoding: str = "utf-8-sig") -> str:
    raw = blob.download_as_bytes(timeout=120, retry=DOWNLOAD_RETRY)
    return raw.decode(encoding)


def _manifest_rows(blob: storage.Blob) -> tuple[list[str], list[dict[str, str]], int]:
    text = _download_text(blob)
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise RuntimeError("manifest has no header")
    missing = REQUIRED_COLUMNS - set(reader.fieldnames)
    if missing:
        raise RuntimeError(f"manifest missing required columns: {sorted(missing)}")
    rows = list(reader)
    malformed = sum(1 for row in rows if None in row)
    return list(reader.fieldnames), rows, malformed


def _audit_reports(client: storage.Client, bucket_name: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for blob in client.list_blobs(bucket_name, prefix="cleaning/"):
        if not blob.name.endswith("/auto_v1/audit_report.json"):
            continue
        try:
            doc = json.loads(_download_text(blob, "utf-8"))
        except Exception:
            continue
        incoming_uri = doc.get("incoming_uri")
        if incoming_uri:
            result[incoming_uri.rstrip("/") + "/"] = doc
    return result


def list_incoming_batches(bucket_name: str | None = None) -> list[dict]:
    bucket_name = bucket_name or get_bucket_name()
    client = storage.Client()
    reports = _audit_reports(client, bucket_name)

    iterator = client.list_blobs(bucket_name, prefix="incoming/", delimiter="/")
    prefixes: set[str] = set()
    for page in iterator.pages:
        prefixes.update(page.prefixes)

    result = []
    for prefix in sorted(prefixes):
        blobs = [b for b in client.list_blobs(bucket_name, prefix=prefix) if not b.name.endswith("/")]
        images = [b for b in blobs if PurePosixPath(b.name).suffix.lower() in IMAGE_EXTS]
        manifests = [b for b in blobs if b.name.endswith("/fish_manifest.csv") or b.name == prefix + "fish_manifest.csv"]
        uri = f"gs://{bucket_name}/{prefix}"
        audit = reports.get(uri.rstrip("/") + "/")
        canonical_batch = (audit or {}).get("batch_id") or prefix.rstrip("/").split("/")[-1]
        source = (audit or {}).get("source") or ("pilot" if "PILOT" in canonical_batch.upper() else "")
        raw_marker = client.bucket(bucket_name).blob(f"raw/batches/{canonical_batch}/batch.json")
        result.append(
            {
                "incoming_prefix": prefix,
                "folder": prefix.rstrip("/").split("/")[-1],
                "image_count": len(images),
                "manifest_count": len(manifests),
                "size_bytes": sum(b.size or 0 for b in blobs),
                "canonical_batch_id": canonical_batch,
                "source": source,
                "audit": audit,
                "promoted": raw_marker.exists(client),
            }
        )
    return result


def audit_incoming_batch(
    incoming_prefix: str,
    batch_id: str,
    source: str,
    bucket_name: str | None = None,
) -> dict:
    bucket_name = bucket_name or get_bucket_name()
    if not batch_id.startswith("BATCH_"):
        raise ValueError("batch_id must start with BATCH_")
    prefix = incoming_prefix.strip("/") + "/"
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = [b for b in client.list_blobs(bucket_name, prefix=prefix) if not b.name.endswith("/")]
    if not blobs:
        raise RuntimeError(f"no objects under gs://{bucket_name}/{prefix}")
    manifests = [b for b in blobs if b.name.endswith("/fish_manifest.csv") or b.name == prefix + "fish_manifest.csv"]
    if len(manifests) != 1:
        raise RuntimeError(f"expected exactly one fish_manifest.csv, found {len(manifests)}")

    _, rows, malformed_rows = _manifest_rows(manifests[0])
    image_blobs = [b for b in blobs if PurePosixPath(b.name).suffix.lower() in IMAGE_EXTS]
    rel_blob = {b.name[len(prefix):]: b for b in image_blobs}
    basename_index: dict[str, list[storage.Blob]] = defaultdict(list)
    for rel, blob in rel_blob.items():
        basename_index[PurePosixPath(rel).name].append(blob)

    id_counts = Counter((r.get("image_id") or "").strip() for r in rows if (r.get("image_id") or "").strip())
    file_counts = Counter(norm_path(r.get("file_name")) for r in rows if (r.get("file_name") or "").strip())
    url_counts = Counter((r.get("source_url") or "").strip() for r in rows if (r.get("source_url") or "").strip())

    linked_blob_names: set[str] = set()
    resolved: list[dict] = []
    md5_first: dict[str, str] = {}

    for idx, row in enumerate(rows, start=1):
        image_id = (row.get("image_id") or "").strip()
        file_name = norm_path(row.get("file_name"))
        species = (row.get("claimed_species") or "").strip()
        source_url = (row.get("source_url") or "").strip()
        reasons: list[str] = []
        status = "CANDIDATE"
        blob = None

        if not image_id:
            reasons.append("missing_image_id")
        if not file_name:
            reasons.append("missing_file_name")
        if not species:
            reasons.append("missing_claimed_species")
        elif species not in TARGET_SPECIES_SET:
            reasons.append("non_target_or_unknown_species")

        if file_name in rel_blob:
            blob = rel_blob[file_name]
        elif file_name:
            matches = basename_index.get(PurePosixPath(file_name).name, [])
            if len(matches) == 1:
                blob = matches[0]
                reasons.append("resolved_by_basename")
            elif len(matches) > 1:
                reasons.append("ambiguous_image_path")
            else:
                reasons.append("missing_image_object")

        if image_id and id_counts[image_id] > 1:
            reasons.append("duplicate_image_id")
        if file_name and file_counts[file_name] > 1:
            reasons.append("duplicate_file_name")
        if source_url and url_counts[source_url] > 1:
            reasons.append("duplicate_source_url")

        if blob:
            linked_blob_names.add(blob.name)
            if blob.md5_hash:
                if blob.md5_hash in md5_first:
                    reasons.append("exact_duplicate_md5")
                else:
                    md5_first[blob.md5_hash] = blob.name

        hard_reject = {
            "missing_image_id",
            "missing_file_name",
            "missing_image_object",
            "ambiguous_image_path",
            "duplicate_image_id",
            "duplicate_file_name",
            "exact_duplicate_md5",
        }
        if any(reason in hard_reject for reason in reasons):
            status = "AUTO_REJECT"
        elif any(reason in {"missing_claimed_species", "non_target_or_unknown_species", "duplicate_source_url"} for reason in reasons):
            status = "NEEDS_REVIEW"

        resolved.append(
            {
                "row_number": idx,
                "image_id": image_id,
                "file_name": file_name,
                "resolved_gcs_name": blob.name if blob else "",
                "claimed_species": species,
                "source_url": source_url,
                "auto_status": status,
                "auto_reasons": ";".join(reasons),
                "size_bytes": blob.size if blob else "",
                "generation": str(blob.generation) if blob else "",
                "md5_hash": blob.md5_hash if blob else "",
                "review_status": "",
                "review_truth_species": "",
                "review_notes": "",
            }
        )

    orphan_blobs = [b for b in image_blobs if b.name not in linked_blob_names]
    status_counts = Counter(row["auto_status"] for row in resolved)
    species_counts = Counter(row["claimed_species"] for row in resolved)
    report = {
        "batch_id": batch_id,
        "source": source,
        "incoming_uri": f"gs://{bucket_name}/{prefix}",
        "created_at": utcnow_iso(),
        "object_count": len(blobs),
        "image_object_count": len(image_blobs),
        "manifest_rows": len(rows),
        "malformed_csv_rows": malformed_rows,
        "linked_unique_images": len(linked_blob_names),
        "orphan_image_count": len(orphan_blobs),
        "status_counts": dict(status_counts),
        "species_counts": dict(species_counts),
        "orphan_images": [b.name[len(prefix):] for b in orphan_blobs],
    }

    out_prefix = f"cleaning/{batch_id}/auto_v1/"
    bucket.blob(out_prefix + "audit_report.json").upload_from_string(
        json.dumps(report, ensure_ascii=False, indent=2), content_type="application/json"
    )
    buf = io.StringIO()
    cols = list(resolved[0].keys()) if resolved else []
    writer = csv.DictWriter(buf, fieldnames=cols)
    writer.writeheader()
    writer.writerows(resolved)
    bucket.blob(out_prefix + "review_queue.csv").upload_from_string(buf.getvalue(), content_type="text/csv")

    orphan_buf = io.StringIO()
    orphan_writer = csv.writer(orphan_buf)
    orphan_writer.writerow(["orphan_gcs_relative_path"])
    for blob in orphan_blobs:
        orphan_writer.writerow([blob.name[len(prefix):]])
    bucket.blob(out_prefix + "orphan_images.csv").upload_from_string(orphan_buf.getvalue(), content_type="text/csv")
    return report


def promote_incoming_batch(
    incoming_prefix: str,
    batch_id: str,
    source: str,
    bucket_name: str | None = None,
) -> dict:
    bucket_name = bucket_name or get_bucket_name()
    if not batch_id.startswith("BATCH_"):
        raise ValueError("batch_id must start with BATCH_")
    prefix = incoming_prefix.strip("/") + "/"
    raw_prefix = f"raw/batches/{batch_id}/"
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    marker = bucket.blob(raw_prefix + "batch.json")
    if marker.exists(client):
        doc = json.loads(_download_text(marker, "utf-8"))
        doc["already_exists"] = True
        return doc

    incoming = [b for b in client.list_blobs(bucket_name, prefix=prefix) if not b.name.endswith("/")]
    if not incoming:
        raise RuntimeError(f"no objects found under gs://{bucket_name}/{prefix}")
    manifests = [b for b in incoming if b.name.endswith("/fish_manifest.csv") or b.name == prefix + "fish_manifest.csv"]
    if len(manifests) != 1:
        raise RuntimeError(f"expected exactly one fish_manifest.csv, found {len(manifests)}")
    _, rows, malformed = _manifest_rows(manifests[0])
    if malformed:
        raise RuntimeError(f"manifest contains {malformed} malformed CSV rows")
    images = [b for b in incoming if PurePosixPath(b.name).suffix.lower() in IMAGE_EXTS]
    if not images:
        raise RuntimeError("no image objects found in incoming prefix")

    copied = skipped = 0
    for source_blob in incoming:
        rel = source_blob.name[len(prefix):]
        destination_name = raw_prefix + rel
        destination = bucket.blob(destination_name)
        if destination.exists(client):
            destination.reload(client)
            same = destination.size == source_blob.size and (
                not source_blob.md5_hash or destination.md5_hash == source_blob.md5_hash
            )
            if not same:
                raise RuntimeError(f"destination differs from source: gs://{bucket_name}/{destination_name}")
            skipped += 1
            continue
        bucket.copy_blob(source_blob, bucket, destination_name)
        copied += 1

    manifest_rel = manifests[0].name[len(prefix):]
    batch = {
        "batch_id": batch_id,
        "source": source,
        "created_at": utcnow_iso(),
        "image_count": len(images),
        "manifest_rows": len(rows),
        "status": "INGESTED",
        "incoming_uri": f"gs://{bucket_name}/{prefix}",
        "raw_uri": f"gs://{bucket_name}/{raw_prefix}",
        "manifest_uri": f"gs://{bucket_name}/{raw_prefix}{manifest_rel}",
        "copied_objects": copied,
        "skipped_existing_objects": skipped,
    }
    marker.upload_from_string(
        json.dumps(batch, ensure_ascii=False, indent=2),
        content_type="application/json",
        if_generation_match=0,
    )
    return batch


def _audit_queue(bucket: storage.Bucket, batch_id: str) -> dict[str, dict]:
    blob = bucket.blob(f"cleaning/{batch_id}/auto_v1/review_queue.csv")
    if not blob.exists():
        return {}
    try:
        rows = list(csv.DictReader(io.StringIO(_download_text(blob))))
    except Exception:
        return {}
    return {(row.get("image_id") or "").strip(): row for row in rows if (row.get("image_id") or "").strip()}


def sync_batch_registry(db: Session, batch_id: str, bucket_name: str | None = None) -> dict:
    bucket_name = bucket_name or get_bucket_name()
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    prefix = f"raw/batches/{batch_id}"
    marker = bucket.blob(f"{prefix}/batch.json")
    if not marker.exists(client):
        raise RuntimeError(f"batch marker not found: gs://{bucket_name}/{prefix}/batch.json")
    batch_doc = json.loads(_download_text(marker, "utf-8"))

    manifest_uri = batch_doc["manifest_uri"]
    body = manifest_uri[5:]
    manifest_bucket, manifest_object = body.split("/", 1)
    if manifest_bucket != bucket_name:
        raise RuntimeError("manifest bucket differs from target bucket")
    rows = list(csv.DictReader(io.StringIO(_download_text(bucket.blob(manifest_object)))))

    object_names = [b.name for b in client.list_blobs(bucket_name, prefix=prefix + "/")]
    object_set = set(object_names)
    image_objects = [name for name in object_names if PurePosixPath(name).suffix.lower() in IMAGE_EXTS]
    by_basename: dict[str, list[str]] = defaultdict(list)
    for name in image_objects:
        by_basename[PurePosixPath(name).name].append(name)
    audit_by_id = _audit_queue(bucket, batch_id)

    batch = db.get(Batch, batch_id)
    if not batch:
        batch = Batch(
            batch_id=batch_id,
            source=batch_doc.get("source", "unknown"),
            image_count=batch_doc.get("image_count", len(image_objects)),
            manifest_uri=manifest_uri,
            raw_uri=batch_doc["raw_uri"],
            status="REGISTERED",
        )
        db.add(batch)
    else:
        batch.image_count = batch_doc.get("image_count", batch.image_count)
        batch.manifest_uri = manifest_uri
        batch.raw_uri = batch_doc["raw_uri"]
        batch.status = "REGISTERED"

    inserted = updated = missing = 0
    for row in rows:
        file_name = norm_path(row.get("file_name") or row.get("filename"))
        relative_path = norm_path(row.get("relative_path") or row.get("file_path") or row.get("path"))
        object_name = None
        for rel in (relative_path, file_name):
            if not rel:
                continue
            candidate = rel if rel.startswith(prefix + "/") else f"{prefix}/{rel}"
            if candidate in object_set:
                object_name = candidate
                break
        if not object_name and file_name:
            matches = by_basename.get(PurePosixPath(file_name).name, [])
            if len(matches) == 1:
                object_name = matches[0]
        if not object_name:
            missing += 1
            continue

        resolved_file = PurePosixPath(object_name).name
        image_id = str((row.get("image_id") or "").strip() or PurePosixPath(resolved_file).stem)
        existing = db.scalar(
            select(ImageAsset).where(ImageAsset.batch_id == batch_id, ImageAsset.image_id == image_id)
        )
        manifest_review = (row.get("review_status") or "pending").strip().lower()
        if manifest_review not in VALID_REVIEW:
            manifest_review = "pending"
        audit = audit_by_id.get(image_id, {})
        if manifest_review == "pending":
            auto_status = audit.get("auto_status")
            manifest_review = {
                "AUTO_REJECT": "rejected",
                "NEEDS_REVIEW": "needs_review",
                "CANDIDATE": "pending",
            }.get(auto_status, manifest_review)

        claimed_species = (row.get("claimed_species") or row.get("species") or row.get("class_name") or "").strip() or None
        truth_species = (row.get("truth_species") or row.get("species_truth") or "").strip() or None
        # An external manifest may claim a review result, but approved is only valid with explicit Ground Truth.
        if manifest_review == "approved" and not truth_species:
            manifest_review = "pending"
        notes = row.get("notes") or ""
        if audit.get("auto_reasons"):
            auto_note = f"[auto_v1:{audit.get('auto_status')}] {audit.get('auto_reasons')}"
            notes = f"{notes}\n{auto_note}".strip()
        values = {
            "file_name": resolved_file,
            "object_name": object_name,
            "gcs_uri": f"gs://{bucket_name}/{object_name}",
            "source_url": row.get("source_url") or row.get("url"),
            "source_platform": row.get("source_platform") or row.get("platform"),
            "claimed_species": claimed_species,
            "truth_species": truth_species,
            "scene": row.get("scene"),
            "lighting": row.get("lighting"),
            "quality": row.get("image_quality") or row.get("quality") or row.get("quality_score"),
            "group_id": row.get("group_id") or row.get("capture_event_id") or row.get("event_id"),
            "notes": notes or None,
        }
        if existing:
            for key, value in values.items():
                if value is not None:
                    setattr(existing, key, value)
            updated += 1
        else:
            db.add(
                ImageAsset(
                    batch_id=batch_id,
                    image_id=image_id,
                    review_status=manifest_review,
                    truth_status="LIKELY_CORRECT" if manifest_review == "approved" and truth_species else "UNCERTAIN",
                    **values,
                )
            )
            inserted += 1

    db.commit()
    return {
        "batch_id": batch_id,
        "manifest_rows": len(rows),
        "gcs_images": len(image_objects),
        "inserted": inserted,
        "updated": updated,
        "missing": missing,
    }


def stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def choose_split(key: str, seed: int, train: float, val: float) -> str:
    p = stable_fraction(key, seed)
    if p < train:
        return "train"
    if p < train + val:
        return "val"
    return "test"


def approved_summary(db: Session) -> dict:
    truth = func.nullif(func.trim(ImageAsset.truth_species), "")
    rows = db.execute(
        select(ImageAsset.batch_id, truth, func.count())
        .where(ImageAsset.review_status == "approved")
        .group_by(ImageAsset.batch_id, truth)
    ).all()
    batches: dict[str, dict] = {}
    total = 0
    for batch_id, species, count in rows:
        name = species or "未确认真实鱼种"
        entry = batches.setdefault(batch_id, {"batch_id": batch_id, "approved": 0, "species": {}})
        entry["approved"] += count
        entry["species"][name] = count
        total += count
    return {"total_approved": total, "batches": list(batches.values())}


def list_datasets(db: Session) -> list[dict]:
    rows = db.scalars(select(DatasetVersion).order_by(DatasetVersion.created_at.desc())).all()
    return [
        {
            "dataset_version": row.dataset_version,
            "parent_version": row.parent_version,
            "created_at": row.created_at.isoformat(),
            "manifest_uri": row.manifest_uri,
            "train_count": row.train_count,
            "val_count": row.val_count,
            "test_count": row.test_count,
            "git_commit": row.git_commit,
            "status": row.status,
        }
        for row in rows
    ]


def freeze_dataset(
    db: Session,
    dataset_version: str,
    batch_ids: list[str],
    git_commit: str,
    seed: int = 20260826,
    train: float = 0.70,
    val: float = 0.15,
    parent_version: str | None = None,
    bucket_name: str | None = None,
) -> dict:
    raise RuntimeError("Legacy factory.freeze_dataset is disabled; use POST /api/dataset-freeze/preview then POST /api/datasets/freeze")
    bucket_name = bucket_name or get_bucket_name()
    if not dataset_version.startswith("DS_"):
        raise ValueError("dataset_version must start with DS_")
    if not batch_ids:
        raise ValueError("select at least one batch")
    if not (0 < train < 1 and 0 <= val < 1 and train + val < 1):
        raise ValueError("invalid split ratios")
    if db.get(DatasetVersion, dataset_version):
        raise ValueError(f"dataset already registered: {dataset_version}")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    out_prefix = f"datasets/{dataset_version}/"
    marker = bucket.blob(out_prefix + "dataset.json")
    if marker.exists(client):
        raise ValueError(f"dataset already exists in GCS: gs://{bucket_name}/{out_prefix}")

    images = db.scalars(
        select(ImageAsset)
        .where(ImageAsset.review_status == "approved", ImageAsset.batch_id.in_(batch_ids))
        .order_by(ImageAsset.batch_id, ImageAsset.id)
    ).all()
    if not images:
        raise ValueError("no approved images in selected batches")

    seen: set[str] = set()
    frozen: list[dict] = []
    for image in images:
        unique_key = image.gcs_uri
        if unique_key in seen:
            continue
        seen.add(unique_key)
        species = image.truth_species or image.claimed_species or "unknown"
        group = image.group_id or f"{image.batch_id}:{image.image_id}"
        split = choose_split(group, seed, train, val)
        frozen.append(
            {
                "dataset_version": dataset_version,
                "batch_id": image.batch_id,
                "image_id": image.image_id,
                "file_name": image.file_name,
                "gcs_uri": image.gcs_uri,
                "object_name": image.object_name,
                "source_url": image.source_url or "",
                "source_platform": image.source_platform or "",
                "claimed_species": image.claimed_species or "",
                "truth_species": image.truth_species or "",
                "species": species,
                "truth_status": image.truth_status,
                "review_status": image.review_status,
                "group_id": image.group_id or "",
                "split": split,
            }
        )

    fields = list(frozen[0].keys())
    csv_buf = io.StringIO()
    writer = csv.DictWriter(csv_buf, fieldnames=fields)
    writer.writeheader()
    writer.writerows(frozen)
    manifest_uri = f"gs://{bucket_name}/{out_prefix}dataset_manifest.csv"
    bucket.blob(out_prefix + "dataset_manifest.csv").upload_from_string(
        csv_buf.getvalue(), content_type="text/csv", if_generation_match=0
    )

    split_counts = Counter(row["split"] for row in frozen)
    species_counts = Counter(row["species"] for row in frozen)
    meta = {
        "dataset_version": dataset_version,
        "parent_version": parent_version,
        "created_at": utcnow_iso(),
        "git_commit": git_commit or "unknown",
        "seed": seed,
        "batch_ids": batch_ids,
        "image_count": len(frozen),
        "split_counts": dict(split_counts),
        "species_counts": dict(species_counts),
        "manifest_uri": manifest_uri,
        "immutable": True,
    }
    marker.upload_from_string(
        json.dumps(meta, ensure_ascii=False, indent=2),
        content_type="application/json",
        if_generation_match=0,
    )
    db.add(
        DatasetVersion(
            dataset_version=dataset_version,
            parent_version=parent_version,
            manifest_uri=manifest_uri,
            train_count=split_counts.get("train", 0),
            val_count=split_counts.get("val", 0),
            test_count=split_counts.get("test", 0),
            git_commit=git_commit or "unknown",
            status="FROZEN",
        )
    )
    db.commit()
    return meta
