#!/usr/bin/env python3
import argparse
import csv
import io
import json
import os
import sys
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from google.cloud import storage
from sqlalchemy import select

from app.db import SessionLocal, init_db
from app.models import Batch, ImageAsset

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VALID_REVIEW = {"approved", "needs_review", "rejected", "hard_case", "pending"}


def gs_object(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"not a GCS URI: {uri}")
    body = uri[5:]
    bucket, obj = body.split("/", 1)
    return bucket, obj


def norm_path(value: str | None) -> str:
    return (value or "").replace("\\", "/").lstrip("./")


def first(row: dict, *keys: str):
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def main():
    ap = argparse.ArgumentParser(description="Sync one immutable GCS batch into YuJian Registry")
    ap.add_argument("--bucket", default=os.getenv("GCS_BUCKET"), required=False)
    ap.add_argument("--batch-id", required=True)
    args = ap.parse_args()
    if not args.bucket:
        raise SystemExit("--bucket or GCS_BUCKET is required")

    init_db()
    client = storage.Client()
    bucket = client.bucket(args.bucket)
    prefix = f"raw/batches/{args.batch_id}"
    marker = bucket.blob(f"{prefix}/batch.json")
    if not marker.exists(client):
        raise SystemExit(f"batch marker not found: gs://{args.bucket}/{prefix}/batch.json")
    batch_doc = json.loads(marker.download_as_text(encoding="utf-8"))

    manifest_bucket, manifest_object = gs_object(batch_doc["manifest_uri"])
    if manifest_bucket != args.bucket:
        raise SystemExit("manifest bucket differs from target bucket")
    manifest_blob = bucket.blob(manifest_object)
    rows = list(csv.DictReader(io.StringIO(manifest_blob.download_as_text(encoding="utf-8-sig"))))

    objects = [b.name for b in client.list_blobs(args.bucket, prefix=prefix + "/")]
    object_set = set(objects)
    image_objects = [o for o in objects if PurePosixPath(o).suffix.lower() in IMAGE_EXTS]
    by_basename: dict[str, list[str]] = {}
    for obj in image_objects:
        by_basename.setdefault(PurePosixPath(obj).name, []).append(obj)

    db = SessionLocal()
    inserted = updated = missing = 0
    try:
        batch = db.get(Batch, args.batch_id)
        if not batch:
            batch = Batch(
                batch_id=args.batch_id,
                source=batch_doc.get("source", "unknown"),
                image_count=batch_doc.get("image_count", len(image_objects)),
                manifest_uri=batch_doc["manifest_uri"],
                raw_uri=batch_doc["raw_uri"],
                status=batch_doc.get("status", "INGESTED"),
            )
            db.add(batch)
        else:
            batch.image_count = batch_doc.get("image_count", batch.image_count)
            batch.manifest_uri = batch_doc["manifest_uri"]
            batch.raw_uri = batch_doc["raw_uri"]

        for index, row in enumerate(rows, start=1):
            file_name = first(row, "image_path", "file_name", "filename", "image_name")
            rel = norm_path(first(row, "relative_path", "file_path", "path"))
            object_name = None
            if rel:
                candidate = rel if rel.startswith(prefix + "/") else f"{prefix}/{rel}"
                if candidate in object_set:
                    object_name = candidate
            if not object_name and file_name:
                matches = by_basename.get(PurePosixPath(norm_path(file_name)).name, [])
                if len(matches) == 1:
                    object_name = matches[0]
            if not object_name:
                missing += 1
                print(f"WARN row {index}: cannot resolve image object for {file_name or rel}")
                continue

            resolved_file = PurePosixPath(object_name).name
            image_id = str(first(row, "image_id") or PurePosixPath(resolved_file).stem)
            existing = db.scalar(
                select(ImageAsset).where(
                    ImageAsset.batch_id == args.batch_id,
                    ImageAsset.image_id == image_id,
                )
            )
            incoming_review = (first(row, "review_status") or "pending").strip().lower()
            if incoming_review not in VALID_REVIEW:
                incoming_review = "pending"

            values = dict(
                file_name=resolved_file,
                object_name=object_name,
                gcs_uri=f"gs://{args.bucket}/{object_name}",
                source_url=first(row, "source_url", "url"),
                source_platform=first(row, "source_platform", "platform", "source"),
                claimed_species=first(row, "claimed_species", "species", "class_name"),
                scene=first(row, "scene"),
                lighting=first(row, "lighting"),
                quality=first(row, "image_quality", "quality", "quality_score"),
                group_id=first(row, "group_id", "capture_event_id", "event_id"),
                notes=first(row, "notes"),
            )
            if existing:
                for key, value in values.items():
                    if value is not None:
                        setattr(existing, key, value)
                updated += 1
            else:
                db.add(
                    ImageAsset(
                        batch_id=args.batch_id,
                        image_id=image_id,
                        review_status=incoming_review,
                        truth_status="LIKELY_CORRECT" if incoming_review == "approved" else "UNCERTAIN",
                        **values,
                    )
                )
                inserted += 1

        db.commit()
    finally:
        db.close()

    print(json.dumps({
        "batch_id": args.batch_id,
        "manifest_rows": len(rows),
        "gcs_images": len(image_objects),
        "inserted": inserted,
        "updated": updated,
        "missing": missing,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
