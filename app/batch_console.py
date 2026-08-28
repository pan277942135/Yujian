from __future__ import annotations

import csv
import io
import json
from pathlib import PurePosixPath

from google.cloud import storage
from sqlalchemy.orm import Session

from app.factory import IMAGE_EXTS, audit_incoming_batch, get_bucket_name
from app.flywheel import species_names


def _audit_reports(client: storage.Client, bucket_name: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for blob in client.list_blobs(bucket_name, prefix="cleaning/"):
        if not blob.name.endswith("/auto_v1/audit_report.json"):
            continue
        try:
            doc = json.loads(blob.download_as_text(encoding="utf-8"))
        except Exception:
            continue
        uri = (doc.get("incoming_uri") or "").rstrip("/") + "/"
        if uri != "/":
            result[uri] = doc
    return result


def list_incoming_batches(bucket_name: str | None = None) -> list[dict]:
    bucket_name = bucket_name or get_bucket_name()
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    reports = _audit_reports(client, bucket_name)

    iterator = client.list_blobs(bucket_name, prefix="incoming/", delimiter="/")
    list(iterator)  # populate iterator.prefixes
    prefixes = sorted(iterator.prefixes)

    result = []
    for prefix in prefixes:
        blobs = [b for b in client.list_blobs(bucket_name, prefix=prefix) if not b.name.endswith("/")]
        images = [b for b in blobs if PurePosixPath(b.name).suffix.lower() in IMAGE_EXTS]
        manifests = [b for b in blobs if b.name.endswith("/fish_manifest.csv") or b.name == prefix + "fish_manifest.csv"]
        uri = f"gs://{bucket_name}/{prefix}"
        audit = reports.get(uri.rstrip("/") + "/")
        canonical_batch = (audit or {}).get("batch_id") or prefix.rstrip("/").split("/")[-1]
        source = (audit or {}).get("source") or ("pilot" if "PILOT" in canonical_batch.upper() else "")
        raw_marker = bucket.blob(f"raw/batches/{canonical_batch}/batch.json")
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


def audit_with_species_catalog(
    db: Session,
    *,
    incoming_prefix: str,
    batch_id: str,
    source: str,
    bucket_name: str | None = None,
) -> dict:
    """Run deterministic audit, then recognize newly-activated Catalog species.

    The low-level audit intentionally remains conservative. This Console adapter
    upgrades rows whose *only* problem is the old fixed target-species list when
    their claimed species is now active/candidate in Species Catalog.
    """
    bucket_name = bucket_name or get_bucket_name()
    report = audit_incoming_batch(incoming_prefix, batch_id, source, bucket_name)
    accepted_names = set(species_names(db, include_candidates=True))
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    queue_blob = bucket.blob(f"cleaning/{batch_id}/auto_v1/review_queue.csv")
    if not queue_blob.exists(client):
        return report

    rows = list(csv.DictReader(io.StringIO(queue_blob.download_as_text(encoding="utf-8-sig"))))
    changed = 0
    for row in rows:
        if row.get("auto_status") != "NEEDS_REVIEW":
            continue
        reasons = [x for x in (row.get("auto_reasons") or "").split(";") if x]
        if row.get("claimed_species") in accepted_names and set(reasons) <= {"non_target_or_unknown_species", "resolved_by_basename"}:
            row["auto_status"] = "CANDIDATE"
            row["auto_reasons"] = ";".join(x for x in reasons if x != "non_target_or_unknown_species")
            changed += 1

    if changed:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        queue_blob.upload_from_string(buf.getvalue(), content_type="text/csv")
        counts: dict[str, int] = {}
        for row in rows:
            status = row.get("auto_status") or "UNKNOWN"
            counts[status] = counts.get(status, 0) + 1
        report["status_counts"] = counts
        report["catalog_species_upgraded"] = changed
        report_blob = bucket.blob(f"cleaning/{batch_id}/auto_v1/audit_report.json")
        report_blob.upload_from_string(json.dumps(report, ensure_ascii=False, indent=2), content_type="application/json")
    return report
