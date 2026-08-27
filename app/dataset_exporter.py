"""Dataset Freeze V0.1 export service.

Writes immutable metadata artifacts for training datasets.
"""

import csv
import io
import json
from collections import Counter
from datetime import datetime, timezone

from google.cloud import storage


def utcnow_iso():
    return datetime.now(timezone.utc).isoformat()


def upload_text(bucket_name: str, object_name: str, content: str, content_type: str):
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(object_name)
    blob.upload_from_string(content, content_type=content_type)
    return f"gs://{bucket_name}/{object_name}"


def export_dataset_metadata(
    bucket_name: str,
    dataset_version: str,
    items: list[dict],
    class_mapping: dict,
):
    prefix = f"datasets/{dataset_version}/metadata"

    csv_buffer = io.StringIO()
    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=["image_id", "batch_id", "gcs_uri", "species", "split"],
    )
    writer.writeheader()
    writer.writerows(items)

    manifest_uri = upload_text(
        bucket_name,
        f"{prefix}/dataset_manifest.csv",
        csv_buffer.getvalue(),
        "text/csv",
    )

    class_uri = upload_text(
        bucket_name,
        f"{prefix}/class_mapping.json",
        json.dumps(class_mapping, ensure_ascii=False, indent=2),
        "application/json",
    )

    report = {
        "dataset_version": dataset_version,
        "created_at": utcnow_iso(),
        "total_images": len(items),
        "species_distribution": dict(Counter(x["species"] for x in items)),
        "split_distribution": dict(Counter(x["split"] for x in items)),
        "status": "FROZEN",
    }

    report_uri = upload_text(
        bucket_name,
        f"{prefix}/freeze_report.json",
        json.dumps(report, ensure_ascii=False, indent=2),
        "application/json",
    )

    return {
        "manifest_uri": manifest_uri,
        "class_map_uri": class_uri,
        "report_uri": report_uri,
        "report": report,
    }
