#!/usr/bin/env python3
"""Capture the non-mutating authority proof used by the cleanup closure window."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import storage

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.db import SessionLocal
from app.historical_duplicate_closure import authority_fingerprint, write_fence_active


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture the historical cleanup authority fingerprint")
    parser.add_argument("--output", default="/tmp/historical-duplicate-closure-fingerprint.json")
    args = parser.parse_args(argv)

    db = SessionLocal()
    try:
        payload = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "write_fence_active": write_fence_active(),
            "app_git_commit": os.getenv("APP_GIT_COMMIT", "").strip() or None,
            **authority_fingerprint(db),
        }
    finally:
        db.close()

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    destination.write_text(body, encoding="utf-8")
    gcs_uri = os.getenv("CLOSURE_FINGERPRINT_GCS_URI", "").strip()
    if gcs_uri:
        if not gcs_uri.startswith("gs://") or "/" not in gcs_uri[5:]:
            raise SystemExit("CLOSURE_FINGERPRINT_GCS_URI must be a gs://bucket/object URI")
        bucket_name, object_name = gcs_uri[5:].split("/", 1)
        storage.Client().bucket(bucket_name).blob(object_name).upload_from_string(
            body, content_type="application/json"
        )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
