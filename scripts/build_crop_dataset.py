#!/usr/bin/env python3
"""Build DS_CROP_M1_v0.1 from reviewed inference records.

The default source is the registry DB + immutable GCS InferenceRecord assets.
Use ``--freeze`` only after inspecting the validation report; it publishes and
registers the dataset as READY_FOR_TRAINING but never starts a training run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trainer.build_reviewed_datasets import load_record_directory
from trainer.crop_dataset_pipeline import (
    CROP_DATASET_VERSION,
    build_reviewed_crop_dataset,
    build_reviewed_crop_dataset_from_db,
    freeze_crop_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-version", default=CROP_DATASET_VERSION)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--records-root", default=None, help="local JSON InferenceRecord directory; defaults to reviewed registry assets")
    parser.add_argument("--bucket", default=os.getenv("GCS_BUCKET"))
    parser.add_argument("--expand-ratio", type=float, default=0.15)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--git-commit", default=os.getenv("APP_GIT_COMMIT", "unknown"))
    parser.add_argument("--freeze", action="store_true", help="publish/register READY_FOR_TRAINING after validation")
    args = parser.parse_args()

    root_context = tempfile.TemporaryDirectory(prefix="yujian-crop-") if args.output_root is None else None
    output_root = Path(args.output_root) if args.output_root else Path(root_context.name)
    db = None
    try:
        if args.records_root:
            report = build_reviewed_crop_dataset(
                load_record_directory(args.records_root),
                output_root,
                dataset_version=args.dataset_version,
                expand_ratio=args.expand_ratio,
            )
        else:
            from app.db import SessionLocal, init_db

            init_db()
            db = SessionLocal()
            report = build_reviewed_crop_dataset_from_db(
                db,
                output_root,
                dataset_version=args.dataset_version,
                expand_ratio=args.expand_ratio,
                limit=args.limit,
            )
        result: dict = {"build": report, "output_root": str(output_root)}
        if not report.get("validation", {}).get("valid"):
            raise SystemExit(json.dumps(result, ensure_ascii=False, indent=2))
        if args.freeze:
            if not args.bucket:
                raise SystemExit("--bucket or GCS_BUCKET is required with --freeze")
            result["freeze"] = freeze_crop_dataset(
                output_root,
                dataset_version=args.dataset_version,
                bucket_name=args.bucket,
                db=db,
                git_commit=args.git_commit,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        if db is not None:
            db.close()
        if root_context is not None:
            root_context.cleanup()


if __name__ == "__main__":
    main()
