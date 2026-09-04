#!/usr/bin/env python3
"""Generate MODEL_COMPARE_REPORT from two local evaluation artifact bundles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.model_compare import compare_model_artifacts, write_model_compare_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="MODEL_M1_v0.5 artifact directory or metrics.json")
    parser.add_argument("--candidate", required=True, help="MODEL_CROP_M1_v0.1 artifact directory or metrics.json")
    parser.add_argument("--output", default="MODEL_COMPARE_REPORT.json")
    args = parser.parse_args()
    report = compare_model_artifacts(args.baseline, args.candidate)
    write_model_compare_report(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
