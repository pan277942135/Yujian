#!/usr/bin/env python3
"""Create an immutable training-run descriptor before GPU execution."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--dataset-version", required=True)
    ap.add_argument("--git-commit", required=True)
    ap.add_argument("--model-family", required=True)
    ap.add_argument("--params-json", required=True, help="JSON object or path to .json")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if not args.run_id.startswith("RUN_"):
        raise SystemExit("run-id must start with RUN_")

    raw = Path(args.params_json)
    params = json.loads(raw.read_text(encoding="utf-8")) if raw.exists() else json.loads(args.params_json)
    if not isinstance(params, dict):
        raise SystemExit("params-json must be a JSON object")

    descriptor = {
        "run_id": args.run_id,
        "dataset_version": args.dataset_version,
        "git_commit": args.git_commit,
        "model_family": args.model_family,
        "params": params,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "REGISTERED",
        "immutable": True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing run descriptor: {output}")
    output.write_text(json.dumps(descriptor, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(descriptor, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
