#!/usr/bin/env python3
"""Freeze one immutable YuJian dataset version from reviewed manifests.

Produces a canonical dataset manifest and dataset.json. Split assignment is
deterministic at group level to prevent same-event leakage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

APPROVED = {"approved", "confirmed"}
GROUP_FIELDS = ("group_id", "capture_event_id", "event_id")


def stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def group_key(row: dict[str, str]) -> str:
    for field in GROUP_FIELDS:
        value = (row.get(field) or "").strip()
        if value:
            return f"{field}:{value}"
    image_id = (row.get("image_id") or row.get("file_name") or "").strip()
    if not image_id:
        raise ValueError("row has no group id, image_id, or file_name")
    return f"image:{image_id}"


def choose_split(key: str, seed: int, train: float, val: float) -> str:
    p = stable_fraction(key, seed)
    if p < train:
        return "train"
    if p < train + val:
        return "val"
    return "test"


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["_source_manifest"] = str(path)
                rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-version", required=True)
    ap.add_argument("--manifest", action="append", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--git-commit", required=True)
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    args = ap.parse_args()

    if not args.dataset_version.startswith("DS_"):
        raise SystemExit("dataset version must start with DS_")
    if not (0 < args.train < 1 and 0 <= args.val < 1 and args.train + args.val < 1):
        raise SystemExit("invalid split ratios")

    rows = read_rows([Path(p) for p in args.manifest])
    approved = [r for r in rows if (r.get("review_status") or "").strip().lower() in APPROVED]
    if not approved:
        raise SystemExit("no approved rows found")

    seen_images: set[str] = set()
    frozen: list[dict[str, str]] = []
    for row in approved:
        image_id = (row.get("image_id") or row.get("file_name") or "").strip()
        if image_id in seen_images:
            continue
        seen_images.add(image_id)
        row = dict(row)
        row["dataset_version"] = args.dataset_version
        row["split"] = choose_split(group_key(row), args.seed, args.train, args.val)
        frozen.append(row)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    fields = sorted({k for row in frozen for k in row.keys()})
    manifest_path = out / "dataset_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(frozen)

    species = Counter((r.get("species") or r.get("species_truth") or r.get("claimed_species") or "unknown") for r in frozen)
    splits = Counter(r["split"] for r in frozen)
    meta = {
        "dataset_version": args.dataset_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": args.git_commit,
        "seed": args.seed,
        "source_manifests": args.manifest,
        "image_count": len(frozen),
        "split_counts": dict(splits),
        "species_counts": dict(species),
        "immutable": True,
    }
    (out / "dataset.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
