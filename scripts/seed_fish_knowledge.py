#!/usr/bin/env python3
"""Initialize the v2 Fish Knowledge editorial content package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db import SessionLocal, init_db  # noqa: E402
from app.fish_knowledge.content_seed import seed_fish_knowledge_content  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-file", type=Path, default=None, help="optional Fish Knowledge JSON seed path")
    args = parser.parse_args()

    init_db()
    with SessionLocal() as db:
        result = seed_fish_knowledge_content(db, args.seed_file)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

