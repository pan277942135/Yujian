#!/usr/bin/env python3
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("REGISTRY_DB_URL", "sqlite:///:memory:")

from app.db import SessionLocal, init_db  # noqa: E402
from app.flywheel import (  # noqa: E402
    create_species_candidate,
    ensure_species_catalog,
    flywheel_summary,
    list_species,
    record_feedback,
)


def main():
    init_db()
    db = SessionLocal()
    try:
        ensure_species_catalog(db)
        seed = list_species(db)
        assert len(seed) >= 10, seed
        assert any(x["species_key"] == "grass_carp" and x["status"] == "active" for x in seed)

        candidate = create_species_candidate(db, common_name_zh="测试鱼种", species_key="test_species")
        assert candidate["status"] == "candidate"

        feedback = record_feedback(
            db,
            source_event_id="SMOKE_FEEDBACK_001",
            feedback_type="corrected",
            predicted_species="鲫鱼",
            corrected_species="测试新鱼种二",
            confidence=0.51,
        )
        assert feedback["feedback_type"] == "new_species_candidate"
        names = {x["common_name_zh"]: x for x in list_species(db)}
        assert names["测试新鱼种二"]["status"] == "candidate"

        summary = flywheel_summary(db)
        assert summary["active_species"] >= 10
        assert summary["candidate_species"] >= 2
        assert summary["new_feedback"] == 1
        print("Console flywheel smoke test: OK")
    finally:
        db.close()


if __name__ == "__main__":
    main()
