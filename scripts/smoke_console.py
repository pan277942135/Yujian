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
from app.presence import classify_presence  # noqa: E402,F401  (registers presence table)


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

        fish = classify_presence(
            objects=[{"name": "Fish", "score": 0.91, "vertices": [{"x": 0.1, "y": 0.2}, {"x": 0.8, "y": 0.2}, {"x": 0.8, "y": 0.7}, {"x": 0.1, "y": 0.7}]}],
            labels=[{"name": "Fish", "score": 0.98}, {"name": "Animal", "score": 0.85}],
        )
        assert fish["status"] == "fish_present", fish
        assert fish["fish_count"] == 1, fish
        assert fish["max_box_area_ratio"] > 0.30, fish

        no_fish = classify_presence(
            objects=[{"name": "Person", "score": 0.95, "vertices": []}],
            labels=[
                {"name": "Person", "score": 0.99},
                {"name": "Clothing", "score": 0.94},
                {"name": "Road", "score": 0.90},
                {"name": "Sky", "score": 0.88},
                {"name": "Vehicle", "score": 0.82},
            ],
        )
        assert no_fish["status"] == "no_fish", no_fish

        uncertain = classify_presence(objects=[], labels=[{"name": "Outdoor", "score": 0.64}])
        assert uncertain["status"] == "uncertain", uncertain

        print("Console flywheel + fish presence smoke test: OK")
    finally:
        db.close()


if __name__ == "__main__":
    main()
