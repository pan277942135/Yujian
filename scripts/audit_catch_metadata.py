"""Read-only audit and conservative recovery proposal for legacy fish catches.

Run with the normal database environment configured. No update statement is
issued. Detailed output may contain private locations and should be saved only
to an access-controlled local file, never to public CI logs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from sqlalchemy import text

from app.db import engine


FIELDS = {
    "length_cm": ("length_cm", "length"),
    "weight_kg": ("weight_kg", "weight"),
    "location": ("location", "location_name"),
    "story": ("story",),
}


def valid_candidate(field: str, value: Any) -> Any:
    if field in {"length_cm", "weight_kg"}:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        return number if math.isfinite(number) and 0 < number <= 1000 else None
    if not isinstance(value, str):
        return None
    value = value.strip()
    max_length = 512 if field == "location" else 4096
    return value if value and len(value) <= max_length else None


def classifier_json(raw: str | None) -> dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def audit_rows(records):
    details: list[dict[str, str]] = []
    formal_records = json_only_records = missing_records = conflict_fields = 0
    unrecoverable_records: set[str] = set()
    for record in records:
        legacy = classifier_json(record["classifier_result_json"])
        version = int(record["metadata_version"] or 0)
        formal_present = any(record[field] is not None for field in FIELDS)
        valid_legacy = {
            field: valid_candidate(field, next((legacy[key] for key in keys if key in legacy), None))
            for field, keys in FIELDS.items()
        }
        legacy_present = any(value is not None for value in valid_legacy.values())
        if formal_present:
            formal_records += 1
        if not formal_present and legacy_present and version == 0:
            json_only_records += 1
        if not formal_present and (version > 0 or not legacy_present):
            missing_records += 1
            unrecoverable_records.add(str(record["id"]))

        for field in FIELDS:
            current = record[field]
            candidate = valid_legacy[field] if version == 0 else None
            if current is not None:
                status = "EXISTS"
                suggested = ""
                source = "fish_catches." + field
                if candidate is not None and str(candidate) != str(current):
                    status = "CONFLICT_NO_OVERWRITE"
                    suggested = str(candidate)
                    source = "classifier_result_json"
                    conflict_fields += 1
                    unrecoverable_records.add(str(record["id"]))
            elif version > 0:
                status = "EXPLICITLY_ABSENT"
                suggested = ""
                source = "metadata_version=1"
            elif candidate is not None:
                status = "AUTO_RESTORE_CANDIDATE"
                suggested = str(candidate)
                source = "same recordId: classifier_result_json"
            else:
                status = "NO_VERIFIED_SOURCE"
                suggested = ""
                source = "none"
            details.append(
                {
                    "recordId": str(record["id"]),
                    "field": field,
                    "source": source,
                    "current_value": "" if current is None else str(current),
                    "suggested_value": suggested,
                    "automatic_recovery": "yes" if status == "AUTO_RESTORE_CANDIDATE" else "no",
                    "status": status,
                }
            )
    return details, {
        "total_catch_records": len(records),
        "records_with_formal_metadata": formal_records,
        "records_with_only_classifier_json_values": json_only_records,
        "records_missing_all_metadata": missing_records,
        "field_conflicts": conflict_fields,
        "records_unrecoverable_or_conflicted": len(unrecoverable_records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="private CSV path for recordId-level proposals")
    args = parser.parse_args()

    with engine.connect() as connection:
        if engine.dialect.name == "postgresql":
            connection.exec_driver_sql("SET TRANSACTION READ ONLY")
        elif engine.dialect.name == "sqlite":
            connection.exec_driver_sql("PRAGMA query_only = ON")
        else:
            raise RuntimeError("Audit supports only PostgreSQL or SQLite read-only sessions")
        records = connection.execute(
            text(
                "SELECT id, length_cm, weight_kg, location, story, classifier_result_json, metadata_version "
                "FROM fish_catches ORDER BY id"
            )
        ).mappings().all()

    details, summary = audit_rows(records)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(details[0]) if details else [
                "recordId", "field", "source", "current_value", "suggested_value", "automatic_recovery", "status"
            ])
            writer.writeheader()
            writer.writerows(details)


if __name__ == "__main__":
    main()
