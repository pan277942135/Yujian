from __future__ import annotations

import json

from app.intelligence_api import analyze_and_write_artifacts, build_intelligence_payload


class EmptyDB:
    def scalars(self, _statement):
        return _Rows([])

    def scalar(self, _statement):
        return None

    def get(self, _model, _key):
        return None


class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def all(self):
        return list(self.rows)


def evaluation_document():
    return {
        "model_version": "MODEL_M1_v0.3",
        "classes": [{"class_index": 0, "species_key": "grass_carp"}, {"class_index": 1, "species_key": "common_carp"}],
        "test": {"confusion_matrix": [[7, 3], [1, 8]]},
        "samples": [],
    }


def test_intelligence_payload_keeps_analysis_and_task_proposal_separate():
    payload = build_intelligence_payload(EmptyDB(), evaluation_document=evaluation_document())
    assert payload["model"]["model_version"] == "MODEL_M1_v0.3"
    assert payload["confusion_report"]["top_confusions"][0]["priority"] == "P0"
    assert payload["production_tasks"][0]["batch_suggestion"]["source"] == "MODEL_ERROR_DRIVEN"
    assert payload["production_tasks"][0]["safety"]["creates_batch"] is False


def test_analyze_and_write_artifacts_writes_three_reviewable_outputs(tmp_path):
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(json.dumps(evaluation_document()), encoding="utf-8")
    result = analyze_and_write_artifacts(EmptyDB(), evaluation_path=str(evaluation_path), output_root=tmp_path / "out")
    assert (tmp_path / "out" / "MODEL_M1_v0.3" / "confusion_report.json").exists()
    assert (tmp_path / "out" / "MODEL_M1_v0.3" / "data_gap_report.json").exists()
    assert (tmp_path / "out" / "MODEL_M1_v0.3" / "DATA_PRODUCTION_TASK.json").exists()
    assert result["task"]["task_type"] == "HARD_CASE_COLLECTION"
