"""Negative scope routing assertions; no application, network or cloud imports."""
import importlib.util
from pathlib import Path
from unittest import TestCase

spec = importlib.util.spec_from_file_location('scope', Path(__file__).parents[1] / 'scripts/classify_candidate_infra_change.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CandidateInfraScopeTest(TestCase):
    def test_exact_task_paths_are_allowed(self):
        self.assertTrue(module.infrastructure_only(module.ALLOWED_PATHS))

    def test_product_file_never_receives_infra_pass(self):
        self.assertFalse(module.infrastructure_only(['.github/workflows/ci.yml', 'app/main.py']))
        self.assertFalse(module.infrastructure_only(['app/catch_recognition_api.py']))

    def test_model_and_unknown_tests_are_not_safe_infra_scope(self):
        self.assertFalse(module.infrastructure_only(['tests/test_production_recognition_pipeline.py']))
        self.assertFalse(module.infrastructure_only(['app/detector_runtime.py']))
        self.assertFalse(module.infrastructure_only(['docs/arbitrary.md']))

    def test_missing_scope_fails_closed(self):
        self.assertFalse(module.infrastructure_only([]))
