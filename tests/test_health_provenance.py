"""Isolated basic health provenance: no app, model or cloud imports."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch


class HealthProvenanceTest(TestCase):
    def test_basic_health_returns_exact_runtime_source(self):
        source = (Path(__file__).parents[1] / 'app/main.py').read_text()
        function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == 'health')
        function.decorator_list = []
        namespace = {'os': os, 'app': SimpleNamespace(version='test')}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), '<isolated-health>', 'exec'), namespace)
        with patch.dict(os.environ, {'APP_GIT_COMMIT': 'b'*40, 'K_REVISION': 'candidate-00003', 'K_SERVICE': 'test-console'}):
            payload = namespace['health']()
        self.assertEqual(payload, {'status': 'ok', 'version': 'test', 'git_commit': 'b'*40, 'revision': 'candidate-00003', 'service': 'test-console'})
