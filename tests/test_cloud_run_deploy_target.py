"""Pure deployment route/provenance cases. No application/model/cloud imports."""
import ast
import os
from types import SimpleNamespace
from unittest.mock import patch
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('target', Path(__file__).parents[1] / 'scripts/cloud_run_deploy_target.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DeployTargetTest(unittest.TestCase):
    def setUp(self):
        self.sha = 'a' * 40
        self.revision = {'metadata': {'name': 'service-00002-new'}, 'spec': {'containers': [{'env': [{'name': 'APP_GIT_COMMIT', 'value': self.sha}]}]}}
        self.service = {'status': {'latestReadyRevisionName': 'service-00002-new', 'url': 'https://service.run.app', 'traffic': [
            {'revisionName': 'service-00001-old', 'percent': 100},
            {'revisionName': 'service-00002-new', 'tag': 'candidate', 'url': 'https://candidate---service.run.app', 'percent': 0}]}}

    def test_zero_traffic_candidate_uses_tag_not_service_url(self):
        result = module.resolve_target(self.service, self.revision, self.sha)
        self.assertEqual(result['service_url'], 'https://candidate---service.run.app')
        self.assertEqual(result['route_type'], 'REVISION_TAG')

    def test_wrong_revision_sha_rejected_before_http(self):
        with self.assertRaisesRegex(ValueError, 'APP_GIT_COMMIT'):
            module.resolve_target(self.service, self.revision, 'b' * 40)

    def test_unroutable_zero_traffic_revision_never_uses_service_url(self):
        self.service['status']['traffic'].pop()
        with self.assertRaisesRegex(ValueError, 'refuse to smoke another revision'):
            module.resolve_target(self.service, self.revision, self.sha)

    def test_default_url_allowed_only_for_exclusive_revision(self):
        self.service['status']['traffic'] = [{'revisionName': 'service-00002-new', 'percent': 100}]
        self.assertEqual(module.resolve_target(self.service, self.revision, self.sha)['service_url'], 'https://service.run.app')

    def test_split_default_traffic_requires_exact_tag(self):
        self.service['status']['traffic'] = [{'revisionName': 'service-00002-new', 'percent': 50}, {'revisionName': 'old', 'percent': 50}]
        with self.assertRaises(ValueError):
            module.resolve_target(self.service, self.revision, self.sha)

    def test_latest_ready_revision_race_rejected(self):
        self.revision['metadata']['name'] = 'service-00003-race'
        with self.assertRaisesRegex(ValueError, 'latest ready'):
            module.resolve_target(self.service, self.revision, self.sha)


if __name__ == '__main__':
    unittest.main()
