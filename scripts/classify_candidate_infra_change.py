#!/usr/bin/env python3
"""Fail-closed, task-local safe CI routing. Does not grant product test PASS."""
import argparse
from pathlib import Path

ALLOWED_PATHS = frozenset({
    '.github/workflows/ci.yml',
    '.github/workflows/uat-deploy.yml',
    '.github/workflows/catch-recognition-pr177-no-traffic-deploy.yml',
    'scripts/classify_candidate_infra_change.py',
    'scripts/cloud_run_deploy_target.py',
    'scripts/deploy_console_runtime.sh',
    'tests/test_cloud_run_deploy_target.py',
    'tests/test_candidate_infra_scope.py',
    'docs/catch_candidate_provenance_repair_v1.md',
    'docs/catch_web_preview_pin_v2.md',
})


def infrastructure_only(paths):
    paths = [p for p in paths if p]
    return bool(paths) and all(p in ALLOWED_PATHS for p in paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('changed_files', type=Path)
    args = parser.parse_args()
    print('true' if infrastructure_only(args.changed_files.read_text().splitlines()) else 'false')


if __name__ == '__main__':
    main()
