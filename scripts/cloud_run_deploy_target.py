#!/usr/bin/env python3
"""Resolve the deployed revision's real route, never another traffic target.

Pure JSON validation: no cloud SDK or HTTP calls, credentials, or mutations.
"""
import argparse
import json
from pathlib import Path


def resolve_target(service, revision, expected_sha):
    status = service.get('status') or {}
    revision_name = (revision.get('metadata') or {}).get('name')
    if not revision_name or revision_name != status.get('latestReadyRevisionName'):
        raise ValueError('Deployed revision is not the expected latest ready revision')
    containers = (revision.get('spec') or {}).get('containers') or []
    env = (containers[0].get('env') if containers else []) or []
    actual_sha = next((e.get('value', '') for e in env if e.get('name') == 'APP_GIT_COMMIT'), '')
    if actual_sha != expected_sha:
        raise ValueError('Revision APP_GIT_COMMIT does not match exact deploy source')
    routes = [t for t in status.get('traffic', []) if t.get('revisionName') == revision_name]
    # A tag routes to one revision; the service URL can route somewhere else.
    tagged = sorted([t for t in routes if t.get('tag') and t.get('url')], key=lambda t: t['tag'])
    if tagged:
        url = tagged[0]['url']
        route_type = 'REVISION_TAG'
    elif sum(t.get('percent', 0) for t in routes) == 100:
        url = status.get('url')
        route_type = 'SOLE_DEFAULT_TRAFFIC'
    else:
        raise ValueError('Revision lacks a tag URL and exclusive default traffic; refuse to smoke another revision')
    if not isinstance(url, str) or not url.startswith('https://'):
        raise ValueError('Missing HTTPS deployed-revision route')
    return {'service_url': url.rstrip('/'), 'revision': revision_name,
            'git_commit': actual_sha, 'route_type': route_type,
            'default_service_url': status.get('url')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--service-json', type=Path, required=True)
    parser.add_argument('--revision-json', type=Path, required=True)
    parser.add_argument('--expected-sha', required=True)
    args = parser.parse_args()
    if len(args.expected_sha) != 40 or any(c not in '0123456789abcdef' for c in args.expected_sha):
        parser.error('expected-sha must be a full lowercase hexadecimal Git SHA')
    print(json.dumps(resolve_target(json.loads(args.service_json.read_text()),
                                    json.loads(args.revision_json.read_text()), args.expected_sha)))


if __name__ == '__main__':
    main()
