#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import storage


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def unpack_input(source: Path) -> Path:
    if source.is_dir():
        return source
    if source.suffix.lower() != '.zip':
        raise ValueError('input must be a directory or .zip')
    tmp = Path(tempfile.mkdtemp(prefix='yujian_batch_'))
    with zipfile.ZipFile(source) as zf:
        zf.extractall(tmp)
    children = [p for p in tmp.iterdir() if p.name != '__MACOSX']
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return tmp


def find_manifest(root: Path) -> Path:
    candidates = list(root.rglob('fish_manifest.csv'))
    if len(candidates) != 1:
        raise RuntimeError(f'expected exactly one fish_manifest.csv, found {len(candidates)}')
    return candidates[0]


def read_manifest(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError('fish_manifest.csv is empty')
    return rows


def collect_images(root: Path):
    exts = {'.jpg', '.jpeg', '.png', '.webp'}
    return sorted([p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in exts])


def main():
    ap = argparse.ArgumentParser(description='Ingest one immutable YuJian data batch into GCS')
    ap.add_argument('--bucket', required=True)
    ap.add_argument('--batch-id', required=True)
    ap.add_argument('--source', required=True)
    ap.add_argument('--input', required=True)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    if not args.batch_id.startswith('BATCH_'):
        raise SystemExit('batch-id must start with BATCH_')

    root = unpack_input(Path(args.input).resolve())
    manifest = find_manifest(root)
    rows = read_manifest(manifest)
    images = collect_images(root)
    if not images:
        raise RuntimeError('no images found')

    client = storage.Client()
    bucket = client.bucket(args.bucket)
    prefix = f'raw/batches/{args.batch_id}'
    marker = bucket.blob(f'{prefix}/batch.json')
    if marker.exists(client):
        raise RuntimeError(f'batch already exists and is immutable: gs://{args.bucket}/{prefix}')

    hashes = {}
    for image in images:
        digest = sha256(image)
        hashes[str(image.relative_to(root))] = digest

    batch = {
        'batch_id': args.batch_id,
        'source': args.source,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'image_count': len(images),
        'manifest_rows': len(rows),
        'status': 'INGESTED',
        'raw_uri': f'gs://{args.bucket}/{prefix}/',
        'manifest_uri': f'gs://{args.bucket}/{prefix}/{manifest.relative_to(root).as_posix()}',
        'sha256': hashes,
    }

    print(json.dumps(batch, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    for path in [p for p in root.rglob('*') if p.is_file()]:
        rel = path.relative_to(root).as_posix()
        blob = bucket.blob(f'{prefix}/{rel}')
        blob.upload_from_filename(str(path))

    marker.upload_from_string(
        json.dumps(batch, ensure_ascii=False, indent=2),
        content_type='application/json',
        if_generation_match=0,
    )
    print(f'uploaded gs://{args.bucket}/{prefix}/')


if __name__ == '__main__':
    main()
