#!/usr/bin/env python3
import argparse
import csv
import io
import json
from datetime import datetime, timezone

from google.cloud import storage

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.webp')


def main():
    ap = argparse.ArgumentParser(
        description='Promote an already-uploaded GCS folder from incoming/ into an immutable raw batch.'
    )
    ap.add_argument('--bucket', required=True)
    ap.add_argument('--batch-id', required=True)
    ap.add_argument('--source', required=True)
    ap.add_argument('--incoming-prefix', required=True,
                    help='Example: incoming/BATCH_20260826_WB_001/')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--delete-source', action='store_true')
    args = ap.parse_args()

    if not args.batch_id.startswith('BATCH_'):
        raise SystemExit('batch-id must start with BATCH_')

    incoming_prefix = args.incoming_prefix.strip('/') + '/'
    raw_prefix = f'raw/batches/{args.batch_id}/'

    client = storage.Client()
    bucket = client.bucket(args.bucket)

    marker = bucket.blob(raw_prefix + 'batch.json')
    if marker.exists(client):
        raise RuntimeError(
            f'batch already exists and is immutable: gs://{args.bucket}/{raw_prefix}'
        )

    incoming = [b for b in client.list_blobs(args.bucket, prefix=incoming_prefix) if not b.name.endswith('/')]
    if not incoming:
        raise RuntimeError(f'no objects found under gs://{args.bucket}/{incoming_prefix}')

    manifests = [b for b in incoming if b.name.endswith('/fish_manifest.csv') or b.name == incoming_prefix + 'fish_manifest.csv']
    if len(manifests) != 1:
        raise RuntimeError(f'expected exactly one fish_manifest.csv, found {len(manifests)}')

    manifest_blob = manifests[0]
    manifest_text = manifest_blob.download_as_text(encoding='utf-8-sig')
    rows = list(csv.DictReader(io.StringIO(manifest_text)))
    if not rows:
        raise RuntimeError('fish_manifest.csv is empty')

    images = [b for b in incoming if b.name.lower().endswith(IMAGE_EXTS)]
    if not images:
        raise RuntimeError('no image objects found in incoming prefix')

    objects = []
    for blob in incoming:
        rel = blob.name[len(incoming_prefix):]
        objects.append({
            'path': rel,
            'size': blob.size,
            'generation': str(blob.generation),
            'md5_hash': blob.md5_hash,
            'crc32c': blob.crc32c,
        })

    batch = {
        'batch_id': args.batch_id,
        'source': args.source,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'image_count': len(images),
        'manifest_rows': len(rows),
        'status': 'INGESTED',
        'incoming_uri': f'gs://{args.bucket}/{incoming_prefix}',
        'raw_uri': f'gs://{args.bucket}/{raw_prefix}',
        'manifest_uri': f'gs://{args.bucket}/{raw_prefix}{manifest_blob.name[len(incoming_prefix):]}',
        'objects': objects,
    }

    print(json.dumps(batch, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    # Server-side GCS copy. Data does not pass through Cloud Shell/local disk.
    for blob in incoming:
        rel = blob.name[len(incoming_prefix):]
        destination_name = raw_prefix + rel
        destination = bucket.blob(destination_name)
        if destination.exists(client):
            raise RuntimeError(f'destination object already exists: gs://{args.bucket}/{destination_name}')
        bucket.copy_blob(blob, bucket, destination_name)

    marker.upload_from_string(
        json.dumps(batch, ensure_ascii=False, indent=2),
        content_type='application/json',
        if_generation_match=0,
    )

    if args.delete_source:
        for blob in incoming:
            blob.delete()

    print(f'promoted gs://{args.bucket}/{incoming_prefix} -> gs://{args.bucket}/{raw_prefix}')


if __name__ == '__main__':
    main()
