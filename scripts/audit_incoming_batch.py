#!/usr/bin/env python3
import argparse
import csv
import io
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from pathlib import PurePosixPath

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from google.api_core.retry import Retry
from google.cloud import storage
from app.services.manifest_normalizer import (
    ManifestNormalizationError,
    normalize_manifest_text,
    validate_fish_manifest_text,
)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}
TARGET_SPECIES = {
    '草鱼', '鳙鱼', '白鲢', '鲤鱼', '鲫鱼', '加州鲈', '黑鱼', '黄骨鱼', '青鱼'
}
REQUIRED_COLUMNS = {'image_id', 'claimed_species'}
IMAGE_PATH_COLUMNS = ('image_path', 'file_name', 'filename', 'image_name')
DOWNLOAD_RETRY = Retry(initial=1.0, maximum=20.0, multiplier=2.0, deadline=600.0)


def norm_path(value: str) -> str:
    return str(PurePosixPath((value or '').strip().lstrip('/')))


def load_manifest(blob):
    raw = blob.download_as_bytes(timeout=120, retry=DOWNLOAD_RETRY)
    text = raw.decode('utf-8-sig')
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise RuntimeError('manifest has no header')
    fields = set(reader.fieldnames)
    missing = REQUIRED_COLUMNS - fields
    if not fields.intersection(IMAGE_PATH_COLUMNS):
        missing.add('image_path/file_name')
    if missing:
        raise RuntimeError(f'manifest missing required columns: {sorted(missing)}')
    rows = list(reader)
    malformed = sum(1 for r in rows if None in r)
    return reader.fieldnames, rows, malformed


def select_manifests(blobs, prefix):
    canonical_name = prefix.rstrip('/') + '/metadata/fish_manifest.csv'
    canonical = [b for b in blobs if b.name == canonical_name]
    if canonical:
        return canonical
    return [b for b in blobs if b.name.endswith('/fish_manifest.csv') or b.name == prefix + 'fish_manifest.csv']


def ensure_manifest(blobs, bucket, prefix):
    manifests = select_manifests(blobs, prefix)
    if manifests:
        validate_fish_manifest_text(
            manifests[0].download_as_text(encoding='utf-8-sig'),
            source_name=manifests[0].name,
        )
        return manifests, False

    sources = [
        b for b in blobs
        if b.name.endswith('/metadata/manifest.csv') or b.name.endswith('/manifest.csv')
    ]
    sources.sort(key=lambda b: (0 if b.name.endswith('/metadata/manifest.csv') else 1, len(b.name)))
    if not sources:
        raise ManifestNormalizationError('missing metadata/manifest.csv')
    source = sources[0]
    normalized, rows = normalize_manifest_text(
        source.download_as_text(encoding='utf-8-sig'), source_name=source.name
    )
    output_name = prefix + 'metadata/fish_manifest.csv'
    output = bucket.blob(output_name)
    try:
        output.upload_from_string(
            normalized, content_type='text/csv; charset=utf-8', if_generation_match=0
        )
    except Exception:
        if not output.exists():
            raise
    return [output], True


def main():
    ap = argparse.ArgumentParser(description='Audit one YuJian incoming GCS batch before promotion/review.')
    ap.add_argument('--bucket', required=True)
    ap.add_argument('--incoming-prefix', required=True)
    ap.add_argument('--batch-id', required=True, help='Canonical batch ID to use in generated audit artifacts')
    ap.add_argument('--source', required=True)
    ap.add_argument('--write-report', action='store_true', help='Write report JSON/CSV to GCS cleaning/<batch-id>/')
    args = ap.parse_args()

    prefix = args.incoming_prefix.strip('/') + '/'
    client = storage.Client()
    bucket = client.bucket(args.bucket)

    blobs = [b for b in client.list_blobs(args.bucket, prefix=prefix) if not b.name.endswith('/')]
    if not blobs:
        raise RuntimeError(f'no objects under gs://{args.bucket}/{prefix}')

    try:
        manifests, manifest_normalized = ensure_manifest(blobs, bucket, prefix)
    except ManifestNormalizationError as exc:
        raise SystemExit(f'{exc.code}: {exc.reason}') from exc
    if len(manifests) != 1:
        raise RuntimeError(f'expected exactly one fish_manifest.csv, found {len(manifests)}')

    fieldnames, rows, malformed_rows = load_manifest(manifests[0])
    image_blobs = [b for b in blobs if PurePosixPath(b.name).suffix.lower() in IMAGE_EXTS]

    rel_blob = {b.name[len(prefix):]: b for b in image_blobs}
    basename_index = defaultdict(list)
    for rel, blob in rel_blob.items():
        basename_index[PurePosixPath(rel).name].append(blob)

    id_counts = Counter((r.get('image_id') or '').strip() for r in rows if (r.get('image_id') or '').strip())
    def image_value(row):
        return next((row.get(key) for key in IMAGE_PATH_COLUMNS if (row.get(key) or '').strip()), '')

    file_counts = Counter(norm_path(image_value(r)) for r in rows if image_value(r).strip())
    url_counts = Counter((r.get('source_url') or '').strip() for r in rows if (r.get('source_url') or '').strip())

    linked_blob_names = set()
    resolved = []
    md5_first = {}

    for idx, row in enumerate(rows, start=1):
        image_id = (row.get('image_id') or '').strip()
        file_name = norm_path(image_value(row))
        species = (row.get('claimed_species') or '').strip()
        source_url = (row.get('source_url') or '').strip()
        reasons = []
        status = 'CANDIDATE'
        blob = None

        if not image_id:
            reasons.append('missing_image_id')
        if not file_name:
            reasons.append('missing_file_name')
        if not species:
            reasons.append('missing_claimed_species')
        elif species not in TARGET_SPECIES:
            reasons.append('non_target_or_unknown_species')

        if file_name in rel_blob:
            blob = rel_blob[file_name]
        elif file_name:
            matches = basename_index.get(PurePosixPath(file_name).name, [])
            if len(matches) == 1:
                blob = matches[0]
                reasons.append('resolved_by_basename')
            elif len(matches) > 1:
                reasons.append('ambiguous_image_path')
            else:
                reasons.append('missing_image_object')

        if image_id and id_counts[image_id] > 1:
            reasons.append('duplicate_image_id')
        if file_name and file_counts[file_name] > 1:
            reasons.append('duplicate_file_name')
        if source_url and url_counts[source_url] > 1:
            reasons.append('duplicate_source_url')

        if blob:
            linked_blob_names.add(blob.name)
            if blob.md5_hash:
                if blob.md5_hash in md5_first:
                    reasons.append('exact_duplicate_md5')
                else:
                    md5_first[blob.md5_hash] = blob.name

        hard_reject = {
            'missing_image_id', 'missing_file_name', 'missing_image_object',
            'ambiguous_image_path', 'duplicate_image_id', 'duplicate_file_name',
            'exact_duplicate_md5'
        }
        if any(r in hard_reject for r in reasons):
            status = 'AUTO_REJECT'
        elif any(r in {'missing_claimed_species', 'non_target_or_unknown_species', 'duplicate_source_url'} for r in reasons):
            status = 'NEEDS_REVIEW'

        resolved.append({
            'row_number': idx,
            'image_id': image_id,
            'file_name': file_name,
            'resolved_gcs_name': blob.name if blob else '',
            'claimed_species': species,
            'source_url': source_url,
            'auto_status': status,
            'auto_reasons': ';'.join(reasons),
            'size_bytes': blob.size if blob else '',
            'generation': str(blob.generation) if blob else '',
            'md5_hash': blob.md5_hash if blob else '',
            'review_status': '',
            'review_truth_species': '',
            'review_notes': '',
        })

    orphan_blobs = [b for b in image_blobs if b.name not in linked_blob_names]
    status_counts = Counter(r['auto_status'] for r in resolved)
    species_counts = Counter(r['claimed_species'] for r in resolved)

    report = {
        'batch_id': args.batch_id,
        'source': args.source,
        'incoming_uri': f'gs://{args.bucket}/{prefix}',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'object_count': len(blobs),
        'image_object_count': len(image_blobs),
        'manifest_rows': len(rows),
        'malformed_csv_rows': malformed_rows,
        'linked_unique_images': len(linked_blob_names),
        'orphan_image_count': len(orphan_blobs),
        'status_counts': dict(status_counts),
        'species_counts': dict(species_counts),
        'orphan_images': [b.name[len(prefix):] for b in orphan_blobs],
        'manifest_normalized': manifest_normalized,
        'manifest_path': manifests[0].name[len(prefix):] if manifests[0].name.startswith(prefix) else manifests[0].name,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.write_report:
        out_prefix = f'cleaning/{args.batch_id}/auto_v1/'
        bucket.blob(out_prefix + 'audit_report.json').upload_from_string(
            json.dumps(report, ensure_ascii=False, indent=2), content_type='application/json', if_generation_match=0
        )
        buf = io.StringIO()
        cols = list(resolved[0].keys()) if resolved else []
        writer = csv.DictWriter(buf, fieldnames=cols)
        writer.writeheader()
        writer.writerows(resolved)
        bucket.blob(out_prefix + 'review_queue.csv').upload_from_string(
            buf.getvalue(), content_type='text/csv', if_generation_match=0
        )
        orphan_buf = io.StringIO()
        ow = csv.writer(orphan_buf)
        ow.writerow(['orphan_gcs_relative_path'])
        for b in orphan_blobs:
            ow.writerow([b.name[len(prefix):]])
        bucket.blob(out_prefix + 'orphan_images.csv').upload_from_string(
            orphan_buf.getvalue(), content_type='text/csv', if_generation_match=0
        )
        print(f'wrote gs://{args.bucket}/{out_prefix}')


if __name__ == '__main__':
    main()
