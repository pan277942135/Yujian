#!/usr/bin/env python3
from pathlib import Path

p=Path('scripts/smoke_business_state.py')
s=p.read_text(encoding='utf-8')
s=s.replace(
    '        assert preview1["excluded_quality_counts"].get("presence_not_scanned") == 1, preview1\n',
    '        assert preview1["excluded_quality_counts"].get("dedupe_not_scanned") == 1, preview1\n',
    1,
)
p.write_text(s,encoding='utf-8')

p=Path('scripts/smoke_dataset_freeze.py')
s=p.read_text(encoding='utf-8')
needle='                FishPresenceResult(image_asset_id=no_fish.id, batch_id=batch_id, status="no_fish", fish_count=0),\n'
replacement=needle+'                FishPresenceResult(image_asset_id=duplicate.id, batch_id=batch_id, status="single_fish", fish_count=1),\n'
if needle not in s:
    raise RuntimeError('Dataset Freeze smoke presence fixture target not found')
s=s.replace(needle,replacement,1)
p.write_text(s,encoding='utf-8')

Path('scripts/_p0_fix_test.py').unlink()
print('adjusted P0 and legacy Dataset Freeze smoke fixtures')
