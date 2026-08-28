#!/usr/bin/env python3
from pathlib import Path
p=Path('scripts/smoke_business_state.py')
s=p.read_text(encoding='utf-8')
s=s.replace('        assert preview1["excluded_quality_counts"].get("presence_not_scanned") == 1, preview1\n','        assert preview1["excluded_quality_counts"].get("dedupe_not_scanned") == 1, preview1\n',1)
p.write_text(s,encoding='utf-8')
Path('scripts/_p0_fix_test.py').unlink()
print('adjusted P0 smoke expectation')
