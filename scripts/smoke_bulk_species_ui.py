#!/usr/bin/env python3
from pathlib import Path
import re

text = Path("app/templates/bulk_review.html").read_text(encoding="utf-8")

required = [
    'id="selectAll"',
    'id="bulkSpecies"',
    'id="selectedCount"',
    'function selectedIndexes()',
    'function toggleSelectAll(checked)',
    'function applyBulkSpecies()',
    'function renderBulkSpeciesOptions()',
    "document.getElementById('species-'+i).value=species",
    '审核状态未改变',
    "function stateForCard(status)",
    "const initial=stateForCard(x.review_status);",
    "items.forEach((x,i)=>{x._state=x.review_status||'pending'});",
    '手动框选',
    '清空框',
    'function bboxPoint(i,event)',
    'function clearBbox(i)',
    'pointerdown',
]
for token in required:
    assert token in text, f"missing quick-review bulk species control: {token}"

match = re.search(r"function applyBulkSpecies\(\)\{(.*?)\}\nasync function submitPage", text, re.S)
assert match, "applyBulkSpecies function not found"
body = match.group(1)
assert "setState(" not in body, "batch species action must not change review status"
assert "._state=" not in body, "batch species action must not mutate review state"
assert "review_status" not in body
assert "catalog.filter" not in text.split("function renderBulkSpeciesOptions()", 1)[0].split("async function init()", 1)[1], "batch species action must remain species-only"

review_text = Path("app/templates/review.html").read_text(encoding="utf-8")
for token in ("手动框选", "清空框", "function reviewBboxPoint(event)", "function clearCurrentBbox()"):
    assert token in review_text, f"missing single-review bbox control: {token}"

print("Quick-review batch truth-species + bbox UI smoke test: OK")
