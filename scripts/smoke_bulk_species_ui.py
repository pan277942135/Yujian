#!/usr/bin/env python3
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

text = Path("app/templates/bulk_review.html").read_text(encoding="utf-8")

required = [
    'id="selectAll"',
    'id="bulkSpecies"',
    'id="selectedCount"',
    'function selectedIndexes()',
    'function toggleSelectAll(checked)',
    'function assignableSpeciesRows()',
    'function assignableSpeciesNames()',
    'function renderBulkSpeciesOptions()',
    'function applyBulkSpecies()',
    "['active','candidate']",
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

options_match = re.search(
    r"function renderBulkSpeciesOptions\(\)\{(.*?)\}\nfunction renderTabs",
    text,
    re.S,
)
assert options_match, "renderBulkSpeciesOptions function not found"
options_body = options_match.group(1)
assert "assignableSpeciesRows()" in options_body
assert "speciesRows" not in options_body, "review groups must not populate the Ground Truth selector"
assert "common_name_zh" in options_body, "Ground Truth options must use SpeciesCatalog names"

apply_match = re.search(
    r"function applyBulkSpecies\(\)\{(.*?)\}\nfunction boxValues",
    text,
    re.S,
)
assert apply_match, "applyBulkSpecies function not found"
apply_body = apply_match.group(1)
assert "change.select.options" in apply_body, "every card must validate that the option exists"
assert "option.value===species" in apply_body, "bulk assignment must validate the canonical option"
assert "change.select.value=species" in apply_body, "bulk assignment must perform the selected assignment"
assert "change.select.value!==species" in apply_body, "bulk assignment must verify the assignment"
assert "change.previous" in apply_body, "bulk assignment must support rollback"
assert "未修改任何卡片" in apply_body, "invalid bulk assignment must not report false success"
assert "msg(" in apply_body and ",true)" in apply_body, "invalid bulk assignment must surface an error"
assert "document.getElementById('species-'+i).value=species" not in apply_body
assert "setState(" not in apply_body, "batch species action must not change review status"
assert "._state=" not in apply_body, "batch species action must not mutate review state"
assert "review_status" not in apply_body

submit_match = re.search(r"async function submitPage\(\)\{(.*?)\n  const fallbackBatch", text, re.S)
assert submit_match, "submitPage validation block not found"
submit_body = submit_match.group(1)
for token in (
    "const assignableNames=assignableSpeciesNames()",
    "!assignableNames.has(truth)",
    "canonical Ground Truth",
):
    assert token in submit_body, f"submit must enforce canonical Ground Truth: {token}"

load_species = text.split("async function loadSpecies()", 1)[1].split("function renderBulkSpeciesOptions()", 1)[0]
assert "/api/bulk-review/species?" in load_species, "review groups must come from the review-group endpoint"
assert "speciesRows=" in load_species, "review groups must remain separate UI state"

alias_text = Path("app/species_alias.py").read_text(encoding="utf-8")
assert 'SEARCH_ONLY_ALIASES = {"鲢鱼", "鳊鱼"}' in alias_text
from app.species_alias import normalize_species_name

assert normalize_species_name("鳊鱼") == "鳊鱼", "ambiguous 鳊鱼 alias must not auto-map"
assert normalize_species_name("鲢鱼") == "鲢鱼", "ambiguous 鲢鱼 alias must not auto-map"

review_text = Path("app/templates/review.html").read_text(encoding="utf-8")
for token in ("手动框选", "清空框", "function reviewBboxPoint(event)", "function clearCurrentBbox()"):
    assert token in review_text, f"missing single-review bbox control: {token}"

print("Quick-review canonical species mapping UI smoke test: OK")
