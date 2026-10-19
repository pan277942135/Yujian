# Case A low-objectness root-cause investigation V1

**Task:** `CASE_A_LOW_OBJECTNESS_ROOT_CAUSE_V1`
**Mode:** Evidence first; no model or production inference changes.
**Investigation date:** 2026-10-19 UTC
**Backend repository:** `pan277942135/Yujian`
**Android consumer:** `pan277942135/Yujian_App`

## 1. Conclusion

**The underlying reason the correct Case A boxes received very low objectness is not yet established.** The only Case A scores available are from the historical P0 report, not a new replay. That report's highest-IoU correct candidates had moderate fish-class probabilities (0.584–0.670) but objectness of only 0.001889–0.006338. The declared decoder multiplies those fields, so objectness is the immediate numerical factor suppressing confidence below the unchanged 0.20 weak threshold.

The artifact chain now has a confirmed identity problem: the historical Case A ONNX and released Android bundle are `12b97f7c…`, while the last observed production GCS ONNX was `2338f44c…`; an October 7 production smoke failed because the GCS ONNX did not match the Golden Manifest. Current GCS generation and bytes cannot be reread because GCS returns 403 while the owning project's billing account is in state `absent`. The Case A original bytes also could not be recovered by SHA from the authorized repository/workspace sources. Consequently, the old low-objectness rows cannot be attributed to the currently deployed production model, and no Case A orientation replay was run in this investigation.

**Root-cause classification:** `ARTIFACT_MISMATCH` is confirmed for the last observed production/GitHub release chain. `BLOCKED_INSUFFICIENT_EVIDENCE` is the conclusion for the cause of Case A's low objectness. `TRAINING_LABEL_NOISE` is a high-priority hypothesis supported by a concrete builder logic risk, but no affected DET_DS_v0.1 row has been identified. `DOMAIN_RECALL_GAP` is not established for the last observed GCS artifact. No model, threshold, decoder, NMS, quality gate, or retry policy was changed.

| Question | Finding | Evidence grade |
|---|---|---|
| Did the historical diagnostic use the same ONNX as the Android Release bundle? | Yes. Both are SHA `12b97f7c…`; bundle ZIP hash and extracted ONNX hash were checked separately. | Verified |
| Was that the same as the last observed production GCS ONNX? | No. The last observed GCS hash was `2338f44c…`; runtime smoke rejected it against Golden Manifest. | Verified historical observation |
| Is the current GCS object SHA/generation known? | No. GCS read blocked by billing-account 403; public detector health returned 503. | Blocked |
| Is the Case A exact source available? | No. SHA search in available repository checkouts and local registered image assets did not find it. | `BLOCKED_SOURCE_UNAVAILABLE` |
| Are static backend/Android input and decode contracts obviously different? | No mismatch found by source comparison. Pixel/tensor and device/runtime parity were not measured. | Static only |
| Is a Case A-specific training error proven? | No. Actual v0.1 manifest and source images are unreadable; per-scene coverage is unknown. | Blocked |

## 2. Live repository, CI, and release baselines

| Source | Live baseline | Relevant run / result |
|---|---|---|
| Backend `main` | `05554ce1b7104c7bce05610979dfc3478b6d24a5` | Main CI #1204, run `38034915879`, success on 2026-10-10 |
| Android `main` | `0d3238e8a41b87363e9a17b04c9cc45fca4c426d` | Latest main Android CI #2316, run `39016335427`, queued at query time |
| Android PR #166 | `2e31e75d95f052a0fa589a29fd88a09bc8571b2b` | Detector bootstrap/engine files checked against Android main; no change in these relevant files. PR #166 was not modified. |
| Historical P0 report | `reports/recognition_detector_vertical_hardcase_p0_v1.md` at `05554ce1…` | Report file SHA-256 `dd87155a646769b33d5ffac4efcc72ea655dc1e9a3a1f17af3c24370f6227e79` |

Relevant immutable run links: [October 7 Detector Bootstrap #26](https://github.com/pan277942135/Yujian/actions/runs/37603789137), [October 7 production detector smoke #23](https://github.com/pan277942135/Yujian/actions/runs/37603789067), and [October 10 backend main CI #1204](https://github.com/pan277942135/Yujian/actions/runs/38034915879).

## 3. Gate A — model artifact identity

### Verified Android Release bundle

The public Release asset `det_fish_v0_1_android_bundle.zip` was downloaded and hashed. Its ZIP SHA is **`246ddacaf89ca7ecc9c64a47d3e12c5ee9088d92c763a38626778505ae8c15ee`** (4,064,690 bytes). The extracted `fish_detector_yolox_nano_v0_1.onnx` SHA is **`12b97f7c081987f33f99d255cdd2e935fb9cf93b893146f54ff98b9c4e3a8e4f`** (3,753,998 bytes). These are different object types and were not compared to one another as model hashes.

The extracted ONNX SHA and byte count match its `detector_metadata.json` and embedded `golden_cases.json`. Metadata declares:

- Model/version: `DET_FISH_v0.1`, family `YOLOX_NANO`, dataset `DET_DS_v0.1`.
- Input: name `images`, shape `[1,3,416,416]`.
- Output: `[1,3549,6]`, `decoded_xywh_objectness_class_probability`, opset 17.
- Metadata source commit: `eea53de7064fb79bb3e3f4ebaa741ff6e5ef55ad`.
- Embedded checkpoint SHA: `4b7945b85a0f890e9da0fd651c45ae6406025b2bf19ec948da24771fa5e31db8`.
- Golden Manifest ONNX SHA: `12b97f7c…`, matching the extracted Release ONNX.

The historical Case A P0 report records the same ONNX SHA `12b97f7c…`. Thus **historical report ↔ Android Release bundle identity matches**.

### Production GCS artifact history

The successful October 7 training/bootstrap job `112734396350` in run `37603789137` loaded a GCS ONNX with SHA **`2338f44c3c4913764af401dc3a3585287f36659be7560b3b7b41efc9931634b1`**, 3,753,998 bytes. It still labeled this model `DET_FISH_v0.1` / `DET_DS_v0.1`. This is the complete SHA corresponding to the previously noted `2338f44c…` record.

The same day's production smoke job `112734254145` loaded the production artifact and failed with `RuntimeError: production detector SHA does not match golden manifest`. Android Release bundle publication was skipped after Golden Cases jobs failed to upload because GCS returned 403. Therefore the last observed GCS ONNX is not the Android Release or Golden Manifest ONNX. The record identifies it as a newly trained/re-exported candidate on the same version label; available logs do not expose its checkpoint SHA or establish that it became a healthy current production revision.

On October 19, a read-only rerun of the production smoke job (`117953666299`, run `37603789067`, attempt 2) failed before object metadata retrieval: **GCS 403, “The billing account for the owning project is disabled in state absent.”** Current object generation, update time, bytes, SHA, metadata, and checkpoint are consequently unknown. A public detector health probe returned HTTP 503. No credentials were requested or changed.

The Android bootstrap script on live main pins the Release bundle URL and ZIP SHA `246ddac…`. Android engine code verifies the SHA in its bundled metadata against the loaded ONNX bytes. This proves what the published bootstrap package is pinned to, but there is no installed APK/device trace proving the ONNX loaded on a particular current device. **Gate A: FAIL for the last observed GCS-vs-Release chain; current GCS generation/live runtime additionally BLOCKED.**

Machine-readable detail: [`artifact_identity_audit.json`](artifact_identity_audit.json) and [`artifact_bundle_preflight.json`](artifact_bundle_preflight.json).

## 4. Gate B — three-direction replay and historical scores

### Replay status

No inference was run for Case A during this investigation. The exact source bytes with SHA `2dd703a46b378093604e10f0add454606393138f7501774bd3bbcca75352c9fe` and dimensions `1440×1920` were not found. The current GCS ONNX identity is not readable, and the last observed GCS hash mismatches the Android Release/Golden model. Per the identity and source gates, using the historical report's `12b97f7c…` output as a fresh production replay would mix artifact lineages. The source is **`BLOCKED_SOURCE_UNAVAILABLE`**; replay is **`BLOCKED_NOT_REPRODUCED`**.

### Historical P0 candidates (not a current replay)

The values below are retained from the historical report at `05554ce1…`. It identified the highest-confidence candidate among rows with GT IoU ≥ 0.50. Full-precision confidence and IoU were reported in its structured summary; printed objectness/class columns are rounded to six decimal places. Recomputed products below use those rounded printed fields and therefore differ slightly from the reported rounded confidence.

| View | Objectness | Fish probability | Reported confidence | Recomputed rounded-field product | GT IoU | Candidate rank | Historical gate |
|---|---:|---:|---:|---:|---:|---:|---|
| ORIGINAL | 0.001889 | 0.584072 | 0.001103288646 (table 0.001103) | 0.001103312008 | 0.532909 (table 0.533) | 5 | NO_FISH |
| CW90 | 0.003726 | 0.595116 | 0.002217599799 (table 0.002218) | 0.002217402216 | 0.765136 (table 0.765) | 6 | NO_FISH |
| CCW90 | 0.006338 | 0.670392 | 0.004248765491 (table 0.004249) | 0.004248944496 | 0.643382 (table 0.643) | 1 | NO_FISH |

All three historical boxes overlap the independently reviewed GT `[0.105, 0.220, 0.380, 1.000]`, while all historical confidences are far below the unchanged weak threshold `0.20`. Because the declared output uses `confidence = objectness × fish_probability`, the low objectness—not an equally low fish-class probability—is the direct numeric mechanism in those rows. That does **not** identify why the model assigned low objectness.

The historical report also retained Top-20 decoded candidates for each view. Their confidence ordering may include a higher-IoU box at a lower confidence than the selected candidate; the “best correct” row above is specifically the maximum-confidence row among candidates meeting IoU ≥ 0.50, not the maximum-IoU row. Historical raw tensor values and per-column logits/probability distributions were not retained, so those cannot be revalidated. See `orientation_replay.json` for all 60 historical table rows, rounded product arithmetic, original-coordinate boxes, and provenance flags.

Production replay settings remain recorded as weak `0.20`, strong `0.35`, NMS IoU `0.45`, minimum primary area `0.08`, diagnostic candidate threshold `0`, no diagnostic NMS, Top-20. **These settings were not changed or run for Case A.** The historical bounded-retry result was `NO_FISH` on all three views and selected `NONE`; it is historical only.

Case B asset SHA `a7ed3bb0410c191364f6d853b4078aebdc2f3b9baa4da34b7b3f6e8e0e93a56` was verified against the Android instrumentation asset. Its historical P0 report records CCW90 confidence `0.6355004065`, GT IoU `0.7135640`, and `READY` on the old `12b97f7c…` model; this investigation did not rerun the positive control. Gate B is **BLOCKED** for this investigation.

## 5. Gate C — backend/Android preprocessing and decode parity

Static inspection found no obvious mismatch in the declared 416 input contract:

| Contract item | Backend | Android | Static finding |
|---|---|---|---|
| EXIF orientation | Pillow `exif_transpose`, then RGB | `ExifInterface.rotationDegrees`, rotate decoded bitmap | Same intent; decoders differ |
| Source downsample | power-of-two sample to ≤2048 | `BitmapFactory.inSampleSize` power-of-two to ≤2048 | Same policy; exact raster untested |
| Model resize | scale to fit, integer-truncated dimensions, bilinear | same dimensions; custom Pillow-compatible separable bilinear | Intended parity; pixels not compared |
| Letterbox | top-left, fill 114 | top-left, fill 114 | Match by code |
| Tensor | RGB→BGR, float32 NCHW, raw 0–255 | BGR planes, float32 NCHW, raw 0–255 | Match by code |
| Output | decoded xywh/objectness/class probability; confidence product | same confidence product and normalized xywh inverse mapping | Match by code/metadata contract |
| Gate and NMS | weak .20, strong .35, IoU .45, area .08 | same constants | Match by code |

Android source decode is `BitmapFactory`; backend uses Pillow. Android's resize implementation deliberately mirrors Pillow's two-pass bilinear filter with fixed-point coefficients; backend directly uses Pillow bilinear. The intended resize policy matches, but there is no same-byte, same-pixel comparison. Android's normalized JPEG92 file is not the actual `detect()` tensor source; the detector receives the oriented in-memory bitmap. Case A's historical EXIF is 1 and its dimensions are below the 2048 sampling bound, but those facts do not establish byte-identical RGB decoding.

Backend and Android both decode output field 4 × field 5 for confidence and use the same scale/source-dimension inverse map. Metadata labels output as decoded probabilities; source inspection shows no additional sigmoid in either consumer. Orientation coordinate transforms, bounded retry trigger, NMS, and threshold constants are aligned by code.

**Parity conclusion:** static source-contract comparison found no confirmed `PREPROCESSING_PARITY_BUG` or `OUTPUT_DECODE_BUG`; end-to-end parity is **not established**. No 416×416 input tensor SHA/statistics, Android output tensor, Case A runtime trace, or Case B runtime replay was captured. Do not call this a PASS. Machine-readable detail: [`preprocessing_parity.json`](preprocessing_parity.json).

## 6. Gate D — data and label audit

### Available dataset summary

The published Android Release metadata and October 7 reuse-frozen-dataset gate report `DET_DS_v0.1` had 594 selected images: 440 positives, 154 negatives, and 222 incomplete candidates. Split summary in the Release metadata is train 477 images / 481 boxes, validation 50 / 50, test 67 / 54. The October 7 prepare-dataset job `112734253323` reported reuse of the frozen set and `train_images=477 train_boxes=481 val_boxes=50 test_images=67 no_fish_negatives=154`.

Those are release metadata/CI summaries, not a fresh manifest/image review. On October 19, the GCS billing error blocked retrieval of `detector-datasets/DET_DS_v0.1/bootstrap_manifest.json`, archive, and current report. Case A and Case B have not been established as v0.1 training rows; both remain held-out incidents in source documentation. The actual manifest was not available to independently prove exclusion.

### Source-code findings (risk, not Case A-specific proof)

- `trainer/build_detector_dataset.py::_collect` takes fish boxes from Google Vision object names and filters them at the contract's strong score `0.35`. A presence status of `single_fish`/`multi_fish` is included only if a qualifying box remains. Thus a real fish with only lower-score evidence may be omitted from positive training data.
- If the effective status is `no_fish`, the builder emits an empty-box negative row without checking whether qualifying fish boxes were collected in that same saved evidence. This is a concrete inconsistent-evidence path that can create a false negative if `no_fish` is wrong or manually overridden while fish detections remain. No affected manifest entry was available to establish that it happened in v0.1.
- `app/presence.py` stores a valid `human_override` as `row.status`; the builder uses that effective status. The override does not provide a manually reviewed bounding box; box geometry remains Google Vision evidence. For positive overrides, no qualifying machine box means no positive row.
- Split assignment hashes `batch_id:image_id` independently per image. The builder does not group related items by source/session; actual cross-split batch leakage is unknown without the manifest.
- Emitted manifest rows include asset ID, batch/image IDs, GCS URI, split, status, negative flag, boxes, and dimensions, but no source SHA, reviewer, manual bbox provenance, license/permission record, source group, or hard-scene tags. Scene-specific coverage/recall cannot be reproduced from the manifest schema alone.

No audited source image/label pair, sample ID with original label and review decision, or per-scene recall count was available. Orientation and scene class remain `UNKNOWN`; none is inferred from image dimensions. The confirmed facts support a **high-priority label-noise/coverage audit hypothesis**, not a confirmed Case A training-label root cause. `DET_DS_v0.1` remains untouched and frozen; no Case A or Case B image was added.

Machine-readable detail: [`dataset_audit.json`](dataset_audit.json). To re-audit an authorized manifest without opening/uploading photos: `python3 scripts/audit_detector_dataset_manifest.py --manifest <authorized-manifest.json> --report <bootstrap-report.json> --output <audit.json>`.

## 7. Root-cause hypotheses and confidence

| Classification | Status | Evidence / falsifier |
|---|---|---|
| `ARTIFACT_MISMATCH` | **Confirmed** for last observed GCS vs Release/Golden chain | GCS hash `2338f44c…` logged by October 7 training job; production smoke failed GCS-vs-Golden hash check. Android Release and historical Case A report both use `12b97f7c…`. Current GCS identity remains blocked. |
| `PREPROCESSING_PARITY_BUG` | **Not confirmed; still open** | Static input contract aligns. Falsifier needed: same exact source -> Android vs backend 416 tensor hashes/statistics and output traces. |
| `OUTPUT_DECODE_BUG` | **Not confirmed** | Confidence and bbox equations match by source inspection. Need raw same-model output from both runtimes and test coverage for decoded logits/probabilities, mapping and NMS. |
| `TRAINING_LABEL_NOISE` | **High-priority hypothesis, not Case A-proven** | Builder can emit a negative from `no_fish` despite collected fish boxes; overrides supply status without manual boxes; actual manifest/sample review unavailable. |
| `DOMAIN_RECALL_GAP` | **Unproven for the last-observed GCS model** | Historical low objectness pertains to `12b97f7c…`, not last-observed `2338f44c…`; no same-artifact replay or reliable hard-scene GT set. |
| `MULTI_FACTOR` | **Not assigned** | More than one cause is plausible, but insufficient aligned evidence to identify contributing causes. |
| `BLOCKED_INSUFFICIENT_EVIDENCE` | **Confirmed final Case A root-cause disposition** | Exact source unavailable, current GCS unreadable, artifact chain mismatch, no runtime tensor parity, and training manifest unavailable. |

Confidence: **high** that the last-observed production GCS lineage diverged from Android Release/Golden. **High** that the historical reported confidence product was dominated by low objectness. **Low/undetermined** on why objectness was low in the model; there is not enough evidence to choose between artifact-specific model behavior, pixel/input differences, decode/runtime differences, or training data/label coverage.

## 8. Reproducible diagnostics and run record

Added read-only tooling:

- `scripts/audit_case_a_artifacts.py`: hashes an Android model ZIP and separately validates its extracted ONNX, metadata, contract, Golden Manifest, and optional Case A source hash/dimensions/EXIF. It never fetches GCS or stores source bytes in the repository.
- `scripts/diagnose_detector_orientation.py`: gated diagnostic replay. Requires explicit ONNX/metadata/images; validates ONNX hash and bytes; optional `--expected-image-sha256` guard; records source ID, input tensor SHA/statistics, raw output summaries, Top-K un-NMSed candidates, GT IoU, production gate outputs, and bounded-retry result. Default Top-K is 20. It has not been run on Case A in this work.
- `scripts/audit_detector_dataset_manifest.py`: reads an authorized manifest/report and computes split, labels, duplicates, cross-split batches, and provenance coverage without opening image files.
- `tests/test_case_a_investigation_tools.py`: deterministic ZIP identity, strict source hash guard, and manifest audit tests.
- [`validation_evidence.json`](validation_evidence.json): reproducible local command outcomes and SHA-256 values for the delivered scripts, tests, and evidence files.

Run records: backend main CI #1204 (`38034915879`, success); Android main run #2316 (`39016335427`, queued when checked); detector training/bootstrap run #26 (`37603789137`, success); production smoke #23 (`37603789067`, failed on artifact SHA mismatch in the original attempt and GCS billing 403 in the read-only rerun). The Case A diagnostic replay was deliberately **not run**. No CI run is represented as a Case A inference pass.

Local validation commands for this branch:

```bash
python3 -m py_compile scripts/diagnose_detector_orientation.py scripts/audit_case_a_artifacts.py scripts/audit_detector_dataset_manifest.py tests/test_case_a_investigation_tools.py
python3 -m unittest discover -s tests -p 'test_case_a_investigation_tools.py' -v
```

Both commands exited 0 locally; all 3 investigation-tool unit tests passed. The Android Release bundle preflight also exited 0 and independently re-confirmed ZIP SHA `246ddac…`, extracted ONNX SHA `12b97f7c…`, metadata and Golden Manifest identity. All five machine-readable audit JSON files parse successfully, and `git diff --check` exited 0. This local validation does not substitute for PR CI or runtime inference.

## 9. Gate status

| Gate | Status | Reason |
|---|---|---|
| A — model identity | **FAIL / BLOCKED** | Last observed GCS hash differs from Release/Golden `12b97f7c…`; current GCS generation/hash unreadable due billing 403; no installed-device load trace. |
| B — detection replay | **BLOCKED** | Case A SHA-matched source not available; current production artifact identity unresolved. Case B source hash verified, but this investigation did not rerun it. Historical rows are explicitly not replay evidence. |
| C — inference consistency | **STATIC PARTIAL / RUNTIME BLOCKED** | Static preprocessor/decode source inspection found no mismatch; tensor and Android instrumentation comparison not performed. |
| D — data audit | **STATIC PARTIAL / MANIFEST BLOCKED** | Training code and release/run summaries inspected; actual manifest, archive, image labels and scene coverage unavailable. |
| E — engineering closeout | **LOCAL VALIDATION PASS; PR/CI PENDING** | Local scripts compile, 3 focused tests pass, artifact preflight and JSON validation pass. Independent Draft PR and remote CI remain to be recorded after push. |

## 10. Next evidence required before model work

1. Restore read access to the owning GCS project/model prefix (the current service error is billing-account state `absent`), or provide an already authorized read-only object export path. Capture exact object generation/time, ONNX SHA/bytes, metadata/contract and checkpoint SHA. Do not rewrite the objects as part of this investigation.
2. Locate the authorized exact Case A original by SHA `2dd703a…` and confirm 1440×1920 plus EXIF orientation 1. Keep the identifiable photo outside GitHub and do not substitute a re-encoded copy.
3. After artifact identity resolves to one explicit ONNX, rerun ORIGINAL/CW90/CCW90 from the same bytes with zero-threshold/no-NMS Top-20 diagnostics and unchanged production-gate replay. Then run the Case B positive control using its verified fixture and record both tensor and output identities.
4. Retrieve the actual v0.1 manifest/archive under authorized access. Review candidate records where status/evidence disagree, verify manual label provenance and image-level held-out exclusions, then compute scene coverage only from human-reviewed tags and reliable GT.
5. Only after these comparisons identify whether input, decoder/artifact, or labels explain the failure should the team propose the smallest corresponding correction. This investigation performed no training or model release.

## 11. Scope and privacy statement

No ONNX weights, model structure, export settings, thresholds, NMS, quality gate, orientation retry logic, or frozen dataset were modified. No model was trained or published. The identifiable Case A original was not copied into the repository, tests, PR, or public CI artifacts; only its expected SHA, dimensions, and historical report reference are recorded.
