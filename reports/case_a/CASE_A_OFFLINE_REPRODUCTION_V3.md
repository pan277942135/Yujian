# Case A Offline Evidence Recovery and Three-Direction Replay V3

**Task:** `CASE_A_OFFLINE_REPRODUCTION_V3`
**Repository:** `pan277942135/Yujian`
**Continuation PR:** #164, Draft / Open / Unmerged
**Mode:** Offline evidence recovery; exact artifact verification; no model or production inference changes
**V3 start HEAD:** `c526d2e6a2618d6b81d6d301318e5eb0e1581ef1`

## 1. Conclusion

The historical Case A failure was **not reproduced**. The files attached to this Work session are not the historical source bytes: `IMG_1415.jpg` is 1152×1536 with SHA-256 `0fc341627e4345131a15fbf3514454f053dcb349db17bf1171c8f2f8756226cf`; the expected historical source is 1440×1920 with SHA-256 `2dd703a46b378093604e10f0add454606393138f7501774bd3bbcca75352c9fe`. The second attachment is a 911×1536 screenshot and was not used as an inference source. A byte-hash scan of 1,993 local image files found no exact Case A source.

The archived GitHub Android Release bundle was recovered and hashed locally. Its ZIP SHA-256 is `246ddacaf89ca7ecc9c64a47d3e12c5ee9088d92c763a38626778505ae8c15ee`; the extracted ONNX SHA-256 is `12b97f7c081987f33f99d255cdd2e935fb9cf93b893146f54ff98b9c4e3a8e4f`, matching its metadata, Golden Manifest, and the historical Case A report. This verifies the **archived Android Release / historical diagnostic model**, not a currently installed device or the unavailable current GCS object.

The ONNX graph check supports the declared decoded-probability output contract for this archived model: its detection head applies Sigmoid to objectness and class prediction tensors at all three scales, and the graph contains no Softmax. The backend decoder multiplies the two output columns once. A fresh Case B CPU replay reproduced the historical CCW90 `READY` result and confidence exactly. The nonmatching user-provided candidate was also run twice; its tensors and raw model outputs were bitwise identical between runs. Neither run can replace the missing historical Case A input.

**Case-specific root cause remains `BLOCKED_INSUFFICIENT_EVIDENCE`.** The historical data still show that low objectness is the direct numeric reason confidence fell below 0.20, but these new experiments cannot establish why the original image received low objectness. Historical artifact mismatch remains a confirmed lineage issue, not a proven cause of Case A’s score. No GCP billing or resource was changed or accessed.

## 2. Live baseline and scope

At V3 start, GitHub reported backend `main@05554ce1b7104c7bce05610979dfc3478b6d24a5`, Android `main@0d3238e8a41b87363e9a17b04c9cc45fca4c426d`, and PR #164 at `c526d2e6a2618d6b81d6d301318e5eb0e1581ef1` over that backend `main`. The PR was Open / Draft / Unmerged. Baseline CI run `39223185096` (run #1208) was successful on that head; its `validate` job completed production recognition pipeline tests.

V3 adds offline evidence files and diagnostic-only inspection/reporting. It does not alter ONNX weights, training, thresholds, NMS, quality gates, retry behavior, frozen datasets, or production model loading. The identifiable source photo and screenshot remain outside the repository.

## 3. Artifact identity recovered offline

| Item | Verified result | Scope / limitation |
|---|---|---|
| GitHub Release | `detector-model-v0.1`, Release ID `379655230`; asset ID `537791631`, published 2026-08-31 | Public archived Release |
| Android bundle ZIP | 4,064,690 bytes; downloaded SHA-256 `246ddacaf89ca7ecc9c64a47d3e12c5ee9088d92c763a38626778505ae8c15ee` | Matches GitHub Release asset digest |
| Bundle ONNX | 3,753,998 bytes; SHA-256 `12b97f7c081987f33f99d255cdd2e935fb9cf93b893146f54ff98b9c4e3a8e4f` | Matches metadata and historical Case A ONNX |
| Detector metadata | SHA-256 `53994a745c192e3c2afc31ad256800bb87f841fafd07977f0fd38fa7e1f8327e`; declares the same ONNX SHA and byte count | Metadata is inside the verified bundle |
| Golden Manifest | SHA-256 `883ab6e54cd7aa2c66907765d401ff22385384c1a8ba242167e8c10ec2ae0d19`; ONNX SHA matches | Five bundled golden cases |
| Training checkpoint | Metadata declares `4b7945b85a0f890e9da0fd651c45ae6406025b2bf19ec948da24771fa5e31db8` | Checkpoint bytes are unavailable; the declared digest was not re-hashed |
| Last observed GCS ONNX | Historical V1 evidence records `2338f44c3c4913764af401dc3a3585287f36659be7560b3b7b41efc9931634b1` | Differs from archived Release; current GCS object is unavailable and was not accessed |
| Installed Android device | No device-load trace available | `adb` and `emulator` are not installed in this Work runtime |

The Release bundle and its ONNX are different objects; their hashes are recorded separately. The archived bundle proves identity with the historical diagnostic model, but does not prove which model is currently loaded by any Android device or current production service.

Machine-readable detail: [`offline_artifact_preflight_v3.json`](offline_artifact_preflight_v3.json), [`onnx_output_contract_v3.json`](onnx_output_contract_v3.json), [`offline_recovery_inventory_v3.json`](offline_recovery_inventory_v3.json), and [`validation_evidence_v3.json`](validation_evidence_v3.json).

## 4. ONNX output semantics

ONNX checker passed on the extracted `12b97f7c…` model. The graph has opset 17, one FLOAT input `images` with shape `[1,3,416,416]`, and one FLOAT output `output` with shape `[1,3549,6]`. It contains 562 nodes: 110 Sigmoid nodes overall, including three detection-head Sigmoids fed by `obj_preds` and three fed by `cls_preds`; it contains no Softmax or LogSoftmax and one `Exp` decode node. Other Sigmoid nodes implement the model’s internal activations, so the total count of 110 should not be read as 110 output activations.

The verified head graph therefore applies probability activation before assembling the objectness and class output columns. The backend consumer does not apply another Sigmoid; it computes `confidence = objectness × fish_probability` once. This is graph evidence for the archived Release ONNX only. The graph for the last observed GCS SHA and the current unavailable GCS object was not inspected.

## 5. Source identity and three-direction experiments

### Historical Case A

The historical report’s candidate rows remain `HISTORICAL`, not new replay evidence:

| Direction | Historical objectness | Historical fish probability | Historical confidence | Historical GT IoU | Historical gate |
|---|---:|---:|---:|---:|---|
| ORIGINAL | 0.001889 | 0.584072 | 0.001103289 | 0.532909 | `NO_FISH` |
| CW90 | 0.003726 | 0.595116 | 0.002217600 | 0.765136 | `NO_FISH` |
| CCW90 | 0.006338 | 0.670392 | 0.004248765 | 0.643382 | `NO_FISH` |

The old confidence values are consistent with the declared product to the precision retained in the historical candidate rows. All remain far below the unchanged weak threshold 0.20. The original input and its output tensors were not recovered, so the historical rows cannot be recomputed from the original bytes in this turn.

### User-provided candidate (not historical Case A)

`IMG_1415.jpg` (292,569 bytes, JPEG, 1152×1536) was labeled `USER_PROVIDED_CASE_A_CANDIDATE` and run twice with the archived Release ONNX. Its EXIF orientation is 1, so input normalization required no orientation transform. No historical Ground Truth box was applied; each candidate's `gt_iou` is `null`, not zero, because no candidate Ground Truth was supplied.

| Direction | Objectness | Fish probability | Confidence | Gate | BBox after mapping to candidate source |
|---|---:|---:|---:|---|---|
| ORIGINAL | 0.046929 | 0.767433 | 0.036015 | `NO_FISH` | `[0.203844,0.209540,0.406149,0.827160]` |
| CW90 | 0.540539 | 0.795761 | 0.430140 | `READY` | `[0.186673,0.266021,0.414967,0.791586]` |
| CCW90 | 0.453421 | 0.793069 | 0.359594 | `FISH_TOO_SMALL` | `[0.612116,0.445006,0.993586,0.566836]` |

Bounded replay selected CW90. This is a direct result for the uploaded candidate only. It suggests orientation sensitivity for that candidate; it does not establish an orientation or training-data cause for historical Case A.

### Case B positive control

The actual fixture is a 151,870-byte JPEG, 1152×1536, SHA-256 `a7ed3bb0410c191364f6d853b4078aebdc2f3b9baa4da34b7b3f6e8e0e93a56b`, matching the Android test constant and local fixture bytes. Its EXIF orientation is 1. The V3 task text omitted the final `b`; an initial strict check caught that mismatch and the run was repeated using the 64-character SHA from the fixture/test. Case B has no V3 Ground Truth file either, so its per-candidate `gt_iou` fields are `null`; the historical IoU remains historical evidence.

| Direction | Objectness | Fish probability | Confidence | Gate | BBox after mapping to source |
|---|---:|---:|---:|---|---|
| ORIGINAL | 0.012112 | 0.675594 | 0.008183 | `NO_FISH` | `[0.558816,0.363477,0.874994,0.818982]` |
| CW90 | 0.250154 | 0.787933 | 0.197105 | `NO_FISH` | `[0.522416,0.329314,0.972723,0.771111]` |
| CCW90 | 0.823237 | 0.771953 | 0.635500 | `READY` | `[0.342930,0.363650,0.605436,0.924975]` |

The production replay selected CCW90 with confidence `0.6355004065`, area ratio `0.147351`, and two post-threshold/NMS detections. Its primary bbox, gate, and confidence match the historical Case B result. Historical GT IoU was `0.713564`; V3 did not recalculate it because no separate Case B GT file was available to this run.

Both the candidate and Case B were run twice from diagnostic commit `f81cb21abe520dae11ad9f4a677567b1a2103b26`. All three orientation input tensor SHA-256 values, raw ONNX output payload SHA-256 values, Top-20 candidate records, and production gate outputs matched exactly between each pair of runs (excluding execution timestamps and optional raw-output artifact pointers). This confirms repeatability of this backend CPU environment and script for those exact input files.

Full per-orientation input statistics and tensor hashes, raw output hashes/ranges, Top-20 candidates, and production gates are in [`user_provided_candidate_replay_v3.json`](user_provided_candidate_replay_v3.json) and [`case_b_positive_control_replay_v3.json`](case_b_positive_control_replay_v3.json). The exact `[1,3549,6]` float32 ONNX output arrays are also preserved as lossless `.npy` evidence; each JSON record contains both the file SHA and raw array SHA.

| Source | ORIGINAL output | CW90 output | CCW90 output |
|---|---|---|---|
| User candidate (`0fc341627e43…`) | [`.npy`](raw_outputs_v3/0fc341627e43.original.npy) | [`.npy`](raw_outputs_v3/0fc341627e43.cw90.npy) | [`.npy`](raw_outputs_v3/0fc341627e43.ccw90.npy) |
| Case B (`a7ed3bb0410c…`) | [`.npy`](raw_outputs_v3/a7ed3bb0410c.original.npy) | [`.npy`](raw_outputs_v3/a7ed3bb0410c.cw90.npy) | [`.npy`](raw_outputs_v3/a7ed3bb0410c.ccw90.npy) |

## 6. Preprocessing parity and execution boundary

The backend replay used the existing production preprocessor: Pillow RGB decode, rotation before model preprocessing, 416×416 input, top-left letterbox with fill 114, RGB-to-BGR, float32 NCHW, raw 0–255 values, and the unchanged confidence/NMS/gate contract. The source files had EXIF orientation 1. Both replay JSON files retain, for each orientation, the input tensor SHA-256, shape/dtype/range, per-channel statistics, scale/draw dimensions, ONNX output hash, output column ranges, and all Top-20 candidates.

The Android and backend code-level comparison from V1 remains static evidence. V3 did not execute Android bitmap decode or the ONNX on an Android device/emulator. Therefore Android-vs-backend byte-level tensor parity and runtime parity remain `NOT_RUN`; the backend tensor hashes do not prove Android parity. Case A’s exact tensor cannot be generated until the historical source bytes are available.

## 7. Training data and labels

Offline search found no tracked `DET_DS_v0.1` bootstrap manifest, dataset archive, or `best_ckpt.pth` bytes in the available repository checkout. The 594-image / 440-positive / 154-negative summary and split counts are still metadata summaries, not a per-image audit. There is no evidence in this run identifying a real mislabeled sample, false negative, manual bbox, or Case A training membership. The existing builder-code risks in V1 remain hypotheses until an actual manifest and evidence rows can be inspected.

Case A and Case B were not added to training. The attached images were used only as local read-only diagnostic inputs; no source photo, input pixel tensor, model, dataset, or checkpoint was added to the repository. Exact model-output arrays are included as the six `.npy` replay artifacts listed above.

## 8. Root-cause disposition and gates

| Classification | V3 disposition |
|---|---|
| `ARTIFACT_MISMATCH` | Confirmed historically: last observed GCS ONNX `2338f44c…` differed from the archived Release / Golden ONNX `12b97f7c…`. Current GCS identity is unknown. This is not proven to be the cause of the historical Case A score. |
| `ONNX_OUTPUT_SEMANTICS_MISMATCH` | Not supported for archived Release SHA `12b97f7c…`: graph head Sigmoids are present for objectness and class, output fields are already activated, and backend code multiplies them once. Current GCS graph remains unknown. |
| `PREPROCESSING_MISMATCH` | Not established. Backend tensors are now recorded for the candidate and Case B; Android byte-level and runtime comparison was not run. |
| `TRAINING_LABEL_NOISE` | Not established; no actual manifest/source row is available. |
| `DIFFICULT_DOMAIN_COVERAGE_GAP` | Not established for historical Case A. The candidate and Case B are not substitutes for the exact Case A input. |
| `MODEL_DETECTION_FAILURE` | Not determined for historical Case A. Case B replay succeeded, and the candidate produced an orientation-dependent result. |
| `BLOCKED_INSUFFICIENT_EVIDENCE` | **Final Case A root-cause disposition.** Exact source SHA and current production model remain unavailable. |

| Gate | Status | Evidence |
|---|---|---|
| A — archived artifact identity | **PASS for archived Android Release; BLOCKED for current production** | ZIP, ONNX, metadata and Golden Manifest hashes match; current GCS and installed-device identities unknown. |
| B — replay | **PASS for Case B; candidate replay repeatable; BLOCKED for historical Case A** | Case B reproduces; neither attachment matches the Case A SHA. |
| C — output/preprocessing | **PARTIAL** | Archived ONNX graph and backend tensors audited; Android-vs-backend runtime parity not run. |
| D — dataset audit | **BLOCKED** | Manifest, archive and checkpoint bytes absent. |
| E — engineering | **V3 code/report validation pending PR CI** | New diagnostic-only code, tests and reports are in this PR branch; final CI run will be recorded in the PR checks and follow-up validation evidence. |

## 9. Reproduction commands

With the archived bundle extracted to `$RELEASE_DIR`, install the repository detector dependencies in an isolated Python environment, then run:

```bash
python scripts/inspect_detector_onnx_contract.py \
  --onnx "$RELEASE_DIR/fish_detector_yolox_nano_v0_1.onnx" \
  --metadata "$RELEASE_DIR/detector_metadata.json" \
  --output reports/case_a/onnx_output_contract_v3.json

python scripts/diagnose_detector_orientation.py \
  --onnx "$RELEASE_DIR/fish_detector_yolox_nano_v0_1.onnx" \
  --metadata "$RELEASE_DIR/detector_metadata.json" \
  --expected-image-sha256 a7ed3bb0410c191364f6d853b4078aebdc2f3b9baa4da34b7b3f6e8e0e93a56b \
  --raw-output-dir reports/case_a/raw_outputs_v3 \
  --json-out reports/case_a/case_b_positive_control_replay_v3.json \
  app/src/androidTest/assets/detector_case_b_night_flash.jpg
```

For historical Case A, use `--expected-image-sha256 2dd703a46b378093604e10f0add454606393138f7501774bd3bbcca75352c9fe` with the unmodified 1440×1920 original. The script’s production gate remains weak 0.20, strong 0.35, NMS IoU 0.45, and minimum primary area 0.08; the diagnostic Top-20 probe uses threshold zero and no NMS.

## 10. Minimum evidence needed to resolve Case A

Please attach the unmodified original 1440×1920 file whose SHA-256 is `2dd703a46b378093604e10f0add454606393138f7501774bd3bbcca75352c9fe`. The two files attached to this run do not match it; do not send a screenshot or resized/re-encoded copy. The archived `12b97f7c…` ONNX is sufficient to replay the historical report’s model identity. A new model-file input is not needed unless the goal is specifically to establish the identity of current production, which this offline V3 intentionally does not access.
