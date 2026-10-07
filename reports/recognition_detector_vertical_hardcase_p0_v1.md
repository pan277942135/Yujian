# Recognition Detector Vertical / Hard-Case Miss P0 — V1.1

Frozen Android acceptance base: `integration/android-product-acceptance-v1` / `06837c6d402c920fbc29e0a4d34595eab5da1c58`.
Android repair target is based on the existing WIP commit `f0fe3dffefa556eed2e79ecb4c48d9e14afff58b`; the fix is kept separate from `main`, PR #97, and the frozen acceptance branch. Model Factory work stays on the detector repair feature line.

## Production detector and diagnostic method

- Production model: `DET_FISH_v0.1`, `YOLOX_NANO`, input `416x416`.
- ONNX SHA-256: `12b97f7c081987f33f99d255cdd2e935fb9cf93b893146f54ff98b9c4e3a8e4f`.
- Production thresholds unchanged: weak `0.20`, strong `0.35`, NMS IoU `0.45`, minimum primary area `0.08`.
- Raw diagnostic probe: decode with `min_confidence=0`, no NMS, rank raw rows by `objectness × fish_probability`; production gate replay uses the unchanged production threshold and NMS.
- Case A source is read as the supplied JPEG bytes, decoded directly to RGB at `1440x1920`, EXIF orientation `1`; no crop, resize, normalization, or re-encoding is applied before the raw probe. Source SHA-256: `2dd703a46b378093604e10f0add454606393138f7501774bd3bbcca75352c9fe`.
- Case A ground-truth bbox is manually reviewed from the original source, independent of detector output: `[0.105, 0.220, 0.380, 1.000]`. The lower edge is at the source frame boundary because the fish continues out of frame. `IoU_GT` is an approximate human annotation metric and should not be used as a training label without a separate review pass.
- IoU match criterion for the “correct candidate” analysis is `IoU_GT >= 0.50`. The borderline band is diagnostic-only `[0.18, 0.20)`.

## Incident classification

| Fixture | ORIGINAL best correct confidence | CW90 | CCW90 | Production replay | Classification |
|---|---:|---:|---:|---|---|
| Case A original, 1440x1920 | 0.001103 | 0.002218 | 0.004249 | all views NO_FISH; selected NONE | `DOMAIN_RECALL_GAP` |
| Case B night/flash, 1152x1536 | 0.004820 | 0.018577 | 0.635500 | CCW90 selected; READY | `ORIENTATION_RECALL_GAP` |

Case A's earlier `PROVISIONAL_DOMAIN_RECALL_GAP` based on a `327x436` result-screen viewport proxy is revoked. The original source confirms that all correct candidates remain well below `0.20`; this is not an orientation-recall or threshold-borderline case. The best correct confidence across all Case A views is `0.004249`.

Case B confirms orientation recall recovery: the correct fish candidate is under `0.20` in ORIGINAL and CW90, then reaches `0.635500` in CCW90. No evidence indicates a decode/NMS defect for either incident: neither original view has a correct candidate at or above `0.20` while production emits NO_FISH.

## Case A original raw ONNX Top-20

The following table was generated from the original source raster and the ONNX SHA above. Bboxes are mapped into original-source normalized coordinates for all three views. The diagnostic decoder used threshold `0` and no NMS.

Highest-confidence correct candidate per view (candidate must have `IoU_GT >= 0.50`):

| orientation | rank | objectness | fish_probability | confidence | bbox_original_xyxy | area_ratio | IoU_GT |
|---|---:|---:|---:|---:|---|---:|---:|
| ORIGINAL | 5 | 0.001889 | 0.584072 | 0.001103 | `[0.064756,0.171667,0.550681,1.000000]` | 0.402507 | 0.533 |
| CW90 | 6 | 0.003726 | 0.595116 | 0.002218 | `[0.091187,0.166099,0.350108,0.965291]` | 0.206927 | 0.765 |
| CCW90 | 1 | 0.006338 | 0.670392 | 0.004249 | `[0.145772,0.209198,0.464641,1.000000]` | 0.252162 | 0.643 |

model_version=DET_FISH_v0.1
onnx_sha256=12b97f7c081987f33f99d255cdd2e935fb9cf93b893146f54ff98b9c4e3a8e4f
weak_confidence=0.2 nms_iou=0.45

## 92f98cf35ad9659ab79e7b930062ec17.jpg (1440x1920)
| orientation | rank | objectness | fish_probability | confidence | bbox_original_xyxy | area_ratio | IoU_GT |
|---|---:|---:|---:|---:|---|---:|---:|
| ORIGINAL | 1 | 0.002482 | 0.599483 | 0.001488 | [0.665196,0.945632,1.000000,1.000000] | 0.018203 | 0.000 |
| ORIGINAL | 2 | 0.002385 | 0.614429 | 0.001465 | [0.607136,0.944308,1.000000,1.000000] | 0.021879 | 0.000 |
| ORIGINAL | 3 | 0.001445 | 0.772643 | 0.001116 | [0.521673,0.224455,0.779777,0.436705] | 0.054783 | 0.000 |
| ORIGINAL | 4 | 0.001509 | 0.734013 | 0.001108 | [0.126708,0.141893,0.278862,0.621092] | 0.072912 | 0.270 |
| ORIGINAL | 5 | 0.001889 | 0.584072 | 0.001103 | [0.064756,0.171667,0.550681,1.000000] | 0.402507 | 0.533 |
| ORIGINAL | 6 | 0.002565 | 0.422007 | 0.001082 | [0.002721,0.182064,0.417147,1.000000] | 0.338974 | 0.633 |
| ORIGINAL | 7 | 0.001979 | 0.535857 | 0.001061 | [0.031973,0.172391,0.664288,1.000000] | 0.523309 | 0.410 |
| ORIGINAL | 8 | 0.001481 | 0.647179 | 0.000958 | [0.495766,0.218587,0.875099,0.778683] | 0.212463 | 0.000 |
| ORIGINAL | 9 | 0.001316 | 0.690691 | 0.000909 | [0.520554,0.227094,0.780040,0.423898] | 0.051068 | 0.000 |
| ORIGINAL | 10 | 0.001617 | 0.549852 | 0.000889 | [0.630062,0.949634,1.000000,1.000000] | 0.018632 | 0.000 |
| ORIGINAL | 11 | 0.001527 | 0.578320 | 0.000883 | [0.683421,0.947681,1.000000,1.000000] | 0.016563 | 0.000 |
| ORIGINAL | 12 | 0.001236 | 0.687533 | 0.000850 | [0.524782,0.229306,0.768168,0.419643] | 0.046325 | 0.000 |
| ORIGINAL | 13 | 0.001030 | 0.790047 | 0.000814 | [0.528876,0.227966,0.761717,0.430756] | 0.047218 | 0.000 |
| ORIGINAL | 14 | 0.001077 | 0.731478 | 0.000788 | [0.114593,0.155619,0.308016,0.650640] | 0.095748 | 0.367 |
| ORIGINAL | 15 | 0.001353 | 0.529264 | 0.000716 | [0.387110,0.217884,0.918007,0.866598] | 0.344400 | 0.000 |
| ORIGINAL | 16 | 0.001595 | 0.441303 | 0.000704 | [0.040848,0.170229,0.399230,1.000000] | 0.297375 | 0.721 |
| ORIGINAL | 17 | 0.001280 | 0.537276 | 0.000688 | [0.633881,0.950827,1.000000,1.000000] | 0.018003 | 0.000 |
| ORIGINAL | 18 | 0.000964 | 0.675291 | 0.000651 | [0.527310,0.222027,0.756708,0.345411] | 0.028304 | 0.000 |
| ORIGINAL | 19 | 0.000873 | 0.678253 | 0.000592 | [0.525521,0.221440,0.745531,0.314447] | 0.020462 | 0.000 |
| ORIGINAL | 20 | 0.000742 | 0.747242 | 0.000554 | [0.003720,0.642471,0.195605,1.000000] | 0.068604 | 0.129 |
| CW90 | 1 | 0.016292 | 0.537799 | 0.008762 | [0.679470,0.170959,1.000000,0.678125] | 0.162562 | 0.000 |
| CW90 | 2 | 0.007222 | 0.682233 | 0.004927 | [0.578058,0.190631,0.969717,0.779932] | 0.230805 | 0.000 |
| CW90 | 3 | 0.006821 | 0.668591 | 0.004560 | [0.592056,0.215677,0.957502,0.887898] | 0.245661 | 0.000 |
| CW90 | 4 | 0.007167 | 0.634713 | 0.004549 | [0.695006,0.176619,0.999343,0.713614] | 0.163428 | 0.000 |
| CW90 | 5 | 0.005077 | 0.607636 | 0.003085 | [0.566303,0.197428,0.985137,0.743438] | 0.228687 | 0.000 |
| CW90 | 6 | 0.003726 | 0.595116 | 0.002218 | [0.091187,0.166099,0.350108,0.965291] | 0.206927 | 0.765 |
| CW90 | 7 | 0.003138 | 0.564550 | 0.001772 | [0.694793,0.178678,0.996712,0.664538] | 0.146690 | 0.000 |
| CW90 | 8 | 0.002304 | 0.720904 | 0.001661 | [0.102486,0.243741,0.346566,0.844130] | 0.146543 | 0.671 |
| CW90 | 9 | 0.002453 | 0.675186 | 0.001656 | [0.560980,0.220945,0.969098,0.943389] | 0.294842 | 0.000 |
| CW90 | 10 | 0.002326 | 0.679627 | 0.001581 | [0.553040,0.201735,0.978111,0.808873] | 0.258077 | 0.000 |
| CW90 | 11 | 0.002175 | 0.672746 | 0.001463 | [0.139451,0.198337,0.280203,0.541317] | 0.048275 | 0.208 |
| CW90 | 12 | 0.002122 | 0.663508 | 0.001408 | [0.093878,0.188438,0.360343,0.842566] | 0.174302 | 0.692 |
| CW90 | 13 | 0.002324 | 0.602528 | 0.001400 | [0.101309,0.148121,0.362241,0.982957] | 0.217836 | 0.831 |
| CW90 | 14 | 0.001963 | 0.609749 | 0.001197 | [0.082888,0.149306,0.419557,0.995050] | 0.284736 | 0.745 |
| CW90 | 15 | 0.001679 | 0.666389 | 0.001119 | [0.146602,0.187825,0.272741,0.508783] | 0.040486 | 0.167 |
| CW90 | 16 | 0.001390 | 0.739308 | 0.001028 | [0.139830,0.197600,0.277123,0.541347] | 0.047194 | 0.203 |
| CW90 | 17 | 0.001377 | 0.738270 | 0.001017 | [0.143331,0.189419,0.275902,0.511299] | 0.042672 | 0.177 |
| CW90 | 18 | 0.002079 | 0.485032 | 0.001008 | [0.082469,0.122291,0.339488,0.932426] | 0.208220 | 0.653 |
| CW90 | 19 | 0.001397 | 0.712867 | 0.000996 | [0.126611,0.206516,0.285890,0.685978] | 0.076368 | 0.343 |
| CW90 | 20 | 0.001631 | 0.607231 | 0.000990 | [0.710678,0.195572,0.997218,0.700700] | 0.144739 | 0.000 |
| CCW90 | 1 | 0.006338 | 0.670392 | 0.004249 | [0.145772,0.209198,0.464641,1.000000] | 0.252162 | 0.643 |
| CCW90 | 2 | 0.007035 | 0.527956 | 0.003714 | [0.131939,0.200027,0.393063,1.000000] | 0.208892 | 0.842 |
| CCW90 | 3 | 0.004032 | 0.591727 | 0.002386 | [0.121279,0.175211,0.514033,0.984313] | 0.317778 | 0.591 |
| CCW90 | 4 | 0.004328 | 0.353311 | 0.001529 | [0.114771,0.175008,0.445714,0.991589] | 0.270242 | 0.731 |
| CCW90 | 5 | 0.002968 | 0.458639 | 0.001361 | [0.093810,0.179507,0.574521,0.995789] | 0.392396 | 0.542 |
| CCW90 | 6 | 0.001072 | 0.675809 | 0.000725 | [0.122373,0.216967,0.637302,0.995606] | 0.400943 | 0.481 |
| CCW90 | 7 | 0.002971 | 0.212295 | 0.000631 | [0.057450,0.191955,0.519573,0.997871] | 0.372433 | 0.573 |
| CCW90 | 8 | 0.000958 | 0.649895 | 0.000622 | [0.124645,0.175763,0.591583,0.990485] | 0.380424 | 0.494 |
| CCW90 | 9 | 0.000916 | 0.655343 | 0.000600 | [0.601831,0.036995,1.000000,1.000000] | 0.383438 | 0.000 |
| CCW90 | 10 | 0.000918 | 0.570198 | 0.000524 | [0.137116,0.167692,0.631005,1.000000] | 0.411068 | 0.434 |
| CCW90 | 11 | 0.000751 | 0.649833 | 0.000488 | [0.606405,0.133707,1.000000,0.942364] | 0.318283 | 0.000 |
| CCW90 | 12 | 0.000654 | 0.635254 | 0.000415 | [0.623798,0.084421,1.000000,0.985448] | 0.338968 | 0.000 |
| CCW90 | 13 | 0.000559 | 0.716727 | 0.000401 | [0.525657,0.256600,0.769935,0.513822] | 0.062834 | 0.000 |
| CCW90 | 14 | 0.000592 | 0.619899 | 0.000367 | [0.632597,0.169428,1.000000,0.948965] | 0.286404 | 0.000 |
| CCW90 | 15 | 0.000391 | 0.726804 | 0.000284 | [0.523613,0.259941,0.766114,0.536751] | 0.067127 | 0.000 |
| CCW90 | 16 | 0.000390 | 0.708160 | 0.000276 | [0.703672,0.947517,1.000000,0.996858] | 0.014621 | 0.000 |
| CCW90 | 17 | 0.000340 | 0.763369 | 0.000259 | [0.528928,0.253193,0.771563,0.479281] | 0.054857 | 0.000 |
| CCW90 | 18 | 0.000400 | 0.628405 | 0.000251 | [0.574983,0.043063,1.000000,0.938322] | 0.380501 | 0.000 |
| CCW90 | 19 | 0.000384 | 0.641192 | 0.000246 | [0.184535,0.153276,0.867235,0.999834] | 0.577945 | 0.238 |
| CCW90 | 20 | 0.000375 | 0.629522 | 0.000236 | [0.585064,0.167017,1.000000,0.999258] | 0.345326 | 0.000 |
classification=DOMAIN_RECALL_GAP original_gate=no_fish
threshold_borderline_band=[0.18,0.2)
best_correct_confidence_by_orientation={"CCW90": [0.004248765491018602, 0.6433822044798577], "CW90": [0.0022175997989215546, 0.7651359841210706], "ORIGINAL": [0.0011032886460728974, 0.532909387899372]}


## Case B raw ONNX Top-10 (confirmed probe)

## 181193.jpg (1152x1536)
| orientation | rank | objectness | fish_probability | confidence | bbox_original_xyxy | area_ratio | IoU_GT |
|---|---:|---:|---:|---:|---|---:|---:|
| ORIGINAL | 1 | 0.012112 | 0.675594 | 0.008183 | [0.558816,0.363477,0.874994,0.818982] | 0.144021 | 0.169 |
| ORIGINAL | 2 | 0.011377 | 0.673424 | 0.007661 | [0.558700,0.326340,0.920535,0.829910] | 0.182209 | 0.159 |
| ORIGINAL | 3 | 0.007959 | 0.735102 | 0.005851 | [0.552715,0.372388,0.884833,0.822614] | 0.149528 | 0.174 |
| ORIGINAL | 4 | 0.006429 | 0.749684 | 0.004820 | [0.331521,0.384417,0.583901,0.841953] | 0.115473 | 0.559 |
| ORIGINAL | 5 | 0.005633 | 0.743203 | 0.004187 | [0.327011,0.387232,0.579312,0.844506] | 0.115371 | 0.559 |
| ORIGINAL | 6 | 0.005024 | 0.661525 | 0.003323 | [0.054587,0.799867,0.554009,1.000000] | 0.099951 | 0.120 |
| ORIGINAL | 7 | 0.004500 | 0.723778 | 0.003257 | [0.323845,0.397094,0.563259,0.850849] | 0.108635 | 0.526 |
| ORIGINAL | 8 | 0.004221 | 0.744354 | 0.003142 | [0.557297,0.326685,0.902945,0.834983] | 0.175692 | 0.167 |
| ORIGINAL | 9 | 0.003386 | 0.657419 | 0.002226 | [0.067957,0.802420,0.548693,1.000000] | 0.094984 | 0.117 |
| ORIGINAL | 10 | 0.002714 | 0.778305 | 0.002112 | [0.329087,0.382146,0.557198,0.854760] | 0.107809 | 0.522 |
| CW90 | 1 | 0.250154 | 0.787933 | 0.197105 | [0.522416,0.329314,0.972723,0.771111] | 0.198944 | 0.181 |
| CW90 | 2 | 0.238353 | 0.795306 | 0.189564 | [0.543203,0.339672,0.962835,0.779979] | 0.184767 | 0.162 |
| CW90 | 3 | 0.169148 | 0.808858 | 0.136816 | [0.534210,0.336229,0.955226,0.786709] | 0.189659 | 0.176 |
| CW90 | 4 | 0.168029 | 0.801437 | 0.134665 | [0.518398,0.259970,0.958184,0.781574] | 0.229394 | 0.177 |
| CW90 | 5 | 0.025800 | 0.720025 | 0.018577 | [0.308856,0.351716,0.615880,0.896091] | 0.167136 | 0.758 |
| CW90 | 6 | 0.023280 | 0.762731 | 0.017757 | [0.530313,0.321672,0.911489,0.812244] | 0.186994 | 0.196 |
| CW90 | 7 | 0.019097 | 0.794886 | 0.015180 | [0.544446,0.340648,0.925197,0.788849] | 0.170653 | 0.171 |
| CW90 | 8 | 0.020065 | 0.740906 | 0.014867 | [0.294660,0.359663,0.585096,0.916688] | 0.161780 | 0.669 |
| CW90 | 9 | 0.016773 | 0.753194 | 0.012633 | [0.486154,0.025189,0.962332,0.774748] | 0.356924 | 0.161 |
| CW90 | 10 | 0.015452 | 0.748470 | 0.011565 | [0.308487,0.373993,0.588937,0.848250] | 0.133006 | 0.602 |
| CCW90 | 1 | 0.823237 | 0.771953 | 0.635500 | [0.342930,0.363650,0.605436,0.924975] | 0.147351 | 0.714 |
| CCW90 | 2 | 0.787502 | 0.792310 | 0.623946 | [0.357972,0.364683,0.614088,0.920467] | 0.142345 | 0.689 |
| CCW90 | 3 | 0.759928 | 0.770428 | 0.585470 | [0.345509,0.360305,0.595611,0.895984] | 0.133974 | 0.649 |
| CCW90 | 4 | 0.739170 | 0.777195 | 0.574479 | [0.346189,0.365409,0.610439,0.961178] | 0.157432 | 0.716 |
| CCW90 | 5 | 0.642886 | 0.781519 | 0.502427 | [0.334322,0.359982,0.600488,0.986614] | 0.166788 | 0.705 |
| CCW90 | 6 | 0.438179 | 0.780515 | 0.342005 | [0.326297,0.360550,0.605704,0.977351] | 0.172339 | 0.746 |
| CCW90 | 7 | 0.369101 | 0.813916 | 0.300417 | [0.326271,0.380471,0.614439,0.889371] | 0.146649 | 0.710 |
| CCW90 | 8 | 0.373221 | 0.792990 | 0.295961 | [0.604761,0.334585,0.885664,0.765435] | 0.121027 | 0.090 |
| CCW90 | 9 | 0.318909 | 0.824387 | 0.262904 | [0.322434,0.349801,0.624480,0.934288] | 0.176542 | 0.854 |
| CCW90 | 10 | 0.257112 | 0.797711 | 0.205101 | [0.592091,0.343737,0.892761,0.771487] | 0.128612 | 0.109 |
classification=ORIENTATION_RECALL_GAP original_gate=no_fish
threshold_borderline_band=[0.18,0.2)
best_correct_confidence_by_orientation={"CCW90": [0.6355004065391654, 0.7135640302038609], "CW90": [0.018576850706601533, 0.7577393796148602], "ORIGINAL": [0.004820067387921512, 0.55919072036095]}

## Production bounded-retry replay

Replayed against the same ONNX SHA and the unchanged production gate. Each retry attempt is retained in trace; CW90 and CCW90 both execute only because ORIGINAL is NO_FISH.

| Fixture | Attempts | Selected | Selection reason | Gate | Original-coordinate bbox | Area ratio | Crop from original pixels |
|---|---|---|---|---|---|---:|---|
| Case A original | ORIGINAL, CW90, CCW90 | NONE | `NO_FISH_AFTER_BOUNDED_RETRIES` | NO_FISH | — | — | — |
| Case B night/flash | ORIGINAL, CW90, CCW90 | CCW90 | `GOOD_HIGHEST_RANK_SCORE` | READY | `[0.342930, 0.363650, 0.605436, 0.924975]` | `0.147351` | `[349, 429, 743, 1536]` |

Case B’s detector replay uses a byte-for-byte Android instrumentation fixture (`detector_case_b_night_flash.jpg`, SHA-256 `a7ed3bb0410c191364f6d853b4078aebdc2f3b9baa4da34b7b3f6e8e0e93a56b`) and asserts the original-coordinate bbox within `±0.015` per coordinate, area within `±0.02`, and crop edges within `±24` pixels. The instrumentation test also requires the production pipeline to reach `CLASSIFYING`; it does not require a species label.

## Runtime arbitration and trace contract

Policy version: `DETECTOR_ORIENTATION_RETRY_v2`.

1. Always evaluate ORIGINAL first.
2. If ORIGINAL is not `NO_FISH`, preserve the original result and stop after one detector evaluation.
3. If ORIGINAL is `NO_FISH`, run both CW90 and CCW90, map detections to original coordinates, and assess both with the unchanged quality gate.
4. Select by classifier-eligible GOOD, classifier-eligible WARNING, non-eligible fish-present, then NO_FISH. Within a class use the existing `confidence × sqrt(areaRatio)` score; exact ties preserve CW90 before CCW90.
5. Maximum calls remain one on original success and three after original NO_FISH. No 180° attempt, angle sweep, or threshold scan is used.

Each executed attempt records orientation, detection count, top confidence, quality status, and quality level. The selected attempt, selection reason, original image dimensions, detector version/SHA, and original-coordinate final bbox are carried in the inference trace. Classifier crops are cut from original source pixels. Raw diagnostic candidates remain outside consumer-facing output.

## Detector V0.2 data and training status

`DET_DS_v0.1` remains frozen. Case A and Case B remain held-out incident regressions and are not training samples. DET_DS_v0.2 preparation continues independently; it requires human-reviewed positive and matched negative labels before training. `DET_FISH_v0.2` training has not started. The training gate requires explicit 90°, 180°, and 270° orientation augmentation and separate overall recall, hard-case recall, negative false-positive rate, AP50, and AP50:95 reporting.

## Acceptance status

- Case A diagnosis: `DOMAIN_RECALL_GAP`, final on original source.
- Case B bounded detector implementation: awaiting Android CI and physical-device acceptance.
- Android regression fixture and instrumentation test are authored; CI runtime matrix remains the required execution gate.
- APK status: not produced until the Android build and required gates pass.
- Overall incident remains `IN_PROGRESS_CASE_A_FIXTURE_AND_DET_FISH_V0_2`; the Case A diagnosis is closed, but this detector repair does not recover Case A and v0.2 data/training remains open.
