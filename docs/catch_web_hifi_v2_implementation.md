# AI 渔获识别 Web Hi-Fi V2 — B2 实施记录

Task: `YUJIAN_AI_CATCH_RECOGNITION_WEB_FOUNDATION_AND_ARCHITECTURE_V2`

## Authority and admission

- Web PR #177 remains Draft/Open/Unmerged; implementation is based on the live product branch and does not touch `main` or production traffic.
- Work A / `Yujian_App` PR #190 HEAD `33e781e2fb2374138b14713b41fb91f5f528bf83` supplies the candidate visual contract, Handoff A/B, nine machine contracts, and G01–G18 gap matrix. It remains Draft and `MAIN_ADOPTED=NO`; all values are implemented as candidate direction, not a production or pixel-freeze authority.
- The product repository has no `AGENTS.md` and no `CONTROL_PLANE_REGISTRY.yaml`; no registered Admission entry or product control-plane rules were found. Admission is `BLOCKED_MISSING_REGISTRY`, not PASS.
- Candidate source artwork SHA `5f07c2f65ea33f14c8517dddbb7f8d6b7369a6ed980cba11cc92014f4e6a75f9` has verified bytes but no traced user approval or public publication permission. It is not bundled. The `208758.jpg` reference also lacks publication permission and is not bundled.

## Implemented contract behavior

- Existing page route, authentication, API adapter, save endpoint, and 3+2 response handling are retained. The product shell is page-scoped; the platform catch-recognition menu route remains intact.
- Desktop keeps three equal stage columns at 1280px and above; 1024–1279px and 768–1023px use two columns plus a full-width result. Below 768px the selected stage is shown with enabled stage navigation. The five-step strip scrolls horizontally within its own region.
- Upload and Processing continue to use EXIF-oriented source dimensions and `object-fit: contain`. The Result Hero uses the same-source foreground with a low-contrast, blurred same-source backdrop. At a 360px content width the high-state Hero uses the candidate 322:210 aspect ratio. No fish image is committed.
- SWOS renders seven photo-relative layers: Photo Base, Ambient Water, Caustic Field, Ripple Field, Peripheral Refraction, Fish Focus, and Residual Gold Trace. A deterministic timeline samples the eight Processing landmarks and both frozen relative-strength curves. Relative strength values are transformed into bounded layer opacity; they are not treated as CSS alpha.
- A real detector BBox may enable only a soft, broad local glow after the real response arrives. Its geometry travels inside the same PhotoStage as the source image. No precise outline is rendered. The three-layer Gold Core/Soft Glow/Halo and Chinese glyph-mask transfer remain unavailable until a separately qualified real mask exists.
- RR09 keeps seven frame landmarks at 0/100/200/380/550/680/850ms and total 850ms. The actual source image stage, its photo-relative optical layers, and optional BBox diagnostics move together through the 200ms R0→R0.5→R1 geometry transition. `R0.5` is explicitly painted at the real 100ms midpoint. Only a real `RESULT_HIGH` enters RR09; the real Chinese species name becomes visible at R4/680ms. Medium, Low, recovery, and transport failure do not play the High reveal.
- Existing model thresholds are unchanged. The 0.45/0.12 adapter decision versus the legacy 0.55 reference remains an authority conflict.
- Candidate muted token `#748897` is preserved as a reference token. Normal secondary copy uses the measured darker derivative `#637480` so 14px text clears 4.5:1 contrast on the page background. Pixel matching remains unmeasured.

## G01–G18 status against current PR #177 source

| Gap | Status | Implementation evidence / remaining condition |
|---|---|---|
| G01 | FIXED | `base.html` page-scoped `product_shell`; product header and return route retained. |
| G02 | FIXED | `catch_recognition.css`: 3 equal columns at 1280+, 2 columns at 768–1279, Result full row below 1280. Pixel comparison remains unavailable. |
| G03 | FIXED | `catch-page` lake-white background and restrained ambient accents; photo itself is not transformed into an underwater scene. |
| G04 | FIXED | 14px minimum ordinary copy, 16px form controls, contrast derivative documented above. |
| G05 | FIXED | Upload drop area min-height 260px; oriented images remain `contain`, without stretching or crop. |
| G06 | FIXED | Normal Processing view prioritizes the photo and real request state; technical pipeline list is inside closed diagnostics. |
| G07 | FIXED | Model JSON, version, SAM and BBox inspection stay behind Advanced Diagnostics. |
| G08 | FIXED | Seven independent photo-relative SWOS layers; deterministic P0–P7 curve interpolation; no dense web or mechanical scan. |
| G09 | PARTIAL_MASK_QUALIFICATION_BLOCKED | Real BBox soft glow only. Trusted Mask three-layer geometry is not available from the current API. |
| G10 | FIXED_CANDIDATE_GEOMETRY | Existing frozen Result form, metadata, story and dual CTA retained; Hero aspect ratio and same-source fallback implemented. CTA copy remains as existing frozen text. |
| G11 | FIXED | Same `PhotoSession` source and one moving PhotoStage contain both image and optical overlays; true 100ms R0.5 geometry is sampled. |
| G12 | PARTIAL_MASK_QUALIFICATION_BLOCKED | HIGH-only timing/name reveal is implemented. Gold-to-glyph path is deliberately absent without trusted fish mask geometry. |
| G13 | FIXED | Three navigable page stages and five visual journey nodes, driven by actual session/result state rather than a progress clock. |
| G14 | IMPLEMENTED_AWAITING_UPDATED_CHROMIUM_CI | Nine candidate viewports and mobile stage/strip assertions are in `smoke_catch_recognition_chromium.py`; updated exact-SHA run is required. |
| G15 | FIXED | Catch-recognition menu fish and empty-result mark use SVG; no emoji is used for those product icons. |
| G16 | FIXED | The reveal entry remains guarded by a real HIGH response; MEDIUM/LOW and recovery use their existing direct states. |
| G17 | BLOCKED_AUTHORITY_CONFLICT | Existing 0.45/0.12 versus 0.55 references are recorded; no threshold was changed. |
| G18 | FIXED | No sample photo or generated result asset is included; selected image remains in browser-local object URL until explicit save. |

## Validation and release boundary

- Local Node geometry/controller/SWOS tests, JavaScript syntax, Python compilation of touched modules/scripts, and `git diff --check` are required on the implementation SHA.
- The local environment has no Chromium, Playwright, or FastAPI installation; real template rendering and browser capture must be evidenced by the repository's exact-SHA CI job. A UI fixture remains explicitly `UI_SIMULATION_ONLY` and never counts as live inference.
- Tagged-preview infrastructure PR #180 is the trusted main-path deployment route. In the B1 audit it was Draft/Open/Unmerged, had no review submissions, intentionally skipped infrastructure CI, and was pinned to B1 SHA `655c4af07a0244fcf29441b5652c48ca74f2e6a2`. The B2 source must be pinned there before any release; independent review/approval is still required. Do not merge #180 or run cloud deployment through another path to bypass it.
- No new Cloud Run revision, preview URL, production traffic change, model inference, real-photo visual acceptance, or user visual signoff is claimed here. Updated B2 Chromium evidence and a separately reviewed preview pin to the final exact SHA remain necessary.
