# AI 渔获识别 Web — Foundation V2 / Checkpoint B1

Task: `YUJIAN_AI_CATCH_RECOGNITION_WEB_FOUNDATION_AND_ARCHITECTURE_V2`

## Baseline and authority

- Product repository: `pan277942135/Yujian`; initial main `1d01231477645f5b888b8815c5602576dac90c61`.
- PR #177 initial head: `7da5c1100d981cfe88317b39b5b95713bdba91d7`; Draft/Open/Unmerged. Changes increment that product work line; no main merge or default traffic change.
- No `AGENTS.md` or `CONTROL_PLANE_REGISTRY.yaml` exists in the audited product tree. Actual entry is README → `docs/PROJECT_GUIDE.md`, which was read. Product formal Admission remains `BLOCKED_MISSING_REGISTRY`, not PASS.
- Work A: Yujian_App PR #190 at `33e781e2fb2374138b14713b41fb91f5f528bf83`. Handoff A/B, geometry, tokens, responsive, state, gap and authority references were read. It remains Draft and depends on #186/#187. Candidate geometry is design completion, not measurement from an approved image; `MAIN_ADOPTED=NO`, `RUNTIME_VISUAL_VERIFIED=NO`.
- Exact candidate references and implementation/pending scope: `docs/contracts/catch_web_foundation_v2.json`. No candidate has been relabeled as canonical authority.

## Implemented foundation

The existing Jinja base has a page-scoped `product_shell` mode. Only Catch Recognition omits the fixed sidebar; its route and the existing Model Factory menu are unchanged. The shared base continues to own the header and navigation. Other page templates still render their normal sidebar. No new frontend framework, inference backend, database or authentication system was added.

Three stage roots expose stable component identities. Desktop uses three columns from 1280px, intermediate screens two columns and a full-width result, and narrow screens a single vertical flow. The separate five-step journey describes the visual experience; water and fish-light effects explicitly remain unavailable. All optical hosts are inert, photo-relative mounts. Technical stage details and JSON are inside a closed diagnostic disclosure.

The existing view code was progressively migrated to separate API, controller, photo session/viewport, stage view and transition modules. One controller retains the file, request sequence, result and explicit-save state. New selection, cancellation and clearing invalidate older requests and every pending reveal step. Async decode and login completion are also session guarded. New photo selection is held while an explicitly initiated save is in flight.

PhotoSession stores source/session identity, browser EXIF-oriented dimensions, source aspect ratio, geometry, validated BBox, unavailable mask and actual result. The same contain/cover geometry function supplies photo content rectangles and BBox mapping. ResizeObserver recomputes geometry. During R0→R1 the actual Processing image node moves into one TransitionHost; no new transition image is loaded, and the destination image is hidden until cleanup. Content rectangles are interpolated with uniform aspect, remeasured on every animation frame, and the original node returns on completion/cancel.

The existing 2800ms Processing and 850ms RR09 clocks and all 15 frame definitions are preserved, including R0.5 at 100ms. They match PR190's references to PR186/187. Only a real HIGH response can enter the nominal handoff. Slow responses extend processing. MEDIUM/LOW and recovery states do not play the HIGH reveal. No precise mask or glyph-gold effect is fabricated.

Console authentication still protects both the page and `POST /api/product/catch-recognition/infer`. Login now carries an origin-relative, escaped return path to the same recognition route; unsafe/external paths are rejected. Existing app-account bearer login remains required only for explicit catch saves, as before. No internal service key is exposed. Detector, crop, classifier, save endpoints and thresholds were not modified. The observed 0.45/0.12 versus legacy 0.55 threshold discrepancy remains unresolved and documented.

## Validation and limits

- Python compile, JS syntax for all six frontend scripts, Jinja render and referenced assets, `git diff --check`: PASS.
- Safe targeted Python allowlist: 34 passed. Includes existing product API/pipeline tests, all platform template rendering, cookie/return-route/API protection, and test-network protection.
- Node geometry/controller tests: 3 passed; portrait/landscape/narrow sources, cover/contain, invalid BBox, real midpoint and stale requests.
- Chromium local UI simulation: nine viewports (1440×1000, 1280×800, 1024×768, 768×1024, 480×900, 430×932, 390×844, 360×640, 320×480), shell, empty result, diagnostics default closed, no horizontal overflow, same photo transition, nominal timing, cancel/reselect during reveal, HIGH/non-HIGH/error, slow response and Reduce Motion. Results are explicitly `UI_SIMULATION_ONLY`; the harness generates plain color fixtures, never uses a real fish photo and never calls production inference. Screenshots are CI artifacts, not source assets.
- Full suite: `BLOCKED_AUTO_REVIEW_METADATA_ACCESS`, NOT PASS. A full run was attempted with an initial socket fence; automatic approval rejected its subsequent operation because an unintended cloud metadata probe could access instance credentials. It was not rerun locally or remotely. Test setup now supplies deterministic anonymous credentials before collection and blocks metadata ping, plus outbound socket/DNS protection. The specific full-suite caller and its first failure remain unresolved; these defenses do not constitute full-suite completion.
- PR177 CI explicitly runs the safe targeted allowlist and labels the full suite NOT_RUN; this narrower job is not full-regression or production-release evidence. Existing unrelated smoke scripts retain their fake dependencies. No cloud inference or save was exercised.

## Environment and release handoff

Historical tagged deployment was confirmed from successful workflow run `39776248323`: `catch-pr177`, revision `yujian-model-factory-console-00344-rub`, source `7da5c1100d981cfe88317b39b5b95713bdba91d7`, no default traffic change. This is the OLD implementation, not B1.

The trusted main workflow pins that old SHA. Publishing B1 requires an independent infrastructure PR to update the explicit approved source SHA, verify same repository + expected branch + exact head and successful necessary CI before authentication, and retain `--no-traffic`/fixed tag/default-traffic comparison/deploy-health SHA checks. That PR must not be merged without approval under this task's section 2.3/11.2. Product #177 remains Draft and unmerged.

B2 final visual closure remains pending authoritative adoption of Work A, approved-image measurement/permission, full SWOS/RR09 renderer, mobile active-stage focus and Result media/CTA authority conflicts. The existing full-source contain fallback protects photo geometry for B1. No Android source or APK work was performed.
