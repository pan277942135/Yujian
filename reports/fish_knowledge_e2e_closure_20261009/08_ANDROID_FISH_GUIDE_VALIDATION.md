# 08 — Android Fish Guide Validation

**Candidate:** `pan277942135/Yujian_App` / `fix/fish-guide-published-asset-render-parity-v1` @ `505690f0a681334b6c3411de50f55e026076b464`; PR #125 remains Open / Draft / Unmerged.  
**Status:** `BLOCKED_ANDROID_RUNTIME_EVIDENCE`.

## CI artifact/build evidence

GitHub Actions run `37886891748` (`Android CI`, run 1535) completed successfully. Its build job reports success for unit tests, runtime harness contract tests, debug APK build, AndroidTest APK compilation, lint and APK artifact upload. The separate `android-runtime` job is **skipped**. This Work did not operate the self-hosted runner, install an APK, launch Fish Guide, or capture device screenshots.

The run lists an artifact named `YuJian-debug-72c9f8a7c722a2e4e41a86cd564961dc143962de-MODEL_CROP_M1_v0.2`, archive size 105,845,514 bytes and archive digest `sha256:07863db22d2b1797362dca8a75c6e5740dcf2bd83b294d7401128632180e4b3f`. This is the GitHub artifact archive digest/size, **not** an independently extracted APK SHA-256 or runtime proof.

## Runtime matrix

| Check | Result |
|---|---|
| Staging public API connected to `FishGuideRepository` | BLOCKED — no Staging deployment/assets |
| Same published version ID through API, cache and UI | NOT RUN |
| LIT / UNLIT screenshots and Frozen geometry/media comparison | NOT RUN |
| Five card roles against real ACTIVE assets | BLOCKED — production role/version bindings not proven; no Staging role versions |
| Image failure, wrong ID, DRAFT suppression, cache update, species switch | NOT RUN at runtime |
| Adjacent cards, pager restoration, 0/1/N saved catches | NOT RUN at runtime |
| APK SHA, device/API/resolution/density/font-scale/foreground Activity | NOT CAPTURED |

No static Frozen image was used as runtime evidence. Android status remains `CLIENT_PENDING`; the independent Android Validation Work owner must run the exact APK against authorized Staging assets and retain the requested screenshot/API provenance. Production is not approved by this report.
