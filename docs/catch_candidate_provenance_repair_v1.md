# Candidate provenance repair and safe infrastructure release V1

Base main: `3c79c00db68ace94d8f6a7dcb4105d0026a09323`. Candidate product source: `32b9463887c6620b32a3dfdd986eccc6ff4a569b` in PR177, which remains Draft/Open/Unmerged. The source adds /health git_commit/revision/service from runtime environment; its existing safe targeted CI includes an isolated AST health test. The product is not merged by this infrastructure proposal.

## Root cause and repair

Failed generic UAT39816120669 checked out main3c79 and combined service-level status.url with latestReadyRevisionName. Default service traffic and zero-traffic tagged candidates are not the same route. The successful dedicated B2 tagged run39815023184 used exact PR source cb9e and validated its tagged candidate00345; generic UAT failure does not mean that older candidate failed its own source check.

Runtime deployment now verifies revision-level APP_GIT_COMMIT and resolves that revision's tag URL. Untagged route is allowed only for exclusive100% expected-revision traffic; mismatched source, zero-traffic unroutable revision and split default traffic fail closed. Service-template SHA is no longer treated as responding-revision proof.

Generic UAT preflight checks actual infrastructure-only file scope instead of relying on a marker omitted by generated merge subjects. The dedicated main-only deployment remains fixed to the exact PR177 HEAD and same repository/branch, requires latest exact-source necessary CI SUCCESS before cloud auth, uses --no-traffic with the existing tag/service/WIF, checks default traffic equality, and now validates BOTH /health and /health/deploy exact SHA/Revision. No IAM or production traffic mutation is introduced.

## Explicit safe CI boundary

Current user P0 scope authorizes a materially safer infrastructure test route: a strict task-specific exact path allowlist in scripts/classify_candidate_infra_change.py. Empty, unknown, application, model or unrelated test paths are NOT infrastructure-only. Negative tests prove app/main.py, app/catch_recognition_api.py, app/detector_runtime.py and the production recognition test cannot be classified as safe infra changes.

For the exact allowlisted infrastructure scope ONLY, CI runs6 pure JSON routing tests+4 negative scope tests, shell syntax, Python compile and YAML parsing. It does not import the application, models or cloud SDK, contact metadata, run a browser/photo fixture, or launch the previously blocked full test suite. CI summary explicitly records FULL_SUITE=BLOCKED_AUTO_REVIEW_METADATA_ACCESS and does not claim product/runtime/inference PASS. All non-allowlisted changes retain the original existing jobs; source PR177 retains its separately approved targeted source CI. This is not rerunning prohibited tests in another environment and does not weaken source deployment guards.

## Validation and promotion

Local6 route tests+4 negative scope tests PASS; shell/Python/YAML/diff checks PASS. The /health product-source test separately passed in PR177 local isolated stdlib execution. Source CI and this infrastructure PR CI must both pass their applicable scopes before main promotion. Production inference/model accuracy, authenticated product page visual acceptance and user-uploaded-photo UAT remain NOT_RUN. Root AGENTS/registry/admission are absent in Yujian; no Admission PASS is invented.

The earlier independent preparation branch9bf0d9aa remains historical staging. This main promotion omits app/main.py and product source; its pipeline pin points at the separately tested, unmerged PR177 candidate. No private image/ZIP is included.
