# Candidate provenance repair V1

Scope: user P0 Work B; base main `3c79c00db68ace94d8f6a7dcb4105d0026a09323`. Product PR177 remains Draft/Open/Unmerged at `cb9e3780afb5e2904e3dfe5148210a4ccc3747b4`; this repair does not change that branch, IAM, or production traffic.

## Observed deployment distinction

The successful dedicated tagged preview run is https://github.com/pan277942135/Yujian/actions/runs/39815023184 (deploy job119951230406). It checks out PR177 exact source cb9e, uses --no-traffic and tag catch-pr177, verifies default traffic equality, and validates /health/deploy SHA and K_REVISION.

The failed run https://github.com/pan277942135/Yujian/actions/runs/39816120669 is the generic UAT workflow, not the dedicated product-preview workflow. Its checkout/GIT_SHA is main3c79. The generic runtime script reads status.url (default traffic) and latestReadyRevisionName, then expects both to identify one revision. Zero-traffic tagged previews make that assumption invalid; service URLs and candidate tag URLs are different routes. A service-template SHA also does not prove which revision answered HTTP.

## Minimal staged repair

- Pure JSON resolver uses revision-level APP_GIT_COMMIT, checks latest-ready identity, and chooses a matching tagged revision URL. Service URL is allowed only when the expected revision exclusively owns default traffic. Wrong SHA, split default traffic without a tag, or an unroutable zero-traffic revision fails before HTTP.
- Runtime /health smoke and /health/deploy smoke both require exact source SHA and revision. /health is minimally extended to return git_commit / revision / service from APP_GIT_COMMIT / K_REVISION / K_SERVICE.
- Generic UAT preflight classifies infrastructure-only changed files instead of relying solely on a commit-message marker. Dedicated preview infrastructure changes do not trigger an unrelated normal UAT production deploy.

## Safe local verification

Seven stdlib unittest cases PASS: exact health-handler values extracted using AST (no application/model/cloud imports); tagged route selection; wrong SHA rejection; missing candidate route rejection; exclusive default traffic; split traffic rejection; latest-ready race rejection. Two executions of the actual preflight shell with stubbed gh/git PASS: infrastructure-only should_deploy=false; product source should_deploy=true. Shell syntax, Python compile, YAML parse, git diff check PASS.

No broad pytest or cloud metadata/telemetry test was run. Root AGENTS.md, CONTROL_PLANE_REGISTRY.yaml and admission script are absent from this product repository; no product Admission PASS is claimed.

## Live candidate recheck (2026-10-23 Beijing time)

URL: https://catch-pr177---yujian-model-factory-console-6254hdioya-de.a.run.app/platform/model/catch-recognition

- /health: HTTP200, status=ok, version=0.1.0; current deployed source lacks SHA/revision fields. This requirement remains OPEN until repaired source is safely deployed.
- /health/deploy: HTTP200, git_commit=cb9e3780afb5e2904e3dfe5148210a4ccc3747b4, revision=yujian-model-factory-console-00345-geg, service=yujian-model-factory-console.
- /login: HTTP200; authenticated product page unauthenticated request returns303 to /login?next=%2Fplatform%2Fmodel%2Fcatch-recognition.
- POST /api/product/catch-recognition/infer with empty request and no credentials returns401 requiring console login. No image was sent and no inference success is claimed.

## Remaining release gate

This is a controlled infra repair branch only, not a deployed repair. No PR is created: main ci.yml runs the previously safety-blocked full pytest suite on every pull_request; launching it elsewhere would bypass the explicit metadata-access safety boundary. The safe targeted tests above do not claim that suite PASS.

The existing dedicated no-traffic workflow has no source input: TARGET_SHA is fixed to cb9e and its preflight requires live PR177 HEAD plus exact-source CI. It cannot deploy this independent repair branch without a separately approved source/pipeline review. Do not weaken or bypass its guard, change IAM, or repoint production traffic. Product rendering after authentication and real-photo inference remain user-login/UAT pending; user uploads photos personally.
