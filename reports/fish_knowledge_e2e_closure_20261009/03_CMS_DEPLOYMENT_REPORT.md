# 03 — CMS Deployment Report

**Candidate:** `ea7f50d5b4d5beadf51564bc2f0d75854ee49eed`  
**Status:** `BLOCKED_STAGING_ENVIRONMENT` — candidate was not deployed.

## What ran

- GitHub Actions `ci` run `37897272676` (run 1193), `validate` job: **success**. It compiled Python, checked deployment shell syntax, imported the app and ran the repository smoke/test steps.
- CMS PR #159 remains Draft / unmerged at the candidate commit.
- A fresh clean checkout was verified at the exact candidate SHA.

## What did not run

- No image build or Cloud Run revision was created by this Work.
- No Staging service URL, revision name, image digest, health response, DB connection or authorization identity was obtained.
- CMS operator and non-operator authorization tests were not run against a deployed service.
- CMS UI controls were not tested in a live browser session against Staging.

## Deployment safety finding

The only discovered deploy workflow is `.github/workflows/uat-deploy.yml`; its configured project, service and bucket are the production-linked values documented in `01_ENVIRONMENT_AUDIT.md`. Its `workflow_dispatch` path was **not** invoked. It cannot satisfy Gate 1 as a separate Staging deployment.

`/health` was not used to claim a deployment. Public production API responses are recorded separately and do not identify the deployed CMS commit or Cloud Run revision.
