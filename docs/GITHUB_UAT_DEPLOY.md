# GitHub OIDC/WIF → Cloud Run UAT Deploy

YuJian uses GitHub Actions CI and a separate, gated UAT continuous-deployment workflow.

## Deployment contract

After one-time Workload Identity Federation bootstrap and after repository variable `YUJIAN_UAT_DEPLOY_ENABLED=true` is set:

1. code is merged to `main`;
2. the existing `ci` workflow validates the exact main SHA;
3. `UAT Deploy` is triggered only after that main CI run completes successfully;
4. GitHub requests a short-lived OIDC token;
5. Google Workload Identity Federation allows only `pan277942135/Yujian` on `refs/heads/main` to impersonate the dedicated deployer service account;
6. `scripts/deploy_console_runtime.sh` creates a new Cloud Run source revision without re-running infrastructure bootstrap or deploying the trainer;
7. the deployed service receives `APP_GIT_COMMIT=<exact tested SHA>`;
8. `/health` must return `status=ok`;
9. `/health/deploy` must return the exact Git SHA and the exact latest-ready Cloud Run revision, otherwise the deployment job fails.

## Fixed UAT resources

- GCP project: `gemini-api-project-503706`
- Project number: `571785698442`
- Region: `asia-east1`
- Cloud Run service: `yujian-model-factory-console`
- Runtime service account: `yujian-model-factory@gemini-api-project-503706.iam.gserviceaccount.com`
- GitHub deployer service account: `yujian-github-deployer@gemini-api-project-503706.iam.gserviceaccount.com`
- WIF pool: `github-actions`
- WIF provider: `yujian-main`

No long-lived Google service-account JSON key is stored in GitHub.

## One-time bootstrap

Run from an already authenticated Google Cloud Shell or another administrator shell:

```bash
TMP="$(mktemp -d)" && \
git clone --depth 1 https://github.com/pan277942135/Yujian.git "$TMP" && \
cd "$TMP" && \
bash scripts/bootstrap_github_wif.sh
```

The bootstrap is idempotent. It:

- enables IAM Credentials / STS / Cloud Run / Cloud Build / Artifact Registry APIs;
- creates the dedicated GitHub deployer service account if missing;
- grants only the source-deploy roles required for the deployer;
- grants `iam.serviceAccountUser` on the existing Cloud Run runtime identity;
- ensures the default source-build service account has `roles/run.builder`;
- creates or updates the Workload Identity Pool and GitHub OIDC provider;
- restricts the provider to this repository and `refs/heads/main`;
- grants `roles/iam.workloadIdentityUser` for this repository identity.

If GitHub CLI is authenticated in the same shell, bootstrap also sets:

```text
YUJIAN_UAT_DEPLOY_ENABLED=true
```

Otherwise set it once under GitHub repository Actions Variables, or run:

```bash
gh variable set YUJIAN_UAT_DEPLOY_ENABLED \
  --repo pan277942135/Yujian \
  --body true
```

## Deploy current main after bootstrap

Either wait for the next successful main CI run, or manually dispatch:

```bash
gh workflow run 'UAT Deploy' --repo pan277942135/Yujian
```

## Runtime-only means runtime-only

The automatic workflow does **not**:

- create Cloud SQL;
- rotate database or Console secrets;
- change bucket IAM;
- recreate the runtime service account;
- enable unrelated APIs;
- deploy the classifier trainer.

Those remain responsibilities of explicit infrastructure/bootstrap workflows. A Console-only code change therefore does not rebuild the training worker.

## Failure behavior

`UAT Deploy` does not deploy when:

- the enabling repository variable is not `true`;
- the upstream main CI run failed;
- the CI-tested SHA is already stale because a newer commit is on `main`.

The deployment fails if:

- GitHub OIDC/WIF authentication fails;
- source deployment fails;
- the Cloud Run template does not contain the expected `APP_GIT_COMMIT`;
- basic `/health` fails;
- `/health/deploy` reports a different SHA or revision.

This keeps GitHub `main`, the CI-tested commit, and the serving Cloud Run revision explicitly traceable.
