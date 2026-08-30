#!/usr/bin/env python3
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("REGISTRY_DB_URL", "sqlite:///:memory:")
os.environ["APP_GIT_COMMIT"] = "a" * 40
os.environ["K_REVISION"] = "yujian-model-factory-console-smoke"
os.environ["K_SERVICE"] = "yujian-model-factory-console"

from app.entry import app, deployment_health  # noqa: E402
from app.secure import PUBLIC_PATHS  # noqa: E402


def require_text(path: str, tokens: list[str]) -> None:
    text = (ROOT / path).read_text(encoding="utf-8")
    for token in tokens:
        assert token in text, f"{path} missing {token!r}"


def main() -> None:
    payload = deployment_health()
    assert payload["status"] == "ok", payload
    assert payload["git_commit"] == "a" * 40, payload
    assert payload["revision"] == "yujian-model-factory-console-smoke", payload
    assert payload["service"] == "yujian-model-factory-console", payload
    assert "/health/deploy" in PUBLIC_PATHS
    assert "/health/deploy" in app.openapi()["paths"]

    require_text(
        ".github/workflows/uat-deploy.yml",
        [
            "workflow_run:",
            "branches:\n      - main",
            "YUJIAN_UAT_DEPLOY_ENABLED",
            "id-token: write",
            "google-github-actions/auth@v3.0.0",
            "google-github-actions/setup-gcloud@v3.0.1",
            "projects/571785698442/locations/global/workloadIdentityPools/github-actions/providers/yujian-main",
            "scripts/deploy_console_runtime.sh",
        ],
    )
    require_text(
        "scripts/deploy_console_runtime.sh",
        [
            'PROJECT_NUMBER="${PROJECT_NUMBER:-571785698442}"',
            'BUILD_SA="${BUILD_SA:-${PROJECT_NUMBER}-compute@developer.gserviceaccount.com}"',
            "--source .",
            '--build-service-account "$BUILD_SA"',
            "--update-env-vars=\"APP_GIT_COMMIT=${GIT_SHA}\"",
            "roles/run.builder",
            "roles/iam.serviceAccountUser",
            "/health",
            "/health/deploy",
            "DEPLOYED_SHA",
            "HEALTH_REVISION",
        ],
    )
    bootstrap = (ROOT / "scripts/bootstrap_github_wif.sh").read_text(encoding="utf-8")
    for token in (
        "workload-identity-pools",
        "roles/run.sourceDeveloper",
        "roles/serviceusage.serviceUsageConsumer",
        "roles/run.builder",
        "roles/iam.workloadIdentityUser",
        "assertion.repository",
        "assertion.ref=='refs/heads/main'",
        'gcloud iam service-accounts add-iam-policy-binding "$BUILD_SA"',
        '--member="serviceAccount:${DEPLOY_SA}"',
        '--role="roles/iam.serviceAccountUser"',
    ):
        assert token in bootstrap, f"bootstrap_github_wif.sh missing {token!r}"
    assert bootstrap.count('roles/iam.serviceAccountUser') >= 2, "runtime and build identities both require deployer actAs"

    for path in (".gitignore", ".gcloudignore", ".dockerignore"):
        require_text(path, ["gha-creds-*.json"])

    print("UAT deploy config smoke OK", payload)


if __name__ == "__main__":
    main()
