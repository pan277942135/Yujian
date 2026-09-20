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
os.environ.pop("FEEDBACK_INGEST_KEY", None)

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
    assert payload["feedback_ingest_path"] == "/api/feedback/ingest", payload
    assert payload["feedback_ingest_key_configured"] is False, payload
    assert payload["qwen_refine_worker_configured"] is False, payload
    assert payload["qwen_refine_worker_url"] is None, payload
    assert "/health/deploy" in PUBLIC_PATHS
    assert "/health/deploy" in app.openapi()["paths"]
    assert "/health/detector" in PUBLIC_PATHS
    assert "/health/detector" in app.openapi()["paths"]
    assert "/api/feedback/ingest" in app.openapi()["paths"]

    require_text(
        ".github/workflows/uat-deploy.yml",
        [
            "workflow_run:",
            "branches:\n      - main",
            "YUJIAN_UAT_DEPLOY_ENABLED",
            "FISH_QWEN_REFINE_WORKER_URL: http://34.69.75.199:8002",
            "Qwen Refine Worker connectivity smoke",
            "Legacy Fish Portrait Worker connectivity smoke (non-blocking)",
            "continue-on-error: true",
            "fish-qwen-refine-worker",
            "Qwen-Image-Edit-2511",
            "FISH_QWEN_REFINE_WORKER_PATH: ${{ env.FISH_QWEN_REFINE_WORKER_PATH }}",
            "qwen_refine_worker_configured",
            "qwen_refine_worker_url",
            "FISH_QWEN_REFINE_WORKER_PATH",
            "gcloud run services describe",
            "FISH_QWEN_REFINE_WORKER_PATH: /refine",
            "id-token: write",
            "google-github-actions/auth@v3.0.0",
            "google-github-actions/setup-gcloud@v3.0.1",
            "projects/571785698442/locations/global/workloadIdentityPools/github-actions/providers/yujian-main",
            "scripts/deploy_console_runtime.sh",
            "repository: pan277942135/Yujian_App",
            "MODEL_TFLITE_URL: https://github.com/pan277942135/Yujian/releases/download/mobile-model-v0.2/fish_classifier_v0_2.tflite",
            "MODEL_TFLITE_SHA256: b77ea78e7f8554078ea3a79051039af1ace04f0ac4e2604da57d1dd8f0b010e7",
            "MODEL_TENSOR_CONTRACT_URL: https://github.com/pan277942135/Yujian/releases/download/mobile-model-v0.2/tensor_contract.json",
            "MODEL_TENSOR_CONTRACT_SHA256: f9a477f4f9ecd23b0162ee7f06c0f6965f005a52f17755e11f7b9283e104b1d8",
            "model_tensor_contract.json",
            "tensor_contract.json",
            "python3 scripts/verify_production_model.py",
            "actions/setup-java@v4",
            "gradle/actions/setup-gradle@v4",
            "YUJIAN_FEEDBACK_BASE_URL",
            "YUJIAN_FEEDBACK_INGEST_KEY",
            "::add-mask::$INGEST_KEY",
            "reactivecircus/android-emulator-runner@v2",
            "api-level: 28",
            ":app:connectedDebugAndroidTest",
        ],
    )
    require_text(
        "scripts/deploy_console_runtime.sh",
        [
            'PROJECT_NUMBER="${PROJECT_NUMBER:-571785698442}"',
            'BUILD_SA="${BUILD_SA:-${PROJECT_NUMBER}-compute@developer.gserviceaccount.com}"',
            'BUILD_SA_RESOURCE="${BUILD_SA_RESOURCE:-projects/${PROJECT_ID}/serviceAccounts/${BUILD_SA}}"',
            "--source .",
            '--build-service-account "$BUILD_SA_RESOURCE"',
            'DEPLOY_ENV_VARS="APP_GIT_COMMIT=${GIT_SHA}"',
            'FEEDBACK_INGEST_KEY="$(python -c',
            '::add-mask::%s',
            "--timeout=1200s",
            '--update-env-vars="$DEPLOY_ENV_VARS"',
            "roles/run.builder",
            "roles/iam.serviceAccountUser",
            "/health",
            "/health/deploy",
            "/api/feedback/ingest",
            'smoke=true',
            'feedback_ingest_key_configured',
            'gcs_write_delete',
            'db_reachable',
            "DEPLOYED_SHA",
            "HEALTH_REVISION",
        ],
    )
    require_text(
        "app/feedback_ingest_api.py",
        [
            'smoke: bool = Form(default=False)',
            'prefix = "feedback/smoke" if smoke else "feedback/app"',
            'blob.upload_from_string',
            'blob.delete',
            'db.scalar(select(1))',
            '"gcs_write_delete": True',
            '"db_reachable": True',
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

    print("UAT deploy + feedback backend smoke + Android API28 E2E contract OK", payload)


if __name__ == "__main__":
    main()
