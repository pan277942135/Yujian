#!/usr/bin/env bash
set -euo pipefail

IMAGE="${WORKER_IMAGE:-}"
CHECKPOINT_DIR="${POWERPAINT_CHECKPOINT_DIR:-/models/ppt-v1}"
OUTPUT_BUCKET="${COMPLETION_OUTPUT_BUCKET:-}"
PORT="${WORKER_PORT:-8080}"
CONTAINER_NAME="${WORKER_CONTAINER_NAME:-fish-completion-worker}"
TOKEN="${WORKER_AUTH_TOKEN:-}"

usage() {
  cat <<'EOF'
Usage:
  bash deploy_fish_completion_worker.sh --image IMAGE --bucket GCS_BUCKET [options]

Options:
  --image IMAGE              Worker image to pull and run.
  --checkpoint-dir PATH      Existing local PowerPaint checkpoint directory.
  --bucket NAME              GCS bucket for generated results.
  --port PORT                Host port, default 8080.
  --token TOKEN              Optional worker bearer token.
  --name NAME                Docker container name.
  --help                     Show this help.

This script does not download or create the PowerPaint checkpoint.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
    --bucket) OUTPUT_BUCKET="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --token) TOKEN="$2"; shift 2 ;;
    --name) CONTAINER_NAME="$2"; shift 2 ;;
    --help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$IMAGE" ]] || { echo "ERROR: --image or WORKER_IMAGE is required" >&2; exit 2; }
[[ -n "$OUTPUT_BUCKET" ]] || { echo "ERROR: --bucket or COMPLETION_OUTPUT_BUCKET is required" >&2; exit 2; }
[[ -d "$CHECKPOINT_DIR" ]] || { echo "ERROR: checkpoint directory does not exist: $CHECKPOINT_DIR" >&2; exit 2; }
command -v docker >/dev/null || { echo "ERROR: docker is not installed" >&2; exit 2; }

docker pull "$IMAGE"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker_args=(
  run -d
  --name "$CONTAINER_NAME"
  --restart unless-stopped
  --gpus all
  -p "$PORT:8080"
  -v "$CHECKPOINT_DIR:/models/ppt-v1:ro"
  -e "POWERPAINT_CHECKPOINT_DIR=/models/ppt-v1"
  -e "COMPLETION_OUTPUT_BUCKET=$OUTPUT_BUCKET"
  -e "POWERPAINT_LOCAL_FILES_ONLY=true"
)
if [[ -n "$TOKEN" ]]; then
  docker_args+=( -e "WORKER_AUTH_TOKEN=$TOKEN" )
fi
docker_args+=( "$IMAGE" )

docker "${docker_args[@]}"
echo "Worker started: http://127.0.0.1:$PORT"
