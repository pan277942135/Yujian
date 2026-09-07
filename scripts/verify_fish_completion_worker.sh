#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${WORKER_URL:-http://127.0.0.1:8080}"
IMAGE_URI=""
MASK_URI=""
PROMPT="complete missing fish body"
TOKEN="${WORKER_AUTH_TOKEN:-}"

usage() {
  cat <<'EOF'
Usage:
  bash verify_fish_completion_worker.sh --image-uri URI --mask-uri URI [options]

Options:
  --url URL                  Worker base URL, default http://127.0.0.1:8080.
  --image-uri URI            GCS URI or local URI for the ROI image.
  --mask-uri URI             GCS URI or local URI for the completion mask.
  --prompt TEXT              Prompt, default "complete missing fish body".
  --token TOKEN              Optional worker bearer token.
  --help                     Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url) BASE_URL="$2"; shift 2 ;;
    --image-uri) IMAGE_URI="$2"; shift 2 ;;
    --mask-uri) MASK_URI="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --token) TOKEN="$2"; shift 2 ;;
    --help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$IMAGE_URI" ]] || { echo "ERROR: --image-uri is required" >&2; exit 2; }
[[ -n "$MASK_URI" ]] || { echo "ERROR: --mask-uri is required" >&2; exit 2; }
command -v curl >/dev/null || { echo "ERROR: curl is not installed" >&2; exit 2; }
command -v python3 >/dev/null || { echo "ERROR: python3 is not installed" >&2; exit 2; }

auth_args=()
if [[ -n "$TOKEN" ]]; then
  auth_args=(-H "Authorization: Bearer $TOKEN")
fi

health_body="$(mktemp)"
trap 'rm -f "$health_body" "$response_body"' EXIT
health_code="$(curl --silent --show-error --output "$health_body" --write-out '%{http_code}' "${auth_args[@]}" "$BASE_URL/health")"
[[ "$health_code" == "200" ]] || { echo "HEALTH FAIL HTTP $health_code"; cat "$health_body"; exit 1; }
python3 - "$health_body" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
if data.get("status") != "ok":
    raise SystemExit(f"HEALTH FAIL: {data}")
print(f"HEALTH PASS: gpu={data.get('gpu', 'unknown')}")
PY

payload="$(python3 - "$IMAGE_URI" "$MASK_URI" "$PROMPT" <<'PY'
import json, sys
print(json.dumps({"image_uri": sys.argv[1], "mask_uri": sys.argv[2], "prompt": sys.argv[3]}))
PY
)"
response_body="$(mktemp)"
completion_code="$(curl --silent --show-error --output "$response_body" --write-out '%{http_code}' -H 'Content-Type: application/json' "${auth_args[@]}" --data "$payload" "$BASE_URL/completion")"
[[ "$completion_code" == "200" ]] || { echo "COMPLETION FAIL HTTP $completion_code"; cat "$response_body"; exit 1; }
python3 - "$response_body" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
for key in ("result_uri", "model_version", "inference_time_ms"):
    if not data.get(key):
        raise SystemExit(f"COMPLETION FAIL: missing {key}: {data}")
print(f"COMPLETION PASS: model={data['model_version']} inference_time_ms={data['inference_time_ms']} result_uri={data['result_uri']}")
PY
