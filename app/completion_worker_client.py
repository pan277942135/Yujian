"""Strict HTTP client for the experimental GPU Completion Worker."""
from __future__ import annotations
import json, os, urllib.error, urllib.request
from typing import Any
class CompletionWorkerError(RuntimeError):
    def __init__(self, error_code: str, message: str, *, status_code: int | None = None):
        super().__init__(message); self.error_code=error_code; self.status_code=status_code
def invoke_completion_worker(*, image_uri: str, mask_uri: str, prompt: str) -> dict[str, Any]:
    base_url=os.getenv("FISH_COMPLETION_WORKER_URL","").strip().rstrip("/")
    if not base_url: raise CompletionWorkerError("COMPLETION_WORKER_NOT_CONFIGURED","FISH_COMPLETION_WORKER_URL is not configured")
    payload=json.dumps({"image_uri":image_uri,"mask_uri":mask_uri,"prompt":prompt},separators=(",",":")).encode()
    req=urllib.request.Request(f"{base_url}/completion",data=payload,method="POST",headers={"Content-Type":"application/json","Accept":"application/json"})
    token=os.getenv("FISH_COMPLETION_WORKER_TOKEN","").strip()
    if token: req.add_header("Authorization",f"Bearer {token}")
    try:
        with urllib.request.urlopen(req,timeout=float(os.getenv("FISH_COMPLETION_WORKER_TIMEOUT_SECONDS","900"))) as response:
            status_code=response.status; body=response.read()
    except urllib.error.HTTPError as exc:
        raise CompletionWorkerError("COMPLETION_WORKER_HTTP_ERROR",exc.read().decode("utf-8","replace")[:2000] or str(exc),status_code=exc.code) from exc
    except (urllib.error.URLError,TimeoutError) as exc:
        raise CompletionWorkerError("COMPLETION_WORKER_UNREACHABLE",str(exc)) from exc
    try: result=json.loads(body.decode())
    except (UnicodeDecodeError,json.JSONDecodeError) as exc:
        raise CompletionWorkerError("COMPLETION_WORKER_INVALID_JSON","Worker returned a non-JSON response",status_code=status_code) from exc
    if not isinstance(result,dict): raise CompletionWorkerError("COMPLETION_WORKER_INVALID_RESPONSE","Worker response is not an object")
    missing=[key for key in ("result_uri","model_version","inference_time_ms") if not result.get(key)]
    if missing: raise CompletionWorkerError("COMPLETION_WORKER_INVALID_RESPONSE",f"Worker response is missing: {', '.join(missing)}",status_code=status_code)
    return result
