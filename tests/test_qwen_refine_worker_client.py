from __future__ import annotations

import base64

from app.qwen_refine_worker_client import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_PROMPT,
    QWEN_MODE,
    QWEN_MODEL_LABEL,
    check_qwen_refine_worker,
    invoke_qwen_refine_worker,
)


class _Response:
    status = 200

    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


def test_qwen_worker_health_and_refine_contract(monkeypatch):
    captured = []

    def _urlopen(request, timeout):
        captured.append((request, timeout))
        if request.method == "GET":
            return _Response(
                b'{"status":"ok","service":"fish-qwen-refine-worker","model":"Qwen-Image-Edit-2511"}'
            )
        return _Response(
            b'{"status":"success","refine_result_uri":"/output/refined.png","final_asset_uri":"/output/final.png","seed":12345,"elapsed_ms":1234.5}'
        )

    monkeypatch.setenv("FISH_QWEN_REFINE_WORKER_URL", "http://worker")
    monkeypatch.delenv("FISH_QWEN_REFINE_WORKER_TOKEN", raising=False)
    monkeypatch.setattr("urllib.request.urlopen", _urlopen)

    health = check_qwen_refine_worker()
    assert health["status"] == "READY"
    assert health["worker_model"] == QWEN_MODEL_LABEL

    source = "data:image/png;base64," + base64.b64encode(b"SAM_VISIBLE").decode("ascii")
    result = invoke_qwen_refine_worker(
        visible_fish_refined_image_uri=source,
        source_run_id="FCL_123",
        prompt=DEFAULT_PROMPT,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        steps=20,
        seed=12345,
        auto_straighten=False,
    )

    request = captured[1][0]
    assert request.full_url == "http://worker/refine"
    assert b'name="image"; filename="visible_fish_refined.png"' in request.data
    assert b'name="params"' in request.data
    assert b'"mode":"fish_preserve_refine_qwen_v1"' in request.data
    assert b'"source_run_id":"FCL_123"' in request.data
    assert b'"steps":20' in request.data
    assert b'"seed":12345' in request.data
    assert b"SAM_VISIBLE" in request.data
    assert result["refine_result_uri"] == "http://worker/output/refined.png"
    assert result["final_asset_uri"] == "http://worker/output/final.png"
    assert result["elapsed_ms"] == 1234.5
    assert result["worker_protocol"]["input_source"] == "visible_fish_refined"
    assert QWEN_MODE == "fish_preserve_refine_qwen_v1"
