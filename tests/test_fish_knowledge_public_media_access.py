from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.responses import Response

from app.fish_knowledge.api import get_knowledge_media
from app.secure import install_access_guard


def _request(app: FastAPI, method: str, path: str) -> tuple[int, bytes]:
    async def run() -> tuple[int, bytes]:
        delivered = False
        sent: list[dict] = []

        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": b"", "more_body": False}
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        await app(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": method,
                "scheme": "http",
                "path": path,
                "raw_path": path.encode("utf-8"),
                "query_string": b"",
                "root_path": "",
                "headers": [],
                "client": ("test", 50000),
                "server": ("testserver", 80),
            },
            receive,
            send,
        )
        start = next(item for item in sent if item["type"] == "http.response.start")
        body = b"".join(item.get("body", b"") for item in sent if item["type"] == "http.response.body")
        return start["status"], body

    return asyncio.run(run())


def _guarded_app() -> FastAPI:
    app = FastAPI()
    install_access_guard(app)

    @app.get("/api/v1/fish/knowledge-media/{asset_key}")
    def media(asset_key: str):
        assert asset_key
        return Response(content=b"webp-bytes", media_type="image/webp")

    @app.post("/api/admin/fish/assets/upload")
    def admin_upload():
        return {"success": True}

    return app


def test_anonymous_fish_knowledge_media_read_is_public(monkeypatch):
    monkeypatch.setenv("CONSOLE_ACCESS_KEY", "console-secret")
    status, body = _request(_guarded_app(), "GET", "/api/v1/fish/knowledge-media/test.webp")
    assert status == 200
    assert body == b"webp-bytes"


def test_non_get_fish_knowledge_media_is_still_protected(monkeypatch):
    monkeypatch.setenv("CONSOLE_ACCESS_KEY", "console-secret")
    status, _ = _request(_guarded_app(), "POST", "/api/v1/fish/knowledge-media/test.webp")
    assert status == 401


def test_admin_asset_upload_is_still_protected(monkeypatch):
    monkeypatch.setenv("CONSOLE_ACCESS_KEY", "console-secret")
    status, body = _request(_guarded_app(), "POST", "/api/admin/fish/assets/upload")
    assert status == 401
    assert "需要先登录模型工厂控制台" in body.decode("utf-8")


@pytest.mark.parametrize("asset_key", ["../../secret", "cover.gif", "cover.webp/../secret"])
def test_knowledge_media_rejects_unsafe_or_unsupported_asset_keys(asset_key):
    with pytest.raises(Exception) as error:
        get_knowledge_media("grass_carp", "cover", asset_key, None)
    assert getattr(error.value, "status_code", None) == 404
