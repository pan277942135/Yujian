import hashlib
import os
import secrets

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

COOKIE_NAME = "yujian_console"
PUBLIC_PATHS = {"/health", "/health/deploy", "/health/detector", "/login"}
PUBLIC_API_PATH_PREFIXES = (
    # These routes authenticate App users with their own Bearer token.  They
    # must not be intercepted by the Model Factory console-cookie middleware.
    "/api/v1/auth",
    "/api/v1/catches",
)
PUBLIC_GET_PATH_PREFIXES = (
    "/api/v1/fish/species",
    "/api/v1/fish/gallery",
    # Fish Knowledge cover/card assets are public read-only resources for App clients.
    "/api/v1/fish/knowledge-media",
)
FEEDBACK_INGEST_PATHS = {
    "/api/feedback",
    "/api/feedback/ingest",
    "/api/v1/inference/upload",
}


def _configured_key() -> str:
    return os.getenv("CONSOLE_ACCESS_KEY", "").strip()


def _feedback_ingest_key() -> str:
    return os.getenv("FEEDBACK_INGEST_KEY", "").strip()


def _cookie_value(key: str) -> str:
    return hashlib.sha256(("yujian-console:" + key).encode("utf-8")).hexdigest()


def install_access_guard(app: FastAPI) -> None:
    @app.middleware("http")
    async def console_access_guard(request: Request, call_next):
        key = _configured_key()
        public_fish_read = request.method == "GET" and any(
            request.url.path == prefix or request.url.path.startswith(prefix + "/")
            for prefix in PUBLIC_GET_PATH_PREFIXES
        )
        app_api_request = any(
            request.url.path == prefix or request.url.path.startswith(prefix + "/")
            for prefix in PUBLIC_API_PATH_PREFIXES
        )
        if not key or request.url.path in PUBLIC_PATHS or public_fish_read or app_api_request:
            return await call_next(request)

        ingest_key = _feedback_ingest_key()
        if request.method == "POST" and request.url.path in FEEDBACK_INGEST_PATHS and ingest_key:
            supplied = request.headers.get("X-YuJian-Ingest-Key", "")
            if supplied and secrets.compare_digest(supplied, ingest_key):
                return await call_next(request)

        expected = _cookie_value(key)
        actual = request.cookies.get(COOKIE_NAME, "")
        if actual and secrets.compare_digest(actual, expected):
            return await call_next(request)

        if request.url.path.startswith("/api/") or request.url.path.startswith("/media/"):
            return JSONResponse({"detail": "需要先登录模型工厂控制台"}, status_code=401)
        return RedirectResponse("/login", status_code=303)

    @app.get("/login", response_class=HTMLResponse)
    async def login_page():
        if not _configured_key():
            return RedirectResponse("/", status_code=303)
        return HTMLResponse(
            """<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>渔见 AI 模型工厂登录</title><style>body{font-family:system-ui;margin:0;background:#f6f7f8;display:grid;place-items:center;min-height:100vh;color:#171717}.box{width:min(380px,calc(100vw - 48px));background:white;border:1px solid #e5e7eb;border-radius:16px;padding:24px}input,button{box-sizing:border-box;width:100%;padding:11px;border-radius:9px;border:1px solid #d1d5db;font:inherit;margin-top:10px}button{background:#111827;color:white;font-weight:700;cursor:pointer}.muted{color:#6b7280;font-size:13px}</style></head><body><form class='box' method='post' action='/login'><h2>渔见 AI 模型工厂</h2><div class='muted'>请输入访问口令</div><input type='password' name='access_key' autofocus required><button type='submit'>进入模型工厂</button></form></body></html>"""
        )

    @app.post("/login")
    async def login(request: Request):
        key = _configured_key()
        if not key:
            return RedirectResponse("/", status_code=303)
        form = await request.form()
        supplied = str(form.get("access_key", ""))
        if not secrets.compare_digest(supplied, key):
            return HTMLResponse("访问口令错误。<a href='/login'>返回重新输入</a>", status_code=401)
        response = RedirectResponse("/", status_code=303)
        response.set_cookie(
            COOKIE_NAME,
            _cookie_value(key),
            httponly=True,
            secure=os.getenv("CONSOLE_COOKIE_SECURE", "1") != "0",
            samesite="strict",
            max_age=60 * 60 * 12,
        )
        return response

    @app.get("/logout")
    async def logout():
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie(COOKIE_NAME)
        return response
