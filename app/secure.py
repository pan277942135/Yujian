import hashlib
import os
import secrets

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from app.main import app

COOKIE_NAME = "yujian_console"


def _configured_key() -> str:
    return os.getenv("CONSOLE_ACCESS_KEY", "").strip()


def _cookie_value(key: str) -> str:
    return hashlib.sha256(("yujian-console:" + key).encode("utf-8")).hexdigest()


@app.middleware("http")
async def console_access_guard(request: Request, call_next):
    key = _configured_key()
    if not key or request.url.path in {"/health", "/login"}:
        return await call_next(request)

    expected = _cookie_value(key)
    actual = request.cookies.get(COOKIE_NAME, "")
    if actual and secrets.compare_digest(actual, expected):
        return await call_next(request)

    if request.url.path.startswith("/api/") or request.url.path.startswith("/media/"):
        return JSONResponse({"detail": "console authentication required"}, status_code=401)
    return RedirectResponse("/login", status_code=303)


@app.get("/login", response_class=HTMLResponse)
async def login_page():
    if not _configured_key():
        return RedirectResponse("/", status_code=303)
    return HTMLResponse(
        """<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>YuJian Console Login</title><style>body{font-family:system-ui;margin:0;background:#f6f7f8;display:grid;place-items:center;min-height:100vh;color:#171717}.box{width:min(380px,calc(100vw - 48px));background:white;border:1px solid #e5e7eb;border-radius:16px;padding:24px}input,button{box-sizing:border-box;width:100%;padding:11px;border-radius:9px;border:1px solid #d1d5db;font:inherit;margin-top:10px}button{background:#111827;color:white;font-weight:700;cursor:pointer}.muted{color:#6b7280;font-size:13px}</style></head><body><form class='box' method='post' action='/login'><h2>YuJian Model Factory</h2><div class='muted'>输入 Console 访问口令</div><input type='password' name='access_key' autofocus required><button type='submit'>进入 Console</button></form></body></html>"""
    )


@app.post("/login")
async def login(request: Request):
    key = _configured_key()
    if not key:
        return RedirectResponse("/", status_code=303)
    form = await request.form()
    supplied = str(form.get("access_key", ""))
    if not secrets.compare_digest(supplied, key):
        return HTMLResponse("访问口令错误。<a href='/login'>返回</a>", status_code=401)
    response = RedirectResponse("/", status_code=303)
    response.set_cookie(
        COOKIE_NAME,
        _cookie_value(key),
        httponly=True,
        secure=True,
        samesite="strict",
        max_age=60 * 60 * 12,
    )
    return response


@app.get("/logout")
async def logout():
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(COOKIE_NAME)
    return response
