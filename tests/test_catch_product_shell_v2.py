"""Offline tests: import only page registry and cookie guard, never app.entry/cloud clients."""
from pathlib import Path
import re

from fastapi import FastAPI
import asyncio
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request
from jinja2 import Environment, FileSystemLoader
import pytest

from app.secure import install_access_guard, _safe_return_path
from app.platform.routes.pages import PLATFORM_PAGES

ROOT = Path(__file__).resolve().parents[1]


def test_shell_scope_and_assets():
    env=Environment(loader=FileSystemLoader(ROOT/'app/templates'))
    page=next(p for p in PLATFORM_PAGES if p.path=='/platform/model/catch-recognition')
    html=env.get_template(page.template).render(page=page,page_title=page.title,platform_pages=PLATFORM_PAGES)
    assert 'data-component="ProductShell"' in html
    assert '<aside' not in html
    assert 'class="platform-sidebar' not in html
    for component in ['ProductHeader','ProductContent','StageNavigation','UploadStage','ProcessingStage','ResultStage','AdvancedDiagnostics','ProcessJourney','ProductFooter']:
        assert f'data-component="{component}"' in html
    assert html.count('data-journey-step=') == 5
    assert html.count('data-optical-layer=') == 7
    ids=re.findall(r'\bid="([^"]+)"',html)
    assert len(ids)==len(set(ids))
    for asset in re.findall(r'(?:src|href)="(/static/[^"?]+)',html):
        assert (ROOT/'app'/asset.lstrip('/')).is_file(),asset
    ordinary=env.get_template('platform/base.html').render(page_title='模型仓库',platform_pages=PLATFORM_PAGES,page=page,request=Request({'type':'http','path':'/platform/model/registry','headers':[],'query_string':b''}))
    assert 'class="platform-sidebar' in ordinary
    assert 'data-component="ProductShell"' not in ordinary
    assert 'catch_recognition.css' not in ordinary


@pytest.mark.parametrize('bad',['https://example.com','//example.com','/\\evil','/%2f%2fevil','/\nevil','/login','/logout'])
def test_login_return_rejects_external_or_ambiguous_path(bad):
    assert _safe_return_path(bad)=='/'


def test_console_auth_return_cookie_refresh_and_api_protection(monkeypatch):
    monkeypatch.setenv('CONSOLE_ACCESS_KEY','offline-test-password')
    monkeypatch.setenv('CONSOLE_COOKIE_SECURE','0')
    app=FastAPI()
    install_access_guard(app)
    route='/platform/model/catch-recognition'
    app.add_api_route(route,lambda:{'shell':'product'})
    app.add_api_route('/api/product/catch-recognition/infer',lambda:{'ok':True},methods=['POST'])
    async def check():
        async with AsyncClient(transport=ASGITransport(app=app),base_url='http://testserver') as client:
            redirect=await client.get(route,follow_redirects=False)
            assert redirect.status_code==303
            assert 'next=%2Fplatform%2Fmodel%2Fcatch-recognition' in redirect.headers['location']
            assert (await client.post('/api/product/catch-recognition/infer')).status_code==401
            login=await client.get(redirect.headers['location'])
            assert "name='next' value='/platform/model/catch-recognition'" in login.text
            response=await client.post('/login',data={'access_key':'offline-test-password','next':route},follow_redirects=False)
            assert response.status_code==303 and response.headers['location']==route
            assert 'HttpOnly' in response.headers['set-cookie']
            assert (await client.get(route)).status_code==200
            assert (await client.get(route)).status_code==200
            assert (await client.post('/api/product/catch-recognition/infer')).status_code==200
            assert (await client.post('/login',data={'access_key':'offline-test-password','next':'//example.com'},follow_redirects=False)).headers['location']=='/'
    asyncio.run(check())
