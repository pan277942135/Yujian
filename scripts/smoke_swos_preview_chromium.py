#!/usr/bin/env python3
"""Actual Chromium rendering of the local-photo SWOS review surface; no inference."""
from __future__ import annotations
import json, os, socket, threading, time
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from playwright.sync_api import sync_playwright
import uvicorn
from app.platform.routes.pages import templates, PlatformPage, PLATFORM_PAGES

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'test-results'/'swos-preview-chromium'
PAGE=PlatformPage('/swos-preview','platform/swos_preview.html','SWOS 动效验收','模型工厂','本地视觉审查')
app=FastAPI(docs_url=None,openapi_url=None)
app.mount('/static',StaticFiles(directory=str(ROOT/'app'/'static')),name='static')
@app.get('/swos-preview')
def preview(request:Request):
    return templates.TemplateResponse(request=request,name=PAGE.template,context={'page':PAGE,'page_title':PAGE.title,'platform_pages':PLATFORM_PAGES})

def main():
    source=os.environ.get('DET_FISH_PORTRAIT_PHOTO')
    assert source and Path(source).is_file(), 'verified public photo fixture required'
    OUT.mkdir(parents=True,exist_ok=True)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
    server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=port,log_level='warning',lifespan='off'))
    thread=threading.Thread(target=server.run,daemon=True);thread.start()
    try:
        for _ in range(150):
            if server.started:break
            time.sleep(.1)
        assert server.started
        with sync_playwright() as pw:
            browser=pw.chromium.launch()
            try:
                page=browser.new_page(viewport={'width':1440,'height':1000})
                errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
                page.goto(f'http://127.0.0.1:{port}/swos-preview',wait_until='networkidle')
                page.locator('#swosPhoto').set_input_files(source)
                page.locator('#swosEmpty').wait_for(state='hidden')
                records=[]
                for id,at in [('P0',0),('P1',350),('P2',950),('P3',1200),('P4',1550),('P5',2050),('P6',2600),('P7',2800),('R0',2800),('R0.5',2900),('R1',3000),('R2',3180),('R3',3350),('R4',3480),('R5',3650)]:
                    page.locator('#swosSeek').evaluate('(n,v)=>{n.value=v;n.dispatchEvent(new Event("input",{bubbles:true}))}',str(at))
                    trace=json.loads(page.locator('#swosInfo').inner_text())
                    assert trace['elapsed_ms']==at and trace['mask_trust']=='NONE'
                    page.locator('#swosCanvas').screenshot(path=str(OUT/f'{id.replace(".","_")}-{at}.png'))
                    records.append({'key':id,'at':at,'photo_rect':trace['photo_rect'],'outer_rect':trace['outer_rect']})
                assert records[7]['photo_rect']==records[8]['photo_rect']
                mid=records[9]['photo_rect'];assert records[8]['photo_rect']['w']>mid['w']>records[10]['photo_rect']['w']
                for w,h in [(1440,1000),(1280,800),(1024,768),(768,1024),(480,800),(430,932),(390,844),(360,640),(320,480)]:
                    page.locator('#swosWidth').fill(str(w));page.locator('#swosHeight').fill(str(h));page.locator('#swosHeight').dispatch_event('change')
                    page.locator('#swosSeek').evaluate('(n)=>{n.value=2900;n.dispatchEvent(new Event("input",{bubbles:true}))}')
                    trace=json.loads(page.locator('#swosInfo').inner_text());assert trace['viewport_px']==[w,h]
                    assert trace['photo_rect']['w']>0 and trace['photo_rect']['h']>0
                    page.locator('#swosCanvas').screenshot(path=str(OUT/f'viewport-{w}x{h}.png'))
                assert not errors,errors
                (OUT/'trace.json').write_text(json.dumps({'source':'verified public fixture; visual test only','frames':records,'browser':browser.version,'errors':errors},ensure_ascii=False,indent=2))
                page.close()
            finally:browser.close()
    finally:server.should_exit=True;thread.join(timeout=5)
if __name__=='__main__':main()
