#!/usr/bin/env python3
"""Actual Chromium rendering of the local-photo SWOS review surface; no inference."""
from __future__ import annotations
import json, os, socket, threading, time
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from playwright.sync_api import sync_playwright
import uvicorn
import sys
from PIL import Image, ImageChops, ImageStat
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from app.platform.routes.pages import templates, PlatformPage, PLATFORM_PAGES
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
                errors=[];requests=[];page.on('pageerror',lambda error:errors.append(str(error)))
                page.on('request',lambda request:requests.append(request.url) if request.method=='POST' else None)
                page.goto(f'http://127.0.0.1:{port}/swos-preview',wait_until='networkidle')
                page.locator('#swosPhoto').set_input_files(source)
                page.locator('#swosEmpty').wait_for(state='hidden')
                records=[]
                for id,at in [('P0',0),('P1',350),('P2',950),('P3',1200),('P4',1550),('P5',2050),('P6',2600),('P7',2800),('R0',2800),('R0.5',2900),('R1',3000),('R2',3180),('R3',3350),('R4',3480),('R5',3650)]:
                    page.get_by_role('button',name=id,exact=True).click()
                    trace=json.loads(page.locator('#swosInfo').inner_text())
                    assert trace['elapsed_ms']==at and trace['frame']==id and trace['mask_trust']=='TEST_FIXTURE'
                    assert trace['fixture_only'] is True
                    page.locator('#swosCanvas').screenshot(path=str(OUT/f'{id.replace(".","_")}-{at}.png'))
                    records.append({'key':id,'at':at,'photo_rect':trace['photo_rect'],'outer_rect':trace['outer_rect']})
                assert records[7]['photo_rect']==records[8]['photo_rect']
                mid=records[9]['photo_rect'];assert records[8]['photo_rect']['w']>mid['w']>records[10]['photo_rect']['w']
                page.locator('#swosSeek').evaluate('(n)=>{n.value=1200;n.dispatchEvent(new Event("input",{bubbles:true}))}')
                page.locator('#swosWater').uncheck()
                page.locator('#swosCanvas').screenshot(path=str(OUT/'P3-original-comparison.png'))
                page.locator('#swosWater').check()
                with Image.open(OUT/'P3-1200.png') as lit, Image.open(OUT/'P3-original-comparison.png') as original:
                    rect=records[3]['photo_rect']
                    crop=(int(rect['x']),int(rect['y']),int(rect['x']+rect['w']),int(rect['y']+rect['h']))
                    difference=ImageChops.difference(lit.crop(crop).convert('RGB'),original.crop(crop).convert('RGB'))
                    ambient_pixel_delta=sum(ImageStat.Stat(difference).mean)/3
                    assert ambient_pixel_delta>2.5, f'ambient optical field imperceptible: {ambient_pixel_delta:.2f}'
                assert not page.locator('#swosFish').is_enabled()
                assert page.locator('#swosInfer').is_enabled()
                assert not requests, f'choosing a local photo unexpectedly sent a request: {requests}'
                for w,h in [(1440,1000),(1280,800),(1024,768),(768,1024),(480,800),(430,932),(390,844),(360,640),(320,480)]:
                    page.locator('#swosWidth').fill(str(w));page.locator('#swosHeight').fill(str(h));page.locator('#swosHeight').dispatch_event('change')
                    page.locator('#swosSeek').evaluate('(n)=>{n.value=2900;n.dispatchEvent(new Event("input",{bubbles:true}))}')
                    trace=json.loads(page.locator('#swosInfo').inner_text());assert trace['viewport_px']==[w,h]
                    assert trace['photo_rect']['w']>0 and trace['photo_rect']['h']>0
                    page.locator('#swosCanvas').screenshot(path=str(OUT/f'viewport-{w}x{h}.png'))
                page.locator('#swosWidth').fill('1440');page.locator('#swosHeight').fill('1000');page.locator('#swosHeight').dispatch_event('change')
                page.get_by_role('button',name='P4',exact=True).click()
                page.locator('#swosCanvas').screenshot(path=str(OUT/'P4-repeat-a.png'))
                page.get_by_role('button',name='P4',exact=True).click()
                page.locator('#swosCanvas').screenshot(path=str(OUT/'P4-repeat-b.png'))
                with Image.open(OUT/'P4-repeat-a.png') as first, Image.open(OUT/'P4-repeat-b.png') as second:
                    repeat_delta=sum(ImageStat.Stat(ImageChops.difference(first.convert('RGB'),second.convert('RGB'))).mean)/3
                    assert repeat_delta==0, f'same-time fixture frame is not deterministic: {repeat_delta:.4f}'
                page.locator('#swosRate').select_option('1')
                with page.expect_download(timeout=15000) as download_info:
                    page.locator('#swosRecord').click()
                recording=download_info.value
                recording_path=OUT/'fixture-full-playback-3650ms.webm'
                recording.save_as(str(recording_path))
                assert recording_path.is_file() and recording_path.stat().st_size>10000
                assert page.locator('#swosTime').inner_text()=='R5 · 3650ms'
                assert not errors,errors
                (OUT/'trace.json').write_text(json.dumps({'source':'verified public fixture; visual test only','frames':records,'ambient_pixel_delta_P3':ambient_pixel_delta,'repeat_frame_delta_P4':repeat_delta,'full_playback':{'nominal_ms':3650,'final_state':page.locator('#swosTime').inner_text(),'recording':'fixture-full-playback-3650ms.webm','bytes':recording_path.stat().st_size},'mask_trust':'TEST_FIXTURE','real_inference_requested':False,'browser':browser.version,'errors':errors},ensure_ascii=False,indent=2))
                page.close()
            finally:browser.close()
    finally:server.should_exit=True;thread.join(timeout=5)
if __name__=='__main__':main()
