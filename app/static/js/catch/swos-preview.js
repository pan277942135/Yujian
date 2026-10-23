(() => {
  "use strict";
  const keys=[['P0',0,0,0],['P1',350,10,0],['P2',950,45,15],['P3',1200,75,45],['P4',1550,60,100],['P5',2050,35,100],['P6',2600,10,100],['P7',2800,0,100],['R0',2800,0,100],['R0.5',2900,0,100],['R1',3000,0,100],['R2',3180,0,30],['R3',3350,0,0],['R4',3480,0,0],['R5',3650,0,0]];
  const $=id=>document.getElementById(id), canvas=$('swosCanvas'), ctx=canvas.getContext('2d');
  const core=window.YujianCatch;
  const nodes={photo:$('swosPhoto'),name:$('swosName'),infer:$('swosInfer'),status:$('swosRunStatus'),play:$('swosPlay'),seek:$('swosSeek'),rate:$('swosRate'),time:$('swosTime'),info:$('swosInfo'),water:$('swosWater'),region:$('swosRegion'),fish:$('swosFish'),compare:$('swosCompare'),reduce:$('swosReduce'),goldCore:$('swosCore'),softGlow:$('swosSoft'),halo:$('swosHalo'),transfer:$('swosTransfer'),textMask:$('swosTextMask')};
  let file=null,image=null,photoSha='',time=0,currentFrame='P0',playing=false,anchor=0,anchorTime=0,raf=0,recorder=null,recordChunks=[];
  let mode='FIXTURE',requestPending=false,liveResult=null,livePhotoSession=null,liveController=null,maskImage=null,maskTexture=null,renderFrames=0,frameTimeP95Ms=0,frameTimes=[];
  const clamp=(n,a,b)=>Math.max(a,Math.min(b,n));
  const lerp=(a,b,p)=>a+(b-a)*p;
  const ease=p=>p*p*(3-2*p);
  const rectLerp=(a,b,p)=>Object.fromEntries(['x','y','w','h'].map(k=>[k,lerp(a[k],b[k],p)]));
  function curve(t,i){let prev=keys[0];for(const next of keys.slice(1)){if(t<=next[1])return lerp(prev[i],next[i],next[1]===prev[1]?1:(t-prev[1])/(next[1]-prev[1]));prev=next;}return prev[i];}
  function keyAt(t){return [...keys].reverse().find(k=>k[1]<=t)||keys[0];}
  function contained(source,box){const s=Math.min(box.w/source.width,box.h/source.height),w=source.width*s,h=source.height*s;return{x:box.x+(box.w-w)/2,y:box.y+(box.h-h)/2,w,h};}
  function geometry(t,w,h){
    const processing={x:0,y:0,w,h},hero={x:w*19/360,y:h*53/640,w:w*322/360,h:h*210/640};
    const p=clamp((t-2800)/200,0,1),outer=rectLerp(processing,hero,p),first=contained(image,processing),last=contained(image,hero),photo=rectLerp(first,last,p);
    return{outer,photo,sourceMatrix:[photo.w/image.width,0,0,photo.h/image.height,photo.x,photo.y]};
  }
  function waterField(g,t,strength,photo){
    if(strength<=0)return;
    const v=Math.pow(strength/75,.82),w=photo.w,h=photo.h,spread=ease(clamp((t-350)/850,0,1));
    g.save();g.beginPath();g.rect(photo.x,photo.y,w,h);g.clip();g.translate(photo.x,photo.y);g.globalCompositeOperation='screen';
    // Broad, uneven light patches give the field depth before its fine caustics appear.
    for(let i=0;i<9;i++){
      const phase=i*2.399963,drift=t*.00011*(i%2?1:-.75),x=w*(.1+.8*((i*.618034+.13)%1));
      const y=h*(.83-spread*(.13+.63*((i*.414214+.21)%1))),rx=w*(.12+.16*((i*.754878)%1)),ry=h*(.06+.075*((i*.569841)%1));
      g.save();g.translate(x+Math.sin(phase+drift)*w*.014,y);g.rotate(Math.sin(phase*.7+drift)*.12);g.scale(1,ry/rx);
      const halo=g.createRadialGradient(0,0,0,0,0,rx);halo.addColorStop(0,`rgba(249,239,196,${.35*v})`);halo.addColorStop(.42,`rgba(230,241,213,${.20*v})`);halo.addColorStop(1,'rgba(232,240,207,0)');
      g.fillStyle=halo;g.beginPath();g.arc(0,0,rx,0,Math.PI*2);g.fill();g.restore();
    }
    // Broken, short refracted caustic patches at three scales; their drift is deterministic.
    for(let band=0;band<3;band++){
      const count=[12,20,27][band];
      for(let j=0;j<count;j++){
        const seed=j*2.399963+band*13.17,rise=(j*.618034+band*.19)%1,y=h*(.86-spread*(.12+.76*rise));
        const cx=w*(.08+.84*((j*.754878+band*.32)%1)),span=w*[.18,.105,.052][band]*(.65+.55*((j*.414214)%1));
        const travel=t*.00032*(band+1),bend=Math.sin(seed+travel)*h*.013,fade=(.55+.45*Math.sin(seed*2+travel*.7)**2)*v;
        const gradient=g.createLinearGradient(cx-span,y,cx+span,y);gradient.addColorStop(0,'rgba(239,243,210,0)');
        gradient.addColorStop(.38,`rgba(252,247,206,${[.40,.31,.2][band]*fade})`);gradient.addColorStop(.64,`rgba(222,245,224,${[.28,.22,.16][band]*fade})`);gradient.addColorStop(1,'rgba(222,245,224,0)');
        g.strokeStyle=gradient;g.lineCap='round';g.lineWidth=[13,6,3][band]*w/360;g.shadowColor=`rgba(239,235,190,${.38*fade})`;g.shadowBlur=[16,9,4][band]*w/360;
        g.beginPath();g.moveTo(cx-span,y+bend);g.bezierCurveTo(cx-span*.4,y-h*.018*Math.sin(seed+travel),cx+span*.28,y+h*.025*Math.sin(seed*1.7-travel),cx+span,y-bend*.6);g.stroke();
      }
    }
    g.restore();
  }
  function drawBBoxSoftRegion(g,photo,bbox,strength){
    if(!bbox||strength<=0)return;
    const x1=photo.x+bbox.x1*photo.w,y1=photo.y+bbox.y1*photo.h,x2=photo.x+bbox.x2*photo.w,y2=photo.y+bbox.y2*photo.h;
    const cx=(x1+x2)/2,cy=(y1+y2)/2,rx=Math.max(12,(x2-x1)*.72),ry=Math.max(12,(y2-y1)*.72),v=clamp(strength/100,0,1);
    g.save();g.globalCompositeOperation='screen';g.translate(cx,cy);g.scale(1,ry/rx);
    const glow=g.createRadialGradient(0,0,0,0,0,rx);glow.addColorStop(0,`rgba(247,207,117,${.22*v})`);glow.addColorStop(.48,`rgba(219,168,72,${.12*v})`);glow.addColorStop(1,'rgba(219,168,72,0)');
    g.fillStyle=glow;g.beginPath();g.arc(0,0,rx,0,Math.PI*2);g.fill();g.restore();
  }
  function drawMask(g,photo,strength){
    if(!maskTexture||!livePhotoSession?.mask.renderAllowed||!nodes.fish.checked||strength<=0)return;
    const v=clamp(strength/100,0,1);g.save();g.globalCompositeOperation='screen';
    if(nodes.halo.checked){g.filter=`blur(${Math.max(5,canvas.width*.025)}px)`;g.globalAlpha=.14*v;g.drawImage(maskTexture,photo.x,photo.y,photo.w,photo.h);}
    if(nodes.softGlow.checked){g.filter=`blur(${Math.max(2,canvas.width*.01)}px)`;g.globalAlpha=.24*v;g.drawImage(maskTexture,photo.x,photo.y,photo.w,photo.h);}
    if(nodes.goldCore.checked){g.filter='none';g.globalAlpha=.22*v;g.drawImage(maskTexture,photo.x,photo.y,photo.w,photo.h);}
    g.restore();
  }
  function drawTransfer(g,photo,geom,titleY,t){
    const mask=livePhotoSession?.mask;
    if(mode!=='REAL_HIGH'||mask?.status!=='REVIEW_PROXY'||!mask.renderAllowed||!nodes.transfer.checked||t<3000||t>=3180)return;
    const centroid=mask.centroid;if(!centroid)return;
    const from={x:photo.x+centroid.x*photo.w,y:photo.y+centroid.y*photo.h},to={x:canvas.width/2,y:titleY};
    const progress=clamp((t-3000)/180,0,1),control={x:(from.x+to.x)/2+canvas.width*.08,y:Math.min(from.y,to.y)-canvas.height*.08};
    g.save();g.globalCompositeOperation='screen';g.globalAlpha=.9*(1-progress*.25);
    const path=new Path2D();path.moveTo(from.x,from.y);path.quadraticCurveTo(control.x,control.y,to.x,to.y);
    const stroke=g.createLinearGradient(from.x,from.y,to.x,to.y);stroke.addColorStop(0,'rgba(240,190,82,0)');stroke.addColorStop(.32,'rgba(255,221,138,.86)');stroke.addColorStop(.78,'rgba(255,242,196,.84)');stroke.addColorStop(1,'rgba(244,202,115,.15)');
    g.strokeStyle=stroke;g.lineWidth=Math.max(2,canvas.width*.006);g.lineCap='round';g.shadowColor='rgba(255,210,112,.85)';g.shadowBlur=canvas.width*.035;g.stroke(path);g.restore();
  }
  function makeGlyphMask(title,fontSize,y){
    const glyph=document.createElement('canvas');glyph.width=canvas.width;glyph.height=canvas.height;const text=glyph.getContext('2d');
    text.font=`700 ${fontSize}px system-ui, sans-serif`;text.textAlign='center';text.textBaseline='middle';text.fillStyle='#fff';text.fillText(title,canvas.width/2,y,canvas.width*.8);return glyph;
  }
  function renderTitle(g,geom,t){
    const real=mode==='REAL_HIGH',allowedFixture=mode==='FIXTURE',title=real?(liveResult?.classifier?.top1?.species||''):(allowedFixture?(nodes.name.value.trim()||'鱼种名称'):'');
    if(!title)return;
    const y=clamp(geom.outer.y+geom.outer.h+Math.max(28,canvas.height*.065),0,canvas.height-35),fontSize=clamp(canvas.width*.065,17,38),glyph=makeGlyphMask(title,fontSize,y);
    drawTransfer(g,geom.photo,geom,y,t);
    if(nodes.reduce.checked||t>=3480){
      g.save();if(real&&t<3650){g.shadowColor='rgba(220,171,76,.3)';g.shadowBlur=nodes.reduce.checked?0:canvas.width*.012;}
      g.font=`700 ${fontSize}px system-ui, sans-serif`;g.textAlign='center';g.textBaseline='middle';g.fillStyle='#294c43';g.fillText(title,canvas.width/2,y,canvas.width*.8);g.restore();return;
    }
    if(t>=3350&&t<3480){
      // R3 uses the real text glyph mask but reveals only disconnected, unreadable strokes.
      g.save();g.beginPath();g.rect(canvas.width*.22,y-fontSize*.48,canvas.width*.055,fontSize*.82);g.rect(canvas.width*.47,y-fontSize*.38,canvas.width*.035,fontSize*.62);g.rect(canvas.width*.71,y-fontSize*.5,canvas.width*.045,fontSize*.72);g.clip();
      g.globalAlpha=nodes.textMask.checked?.8:.18;g.globalCompositeOperation=nodes.textMask.checked?'source-over':'source-over';g.drawImage(glyph,0,0);g.restore();
    }
  }
  function draw(t){
    const w=canvas.width,h=canvas.height;ctx.clearRect(0,0,w,h);ctx.fillStyle='#d9e8e5';ctx.fillRect(0,0,w,h);
    if(!image){nodes.info.textContent='无照片 · 尚未产生检测、Mask 或识别结果';return;}
    const geom=geometry(t,w,h),o=geom.outer,p=geom.photo,water=nodes.reduce.checked?0:curve(t,2),gold=curve(t,3);
    ctx.save();ctx.beginPath();ctx.rect(o.x,o.y,o.w,o.h);ctx.clip();
    const coverScale=Math.max(o.w/image.width,o.h/image.height);ctx.globalAlpha=.17;ctx.filter='blur(18px)';
    ctx.drawImage(image,o.x+(o.w-image.width*coverScale)/2,o.y+(o.h-image.height*coverScale)/2,image.width*coverScale,image.height*coverScale);
    ctx.filter='none';ctx.globalAlpha=1;ctx.drawImage(image,p.x,p.y,p.w,p.h);
    if(!nodes.compare.checked){
      if(nodes.water.checked&&water>0)waterField(ctx,t,water,p);
      if(nodes.region.checked&&livePhotoSession?.bbox)drawBBoxSoftRegion(ctx,p,livePhotoSession.bbox,gold);
      if(nodes.fish.checked&&nodes.goldCore.checked||nodes.fish.checked&&nodes.softGlow.checked||nodes.fish.checked&&nodes.halo.checked)drawMask(ctx,p,gold);
    }
    ctx.restore();renderTitle(ctx,geom,t);
    const trust=mode==='FIXTURE'?'TEST_FIXTURE':(livePhotoSession?.mask?.status||'NONE');
    nodes.info.textContent=JSON.stringify({mode,photo_sha256:photoSha,source_px:[image.width,image.height],viewport_px:[w,h],dpr:window.devicePixelRatio||1,browser:navigator.userAgent,frame:currentFrame,elapsed_ms:Math.round(t),photo_rect:p,outer_rect:o,source_matrix:geom.sourceMatrix,recognition_state:liveResult?.result_state||null,detector_model:liveResult?.detector?.model_version||null,detector_onnx_sha256:liveResult?.detector?.onnx_sha256||null,classifier_model:liveResult?.classifier?.model_version||null,mask_trust:trust,mask_sha256:livePhotoSession?.mask?.sha256||null,mask_quality:liveResult?.segmentation?.quality||null,sam_checkpoint_sha256:liveResult?.segmentation?.checkpoint_sha256||null,renderer:'Canvas 2D',render_frame_count:renderFrames,frame_time_p95_ms:frameTimeP95Ms,reduced_motion:nodes.reduce.checked,fixture_only:mode==='FIXTURE'},null,2);
  }
  function maxPlayableTime(){return mode==='REAL_PENDING'||mode==='REAL_NO_SUCCESS'||mode==='REAL_FAILED'?2800:3650;}
  function setTime(value,requestedFrame=null){
    const limit=maxPlayableTime(),requested=clamp(Number(value)||0,0,3650);time=Math.min(requested,limit);
    currentFrame=requestedFrame&&keys.some(k=>k[0]===requestedFrame&&k[1]===time)?requestedFrame:keyAt(time)[0];
    nodes.seek.value=String(Math.round(time));nodes.time.textContent=`${currentFrame} · ${Math.round(time)}ms`;
    document.querySelectorAll('#swosKeys button').forEach(b=>b.setAttribute('aria-current',String(b.dataset.frame===currentFrame)));
    if(requested>limit)nodes.status.textContent='R0–R5 需要真实 RESULT_HIGH；当前只显示到 P7。';
    draw(time);
  }
  function pause(){playing=false;cancelAnimationFrame(raf);raf=0;nodes.play.textContent='播放';}
  function tick(now){
    if(!playing)return;
    const raw=anchorTime+(now-anchor)*Number(nodes.rate.value),limit=maxPlayableTime(),next=Math.min(raw,limit);
    setTime(next,raw>=2800?(mode==='REAL_PENDING'||mode==='REAL_NO_SUCCESS'?'P7':null):null);
    if(next<limit||mode==='REAL_PENDING')raf=requestAnimationFrame(tick);
    else{pause();if(recorder?.state==='recording')recorder.stop();}
  }
  function play(){
    if(!image)return;
    if(time>=maxPlayableTime()&&mode!=='REAL_PENDING'&&mode!=='REAL_HIGH'&&mode!=='FIXTURE')return;
    if(time>=3650)setTime(0);
    anchorTime=time;anchor=performance.now();playing=true;nodes.play.textContent='暂停';raf=requestAnimationFrame(tick);
  }
  function resetLiveState(){
    liveController?.abort();liveController=null;requestPending=false;liveResult=null;mode='FIXTURE';maskImage=null;maskTexture=null;
    livePhotoSession?.dispose();livePhotoSession=null;nodes.region.disabled=true;nodes.region.checked=false;nodes.fish.disabled=true;nodes.fish.checked=false;nodes.infer.disabled=!file;
    nodes.name.disabled=false;nodes.status.textContent='Fixture 模式：时间线和中文测试标题不代表真实识别。';
  }
  function rgbaMask(source){
    const maxSide=1024,scale=Math.min(1,maxSide/Math.max(source.naturalWidth,source.naturalHeight)),texture=document.createElement('canvas');
    texture.width=Math.max(1,Math.round(source.naturalWidth*scale));texture.height=Math.max(1,Math.round(source.naturalHeight*scale));
    const maskCtx=texture.getContext('2d',{willReadFrequently:true});maskCtx.drawImage(source,0,0,texture.width,texture.height);
    const imageData=maskCtx.getImageData(0,0,texture.width,texture.height),pixels=imageData.data;
    for(let i=0;i<pixels.length;i+=4){const alpha=pixels[i];pixels[i]=245;pixels[i+1]=192;pixels[i+2]=86;pixels[i+3]=alpha;}
    maskCtx.putImageData(imageData,0,0);return texture;
  }
  async function acceptLiveResult(payload){
    if(payload.input?.source_sha256!==photoSha)throw new Error('返回结果照片 SHA 与当前照片不一致');
    liveResult=payload;livePhotoSession.accept(payload);
    const mask=payload.segmentation?.mask;
    if(livePhotoSession.mask.status==='REVIEW_PROXY'&&livePhotoSession.mask.renderAllowed&&mask?.data_base64){
      maskImage=new Image();maskImage.src=`data:image/png;base64,${mask.data_base64}`;await maskImage.decode();maskTexture=rgbaMask(maskImage);
      nodes.fish.disabled=false;
    }
    if(livePhotoSession.bbox)nodes.region.disabled=false;
    const high=payload.result_state==='RESULT_HIGH'&&/^[\u3400-\u9fff]+$/u.test(String(payload.classifier?.top1?.species||''));
    if(high){
      mode='REAL_HIGH';nodes.name.value=payload.classifier.top1.species;nodes.name.disabled=true;
      const elapsed=playing?anchorTime+(performance.now()-anchor)*Number(nodes.rate.value):time;
      if(elapsed>=2800){setTime(2800,'R0');anchorTime=2800;anchor=performance.now();if(playing)raf=requestAnimationFrame(tick);}
      nodes.status.textContent=`真实 RESULT_HIGH · ${payload.classifier.top1.species} · detector ${payload.detector?.model_version||'—'} · classifier ${payload.classifier?.model_version||'—'} · SAM ${payload.segmentation?.model_version||'未启用'} · Mask ${livePhotoSession.mask.status} · ${photoSha}`;
      nodes.fish.title=livePhotoSession.mask.status==='REVIEW_PROXY'?'仅用于调试目视审查；生产 Mask 未获资格':'未取得可显示的候选 Mask';
    }else{
      mode='REAL_NO_SUCCESS';nodes.name.disabled=true;pause();setTime(Math.min(time,2800),'P7');
      nodes.status.textContent=`真实结果 ${payload.result_state}；没有 RESULT_HIGH，因此没有 R0–R5 成功揭晓。Detector ${payload.detector?.model_version||'—'} · SHA ${photoSha}`;
    }
    draw(time);
  }
  async function runRealRecognition(){
    if(!file||!image||requestPending)return;
    pause();resetLiveState();mode='REAL_PENDING';requestPending=true;nodes.infer.disabled=true;nodes.status.textContent=`真实推理进行中 · 照片 SHA ${photoSha} · 到 P7 后等待真实模型结果，不提前显示成功。`;
    livePhotoSession=new core.PhotoSession(file,image.width,image.height);setTime(0,'P0');anchorTime=0;anchor=performance.now();playing=true;nodes.play.textContent='暂停';raf=requestAnimationFrame(tick);
    liveController=new AbortController();const timeout=setTimeout(()=>liveController?.abort('request-timeout'),120000);
    try{
      const payload=await core.infer('/api/product/catch-recognition/infer',file,liveController.signal,`swos_${photoSha.slice(0,24)}`);
      requestPending=false;await acceptLiveResult(payload);
    }catch(error){
      requestPending=false;mode='REAL_FAILED';pause();setTime(Math.min(time,2800),'P7');
      nodes.status.textContent=`真实推理未完成：${error?.message||error} · 没有生成成功揭晓。`;
    }finally{clearTimeout(timeout);nodes.infer.disabled=!file;}
  }
  nodes.play.onclick=()=>playing?pause():play();
  $('swosRestart').onclick=()=>{pause();setTime(0,'P0');play();};
  nodes.seek.oninput=e=>{pause();setTime(e.target.value);};
  nodes.rate.onchange=()=>{if(playing){anchorTime=time;anchor=performance.now();}};
  const step=d=>{pause();let index=keys.findIndex(k=>k[0]===currentFrame);if(index<0)index=0;index=clamp(index+d,0,keys.length-1);setTime(keys[index][1],keys[index][0]);};
  $('swosPrev').onclick=()=>step(-1);$('swosNext').onclick=()=>step(1);
  keys.forEach(([id,at])=>{const b=document.createElement('button');b.type='button';b.textContent=id;b.title=`${id} · ${at}ms`;b.dataset.at=at;b.dataset.frame=id;b.onclick=()=>{pause();setTime(at,id);};$('swosKeys').append(b);});
  function resize(){canvas.width=clamp(Number($('swosWidth').value)||360,240,2000);canvas.height=clamp(Number($('swosHeight').value)||640,320,2000);draw(time);}
  $('swosViewport').onchange=e=>{if(e.target.value==='free')return;const [w,h]=e.target.value.split(',');$('swosWidth').value=w;$('swosHeight').value=h;resize();};
  for(const id of ['swosWidth','swosHeight'])$(id).onchange=()=>{$('swosViewport').value='free';resize();};
  for(const node of [nodes.water,nodes.region,nodes.fish,nodes.compare,nodes.reduce,nodes.goldCore,nodes.softGlow,nodes.halo,nodes.transfer,nodes.textMask,nodes.name])node.oninput=()=>draw(time);
  nodes.infer.onclick=runRealRecognition;
  nodes.photo.onchange=async()=>{
    pause();resetLiveState();file=nodes.photo.files?.[0]||null;if(!file)return;
    try{
      const buffer=await file.arrayBuffer(),digest=await crypto.subtle.digest('SHA-256',buffer);photoSha=[...new Uint8Array(digest)].map(x=>x.toString(16).padStart(2,'0')).join('');
      const decoded=await createImageBitmap(file,{imageOrientation:'from-image'});image?.close();image=decoded;
      livePhotoSession=new core.PhotoSession(file,image.width,image.height);livePhotoSession.dispose();livePhotoSession=null;
      nodes.infer.disabled=false;$('swosEmpty').hidden=true;nodes.status.textContent=`Fixture 模式 · 照片未上传 · SHA ${photoSha}`;setTime(0,'P0');
    }catch(error){nodes.status.textContent=`无法读取照片：${error.message}`;nodes.infer.disabled=true;}
  };
  $('swosExport').onclick=()=>{if(!image)return;canvas.toBlob(blob=>{if(!blob)return;const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`SWOS_${mode}_${currentFrame.replace('.','_')}_${Math.round(time)}ms.png`;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),3000);});};
  $('swosExportTrace').onclick=()=>{
    const trace={source_sha256:photoSha,source_px:image?[image.width,image.height]:null,viewport_px:[canvas.width,canvas.height],device_pixel_ratio:window.devicePixelRatio||1,browser:navigator.userAgent,mode,recognition_result_state:liveResult?.result_state||null,detector:liveResult?.detector||null,classifier:liveResult?.classifier||null,segmentation:liveResult?.segmentation?{status:liveResult.segmentation.status,geometry_trust_state:liveResult.segmentation.geometry_trust_state,model_version:liveResult.segmentation.model_version,checkpoint_sha256:liveResult.segmentation.checkpoint_sha256,quality:liveResult.segmentation.quality,mask_sha256:liveResult.segmentation.mask?.sha256}:null,elapsed_ms:time,frame:currentFrame,renderer:'Canvas 2D',render_frame_count:renderFrames,frame_time_p95_ms:frameTimeP95Ms,layer_toggles:{water:nodes.water.checked,bbox_soft_region:nodes.region.checked,review_proxy:nodes.fish.checked,core:nodes.goldCore.checked,soft_glow:nodes.softGlow.checked,halo:nodes.halo.checked,gold_transfer:nodes.transfer.checked,text_mask:nodes.textMask.checked,reduced_motion:nodes.reduce.checked}};
    const a=document.createElement('a'),url=URL.createObjectURL(new Blob([JSON.stringify(trace,null,2)],{type:'application/json'}));a.href=url;a.download=`SWOS_${mode}_${photoSha||'no-photo'}_evidence.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),3000);
  };
  $('swosRecord').onclick=()=>{
    if(!image||!canvas.captureStream||!window.MediaRecorder)return;pause();setTime(0,'P0');recordChunks=[];
    const mime=MediaRecorder.isTypeSupported('video/webm;codecs=vp9')?'video/webm;codecs=vp9':'video/webm';recorder=new MediaRecorder(canvas.captureStream(30),{mimeType:mime});
    recorder.ondataavailable=e=>{if(e.data.size)recordChunks.push(e.data);};recorder.onstop=()=>{const url=URL.createObjectURL(new Blob(recordChunks,{type:'video/webm'}));const a=document.createElement('a');a.href=url;a.download=`SWOS_${mode}_${photoSha.slice(0,12)}_3650ms.webm`;a.click();setTimeout(()=>URL.revokeObjectURL(url),3000);};
    recorder.start();play();
  };
  const timingObserver=(()=>{let last=0;return now=>{if(last){const delta=now-last;frameTimes.push(delta);if(frameTimes.length>240)frameTimes.shift();const sorted=[...frameTimes].sort((a,b)=>a-b);frameTimeP95Ms=sorted[Math.floor((sorted.length-1)*.95)]||0;renderFrames++;}last=now;};})();
  const originalDraw=draw;draw=(t)=>{timingObserver(performance.now());originalDraw(t);};
  resize();
})();
