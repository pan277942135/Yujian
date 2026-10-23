(() => {
  "use strict";
  const keys=[['P0',0,0,0],['P1',350,10,0],['P2',950,45,15],['P3',1200,75,45],['P4',1550,60,100],['P5',2050,35,100],['P6',2600,10,100],['P7',2800,0,100],['R0',2800,0,100],['R0.5',2900,0,100],['R1',3000,0,100],['R2',3180,0,30],['R3',3350,0,0],['R4',3480,0,0],['R5',3650,0,0]];
  const $=id=>document.getElementById(id), canvas=$('swosCanvas'), ctx=canvas.getContext('2d');
  const nodes={photo:$('swosPhoto'),name:$('swosName'),play:$('swosPlay'),seek:$('swosSeek'),rate:$('swosRate'),time:$('swosTime'),info:$('swosInfo')};
  let image=null,photoSha='',time=0,playing=false,anchor=0,anchorTime=0,raf=0,recorder=null,recordChunks=[];
  const clamp=(n,a,b)=>Math.max(a,Math.min(b,n));
  const lerp=(a,b,p)=>a+(b-a)*p;
  const ease=p=>p*p*(3-2*p);
  const rectLerp=(a,b,p)=>Object.fromEntries(['x','y','w','h'].map((k)=>[k,lerp(a[k],b[k],p)]));
  function curve(t,i){
    let prev=keys[0];for(const next of keys.slice(1)){
      if(t<=next[1])return lerp(prev[i],next[i],next[1]===prev[1]?1:(t-prev[1])/(next[1]-prev[1]));prev=next;
    }return prev[i];
  }
  function keyAt(t){return [...keys].reverse().find(k=>k[1]<=t)||keys[0];}
  function contained(source,box){let s=Math.min(box.w/source.width,box.h/source.height),w=source.width*s,h=source.height*s;return{x:box.x+(box.w-w)/2,y:box.y+(box.h-h)/2,w,h};}
  function geometry(t,w,h){
    const processing={x:0,y:0,w,h}, hero={x:w*19/360,y:h*53/640,w:w*322/360,h:h*210/640};
    const p=clamp((t-2800)/200,0,1);
    const outer=rectLerp(processing,hero,p), first=contained(image,processing), last=contained(image,hero);
    // One oriented source, one center and uniform-scale interpolation. At 360x640 and a 3:4 source,
    // the 100ms midpoint is (50.625,66.5,258.75,345) exactly.
    const photo=rectLerp(first,last,p);
    return{outer,photo,sourceMatrix:[photo.w/image.width,0,0,photo.h/image.height,photo.x,photo.y]};
  }
  function waterField(g,t,strength,photo){
    if(strength<=0)return;
    const v=Math.pow(strength/75,.82),w=photo.w,h=photo.h;
    // The photo, cover and optical layer share the same source-to-canvas rectangle.
    // Expansion rises through P3; after P4 only intensity falls, without a new loop.
    const spread=ease(clamp((t-350)/850,0,1));
    g.save();g.beginPath();g.rect(photo.x,photo.y,w,h);g.clip();
    g.translate(photo.x,photo.y);g.globalCompositeOperation='screen';
    for(let i=0;i<7;i++){
      const phase=i*2.399963,drift=t*.00014*(i%2?1:-.7);
      const x=w*(.14+.72*((i*.618034+.13)%1));
      const y=h*(.8-spread*(.16+.62*((i*.414214+.21)%1)));
      const rx=w*(.17+.13*((i*.754878)%1)),ry=h*(.08+.06*((i*.569841)%1));
      g.save();g.translate(x+Math.sin(phase+drift)*w*.012,y);g.scale(1,ry/rx);
      const halo=g.createRadialGradient(0,0,0,0,0,rx);
      halo.addColorStop(0,`rgba(246,235,187,${.30*v})`);
      halo.addColorStop(.48,`rgba(232,240,207,${.16*v})`);
      halo.addColorStop(1,'rgba(232,240,207,0)');
      g.fillStyle=halo;g.beginPath();g.arc(0,0,rx,0,Math.PI*2);g.fill();g.restore();
    }
    // Unequal, broken refracted bands; their positions are a pure function of time.
    for(let band=0;band<3;band++){
      const count=[13,19,23][band];
      for(let j=0;j<count;j++){
        const seed=j*2.399963+band*13.17;
        const rise=(j*.618034+band*.19)%1;
        const y=h*(.86-spread*(.12+.76*rise));
        const cx=w*(.08+.84*((j*.754878+band*.32)%1));
        const span=w*[.23,.14,.075][band]*(.65+.55*((j*.414214)%1));
        const travel=t*.00038*(band+1),bend=Math.sin(seed+travel)*h*.013;
        const fade=(.55+.45*Math.sin(seed*2+travel*.7)**2)*v;
        const gradient=g.createLinearGradient(cx-span,y,cx+span,y);
        gradient.addColorStop(0,'rgba(239,243,210,0)');
        gradient.addColorStop(.38,`rgba(252,247,206,${[.34,.26,.17][band]*fade})`);
        gradient.addColorStop(.64,`rgba(222,245,224,${[.23,.18,.13][band]*fade})`);
        gradient.addColorStop(1,'rgba(222,245,224,0)');
        g.strokeStyle=gradient;g.lineCap='round';g.lineWidth=[11,5,2.5][band]*w/360;
        g.shadowColor=`rgba(239,235,190,${.35*fade})`;
        g.shadowBlur=[14,8,4][band]*w/360;
        g.beginPath();g.moveTo(cx-span,y+bend);
        g.bezierCurveTo(cx-span*.4,y-h*.018*Math.sin(seed+travel),cx+span*.28,y+h*.025*Math.sin(seed*1.7-travel),cx+span,y-bend*.6);
        g.stroke();
      }
    }
    g.restore();
  }
  function draw(t){
    const w=canvas.width,h=canvas.height;
    ctx.clearRect(0,0,w,h);ctx.fillStyle='#d9e8e5';ctx.fillRect(0,0,w,h);
    if(!image){nodes.info.textContent='无本地照片 · 不生成鱼体轮廓或推理结果';return;}
    const geom=geometry(t,w,h),o=geom.outer,p=geom.photo;
    ctx.save();ctx.beginPath();ctx.rect(o.x,o.y,o.w,o.h);ctx.clip();
    // Same source image forms the quiet cover and the full, uncropped foreground.
    ctx.globalAlpha=.17;ctx.filter='blur(18px)';const coverScale=Math.max(o.w/image.width,o.h/image.height);
    ctx.drawImage(image,o.x+(o.w-image.width*coverScale)/2,o.y+(o.h-image.height*coverScale)/2,image.width*coverScale,image.height*coverScale);
    ctx.filter='none';ctx.globalAlpha=1;ctx.drawImage(image,p.x,p.y,p.w,p.h);
    if(!$('swosCompare').checked){
      const water=$('swosReduce').checked?0:curve(t,2);
      if($('swosWater').checked && water>0)waterField(ctx,t,water,p);
      // No qualified production Mask is supplied. BBox is also absent: fish-attached light stays off.
    }
    ctx.restore();
    if(t>=3000){
      const title=$('swosName').value.trim()||'鱼种名称';
      const y=clamp(o.y+o.h+Math.max(28,h*.065),0,h-35);
      const fontSize=clamp(w*.065,17,38);
      ctx.font=`600 ${fontSize}px system-ui, sans-serif`;ctx.textAlign='center';ctx.textBaseline='middle';
      if(t>=3480||$('swosReduce').checked){ctx.fillStyle='#294c43';ctx.fillText(title,w/2,y);}
      else if(t>=3350){ctx.save();ctx.beginPath();ctx.rect(w*.2,y-fontSize,w*.14,fontSize*1.3);ctx.rect(w*.56,y-fontSize,w*.08,fontSize*1.3);ctx.clip();ctx.fillStyle='#af9764';ctx.fillText(title,w/2,y);ctx.restore();}
    }
    nodes.info.textContent=JSON.stringify({photo_sha256:photoSha,source_px:[image.width,image.height],viewport_px:[w,h],frame:keyAt(t)[0],elapsed_ms:Math.round(t),photo_rect:p,outer_rect:o,source_matrix:geom.sourceMatrix,mask_trust:'NONE',fish_gold:'DISABLED_NO_TRUSTED_MASK',gold_transfer:'DISABLED_NO_TRUSTED_MASK',renderer:'Canvas 2D',webgl2_available:Boolean(document.createElement('canvas').getContext('webgl2')),reduced_motion:$('swosReduce').checked,visual_fixture_only:true},null,2);
  }
  function setTime(value){time=clamp(Number(value)||0,0,3650);nodes.seek.value=String(Math.round(time));nodes.time.textContent=`${keyAt(time)[0]} · ${Math.round(time)}ms`;document.querySelectorAll('#swosKeys button').forEach(b=>b.setAttribute('aria-current',String(Number(b.dataset.at)===time)));draw(time);}
  function pause(){playing=false;cancelAnimationFrame(raf);nodes.play.textContent='播放';}
  function tick(now){if(!playing)return;setTime(anchorTime+(now-anchor)*Number(nodes.rate.value));if(time<3650)raf=requestAnimationFrame(tick);else{pause();if(recorder?.state==='recording')recorder.stop();}}
  function play(){if(!image)return;if(time>=3650)setTime(0);anchorTime=time;anchor=performance.now();playing=true;nodes.play.textContent='暂停';raf=requestAnimationFrame(tick);}
  nodes.play.onclick=()=>playing?pause():play();$('swosRestart').onclick=()=>{pause();setTime(0);play();};
  nodes.seek.oninput=e=>{pause();setTime(e.target.value);};nodes.rate.onchange=()=>{if(playing){anchorTime=time;anchor=performance.now();}};
  const step=d=>{pause();let index=keys.findIndex(k=>k[1]>time);if(index<0)index=keys.length;const current=keys.findIndex(k=>k[1]===time);setTime(keys[clamp((current>=0?current:index)+d,0,keys.length-1)][1]);};
  $('swosPrev').onclick=()=>step(-1);$('swosNext').onclick=()=>step(1);
  keys.forEach(([id,at],i)=>{const b=document.createElement('button');b.type='button';b.textContent=id;b.title=`${id} · ${at}ms`;b.dataset.at=at;b.onclick=()=>{pause();setTime(at);};$('swosKeys').append(b);});
  function resize(){canvas.width=clamp(Number($('swosWidth').value)||360,240,2000);canvas.height=clamp(Number($('swosHeight').value)||640,320,2000);draw(time);}
  $('swosViewport').onchange=e=>{if(e.target.value==='free')return;const [w,h]=e.target.value.split(',');$('swosWidth').value=w;$('swosHeight').value=h;resize();};
  for(const id of ['swosWidth','swosHeight'])$(id).onchange=()=>{$('swosViewport').value='free';resize();};
  for(const id of ['swosWater','swosCompare','swosReduce','swosName'])$(id).oninput=()=>draw(time);
  nodes.photo.onchange=async()=>{pause();const file=nodes.photo.files?.[0];if(!file)return;try{
    const buffer=await file.arrayBuffer(),digest=await crypto.subtle.digest('SHA-256',buffer);photoSha=[...new Uint8Array(digest)].map(x=>x.toString(16).padStart(2,'0')).join('');
    const decoded=await createImageBitmap(file,{imageOrientation:'from-image'});image?.close();image=decoded;$('swosEmpty').hidden=true;setTime(0);
  }catch(error){nodes.info.textContent=`无法读取本地照片：${error.message}`;}};
  $('swosExport').onclick=()=>{if(!image)return;canvas.toBlob(blob=>{if(!blob)return;const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`SWOS_${keyAt(time)[0].replace('.','_')}_${Math.round(time)}ms.png`;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),3000);});};
  $('swosRecord').onclick=()=>{if(!image||!canvas.captureStream||!window.MediaRecorder)return;pause();setTime(0);recordChunks=[];recorder=new MediaRecorder(canvas.captureStream(30),{mimeType:MediaRecorder.isTypeSupported('video/webm;codecs=vp9')?'video/webm;codecs=vp9':'video/webm'});recorder.ondataavailable=e=>{if(e.data.size)recordChunks.push(e.data);};recorder.onstop=()=>{const url=URL.createObjectURL(new Blob(recordChunks,{type:'video/webm'}));const a=document.createElement('a');a.href=url;a.download='SWOS_review_3650ms.webm';a.click();setTimeout(()=>URL.revokeObjectURL(url),3000);};recorder.start();play();};
  resize();
})();
