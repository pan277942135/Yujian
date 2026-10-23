(() => {
  "use strict";
  const core=window.YujianCatch=window.YujianCatch||{};
  const PROCESSING_MS=2800, REVEAL_MS=850, TOTAL_MS=3650;

  function sampleCurve(frames,values,elapsedMs){
    if(!Array.isArray(frames)||!Array.isArray(values)||frames.length!==values.length||!frames.length)return 0;
    const t=Math.max(0,Number(elapsedMs)||0);
    if(t<=frames[0].at)return values[0];
    for(let index=1;index<frames.length;index++){
      if(t<=frames[index].at){
        const from=frames[index-1],to=frames[index];
        const progress=(t-from.at)/(to.at-from.at);
        return values[index-1]+(values[index]-values[index-1])*progress;
      }
    }
    return values[values.length-1];
  }

  function frameAt(frames,elapsedMs){
    let current=frames?.[0]?.id||"P0";
    for(const frame of frames||[]){if(frame.at>elapsedMs)break;current=frame.id;}
    return current;
  }

  // Contract strengths are perceptual relative values, never CSS alpha. These
  // bounded curves translate them into a restrained, layered browser treatment.
  function paintForStrength(waterStrength,goldStrength){
    const water=Math.max(0,Math.min(1,waterStrength/75));
    const gold=Math.max(0,Math.min(1,goldStrength/100));
    return {
      ambient:(0.12*Math.sqrt(water)).toFixed(3),
      caustic:(0.24*Math.pow(water,0.78)).toFixed(3),
      ripple:(0.09*Math.pow(water,0.7)).toFixed(3),
      refraction:(0.13*Math.sqrt(water)).toFixed(3),
      focus:(0.15*Math.pow(gold,0.8)).toFixed(3),
    };
  }

  const VERTEX_SHADER=`#version 300 es
    precision highp float;
    out vec2 v_uv;
    void main(){
      vec2 p=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2));
      v_uv=p;
      gl_Position=vec4(p*2.0-1.0,0.0,1.0);
    }`;
  const FRAGMENT_SHADER=`#version 300 es
    precision highp float;
    in vec2 v_uv;
    uniform vec2 u_resolution;
    uniform float u_time;
    uniform float u_strength;
    out vec4 out_color;
    float hash21(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453123);}
    float noise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.0-2.0*f);return mix(mix(hash21(i),hash21(i+vec2(1,0)),f.x),mix(hash21(i+vec2(0,1)),hash21(i+vec2(1,1)),f.x),f.y);}
    float fbm(vec2 p){float v=0.0,a=.5;for(int i=0;i<4;i++){v+=noise(p)*a;p=p*2.03+vec2(17.1,9.2);a*=.5;}return v;}
    void main(){
      vec2 uv=v_uv;vec2 p=(uv-.5)*vec2(u_resolution.x/u_resolution.y,1.0);
      float t=u_time*.18;float warp=fbm(p*3.2+vec2(t*.34,-t*.25));
      float low=fbm(p*2.15+vec2(-t*.22,t*.31));
      float a=sin((p.x+warp*.17)*8.4+t*.45);
      float b=sin((p.y+warp*.13)*7.1-t*.38);
      float lattice=abs(a*b);
      float cells=1.0-smoothstep(.12,.63,lattice);
      float patch=smoothstep(.22,.78,low+warp*.2);
      float upward=smoothstep(.02,.92,1.0-uv.y);
      float caustic=cells*patch*(.42+.58*upward);
      float alpha=clamp(caustic*u_strength*.34,0.0,.24);
      vec3 color=mix(vec3(.57,.78,.72),vec3(1.0,.84,.57),smoothstep(.56,.9,warp));
      out_color=vec4(color,alpha);
    }`;

  function compileShader(gl,type,source){
    const shader=gl.createShader(type);gl.shaderSource(shader,source);gl.compileShader(shader);
    if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS)){gl.deleteShader(shader);return null;}
    return shader;
  }

  function createSwosRenderer({media,host,focus,maskCanvas,waterGLCanvas,water2DCanvas,photoStage,getReducedMotion=()=>window.matchMedia?.("(prefers-reduced-motion: reduce)").matches===true}){
    let frames=[],water=[],gold=[],revealFrames=[],revealWater=[],revealGold=[];
    let startAt=0,revealStartAt=0,raf=0,running=false,mode="idle",bbox=null;
    let resultState="",maskTrust="NONE",maskImage=null,maskTexture=null,maskVisible=false,lastElapsed=0;
    const maskContext=maskCanvas?.getContext("2d",{willReadFrequently:true});
    let gl=null,glProgram=null,glMode="STATIC_CSS",water2DContext=null,previousFrameAt=0,droppedFrames=0;
    const frameCosts=[],rafIntervals=[];
    const layers=new Map([...host.querySelectorAll("[data-optical-layer]")].map(node=>[node.dataset.opticalLayer,node]));
    function initWebGL(){
      if(!waterGLCanvas)return false;
      try{gl=waterGLCanvas.getContext("webgl2",{alpha:true,antialias:false,premultipliedAlpha:true});}catch(_error){gl=null;}
      if(!gl)return false;
      const vertex=compileShader(gl,gl.VERTEX_SHADER,VERTEX_SHADER),fragment=compileShader(gl,gl.FRAGMENT_SHADER,FRAGMENT_SHADER);
      if(!vertex||!fragment){gl=null;return false;}
      glProgram=gl.createProgram();gl.attachShader(glProgram,vertex);gl.attachShader(glProgram,fragment);gl.linkProgram(glProgram);
      if(!gl.getProgramParameter(glProgram,gl.LINK_STATUS)){gl=null;glProgram=null;return false;}
      gl.bindVertexArray(gl.createVertexArray());gl.useProgram(glProgram);gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
      waterGLCanvas.hidden=false;if(water2DCanvas)water2DCanvas.hidden=true;glMode="WEBGL2";media.dataset.swosRenderer=glMode;return true;
    }
    function initCanvas2D(){
      if(!water2DCanvas)return false;
      try{water2DContext=water2DCanvas.getContext("2d");}catch(_error){water2DContext=null;}
      if(!water2DContext)return false;
      water2DCanvas.hidden=false;if(waterGLCanvas)waterGLCanvas.hidden=true;glMode="CANVAS2D";media.dataset.swosRenderer=glMode;return true;
    }
    function setWaterSize(canvas,context){
      if(!canvas||!context)return null;
      const dpr=Math.min(2,window.devicePixelRatio||1),width=Math.max(1,Math.round(canvas.clientWidth*dpr)),height=Math.max(1,Math.round(canvas.clientHeight*dpr));
      if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height;}
      return {width,height,dpr,cssWidth:width/dpr,cssHeight:height/dpr};
    }
    function drawWaterCanvas(elapsedMs,strength){
      if(strength<=0){
        for(const [canvas,context] of [[waterGLCanvas,gl],[water2DCanvas,water2DContext]])if(canvas&&context)context.clearRect?.(0,0,canvas.width,canvas.height);
        if(gl){gl.viewport(0,0,waterGLCanvas.width,waterGLCanvas.height);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);}
        return;
      }
      if(glMode==="WEBGL2"&&gl&&glProgram&&waterGLCanvas){
        const size=setWaterSize(waterGLCanvas,gl);gl.viewport(0,0,size.width,size.height);gl.useProgram(glProgram);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);
        gl.uniform2f(gl.getUniformLocation(glProgram,"u_resolution"),size.width,size.height);gl.uniform1f(gl.getUniformLocation(glProgram,"u_time"),elapsedMs/1000);gl.uniform1f(gl.getUniformLocation(glProgram,"u_strength"),Math.max(0,Math.min(1,strength/75)));gl.drawArrays(gl.TRIANGLES,0,3);return;
      }
      if(glMode!=="CANVAS2D"&&!initCanvas2D()){glMode="STATIC_CSS";media.dataset.swosRenderer=glMode;return;}
      const size=setWaterSize(water2DCanvas,water2DContext),ctx=water2DContext,w=size.cssWidth,h=size.cssHeight,v=Math.max(0,Math.min(1,strength/75)),t=elapsedMs/1000;
      ctx.setTransform(size.dpr,0,0,size.dpr,0,0);ctx.clearRect(0,0,w,h);ctx.globalCompositeOperation="screen";
      for(let i=0;i<18;i++){
        const seed=i*2.399963,drift=t*(i%2?.07:-.045),x=w*(.08+.84*((i*.618034+.19)%1)),y=h*(.82-.7*((i*.414214+.13)%1)+Math.sin(seed+drift)*.012);
        const rx=w*(.025+.045*((i*.754878)%1)),ry=h*(.018+.026*((i*.569841)%1));ctx.save();ctx.translate(x,y);ctx.rotate(Math.sin(seed+drift)*.18);ctx.scale(1,ry/rx);
        const g=ctx.createRadialGradient(0,0,0,0,0,rx);g.addColorStop(0,`rgba(246,238,199,${.24*v})`);g.addColorStop(.48,`rgba(225,239,215,${.15*v})`);g.addColorStop(1,"rgba(225,239,215,0)");ctx.fillStyle=g;ctx.beginPath();ctx.arc(0,0,rx,0,Math.PI*2);ctx.fill();ctx.restore();
      }
      for(let band=0;band<3;band++)for(let i=0;i<11;i++){
        const seed=i*2.399963+band*11.3,drift=t*(.12+band*.05),cx=w*(.1+.8*((i*.754878+band*.21)%1)),cy=h*(.18+.72*((i*.618034+band*.17)%1));
        const span=w*[.08,.05,.028][band],bend=Math.sin(seed+drift)*h*.012,alpha=[.25,.18,.12][band]*v;
        const gradient=ctx.createLinearGradient(cx-span,cy,cx+span,cy);gradient.addColorStop(0,"rgba(246,240,203,0)");gradient.addColorStop(.45,`rgba(250,245,216,${alpha})`);gradient.addColorStop(1,"rgba(246,240,203,0)");
        ctx.strokeStyle=gradient;ctx.lineWidth=[4,2.3,1.3][band];ctx.lineCap="round";ctx.shadowColor=`rgba(239,235,190,${alpha})`;ctx.shadowBlur=[6,3,1][band];ctx.beginPath();ctx.moveTo(cx-span,cy+bend);ctx.bezierCurveTo(cx-span*.2,cy-h*.018,cx+span*.4,cy+h*.015,cx+span,cy-bend);ctx.stroke();
      }
      ctx.globalCompositeOperation="source-over";
    }
    if(!initWebGL())initCanvas2D();
    waterGLCanvas?.addEventListener("webglcontextlost",event=>{event.preventDefault();gl=null;glProgram=null;if(!initCanvas2D()){glMode="STATIC_CSS";media.dataset.swosRenderer=glMode;}paint(lastElapsed);});
    waterGLCanvas?.addEventListener("webglcontextrestored",()=>{if(initWebGL())paint(lastElapsed);});
    const paint=(elapsedMs)=>{
      const duration=mode==="reveal"?REVEAL_MS:PROCESSING_MS;
      lastElapsed=Math.max(0,Math.min(duration,elapsedMs));
      const waterStrength=sampleCurve(frames,water,elapsedMs);
      const goldStrength=sampleCurve(frames,gold,elapsedMs);
      const goldAllowed=mode==="reveal"&&resultState==="RESULT_HIGH"&&maskTrust==="TRUSTED_MASK";
      const visual=paintForStrength(waterStrength,goldAllowed?goldStrength:0);
      const frame=frameAt(frames,elapsedMs);
      media.dataset.swosFrame=frame;
      media.dataset.swosPhase=mode==="reveal"?"RESULT_REVEAL":"PROCESSING";
      media.dataset.relativeWaterStrength=String(Math.round(waterStrength));
      media.dataset.relativeFishGoldStrength=String(Math.round(goldStrength));
      media.dataset.swosOpticalState="ACTIVE";
      for(const [name,value] of [["ambient-water",visual.ambient],["caustic-field",visual.caustic],["ripple-field",visual.ripple],["peripheral-refraction",visual.refraction]]){
        const layer=layers.get(name);if(layer)layer.style.opacity=value;
      }
      drawWaterCanvas(elapsedMs,waterStrength);
      if(bbox){
        focus.classList?.add("visible");
        focus.style.left=`${bbox.x1*100}%`;focus.style.top=`${bbox.y1*100}%`;
        focus.style.width=`${(bbox.x2-bbox.x1)*100}%`;focus.style.height=`${(bbox.y2-bbox.y1)*100}%`;
        focus.style.opacity=visual.focus;focus.dataset.geometryTrust="BBOX_ONLY_SOFT_REGION";
      }else{
        focus.classList?.remove("visible");
        focus.style.opacity="0";focus.dataset.geometryTrust="NONE_NO_SPATIAL_FOCUS";
      }
      const residual=layers.get("residual-gold-trace");
      if(residual){residual.style.opacity=goldAllowed?visual.focus:"0";residual.style.visibility=goldAllowed?"visible":"hidden";}
      paintMask(goldAllowed?goldStrength:0);
      return {frame,waterStrength,goldStrength};
    };
    function paintMask(goldStrength){
      if(!maskCanvas||!maskContext)return;
      const dpr=Math.min(2,window.devicePixelRatio||1),width=Math.max(1,Math.round(maskCanvas.clientWidth*dpr)),height=Math.max(1,Math.round(maskCanvas.clientHeight*dpr));
      if(maskCanvas.width!==width||maskCanvas.height!==height){maskCanvas.width=width;maskCanvas.height=height;maskTexture=null;}
      maskContext.setTransform(dpr,0,0,dpr,0,0);maskContext.clearRect(0,0,width/dpr,height/dpr);
      const trust=maskVisible&&maskImage;
      const fallbackTrust=bbox?"BBOX_ONLY":"NONE";
      host.dataset.maskState=trust?"REVIEW_PROXY":fallbackTrust;
      media.dataset.maskTrust=trust?"REVIEW_PROXY":fallbackTrust;
      if(!trust||goldStrength<=0)return;
      if(!maskTexture){
        const source=document.createElement("canvas");source.width=width;source.height=height;
        const sourceContext=source.getContext("2d",{willReadFrequently:true});sourceContext.drawImage(maskImage,0,0,width,height);
        const pixels=sourceContext.getImageData(0,0,width,height),data=pixels.data;
        for(let i=0;i<data.length;i+=4){const alpha=data[i];data[i]=245;data[i+1]=190;data[i+2]=86;data[i+3]=alpha;}
        sourceContext.putImageData(pixels,0,0);maskTexture=source;
      }
      const strength=Math.max(0,Math.min(1,goldStrength/100));maskContext.save();maskContext.globalCompositeOperation="screen";
      maskContext.filter="blur(18px)";maskContext.globalAlpha=.13*strength;maskContext.drawImage(maskTexture,0,0,width/dpr,height/dpr);
      maskContext.filter="blur(6px)";maskContext.globalAlpha=.22*strength;maskContext.drawImage(maskTexture,0,0,width/dpr,height/dpr);
      maskContext.filter="none";maskContext.globalAlpha=.20*strength;maskContext.drawImage(maskTexture,0,0,width/dpr,height/dpr);maskContext.restore();
    }
    const tick=(now)=>{
      if(!running)return;
      if(previousFrameAt){
        const interval=now-previousFrameAt;rafIntervals.push(interval);if(rafIntervals.length>180)rafIntervals.shift();
        if(interval>24)droppedFrames+=Math.max(1,Math.floor(interval/16.7)-1);
        const intervals=[...rafIntervals].sort((a,b)=>a-b);media.dataset.rendererRafIntervalP95Ms=(intervals[Math.floor((intervals.length-1)*.95)]||0).toFixed(2);
      }
      previousFrameAt=now;
      const costStart=performance.now();
      const duration=mode==="reveal"?REVEAL_MS:PROCESSING_MS;
      const origin=mode==="reveal"?revealStartAt:startAt;
      const elapsed=Math.max(0,Math.min(duration,now-origin));
      paint(elapsed);
      frameCosts.push(performance.now()-costStart);if(frameCosts.length>180)frameCosts.shift();
      const sorted=[...frameCosts].sort((a,b)=>a-b);media.dataset.rendererFrameP95Ms=(sorted[Math.floor((sorted.length-1)*.95)]||0).toFixed(2);media.dataset.droppedFrames=String(droppedFrames);
      if(elapsed<duration)raf=requestAnimationFrame(tick);else running=false;
    };
    return Object.freeze({
      start(startTime,timeline){
        this.stop(false);frames=timeline.processingFrames;water=timeline.waterStrength;gold=timeline.goldStrength;
        revealFrames=timeline.revealFrames||revealFrames;revealWater=timeline.revealWaterStrength||revealWater;revealGold=timeline.revealGoldStrength||revealGold;
        startAt=startTime;mode="processing";running=true;paint(getReducedMotion()?PROCESSING_MS:0);
        if(!getReducedMotion())raf=requestAnimationFrame(tick);
      },
      startReveal(startTime,timeline,options={}){
        this.stop(false);frames=timeline.revealFrames||revealFrames;water=timeline.revealWaterStrength||revealWater;gold=timeline.revealGoldStrength||revealGold;
        resultState=options.resultState||"";maskTrust=options.maskTrust||"NONE";revealStartAt=startTime;mode="reveal";running=true;paint(getReducedMotion()?REVEAL_MS:0);
        if(!getReducedMotion())raf=requestAnimationFrame(tick);
      },
      setRevealState(options={}){resultState=options.resultState||resultState;maskTrust=options.maskTrust||maskTrust;if(running)paint(mode==="reveal"?Math.max(0,Math.min(REVEAL_MS,performance.now()-revealStartAt)):Math.max(0,Math.min(PROCESSING_MS,performance.now()-startAt)));},
      setGeometry(box){
        bbox=core.PhotoViewport.validBox(box);
        if(bbox&&running)paint(mode==="reveal"?Math.max(0,Math.min(REVEAL_MS,performance.now()-revealStartAt)):Math.max(0,Math.min(PROCESSING_MS,performance.now()-startAt)));
      },
      async setMask(mask){
        maskVisible=false;maskImage=null;maskTexture=null;
        if(mask?.status!=="REVIEW_PROXY"||mask.renderAllowed!==true||!mask.value){paint(lastElapsed);return false;}
        const image=new Image();image.src=mask.value;
        try{await image.decode();}catch(_error){paint(lastElapsed);return false;}
        maskImage=image;paint(lastElapsed);return true;
      },
      setMaskReviewEnabled(enabled){maskVisible=Boolean(enabled&&maskImage);paint(lastElapsed);},
      stop(reset=true){
        running=false;if(raf)cancelAnimationFrame(raf);raf=0;
        if(reset){
          for(const layer of layers.values())layer.style.opacity="0";
          focus.classList?.remove("visible");focus.style.opacity="0";focus.removeAttribute("data-geometry-trust");
          if(maskContext){maskContext.clearRect(0,0,maskCanvas.width,maskCanvas.height);}
          maskImage=null;maskTexture=null;maskVisible=false;host.dataset.maskState="NONE";media.removeAttribute("data-mask-trust");
          media.removeAttribute("data-swos-frame");media.removeAttribute("data-relative-water-strength");
          media.removeAttribute("data-relative-fish-gold-strength");media.dataset.swosOpticalState="IDLE";
          bbox=null;
        }
      },
      renderAt(elapsedMs){const absolute=Math.max(0,Math.min(TOTAL_MS,Number(elapsedMs)||0));if(absolute<=PROCESSING_MS){mode="processing";return paint(absolute);}mode="reveal";return paint(absolute-PROCESSING_MS);},
    });
  }

  core.SWOSTimeline=Object.freeze({sampleCurve,frameAt,paintForStrength,processingMs:PROCESSING_MS,revealMs:REVEAL_MS,totalMs:TOTAL_MS});
  core.createSwosRenderer=createSwosRenderer;
})();
