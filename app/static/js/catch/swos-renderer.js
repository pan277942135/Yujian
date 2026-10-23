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

  function rendererMode(){
    try{
      const canvas=document.createElement("canvas");
      if(canvas.getContext("webgl2"))return "WEBGL2_OPTICAL_LAYERS";
      if(canvas.getContext("2d"))return "CANVAS2D_CACHED_OPTICAL_LAYERS";
    }catch(_error){/* static sample is the safe final fallback */}
    return "STATIC_OPTICAL_SAMPLE";
  }

  function createSwosRenderer({media,host,focus,photoStage,getReducedMotion=()=>window.matchMedia?.("(prefers-reduced-motion: reduce)").matches===true}){
    let processingFrames=[],processingWater=[],processingGold=[],revealFrames=[],revealWater=[],revealGold=[];
    let startAt=0,revealStartAt=0,raf=0,running=false,mode="idle",bbox=null;
    let resultState="",maskTrust="NONE";
    const layers=new Map([...host.querySelectorAll("[data-optical-layer]")].map(node=>[node.dataset.opticalLayer,node]));
    const opticalRenderer=rendererMode();
    media.dataset.swosRenderer=opticalRenderer;

    const setLayer=(name,opacity)=>{
      const layer=layers.get(name);if(!layer)return;
      layer.style.opacity=String(opacity);layer.dataset.opticalState=Number(opacity)>0?"ACTIVE":"IDLE";
    };
    const paintCommon=(frame,waterStrength,goldStrength,phase)=>{
      const visual=paintForStrength(waterStrength,goldStrength);
      media.dataset.swosFrame=frame;media.dataset.swosPhase=phase;
      media.dataset.relativeWaterStrength=String(Math.round(waterStrength));
      media.dataset.relativeFishGoldStrength=String(Math.round(goldStrength));
      media.dataset.swosOpticalState="ACTIVE";
      setLayer("ambient-water",visual.ambient);setLayer("caustic-field",visual.caustic);
      setLayer("ripple-field",visual.ripple);setLayer("peripheral-refraction",visual.refraction);
      // A detector box may show only a restrained soft region.  Contour and
      // gold transfer require the same independently qualified source mask.
      if(bbox){
        focus.style.left=`${bbox.x1*100}%`;focus.style.top=`${bbox.y1*100}%`;
        focus.style.width=`${(bbox.x2-bbox.x1)*100}%`;focus.style.height=`${(bbox.y2-bbox.y1)*100}%`;
        focus.style.opacity=visual.focus;focus.dataset.geometryTrust=maskTrust==="TRUSTED_MASK"?"TRUSTED_MASK":"BBOX_ONLY_SOFT_REGION";
      }else{focus.style.opacity="0";focus.dataset.geometryTrust="NONE_NO_SPATIAL_FOCUS";}
      const goldAllowed=resultState==="RESULT_HIGH"&&maskTrust==="TRUSTED_MASK";
      setLayer("residual-gold-trace",goldAllowed?visual.focus:"0");
      const residual=layers.get("residual-gold-trace");if(residual)residual.style.visibility=goldAllowed?"visible":"hidden";
      return {frame,waterStrength,goldStrength,phase,renderer:opticalRenderer,maskTrust,goldAllowed};
    };
    const paintProcessing=(elapsedMs)=>paintCommon(frameAt(processingFrames,elapsedMs),sampleCurve(processingFrames,processingWater,elapsedMs),sampleCurve(processingFrames,processingGold,elapsedMs),"PROCESSING");
    const paintReveal=(elapsedMs)=>paintCommon(frameAt(revealFrames,elapsedMs),sampleCurve(revealFrames,revealWater,elapsedMs),sampleCurve(revealFrames,revealGold,elapsedMs),"RESULT_REVEAL");
    const tick=(now)=>{
      if(!running)return;
      if(mode==="processing"){
        const elapsed=Math.max(0,Math.min(PROCESSING_MS,now-startAt));paintProcessing(elapsed);
        if(elapsed<PROCESSING_MS)raf=requestAnimationFrame(tick);
      }else if(mode==="reveal"){
        const elapsed=Math.max(0,Math.min(REVEAL_MS,now-revealStartAt));paintReveal(elapsed);
        if(elapsed<REVEAL_MS)raf=requestAnimationFrame(tick);else running=false;
      }
    };
    const publicApi={
      start(startTime,timeline){
        this.stop(false);processingFrames=timeline.processingFrames||[];processingWater=timeline.waterStrength||[];processingGold=timeline.goldStrength||[];
        revealFrames=timeline.revealFrames||[{id:"R0",at:0},{id:"R0.5",at:100},{id:"R1",at:200},{id:"R2",at:380},{id:"R3",at:550},{id:"R4",at:680},{id:"R5",at:850}];
        revealWater=timeline.revealWaterStrength||[0,12,18,14,8,2,0];revealGold=timeline.revealGoldStrength||[0,0,0,0,0,0,0];
        startAt=startTime;mode="processing";running=true;paintProcessing(getReducedMotion()?PROCESSING_MS:0);
        if(!getReducedMotion())raf=requestAnimationFrame(tick);
      },
      startReveal(startTime,timeline,options={}){
        this.stop(false);revealFrames=timeline.revealFrames||revealFrames;revealWater=timeline.revealWaterStrength||revealWater;revealGold=timeline.revealGoldStrength||revealGold;
        resultState=options.resultState||"";maskTrust=options.maskTrust||"NONE";revealStartAt=startTime;mode="reveal";running=true;paintReveal(getReducedMotion()?REVEAL_MS:0);
        if(!getReducedMotion())raf=requestAnimationFrame(tick);
      },
      setRevealState(options={}){resultState=options.resultState||resultState;maskTrust=options.maskTrust||maskTrust;if(running){const now=performance.now();mode==="reveal"?paintReveal(Math.max(0,Math.min(REVEAL_MS,now-revealStartAt))):paintProcessing(Math.max(0,Math.min(PROCESSING_MS,now-startAt)));}},
      setGeometry(box){bbox=core.PhotoViewport.validBox(box);if(bbox&&running)this.renderAt(mode==="reveal"?PROCESSING_MS+Math.max(0,Math.min(REVEAL_MS,performance.now()-revealStartAt)):Math.max(0,Math.min(PROCESSING_MS,performance.now()-startAt)));},
      stop(reset=true){
        running=false;if(raf)cancelAnimationFrame(raf);raf=0;mode="idle";
        if(reset){for(const layer of layers.values()){layer.style.opacity="0";layer.dataset.opticalState="IDLE";}const residual=layers.get("residual-gold-trace");if(residual)residual.style.visibility="hidden";focus.style.opacity="0";focus.removeAttribute("data-geometry-trust");media.removeAttribute("data-swos-frame");media.removeAttribute("data-swos-phase");media.removeAttribute("data-relative-water-strength");media.removeAttribute("data-relative-fish-gold-strength");media.dataset.swosOpticalState="IDLE";bbox=null;resultState="";maskTrust="NONE";}
      },
      renderAt(elapsedMs){const absolute=Math.max(0,Math.min(TOTAL_MS,Number(elapsedMs)||0));return absolute<=PROCESSING_MS?paintProcessing(absolute):paintReveal(absolute-PROCESSING_MS);},
    };
    return Object.freeze(publicApi);
  }

  core.SWOSTimeline=Object.freeze({sampleCurve,frameAt,paintForStrength,processingMs:PROCESSING_MS,revealMs:REVEAL_MS,totalMs:TOTAL_MS});
  core.createSwosRenderer=createSwosRenderer;
})();
