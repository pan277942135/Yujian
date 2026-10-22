(() => {
  "use strict";
  const core=window.YujianCatch=window.YujianCatch||{};

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

  function createSwosRenderer({media,host,focus,photoStage,getReducedMotion=()=>window.matchMedia?.("(prefers-reduced-motion: reduce)").matches===true}){
    let frames=[],water=[],gold=[],startAt=0,raf=0,running=false,bbox=null;
    const layers=new Map([...host.querySelectorAll("[data-optical-layer]")].map(node=>[node.dataset.opticalLayer,node]));
    const paint=(elapsedMs)=>{
      const waterStrength=sampleCurve(frames,water,elapsedMs);
      const goldStrength=sampleCurve(frames,gold,elapsedMs);
      const visual=paintForStrength(waterStrength,goldStrength);
      const frame=frameAt(frames,elapsedMs);
      media.dataset.swosFrame=frame;
      media.dataset.relativeWaterStrength=String(Math.round(waterStrength));
      media.dataset.relativeFishGoldStrength=String(Math.round(goldStrength));
      media.dataset.swosOpticalState="ACTIVE";
      for(const [name,value] of [["ambient-water",visual.ambient],["caustic-field",visual.caustic],["ripple-field",visual.ripple],["peripheral-refraction",visual.refraction]]){
        const layer=layers.get(name);if(layer)layer.style.opacity=value;
      }
      if(bbox){
        focus.style.left=`${bbox.x1*100}%`;focus.style.top=`${bbox.y1*100}%`;
        focus.style.width=`${(bbox.x2-bbox.x1)*100}%`;focus.style.height=`${(bbox.y2-bbox.y1)*100}%`;
        focus.style.opacity=visual.focus;focus.dataset.geometryTrust="BBOX_ONLY_SOFT_REGION";
      }else{
        focus.style.opacity="0";focus.dataset.geometryTrust="NONE_NO_SPATIAL_FOCUS";
      }
      return {frame,waterStrength,goldStrength};
    };
    const tick=(now)=>{
      if(!running)return;
      const elapsed=Math.max(0,Math.min(2800,now-startAt));
      paint(elapsed);
      if(elapsed<2800)raf=requestAnimationFrame(tick);
    };
    return Object.freeze({
      start(startTime,timeline){
        this.stop(false);frames=timeline.processingFrames;water=timeline.waterStrength;gold=timeline.goldStrength;
        startAt=startTime;running=true;paint(getReducedMotion()?2800:0);
        if(!getReducedMotion())raf=requestAnimationFrame(tick);
      },
      setGeometry(box){
        bbox=core.PhotoViewport.validBox(box);
        if(bbox&&running)paint(Math.max(0,Math.min(2800,performance.now()-startAt)));
      },
      stop(reset=true){
        running=false;if(raf)cancelAnimationFrame(raf);raf=0;
        if(reset){
          for(const layer of layers.values())layer.style.opacity="0";
          focus.style.opacity="0";focus.removeAttribute("data-geometry-trust");
          media.removeAttribute("data-swos-frame");media.removeAttribute("data-relative-water-strength");
          media.removeAttribute("data-relative-fish-gold-strength");media.dataset.swosOpticalState="IDLE";
          bbox=null;
        }
      },
      renderAt:paint,
    });
  }

  core.SWOSTimeline=Object.freeze({sampleCurve,frameAt,paintForStrength});
  core.createSwosRenderer=createSwosRenderer;
})();
