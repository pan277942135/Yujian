(() => {
  "use strict";
  const core=window.YujianCatch=window.YujianCatch||{};
  core.transitionPhoto = ({photo,image,targetImage,source,target,eventStart,duration,isCurrent,onCleanup,onFrame}) => {
    if (!photo || !isCurrent() || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return Promise.resolve();
    const host=document.createElement("div"); host.className="catch-transition-photo";host.dataset.component="TransitionHost";
    const parent=image.parentNode, next=image.nextSibling;
    const sourceVisibility=image.style.visibility, targetVisibility=targetImage.style.visibility;
    const sourceStyle={left:image.style.left,top:image.style.top,width:image.style.width,height:image.style.height};
    targetImage.style.visibility="hidden";
    host.appendChild(image);document.body.appendChild(host);
    Object.assign(image.style,{left:"0px",top:"0px",width:"100%",height:"100%"});
    let done=false,frameId;
    return new Promise(resolve=>{
      const cleanup=()=>{
        if(done)return;done=true;cancelAnimationFrame(frameId);
        parent.insertBefore(image,next);image.style.visibility=sourceVisibility;targetImage.style.visibility=targetVisibility;
        Object.assign(image.style,sourceStyle);
        host.remove();onCleanup(null);resolve();
      };
      onCleanup(cleanup);
      const paint=now=>{
        if(!isCurrent()){cleanup();return;}
        // Remeasure actual content rects every frame: resize and scroll remain valid.
        photo.processingGeometry=photo.geometry(source.getBoundingClientRect());
        photo.resultGeometry=photo.geometry(target.getBoundingClientRect());
        if(!photo.processingGeometry||!photo.resultGeometry){cleanup();return;}
        const progress=Math.max(0,Math.min(1,(now-eventStart)/duration));
        const rect=core.PhotoViewport.interpolate(photo.processingGeometry.rect,photo.resultGeometry.rect,progress);
        host.dataset.progress=progress.toFixed(3);
        const elapsed=Math.max(0,Math.min(duration,now-eventStart));
        const frame=elapsed<100?"R0":elapsed<200?"R0.5":"R1";
        host.dataset.frame=frame;
        host.dataset.elapsedMs=String(Math.round(elapsed));
        for(const key of ["left","top","width","height"])host.style[key]=`${rect[key]}px`;
        onFrame?.({frame,elapsedMs:elapsed,progress,rect});
        if(progress<1)frameId=requestAnimationFrame(paint);else cleanup();
      };
      paint(eventStart);
      frameId=requestAnimationFrame(paint);
      // Force the contract's real 100ms middle geometry even when the display
      // refresh cadence does not land on the R0.5 timestamp.
      window.setTimeout(()=>{if(!done)paint(eventStart+100);},100);
    });
  };
})();
