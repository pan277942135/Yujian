(() => {
  "use strict";
  const core=window.YujianCatch=window.YujianCatch||{};
  core.transitionPhoto = ({photo,image,targetImage,source,target,eventStart,duration,isCurrent,onCleanup}) => {
    if (!photo || !isCurrent() || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return Promise.resolve();
    const host=document.createElement("div"); host.className="catch-transition-photo";host.dataset.component="TransitionHost";
    const parent=image.parentNode, next=image.nextSibling;
    const sourceVisibility=image.style.visibility, targetVisibility=targetImage.style.visibility;
    targetImage.style.visibility="hidden";
    host.appendChild(image);document.body.appendChild(host);
    let done=false,frameId;
    return new Promise(resolve=>{
      const cleanup=()=>{
        if(done)return;done=true;cancelAnimationFrame(frameId);
        parent.insertBefore(image,next);image.style.visibility=sourceVisibility;targetImage.style.visibility=targetVisibility;
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
        for(const key of ["left","top","width","height"])host.style[key]=`${rect[key]}px`;
        if(progress<1)frameId=requestAnimationFrame(paint);else cleanup();
      };
      frameId=requestAnimationFrame(paint);
    });
  };
})();
