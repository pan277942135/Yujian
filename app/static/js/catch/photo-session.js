(() => {
  "use strict";
  const core = window.YujianCatch = window.YujianCatch || {};
  const validBox = (box) => {
    if (!box) return null;
    const x1 = Number(box.x1 ?? box.x), y1 = Number(box.y1 ?? box.y);
    const x2 = Number(box.x2 ?? (x1 + Number(box.width))), y2 = Number(box.y2 ?? (y1 + Number(box.height)));
    return [x1,y1,x2,y2].every(Number.isFinite) && x1 >= 0 && y1 >= 0 && x2 <= 1 && y2 <= 1 && x2 > x1 && y2 > y1 ? {x1,y1,x2,y2} : null;
  };
  function viewport(width, height, frame, fit = "contain") {
    if (!(width > 0 && height > 0 && frame.width > 0 && frame.height > 0)) return null;
    if (!["contain","cover"].includes(fit)) throw new Error("Unsupported photo fit");
    const scale = (fit === "cover" ? Math.max : Math.min)(frame.width / width, frame.height / height);
    const rect = {left:frame.left+(frame.width-width*scale)/2, top:frame.top+(frame.height-height*scale)/2, width:width*scale, height:height*scale};
    return {rect, scale, offsetX:rect.left-frame.left, offsetY:rect.top-frame.top, fit};
  }
  function mapBox(box, geometry) {
    box = validBox(box);
    if (!box || !geometry) return null;
    const r = geometry.rect;
    return {left:r.left+box.x1*r.width, top:r.top+box.y1*r.height, width:(box.x2-box.x1)*r.width, height:(box.y2-box.y1)*r.height};
  }
  function interpolate(from, to, progress) {
    const t=Math.max(0,Math.min(1,progress));
    return Object.fromEntries(["left","top","width","height"].map(k=>[k,from[k]+(to[k]-from[k])*t]));
  }
  class PhotoSession {
    constructor(file, width, height) {
      this.id = crypto.randomUUID(); this.sourceId = this.id; this.file = file;
      this.url = URL.createObjectURL(file); this.width = width; this.height = height;
      this.aspectRatio = width / height; this.orientation = "BROWSER_EXIF_NORMALIZED";
      this.processingGeometry = null; this.resultGeometry = null; this.bbox = null;
      this.sourceSha256 = null; this.mask = {status:"NONE",value:null}; this.result = null;
    }
    geometry(frame, fit = "contain") { return viewport(this.width,this.height,frame,fit); }
    accept(result) {
      this.result = result;
      this.sourceSha256 = result.input?.source_sha256 || null;
      this.bbox = validBox(result.detector?.quality_gate?.primary_bbox || result.detector?.detections?.[0]?.bbox);
      const trust = result.segmentation?.geometry_trust_state;
      const mask = result.segmentation?.mask;
      const sameSource = Boolean(this.sourceSha256 && result.segmentation?.source_image_sha256 === this.sourceSha256
        && mask?.source_image_sha256 === this.sourceSha256);
      const sameDimensions = mask?.width === this.width && mask?.height === this.height;
      const sameBox = validBox(result.segmentation?.bbox) && this.bbox
        && ["x1","y1","x2","y2"].every((key) => Math.abs(result.segmentation.bbox[key] - this.bbox[key]) < 0.00001);
      if (trust === "REVIEW_PROXY" && mask?.data_base64 && sameSource && sameDimensions && sameBox) {
        this.mask = {
          status:"REVIEW_PROXY",
          value:`data:image/png;base64,${mask.data_base64}`,
          sha256:mask.sha256,
          sourceSha256:this.sourceSha256,
          width:mask.width,
          height:mask.height,
          centroid:mask.centroid,
          renderAllowed:result.segmentation.quality?.status !== "INVALID" && mask.render_allowed_in_review_mode === true,
          quality:result.segmentation.quality,
        };
      } else {
        this.mask = {status:this.bbox ? "BBOX_ONLY" : "NONE",value:null};
      }
    }
    dispose() { URL.revokeObjectURL(this.url); }
  }
  Object.assign(core,{PhotoSession,PhotoViewport:Object.freeze({viewport,mapBox,validBox,interpolate})});
})();
