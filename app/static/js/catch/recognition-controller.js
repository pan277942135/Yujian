(() => {
  "use strict";
  const core=window.YujianCatch=window.YujianCatch||{};
  class RecognitionController {
    constructor() {
      Object.assign(this,{file:null,objectUrl:null,inference:null,controller:null,requestSequence:0,busy:false,saved:false,appToken:null,species:[],selectedByUser:false,saveInProgress:false,bbox:null,photo:null,phase:"IDLE",cancelTransition:null,transitionState:"idle",revealState:"idle"});
    }
    current(sequence) { return sequence===this.requestSequence; }
    invalidate() {
      this.requestSequence++;this.controller?.abort();this.controller=null;this.cancelTransition?.();this.cancelTransition=null;this.busy=false;
      this.photo?.dispose();this.photo=null;this.phase="IDLE";this.transitionState="idle";this.revealState="idle";
    }
    select(file,width,height) {
      this.photo=new core.PhotoSession(file,width,height);
      this.file=file;this.objectUrl=this.photo.url;this.phase="UPLOAD";
    }
    begin() {
      this.requestSequence++;this.controller?.abort();this.controller=new AbortController();this.busy=true;this.phase="PROCESSING";this.transitionState="idle";this.revealState="idle";
      return {sequence:this.requestSequence,controller:this.controller};
    }
    accept(payload,sequence) {
      if (!this.current(sequence)) return false;
      this.inference=payload;this.photo?.accept(payload);this.bbox=this.photo?.bbox||null;return true;
    }
  }
  core.RecognitionController=RecognitionController;
})();
