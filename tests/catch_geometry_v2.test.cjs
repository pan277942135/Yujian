const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
let maskDecodeGate=null;
const maskCanvasDraws=[],offscreenMaskCanvases=[];
function createMaskCanvas(main=false){
 const pixels=new Uint8ClampedArray([255,255,255,255,0,0,0,255,128,128,128,255,0,0,0,0]);
 const ctx={globalAlpha:1,globalCompositeOperation:'source-over',filter:'none',setTransform(){},clearRect(){},save(){this.saved={globalAlpha:this.globalAlpha,globalCompositeOperation:this.globalCompositeOperation,filter:this.filter};},restore(){if(this.saved)Object.assign(this,this.saved);},drawImage(image){if(main)maskCanvasDraws.push({image,globalAlpha:this.globalAlpha,composite:this.globalCompositeOperation,filter:this.filter});},getImageData(){return {data:new Uint8ClampedArray(pixels)};},putImageData(image){this.painted=Array.from(image.data);}};
 const canvas={width:2,height:2,clientWidth:2,clientHeight:2,getContext:()=>ctx,ctx};
 if(!main)offscreenMaskCanvases.push(canvas);
 return canvas;
}
const mainMaskCanvas=createMaskCanvas(true);
const context={window:{},crypto:require('node:crypto').webcrypto,performance:{now:()=>0},URL:{createObjectURL:()=> 'blob:fixture',revokeObjectURL:()=>{}},AbortController,Image:class{async decode(){if(maskDecodeGate)await maskDecodeGate;}set src(value){this._src=value;}get src(){return this._src;}},document:{createElement(name){assert.equal(name,'canvas');return createMaskCanvas(false);}}};
vm.createContext(context);
for(const name of ['photo-session','recognition-controller','stage-view','swos-renderer'])vm.runInContext(fs.readFileSync(`app/static/js/catch/${name}.js`,'utf8'),context);
const {PhotoViewport:g,RecognitionController}=context.window.YujianCatch;
test('contain geometry preserves portrait, landscape and narrow source edges',()=>{
 for(const [w,h] of [[800,600],[600,900],[100,1200]]){
 const geom=g.viewport(w,h,{left:10,top:20,width:300,height:200});
 assert.ok(Math.abs(geom.rect.width/geom.rect.height-w/h)<1e-9);
 const box=g.mapBox({x1:0,y1:0,x2:1,y2:1},geom);
 assert.equal(box.left,geom.rect.left);assert.equal(box.width,geom.rect.width);
 assert.ok(box.left>=10 && box.top>=20 && box.left+box.width<=310.001 && box.top+box.height<=220.001);
 }
});
test('cover, midpoint and invalid box rejection',()=>{
 assert.equal(g.viewport(100,200,{left:0,top:0,width:300,height:200},'cover').scale,3);
 assert.equal(g.interpolate({left:0,top:0,width:300,height:200},{left:100,top:200,width:150,height:100},.5).left,50);
 for(const box of [{x1:-.1,y1:0,x2:1,y2:1},{x1:.8,y1:0,x2:.2,y2:1},{x1:0,y1:0,x2:Infinity,y2:1}])assert.equal(g.validBox(box),null);
});
test('one controller invalidates cancelled requests and ignores late response',()=>{
 const c=new RecognitionController();c.select({name:'a'},600,900);const first=c.begin();
 c.invalidate();c.select({name:'b'},800,600);const second=c.begin();
 assert.equal(first.controller.signal.aborted,true);
 assert.equal(c.accept({result_state:'RESULT_HIGH'},first.sequence),false);
 assert.equal(c.accept({result_state:'RESULT_LOW'},second.sequence),true);
 assert.equal(c.photo.result.result_state,'RESULT_LOW');assert.equal(c.photo.width,800);
});
test('SWOS samples the frozen 8-point curves and keeps perceptual strength separate from CSS alpha',()=>{
 const frames=[{id:'P0',at:0},{id:'P1',at:350},{id:'P2',at:950},{id:'P3',at:1200},{id:'P4',at:1550},{id:'P5',at:2050},{id:'P6',at:2600},{id:'P7',at:2800}];
 const water=[0,10,45,75,60,35,10,0],gold=[0,0,15,45,100,100,100,100],t=context.window.YujianCatch.SWOSTimeline;
 assert.equal(frames.length,8);assert.equal(t.sampleCurve(frames,water,0),0);assert.equal(t.sampleCurve(frames,water,1200),75);assert.equal(t.sampleCurve(frames,water,2800),0);
 assert.equal(t.sampleCurve(frames,gold,1550),100);assert.equal(t.sampleCurve(frames,gold,2800),100);
 assert.equal(t.frameAt(frames,1000),'P2');
 const p4FallbackFocus=t.bboxFallbackFocusStrength(60);
 assert.equal(t.bboxFallbackFocusStrength(0),0);
 assert.equal(p4FallbackFocus,80);
 assert.equal(t.bboxFallbackFocusStrength(75),100);
 assert.equal(t.paintForStrength(60,p4FallbackFocus).focus,'0.201');
 const paint=t.paintForStrength(75,100);
 assert.ok(Number(paint.caustic)>0&&Number(paint.caustic)<0.75);
 assert.ok(Number(paint.focus)>0&&Number(paint.focus)<0.25);
 assert.notEqual(Number(paint.caustic),0.75);assert.notEqual(Number(paint.focus),1);
});
test('SWOS exposes the complete 3650ms clock and blocks gold for an unqualified mask',()=>{
 const t=context.window.YujianCatch.SWOSTimeline;
 assert.equal(t.processingMs,2800);assert.equal(t.revealMs,850);assert.equal(t.totalMs,3650);
 const names=['photo-base','ambient-water','caustic-field','ripple-field','peripheral-refraction','fish-focus','residual-gold-trace'];
 const nodes=names.map(name=>({dataset:{opticalLayer:name},style:{},removeAttribute(){},}));
 const host={querySelectorAll:()=>nodes};
 const media={dataset:{}};const focus={style:{},dataset:{},removeAttribute(){}};
 const renderer=context.window.YujianCatch.createSwosRenderer({media,host,focus,getReducedMotion:()=>true});
 renderer.start(0,{
  processingFrames:[{id:'P0',at:0},{id:'P1',at:350},{id:'P2',at:950},{id:'P3',at:1200},{id:'P4',at:1550},{id:'P5',at:2050},{id:'P6',at:2600},{id:'P7',at:2800}],
  waterStrength:[0,10,45,75,60,35,10,0],goldStrength:[0,0,15,45,100,100,100,100],
 });
 renderer.setGeometry({x1:.1,y1:.2,x2:.8,y2:.9});
 renderer.renderAt(1550);
 assert.equal(focus.style.opacity,t.paintForStrength(60,t.bboxFallbackFocusStrength(60)).focus);
 assert.ok(Number(focus.style.opacity)>0);
 assert.equal(focus.dataset.geometryTrust,'BBOX_ONLY_SOFT_REGION');
 assert.equal(nodes[6].style.opacity,'0');
 const p7=renderer.renderAt(2800);
 assert.equal(p7.frame,'P7');
 assert.equal(p7.waterStrength,0);
 assert.equal(p7.goldStrength,100);
 assert.equal(media.dataset.relativeWaterStrength,'0');
 assert.equal(media.dataset.relativeFishGoldStrength,'100');
 renderer.startReveal(0,{revealFrames:[{id:'R0',at:0},{id:'R5',at:850}],revealWaterStrength:[0,0],revealGoldStrength:[0,100]},{resultState:'RESULT_HIGH',maskTrust:'BBOX_ONLY'});
 assert.equal(media.dataset.swosPhase,'RESULT_REVEAL');
 assert.equal(focus.dataset.geometryTrust,'BBOX_ONLY_SOFT_REGION');
 assert.equal(nodes[6].style.opacity,'0');
});
test('REVIEW_PROXY candidate paint is muted, opt-in, and unit-fixture-only (not SAM evidence)',async()=>{
 maskCanvasDraws.length=0;offscreenMaskCanvases.length=0;
 const layers=[];
 const host={dataset:{},querySelectorAll:()=>layers};
 const media={dataset:{},removeAttribute(name){delete this.dataset[name.replace(/-([a-z])/g,(_match,letter)=>letter.toUpperCase())];}};
 const focus={style:{},dataset:{},classList:{add(){},remove(){}},removeAttribute(){}};
 const renderer=context.window.YujianCatch.createSwosRenderer({media,host,focus,maskCanvas:mainMaskCanvas,getReducedMotion:()=>true});
 renderer.start(0,{processingFrames:[{id:'P0',at:0},{id:'P7',at:2800}],waterStrength:[0,0],goldStrength:[0,100]});
 const loaded=await renderer.setMask({status:'REVIEW_PROXY',renderAllowed:true,value:'data:image/png;base64,TEST_FIXTURE_ONLY'});
 assert.equal(loaded,true);
 assert.equal(host.dataset.maskState,'REVIEW_PROXY');
 assert.equal(media.dataset.maskTrust,'REVIEW_PROXY');
 assert.equal(maskCanvasDraws.length,0,'candidate must stay hidden until explicit review opt-in');
 renderer.setMaskReviewEnabled(true);
 assert.equal(maskCanvasDraws.length,1,'explicit review opt-in paints the decoded candidate mask');
 assert.equal(maskCanvasDraws[0].globalAlpha,0.34);
 assert.equal(maskCanvasDraws[0].composite,'source-over');
 assert.equal(maskCanvasDraws[0].filter,'none');
 const painted=offscreenMaskCanvases[0].ctx.painted;
 assert.deepEqual(painted.slice(0,4),[70,145,138,255]);
 assert.deepEqual(painted.slice(4,8),[70,145,138,0]);
 assert.deepEqual(painted.slice(8,12),[70,145,138,128]);
 renderer.setMaskReviewEnabled(false);
 assert.equal(maskCanvasDraws.length,1,'turning review off clears rather than redraws the mask');
 renderer.stop();
 assert.equal(host.dataset.maskState,'NONE');
 assert.equal(media.dataset.maskTrust,undefined);
 let finishDecode;
 maskDecodeGate=new Promise(resolve=>{finishDecode=resolve;});
 const staleLoad=renderer.setMask({status:'REVIEW_PROXY',renderAllowed:true,value:'data:image/png;base64,STALE_TEST_FIXTURE_ONLY'});
 renderer.stop();
 finishDecode();
 assert.equal(await staleLoad,false,'a decode started before reset must be discarded');
 assert.equal(host.dataset.maskState,'NONE');
 assert.equal(media.dataset.maskTrust,undefined);
 assert.equal(maskCanvasDraws.length,1,'the stale decode must not repaint after reset');
 maskDecodeGate=null;
});
test('journey remains explicit about unavailable segmentation during processing',()=>{
 const makeStep=()=>({dataset:{},attrs:{},status:{textContent:''},image:{src:'',hidden:true,removeAttribute(){}},placeholder:{hidden:false},querySelector(selector){return selector==='[data-journey-status]'?this.status:selector==='.journey-placeholder'?this.placeholder:this.image;},setAttribute(key,value){this.attrs[key]=value;},removeAttribute(key){delete this.attrs[key];}});
 const steps=Array.from({length:5},makeStep),root={dataset:{},querySelectorAll:()=>steps};
 const state={phase:'PROCESSING',photo:{url:'blob:private-local',bbox:{x1:.1,y1:.2,x2:.9,y2:.8},mask:{status:'UNAVAILABLE'}},bbox:{x1:.1,y1:.2,x2:.9,y2:.8},busy:true,inference:null,transitionState:'idle',revealState:'idle'};
 context.window.YujianCatch.renderJourney(root,state);
 assert.equal(steps[1].dataset.state,'active');assert.equal(steps[2].dataset.state,'unavailable');
 assert.match(steps[2].status.textContent,/不绘制轮廓/);assert.equal(steps[2].image.src,state.photo.url);
});
