const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const context={window:{},crypto:require('node:crypto').webcrypto,URL:{createObjectURL:()=> 'blob:fixture',revokeObjectURL:()=>{}},AbortController};
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
 const paint=t.paintForStrength(75,100);
 assert.ok(Number(paint.caustic)>0&&Number(paint.caustic)<0.75);
 assert.ok(Number(paint.focus)>0&&Number(paint.focus)<0.25);
 assert.notEqual(Number(paint.caustic),0.75);assert.notEqual(Number(paint.focus),1);
});
test('journey remains explicit about unavailable segmentation during processing',()=>{
 const makeStep=()=>({dataset:{},attrs:{},status:{textContent:''},image:{src:'',hidden:true,removeAttribute(){}},placeholder:{hidden:false},querySelector(selector){return selector==='[data-journey-status]'?this.status:selector==='.journey-placeholder'?this.placeholder:this.image;},setAttribute(key,value){this.attrs[key]=value;},removeAttribute(key){delete this.attrs[key];}});
 const steps=Array.from({length:5},makeStep),root={dataset:{},querySelectorAll:()=>steps};
 const state={phase:'PROCESSING',photo:{url:'blob:private-local',bbox:{x1:.1,y1:.2,x2:.9,y2:.8},mask:{status:'UNAVAILABLE'}},bbox:{x1:.1,y1:.2,x2:.9,y2:.8},busy:true,inference:null,transitionState:'idle',revealState:'idle'};
 context.window.YujianCatch.renderJourney(root,state);
 assert.equal(steps[1].dataset.state,'active');assert.equal(steps[2].dataset.state,'unavailable');
 assert.match(steps[2].status.textContent,/不绘制轮廓/);assert.equal(steps[2].image.src,state.photo.url);
});
