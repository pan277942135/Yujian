const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const context={window:{},crypto:require('node:crypto').webcrypto,URL:{createObjectURL:()=> 'blob:fixture',revokeObjectURL:()=>{}},AbortController};
vm.createContext(context);
for(const name of ['photo-session','recognition-controller'])vm.runInContext(fs.readFileSync(`app/static/js/catch/${name}.js`,'utf8'),context);
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
