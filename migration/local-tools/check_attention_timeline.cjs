const fs=require('fs'),vm=require('vm'),assert=require('assert');
const root=fs.readFileSync('.cache/attention_timeline_root','utf8').trim();
const filename=process.argv[2]||'index.html';
const html=fs.readFileSync(root+'/viewer/'+filename,'utf8');
const elements=new Map();let drawingCalls=0;const lastImages=new Map();let fakeTime=0,intervalCallback=null;
class Element{
 constructor(){this.style={};this.children=[];this.events={};this.value='';this.textContent='';this._id='';}
 set id(id){this._id=id;elements.set(id,this);}get id(){return this._id;}
 appendChild(x){this.children.push(x);return x;}append(...xs){this.children.push(...xs);}replaceChildren(...xs){this.children=[...xs];}
 addEventListener(n,fn){this.events[n]=fn;}
 set innerHTML(text){for(const m of text.matchAll(/id="([^"]+)"/g)){const el=new Element();el.id=m[1];}}
 getContext(){const id=this.id;return{clearRect(){},drawImage(im,...args){assert(args.every(Number.isFinite));lastImages.set(id,im.src);drawingCalls++;},fillRect(...a){assert(a.every(Number.isFinite));drawingCalls++;},beginPath(){},moveTo(){},lineTo(){},stroke(){},fillText(){},setLineDash(){}};}
 getBoundingClientRect(){return{left:0,top:0,width:448,height:448};}
}
for(const m of html.matchAll(/id="([^"]+)"/g)){const el=new Element();el.id=m[1];}
elements.get('time').value='0';elements.get('scale').value='local';elements.get('display').value='overlay';
class Image{set src(s){assert(s.startsWith('data:image/jpeg;base64,'));this._src=s;this.onload();}get src(){return this._src;}}
const ctx=vm.createContext({document:{getElementById(id){assert(elements.has(id),id);return elements.get(id);},createElement(){return new Element();}},Image,console,atob:s=>Buffer.from(s,'base64').toString('binary'),window:{location:{href:''}},performance:{now:()=>fakeTime},setInterval(fn){intervalCallback=fn;return 1;},clearInterval(){intervalCallback=null;}});
vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1],ctx);
(async()=>{
 await vm.runInContext('render()',ctx);
 const data=vm.runInContext('DATA',ctx);
 for(let t=0;t<=280;t++){
  await vm.runInContext(`seek(${t})`,ctx);
  const u=Math.min(275,Math.floor(t/5)*5);
  assert.strictEqual(vm.runInContext('shownPlan',ctx),u);
  for(const mode of data.modes)for(let c=0;c<2;c++){
   const cam=c?'wrist':'front',id=c+'_'+mode;
   assert.strictEqual(lastImages.get('live'+id),'data:image/jpeg;base64,'+data.rollouts[mode].images[cam][t]);
   assert.strictEqual(lastImages.get('heat'+id),'data:image/jpeg;base64,'+data.rollouts[mode].images[cam][u]);
   assert(elements.get('heatLabel'+id).textContent.endsWith(String(u)));
  }
  assert.strictEqual(elements.get('blocks').children.length,11);
 }
 for(let flow=0;flow<3;flow++)for(let layer=0;layer<3;layer++)for(const scale of ['local','shared'])for(const display of ['heat','overlay']){
  for(const [k,v] of Object.entries({flow,layer,scale,display}))elements.get(k).value=String(v);
  await vm.runInContext('render()',ctx);
  assert(!elements.get('mass0_correct').textContent.match(/NaN|undefined/));
 }
 await vm.runInContext('seek(83)',ctx);assert(elements.get('alignment').textContent.includes('右图保留规划步 80'));
 await vm.runInContext('seek(280)',ctx);assert(elements.get('alignment').textContent.includes('轨迹已结束'));
 elements.get('episode').value='4';elements.get('episode').events.change();assert.strictEqual(ctx.window.location.href,'episode_004.html');
 // Simulated real-time playback advances 20 control steps per second.
 await vm.runInContext('seek(0)',ctx);elements.get('play').events.click();fakeTime=1000;intervalCallback();await vm.runInContext('render()',ctx);assert.strictEqual(elements.get('time').value,'20');vm.runInContext('pause()',ctx);
 console.log(`${filename}: all 281 control/anchor image pairs validated across 6 panels; 36 layer/flow/color/display combinations; step 83 and terminal 280; page switch and 20Hz playback passed (${drawingCalls} draws).`);
 console.log('Script/DOM/canvas simulation only; actual browser rendering and speed are not measured.');
})().catch(e=>{console.error(e);process.exit(1);});
