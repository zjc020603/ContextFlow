const fs=require('fs'),vm=require('vm'),assert=require('assert');
const root=fs.readFileSync('.cache/attention_comparison_root','utf8').trim(),html=fs.readFileSync(root+'/index.html','utf8');
const elements=new Map();let draws=0;
class Element{
 constructor(){this.style={};this.children=[];this.events={};this.value='';this.textContent='';}
 appendChild(x){this.children.push(x);return x;} append(...xs){this.children.push(...xs);} replaceChildren(...xs){this.children=[...xs];}
 addEventListener(n,fn){this.events[n]=fn;}
 set innerHTML(s){for(const m of s.matchAll(/id="([^"]+)"/g))elements.set(m[1],new Element());}
 getContext(){return{clearRect(){},drawImage(){draws++;},fillRect(...args){assert(args.every(Number.isFinite));draws++;}};}
 getBoundingClientRect(){return{left:0,top:0,width:448,height:448};}
}
for(const m of html.matchAll(/id="([^"]+)"/g))elements.set(m[1],new Element());
elements.get('scale').value='local';elements.get('display').value='overlay';
class Image{set src(s){assert(s.startsWith('data:image/png;base64,'));this.onload();}}
const ctx=vm.createContext({document:{getElementById(id){assert(elements.has(id),id);return elements.get(id);},createElement(){return new Element();}},Image,console});
vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1],ctx);
assert(elements.get('status').textContent.includes('已载入'));
let n=0;
for(let episode=0;episode<5;episode++)for(let step=0;step<3;step++)for(let layer=0;layer<3;layer++)for(const scale of ['local','shared'])for(const display of ['overlay','heat']){
 for(const [key,value] of Object.entries({episode,step,layer,scale,display}))elements.get(key).value=String(value);
 elements.get('episode').events.change();n++;
 assert.strictEqual(elements.get('blocks').children.length,11);assert.strictEqual(elements.get('tokens').children.length,12);
 for(const camera of [0,1])for(const mode of ['no','wrong']){const text=elements.get('tv'+camera+'_'+mode).textContent;assert(!/NaN|undefined/.test(text));const v=Number(text.split('= ')[1]);assert(v>=0&&v<=1);}
}
elements.get('reset').events.click();assert.strictEqual(elements.get('episode').value,'0');
assert(elements.get('tv0_wrong').textContent.endsWith('0.055'));assert(elements.get('tv1_no').textContent.endsWith('0.849'));
const canvas=elements.get('map0_correct');canvas.events.mousemove({target:canvas,clientX:238,clientY:322});assert(elements.get('hover0_correct').textContent.includes('0.10782%'));
const data=vm.runInContext('DATA',ctx);for(const episode of data.episodes){assert.strictEqual(episode.context.no.first_action_position_id,525);assert.strictEqual(episode.context.correct.first_action_position_id,653);}
console.log(`JavaScript DOM/canvas simulation passed ${n} control combinations, reset, exact example metrics, position metadata and patch tooltip (${draws} draw calls).`);
console.log('Not a real browser pixel-layout test; PNGs visually inspected separately.');
