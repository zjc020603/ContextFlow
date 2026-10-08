const fs=require('fs'),vm=require('vm'),assert=require('assert');
const root=fs.readFileSync('.cache/attention_tutorial_root','utf8').trim();
const html=fs.readFileSync(root+'/index.html','utf8');
const elements=new Map();let drawCalls=0;
class Element{
 constructor(tag='div'){this.tag=tag;this.style={};this.children=[];this.events={};this.value='';this.textContent='';}
 appendChild(x){this.children.push(x);return x;}
 append(...xs){this.children.push(...xs);}
 replaceChildren(...xs){this.children=[...xs];}
 addEventListener(name,fn){this.events[name]=fn;}
 set innerHTML(text){for(const match of text.matchAll(/id="([^"]+)"/g))elements.set(match[1],new Element());}
 getContext(){return {clearRect(){},drawImage(){drawCalls++;},fillRect(x,y,w,h){assert([x,y,w,h].every(Number.isFinite));drawCalls++;}};}
 getBoundingClientRect(){return {left:0,top:0,width:448,height:448};}
}
for(const match of html.matchAll(/id="([^"]+)"/g))elements.set(match[1],new Element());
elements.get('scale').value='shared';elements.get('display').value='overlay';
class MockImage{set src(v){assert(v.startsWith('data:image/png;base64,'));this.onload();}}
const ctx=vm.createContext({document:{getElementById(id){assert(elements.has(id),id);return elements.get(id);},createElement(tag){return new Element(tag);}},Image:MockImage,console});
const source=html.match(/<script>([\s\S]*?)<\/script>/)[1];vm.runInContext(source,ctx);
assert(elements.get('status').textContent.includes('已加载'));
let count=0;
for(let step=0;step<3;step++)for(let layer=0;layer<3;layer++)for(const scope of ['first','first5','all50'])for(const scale of ['shared','local'])for(const display of ['overlay','heat']){
 for(const [k,v] of Object.entries({step,layer,scope,scale,display}))elements.get(k).value=String(v);
 elements.get('step').events.change();count++;
 assert.strictEqual(elements.get('blocks').children.length,11);
 assert.strictEqual(elements.get('tokens').children.length,12);
 assert(!elements.get('total').textContent.match(/NaN|undefined/));
 const values=vm.runInContext('current',ctx);assert.strictEqual(values.length,1027);assert(values.every(Number.isFinite));
}
elements.get('reset').events.click();assert.strictEqual(elements.get('step').value,'2');
const canvas=elements.get('canvas0');canvas.events.mousemove({target:canvas,clientX:238,clientY:322});
assert(elements.get('hover0').textContent.includes('patch 行 11 / 列 8'));
assert(elements.get('hover0').textContent.includes('0.10782%'));
const data=vm.runInContext('DATA',ctx);assert(data.tokens.includes('<newline>'));assert.strictEqual(data.tokens.filter(x=>x==='the').length,2);
console.log(`JavaScript/DOM simulation: ${count} selector combinations rendered; masks/data lengths, reset and patch tooltip passed (${drawCalls} drawing calls).`);
console.log('This checks script behavior, not browser pixel layout. Static figures were visually inspected separately.');
