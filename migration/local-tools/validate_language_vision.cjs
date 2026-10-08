const fs=require('fs'),vm=require('vm'),path=require('path');
const root=fs.readFileSync('.cache/language_vision_root','utf8').trim();
function element(){return {value:'',options:[],paused:true,currentTime:0,src:'',textContent:'',add(o){this.options.push(o);if(this.options.length===1)this.value=String(o.value)},play(){this.paused=false},pause(){this.paused=true}}}
let checked=0;
for(const file of ['index.html','attention_figures/index.html']){
 const html=fs.readFileSync(path.join(root,file),'utf8');const script=html.match(/<script>([\s\S]*)<\/script>/)[1];
 const els={};const context={document:{getElementById(id){if(!els[id]){els[id]=element();els[id].value=({experiment:'language',demo:'milk',camera:'front'})[id]||'';}return els[id]}},Option:function(label,value){return {label,value:String(value)}}};
 vm.runInNewContext(script,context);
 if(file==='index.html'){
  for(const c of els.condition.options)for(const e of els.episode.options){els.condition.value=c.value;els.episode.value=e.value;els.condition.onchange();for(const id of ['environment','input']){if(!fs.existsSync(path.join(root,els[id].src)))throw Error('missing video');}checked++;}
  els.environment.currentTime=5.25;els.environment.onseeked();if(els.input.currentTime!==5.25)throw Error('seek mismatch');els.play.onclick();if(els.environment.paused||els.input.paused)throw Error('play');els.play.onclick();if(!els.environment.paused||!els.input.paused)throw Error('pause');
 }else{
  for(const ex of ['language','vision'])for(const d of ['milk','tomato_sauce'])for(let ep=0;ep<5;ep++)for(const cam of ['front','wrist']){Object.assign(els.experiment,{value:ex});els.demo.value=d;els.episode.value=String(ep);els.camera.value=cam;els.experiment.onchange();if(!fs.existsSync(path.join(root,'attention_figures',els.figure.src)))throw Error('missing figure');checked++;}
 }
}
fs.writeFileSync(path.join(root,'html_validation.json'),JSON.stringify({passed:true,selector_combinations:checked,video_seek_play_pause_logic:true,actual_browser_test:false},null,2));console.log('PASS',checked,'selector combinations; DOM simulation only');
