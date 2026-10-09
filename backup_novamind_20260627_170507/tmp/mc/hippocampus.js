// hippocampus.js — MEMORIA EPISODICA + ESPACIAL (hipocampo de rato/humano).
// Neurociencia (pesquisada): PLACE CELLS = memorias que conjugam LUGAR + atributo nao-espacial (comida/perigo/
// morte aqui); episodica e navegacao sao o MESMO sistema (lobo temporal medial); REPLAY durante o repouso
// reativa sequencias de alto-reward e CONSOLIDA (reforca sinapses, reconstroi 'caminhos virtuais' p/ planejar).
// Preenche o gap: os agentes nao lembravam LUGARES nem EVENTOS especificos (morriam nos mesmos pontos).
const fs=require('fs'), path=require('path');
const HIP_PATH = path.join(__dirname,'hippocampus.json');
const CELL=4;   // tamanho da celula espacial (place cell ~ 4 blocos)

// place memory: chave "cx,cz" -> {food,danger,death,resource,val,n,t}  (lugar + atributos = place cell)
let PLACES={};
let EPISODES=[];   // eventos salientes: {t, cx, cz, type, val}
function load(){ try{ const d=JSON.parse(fs.readFileSync(HIP_PATH,'utf8')); PLACES=d.places||{}; EPISODES=d.episodes||[]; }catch(e){ PLACES={}; EPISODES=[]; } return PLACES; }
function save(){ try{ fs.writeFileSync(HIP_PATH, JSON.stringify({places:PLACES, episodes:EPISODES.slice(-300)})); }catch(e){} }
function key(x,z){ return Math.floor(x/CELL)+','+Math.floor(z/CELL); }

// REGISTRA um evento num lugar (place cell aprende: aqui tem comida / aqui morri / perigo)
function record(pos, type, val){
  if(!pos) return; const k=key(pos.x,pos.z); const p=(PLACES[k]=PLACES[k]||{food:0,danger:0,death:0,resource:0,val:0,n:0});
  p.n++; p.val=p.val*0.85+(val||0)*0.15; p.t=Date.now();
  if(type) p[type]=(p[type]||0)+1;
  if(Math.abs(val||0)>=1.5 || type==='death' || type==='food'){ EPISODES.push({t:Date.now(),cx:Math.floor(pos.x/CELL),cz:Math.floor(pos.z/CELL),type:type||'',val:val||0}); if(EPISODES.length>400) EPISODES.shift(); }
}
function valueAt(pos){ if(!pos) return 0; const p=PLACES[key(pos.x,pos.z)]; return p?p.val:0; }
function dangerAt(pos){ if(!pos) return 0; const p=PLACES[key(pos.x,pos.z)]; return p?(p.death*2+p.danger):0; }

// RECALL: melhor lugar lembrado (mais valioso) dentro de um raio (em celulas), p/ navegar ate ele
function recallBest(pos, attr, maxCells=10){
  if(!pos) return null; const cx=Math.floor(pos.x/CELL), cz=Math.floor(pos.z/CELL);
  let best=null, bv=0.5;
  for(const k in PLACES){ const [kx,kz]=k.split(',').map(Number); const dc=Math.abs(kx-cx)+Math.abs(kz-cz);
    if(dc>maxCells||dc===0) continue; const p=PLACES[k];
    const score=(attr?(p[attr]||0):p.val) - 0.05*dc - p.death*3;   // bom, perto, e nao-mortal
    if(score>bv){ bv=score; best={x:kx*CELL+2, z:kz*CELL+2, score, place:p}; } }
  return best;
}
// lugar lembrado como PERIGOSO/MORTAL por perto? (p/ evitar)
function recallDanger(pos, maxCells=3){
  if(!pos) return 0; const cx=Math.floor(pos.x/CELL), cz=Math.floor(pos.z/CELL); let d=0;
  for(const k in PLACES){ const [kx,kz]=k.split(',').map(Number); if(Math.abs(kx-cx)+Math.abs(kz-cz)<=maxCells) d+=PLACES[k].death*2+PLACES[k].danger; }
  return d;
}

// REPLAY / CONSOLIDACAO (durante sono/repouso): reativa episodios de alto |valor|, reforca os place values
// e devolve o melhor "caminho virtual" lembrado (lugares bons) p/ o agente planejar/ir depois.
function replay(){
  const salient = EPISODES.filter(e=>Math.abs(e.val)>=1.2).slice(-40);
  for(const e of salient){ const k=e.cx+','+e.cz; const p=(PLACES[k]=PLACES[k]||{food:0,danger:0,death:0,resource:0,val:0,n:0});
    p.val = p.val*0.9 + e.val*0.1;   // reforca pela reativacao (consolidacao)
  }
  return salient.length;
}
function prune(cap=500){ const ks=Object.keys(PLACES); if(ks.length>cap){ ks.sort((a,b)=>(Math.abs(PLACES[b].val)+PLACES[b].n*0.01)-(Math.abs(PLACES[a].val)+PLACES[a].n*0.01)); for(let i=cap;i<ks.length;i++) delete PLACES[ks[i]]; } }
function stats(){ return { places:Object.keys(PLACES).length, episodes:EPISODES.length,
  deathSpots:Object.values(PLACES).filter(p=>p.death>0).length, foodSpots:Object.values(PLACES).filter(p=>p.food>0).length }; }

module.exports={ record, valueAt, dangerAt, recallBest, recallDanger, replay, prune, load, save, stats };
