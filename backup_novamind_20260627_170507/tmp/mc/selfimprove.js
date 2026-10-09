// selfimprove.js — AUTO-MELHORIA SEGURA (lacuna #4). O sistema ajusta os PROPRIOS hiperparametros
// cognitivos pela PERFORMANCE MEDIDA (hill-climbing reversivel e BOUNDED). Inspiracao: meta-aprendizado /
// plasticidade homeostatica (o cerebro regula seus proprios ganhos). NAO reescreve codigo arbitrario
// (seguro): so move knobs dentro de faixas seguras, e REVERTE se piorou. Mede fitness da colonia a cada janela.
const fs=require('fs'), path=require('path');
const SI_PATH=path.join(__dirname,'selfimprove.json');

// hiperparametros AJUSTAVEIS (a cognicao le estes em vez de constantes fixas), com faixas seguras
const TUNE = {
  curiosity:    { v:0.45, min:0.15, max:0.75, step:0.05 },  // prob. de explorar (curiosidade) vs explotar
  exploreBonus: { v:0.35, min:0.10, max:0.60, step:0.05 },  // peso da dopamina na exploracao
  planSalience: { v:0.65, min:0.40, max:0.90, step:0.05 },  // saliencia do planejamento (tech tree) no workspace
  mintCooldown: { v:45000,min:20000,max:90000,step:8000 },  // ms entre mints de projeto via LLM (custo x iniciativa)
};
function get(k){ return TUNE[k]?TUNE[k].v:undefined; }

let last={ fitness:null, key:null, dir:1 };
const HIST=[];
function load(){ try{ const d=JSON.parse(fs.readFileSync(SI_PATH,'utf8')); for(const k in (d.tune||{})) if(TUNE[k]) TUNE[k].v=d.tune[k]; last=d.last||last; }catch(e){} }
function save(){ try{ const tune={}; for(const k in TUNE) tune[k]=TUNE[k].v; fs.writeFileSync(SI_PATH, JSON.stringify({tune,last,hist:HIST.slice(-50)})); }catch(e){} }

const keys=Object.keys(TUNE);
// PASSO de auto-melhoria: recebe a fitness da janela; mantem se melhorou, reverte+troca direcao se piorou (hill-climbing)
function step(fitness){
  HIST.push({t:HIST.length, fitness, tune:Object.fromEntries(keys.map(k=>[k,TUNE[k].v]))});
  if(last.fitness===null){ last.fitness=fitness; last.key=keys[0]; last.dir=1; _nudge(last.key,last.dir); save(); return {first:true}; }
  const improved = fitness > last.fitness + 1e-6;
  if(improved){
    // melhorou: continua mexendo no MESMO knob na MESMA direcao (segue o gradiente)
    last.fitness=fitness; _nudge(last.key,last.dir);
    save(); return {kept:last.key, dir:last.dir, fitness};
  } else {
    // piorou/igual: REVERTE o ultimo passo e tenta OUTRO knob (ou direcao oposta) — seguro
    _nudge(last.key, -last.dir);                 // desfaz
    const i=(keys.indexOf(last.key)+1)%keys.length; last.key=keys[i]; last.dir=(Math.random()<0.5?1:-1);
    last.fitness=fitness; _nudge(last.key,last.dir);
    save(); return {reverted:true, now:last.key, dir:last.dir, fitness};
  }
}
function _nudge(k,dir){ const t=TUNE[k]; if(!t) return; t.v=Math.max(t.min,Math.min(t.max, t.v+dir*t.step)); }
function stats(){ return { tune:Object.fromEntries(keys.map(k=>[k,+(TUNE[k].v).toFixed(3)])), tuning:last.key, dir:last.dir }; }

module.exports={ get, step, load, save, stats, TUNE };
