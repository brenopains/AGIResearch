// percept_forge.js — FORJA PERCEPTUAL (porte honesto da ideia do CC Engine / MiniMax).
// A forja de ACOES (forge.js) inventa novos verbos motores. ESTA inventa novas PERCEPCOES:
// quando uma CLASSE de situacao do worldmodel tem erro de predicao ALTO e que NAO CAI (irredutivel com as
// features atuais de classify = ameaca/fome/periodo/obj), o problema e que a classe MISTURA situacoes que o
// agente nao distingue. A forja entao testa FEATURES candidatas (funcoes baratas do estado) e PROMOVE a que
// melhor DIVIDE a classe (maior reducao de variancia do efeito = ganho de informacao) -> a classe vira
// sub-classes previsiveis. Isso e "inventar um novo jeito de VER", nao recombinar acoes. Bounded, sem LLM.
// HONESTO: e bounded (biblioteca de features candidatas e fixa); nao quebra Solomonoff — escolhe QUAL distincao
// perceptual ja-computavel importa, guiado pela propria falha de predicao. Mas e a peca perceptual que faltava.

// ===== BIBLIOTECA DE FEATURES CANDIDATAS (funcoes baratas bot->valor discreto) =====
const CANDIDATES = {
  alt_baixa:    (bot)=>{ try{ return bot.entity.position.y < 60 ? 'baixo':'alto'; }catch(_){ return '?'; } },
  agua_abaixo:  (bot)=>{ try{ const b=bot.blockAt(bot.entity.position.offset(0,-1,0)); return (b&&/water/.test(b.name))?'agua':'solido'; }catch(_){ return '?'; } },
  ceu_aberto:   (bot)=>{ try{ const b=bot.blockAt(bot.entity.position.offset(0,3,0)); return (b&&b.name==='air')?'ceu':'teto'; }catch(_){ return '?'; } },
  aliados2:     (bot)=>{ try{ let n=0; const p=bot.entity.position; for(const id in bot.entities){ const o=bot.entities[id]; if(o&&o.username&&o.username.startsWith('Nova_')&&o.position.distanceTo(p)<12) n++; } return n>=2?'grupo':'sozinho'; }catch(_){ return '?'; } },
  tem_ferramenta:(bot)=>{ try{ return bot.inventory.items().some(i=>/pickaxe|axe|sword/.test(i.name))?'armado':'desarmado'; }catch(_){ return '?'; } },
  vida_baixa:   (bot)=>{ try{ return (bot.health??20)<10?'ferido':'inteiro'; }catch(_){ return '?'; } },
};
const CAND_NAMES = Object.keys(CANDIDATES);
const NOVEL_TYPES = [];   // tipos de bloco NOVOS (introduzidos de FORA) viram features candidatas -> a percepcao CRESCE com o mundo

// PRESSAO EXTERNA: registra um tipo de bloco/objeto nunca-visto como uma NOVA feature candidata.
// Isso EXPANDE o espaco perceptual a partir de um estimulo de FORA (nao do nosso codigo) -> ataca a parede do fecho.
function registerNovel(type){
  if(!type) return false;
  const key='perto_'+type.replace(/[^a-z0-9_]/gi,'');
  if(CANDIDATES[key]) return false;
  CANDIDATES[key]=(bot)=>{ try{ const b=bot.findBlock({matching:x=>x&&x.name===type, maxDistance:8}); return b?'sim':'nao'; }catch(_){ return '?'; } };
  CAND_NAMES.push(key); NOVEL_TYPES.push(type); return true;
}
function novelStats(){ return { types:NOVEL_TYPES.length, list:NOVEL_TYPES.slice(-8) }; }

function snapshot(bot){ const o={}; for(const k of CAND_NAMES){ o[k]=CANDIDATES[k](bot); } return o; }

// ===== LOG bounded de observacoes por CLASSE-BASE (situacao+acao) =====
// LOG[clsBase][act] = [{feats, actual}], cap por chave
const LOG = {};
const LOG_CAP = 60;
function observe(clsBase, act, feats, actual){
  if(!feats) return;
  clsBase = String(clsBase||'').split('#')[0];   // SEMPRE loga na classe-BASE (sem o suffix das features ja promovidas)
  const c=(LOG[clsBase]=LOG[clsBase]||{}); const arr=(c[act]=c[act]||[]);
  arr.push({feats, actual}); if(arr.length>LOG_CAP) arr.shift();
}

// ===== PROMOCOES: clsBase -> [features ativadas] (ate MAX_PER_CLASS) =====
const PROMOTED = {};   // clsBase -> [featureName,...]
const MAX_PER_CLASS = 2;
const STAT = { invented:0, tested:0, lastFeature:null };

// suffix(clsBase, bot): anexa TODAS as features promovidas da classe (refina em sub-classes previsiveis)
function suffix(clsBase, bot){ const fs=PROMOTED[clsBase]; if(!fs||!fs.length) return '';
  try{ return fs.map(f=>'#'+f+':'+CANDIDATES[f](bot)).join(''); }catch(_){ return ''; } }

// variancia de uma lista
function variance(xs){ if(xs.length<2) return 0; const m=xs.reduce((a,b)=>a+b,0)/xs.length; return xs.reduce((a,b)=>a+(b-m)*(b-m),0)/xs.length; }

// PROMOVE: para cada clsBase com variancia de efeito e observacoes suficientes, acha a feature candidata que mais
// REDUZ a variancia ao dividir -> inventa a percepcao. AGORA: ate 2 features/classe, limiares menores, log preservado.
function promote(minObs=12, minGain=0.20){
  for(const clsBase in LOG){
    const already = PROMOTED[clsBase] || [];
    if(already.length >= MAX_PER_CLASS) continue;          // ja refinou o suficiente esta classe
    const obs=[]; for(const act in LOG[clsBase]) for(const o of LOG[clsBase][act]) obs.push(o);
    if(obs.length<minObs) continue;
    const baseVar = variance(obs.map(o=>o.actual));
    if(baseVar < 0.3) continue;                            // classe ja previsivel
    STAT.tested++;
    let bestF=null, bestVar=baseVar;
    for(const f of CAND_NAMES){
      if(already.includes(f)) continue;                    // nao re-promove a mesma feature
      const groups={}; for(const o of obs){ const v=(o.feats&&o.feats[f])||'?'; (groups[v]=groups[v]||[]).push(o.actual); }
      if(Object.keys(groups).length<2) continue;           // feature nao divide nada aqui (ex: sempre 'nao')
      let wvar=0; for(const g in groups){ wvar += (groups[g].length/obs.length)*variance(groups[g]); }
      if(wvar < bestVar){ bestVar=wvar; bestF=f; }
    }
    const gain = bestF ? (baseVar - bestVar)/baseVar : 0;
    if(bestF && gain >= minGain){
      (PROMOTED[clsBase]=PROMOTED[clsBase]||[]).push(bestF);
      STAT.invented++; STAT.lastFeature=clsBase+'->'+bestF+`(ganho ${(gain*100).toFixed(0)}%)`;
      // NAO apaga o log (permite 2a promocao); so limita o tamanho
      for(const act in LOG[clsBase]){ if(LOG[clsBase][act].length>40) LOG[clsBase][act]=LOG[clsBase][act].slice(-40); }
    }
  }
}

function stats(){ return { invented:STAT.invented, tested:STAT.tested, last:STAT.lastFeature, promoted:Object.keys(PROMOTED).length }; }

module.exports = { snapshot, observe, promote, suffix, stats, CAND_NAMES, getPromoted:()=>PROMOTED,
                   registerNovel, novelStats };
