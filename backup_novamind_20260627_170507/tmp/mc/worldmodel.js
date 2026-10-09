// worldmodel.js — MODELO DE MUNDO PREDITIVO (cerebro-inspirado), embodido no agente.
// Neurociencia (pesquisada): (1) modelo forward preve (situacao,acao)->efeito; (2) ERRO DE PREDICAO =
// dopamina fasica -> atualiza o modelo (TD-learning, Schultz); (3) PROGRESSO DE APRENDIZADO (erro CAINDO)
// = dopamina tonica -> CURIOSIDADE dirigida ao que e APRENDIVEL (Oudeyer, Intelligent Adaptive Curiosity).
// O agente e atraido pra acoes onde esta APRENDENDO (erro alto mas decrescente), nao pro ruido nem pro ja-dominado.
// Resultado: ele busca ENTENDER as relacoes das coisas (porta->abre, machado->arvore-rapido).
const fs=require('fs'); const path=require('path');
const pforge=require('./percept_forge');   // FORJA PERCEPTUAL: refina classes com feature inventada
const WM_PATH = path.join(__dirname,'worldmodel_state.json');

// vocabulario de acoes de alto nivel (o "espaco motor" sobre o qual ele preve)
const ACTIONS = ['comer','cacar','fugir','cavar','construir','ativar_objeto','craftar','explorar','seguir_aliado','observar'];

// discretizacao da situacao (classe) — grosseira, p/ o modelo generalizar
function classify(api, bot){
  const h=api.nearestHostile&&api.nearestHostile();
  const threat = (h && api.dist(h)<11) ? 'ameaca' : 'seguro';
  const fome = api.food<7?'faminto':(api.food<14?'fome':'farto');
  const per = api.night?'noite':'dia';
  const inter = (api.nearestInteractable && api.nearestInteractable())?'obj':'';
  const base = `${threat}|${fome}|${per}${inter?'|'+inter:''}`;
  // FORJA PERCEPTUAL: se esta classe-base ganhou uma feature inventada, refina (sub-classe previsivel)
  if(bot){ try{ return base + pforge.suffix(base, bot); }catch(_){} }
  return base;
}

// estado do modelo: M[classe][acao] = {pred, errFast, errSlow, n}
let M={};
function load(){ try{ const d=JSON.parse(fs.readFileSync(WM_PATH,'utf8')); if(d&&d.__M){ M=d.__M; Object.assign(T,d.__T||{}); Object.assign(SR,d.__SR||{}); Object.assign(RWD,d.__RWD||{}); Object.assign(SR_ABS,d.__SRA||{}); Object.assign(RWD_ABS,d.__RWDA||{}); Object.assign(T_ABS,d.__TA||{}); } else { M=d; } }catch(e){ M={}; } return M; }
function save(){ try{ fs.writeFileSync(WM_PATH, JSON.stringify({__M:M,__T:T,__SR:SR,__RWD:RWD,__SRA:SR_ABS,__RWDA:RWD_ABS,__TA:T_ABS})); }catch(e){} }
function cell(cls,act){ const c=(M[cls]=M[cls]||{}); return (c[act]=c[act]||{pred:0,errFast:1.0,errSlow:1.0,n:0}); }

// INJETA REGRA CULTURAL (zero-shot planning)
function injectRule(cls, act, predValue) {
  const e = cell(cls, act);
  if (e.n === 0) { // Only inject if we haven't learned it empirically yet
    e.pred = predValue;
    e.errSlow = 0.1; // High confidence from culture
    e.errFast = 0.1;
    e.n = 1; // Mark as known
  }
}

// PREVE o efeito (Δsobrevivencia) da acao na situacao
function predict(cls,act){ const e=cell(cls,act); return e.pred; }

// OBSERVA o efeito real -> ERRO DE PREDICAO (dopamina fasica) + PROGRESSO DE APRENDIZADO (tonica)
// NEUROMODULACAO COMO PRECISAO (Yu&Dayan 2005; Friston precision-weighting):
// ACh = incerteza ESPERADA -> ganho de aprendizado (aprende mais rapido onde o mundo e incerto-mas-aprendivel);
// NE  = incerteza INESPERADA (surpresa) -> exploracao/reset. Modula a propria taxa de aprendizado do modelo.
const NEURO = { ach:0.3, ne:0.1 };
let _lastRpe = 0;
function observe(cls,act,actual){
  const e=cell(cls,act);
  const rpe = actual - e.pred; _lastRpe=rpe;  // erro de predicao de recompensa (dopamina fasica)
  const prec = 0.12 + 0.4*Math.min(1, e.errSlow);   // PRECISAO (ACh por-celula): + taxa onde ha incerteza
  e.pred = e.pred + prec*rpe;                 // TD update modulado por precisao
  const ae = Math.abs(rpe);
  const surprise = ae - e.errSlow;            // erro ACIMA do esperado = incerteza inesperada (NE)
  e.errFast = e.errFast*0.6 + ae*0.4;
  e.errSlow = e.errSlow*0.95 + ae*0.05;
  e.n++;
  NEURO.ach = NEURO.ach*0.97 + Math.min(1,e.errSlow)*0.03;            // incerteza esperada (global)
  NEURO.ne  = Math.max(0, Math.min(1, NEURO.ne*0.9 + Math.max(0,surprise)*0.3));  // surpresa (global)
  const lp = e.errSlow - e.errFast;
  RWD[cls] = (RWD[cls]||0)*0.9 + actual*0.1;
  const _ab=deriveAbs(cls); RWD_ABS[_ab]=(RWD_ABS[_ab]||0)*0.9 + actual*0.1;   // recompensa ABSTRATA (content-invariante)
  // alimenta o buffer de REPLAY PRIORIZADO (prioridade = |rpe| = utilidade do backup, Mattar&Daw)
  REPLAY_BUF.push({cls, act, rpe:ae}); if(REPLAY_BUF.length>400) REPLAY_BUF.shift();
  return { rpe, lp };
}
function neuro(){ return { ach:+NEURO.ach.toFixed(3), ne:+NEURO.ne.toFixed(3) }; }

// ===== REPLAY PRIORIZADO (Mattar&Daw 2018): reativa as transicoes de MAIOR utilidade p/ propagar valor na SR =====
const REPLAY_BUF = [];   // {cls, act, rpe}
const _lastNext = {};    // cls+act -> ultimo newCls visto (p/ replayar a transicao certa)
function prioritizedReplay(n){
  if(REPLAY_BUF.length<8) return 0;
  const sorted = REPLAY_BUF.slice().sort((a,b)=>b.rpe-a.rpe).slice(0, n||16);  // top-utilidade
  let done=0;
  for(const it of sorted){ const nc=_lastNext[it.cls+'|'+it.act]; if(nc){ srUpdate(it.cls, nc); done++; } }
  return done;
}

// PROGRESSO DE APRENDIZADO de uma (classe,acao): quanto o erro esta CAINDO (=quanto da p/ aprender agora)
function learningProgress(cls,act){ const e=cell(cls,act); return e.errSlow - e.errFast; }

// ACAO CURIOSA: a mais APRENDIVEL agora (maior progresso de aprendizado), com bonus p/ pouco-tentadas
function curiousAction(cls, available){
  const acts = available || ACTIONS;
  let best=acts[0], bv=-1e9;
  for(const a of acts){ const e=cell(cls,a);
    const lp = e.errSlow - e.errFast;                 // progresso de aprendizado
    const noveltyBonus = e.n<3 ? 0.5 : 0;             // pouco-tentada = vale explorar
    const score = lp + noveltyBonus + (Math.random()-0.5)*0.1;
    if(score>bv){ bv=score; best=a; } }
  return { action:best, lp:bv };
}
// ACAO DE VALOR: a com maior efeito-esperado (exploracao do modelo aprendido)
function valuedAction(cls, available){
  const acts=available||ACTIONS; let best=acts[0],bv=-1e9;
  for(const a of acts){ const e=cell(cls,a); if(e.pred>bv){bv=e.pred;best=a;} }
  return { action:best, pred:bv };
}

// ====== MODELO DE TRANSICAO + PLANEJAMENTO POR IMAGINACAO (Mattar&Daw 2018 / DYNA) ======
// O cerebro nao so preve o EFEITO de uma acao — ele IMAGINA sequencias (forward sweep pre-frontal/hipocampal)
// e escolhe a que maximiza o resultado futuro previsto. Aqui aprendemos T[classe][acao]->proxima-classe e fazemos
// um ROLLOUT greedy com desconto. Isso usa o modelo causal ja aprendido p/ ANTECIPAR (ex: evitar perigo SEM morrer).
const T = {};   // T[cls][act] = { nextCls: count }
// ===== REPRESENTACAO SUCESSORA (Dayan 1993 / Stachenfeld 2017: place cells = mapa preditivo do hipocampo) =====
// SR[s][k] = ocupacao futura DESCONTADA do estado k partindo de s. Aprendida por TD, INDEPENDENTE da recompensa.
// valor(s) = sum_k SR[s][k]*R[k]  -> quando o objetivo/perigo (R) muda, o valor RE-PLANEJA na hora, sem re-aprender.
const SR = {};   // SR[cls] = { cls2: occupancy }
const RWD = {};  // RWD[cls] = recompensa imediata media (Δsobrevivencia) observada na classe
const SR_A=0.12, SR_G=0.9;
function srUpdate(s, sp){
  if(!s||!sp) return; const row=(SR[s]=SR[s]||{}); const rowp=SR[sp]||{};
  const keys=new Set([s,sp]); for(const k in row)keys.add(k); for(const k in rowp)keys.add(k);
  for(const k of keys){ const ind=(k===s)?1:0; const cur=row[k]||0; const nxt=rowp[k]||0;
    row[k]=cur + SR_A*(ind + SR_G*nxt - cur); }
}
function srValue(cls){ const row=SR[cls]; if(!row) return 0; let v=0; for(const k in row){ v += row[k]*(RWD[k]||0); } return v; }
function observeTransition(prevCls, act, newCls){
  if(!prevCls||!act||!newCls) return;
  const c=(T[prevCls]=T[prevCls]||{}); const a=(c[act]=c[act]||{}); a[newCls]=(a[newCls]||0)+1;
  _lastNext[prevCls+'|'+act]=newCls;   // p/ replay priorizado
  srUpdate(prevCls, newCls);           // aprende o MAPA PREDITIVO FINO (independente de recompensa)
  const pa=deriveAbs(prevCls), na=deriveAbs(newCls);   // MAPA PREDITIVO ABSTRATO (content-invariante -> transfere)
  const ca=(T_ABS[pa]=T_ABS[pa]||{}); ca[na]=(ca[na]||0)+1; srUpdateAbs(pa, na);
}
// ====== NIVEL ABSTRATO CONTENT-INVARIANTE (Active Predictive Coding 2024 + TEM 2020) ======
// A representacao FINA (classify) e content-especifica -> nao transfere p/ conteudo novo. O nivel ABSTRATO
// guarda so a ESTRUTURA relacional (ameaca?/fome-band) SEM conteudo -> a dinamica aprendida aqui TRANSFERE
// instantaneamente p/ qualquer conteudo novo com a mesma estrutura (o "remapping" da TEM / transition task-invariante).
const SR_ABS={}, RWD_ABS={}, T_ABS={};
const TRANSFER={ applied:0 };   // quantas vezes o nivel abstrato carregou valor p/ um estado fino RARO/novo
function deriveAbs(cls){ return String(cls||'').split('#')[0].split('|').slice(0,2).join('|'); }  // ex: "ameaca|farto" (sem periodo/obj/conteudo)
function srUpdateAbs(s,sp){ if(!s||!sp)return; const row=(SR_ABS[s]=SR_ABS[s]||{}), rowp=SR_ABS[sp]||{};
  const keys=new Set([s,sp]); for(const k in row)keys.add(k); for(const k in rowp)keys.add(k);
  for(const k of keys){ const ind=(k===s)?1:0; row[k]=(row[k]||0)+SR_A*(ind+SR_G*(rowp[k]||0)-(row[k]||0)); } }
function srValueAbs(absCls){ const row=SR_ABS[absCls]; if(!row)return 0; let v=0; for(const k in row)v+=row[k]*(RWD_ABS[k]||0); return v; }

// PLANEJAMENTO HIERARQUICO via SR: valor = efeito imediato + valor sucessor FINO + valor sucessor ABSTRATO (transfere).
function planSR(cls, acts){
  acts=acts||ACTIONS; let best=null,bv=-1e9;
  for(const a of acts){ const nc=nextClass(cls,a);
    const immed=cell(cls,a).pred;
    const fine = nc?srValue(nc):0;
    const absN = nc?deriveAbs(nc):deriveAbs(cls);
    const fineSeen = nc && SR[nc] && Object.keys(SR[nc]).length>0;
    const abs = srValueAbs(absN);
    if(!fineSeen && Math.abs(abs)>0.01) TRANSFER.applied++;   // nivel abstrato carregou um estado fino raro/novo = TRANSFERENCIA
    const v = immed + SR_G*(fine + (fineSeen?0.4:1.0)*abs);   // se o fino e desconhecido, CONFIA mais na estrutura abstrata
    if(v>bv){ bv=v; best=a; } }
  return { action:best, value:bv };
}
function transferStats(){ let rows=Object.keys(SR_ABS).length; return { abs:rows, applied:TRANSFER.applied }; }
function nextClass(cls, act){ const a=T[cls]&&T[cls][act]; if(!a) return null;
  let best=null,bv=-1; for(const nc in a){ if(a[nc]>bv){bv=a[nc];best=nc;} } return best; }
function greedyAction(cls, acts){ acts=acts||ACTIONS; let best=acts[0],bv=-1e9;
  for(const a of acts){ const p=cell(cls,a).pred; if(p>bv){bv=p;best=a;} } return best; }
// PLANEJA: simula cada 1a acao depth passos a frente (rollout greedy, desconto) e devolve a de maior retorno previsto
function planAhead(cls, depth, acts){
  acts=acts||ACTIONS; depth=depth||3; let best=null,bestVal=-1e9;
  for(const a0 of acts){
    let val=0, c=cls, a=a0;
    for(let d=0; d<depth; d++){
      val += cell(c,a).pred * Math.pow(0.8,d);      // efeito previsto, descontado no tempo
      const nc=nextClass(c,a); if(!nc) break; c=nc; a=greedyAction(c,acts);   // continua imaginando greedy
    }
    if(val>bestVal){ bestVal=val; best=a0; }
  }
  return { action:best, value:bestVal };
}

function stats(){
  let cells=0, learned=0, classes=Object.keys(M).length;
  for(const c in M) for(const a in M[c]){ cells++; if(M[c][a].n>=3 && M[c][a].errSlow<0.6) learned++; }
  let trans=0; for(const c in T) for(const a in T[c]) trans+=Object.keys(T[c][a]).length;
  let srLinks=0; for(const c in SR) srLinks+=Object.keys(SR[c]).length;
  return { classes, cells, learned, trans, srRows:Object.keys(SR).length, srLinks };
}
// melhor relacao causal aprendida (p/ inspecao/chat): (classe,acao)->efeito previsto
function topRelations(k=6){
  const rows=[];
  for(const c in M) for(const a in M[c]){ const e=M[c][a]; if(e.n>=3) rows.push({cls:c,act:a,pred:+e.pred.toFixed(2),n:e.n,err:+e.errSlow.toFixed(2)}); }
  rows.sort((x,y)=>Math.abs(y.pred)-Math.abs(x.pred)); return rows.slice(0,k);
}

module.exports={ ACTIONS, classify, predict, observe, learningProgress, curiousAction, valuedAction,
                 load, save, stats, topRelations, getM:()=>M,
                 observeTransition, nextClass, planAhead, planSR, srValue,
                 neuro, prioritizedReplay, transferStats, srValueAbs, deriveAbs, injectRule };
