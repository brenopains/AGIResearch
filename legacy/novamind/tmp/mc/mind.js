// mind.js — TEORIA DA MENTE + PLANEJAMENTO HIERARQUICO (gaps #4/#8 do mapa AGI).
// (1) ToM: cada bot modela os OUTROS (acao/estado recente) e usa p/ COORDENAR (imitar quem prospera,
//     evitar redundancia). Pesquisa MindForge: ToM acelera aprendizado cultural.
// (2) Planejamento hierarquico: uma META (marco da tech tree) vira um PLANO = sequencia de sub-passos,
//     executados ao longo de varios ticks, re-planejando ao concluir/falhar. Comportamento de LONGO PRAZO.
// Tudo BOUNDED (sem crescimento ilimitado de memoria).

// ===== TEORIA DA MENTE =====
// modelo dos outros: shared, atualizado por quem observa. otherModel[name] = {act, food, hp, t}
const OTHERS = {};
function observeOther(name, act, food, hp){ OTHERS[name]={act,food:food,hp:hp,t:Date.now()};
  // bound: remove modelos velhos (>60s)
  const now=Date.now(); for(const k in OTHERS){ if(now-OTHERS[k].t>60000) delete OTHERS[k]; } }
// prediz o que um aliado vai fazer (modelo simples: repete a ultima acao observada)
function predictOther(name){ const o=OTHERS[name]; return o?o.act:null; }
// quem esta prosperando mais (maior food+hp) entre os modelados -> alvo de imitacao (aprendizado social)
function thrivingOther(exclude){ let best=null,bv=-1; for(const k in OTHERS){ if(k===exclude) continue; const o=OTHERS[k]; const v=(o.food||0)+(o.hp||0); if(v>bv){bv=v;best={name:k,...o};} } return best; }
function tomStats(){ return Object.keys(OTHERS).length; }

// ===== PLANEJAMENTO HIERARQUICO =====
// plano por bot: {goal, steps:[...], i, born}. cada step e uma ACAO de alto nivel (vocab do worldmodel).
// decompoe um marco da tech tree em sub-passos concretos (heuristica; sem LLM p/ ser barato e confiavel).
const MILESTONE_PLAN = {
  'madeira':            ['explorar','cavar','cavar'],                 // achar arvore e cavar madeira
  'mesa_trabalho':      ['craftar'],                                  // planks->table
  'ferramenta_madeira': ['craftar','craftar'],                        // sticks + tool
  'pedra':              ['cavar','cavar','cavar'],                    // minerar pedra
  'ferramenta_pedra':   ['craftar'],
  'forno':              ['craftar'],
  'comida_cozida':      ['cacar','craftar'],
  'carvao':             ['cavar','cavar'],
  'ferro':              ['cavar','cavar','cavar'],
  'ferramenta_ferro':   ['craftar'],
};
function makePlan(bot, milestone){
  const steps = (MILESTONE_PLAN[milestone]||['explorar']).slice();
  bot.brain._plan = { goal: milestone, steps, i:0, born:Date.now() };
  return bot.brain._plan;
}
function planStep(bot){ const p=bot.brain._plan; if(!p) return null;
  if(p.i>=p.steps.length || Date.now()-p.born>60000){ bot.brain._plan=null; return null; }   // concluiu/expirou
  return p.steps[p.i]; }
function advancePlan(bot, success){ const p=bot.brain._plan; if(!p) return;
  if(success) p.i++;                       // avanca so se o passo deu certo (senao repete = re-tentativa)
  if(p.i>=p.steps.length) bot.brain._plan=null; }
function hasPlan(bot){ return !!bot.brain._plan; }

// ===== ENSINO GUIADO POR TEORIA DA MENTE (porte do H99/MiniMax: cultura+ToM = 5x o individuo) =====
// Em vez de transmitir cultura ALEATORIA, o tutor MODELA o que o aprendiz ainda NAO sabe mas esta PRONTO
// pra aprender (a FRONTEIRA dele) e ensina EXATAMENTE isso. Aqui a fronteira concreta e: (1) conhecimento
// causal de ameaca (qual mob machuca) que o tutor aprendeu e o aprendiz nao, (2) vocabulario (simbolo cunhado).
// Isso transfere conhecimento SOCIALMENTE (sem o aprendiz precisar apanhar) — o multiplicador 3.4x do H99.
function teachableFrontier(tutorBrain, learnerBrain, COINED){
  if(!tutorBrain||!learnerBrain) return null;
  const tt=tutorBrain.threat||{}, lt=learnerBrain.threat||{};
  // (1) AMEACA: tutor SABE que ent machuca (threat>0.5) e aprendiz NAO sabe (gap grande) -> fronteira ideal
  let best=null, bestGap=0.5;
  for(const ent in tt){ const tv=tt[ent]||0, lv=lt[ent]||0; const gap=tv-lv;
    if(tv>0.5 && gap>bestGap){ bestGap=gap; best={kind:'threat', ent, value:tv,
      symbol:(COINED&&Object.keys(COINED).find(w=>COINED[w]==='ENT:'+ent))||null}; } }
  if(best) return best;
  // (2) VOCABULARIO: simbolo que o tutor usa e o aprendiz ainda nao tem ancorado
  const tb=tutorBrain.bound||{}, lb=learnerBrain.bound||{};
  for(const w in tb){ if(tb[w] && !lb[w]) return {kind:'vocab', symbol:w, concept:tb[w]}; }
  return null;
}
// EFETIVA o ensino no aprendiz: ele ADQUIRE o conhecimento da fronteira (sem apanhar). Retorna o que foi ensinado.
function applyTeaching(learnerBrain, frontier){
  if(!learnerBrain||!frontier) return false;
  if(frontier.kind==='threat'){
    learnerBrain.threat=learnerBrain.threat||{};
    const had=(learnerBrain.threat[frontier.ent]||0)>0.5;
    learnerBrain.threat[frontier.ent]=Math.max(learnerBrain.threat[frontier.ent]||0, frontier.value*0.7); // herda parcial
    if(frontier.symbol){ learnerBrain.bound=learnerBrain.bound||{}; learnerBrain.bound[frontier.symbol]='ENT:'+frontier.ent; }
    return !had;   // true se era fronteira nova (aprendiz nao sabia)
  } else if(frontier.kind==='vocab'){
    learnerBrain.bound=learnerBrain.bound||{}; const had=!!learnerBrain.bound[frontier.symbol];
    learnerBrain.bound[frontier.symbol]=frontier.concept; return !had;
  }
  return false;
}

module.exports={ observeOther, predictOther, thrivingOther, tomStats,
                 makePlan, planStep, advancePlan, hasPlan,
                 teachableFrontier, applyTeaching };
