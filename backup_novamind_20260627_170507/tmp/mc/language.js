// language.js — ANCORAGEM DE LINGUAGEM (symbol grounding). Lacuna #2 rumo a AGI.
// Hoje os bots cunham simbolos (COINED: 'pipi'->ENT:zombie, 'xano'->CAT:perigo). Mas um simbolo so e
// linguagem DE FATO quando esta ANCORADO no CONCEITO PROFUNDO — o conhecimento causal (zumbi->dano, do
// worldmodel/threat), espacial (onde foi visto, do hipocampo) e afetivo (valencia). E quando OUVIR o simbolo
// ATIVA esse conceito no ouvinte (compreensao referencial) — transferindo conhecimento mesmo SEM ver o referente.
// Ex: ouve 'pipi' (zumbi) -> SABE que machuca -> fica alerta/com medo, sem ver o zumbi. Linguagem fazendo
// trabalho cognitivo real (um aviso que transfere o que aprendeu).
const HOST=new Set(['zombie','skeleton','spider','creeper','enderman','witch','phantom']);
const FOOD=new Set(['cow','pig','chicken','sheep','rabbit','cod','salmon']);

const ANCHORS={};   // simbolo -> {concept, n} (registro da ancoragem; COINED ja tem symbol->concept)
const CULTURAL_RULES=[]; // regras compiladas (ex: {cause: 'agua contaminada', effect: 'doenca'})

function anchor(symbol, concept){ if(!symbol||!concept) return; const a=(ANCHORS[symbol]=ANCHORS[symbol]||{concept,n:0}); a.concept=concept; a.n++; }
function injectCulturalRule(cause, effect) { CULTURAL_RULES.push({cause: cause.toLowerCase(), effect: effect.toLowerCase()}); }
function stats(){ return Object.keys(ANCHORS).length; }

// COMPREENSAO: dado um conceito ouvido, retorna a ATIVACAO ancorada usando o conhecimento APRENDIDO do ouvinte
function comprehend(bot, concept){
  const br=bot.brain||{};
  const cLow = (concept||'').toLowerCase();
  
  // 1. Checa regras culturais primeiro (zero-shot language grounding)
  for(const r of CULTURAL_RULES){
    if(cLow.includes(r.cause)){
      // Se ouviu algo que causa doenca/mal, ativa perigo
      if(r.effect.includes('doenca') || r.effect.includes('machucar') || r.effect.includes('quebrar')){
        return {type:'danger', reason: r.effect, intensity: 0.8, knows: true};
      }
      return {type:'cultural_alert', effect: r.effect, intensity: 0.5, knows: true};
    }
  }

  if(concept==='CAT:perigo') return {type:'danger', intensity:0.6};
  if(concept==='CAT:bom')    return {type:'good',   intensity:0.4};
  if(concept && concept.startsWith('ENT:')){
    const ent=concept.slice(4);
    if(HOST.has(ent)){ const lethal=(br.threat&&br.threat[ent])||0;        // conhecimento CAUSAL: quanto esse tipo ja me machucou
      const valence=(br.arch&&br.arch[ent]&&br.arch[ent].val)||0;
      return {type:'danger', ent, intensity:Math.min(1,0.3+lethal*0.15+Math.max(0,-valence)*0.3), knows: lethal>0||valence<0}; }
    if(FOOD.has(ent))  return {type:'food', ent, intensity:0.5};
    return {type:'entity', ent, intensity:0.2};
  }
  return null;
}
module.exports={ anchor, comprehend, stats, ANCHORS, injectCulturalRule };
