// skills.js — FUSAO EMBODIDA: a Ponte dentro do agente do Minecraft.
// Habilidade = programa de acao (JS) que o bot ADQUIRE da experiencia. O bot tenta resolver a
// situacao com a biblioteca (recuperacao associativa por fingerprint da situacao) SEM LLM; so quando
// TRAVA (situacao saliente sem habilidade boa) chama o LLM (Venice) p/ MINTAR uma habilidade nova.
// Habilidade que melhora sobrevivencia (Δhp+food) se FORTALECE; ruim decai. Biblioteca PERSISTE em
// disco e e COMPARTILHADA pela colonia (ratchet cultural H99) -> sobrevive a morte = transmissao geracional.
// SINAL DE AGI: fracao de situacoes resolvidas SEM LLM cresce ao longo do tempo (compounding embodido).
const fs = require('fs');
const path = require('path');

const LIB_PATH = path.join(__dirname, 'skills_library.json');
const ENV_PATH = path.join(__dirname, '..', '..', 'sci', '.env');   // novamind/sci/.env

// --- chave Venice (do .env do sci) ---
let VENICE_KEY = process.env.VENICE_API_KEY || '';
try { const e = fs.readFileSync(ENV_PATH,'utf8'); const m=e.match(/VENICE_API_KEY=(.+)/); if(m && !VENICE_KEY) VENICE_KEY=m[1].trim(); } catch(e){}
const VENICE_URL = 'https://api.venice.ai/api/v1/chat/completions';
const MODEL = process.env.SKILL_MODEL || 'zai-org-glm-4.7';   // rapido p/ tempo-real

// --- contadores globais do SINAL (compounding) ---
const STATS = { used_no_llm:0, minted_llm:0, salient:0, skill_reward:0, mint_fail:0,
                minted_compose:0, minted_mutate:0, chunked:0,   // habilidades NOVAS criadas SEM LLM (lacuna #3) + rotinas abstraidas da experiencia
                window:[] /* historico p/ ver a curva subir */ };

// ====== FINGERPRINT DA SITUACAO (world-model embodido) ======
function situationFP(bot){
  const e=bot.entity; if(!e) return null;
  const hp=bot.health??20, food=bot.food??20;
  const tod=bot.time?bot.time.timeOfDay:0; const night=(tod>13000&&tod<23000)?1:0;
  const pos=e.position;
  let hostD=64, foodD=64, allies=0;
  const HOST=new Set(['zombie','skeleton','spider','creeper','enderman','witch']);
  const FOODM=new Set(['cow','pig','chicken','sheep','rabbit']);
  for(const id in (bot.entities||{})){ const o=bot.entities[id]; if(!o||!o.position||o===e) continue;
    const d=o.position.distanceTo(pos);
    if(o.name&&HOST.has(o.name)) hostD=Math.min(hostD,d);
    else if(o.name&&FOODM.has(o.name)) foodD=Math.min(foodD,d);
    else if(o.username&&o.username.startsWith('Nova_')&&d<12) allies++;
  }
  let sky=0, below='air';
  try{ const b=bot.blockAt(pos.offset(0,3,0)); sky=(b&&b.name==='air')?1:0;
       const bb=bot.blockAt(pos.offset(0,-1,0)); below=bb?bb.name:'air'; }catch(_){}
  // vetor normalizado (0..1 aprox)
  return [ hp/20, food/20, night, Math.min(hostD,32)/32, Math.min(foodD,32)/32,
           Math.min(allies,5)/5, sky, (below==='air'?1:0),
           (below&&below.includes('water')?1:0), (bot.oxygenLevel??20)/20 ];
}
function fpDist(a,b){ if(!a||!b) return 9; let s=0; for(let i=0;i<a.length;i++){ const d=(a[i]||0)-(b[i]||0); s+=d*d;} return Math.sqrt(s); }

// ====== BIBLIOTECA DE HABILIDADES (compartilhada + persistente) ======
let LIB = [];   // {name, code, fp, strength, uses, success, born}
const LIB_CAP = 120;   // poda: evita crescimento ilimitado (causava OOM)
function pruneLib(){ if(LIB.length<=LIB_CAP) return;
  const score=s=>((s.strength||0)+(s.uses||0)*0.02);
  LIB.sort((a,b)=>score(b)-score(a));
  const keep=LIB.slice(0,LIB_CAP);
  // RESERVA p/ OPCOES HIERARQUICAS: garante que ate 15 chunks/composed sobrevivam (senao sao out-competidos
  // por skills individuais e o mecanismo de opcao fica sem material — abstracao temporal precisa persistir).
  const inKeep=new Set(keep); const RES=15;
  const opts=LIB.filter(s=>(s.chunk||s.composed)&&!inKeep.has(s)).sort((a,b)=>score(b)-score(a)).slice(0,RES);
  if(opts.length){ keep.sort((a,b)=>score(a)-score(b)); for(let i=0;i<opts.length&&i<keep.length;i++) keep[i]=opts[i]; }
  LIB.length=0; LIB.push(...keep);
}
function loadLib(){ try{ LIB = JSON.parse(fs.readFileSync(LIB_PATH,'utf8')); }catch(e){ LIB=[]; } pruneLib(); return LIB; }
function saveLib(){ try{ pruneLib(); fs.writeFileSync(LIB_PATH, JSON.stringify(LIB)); }catch(e){} }
function libSize(){ return LIB.length; }

function bestSkill(fp, maxDist=0.55){
  let best=null, bd=maxDist;
  for(const s of LIB){ if((s.strength||0)<=0.15) continue; const d=fpDist(fp,s.fp);
    // score = proximidade + forca
    const eff = d - 0.15*Math.min(s.strength,3);
    if(eff<bd){ bd=eff; best=s; } }
  return best;
}

// ====== MODELO CAUSAL-RELACIONAL DE AFFORDANCES (gap #6) ======
// objeto x acao -> efeito -> valor. O bot DESCOBRE o que cada coisa faz interagindo (motor babbling de objetos).
// Codificacao preditiva: preve o efeito; SURPRESA (efeito != previsto) marca aprendizado. Persistente+compartilhado.
const AFF_PATH = path.join(__dirname, 'affordances.json');
let AFF = {};   // chave 'oak_door|activate' -> {effect, value, n, statechange}
function loadAff(){ try{ AFF=JSON.parse(fs.readFileSync(AFF_PATH,'utf8')); }catch(e){ AFF={}; } return AFF; }
function saveAff(){ try{ fs.writeFileSync(AFF_PATH, JSON.stringify(AFF)); }catch(e){} }
function recordAff(objType, action, effect, dv, statechange){
  const k=objType+'|'+action; const a=(AFF[k]=AFF[k]||{effect:'?',value:0,n:0,statechange:false});
  a.n++; a.value=a.value*0.8+(dv||0)*0.2; if(effect) a.effect=effect; if(statechange) a.statechange=true; return a;
}
function affStats(){ const ks=Object.keys(AFF); return {n:ks.length, withEffect:ks.filter(k=>AFF[k].statechange).length, sample:ks.slice(0,8)}; }
const INTERACTABLE = /door|button|lever|trapdoor|fence_gate|chest|crafting_table|furnace|bed|note_block/;

// ====== API SEGURA exposta as habilidades (a habilidade so pode fazer isto) ======
function makeAPI(bot, goals){
  const e=bot.entity; const pos=e?e.position:null;
  const HOST=new Set(['zombie','skeleton','spider','creeper','enderman','witch']);
  const FOODM=new Set(['cow','pig','chicken','sheep','rabbit']);
  const nearestOf=(pred)=>{ let best=null,bd=1e9; for(const id in (bot.entities||{})){ const o=bot.entities[id]; if(!o||!o.position||o===e) continue; if(pred(o)){ const d=o.position.distanceTo(pos); if(d<bd){bd=d;best=o;} } } return best; };
  const api = {
    hp: bot.health??20, food: bot.food??20, pos,
    night: (bot.time && bot.time.timeOfDay>13000 && bot.time.timeOfDay<23000),
    nearestHostile: ()=> nearestOf(o=>o.name&&HOST.has(o.name)),
    nearestPrey:    ()=> nearestOf(o=>o.name&&FOODM.has(o.name)),
    nearestAlly:    ()=> nearestOf(o=>o.username&&o.username.startsWith('Nova_')),
    dist: (o)=> o&&o.position?o.position.distanceTo(pos):64,
    goto: (x,y,z)=>{ try{ bot.pathfinder.setGoal(new goals.GoalNear(x,y,z,1)); }catch(_){} },
    follow: (ent,r=2)=>{ try{ bot.pathfinder.setGoal(new goals.GoalFollow(ent,r)); }catch(_){} },
    fleeFrom: (p)=>{ try{ if(!p||!pos)return; const away=pos.minus(p).normalize().scaled(8).plus(pos); bot.pathfinder.setGoal(new goals.GoalNear(away.x,pos.y,away.z,1)); }catch(_){} },
    sprint: (on)=>{ try{ bot.setControlState('sprint',!!on); }catch(_){} },
    jump: (on)=>{ try{ bot.setControlState('jump',!!on); }catch(_){} },
    attack: async (ent)=>{ try{ if(ent) await bot.attack(ent); }catch(_){} },
    eat: async ()=>{ try{ const it=bot.inventory.items().find(i=>i.name.match(/beef|porkchop|chicken|mutton|cooked|bread|apple|carrot/)); if(it){ await bot.equip(it,'hand'); await bot.consume(); return true } }catch(_){} return false; },
    digBelow: async ()=>{ try{ const b=bot.blockAt(pos.offset(0,-1,0)); if(b&&bot.canDigBlock(b)){ await bot.dig(b); return true } }catch(_){} return false; },
    digFront: async ()=>{ try{ const b=bot.blockAt(pos.offset(0,0,1)); if(b&&b.name!=='air'&&bot.canDigBlock(b)){ await bot.dig(b); return true } }catch(_){} return false; },
    placeBelow: async ()=>{ try{ const it=bot.inventory.items().find(i=>i.name.match(/dirt|stone|cobble|plank|sand|log/)); if(!it)return false; await bot.equip(it,'hand'); const ref=bot.blockAt(pos.offset(0,-1,0)); if(ref) await bot.placeBlock(ref,{x:0,y:1,z:0}); return true; }catch(_){} return false; },
    look: async (p)=>{ try{ if(p) await bot.lookAt(p.offset?p.offset(0,1,0):p); }catch(_){} },
    // ===== INTERACAO COM OBJETOS (affordances) =====
    nearestInteractable: ()=>{ try{ return bot.findBlock({matching:b=>b&&b.name&&INTERACTABLE.test(b.name), maxDistance:8}); }catch(_){ return null; } },
    // ativa um bloco (botao-direito): abre porta, aperta botao, abre bau/mesa... e DETECTA o efeito (mudou de estado?)
    activate: async (block)=>{ try{ if(!block) return {ok:false};
      // chega perto
      if(block.position.distanceTo(pos)>3){ bot.pathfinder.setGoal(new goals.GoalNear(block.position.x,block.position.y,block.position.z,2)); await new Promise(r=>setTimeout(r,1200)); }
      const before = bot.blockAt(block.position); const bprops = before&&before.getProperties?before.getProperties():{};
      await bot.activateBlock(block); await new Promise(r=>setTimeout(r,300));
      const after = bot.blockAt(block.position); const aprops = after&&after.getProperties?after.getProperties():{};
      const changed = JSON.stringify(bprops)!==JSON.stringify(aprops) || (before&&after&&before.name!==after.name);
      return {ok:true, changed, type:(before?before.name:'?'), effect: changed?'mudou_estado':'sem_efeito_visivel'};
    }catch(e){ return {ok:false}; } },
    craft: async (name)=>{ try{ const md=bot.mcData||require('minecraft-data')(bot.version); const it=md.itemsByName[name]; if(!it) return false;
      const tbl=bot.findBlock({matching:b=>b&&b.name==='crafting_table',maxDistance:6})||null;
      const recipes=bot.recipesFor(it.id,null,1,tbl); if(!recipes||!recipes.length) return false;
      await bot.craft(recipes[0],1,tbl); return true; }catch(_){ return false; } },
    equipTool: async (kind)=>{ try{ const it=bot.inventory.items().find(i=>i.name.includes(kind||'axe')); if(it){ await bot.equip(it,'hand'); return true } }catch(_){} return false; },
    hasItem: (substr)=> { try{ return bot.inventory.items().some(i=>i.name.includes(substr)); }catch(_){ return false; } },
    // ===== RACIOCINIO DE FALHA: por que a acao deu errado? (forma hipotese + sugere proximo passo + mensagem) =====
    entityAt: (ox,oy,oz)=>{ try{ const tp=pos.offset(ox,oy,oz); for(const id in (bot.entities||{})){ const e=bot.entities[id]; if(e&&e.position&&e!==bot.entity&&e.position.distanceTo(tp)<1.3) return e; } }catch(_){} return null; },
    diagnose: (where)=>{ try{   // where: 'below' | 'front'
      const off = where==='front'?[0,0,1]:[0,-1,0];
      const ent = (()=>{ const tp=pos.offset(off[0],off[1],off[2]); for(const id in (bot.entities||{})){ const e=bot.entities[id]; if(e&&e.position&&e!==bot.entity&&e.position.distanceTo(tp)<1.4) return e; } return null; })();
      if(ent){ const who = ent.username ? (ent.username.startsWith('Nova_')?'um amigo':'voce') : (ent.name||'algo'); return {reason:'bloqueado_por_entidade', by:who, msg:'tem '+who+' no caminho, vou pro lado', alt:'mover'}; }
      const blk = bot.blockAt(pos.offset(off[0],off[1],off[2]));
      if(!blk || blk.name==='air') return {reason:'nada_aqui', msg:'nada pra cavar aqui, vou procurar outro lugar', alt:'mover'};
      if(!bot.canDigBlock(blk)) return {reason:'bloco_duro', block:blk.name, msg:blk.name+' e duro, preciso de ferramenta melhor', alt:'craftar'};
      return {reason:'desconhecido', msg:'travei aqui, vou tentar outra coisa', alt:'mover'};
    }catch(_){ return {reason:'erro', msg:'algo deu errado', alt:'mover'}; } },
    moveAside: ()=>{ try{ const a=Math.random()*6.28; bot.pathfinder.setGoal(new goals.GoalNear(pos.x+Math.cos(a)*2.5, pos.y, pos.z+Math.sin(a)*2.5, 1)); }catch(_){} },
    // ===== BOOTSTRAP DA TECH TREE (crafting robusto e dependente do inventario) =====
    getWood: async ()=>{ try{ const tree=bot.findBlock({matching:b=>b&&b.name&&b.name.endsWith('_log'),maxDistance:24});
      if(!tree) return false; if(tree.position.distanceTo(pos)>2.5){ bot.pathfinder.setGoal(new goals.GoalNear(tree.position.x,tree.position.y,tree.position.z,1)); await new Promise(r=>setTimeout(r,1500)); }
      const t2=bot.findBlock({matching:b=>b&&b.name&&b.name.endsWith('_log'),maxDistance:4}); if(t2){ await bot.dig(t2); return true } }catch(_){} return false; },
    mineStone: async ()=>{ try{ // precisa de picareta; junta VARIAS cobblestone (>=3 p/ ferramentas de pedra -> destrava tech tree)
      const pick=bot.inventory.items().find(i=>i.name.includes('pickaxe')); if(pick) await bot.equip(pick,'hand');
      const isStone=n=>/stone|cobble|andesite|diorite|granite|deepslate|tuff/.test(n);
      const have=()=>bot.inventory.items().filter(i=>/cobblestone|^stone$/.test(i.name)).reduce((s,i)=>s+i.count,0);
      let mined=0;
      for(let k=0;k<12 && have()<12;k++){   // junta cobblestone p/ ferramentas de pedra (3) E forno (8) -> destrava marco 6
        const sb=bot.findBlock({matching:b=>b&&isStone(b.name),maxDistance:16});
        if(sb && bot.canDigBlock(sb)){ if(sb.position.distanceTo(pos)>3){ bot.pathfinder.setGoal(new goals.GoalNear(sb.position.x,sb.position.y,sb.position.z,1)); await new Promise(r=>setTimeout(r,900)); }
          const s2=bot.blockAt(sb.position); if(s2&&isStone(s2.name)&&bot.canDigBlock(s2)){ await bot.dig(s2); mined++; await new Promise(r=>setTimeout(r,120)); continue; } }
        // senao DESCE ate a pedra
        const below=bot.blockAt(pos.offset(0,-1,0)); if(!below||below.name==='air'||!bot.canDigBlock(below)) break;
        await bot.dig(below); mined++; await new Promise(r=>setTimeout(r,120));
      }
      return mined>0; }catch(_){} return false; },
    techCraft: async ()=>{ try{ const md=bot.mcData||require('minecraft-data')(bot.version);
      const inv=()=>bot.inventory.items(); const has=(re)=>inv().find(i=>re.test(i.name)); const cnt=(re)=>inv().filter(i=>re.test(i.name)).reduce((s,i)=>s+i.count,0);
      const craftIt=async(name,table)=>{ const it=md.itemsByName[name]; if(!it)return false; const r=bot.recipesFor(it.id,null,1,table||null); if(!r||!r.length)return false; try{ await bot.craft(r[0],1,table||null); return true; }catch(_){ return false; } };
      // garante uma MESA por perto: acha, senao coloca a do inventario, senao crafta uma de planks e coloca
      const ensureTable=async()=>{ let t=bot.findBlock({matching:b=>b&&b.name==='crafting_table',maxDistance:4}); if(t) return t;
        if(!has(/crafting_table/)){ if(cnt(/_planks$/)<4 && has(/_log$/)) await craftIt(has(/_log$/).name.replace('_log','_planks')); if(cnt(/_planks$/)>=4) await craftIt('crafting_table'); }
        if(has(/crafting_table/)){ try{ await bot.equip(has(/crafting_table/),'hand'); for(const off of [[1,0],[-1,0],[0,1],[0,-1]]){ const r2=bot.blockAt(pos.offset(off[0],-1,off[1])); if(r2&&r2.name!=='air'){ try{ await bot.placeBlock(r2,{x:0,y:1,z:0}); break }catch(_){} } } t=bot.findBlock({matching:b=>b&&b.name==='crafting_table',maxDistance:4}); }catch(_){} }
        return t; };
      let made=false, last=false;
      for(let step=0; step<6; step++){            // LOOP: avanca o MAXIMO de sub-passos possivel numa chamada (destrava a tech tree)
        let did=false;
        if(has(/_log$/) && cnt(/_planks$/)<8){ if(await craftIt(has(/_log$/).name.replace('_log','_planks'))){ did=true; last='planks'; } }
        else if(cnt(/_planks$/)>=2 && cnt(/^stick$/)<4){ if(await craftIt('stick')){ did=true; last='sticks'; } }
        else { const tbl=await ensureTable();
          if(tbl){
            if(!has(/_pickaxe/) && cnt(/_planks$/)>=3 && cnt(/^stick$/)>=2){ if(await craftIt('wooden_pickaxe',tbl)){ did=true; last='wooden_pickaxe'; } }
            else if(cnt(/cobblestone/)>=3 && !has(/stone_pickaxe/) && cnt(/^stick$/)>=2){ if(await craftIt('stone_pickaxe',tbl)){ did=true; last='stone_pickaxe'; } }
            else if(cnt(/cobblestone/)>=3 && !has(/stone_axe/) && cnt(/^stick$/)>=2){ if(await craftIt('stone_axe',tbl)){ did=true; last='stone_axe'; } }
            else if(cnt(/cobblestone/)>=3 && !has(/stone_sword/) && cnt(/^stick$/)>=2){ if(await craftIt('stone_sword',tbl)){ did=true; last='stone_sword'; } }
            else if(cnt(/cobblestone/)>=8 && !has(/furnace/)){ if(await craftIt('furnace',tbl)){ did=true; last='furnace'; } }
          }
        }
        if(did){ made=true; await new Promise(r=>setTimeout(r,120)); } else break;
      }
      return made?last:false;
    }catch(e){} return false; },
    say: (m)=>{ try{ if(m && (!bot._sk_say||Date.now()-bot._sk_say>4000)){ bot._sk_say=Date.now(); bot.chat(String(m).slice(0,80)); } }catch(_){} },
    wait: (ms)=> new Promise(r=>setTimeout(r,Math.min(ms||300,1500))),
  };
  // act(verbo): executa uma ACAO de alto nivel pelo despachante -> permite que CHUNKS (rotinas abstraidas) sejam replayados
  api.act = async (verb)=>{ try{ return await dispatch(api, verb); }catch(_){ return false; } };
  return api;
}

// ====== COMPILADOR ROBUSTO: aceita CORPO puro OU funcao completa (arrow/function) ======
function compileSkillFn(code){
  const c=(code||'').trim();
  // forma 1: e uma funcao/arrow completa -> chama com api
  if(/^\s*(async\s*)?(\(|function)/.test(c) && /=>|function/.test(c.slice(0,40))){
    try{ const f=new Function('api','return ('+c+')(api)'); return f; }catch(e){}
  }
  // forma 2: corpo puro -> embrulha
  try{ const f=new Function('api','return (async()=>{ '+c+' \n})()'); return f; }catch(e){}
  return null;
}

// ====== EXEC SEGURA da habilidade (timeout + try/catch) ======
async function runSkill(skill, bot, goals){
  const api = makeAPI(bot, goals);
  const fn = compileSkillFn(skill.code);
  if(!fn) return false;
  try {
    await Promise.race([ Promise.resolve(fn(api)), new Promise((_,rej)=>setTimeout(()=>rej('timeout'),4000)) ]);
    return true;
  } catch(e){ return false; }
}

// ====== MINT via LLM (Venice) — so quando TRAVA ======
const API_DOC = `Voce controla um agente no Minecraft. Escreva o CORPO de uma funcao async (sem assinatura, sem 'def', JS) que usa SO o objeto 'api':
api.hp, api.food, api.pos, api.night (bool)
api.nearestHostile(), api.nearestPrey(), api.nearestAlly() -> entidade ou null (tem .position, .name)
api.dist(ent) -> distancia
await api.attack(ent); await api.eat(); await api.digBelow(); await api.digFront(); await api.placeBelow();
api.goto(x,y,z); api.follow(ent,r); api.fleeFrom(pos); api.sprint(bool); api.jump(bool); await api.look(pos); api.say(txt); await api.wait(ms)
Regras: codigo curto e seguro (sem loops infinitos, sem require/import/fs). Resolva a SITUACAO descrita. Responda APENAS o codigo JS do corpo, sem cercas.`;

let _mintInFlight = 0;
async function mintSkill(situationDesc){
  if(!VENICE_KEY) return null;
  if(_mintInFlight >= 2) return null;   // limita concorrencia
  _mintInFlight++;
  try{
    const body = { model: MODEL, temperature: 0.3, max_tokens: 500,
      messages: [ {role:'system', content: API_DOC}, {role:'user', content: 'SITUACAO: '+situationDesc+'\n\nEscreva o corpo da habilidade (JS, so o codigo):'} ],
      venice_parameters: { disable_thinking: true, strip_thinking_response: true } };
    const r = await fetch(VENICE_URL, { method:'POST', headers:{'Content-Type':'application/json','Authorization':'Bearer '+VENICE_KEY}, body: JSON.stringify(body) });
    if(!r.ok){ STATS.mint_fail++; return null; }
    const d = await r.json();
    let code = (d.choices && d.choices[0] && d.choices[0].message && (d.choices[0].message.content||d.choices[0].message.reasoning_content)) || '';
    code = code.replace(/```(?:js|javascript)?/g,'').replace(/```/g,'').trim();
    if(!code || code.length<8 || /require|import|while\s*\(\s*true|for\s*\(\s*;;/.test(code)){ STATS.mint_fail++; return null; }
    // valida que compila
    if(!compileSkillFn(code)){ STATS.mint_fail++; return null; }
    return code;
  }catch(e){ STATS.mint_fail++; return null; }
  finally{ _mintInFlight--; }
}

function describeSituation(api){
  const h=api.nearestHostile();
  const bits=[`hp=${Math.round(api.hp)}`,`comida=${Math.round(api.food)}`, api.night?'noite':'dia'];
  if(h) bits.push(`AMEACA ${h.name} a ${Math.round(api.dist(h))} blocos`);
  if(api.food<8) bits.push('FOME CRITICA');
  if(api.hp<8) bits.push('VIDA CRITICA');
  const p=api.nearestPrey(); if(p&&api.food<16) bits.push(`presa ${p.name} a ${Math.round(api.dist(p))}`);
  return bits.join(', ');
}

// nomeia a habilidade pela situacao (sem outra chamada LLM)
function nameFor(api){
  const h=api.nearestHostile();
  if(h && api.dist(h)<10) return 'lidar_'+h.name;
  if(api.food<8) return 'fome_critica';
  if(api.hp<8) return 'vida_critica';
  if(api.night) return 'sobreviver_noite';
  return 'situacao_geral';
}

// EMERGENCIA REAL (reflexo de sobrevivencia tem prioridade) -> so perigo IMEDIATO
function isSalient(api){
  const h=api.nearestHostile();
  return (h && api.dist(h)<6) || api.food<5 || api.hp<7;
}
// FORA de emergencia -> AUTONOMIA: persegue OBJETIVO PROPRIO. Este e o estado DOMINANTE (vida dirigida por metas).
// Se faminto-moderado, o projeto que o LLM propoe sera forragear; se farto, construir/explorar/socializar.
function isCalm(api){
  const h=api.nearestHostile();
  return api.hp>=7 && !(h && api.dist(h)<6);
}

// ====== AUTONOMIA: objetivo AUTO-GERADO pelo proprio agente (gap #8: motivacao propria) ======
const GOAL_DOC = `Voce e um agente AUTONOMO no Minecraft, vivo e seguro agora. Em vez de so sobreviver, DECIDA o que VOCE quer fazer e persiga. Escreva o CORPO de uma funcao async JS (so 'api') que realiza UM projeto proprio concreto e util AGORA:
- explorar longe (api.goto a coords distantes), procurar recursos, minerar (api.digFront/digBelow em sequencia curta), erguer estrutura (api.placeBelow repetido = pilar/abrigo), juntar comida proativa (caçar presa: api.follow+api.attack+api.eat), socializar (ir ate aliado api.follow(api.nearestAlly())), etc.
api: api.pos, api.food, api.hp, api.nearestPrey(), api.nearestAlly(), api.dist(e), api.goto(x,y,z), api.follow(e,r), await api.attack(e), await api.eat(), await api.digFront(), await api.digBelow(), await api.placeBelow(), api.sprint(b), api.jump(b), await api.look(p), api.say(txt), await api.wait(ms).
Curto, seguro, SEM loops infinitos/require/import. Mostre INICIATIVA (faça algo, nao só esperar). Responda APENAS o codigo JS do corpo.`;

async function mintGoalSkill(stateDesc){
  if(!VENICE_KEY) return null;
  if(_mintInFlight>=2) return null;
  _mintInFlight++;
  try{
    const body={ model:MODEL, temperature:0.6, max_tokens:500,
      messages:[{role:'system',content:GOAL_DOC},{role:'user',content:'SEU ESTADO: '+stateDesc+'\n\nDecida e escreva a habilidade do seu projeto (so o codigo JS):'}],
      venice_parameters:{disable_thinking:true,strip_thinking_response:true} };
    const r=await fetch(VENICE_URL,{method:'POST',headers:{'Content-Type':'application/json','Authorization':'Bearer '+VENICE_KEY},body:JSON.stringify(body)});
    if(!r.ok){STATS.mint_fail++;return null;}
    const d=await r.json();
    let code=(d.choices&&d.choices[0]&&d.choices[0].message&&(d.choices[0].message.content||d.choices[0].message.reasoning_content))||'';
    code=code.replace(/```(?:js|javascript)?/g,'').replace(/```/g,'').trim();
    if(!code||code.length<8||/require|import|while\s*\(\s*true|for\s*\(\s*;;/.test(code)){STATS.mint_fail++;return null;}
    if(!compileSkillFn(code)){STATS.mint_fail++;return null;}
    return code;
  }catch(e){STATS.mint_fail++;return null;}
  finally{_mintInFlight--;}
}
function describeState(api, inv){
  return [`hp=${Math.round(api.hp)}`,`comida=${Math.round(api.food)}`, api.night?'noite':'dia',
          `inventario=[${(inv||[]).slice(0,6).join(',')||'vazio'}]`,
          api.nearestAlly()?`aliado a ${Math.round(api.dist(api.nearestAlly()))} blocos`:'sozinho'].join(', ');
}
function goalName(){ return 'projeto_'+(Math.floor(Date.now()/1000)%10000); }

// ====== AUTO-VERIFICACAO DE SKILL (Voyager pilar 3): refina habilidade que FALHA repetidamente ======
// quando uma skill da resultado ruim varias vezes, o LLM REESCREVE uma versao corrigida (feedback de execucao).
async function refineSkill(skill, situationDesc){
  if(!VENICE_KEY || _mintInFlight>=2) return null;
  _mintInFlight++;
  try{
    const body={ model:MODEL, temperature:0.3, max_tokens:500,
      messages:[{role:'system',content:API_DOC},
                {role:'user',content:`Esta habilidade vem FALHANDO (resultado ruim) na situacao: ${situationDesc}\n\nCODIGO ATUAL:\n${skill.code}\n\nReescreva CORRIGINDO (mesma intencao, melhor execucao). APENAS o codigo JS do corpo:`}],
      venice_parameters:{disable_thinking:true,strip_thinking_response:true} };
    const r=await fetch(VENICE_URL,{method:'POST',headers:{'Content-Type':'application/json','Authorization':'Bearer '+VENICE_KEY},body:JSON.stringify(body)});
    if(!r.ok){return null;}
    const d=await r.json(); let code=(d.choices&&d.choices[0]&&d.choices[0].message&&(d.choices[0].message.content||d.choices[0].message.reasoning_content))||'';
    code=code.replace(/```(?:js|javascript)?/g,'').replace(/```/g,'').trim();
    if(!code||code.length<8||/require|import|while\s*\(\s*true/.test(code)||!compileSkillFn(code)) return null;
    return code;
  }catch(e){return null;} finally{_mintInFlight--;}
}

// ====== MINT SEM LLM: COMPOSICAO / MUTACAO de habilidades existentes (lacuna #3: independencia do LLM) ======
// O cerebro FABRICA habilidades novas RECOMBINANDO as que ja tem, SEM chamar o LLM. Fundamento: composicao
// motora (o cortex/cerebelo encadeia primitivas aprendidas em sequencias novas) + variacao-e-selecao (a skill
// nova entra FRACA; o loop de recompensa a fortalece se ajudar a sobreviver/progredir, e a poda a descarta se
// nao -> "teste por EXECUCAO"). Assim a biblioteca cresce parte SEM LLM -> reduz a dependencia ao longo do tempo.
function _toFn(code){ const c=(code||'').trim();
  if(/^\s*(async\s*)?(\(|function)/.test(c) && /=>|function/.test(c.slice(0,40))) return c;   // ja e funcao completa
  return 'async(api)=>{ '+c+' \n}';                                                            // corpo -> embrulha
}
let _composeSeq=0;
// COMPOSICAO: pega 2 habilidades PROVADAS (forca alta) e encadeia em SEQUENCIA (faz A, espera, faz B) = repertorio novo
function mintByComposition(){
  // RECURSIVO: composicoes e chunks tambem podem ser pais (torre de abstracao), mas com guarda de TAMANHO p/ limitar profundidade
  const proven=LIB.filter(s=>(s.strength||0)>0.4 && s.code && s.code.length<450);
  if(proven.length<2) return null;
  const a=proven[_composeSeq%proven.length], b=proven[(_composeSeq*7+3)%proven.length]; _composeSeq++;
  if(!a||!b||a===b) return null;
  const code='async(api)=>{ const __a=('+_toFn(a.code)+'); const __b=('+_toFn(b.code)+'); try{await __a(api)}catch(_){}; await api.wait(150); try{await __b(api)}catch(_){} }';
  if(!compileSkillFn(code)) return null;
  const fp=(a.fp&&b.fp)?a.fp.map((x,i)=>((x||0)+(b.fp[i]||0))/2):(a.fp||b.fp);
  return { name:'comp_'+(a.name||'a').slice(0,6)+'+'+(b.name||'b').slice(0,6), code, fp, strength:0.45, uses:0, success:0, born:Date.now(), composed:true, parents:[a.name,b.name] };
}
let _mutSeq=0;
// MUTACAO: pega 1 habilidade e perturba uma CONSTANTE numerica (variacao local), validando compilacao
function mintByMutation(){
  const cand=LIB.filter(s=>(s.strength||0)>0.4 && s.code && /\d/.test(s.code) && s.code.length<700);
  if(!cand.length) return null;
  const s=cand[_mutSeq%cand.length];
  const nums=[...s.code.matchAll(/\b(\d{1,4})\b/g)]; _mutSeq++; if(!nums.length) return null;
  const pick=nums[(_mutSeq*3)%nums.length], orig=+pick[1];
  const mut=Math.max(1,Math.round(orig*[0.5,1.5,2,0.75][_mutSeq%4])); if(mut===orig) return null;
  const code=s.code.slice(0,pick.index)+mut+s.code.slice(pick.index+pick[1].length);
  if(!compileSkillFn(code)) return null;
  return { name:'mut_'+(s.name||'s').slice(0,8), code, fp:s.fp, strength:0.4, uses:0, success:0, born:Date.now(), mutated:true, parent:s.name };
}
// FABRICA SEM LLM: tenta composicao, senao mutacao. Retorna skill nova (validada por compilacao) ou null.
function mintNoLLM(){
  let sk=mintByComposition(); if(sk){ STATS.minted_compose++; return sk; }
  sk=mintByMutation();        if(sk){ STATS.minted_mutate++;  return sk; }
  return null;
}

// ====== CHUNKING: ABSTRACAO HIERARQUICA DE ROTINAS A PARTIR DA EXPERIENCIA (sem LLM) ======
// O cerebro descobre SUAS PROPRIAS sub-rotinas uteis: uma SEQUENCIA de acoes que REPETIDAMENTE precede recompensa
// e ABSTRAIDA num NOVO skill nomeado (ex: chunk_cavar>craftar), que vira uma PRIMITIVA composavel (recursivo).
// Fundamento: chunking (SOAR/Newell), descoberta de OPCOES em RL hierarquico, formacao de HABITO nos ganglios da base
// (de deliberado->automatico). E o mecanismo da GENERATIVIDADE ABERTA: a biblioteca vira uma TORRE de capacidades
// cada vez mais abstratas, construida pela propria experiencia. NENHUMA chamada de LLM.
const DISPATCH_VERBS=new Set(['comer','cacar','fugir','cavar','construir','ativar_objeto','craftar','explorar','seguir_aliado','observar']);
const CHUNKS={};   // 'cavar>craftar' -> {n, reward, abstracted}
function normVerb(lastAction){ if(!lastAction) return null;
  const m=String(lastAction).match(/(comer|cacar|fugir|cavar|construir|ativar_objeto|craftar|explorar|seguir_aliado|observar)/);
  return m?m[1]:null; }
// observa o TRACE recente (verbos) + recompensa; quando uma sequencia recorre o suficiente antes de recompensa, ABSTRAI
function observeChunk(verbs, reward, fp){
  if(!verbs || verbs.length<2 || reward<2) return null;
  const seq=verbs.slice(-3).filter(v=>DISPATCH_VERBS.has(v)); if(seq.length<2) return null;
  const key=seq.join('>'); const c=(CHUNKS[key]=CHUNKS[key]||{n:0,reward:0,abstracted:false});
  c.n++; c.reward=c.reward*0.7+reward*0.3;
  if(c.n>=2 && !c.abstracted){            // recorreu >=2x precedendo recompensa -> vira rotina nomeada
    c.abstracted=true;
    const code='async(api)=>{ '+seq.map(v=>`await api.act('${v}'); await api.wait(120);`).join(' ')+' }';
    if(!compileSkillFn(code)) return null;
    STATS.chunked=(STATS.chunked||0)+1;
    return { name:'chunk_'+key, code, fp, strength:0.6, uses:0, success:0, born:Date.now(), chunk:true, seq };
  }
  return null;
}
function chunkStats(){ const ks=Object.keys(CHUNKS); return {seen:ks.length, abstracted:ks.filter(k=>CHUNKS[k].abstracted).length}; }

// ====== MINERAÇÃO DE ARQUÉTIPOS (COMPRESSÃO RECURSIVA / MDL) ======
// Uma camada ACIMA da Composição (Lacuna #3 resolvida profundamente).
// O sistema analisa suas próprias abstrações (Chunks) buscando regularidades (o Z do A+B).
// Se o agente faz A>C, B>C, D>C... "C" torna-se um Atrator (Arquétipo).
// O arquétipo abstrai a variabilidade: "Condição Preparatória Genérica" -> C.
const ARCHETYPES = {};
function mineArchetypes() {
  const ks = Object.keys(CHUNKS).filter(k=>CHUNKS[k].abstracted);
  if(ks.length < 4) return null;
  // Acha padrões A>B onde B é muito comum (ex: craftar, comer)
  const tails = {}; const pairs = [];
  ks.forEach(k => { const parts=k.split('>'); if(parts.length===2){ const tail=parts[1]; tails[tail]=(tails[tail]||0)+1; pairs.push(parts); } });
  
  let newArch = null;
  for(const tail in tails) {
    if(tails[tail] >= 3 && !ARCHETYPES[tail]) {
      // Encontrou um Arquétipo: Várias ações diferentes convergem para 'tail'
      ARCHETYPES[tail] = true;
      const headOptions = pairs.filter(p=>p[1]===tail).map(p=>p[0]);
      // O Arquétipo "Z" tenta a melhor condição preparatória baseada na experiência, e então executa o Tail.
      const code = `async(api)=>{ 
        const opts = ${JSON.stringify(headOptions)};
        const prep = opts[Math.floor(Math.random()*opts.length)]; 
        await api.act(prep); 
        await api.wait(150); 
        await api.act('${tail}'); 
      }`;
      if(!compileSkillFn(code)) continue;
      newArch = { name:'archetype_prep_'+tail, code, fp:null, strength:0.8, uses:0, success:0, born:Date.now(), archetype:true, tail };
      LIB.push(newArch);
      saveLib();
      STATS.chunked = (STATS.chunked||0) + 1;
      break; // Minera 1 por vez
    }
  }
  return newArch;
}
// OPCAO HIERARQUICA (Sutton-Precup-Singh): uma rotina temporalmente-estendida (chunk forte) tratada como UMA acao.
// Retorna o chunk de maior forca p/ ser selecionado como UNIDADE pelo planejador/workspace (abstracao temporal).
function bestOption(){ let best=null,bv=0.3; for(const s of LIB){ if((s.chunk||s.composed) && (s.strength||0)>bv){ bv=s.strength; best=s; } } return best; }

// CURIOSIDADE DE AFFORDANCE: ha um objeto interativo NOVO (acao ainda nao tentada) por perto?
function novelInteractable(api){
  const b = api.nearestInteractable && api.nearestInteractable();
  if(!b) return null;
  const k = b.name+'|activate';
  if(AFF[k] && AFF[k].n>=2) return null;   // ja explorou esse tipo o suficiente
  return b;
}
// motor babbling de objeto: tenta ativar o objeto novo e APRENDE o que ele faz
async function tryAffordance(api, block){
  const r = await api.activate(block);
  if(r && r.ok){ recordAff(r.type||block.name, 'activate', r.effect, 0, r.changed); saveAff();
    return { learned:true, type:r.type||block.name, changed:!!r.changed }; }
  return { learned:false };
}

// DESPACHANTE: executa uma ACAO de alto nivel do modelo de mundo usando as capacidades reais
async function dispatch(api, action){
  try{
    switch(action){
      case 'comer': return await api.eat();
      case 'cacar': { const p=api.nearestPrey(); if(p){ if(api.dist(p)<2.5){ await api.attack(p); await api.wait(400); await api.eat(); } else api.follow(p,1.5); return true } return false; }
      case 'fugir': { const h=api.nearestHostile(); if(h){ api.sprint(true); api.fleeFrom(h.position); return true } return false; }
      case 'cavar': {  // mineracao INTELIGENTE: sem madeira -> derruba arvore; com picareta -> minera pedra; senao cava
        if(!api.hasItem('log') && !api.hasItem('planks')){ if(await api.getWood()) return true }
        if(api.hasItem('pickaxe')){ if(await api.mineStone()) return true }
        const ok=(await api.getWood()) || (await api.digFront()) || (await api.digBelow())
        if(!ok){ const dg=api.diagnose('below'); api.say(dg.msg); STATS.diagnoses=(STATS.diagnoses||0)+1; STATS.lastDiag=dg.reason
          if(dg.alt==='mover') api.moveAside(); }                            // RACIOCINIO: por que falhou? comunica e adapta
        return ok; }
      case 'construir': { const ok=await api.placeBelow()
        if(!ok){ const dg=api.diagnose('below'); api.say(dg.reason==='nada_aqui'?'nao consigo colocar bloco aqui':dg.msg); STATS.diagnoses=(STATS.diagnoses||0)+1; STATS.lastDiag=dg.reason
          if(dg.reason==='bloqueado_por_entidade') api.moveAside(); }          // alguem/algo ocupa o lugar -> sai do caminho
        return ok; }
      case 'ativar_objeto': { const b=api.nearestInteractable(); if(b){ const r=await api.activate(b); return r&&r.ok && {block:b, res:r} } return false; }
      case 'craftar': { const made=await api.techCraft(); return made || false; }   // bootstrap robusto da tech tree
      case 'explorar': { const p=api.pos; api.goto(p.x+(Math.random()*30-15), p.y, p.z+(Math.random()*30-15)); return true; }
      case 'seguir_aliado': { const a=api.nearestAlly(); if(a){ api.follow(a,2); return true } return false; }
      case 'observar': { const a=api.nearestAlly()||api.nearestPrey(); if(a) await api.look(a.position); return true; }
    }
  }catch(e){}
  return false;
}

module.exports = { situationFP, fpDist, loadLib, saveLib, libSize, bestSkill, runSkill, dispatch,
                   mintSkill, makeAPI, describeSituation, nameFor, isSalient, isCalm,
                   mintGoalSkill, describeState, goalName, refineSkill, compileSkillFn, STATS,
                   mintByComposition, mintByMutation, mintNoLLM, observeChunk, chunkStats, mineArchetypes, normVerb, bestOption,
                   loadAff, saveAff, recordAff, affStats, novelInteractable, tryAffordance, getAff:()=>AFF,
                   getLib:()=>LIB, addSkill:(s)=>{ LIB.push(s); }, hasKey: ()=>!!VENICE_KEY };
