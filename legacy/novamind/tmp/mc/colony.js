// NovaMind — Colonia de agentes no Minecraft real (mineflayer)
// Mecanismos VALIDADOS: sobrevivencia + transmissao cultural (H99) + codigo de sinais EMERGENTE
// (H81, via chat) + correlacao acao<->sobrevivencia (bandit por contexto). Sem programar a lingua:
// os simbolos comecam aleatorios e ALINHAM por sucesso comunicativo.
const mineflayer = require('mineflayer')
const { pathfinder, Movements, goals } = require('mineflayer-pathfinder')
const fs = require('fs')
const skills = require('./skills')   // FUSAO EMBODIDA: biblioteca de habilidades adquiridas (a Ponte)
const wm = require('./worldmodel')   // MODELO DE MUNDO PREDITIVO cerebro-inspirado (curiosidade por progresso de aprendizado)
const prog = require('./progression')   // CURRICULO AUTOMATICO: progressao de capacidade (tech tree, Voyager)
const mind = require('./mind')           // TEORIA DA MENTE + PLANEJAMENTO HIERARQUICO
const hippo = require('./hippocampus')   // MEMORIA EPISODICA + ESPACIAL (place cells + replay/consolidacao)
const workspace = require('./workspace') // WORKSPACE GLOBAL: modulos competem, o saliente ignita e e transmitido
const vision = require('./vision')       // PERCEPCAO VISUAL humana-like (FOV + oclusao, nao-wallhack)
const language = require('./language')   // ANCORAGEM DE LINGUAGEM (symbol grounding + compreensao referencial)
const selfimp = require('./selfimprove') // AUTO-MELHORIA SEGURA (ajusta os proprios hiperparametros pela performance)
const forge = require('./forge')         // FORJA DE PRIMITIVAS ABERTAS: inventa acoes novas abaixo das 10 (motor babbling, sem LLM)
const pforge = require('./percept_forge') // FORJA PERCEPTUAL (porte do CC/MiniMax): inventa novas FEATURES que dividem classes irredutiveis
const curriculum = require('./curriculum') // MOTOR ABERTO (PowerPlay+OMNI): a colonia inventa seus proprios desafios escalonados
skills.loadLib(); skills.loadAff(); wm.load(); prog.load(); hippo.load(); selfimp.load(); curriculum.load()
let SPAWN_REF=null; let ACTIVE_CHALLENGE=null   // motor aberto: referencia de spawn + desafio auto-inventado ativo
console.log('[skills] biblioteca embodida: '+skills.libSize()+' habilidades | affordances: '+skills.affStats().n+' | worldmodel: '+wm.stats().cells+' celulas | LLM(Venice): '+(skills.hasKey()?'ON':'OFF'))

// PERSISTENCIA: recupera lingua/cultura/personalidade ao religar (mundo o MC ja salva)
const STATE = __dirname + '/colony_state.json'
let SAVED = {}
try { SAVED = JSON.parse(fs.readFileSync(STATE)) } catch(e) { SAVED = {} }

const N = parseInt(process.argv[2] || '8')
const VERSION = '1.20.4'
// MANUAL DE PORTUGUES (lexico ancorado): conceito do mundo <-> palavra real. Linguagem legivel, sem LLM.
const LEX = { FOOD:'comida', DANGER:'perigo', FOLLOW:'vem', DIG:'cavar', CRAFT:'fazer', NEW:'novo', AHA:'entendi', HELP:'ajuda', FRIEND:'amigo' }
const MOD = { here:'aqui', near:'perto', far:'longe' }
const REV = {}; Object.entries(LEX).forEach(([m,w])=> REV[w]=m)   // palavra->conceito (entende na hora)
// palavras-funcao que NUNCA viram conceito (artigos, copulas, deiticos, negacao, etc.)
const STOP = new Set(['o','a','os','as','um','uma','uns','umas','e','eh','é','de','do','da','dos','das','que','oque','o-que',
  'nao','não','nem','mas','sim','ou','se','isso','isto','aquilo','aqui','ali','la','lá','esse','essa','este','esta','aquele','aquela',
  'meu','minha','seu','sua','voce','você','eu','ele','ela','atras','atrás','tras','trás','frente','ao','redor','em','no','na',
  'voa','voando','voador','tipo','coisa','bloco','cor','estrutura','sao','são','sendo','agora','tem','ta','tá','esta','está','com','por','pra','para','significa','quer','dizer','é']);
const MEANINGS = Object.keys(LEX)
const FOOD_MOBS = new Set(['cow','pig','chicken','sheep','rabbit'])
const HOSTILE = new Set(['zombie','skeleton','spider','creeper','enderman'])

const culture = { foodSpots: (SAVED.culture||[]).slice(-60) }   // cultura RECUPERADA do disco
const bots = []

// EVOLUCAO: hall-da-fama de genomas (drives) bem-sucedidos + palavras CUNHADAS (lexico emergente)
const HALL = (SAVED.hall||[])      // [{drives, fit}]
const COINED = (SAVED.coined||{})  // palavra-cunhada -> referente (vocabulario alem do manual)
const SITWORD = (SAVED.sitword||{})  // sig-de-situacao -> {word, n} : LEXICO COMPARTILHADO emergente (transmissao cultural)
let GEN = SAVED.gen||0
const TOM = { taught:0, frontierHits:0, threatTaught:0, vocabTaught:0 }   // ensino guiado por Teoria da Mente (H99)
// ===== PRESSAO EXTERNA: deteccao de NOVIDADE no mundo (introduzida pelo player) =====
const KNOWN_TYPES = new Set()                       // tipos de bloco ja conhecidos (semeado com a natureza no warmup)
const NATURAL = /air|stone|dirt|grass|sand|gravel|water|lava|bedrock|_log$|_leaves|_planks|_ore$|cobble|deepslate|granite|diorite|andesite|tuff|clay|snow|ice|fern|flower|poppy|dandelion|mushroom|seagrass|kelp|vine|moss|root|sandstone|terracotta|calcite|dripstone|netherrack|basalt|blackstone|bush|grass_block|podzol/i
const NOVELTY = { detected:0, registered:0, investigated:0, last:null }   // pressao externa (player) crackeando o plato?
const NOVEL_POS = []   // posicoes dos objetos novos do player -> CURIOSIDADE atrai os bots a CONVIVER (gera variancia p/ a forja perceptual promover)
const PLANN = { acted:0, avoided:0 }   // PLANEJAMENTO POR IMAGINACAO (Mattar&Daw): acoes escolhidas por simulacao / evitacoes antecipadas
const OPT = { ran:0 }                  // OPCOES HIERARQUICAS executadas (rotinas como acao unica)
let REPLAYED = 0                       // transicoes replayadas (replay priorizado)
function randDrives(){ return { social:Math.random(), explore:0.4+Math.random()*0.6, forage:0.6+Math.random()*0.4, mirror:0.3+Math.random()*0.4, hunger_thr:14+Math.random()*6 } }
function mutd(d){ const m={}; for(const k in d) m[k]=d[k]*(1+(Math.random()-0.5)*0.3)
  m.hunger_thr=Math.min(20,Math.max(8,m.hunger_thr)); ['social','explore','forage','mirror'].forEach(k=>m[k]=Math.min(1,Math.max(0.02,m[k]))); return m }
function pickParent(){ if(!HALL.length) return null; const a=HALL[(Math.random()*HALL.length)|0], b=HALL[(Math.random()*HALL.length)|0]; return (a.fit>=b.fit?a:b).drives }
function evolveDrives(){ const p=pickParent(); return (p && Math.random()<0.8)? mutd(p) : randDrives() }   // PRIOR EVOLUTIVO

function makeBrain(i){
  if(SAVED.brains && SAVED.brains[i]){                 // RECUPERA cerebro salvo
    const b=SAVED.brains[i]
    if(b.value) for(const k in b.value){ const x=b.value[k]; b.value[k]=(typeof x==='number'&&isFinite(x))?Math.max(-20,Math.min(20,x)):0 }  // sanitiza valores corrompidos
    return { drives:b.drives||randDrives(), nm:{da:0.5,ne:0.3,se:0.5}, attach:b.attach||0.3, intent:null, intentT:0, lastAction:null, foodEaten:0,
             value:b.value||{}, seen:b.seen||{}, assoc:b.assoc||{}, words:b.words||{}, bound:b.bound||{}, arch:b.arch||{}, supercat:b.supercat||{}, self:b.self||null, concepts:b.concepts||{}, pred:b.pred||{}, comp:b.comp||{}, threat:b.threat||{}, wm:b.wm||{}, punished:b.punished||{}, goal:b.goal||null, lastEat:0, born:Date.now() }
  }
  // nm = neuromoduladores (conectoma); attach = APEGO ao player (cresce se voce e relevante = modelo de "pai")
  return { drives:evolveDrives(), nm:{da:0.5,ne:0.3,se:0.5}, attach:0.3, intent:null, intentT:0, lastAction:null, foodEaten:0, value:{}, seen:{}, assoc:{}, words:{}, bound:{}, lastEat:0, born:Date.now() }
}
const clamp01=x=>Math.max(0,Math.min(1,x))
// MODELO DE MUNDO: liga um EVENTO (key) ao seu impacto multidimensional. Memoria = media movel por atributo + ancora espacial.
function recordEvent(bot, key, attrs){
  const W=(bot.brain.wm=bot.brain.wm||{}); const e=(W[key]=W[key]||{n:0})
  for(const k in attrs){ const v=attrs[k]; if(typeof v==='number'&&isFinite(v)) e[k]=(e[k]==null?v:e[k]*0.9+v*0.1) }
  e.n=(e.n||0)+1; if(bot.entity){ const p=bot.entity.position; e.lx=Math.round(p.x); e.ly=Math.round(p.y); e.lz=Math.round(p.z) }
}

// PODA DE MEMORIA: cap mapas cognitivos dos cerebros (cresciam sem limite -> causavam OOM).
function capMap(obj, n, scoreFn){ if(!obj) return; const ks=Object.keys(obj); if(ks.length<=n) return
  ks.sort((a,b)=>(scoreFn?scoreFn(obj[b]):0)-(scoreFn?scoreFn(obj[a]):0))   // mantem os de maior score
  for(let i=n;i<ks.length;i++) delete obj[ks[i]] }
const byN = o => (o&&typeof o==='object'&&o.n)||0
const byMag = o => (typeof o==='number')?Math.abs(o):(o&&o.val!=null?Math.abs(o.val):(o&&o.n)||0)
function pruneBrains(){
  for(const b of bots){ const br=b.brain; if(!br) continue
    capMap(br.concepts,120,byN); capMap(br.comp,120,byN); capMap(br.pred,200,byN)
    capMap(br.wm,80,byN); capMap(br.seen,150,byN); capMap(br.assoc,100,byN)
    capMap(br.words,100,()=>1); capMap(br.bound,100,()=>1); capMap(br.arch,60,byN)
    capMap(br.threat,30,byMag); capMap(br.value,60,byMag); capMap(br.supercat,40,byN); capMap(br.punished,40,byMag)
  }
  capMap(SITWORD,200,o=>o&&o.n||0); capMap(COINED,200,()=>1)
}

function saveState(){
  try{ pruneBrains() }catch(e){}
  const brains={}
  bots.forEach(b=>{ if(b.brain && b.idx!=null) brains[b.idx]={ drives:b.brain.drives, attach:b.brain.attach,
    value:b.brain.value, seen:b.brain.seen, assoc:b.brain.assoc, words:b.brain.words, bound:b.brain.bound, arch:b.brain.arch, supercat:b.brain.supercat, self:b.brain.self, concepts:b.brain.concepts, pred:b.brain.pred, comp:b.brain.comp, threat:b.brain.threat, wm:b.brain.wm, punished:b.brain.punished, goal:b.brain.goal } })
  try{ fs.writeFileSync(STATE, JSON.stringify({culture:culture.foodSpots, brains, hall:HALL.slice(0,12), coined:COINED, sitword:SITWORD, gen:GEN})) }catch(e){}
  try{ skills.saveLib(); skills.saveAff(); wm.save(); prog.save(); hippo.prune(); hippo.save() }catch(e){}
}
setInterval(saveState, 20000)                          // autosave a cada 20s
// META-LOOP DE AUTO-MELHORIA: mede a fitness da colonia e ajusta os PROPRIOS hiperparametros (hill-climbing seguro)
function colonyFitness(){
  const alive=bots.filter(b=>b.entity); if(!alive.length) return 0
  const avgFood=alive.reduce((s,b)=>s+(b.food??0),0)/alive.length
  const avgHp=alive.reduce((s,b)=>s+(b.health??0),0)/alive.length
  let rel=0; for(const b of bots){ if(b.brain&&b.brain.wm) rel+=Object.keys(b.brain.wm).length }
  const S=skills.STATS; const frac=(S.used_no_llm+S.minted_llm)>0?S.used_no_llm/(S.used_no_llm+S.minted_llm):0
  // fitness = sobreviver bem + progredir (tech tree) + aprender (relacoes, hipocampo) + autonomia (compounding)
  return avgFood*0.5 + avgHp*0.3 + alive.length*1.0 + prog.progressLevel()*6 + hippo.stats().places*0.01 + frac*5
}
setInterval(()=>{ try{ const f=colonyFitness(); const r=selfimp.step(f)
  console.log(`[AUTO-MELHORIA] fitness=${f.toFixed(1)} | ajustando=${selfimp.stats().tuning} | tune=${JSON.stringify(selfimp.stats().tune)}`)
}catch(e){} }, 120000)                                 // a cada 2min: auto-ajuste
process.on('SIGINT', ()=>{ saveState(); process.exit(0) })
process.on('SIGTERM', ()=>{ saveState(); process.exit(0) })

function spawnBot(i){
  const bot = mineflayer.createBot({ host:'localhost', port:25565, username:'Nova_'+i, version:VERSION, auth:'offline' })
  bot.idx = i
  bot.brain = makeBrain(i)
  bot.loadPlugin(pathfinder)
  bot.once('spawn', ()=>{
    try{ bot.mcData = require('minecraft-data')(bot.version) }catch(e){ bot.mcData=null }
    const mv = new Movements(bot); mv.allowParkour=true; bot.pathfinder.setMovements(mv)
    bot.chat('Nova_'+i+' nasceu'); loop(bot, i)
  })
  // CORTEX AUDITIVO: ouve sons do mundo. Som de algo que NAO ve -> alerta (noradrenalina) + curiosidade (vira p/ la)
  const onSound=(name,pos)=>{ try{
    if(!bot.entity||!pos) return; const d=pos.distanceTo? pos.distanceTo(bot.entity.position):99; if(d>16) return
    const visible=!bot.world.raycast(bot.entity.position.offset(0,1.6,0), pos.minus(bot.entity.position).scaled(1/Math.max(d,0.1)), d-0.6)
    if(!visible && bot.brain.nm){ bot.brain.nm.ne=clamp01(bot.brain.nm.ne+0.05); bot.brain.heard={pos,t:Date.now(),name:String(name)} }  // ouviu algo INVISIVEL -> tenso
  }catch(e){} }
  bot.on('soundEffectHeard',(n,p)=>onSound(n,p))
  bot.on('hardcodedSoundEffectHeard',(id,cat,p)=>onSound('som',p))
  // APRENDIZADO OBSERVACIONAL DO PLAYER: ve voce ferir bicho->aprende CACAR; ferir MONSTRO->aprende LUTAR (coragem)
  bot.on('entityHurt',(e)=>{ try{
    if(!e||!e.name||!bot.entity) return
    const pname=Object.keys(bot.players||{}).find(u=>!u.startsWith('Nova_'))
    const pl=pname&&bot.players[pname]&&bot.players[pname].entity
    if(!pl || e.position.distanceTo(pl.position)>5 || e.position.distanceTo(bot.entity.position)>16 || !canSee(bot,e)) return
    if(FOOD_MOBS.has(e.name)){ bot.brain.value['cacar']=(bot.brain.value['cacar']||0)+0.6; bot.brain._huntBoost=Date.now() }
    else if(HOSTILE.has(e.name)){ bot.brain.value['lutar']=(bot.brain.value['lutar']||0)+0.5; bot.brain._fightBoost=Date.now() }  // VIU voce LUTAR -> aprende que da p/ enfrentar
    bot.brain.nm.da=clamp01(bot.brain.nm.da+0.1); bot.brain.attach=clamp01((bot.brain.attach||0.3)+0.05)   // aprender COM voce -> vinculo cresce
  }catch(err){} })
  // APRENDER LETALIDADE NA PELE: levei dano -> qual hostil estava perto? constroi MODELO CAUSAL (este tipo me machuca)
  bot.on('health',()=>{ try{
    if(!bot.entity) return; const hp=bot.health??20; const ph=bot.brain._prevHp??hp
    if(hp<ph-0.5){   // PERDI vida
      const T=(bot.brain.threat=bot.brain.threat||{})
      const att=nearest(bot, e=> e.name && HOSTILE.has(e.name), false)
      const dmg=ph-hp
      const myPos=bot.entity.position
      const plName=Object.keys(bot.players||{}).find(u=>!u.startsWith('Nova_'))
      const pl=plName&&bot.players[plName]&&bot.players[plName].entity
      const plClose = pl && pl.position.distanceTo(myPos)<4.5
      const hostClose = att && att.position.distanceTo(myPos)<7
      if(plClose && (!hostClose || pl.position.distanceTo(myPos) < att.position.distanceTo(myPos))){
        // VOCE me bateu -> CONDICIONAMENTO OPERANTE (treino, tipo animal): a ACAO que eu fazia ficou RUIM (aprende por consequencia, sozinho)
        const la=bot.brain.lastAction
        if(la){ const v=bot.brain.value; v[la]=Math.max(-20,(v[la]||0)-dmg*0.8)
          const PB=(bot.brain.punished=bot.brain.punished||{}); PB[la]=(PB[la]||0)+1 }   // memoria: o player reprova esta acao
        bot.brain._scolded={action:la,t:Date.now()}
      } else if(hostClose){
        T[att.name]=(T[att.name]||0)*0.8 + dmg*0.2   // estima dano-por-encontro deste tipo
        bot.brain._underAttack={type:att.name,t:Date.now(),pos:att.position}
        const ar=(bot.brain.arch=bot.brain.arch||{}); ar[att.name]=ar[att.name]||{val:0,n:0}; ar[att.name].val=Math.max(-1,ar[att.name].val-0.3); ar[att.name].n++
        // CONTAGIO DE PERIGO SOCIAL: aliados perto APRENDEM que esse tipo machuca (vi meu amigo apanhar)
        for(const o of bots){ if(o===bot||!o.entity||!o.brain) continue; if(o.entity.position.distanceTo(myPos)<10){
          const OT=(o.brain.threat=o.brain.threat||{}); OT[att.name]=Math.max(OT[att.name]||0, dmg*0.5) } }
      }
    }
    bot.brain._prevHp=hp
  }catch(err){} })
  bot.on('death', ()=> { try{
      try{ if(bot.entity) hippo.record(bot.entity.position, 'death', -4); hippo.save() }catch(e){}   // HIPOCAMPO: marca o lugar da MORTE (forte) -> evita voltar
      bot.chat('perigo '+Math.round(bot.entity.position.x)+' '+Math.round(bot.entity.position.y)+' '+Math.round(bot.entity.position.z))
      // EVOLUCAO: aptidao = tempo de vida + comida obtida; melhores genomas entram no hall e se reproduzem
      const fit=(Date.now()-(bot.brain.born||Date.now()))/1000 + (bot.brain.foodEaten||0)*4
      HALL.push({drives:bot.brain.drives, fit}); HALL.sort((a,b)=>b.fit-a.fit); if(HALL.length>12) HALL.length=12
      bot.brain.drives=evolveDrives(); bot.brain.born=Date.now(); bot.brain.foodEaten=0
      const inst={}; for(const k in (bot.brain.value||{})) inst[k]=bot.brain.value[k]*0.3   // MEMORIA INSTINTIVA: 30% do aprendido passa adiante (acumula entre vidas)
      bot.brain.value=inst; GEN++   // RENASCE evoluido, mas com instinto herdado
    }catch(e){} })
  bot.on('chat', (user, msg)=> onHear(bot, user, msg))
  bot.on('error', e=> {})
  bot.on('kicked', r=> console.log('Nova_'+i+' kicked', JSON.stringify(r).slice(0,80)))
  bot.on('end', ()=>{ const idx=bots.indexOf(bot); if(idx>=0)bots.splice(idx,1); enqueue(i) })  // RENASCE via fila
  bots.push(bot); return bot
}

// FALAR em PORTUGUES composto (com THROTTLE p/ nao floodar o chat: max 1 fala a cada ~5s por agente)
function emit(bot, meaning, pos, mod){
  const w = LEX[meaning]; if(!w) return
  const now=Date.now(); if(bot.brain._lastChat && now-bot.brain._lastChat<5000) return
  bot.brain._lastChat=now
  bot.chat(w + (mod&&MOD[mod]?' '+MOD[mod]:'') + (pos?` ${Math.round(pos.x)} ${Math.round(pos.y)} ${Math.round(pos.z)}`:''))
}
// OUVIR: entende as palavras do MANUAL (REV) na hora + palavras NOVAS que aprendeu do player (bound)
function onHear(bot, user, msg){
  if(user===bot.username) return
  const raw = msg.toLowerCase()
  const words = raw.replace(/[^a-zà-ú0-9\s-]/g,' ').split(/\s+/).filter(Boolean)
  const coords = msg.match(/(-?\d+)\s+(-?\d+)\s+(-?\d+)/)
  // CULTURA: adota palavras CUNHADAS por outros (lexico emergente se espalha)
  words.forEach(w=>{ const nw=norm(w); if(COINED[nw] && !bot.brain.bound[nw] && Math.random()<0.4) bot.brain.bound[nw]=COINED[nw] })
  // TRANSMISSAO OSTENSIVA de CONCEITO-DE-SITUACAO (Nova->Nova): se outro fala "<palavra> perigo/bom" e EU estou
  // numa situacao recorrente com a MESMA valencia AGORA, eu adoto a palavra dele p/ a MINHA situacao -> idioma converge
  if(user.startsWith('Nova_') && bot.entity){
    const val = /\bperigo\b/.test(raw)?-1 : (/\bbom\b/.test(raw)?1 : 0)
    if(val!==0){ const heard=words.find(w=>norm(w).length>=3 && !REV[norm(w)] && !STOP.has(norm(w)) && !/^\d+$/.test(w))
      if(heard){ try{ const p=bot.entity.position, bb=bot.blockAt(p.offset(0,-1,0)), sky=bot.blockAt(p.offset(0,3,0)), tod=bot.time?bot.time.timeOfDay:0
        const mysig=(bb?bb.name:'ar')+'|'+(tod>13000?'noite':'dia')+'|'+(sky&&sky.name==='air'?'ceuaberto':'coberto')
        const P=bot.brain.pred&&bot.brain.pred[mysig]
        if(P && P.n>=4 && Math.sign(P.exp)===val){   // mesma situacao + mesmo sentimento = referente compartilhado
          const C=(bot.brain.concepts=bot.brain.concepts||{}); const hw=norm(heard)
          if(!C[mysig] || C[mysig].word!==hw){ C[mysig]={word:hw,val:P.exp,sig:mysig}
            const S=(SITWORD[mysig]=SITWORD[mysig]||{word:hw,n:0}); S.word=hw; S.n++ }   // converge p/ a palavra ouvida
        } }catch(e){} }
    }
  }
  const meaning = words.map(w=> resolveWord(bot,w)).find(Boolean)        // FUZZY (tolera "veeeem", typos)
  if(meaning==='FOOD' && coords){ const p={x:+coords[1],y:+coords[2],z:+coords[3]}
    culture.foodSpots.push(p); if(culture.foodSpots.length>60) culture.foodSpots.shift() }
  if(meaning==='DANGER'){ const t=nearest(bot,e=>e.name&&HOSTILE.has(e.name),false)
    if(t && bot.entity && t.position.distanceTo(bot.entity.position)<14){   // CONTAGIO so se ha ameaca REAL perto (corta o loop de panico)
      if(bot.brain.nm) bot.brain.nm.ne=clamp01(bot.brain.nm.ne+0.05); const pos=bot.entity.position
      const away=pos.minus(t.position).normalize().scaled(6).plus(pos); bot.pathfinder.setGoal(new goals.GoalNear(away.x,pos.y,away.z,1)) } }
  // FOLLOW ("vem"): VAI ATE QUEM CHAMOU (bug consertado!). Do player depende do APEGO; de outro Nova, do social.
  if(meaning==='FOLLOW' && bot.entity){
    const sp = bot.players && bot.players[user] && bot.players[user].entity
    if(sp){ const isPlayer=!user.startsWith('Nova_')
      bot.brain.attach = bot.brain.attach||0.3
      const willCome = isPlayer ? (Math.random()<bot.brain.attach+0.2) : (Math.random()<bot.brain.drives.social)
      if(willCome){ bot.pathfinder.setGoal(new goals.GoalNear(sp.position.x,sp.position.y,sp.position.z,2)); if(isPlayer&&Math.random()<0.3) bot.chat('vem '+Math.round(sp.position.x)+' '+Math.round(sp.position.y)+' '+Math.round(sp.position.z)) }
    }
  }
  // REAGE a um REFERENTE aprendido (ex: voce diz "abelha" -> ele olha/vai ate a abelha; mostra que entendeu)
  const ref = words.map(w=>resolveWord(bot,w)).find(r=>typeof r==='string'&&r.startsWith('ENT:'))
  if(ref && bot.entity){ const tp=ref.slice(4); const e=nearest(bot,x=>x.name===tp)
    if(e){ bot.lookAt(e.position.offset(0,(e.height||0.5),0)).catch(()=>{}); bot.pathfinder.setGoal(new goals.GoalNear(e.position.x,e.position.y,e.position.z,3)) } }
  // COMPREENSAO REFERENCIAL ANCORADA: ouvir um simbolo cunhado ATIVA o conceito profundo (causal+espacial+afetivo)
  // -> transfere conhecimento mesmo SEM ver o referente (ex: ouve 'pipi'/zumbi -> SABE que machuca -> fica alerta).
  if(bot.entity){ for(const w of words){ const nw=norm(w); const concept=COINED[nw]||bot.brain.bound[nw]; if(!concept) continue
    language.anchor(nw, concept); const g=language.comprehend(bot, concept); if(!g) continue
    bot.brain._comprehended=(bot.brain._comprehended||0)+1
    if(g.type==='danger'){                                  // ENTENDEU um aviso de perigo: ativa medo/vigilancia ancorada
      bot.brain.nm.ne=clamp01((bot.brain.nm.ne||0.3)+0.07*g.intensity); bot.brain._heardDanger={ent:g.ent||null,t:Date.now()}
      if(g.knows && Math.random()<0.5){ const pos=bot.entity.position; const safe=hippo.recallBest(pos,'food',10); // foge p/ um lugar lembrado seguro/bom
        if(safe) bot.pathfinder.setGoal(new goals.GoalNear(safe.x,pos.y,safe.z,2)) }
    } else if(g.type==='food' && (bot.food??20)<16){        // ENTENDEU sobre comida: vai a um lugar lembrado de comida
      const pos=bot.entity.position; const fp=hippo.recallBest(pos,'food',12); if(fp) bot.pathfinder.setGoal(new goals.GoalNear(fp.x,pos.y,fp.z,2))
    }
    break }   // so o 1o simbolo reconhecido por fala
  }
  // ---- PLAYER: pergunta, ensino, aprendizado (com guarda de NEGACAO e palavras-funcao) ----
  if(!user.startsWith('Nova_')){
    bot.brain.attach=clamp01((bot.brain.attach||0.3)+0.02)   // VOCE fala com eles -> vinculo cresce (relevancia social)
    bot.brain._attn=Date.now(); bot.brain._attnUser=user      // ATENCAO: decide() vai te encarar/aproximar (responsivo)
    const isQuestion = /\?|^\s*(o ?que|que)\s+(e|eh|é|significa)/.test(raw)
    const isNegation = /\b(nao|não|nem|nunca)\b/.test(raw)     // "nao e zombie" = correcao -> NAO aprende
    // candidato a palavra NOVA = conteudo, nao funcao, nao numero, nao ja-conhecida
    const noun = words.find(w=> norm(w).length>=3 && !STOP.has(norm(w)) && !resolveWord(bot,w) && !/^\d+$/.test(w))
    // PERGUNTA: "o que e X" -> responde o que entende (so 1 bot responde, p/ nao floodar)
    if(isQuestion){
      const tgt = words.slice().reverse().find(w=> !STOP.has(norm(w)) && norm(w).length>=2)
      const t = tgt? resolveWord(bot,tgt) : null
      if(Math.random()<0.25){ let ans='novo'; if(typeof t==='string'){ ans = t.startsWith('ENT:')? t.slice(4) : (LEX[t]||t.replace(/^STRUCT:/,'casa ')) } bot.chat(ans) }
      return
    }
    if(isNegation) return                                      // negacao: nao aprende nada (so corrige/ignora)
    // ENSINO POR DEFINICAO: "<nova> e/significa <conhecida>"
    const copula = words.some(w=>['e','eh','é','significa','quer'].includes(norm(w))||['é'].includes(w))
    const known = words.find(w=>resolveWord(bot,w))
    if(copula && known && noun){ bot.brain.bound[norm(noun)]=resolveWord(bot,known); if(Math.random()<0.25) bot.chat('entendi '+noun); return }
    // OSTENSIVO/DEICTICO: "aquilo voando e abelha" -> liga a palavra NOVA ao referente perto de VOCE
    const pl = bot.players && bot.players[user] && bot.players[user].entity
    if(pl && noun){
      let r=null,rd=8
      for(const id in bot.entities){ const e=bot.entities[id]; if(!e||!e.position||e===bot.entity)continue
        if(e.username && (e.username.startsWith('Nova_')||e.username===user)) continue
        const d=e.position.distanceTo(pl.position); if(d<rd && e.name){r=e;rd=d} }
      if(r){ bot.brain.bound[norm(noun)]='ENT:'+r.name; if(Math.random()<0.25) bot.chat('entendi '+noun+' = '+r.name); return }
      const blk=bot.findBlock({matching:b=>b&&b.name&&!['air','grass_block','dirt','stone','water','sand','gravel','tall_grass','short_grass','seagrass','snow','poppy','dandelion'].includes(b.name), point:pl.position, maxDistance:5})
      if(blk){ bot.brain.bound[norm(noun)]='STRUCT:'+blk.name; if(Math.random()<0.25) bot.chat('entendi '+noun); return }
    }
    // CO-OCORRENCIA (so palavras de conteudo, nunca funcao)
    const food=bot.food??20; const threat=nearest(bot,e=>e.name&&HOSTILE.has(e.name))
    const ctx = (food<12)?'FOOD' : (threat?'DANGER':'NEUTRO')
    for(const w of words){ const nw=norm(w); if(!nw||nw.length<3||STOP.has(nw)||REV[nw]||resolveWord(bot,w)||/^\d+$/.test(w)) continue
      const c=(bot.brain.words[nw]=bot.brain.words[nw]||{}); c[ctx]=(c[ctx]||0)+1
      const tot=Object.values(c).reduce((a,b)=>a+b,0), top=Object.entries(c).sort((a,b)=>b[1]-a[1])[0]
      if(tot>=4 && top[0]!=='NEUTRO' && top[1]/tot>0.65) bot.brain.bound[nw]=top[0] }
    if(meaning==='FOOD' && !coords){ bot.brain.intent='vagar'; bot.brain.intentT=2; if(Math.random()<0.2) emit(bot,'AHA',bot.entity.position) }
  }
}

async function loop(bot, i){
  let prev=(bot.food??20)+(bot.health??20)
  while(bot.entity){
    try { await decide(bot) } catch(e){}
    await new Promise(r=> setTimeout(r, 1200 + Math.random()*600))
    // CORRELACAO GERAL: ação -> sobrevivencia (Δcomida+Δvida), nao so comida. Aprende o que funciona.
    const cur=(bot.food??0)+(bot.health??0); const r=Math.max(-5,Math.min(5,cur-prev)); prev=cur   // clampa: evita corrupcao por descontinuidade (respawn)
    // FISGADA / FRUSTRAÇÃO: se há progresso, frustração zera. Se estagna, cresce.
    if(r <= 0) bot.brain.frustration = Math.min(1.0, (bot.brain.frustration||0) + 0.05);
    else bot.brain.frustration = 0;
    // REFORCO DA HABILIDADE EMBODIDA: a habilidade usada se FORTALECE se melhorou a sobrevivencia, decai se piorou
    if(bot.brain._skUsed){ const sk=bot.brain._skUsed; const sr=cur-(bot.brain._skBase||cur)
      sk.strength=Math.max(0,Math.min(5,(sk.strength||0.5)+sr*0.08))
      if(sr>0){ sk.success=(sk.success||0)+1; skills.STATS.skill_reward+=sr }
      else { sk.fails=(sk.fails||0)+1 }
      // AUTO-VERIFICACAO (Voyager pilar 3): skill que falhou MUITO e ficou fraca -> LLM REESCREVE corrigida
      if((sk.fails||0)>=4 && (sk.strength||0)<0.25 && !sk._refining && skills.hasKey() && Math.random()<0.5){
        sk._refining=true
        skills.refineSkill(sk, sk.name||'situacao').then(nc=>{ if(nc){ sk.code=nc; sk.strength=0.5; sk.fails=0; try{bot.chat('refinei:'+(sk.name||'skill'))}catch(e){} } sk._refining=false }).catch(()=>{sk._refining=false})
      }
      bot.brain._skUsed=null }
    // TEORIA DA MENTE: publica o proprio estado p/ os outros modelarem (aprendizado social/coordenacao)
    try{ mind.observeOther(bot.username, bot.brain.lastAction, bot.food??20, bot.health??20) }catch(e){}
    // HIPOCAMPO: registra o LUGAR atual + valor (place cell aprende: aqui foi bom/ruim/comida/perigo)
    try{ if(bot.entity){ hippo.record(bot.entity.position, r>=2?'food':(r<=-2?'danger':null), r) } }catch(e){}
    // VISAO: as vezes vira a cabeca p/ ESCANEAR o entorno (compensa o FOV limitado; sacadas visuais humanas)
    if(Math.random()<0.18 && (bot.brain.lastAction||'').match(/^(wm:|vagar|explorar|olhar|memoria|curios)/)){ try{ await vision.scan(bot) }catch(e){} }
    // PLANEJAMENTO: avanca o passo do plano se deu certo (reward>0); senao re-tenta o mesmo passo
    if(bot.brain._planStep){ mind.advancePlan(bot, r>0); bot.brain._planStep=false }
    // MODELO DE MUNDO PREDITIVO: observa efeito real da acao -> ERRO DE PREDICAO (dopamina) + progresso de aprendizado
    if(bot.brain._wmAct){ const actual=Math.max(-5,Math.min(5,cur-(bot.brain._wmBase||cur)))
      const o=wm.observe(bot.brain._wmCls, bot.brain._wmAct, actual)
      try{ pforge.observe(bot.brain._wmCls, bot.brain._wmAct, bot.brain._wmFeats, actual) }catch(e){}  // FORJA PERCEPTUAL: loga p/ detectar classe irredutivel
      try{ const newCls=wm.classify(skills.makeAPI(bot,goals), bot); wm.observeTransition(bot.brain._wmCls, bot.brain._wmAct, newCls) }catch(e){}  // MODELO DE TRANSICAO p/ planejamento (Mattar&Daw)
      bot.brain.nm.da = clamp01((bot.brain.nm.da||0.4)*0.85 + (0.5+o.rpe*0.12)*0.15)   // dopamina fasica = erro de predicao
      bot.brain._wmAct=null }
    // CURRICULO/PROGRESSAO: conquistou marco novo da tech tree? -> RECOMPENSA INTRINSECA (dominio) + anuncia
    try{ if(bot.entity){ const inv=bot.inventory.items().map(i=>i.name); const novos=prog.checkAchievements(inv, bot.username)
      for(const nv of novos){ bot.brain.nm.da=clamp01((bot.brain.nm.da||0.4)+0.3)   // pico de dopamina por dominio
        if(!bot.brain._lastChat||Date.now()-bot.brain._lastChat>=4000){ bot.brain._lastChat=Date.now(); bot.chat('consegui: '+nv.marco+(nv.first?' (1o da colonia!)':'')) }
        // CHUNKING POR MARCO: a sequencia de acoes que LEVOU ao marco e uma rotina valiosa -> abstrai (recompensa alta)
        try{ if(bot.brain._trace && bot.brain._trace.length>=2){ const ck=skills.observeChunk(bot.brain._trace, 4, skills.situationFP(bot))
          if(ck){ skills.addSkill(ck); skills.saveLib(); try{bot.chat('rotina:'+ck.name.slice(0,40))}catch(e){} } } }catch(e){} } } }catch(e){}
    const a=bot.brain.lastAction; if(a!=null){ const v=bot.brain.value; v[a]=Math.max(-20,Math.min(20,(v[a]||0)*0.93 + r*0.07)) }
    // ===== CHUNKING: ABSTRAI ROTINAS UTEIS DA PROPRIA EXPERIENCIA (sem LLM) =====
    // grava o verbo de acao no TRACE; quando uma SEQUENCIA recorre antes de recompensa, vira um SKILL nomeado e composavel
    try{ const verb=skills.normVerb(bot.brain.lastAction)
      if(verb){ (bot.brain._trace=bot.brain._trace||[]).push(verb); if(bot.brain._trace.length>6) bot.brain._trace.shift() }
      if(r>=2 && bot.brain._trace && bot.brain._trace.length>=2){
        const ck=skills.observeChunk(bot.brain._trace, r, skills.situationFP(bot))
        if(ck){ skills.addSkill(ck); skills.saveLib(); try{bot.chat('rotina:'+ck.name.slice(0,40))}catch(e){}; bot.brain._trace=[] } }
    }catch(e){}
    // ===== LINGUAGEM NO DIA A DIA: NOMEAR o que ve (ato comunicativo referencial), nao so alarmar no perigo =====
    // ve uma presa/mob conhecido -> de vez em quando ANUNCIA o simbolo cunhado; os Nova por perto OUVEM e COMPREENDEM.
    try{ if(bot.entity && (!bot.brain._lastName||Date.now()-bot.brain._lastName>=6000) && Math.random()<0.25){
      const seen=nearest(bot, e=> e && e.name && (/cow|pig|chicken|sheep|rabbit/.test(e.name) || HOSTILE.has(e.name)), false)
      if(seen && canSee(bot,seen)){ let sym=Object.keys(COINED).find(w=>COINED[w]==='ENT:'+seen.name)
        if(!sym){ const r=()=>String.fromCharCode(97+Math.floor(Math.random()*26)); sym=r()+r()+r(); COINED[sym]='ENT:'+seen.name; bot.brain.bound[sym]='ENT:'+seen.name }
        bot.brain._lastName=Date.now(); bot.chat(sym) } } }catch(e){}
    // ===== MODELO DE MUNDO MULTIDIMENSIONAL: atrela o EVENTO ao impacto em TODOS os atributos =====
    try{ if(bot.entity){
      const hpNow=bot.health??20, foodNow=bot.food??20, nmm=bot.brain.nm
      const dHp=hpNow-(bot.brain._wmHp??hpNow), dFood=foodNow-(bot.brain._wmFood??foodNow)
      bot.brain._wmHp=hpNow; bot.brain._wmFood=foodNow
      const vel=bot.entity.velocity||{x:0,z:0}; const speed=Math.sqrt(vel.x*vel.x+vel.z*vel.z)
      const host=nearest(bot,e=>e.name&&HOSTILE.has(e.name),false)
      const hostDist=host?host.position.distanceTo(bot.entity.position):64
      const allies=bots.filter(o=>o!==bot&&o.entity&&o.entity.position.distanceTo(bot.entity.position)<10).length
      const attrs={ vida:dHp, fome:dFood, medo:nmm.ne, euforia:nmm.da, calma:nmm.se, social:allies, velocidade:speed, dist_ameaca:hostDist, forca:Math.max(0,-dHp), valencia:r }
      const key=(host&&hostDist<10)?('enc:'+host.name):('ato:'+(bot.brain.lastAction||'nada'))   // evento saliente: encontro domina; senao a acao
      recordEvent(bot,key,attrs)
    }}catch(e){}
    // ===== FORMACAO DE CONCEITO POR SURPRESA (predicao, Friston/Zealot) — primitiva do ZERO da experiencia crua =====
    try{ if(bot.entity){
      const p=bot.entity.position; const bb=bot.blockAt(p.offset(0,-1,0)); const tod=bot.time?bot.time.timeOfDay:0
      const sky=bot.blockAt(p.offset(0,3,0)); // ferramentas de FEATURE BRUTA (nao rotulo de entidade):
      const sig = (bb?bb.name:'ar')+'|'+(tod>13000?'noite':'dia')+'|'+(sky&&sky.name==='air'?'ceuaberto':'coberto')
      const P=(bot.brain.pred=bot.brain.pred||{}); P[sig]=P[sig]||{exp:0,n:0}
      const surpresa=Math.abs(r - P[sig].exp); P[sig].exp = P[sig].exp*0.8 + r*0.2; P[sig].n++
      // SE uma situacao crua recorre com resultado forte e consistente -> FORMA um conceito NOVO (categoria inventada, nao rotulo)
      if(P[sig].n>=4 && Math.abs(P[sig].exp)>=0.6){
        const C=(bot.brain.concepts=bot.brain.concepts||{})
        if(!C[sig]){ const syl=['ka','lo','mi','tu','re','na','zo','pi','xa','bu','sa','ti']
          // ADOTA palavra compartilhada se a cultura ja nomeou esta situacao; senao SEMEIA a sua
          const shared=SITWORD[sig]; const w = shared? shared.word : (syl[(Math.random()*12)|0]+syl[(Math.random()*12)|0])
          if(shared) shared.n++; else SITWORD[sig]={word:w,n:1}
          C[sig]={word:w, val:P[sig].exp, sig}; if(!bot.brain._lastChat||Date.now()-bot.brain._lastChat>=5000){ bot.brain._lastChat=Date.now(); bot.chat(w+(P[sig].exp<0?' perigo':' bom')) } }  // conceito de SITUACAO, nao de coisa
      }
      // CONCEITO COMPOSICIONAL: combina a situacao ANTERIOR + ATUAL numa categoria NOVA se o par tem valor que
      // NENHUMA das duas isoladas previa (nao-aditivo) -> primitiva conjuntiva emergente da experiencia
      const ps=bot.brain._prevSig
      if(ps && ps!==sig){ const C=(bot.brain.concepts=bot.brain.concepts||{}), Cm=(bot.brain.comp=bot.brain.comp||{})
        const key=ps+'>'+sig, A=bot.brain.pred[ps], Bp=P[sig]
        if(A&&A.n>=4&&Bp.n>=4){ const pred=(A.exp+Bp.exp)/2; const cm=(Cm[key]=Cm[key]||{exp:0,n:0}); cm.exp=cm.exp*0.7+r*0.3; cm.n++
          if(cm.n>=3 && Math.abs(cm.exp-pred)>=0.8 && C[ps] && C[sig] && C[ps].word!==C[sig].word && !cm.word){   // PAR de conceitos DISTINTOS que diz algo que as partes nao diziam
            cm.word=(C[ps].word||'')+'-'+(C[sig].word||''); cm.val=cm.exp
            if(!bot.brain._lastChat||Date.now()-bot.brain._lastChat>=5000){ bot.brain._lastChat=Date.now(); bot.chat(cm.word) } } } }
      bot.brain._prevSig=sig
    }}catch(e){}
    // CONSOLIDA SUPERCATEGORIAS periodicamente (agrupa coisas por sentimento -> primitiva-categoria nova)
    bot.brain._tick=(bot.brain._tick||0)+1; 
    if(bot.brain._tick%25===0){ try{ consolidateSuper(bot) }catch(e){} }
    if(bot.brain._tick%60===0 && bot.idx===0){ try{ const arch=skills.mineArchetypes(); if(arch) bot.chat('Eureca! Arquétipo: '+arch.name); }catch(e){} }
    // NEUROMODULADORES (estado interno difuso, conectoma): da=recompensa, ne=stress, se=calma/saciedade
    const nm=bot.brain.nm, food=bot.food??20, hp=bot.health??20
    nm.da = clamp01(nm.da*0.92 + (r>0?0.12:0.0))
    nm.ne = clamp01(nm.ne*0.75 + (hp<10?0.2:0) + (food<8?0.1:0))   // decai sozinho; sobe com perigo/fome
    nm.se = clamp01(nm.se*0.95 + (food>16&&hp>16?0.06:0))
  }
}

// VISAO por LINHA DE VISADA (oclusao real, igual olhos): so "ve" o que nao esta atras de bloco
function canSee(bot, e){
  try{
    if(!bot.entity||!e||!e.position) return false
    if(!vision.inFOV(bot, e.position)) return false   // CAMPO DE VISAO humano (~200deg): nao ve o que esta fora do cone (ex: atras)
    const eye=bot.entity.position.offset(0,1.6,0), tgt=e.position.offset(0,(e.height||1)*0.5,0)
    const dir=tgt.minus(eye), dist=dir.norm()
    if(dist>32) return false           // alcance maximo da vista
    if(dist<1.6) return true
    const hit=bot.world.raycast(eye, dir.scaled(1/dist), dist-0.6)
    return !hit                        // sem bloco no caminho = visivel (oclusao = nao-wallhack)
  }catch(err){ return true }           // fallback se raycast indisponivel
}
function nearest(bot, pred, sight=true){
  let best=null, bd=1e9
  for(const id in bot.entities){ const e=bot.entities[id]; if(!e||e===bot.entity||!e.position) continue
    if(!pred(e)) continue
    if(sight && !canSee(bot,e)) continue          // so percebe o que ENXERGA (oclusao)
    const d=e.position.distanceTo(bot.entity.position); if(d<bd){bd=d;best=e} }
  return best
}
// FUZZY: tolera "veeeem", erros de digitacao. normaliza letras repetidas + distancia de edicao ao vocab conhecido
function norm(w){ return w.toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g,'').replace(/(.)\1+/g,'$1').replace(/[^a-z]/g,'') }
function lev(a,b){ const m=a.length,n=b.length,d=Array.from({length:m+1},(_,i)=>[i,...Array(n).fill(0)]); for(let j=0;j<=n;j++)d[0][j]=j
  for(let i=1;i<=m;i++)for(let j=1;j<=n;j++)d[i][j]=Math.min(d[i-1][j]+1,d[i][j-1]+1,d[i-1][j-1]+(a[i-1]===b[j-1]?0:1)); return d[m][n] }
function resolveWord(bot,w){
  const n=norm(w); if(!n) return null
  if(REV[n]) return REV[n]; if(bot.brain.bound[n]) return bot.brain.bound[n]
  let best=null,bd=99; const vocab=Object.keys(REV).concat(Object.keys(bot.brain.bound))
  for(const k of vocab){ const nk=norm(k); const dist=lev(n,nk); if(dist<bd){bd=dist;best=k} }
  if(best && bd<=Math.max(1,Math.floor(norm(best).length/3))) return REV[norm(best)]||REV[best]||bot.brain.bound[best]   // perto o bastante
  return null
}
async function eatAny(bot){
  const ed=bot.inventory.items().find(it=> it.name.match(/beef|porkchop|chicken|mutton|rabbit|bread|apple|carrot|cooked|melon|berries/))
  if(ed){ try{ await bot.equip(ed,'hand'); await bot.consume(); bot.brain.lastEat=Date.now(); bot.brain.foodEaten=(bot.brain.foodEaten||0)+1; return true }catch(e){} }
  return false
}
// MOTOR BABBLING: o agente EXPLORA o proprio corpo (como crianca) — testa uma acao e o sistema de valor aprende o efeito (affordance)
const BODY=['agachar','pular','correr','colocar','cavar','girar']
async function tryBody(bot, act){ try{
  if(act==='agachar'){ bot.setControlState('sneak',true); setTimeout(()=>{try{bot.setControlState('sneak',false)}catch(e){}},800) }
  else if(act==='pular'){ bot.setControlState('jump',true); setTimeout(()=>{try{bot.setControlState('jump',false)}catch(e){}},300) }
  else if(act==='correr'){ bot.setControlState('sprint',true); bot.setControlState('forward',true); setTimeout(()=>{try{bot.setControlState('sprint',false);bot.setControlState('forward',false)}catch(e){}},700) }
  else if(act==='colocar'){ const it=bot.inventory.items().find(i=>i.name.endsWith('_log')||i.name.endsWith('_planks')||i.name==='dirt'||i.name==='cobblestone'); if(it){ await bot.equip(it,'hand'); const ref=bot.blockAt(bot.entity.position.offset(0,-1,0)); if(ref) await bot.placeBlock(ref,{x:0,y:1,z:0}).catch(()=>{}) } }
  else if(act==='cavar'){ const b=bot.blockAt(bot.entity.position.offset(0,-1,0)); if(b&&bot.canDigBlock(b)) await bot.dig(b).catch(()=>{}) }
  else if(act==='girar'){ await bot.look(Math.random()*6.28,(Math.random()-0.5),false).catch(()=>{}) }
  return true
}catch(e){ return false } }
// SUPERCATEGORIA: agrupa coisas DISTINTAS por SENTIMENTO numa categoria NOVA com nome = primitiva mental nova (abstracao)
function consolidateSuper(bot){
  const ar=bot.brain.arch||{}; const neg=[],pos=[]
  for(const t in ar){ if(ar[t].val<=-0.3) neg.push(t); else if(ar[t].val>=0.3) pos.push(t) }
  bot.brain.supercat=bot.brain.supercat||{}
  const syl=['ka','lo','mi','tu','re','na','zo','pi','xa','bu']; const r=()=>syl[(Math.random()*10)|0]
  if(neg.length>=2 && (!bot.brain.supercat._neg || bot.brain.supercat._neg.membros.length<neg.length)){
    const w=bot.brain.supercat._neg?bot.brain.supercat._neg.word:(r()+'no'); bot.brain.supercat._neg={word:w,sign:'-',membros:neg.slice()}; COINED[w]='CAT:perigo'
    const now=Date.now(); if(!bot.brain._lastChat||now-bot.brain._lastChat>=5000){ bot.brain._lastChat=now; bot.chat(w+' '+neg.slice(0,3).join(' ')) }  // ANUNCIA: "essas coisas sao da classe X"
  }
  if(pos.length>=2 && (!bot.brain.supercat._pos || bot.brain.supercat._pos.membros.length<pos.length)){
    const w=bot.brain.supercat._pos?bot.brain.supercat._pos.word:(r()+'si'); bot.brain.supercat._pos={word:w,sign:'+',membros:pos.slice()}; COINED[w]='CAT:bom'
  }
}
// CONSTRUIR (necessidade de SEGURANCA/ESTIMA na piramide de Maslow): empilha blocos ao redor = abrigo/estrutura
async function buildAround(bot){
  const it=bot.inventory.items().find(i=>i.name.endsWith('_log')||i.name.endsWith('_planks')||['dirt','cobblestone','stone','sand'].includes(i.name))
  if(!it) return false
  try{ await bot.equip(it,'hand')
    for(const d of [[1,0],[-1,0],[0,1],[0,-1]]){
      const cell=bot.entity.position.offset(d[0],0,d[1])
      const here=bot.blockAt(cell), below=bot.blockAt(cell.offset(0,-1,0))
      if(here&&here.name==='air'&&below&&below.name!=='air'){ await bot.placeBlock(below,{x:0,y:1,z:0}).catch(()=>{}); return true }
    }
  }catch(e){}
  return false
}
// CRAFTING (instinto disparado por ter material): log->tabua->graveto->mesa->picareta de madeira
async function tryCraft(bot){
  const md=bot.mcData; if(!md) return false
  const inv=bot.inventory.items(); const has=n=> inv.find(i=>i.name===n)
  const craft=async(name,table)=>{ const it=md.itemsByName[name]; if(!it)return false
    const r=bot.recipesFor(it.id,null,1,table||null)[0]; if(!r)return false; try{ await bot.craft(r,1,table||null); return true }catch(e){return false} }
  const log=inv.find(i=>i.name.endsWith('_log'))
  if(log && !inv.find(i=>i.name.endsWith('_planks'))) return await craft(log.name.replace('_log','_planks'))
  const planks=inv.find(i=>i.name.endsWith('_planks'))
  if(planks && planks.count>=2 && !has('stick')) return await craft('stick')
  if(planks && planks.count>=4 && !has('crafting_table')) return await craft('crafting_table')
  // colocar a mesa e fazer picareta (precisa da mesa por perto)
  const tableItem=has('crafting_table')
  if(tableItem && has('stick') && planks && planks.count>=3){
    try{ const ref=bot.blockAt(bot.entity.position.offset(0,-1,0)); await bot.equip(tableItem,'hand'); await bot.placeBlock(ref, {x:1,y:0,z:0}) }catch(e){}
    const tb=bot.findBlock({matching:md.blocksByName['crafting_table'].id, maxDistance:4})
    if(tb) return await craft('wooden_pickaxe', tb)
  }
  return false
}

async function decide(bot){
  const hp=bot.health??20, food=bot.food??20, pos=bot.entity.position
  try{ bot.setControlState('sprint',false) }catch(e){}   // reseta corrida (so liga em fuga/perseguicao); evita sprint preso
  const D=bot.brain.drives, nm=bot.brain.nm||{da:0.5,ne:0.3,se:0.5}
  const threat = nearest(bot, e=> e.name && HOSTILE.has(e.name))
  const imminent = threat && threat.position.distanceTo(pos) < 4.5    // perigo REAL e proximo (sempre prioritario)
  // 0) PLAYER = LIVRE-ARBITRIO. So PARA pra voce se VOCE o CHAMOU (comunicacao legitima). Se voce so esta perto,
  // ele NAO interrompe a propria agenda — no maximo um olhar curioso ocasional (decidido por ele, nao imposto).
  const chatted = bot.brain._attn && Date.now()-bot.brain._attn<8000
  const plCalled = (chatted && bot.players && bot.brain._attnUser && bot.players[bot.brain._attnUser]) ? bot.players[bot.brain._attnUser].entity : null
  if(!imminent && plCalled){                          // VOCE chamou -> responde (e comunicacao, nao regra)
    try{ await bot.lookAt(plCalled.position.offset(0,1.6,0)) }catch(e){}
    const d=plCalled.position.distanceTo(pos)
    if(d>3 && d<40 && Math.random()<bot.brain.attach+0.3){ bot.pathfinder.setGoal(new goals.GoalNear(plCalled.position.x,plCalled.position.y,plCalled.position.z,2)); bot.brain.lastAction='atender' }
    else bot.brain.lastAction='olhar_voce'
    return
  }
  // player apenas PERTO (nao chamou): livre-arbitrio — segue sua agenda. Talvez um olhar curioso (sem travar/return).
  if(!imminent){ const nearPl=nearest(bot, e=> e.type==='player' && e.username && !e.username.startsWith('Nova_'), true)
    if(nearPl && nearPl.position.distanceTo(pos)<6 && Math.random() < 0.12*(D.social||0.5)){ try{ await bot.lookAt(nearPl.position.offset(0,1.6,0)) }catch(e){} } }
  // ===== ALIMENTACAO PRIORITARIA (necessidade fisiologica vence tudo): garante que NAO morram de fome,
  // alcancem o estado CALMO e o cerebro preditivo/curiosidade/affordances possa ATIVAR. Logica comprovada. =====
  if(food<7){
    if(await eatAny(bot)){ return }
    const drop = nearest(bot, e=> e.name==='item', false)
    if(drop && drop.position.distanceTo(pos)<14){
      if(drop.position.distanceTo(pos)>1.3){ bot.pathfinder.setGoal(new goals.GoalNear(drop.position.x,drop.position.y,drop.position.z,0)) }
      else { await new Promise(r=>setTimeout(r,300)); await eatAny(bot) }
      bot.brain.lastAction='coletar'; return }
    const prey = nearest(bot, e=> e.name && FOOD_MOBS.has(e.name))
    if(prey){ if(prey.position.distanceTo(pos)<2.5){ try{ await bot.attack(prey) }catch(e){}
        await new Promise(r=>setTimeout(r,500)); bot.pathfinder.setGoal(new goals.GoalNear(prey.position.x,prey.position.y,prey.position.z,0)); await new Promise(r=>setTimeout(r,400)); await eatAny(bot) }
      else { try{bot.setControlState('sprint',true)}catch(e){}; bot.pathfinder.setGoal(new goals.GoalFollow(prey,1.5)) }
      bot.brain.lastAction='cacar_fome'; return }
  }
  // ===== COLETA DE LOOT (proativa, nao so quando faminto): item largado por perto -> vai pegar (recurso gratis) =====
  // antes so coletavam comida com fome; agora pegam QUALQUER loot (madeira/ferramenta/cobblestone/comida) que veem cair.
  if(!imminent){ const loot = nearest(bot, e=> e.name==='item', false)
    if(loot && loot.position && loot.position.distanceTo(pos)<10){
      const d=loot.position.distanceTo(pos)
      skills.STATS.loot=(skills.STATS.loot||0)+1
      if(d>1.0){ bot.pathfinder.setGoal(new goals.GoalNear(loot.position.x,loot.position.y,loot.position.z,0)); bot.brain.lastAction='pegar_loot'; return }
      else { await new Promise(r=>setTimeout(r,250)); bot.brain.lastAction='pegar_loot'; return }   // em cima -> auto-pickup
    } }
  // ===== CAMADA DE HABILIDADE (FUSAO EMBODIDA / A PONTE) =====
  // Situacao SALIENTE: tenta resolver com habilidade APRENDIDA (recuperacao associativa, SEM LLM).
  // Se TRAVA (nenhuma habilidade boa), o LLM MINTA uma nova (o bot "pausa pra pensar"). Mede compounding.
  try{
    const api = skills.makeAPI(bot, goals)
    if(skills.isSalient(api)){
      skills.STATS.salient++
      const fp = skills.situationFP(bot)
      const sk = skills.bestSkill(fp)
      if(sk){                                   // JA SABE -> age sem LLM
        skills.STATS.used_no_llm++
        bot.brain._skUsed = sk; bot.brain._skBase = (bot.food??0)+(bot.health??0)
        await skills.runSkill(sk, bot, goals); sk.uses=(sk.uses||0)+1
        bot.brain.lastAction = 'skill:'+sk.name
        return
      } else if((()=>{   // ANTES do LLM: tenta FABRICAR habilidade SEM LLM (composicao/mutacao do que ja sabe)
          const sn=skills.mintNoLLM()
          if(sn){ sn.fp=fp; skills.addSkill(sn); skills.saveLib()
            bot.brain._skUsed=sn; bot.brain._skBase=(bot.food??0)+(bot.health??0)
            api.say('combinei:'+sn.name); bot.brain.lastAction='skill_composto:'+sn.name; return true }
          return false })()){
        await skills.runSkill(bot.brain._skUsed, bot, goals)
        return
      } else if(skills.hasKey() && (!bot.brain._mintCd || Date.now()-bot.brain._mintCd>20000)){
        bot.brain._mintCd = Date.now()
        const desc = skills.describeSituation(api)
        const code = await skills.mintSkill(desc)      // PAUSA PRA PENSAR (chama o LLM 1x)
        if(code){
          const sk2 = { name: skills.nameFor(api), code, fp, strength:0.6, uses:0, success:0, born:Date.now() }
          skills.addSkill(sk2); skills.saveLib(); skills.STATS.minted_llm++
          api.say('aprendi:'+sk2.name)
          bot.brain._skUsed = sk2; bot.brain._skBase = (bot.food??0)+(bot.health??0)
          await skills.runSkill(sk2, bot, goals)
          bot.brain.lastAction = 'skill_novo:'+sk2.name
          return
        }
      }
    }
    // ===== CALMO: WORKSPACE GLOBAL — modulos cognitivos COMPETEM; o mais saliente IGNITA e e transmitido =====
    else if(!skills.isSalient(api) && skills.isCalm(api)){
      const cls=wm.classify(api, bot); const fp=skills.situationFP(bot); const food=bot.food??20
      const invp=bot.inventory.items().map(i=>i.name); const nextM=prog.nextMilestone(invp)
      const base=()=>{ bot.brain._wmBase=(bot.food??0)+(bot.health??0); bot.brain._wmFeats=pforge.snapshot(bot) }  // captura features candidatas NA DECISAO
      const bids=[]
      // MODULO MEMORIA EPISODICA (hipocampo): lembra ONDE tinha comida/recurso e vai ate la (saliencia sobe com fome)
      const goodPlace = hippo.recallBest(pos, food<13?'food':null, 12)
      if(goodPlace){ bids.push(workspace.bid('episodico', 'ir_lugar', (food<13?(13-food)/13:0.3)+goodPlace.score*0.2,
        async()=>{ bot.pathfinder.setGoal(new goals.GoalNear(goodPlace.x, pos.y, goodPlace.z, 2)); bot.brain.lastAction='memoria>lugar_bom' })) }
      // MAESTRO ROUTING: A frustração dita se usamos Simbólico (Tech Tree) ou Descoberta Neural (Forja/Babbling)
      const frust = bot.brain.frustration || 0;
      if (frust > 0.6) {
        // Frustrado com as abordagens de alta cognição: fallback para Babbling (Descoberta Aberta)
        bids.push(workspace.bid('maestro_forja', 'motor_babbling', 0.85, async()=>{
          const act = BODY[Math.floor(Math.random()*BODY.length)];
          await tryBody(bot, act);
          bot.brain.lastAction = 'maestro:babbling:'+act;
        }));
      } else {
        // MODULO PLANEJAMENTO/CURRICULO: persegue o proximo marco da tech tree (plano multi-passo)
        if(nextM){ if(!mind.hasPlan(bot)) mind.makePlan(bot, nextM); const step=mind.planStep(bot)
          if(step) bids.push(workspace.bid('plano', step, selfimp.get('planSalience'),
            async()=>{ bot.brain._wmCls=cls; bot.brain._wmAct=step; base(); bot.brain._planStep=true; await skills.dispatch(api, step); bot.brain.lastAction='plano:'+nextM+'>'+step })) }
      }
      // MODULO PROJETO PROPRIO aprendido (reuso sem LLM)
      const sk = skills.bestSkill(fp)
      if(sk && sk.goal) bids.push(workspace.bid('projeto', sk.name, 0.55,
        async()=>{ skills.STATS.used_no_llm++; bot.brain._skUsed=sk; bot.brain._skBase=(bot.food??0)+(bot.health??0); await skills.runSkill(sk,bot,goals); sk.uses=(sk.uses||0)+1; bot.brain.lastAction='projeto:'+sk.name }))
      // MODULO TEORIA DA MENTE: imita quem prospera (aprendizado social)
      const th=mind.thrivingOther(bot.username)
      if(th && th.act && th.act.startsWith('wm:')) bids.push(workspace.bid('teoria_mente', th.act, 0.4,
        async()=>{ const a=th.act.slice(3); bot.brain._wmCls=cls; bot.brain._wmAct=a; base(); await skills.dispatch(api,a); bot.brain.lastAction='imitar:'+a }))
      // MODULO CURIOSIDADE (modelo de mundo): explora o aprendivel; modulado por dopamina + NE (surpresa->explora/reset)
      const cur=wm.curiousAction(cls); const _ne=(wm.neuro&&wm.neuro().ne)||0
      bids.push(workspace.bid('curiosidade', cur.action, (selfimp.get('curiosity')-0.1) + Math.max(0,cur.lp) + selfimp.get('exploreBonus')*(nm.da||0.3) + 0.15*_ne,
        async()=>{ bot.brain._wmCls=cls; bot.brain._wmAct=cur.action; base()
          const r=await skills.dispatch(api, cur.action)
          if(cur.action==='ativar_objeto' && r && r.res){ skills.recordAff(r.block.name,'activate',r.res.effect,0,r.res.changed); if(r.res.changed) api.say('descobri: '+r.block.name+' reage!'); skills.saveAff() }
          bot.brain.lastAction='wm:'+cur.action }))
      // MODULO VALOR (exploitation do que o modelo aprendeu ser bom)
      const val=wm.valuedAction(cls)
      bids.push(workspace.bid('valor', val.action, 0.3 + Math.max(0,val.pred)*0.3,
        async()=>{ bot.brain._wmCls=cls; bot.brain._wmAct=val.action; base(); await skills.dispatch(api,val.action); bot.brain.lastAction='wm:'+val.action }))
      // MODULO MOTOR-ABERTO (PowerPlay): persegue o DESAFIO auto-inventado ativo (nudge na acao-alvo da fronteira)
      if(ACTIVE_CHALLENGE && ACTIVE_CHALLENGE.nudge) bids.push(workspace.bid('desafio', ACTIVE_CHALLENGE.nudge, 0.5 + Math.min(0.3, (ACTIVE_CHALLENGE.lp||0)),
        async()=>{ bot.brain._wmCls=cls; bot.brain._wmAct=ACTIVE_CHALLENGE.nudge; base(); await skills.dispatch(api, ACTIVE_CHALLENGE.nudge); bot.brain.lastAction='desafio:'+ACTIVE_CHALLENGE.dim }))
      // MODULO PLANEJAMENTO POR IMAGINACAO (Mattar&Daw/DYNA): IMAGINA rollouts 3-passos a frente e age na melhor 1a acao.
      // Saliencia sobe com o retorno previsto e com o RISCO previsto (planejar importa mais quando ha consequencia forte).
      // MODULO OPCAO HIERARQUICA (Sutton-Precup-Singh): roda uma ROTINA inteira (chunk forte) como UMA acao
      const opt=skills.bestOption&&skills.bestOption()
      if(opt) bids.push(workspace.bid('opcao', opt.name, 0.5 + Math.min(0.4,(opt.strength-0.55)*0.3),
        async()=>{ bot.brain._skUsed=opt; bot.brain._skBase=(bot.food??0)+(bot.health??0); await skills.runSkill(opt,bot,goals); opt.uses=(opt.uses||0)+1; bot.brain.lastAction='opcao:'+opt.name; OPT.ran++ }))
      const plan=wm.planSR(cls)   // PLANEJAMENTO via REPRESENTACAO SUCESSORA (horizonte longo, hipocampo-Stachenfeld) > rollout raso
      if(plan && plan.action){ const hh=api.nearestHostile&&api.nearestHostile()
        const stakes = ((hh && api.dist(hh)<14)?0.3:0) + ((api.food<8||api.hp<10)?0.25:0)   // Mattar&Daw: planejar importa MAIS sob risco
        bids.push(workspace.bid('imaginacao', plan.action, 0.45 + Math.min(0.6, Math.abs(plan.value)*0.35) + stakes,
        async()=>{ bot.brain._wmCls=cls; bot.brain._wmAct=plan.action; base(); await skills.dispatch(api,plan.action); bot.brain.lastAction='imaginei:'+plan.action; PLANN.acted++; if(plan.value<0){ PLANN.avoided++; try{ const sym=Object.keys(COINED).find(w=>COINED[w]==='CAT:perigo'); if(sym&&(!bot.brain._lastWarn||Date.now()-bot.brain._lastWarn>2500)){bot.brain._lastWarn=Date.now();bot.chat(sym)} }catch(e){} } })) }
      // MODULO COMPOSITOR (SEM LLM): de vez em quando RECOMBINA 2 habilidades provadas numa nova (cresce a biblioteca sem custo)
      if(skills.libSize()>=2 && (!bot.brain._compCd || Date.now()-bot.brain._compCd>30000))
        bids.push(workspace.bid('compor', 'compor_skill', 0.32, async()=>{ bot.brain._compCd=Date.now()
          const sn=skills.mintNoLLM(); if(sn){ sn.fp=fp; skills.addSkill(sn); skills.saveLib()
            api.say('combinei:'+sn.name); bot.brain._skUsed=sn; bot.brain._skBase=(bot.food??0)+(bot.health??0)
            await skills.runSkill(sn,bot,goals); bot.brain.lastAction='compor:'+sn.name } }))
      // MODULO RARO: LLM minta projeto elaborado rumo ao marco (so as vezes)
      if(skills.hasKey() && (!bot.brain._goalCd || Date.now()-bot.brain._goalCd>selfimp.get('mintCooldown')))
        bids.push(workspace.bid('llm_projeto', 'mint', 0.5, async()=>{ bot.brain._goalCd=Date.now()
          const code=await skills.mintGoalSkill(skills.describeState(api,invp)+(nextM?` | PROXIMO MARCO: ${nextM}`:''))
          if(code){ const g={name:skills.goalName(),code,fp,strength:0.6,uses:0,success:0,goal:true,born:Date.now()}; skills.addSkill(g); skills.saveLib(); skills.STATS.minted_llm++; api.say('vou:'+g.name); bot.brain._skUsed=g; bot.brain._skBase=(bot.food??0)+(bot.health??0); await skills.runSkill(g,bot,goals); bot.brain.lastAction='projeto_novo' } }))
      // IGNICAO: o lance mais saliente vence a "consciencia" e e executado (broadcast)
      const winner = await workspace.ignite(bids)
      if(winner){ bot.brain._wmCls=bot.brain._wmCls||cls; return }
    }
  }catch(e){}
  // PERCEPCAO ANTECIPATORIA: avisto um hostil a DISTANCIA -> RECALL do modelo de mundo -> SINTO a emocao ANTES do dano,
  // e EVITO proativamente o que APRENDI ser letal (nao espero chegar perto). Medo CALIBRADO pela experiencia, nao uniforme.
  if(threat){
    const wmE=(bot.brain.wm||{})['enc:'+threat.name]; const dT=threat.position.distanceTo(pos)
    if(wmE && dT<16){
      nm.ne=clamp01(nm.ne + (wmE.medo||0)*0.04)                       // medo aprendido sobe so de AVISTAR (antecipacao)
      const expHarm=Math.max(0,-(wmE.vida||0))
      if(expHarm>0.8 && dT>5 && dT<14 && food>9 && Math.random()<0.5){ // modelo diz perigoso -> recua ANTES (projecao -> acao)
        const away=pos.minus(threat.position).normalize().scaled(7).plus(pos)
        bot.pathfinder.setGoal(new goals.GoalNear(away.x,pos.y,away.z,2)); bot.brain.lastAction='evitar_'+threat.name; return
      }
    }
  }
  // 1-CREEPER: PREVE a explosao -> foge ANTES (creeper arma a <3 blocos). Prioridade maxima, aprendido ou nao (instinto + memoria).
  const creeper=nearest(bot, e=> e.name==='creeper')
  if(creeper && creeper.position.distanceTo(pos)<6){
    emit(bot,'DANGER',pos)
    const away=pos.minus(creeper.position).normalize().scaled(9).plus(pos)
    try{ bot.setControlState('sprint',true) }catch(e){}
    bot.pathfinder.setGoal(new goals.GoalNear(away.x,pos.y,away.z,1)); bot.brain.lastAction='fugir_creeper'; return
  }
  // FOME CRITICA vence o medo de ameaca DISTANTE: faminto (<7) so interrompe p/ perigo IMINENTE (<4) — senao morre de fome fugindo
  const starving = food<7
  // 1) PERIGO: decide LUTAR EM GRUPO / FUGIR / FECHAR DISTANCIA (vs flecha) usando o MODELO DE LETALIDADE aprendido na pele.
  if(threat && threat.position.distanceTo(pos) < (starving?4:(4+4*nm.ne))){
    emit(bot,'DANGER',pos)
    // GRITA o SIMBOLO do perigo (linguagem ancorada): os bots por perto OUVEM e COMPREENDEM (ficam alertas sem ver)
    try{ let sym=Object.keys(COINED).find(w=>COINED[w]==='ENT:'+threat.name)
      if(!sym){ const r=()=>String.fromCharCode(97+Math.floor(Math.random()*26)); sym=r()+r()+r(); COINED[sym]='ENT:'+threat.name; bot.brain.bound[sym]='ENT:'+threat.name }   // cunha na hora se ainda nao existe
      if(!bot.brain._lastWarn||Date.now()-bot.brain._lastWarn>=2500){ bot.brain._lastWarn=Date.now(); bot.chat(sym) } }catch(e){}   // cooldown PROPRIO de alarme (nao starvado pela conversa de craft)
    const ar=(bot.brain.arch=bot.brain.arch||{}); ar[threat.name]=ar[threat.name]||{val:0,n:0}
    ar[threat.name].val=Math.max(-1,ar[threat.name].val-0.15); ar[threat.name].n++
    const T=bot.brain.threat||{}; const lethal=(T[threat.name]||0)   // QUANTO esse tipo ja me machucou (modelo causal)
    const allies = bots.filter(o=>o!==bot&&o.entity&&o.entity.position.distanceTo(pos)<10 && o.entity.position.distanceTo(threat.position)<12)
    const nAllies=allies.length
    const learnedFight = bot.brain.value.lutar>0.3 || (bot.brain._fightBoost && Date.now()-bot.brain._fightBoost<90000)
    const armed = bot.inventory.items().some(i=>i.name.includes('sword')||i.name.includes('axe'))
    const underAttack = bot.brain._underAttack && Date.now()-bot.brain._underAttack.t<3000
    const dist=threat.position.distanceTo(pos)
    // ESQUELETO/flecha: longe e com linha de visada -> FECHA DISTANCIA correndo (flecha perde eficacia colado) em vez de levar tiro parado
    if(threat.name==='skeleton' && dist>4 && dist<16){
      bot.pathfinder.setGoal(new goals.GoalFollow(threat,2)); bot.brain.lastAction='avancar_flecha'; return   // anda (nao sprinta: poupa fome)
    }
    // CONSULTA O MODELO DE MUNDO: o que costuma acontecer no ENCONTRO com este tipo? (impacto na vida esperado)
    const wmE=(bot.brain.wm||{})['enc:'+threat.name]
    const expHarm = wmE? Math.max(0,-(wmE.vida||0)) : 0      // dano-na-vida medio esperado deste encontro (multidimensional, aprendido)
    const tooDeadly = expHarm>1.2 && nAllies===0 && hp<14    // modelo diz: sozinho isso me destroi -> recua
    // CORAGEM EMERGENTE (modulada pelo modelo de mundo): defesa propria / forca-em-numero / armado / aprendeu — mas recua se o modelo preve massacre sozinho
    const groupBrave = nAllies>=1 && hp>7              // seguranca em numero: 2+ enfrentam juntos
    const selfDefense = underAttack && hp>5            // estao me batendo -> revido (e assim APRENDO a lutar)
    const brave = !tooDeadly && (groupBrave || selfDefense || (armed&&hp>10) || (learnedFight&&hp>8&&nm.da>0.15))
    if(brave){
      // COORDENA: chama os aliados pro MESMO alvo (ataque conjunto)
      if(nAllies>=1 && Math.random()<0.4) emit(bot,'DANGER',threat.position)
      if(dist<3){ try{ await bot.attack(threat); bot.brain.lastAction='lutar'; bot.brain.value['lutar']=Math.min(20,(bot.brain.value['lutar']||0)+0.3) }catch(e){} }  // ATACAR reforca a coragem (aprende fazendo)
      else { bot.pathfinder.setGoal(new goals.GoalFollow(threat,2)); bot.brain.lastAction='lutar' }
      return
    }
    const away = pos.minus(threat.position).normalize().scaled(8).plus(pos)
    if(dist<4){ try{ bot.setControlState('sprint',true) }catch(e){} }   // so sprinta se a ameaca esta COLADA (poupa fome em fuga distante)
    bot.pathfinder.setGoal(new goals.GoalNear(away.x, pos.y, away.z, 1)); bot.brain.lastAction='fugir'; return
  }
  // 1a) AUDICAO: ouviu algo invisivel ha pouco -> investiga (vira/anda na direcao do som)
  const h=bot.brain.heard
  if(h && Date.now()-h.t<3000 && food>=9){
    try{ await bot.lookAt(h.pos.offset?h.pos.offset(0,1,0):h.pos) }catch(e){}
    if(Math.random()<0.5){ bot.pathfinder.setGoal(new goals.GoalNear(h.pos.x,h.pos.y,h.pos.z,3)) }
    bot.brain.heard=null; return
  }
  const night = bot.time && bot.time.timeOfDay>13000 && bot.time.timeOfDay<23000
  const hasBlocks = bot.inventory.items().some(i=>i.name.endsWith('_log')||i.name.endsWith('_planks')||['dirt','cobblestone','stone','sand'].includes(i.name))
  // MASLOW nivel 2 (SEGURANCA): SO depois de comer (nivel 1 fisiologico vem primeiro!). de noite/stress e exposto -> CONSTROI abrigo
  if((night||nm.ne>0.5) && !threat && food>=12){
    const adj=[[1,0],[-1,0],[0,1],[0,-1]].filter(d=>{const b=bot.blockAt(pos.offset(d[0],0,d[1])); return b&&b.name!=='air'}).length
    if(adj<3 && hasBlocks){ if(await buildAround(bot)){ bot.brain.lastAction='construir'; bot.brain.nm.ne=clamp01(nm.ne-0.05); return } }
  }
  // 1c) SONHO/SONO (wake-sleep H53b): de noite, seguro E abrigado -> dorme e CONSOLIDA (replay + "aha")
  if(night && !threat && food>=10 && hp>=12){
    bot.brain.intent='dormir'; bot.brain.lastAction='dormir'
    try{ const bed=bot.findBlock({matching:b=>b&&b.name&&b.name.endsWith('_bed'),maxDistance:6}); if(bed && bot.isABed&&bot.isABed(bed)){ await bot.sleep(bed).catch(()=>{}) } }catch(e){}
    // CONSOLIDACAO (replay): reforca os valores de acoes que deram certo; transforma fragmento em "aha"
    const v=bot.brain.value; let best=null,bv=-1e9; for(const k in v){ if(v[k]>bv){bv=v[k];best=k} }
    if(best) v[best]=v[best]*1.05
    try{ hippo.replay() }catch(e){}   // HIPOCAMPO: REPLAY/consolidacao das memorias espaciais durante o sono
    const assoc=bot.brain.assoc; for(const t in assoc){ if(assoc[t]['fome']>=2 && !assoc[t]['comida']){ assoc[t]['comida']=1; if(Math.random()<0.15) bot.chat('entendi '+t) } }  // sonho reconecta -> aha matinal
    bot.brain.nm.se=clamp01(bot.brain.nm.se+0.05)   // descanso acalma
    // SONHO forma um "EU" (self-schema): quem sou = minha acao mais forte + meu vinculo + meu humor (inicio de um "ser")
    const ve=Object.entries(bot.brain.value||{}).sort((a,b)=>b[1]-a[1])[0]
    bot.brain.self={ sou:ve?ve[0]:'?', apego:Math.round((bot.brain.attach||0.3)*100)/100, humor: bot.brain.nm.se>bot.brain.nm.ne?'calmo':'tenso' }
    return
  }
  // 1b) CURIOSIDADE: se ve algo NOVO (pouco familiar) e nao esta desesperado -> PARA, observa, fragmenta na memoria
  if(food>=10 && hp>=10){
    const obj = nearest(bot, e=> { const t=e.name||e.username; return t && t!=='item' && !(e.username&&e.username.startsWith('Nova_')) })
    if(obj && obj.position.distanceTo(pos)<7){
      const t=obj.name||obj.username
      const seen=(bot.brain.seen[t]=(bot.brain.seen[t]||0)+1)
      if(seen<=2){                                         // NOVO -> presta atencao
        try{ await bot.lookAt(obj.position.offset(0,1,0)) }catch(e){}
        bot.chat('[?] '+t)                                 // simbolo de curiosidade ("o que e isso?")
        const ctx = (food<14?'fome':'neutro')
        const a=(bot.brain.assoc[t]=bot.brain.assoc[t]||{}); a[ctx]=(a[ctx]||0)+1   // fragmenta com contexto
        // ARQUETIPO com SENTIMENTO (valencia): cada coisa carrega uma carga afetiva que cresce com a experiencia
        const ar=(bot.brain.arch=bot.brain.arch||{}); ar[t]=ar[t]||{val:0,n:0}
        ar[t].val=Math.max(-1,Math.min(1, ar[t].val + (FOOD_MOBS.has(t)?0.12 : HOSTILE.has(t)?-0.18 : 0.0))); ar[t].n++
        // AHA: se ja viu esse tipo associado a comida antes, RECONECTA (vira fonte de comida conhecida)
        if(a['comida']>=2 && !FOOD_MOBS.has(t)){ bot.chat('[!] '+t); }
        return
      }
      // CUNHAGEM: ja e familiar mas ainda NAO tem nome -> inventa uma palavra nova e anuncia (lexico emergente)
      else if(seen===3 && !Object.values(bot.brain.bound).includes('ENT:'+t) && !Object.keys(COINED).some(w=>COINED[w]==='ENT:'+t)){
        const syl=['ka','lo','mi','tu','re','na','zo','pi','xa','bu','sa','ti']
        const w=syl[(Math.random()*syl.length)|0]+syl[(Math.random()*syl.length)|0]
        bot.brain.bound[w]='ENT:'+t; COINED[w]='ENT:'+t
        const now=Date.now(); if(!bot.brain._lastChat||now-bot.brain._lastChat>=5000){ bot.brain._lastChat=now; bot.chat(w+' '+Math.round(pos.x)+' '+Math.round(pos.y)+' '+Math.round(pos.z)) }
      }
      // reforco do aha: se um tipo apareceu junto de comida obtida, marca associacao
      if(bot.brain.lastEat && Date.now()-bot.brain.lastEat<3000){ const a=(bot.brain.assoc[t]=bot.brain.assoc[t]||{}); a['comida']=(a['comida']||0)+1 }
    }
  }
  // 1d) MOTOR BABBLING (explora o corpo, tipo crianca): seguro+curioso -> testa a acao corporal MENOS explorada
  if(food>=13 && hp>=13 && Math.random() < 0.15+0.3*nm.da){
    bot.brain.tries=bot.brain.tries||{}
    let act=BODY[0], mn=1e9; for(const b of BODY){ const c=bot.brain.tries[b]||0; if(c<mn){mn=c;act=b} }   // curiosidade = tenta o pouco-explorado
    bot.brain.tries[act]=(bot.brain.tries[act]||0)+1
    await tryBody(bot,act); bot.brain.lastAction='corpo:'+act   // o valor (loop) aprende o EFEITO -> affordance descoberta por ELE
    return
  }
  // 2) FOME: limiar PROPRIO. come o que tem; COLETA drops do chao (carne caiu da presa); senao caca; senao spot cultural
  if(food<D.hunger_thr){
    if(await eatAny(bot)){ bot.brain.value['hunt']=(bot.brain.value['hunt']||0)*0.9+1; return }
    // COLETA ITENS LARGADOS (consertou a fome cronica: matavam a presa mas NAO pegavam a carne no chao)
    const drop = nearest(bot, e=> e.name==='item', false)
    if(drop && drop.position.distanceTo(pos)<12){
      if(drop.position.distanceTo(pos)>1.3){ bot.pathfinder.setGoal(new goals.GoalNear(drop.position.x,drop.position.y,drop.position.z,0)); bot.brain.lastAction='coletar'; return }
      else { await new Promise(r=>setTimeout(r,350)); await eatAny(bot); bot.brain.lastAction='coletar'; return }
    }
    const prey = nearest(bot, e=> e.name && FOOD_MOBS.has(e.name))
    if(prey){
      if(prey.position.distanceTo(pos)<2.5){ try{ await bot.attack(prey); emit(bot,'FOOD',pos) }catch(e){}
        await new Promise(r=>setTimeout(r,600))
        bot.pathfinder.setGoal(new goals.GoalNear(prey.position.x,prey.position.y,prey.position.z,0))  // VAI ATE o drop coletar
        await new Promise(r=>setTimeout(r,500)); await eatAny(bot) }
      else bot.pathfinder.setGoal(new goals.GoalFollow(prey,1.5))
      return
    }
    if(culture.foodSpots.length){ const p=culture.foodSpots[culture.foodSpots.length-1]
      bot.pathfinder.setGoal(new goals.GoalNear(p.x,p.y,p.z,2)); return }
  }
  // 2b) DESESPERO (como a mente humana sob pressao): fome/HP MUITO baixos e nada deu certo -> VARIA tudo
  if(food<8 || hp<8){
    emit(bot,'DANGER',pos)                                   // pede ajuda / avisa
    const tries=[
      async()=>{ const m=nearest(bot,e=>e.type==='mob'||e.type==='animal'); if(m&&m.position.distanceTo(pos)<4){await bot.attack(m); await new Promise(r=>setTimeout(r,500)); await eatAny(bot); return true} },
      async()=> await tryCraft(bot),
      async()=>{ try{ const b=bot.blockAt(pos.offset(0,-1,0)); if(b&&bot.canDigBlock(b)){await bot.dig(b); return true} }catch(e){} },
      async()=>{ const a=Math.random()*6.28,r=20; bot.pathfinder.setGoal(new goals.GoalNear(pos.x+Math.cos(a)*r,pos.y,pos.z+Math.sin(a)*r,2)); return true }
    ]
    for(const t of tries){ try{ if(await t()) break }catch(e){} }
    return
  }
  // APEGO/PAI: inseguro (fome/stress moderado) e APEGADO -> volta a OBSERVAR voce (busca o "pai" p/ seguranca/aprender).
  // Confiante (apego baixo OU tudo bem) -> NAO te procura, explora sozinho. (o modelo que voce descreveu)
  if((food<11 || nm.ne>0.6) && (bot.brain.attach||0.3)>0.45){
    const pn=Object.keys(bot.players||{}).find(u=>!u.startsWith('Nova_')); const pl=pn&&bot.players[pn]&&bot.players[pn].entity
    if(pl && pl.position.distanceTo(pos)<45){ bot.pathfinder.setGoal(new goals.GoalNear(pl.position.x,pl.position.y,pl.position.z,3)); bot.brain.lastAction='observar_pai'; return }
  }
  // COOPERACAO (SELF primeiro, depois o outro): bem-alimentado + aliado faminto perto -> AJUDA (doa comida/avisa). + sobrevivencia do grupo
  if(food>16 && hp>14){
    const ally = bots.find(o=>o!==bot && o.entity && o.entity.position.distanceTo(pos)<6 && (o.food??20)<9)
    if(ally){
      const meat=bot.inventory.items().find(i=>i.name.match(/beef|porkchop|chicken|mutton|cooked|bread/))
      if(meat){ try{ await bot.lookAt(ally.entity.position); await bot.toss(meat.type,null,1) }catch(e){}; bot.brain.lastAction='ajudar'; bot.brain.value['ajudar']=(bot.brain.value['ajudar']||0)+0.3; return }
      if(culture.foodSpots.length){ emit(bot,'FOOD',culture.foodSpots[culture.foodSpots.length-1]); return }   // avisa onde tem comida
    }
  }
  // PROATIVIDADE: nao espera a fome chegar — seguro e nao-cheio -> ja vai juntar comida (prevencao = + sobrevivencia)
  if(food<18 && hp>12 && Math.random()<0.45){
    const prey=nearest(bot,e=>e.name&&FOOD_MOBS.has(e.name))
    if(prey){ if(prey.position.distanceTo(pos)<2.5){ try{await bot.attack(prey)}catch(e){}; await new Promise(r=>setTimeout(r,400)); await eatAny(bot) }
      else bot.pathfinder.setGoal(new goals.GoalFollow(prey,1.5)); bot.brain.lastAction='cacar'; return }
  }
  // 3) SOCIAL modulado pela SEROTONINA (calmo/saciado -> mais sociavel; estressado -> isola)
  if(Math.random() < D.social*(0.5+nm.se)*(1-0.5*nm.ne)){
    const friend = nearest(bot, e=> e.username && e.username.startsWith('Nova_'))   // SO outros agentes
    if(friend){ if(Math.random()<0.5) emit(bot,'FOLLOW',pos)
      bot.pathfinder.setGoal(new goals.GoalNear(friend.position.x,friend.position.y,friend.position.z,3)); return }
  }
  // MASLOW nivel 4 (ESTIMA/AUTORREALIZACAO): prospero e calmo -> CRIA estruturas (entusiasmo construtivo)
  if(food>16 && hp>16 && nm.se>0.4 && hasBlocks && Math.random()<0.4){ if(await buildAround(bot)){ bot.brain.lastAction='construir'; bot.brain.nm.da=clamp01(nm.da+0.05); return } }
  // 3b) CRAFTAR (instinto proativo): se tem material, tenta evoluir a tecnologia (tabua->graveto->mesa->picareta)
  const hasLog=bot.inventory.items().some(i=>i.name.endsWith('_log')||i.name.endsWith('_planks'))
  if(hasLog && Math.random()<0.4){ if(await tryCraft(bot)) { emit(bot,'FOLLOW',pos); return } }
  // 3c) NEURONIO-ESPELHO: observa o Nova vizinho que esta PROSPERANDO (mais comida) e IMITA a intencao dele
  if(Math.random()<D.mirror){
    let model=null, bf=(bot.food??20)+2
    for(const o of bots){ if(o===bot||!o.entity||!o.brain) continue
      if(o.entity.position.distanceTo(pos)<10 && (o.food??0)>bf && o.brain.lastAction){ model=o; bf=o.food } }
    if(model){ bot.brain.intent=model.brain.lastAction; bot.brain.intentT=4; emit(bot,'FOLLOW',pos) }   // aprende vendo
  }
  // CACAR aprendido OBSERVANDO o player: se aprendeu vendo voce, caca proativamente ao ver presa (nao so faminto)
  if((bot.brain.value.cacar>0.5 || (bot.brain._huntBoost && Date.now()-bot.brain._huntBoost<60000)) && food<18){
    const prey=nearest(bot, e=> e.name && FOOD_MOBS.has(e.name))
    if(prey){ if(prey.position.distanceTo(pos)<2.5){ try{ await bot.attack(prey) }catch(e){} await new Promise(r=>setTimeout(r,500))
        if(await eatAny(bot)){ const ar=(bot.brain.arch=bot.brain.arch||{}); ar[prey.name]=ar[prey.name]||{val:0,n:0}; ar[prey.name].val=Math.min(1,ar[prey.name].val+0.2) } }  // comer -> SENTIMENTO + por esse tipo
      else bot.pathfinder.setGoal(new goals.GoalFollow(prey,1.5)); bot.brain.lastAction='cacar'; return }
  }
  // 3d) META AUTO-GERADA POR CONCEITO: age sobre a PROPRIA categoria que inventou — busca o bloco da situacao que
  // seus conceitos dizem ser BOA (val>0). Conceito->objetivo->acao: subobjetivo nao programado na lista de prioridades.
  const C=bot.brain.concepts||{}
  if(food>=14 && food<19 && Math.random()<0.4+0.4*nm.da){   // SO quando saciado: exploracao proativa por conceito, NAO compete com comer
    let bestSig=null,bestVal=0.8
    for(const sig in C){ if(C[sig].val>bestVal){ bestVal=C[sig].val; bestSig=sig } }   // a situacao que ELE aprendeu ser otima
    if(bestSig){ const want=bestSig.split('|')[0]    // tipo de bloco-alvo extraido da categoria inventada
      if(want && want!=='ar' && want!=='air'){
        try{ const tgt=bot.findBlock({matching:b=>b&&b.name===want, maxDistance:24})
          if(tgt && tgt.position.distanceTo(pos)>2){ bot.pathfinder.setGoal(new goals.GoalNear(tgt.position.x,tgt.position.y,tgt.position.z,1))
            bot.brain.lastAction='buscar_'+want; bot.brain._goalSig=bestSig; if(Math.random()<0.15&&C[bestSig].word&&(!bot.brain._lastChat||Date.now()-bot.brain._lastChat>=5000)){bot.brain._lastChat=Date.now();bot.chat(C[bestSig].word)} return } }catch(e){}
      }
    }
  }
  // 3e) PLANEJAMENTO POR CONCEITO DIRECIONAL: usa a composicao NAO-COMUTATIVA que emergiu p/ AGIR — da situacao ATUAL,
  // busca a transicao "atual>proxima" que aprendeu ser BOA (assimetria de ordem) e navega ao bloco-alvo. Planeja 1 passo.
  if(food>=12 && Math.random()<0.3+0.3*nm.da){
    const Cm=bot.brain.comp||{}
    const bb=bot.blockAt(pos.offset(0,-1,0)), tod=bot.time?bot.time.timeOfDay:0, sky=bot.blockAt(pos.offset(0,3,0))
    const curSig=(bb?bb.name:'ar')+'|'+(tod>13000?'noite':'dia')+'|'+(sky&&sky.name==='air'?'ceuaberto':'coberto')
    let bestNext=null,bestV=0.8
    for(const k in Cm){ if(!Cm[k].word) continue; const parts=k.split('>'); if(parts[0]===curSig && (Cm[k].val||0)>bestV){ bestV=Cm[k].val; bestNext=parts[1] } }
    if(bestNext){ const want=bestNext.split('|')[0]
      if(want && want!=='ar' && want!=='air'){ try{ const tgt=bot.findBlock({matching:b=>b&&b.name===want, maxDistance:20})
        if(tgt && tgt.position.distanceTo(pos)>2){ bot.pathfinder.setGoal(new goals.GoalNear(tgt.position.x,tgt.position.y,tgt.position.z,1)); bot.brain.lastAction='planejar_'+want; return } }catch(e){} }
    }
  }
  // 4) PROJECAO BASEADA EM MODELO DE MUNDO + OBJETIVO PROPRIO: SIMULA o resultado esperado (wm) de cada acao e
  // escolhe a que mais avanca o objetivo INDIVIDUAL do agente; EVITA acoes que VOCE reprovou (treino tipo casa). Jornada autonoma.
  if(!bot.brain.intent || bot.brain.intentT<=0){
    // objetivo proprio = atributo que MAIS quero mover agora, pelo drive dominante + deficit atual (individual, autonomo)
    bot.brain.goal = food<14?'fome' : (nm.se<0.3?'calma' : (D.social>0.6?'social' : (D.explore>0.6?'euforia':'valencia')))
    const W=bot.brain.wm||{}, PB=bot.brain.punished||{}, opts=['explorar_longe','vagar','cavar']
    let best='vagar',bestScore=-1e9
    for(const act of opts){
      const e=W['ato:'+act]
      let score=(e?(e[bot.brain.goal]||0):0) + (bot.brain.value[act]||0)*0.1 - (PB[act]||0)*0.6 + (Math.random()-0.5)*0.25   // PROJECAO + valor - reprovacao + ruido
      if(score>bestScore){ bestScore=score; best=act }
    }
    bot.brain.intent=best; bot.brain.intentT = 4+Math.floor(Math.random()*6)
  }
  bot.brain.intentT--
  bot.brain.lastAction = bot.brain.intent   // registra p/ os neuronios-espelho dos outros observarem
  // cavar madeira se houver arvore por perto (vira material de craft)
  if(bot.brain.intent==='cavar'){
    try{ const tree=bot.findBlock({matching:b=>b&&b.name&&b.name.endsWith('_log'),maxDistance:6})
      if(tree){ bot.pathfinder.setGoal(new goals.GoalNear(tree.position.x,tree.position.y,tree.position.z,1)); if(tree.position.distanceTo(pos)<3) await bot.dig(tree); return }
      const b=bot.blockAt(pos.offset(0,-1,0)); if(b&&bot.canDigBlock(b)){ await bot.dig(b) } }catch(e){}
    return
  }
  const far = bot.brain.intent==='explorar_longe'
  const a=Math.random()*Math.PI*2, r=(far?20:6)+Math.random()*(far?30:10)
  bot.pathfinder.setGoal(new goals.GoalNear(pos.x+Math.cos(a)*r, pos.y, pos.z+Math.sin(a)*r, 2))
}

console.log('NovaMind colonia: subindo', N, 'agentes (fila de login serial)...')
// FILA GLOBAL DE LOGIN: 1 conexao a cada 5.5s (inicial E reconexao) -> nunca dispara o throttle
const connQueue=[]
function enqueue(i){ if(!connQueue.includes(i)) connQueue.push(i) }
for(let i=1;i<=N;i++) enqueue(i)
setInterval(()=>{ if(connQueue.length){ spawnBot(connQueue.shift()) } }, 5500)
// status periodico
setInterval(()=>{
  const alive=bots.filter(b=>b.entity)
  const avgFood = alive.length? (alive.reduce((s,b)=>s+(b.food??0),0)/alive.length).toFixed(1):0
  // lingua = manual PT compartilhado; mede PALAVRAS NOVAS aprendidas do player + tipos observados (curiosidade)
  const learned = new Set(); alive.forEach(b=> Object.keys(b.brain.bound||{}).forEach(w=>learned.add(w)))
  const bestFit = HALL.length? HALL[0].fit.toFixed(0):0
  const nmAvg = alive.length? {da:0,ne:0,se:0}:null
  if(nmAvg){ alive.forEach(b=>{ if(b.brain.nm){nmAvg.da+=b.brain.nm.da;nmAvg.ne+=b.brain.nm.ne;nmAvg.se+=b.brain.nm.se} }); for(const k in nmAvg)nmAvg[k]=(nmAvg[k]/alive.length).toFixed(2) }
  const attAvg = alive.length? (alive.reduce((s,b)=>s+(b.brain.attach||0.3),0)/alive.length).toFixed(2):0
  const S=skills.STATS; const semLLM=S.used_no_llm, comLLM=S.minted_llm, frac=(semLLM+comLLM)>0?(semLLM/(semLLM+comLLM)):0
  console.log(`[colonia] vivos=${alive.length}/${N} comida=${avgFood} GER=${GEN} apt-max=${bestFit} cunhadas=${Object.keys(COINED).length} vocab=${learned.size} APEGO-a-voce=${attAvg} | da=${nmAvg?nmAvg.da:0} ne=${nmAvg?nmAvg.ne:0} se=${nmAvg?nmAvg.se:0}`)
  // COMPOSITOR DE FUNDO (SEM LLM): a cada janela, recombina 2 habilidades provadas numa nova. Entra FRACA;
  // o loop de recompensa a fortalece se ajudar e a poda a descarta se nao (variacao-e-selecao, teste por execucao).
  try{ const sn=skills.mintNoLLM(); if(sn){ skills.addSkill(sn); skills.saveLib() } }catch(_){}
  // TECH TICK GARANTIDO: destrava a tech tree independente da competicao do workspace. Um bot ativo tenta
  // EXTRAIR madeira/pedra e AVANCAR o crafting (techCraft em loop) a cada janela -> progressao visivel mais rapida.
  // FERREIRO FIXO: sempre o mesmo bot conduz o tech tick -> ACUMULA recursos (sem round-robin) p/ chegar aos ~17
  // cobblestone do forno. Se o ferreiro morrer, o proximo vivo assume. Os outros 5 seguem a vida emergente normal.
  ;(async()=>{ try{ const b=alive.find(x=>x.entity)||null; if(!b) return
    const api=skills.makeAPI(b, goals)
    if(!api.hasItem('log')&&!api.hasItem('planks')) await api.getWood()
    if(!api.hasItem('pickaxe')) await api.techCraft()        // garante picareta antes de minerar
    if(api.hasItem('pickaxe')) await api.mineStone()
    const made=await api.techCraft(); if(made) try{ b.chat('craftei:'+made) }catch(e){}
  }catch(_){} })()
  // FORJA DE PRIMITIVAS (motor babbling, SEM LLM): um bot SEGURO experimenta uma acao crua nova e, se o EFEITO
  // for inedito, vira uma PRIMITIVA nomeada que entra na biblioteca (e dali pode ser reusada/composta/chunkada).
  ;(async()=>{ try{ const b=alive.find(x=>x.entity&&(x.health??20)>=12&&(x.food??20)>=8); if(!b) return
    const prim=await forge.babble(b, (code)=> skills.runSkill({code}, b, goals))
    if(prim){ prim.fp=skills.situationFP(b); skills.addSkill(prim); skills.saveLib(); try{ b.chat('inventei:'+prim.effect) }catch(e){} }
  }catch(_){} })()
  // PRESSAO EXTERNA: NOVIDADE no mundo (player coloca algo nunca-visto) -> a colonia DETECTA (percepcao falha),
  // REGISTRA o tipo como nova feature candidata (a forja perceptual cresce a partir do estimulo de FORA) e INVESTIGA
  // (affordance babbling). E o unico lever que cruza o fecho: informacao de fora vira capacidade nova. SEM truque:
  // eles so VEEM e INTERAGEM com o que esta no mundo (nao treino com codigo).
  ;(async()=>{ try{ const b=alive.find(x=>x.entity&&(x.health??20)>=10); if(!b) return
    const novel=b.findBlock({matching:x=>x&&x.name&&!NATURAL.test(x.name)&&!KNOWN_TYPES.has(x.name), maxDistance:8})
    if(novel){ const t=novel.name; KNOWN_TYPES.add(t); NOVELTY.detected++; NOVELTY.last=t
      try{ NOVEL_POS.push({x:novel.position.x,y:novel.position.y,z:novel.position.z}); if(NOVEL_POS.length>12) NOVEL_POS.shift() }catch(e){}  // CONVIVENCIA
      if(pforge.registerNovel(t)) NOVELTY.registered++          // tipo novo -> NOVA feature perceptual (espaco cresce!)
      try{ b.chat('o que e isso? '+t.slice(0,24)) }catch(e){}     // curiosidade vocalizada
      // INVESTIGA: vai ate o objeto novo e tenta uma affordance (motor babbling de objeto)
      try{ const api=skills.makeAPI(b, goals)
        if(novel.position.distanceTo(b.entity.position)>3){ b.pathfinder.setGoal(new goals.GoalNear(novel.position.x,novel.position.y,novel.position.z,2)); await new Promise(r=>setTimeout(r,1500)) }
        const r=await skills.tryAffordance(api, b.blockAt(novel.position)||novel); if(r&&r.learned){ NOVELTY.investigated++ }
      }catch(e){}
    }
  }catch(_){} })()
  // DWELL/CONVIVENCIA: a curiosidade leva um bot calmo a CONVIVER perto de um objeto novo do player. Isso da
  // VARIANCIA a feature "perto_X" (as vezes perto, as vezes longe, com efeitos diferentes) -> a forja perceptual
  // pode finalmente PROMOVER a percepcao. Sem isso, perto_X fica sempre 'nao' e nunca discrimina nada (esparsidade).
  try{ if(NOVEL_POS.length){ const cands=alive.filter(x=>x.entity&&(x.health??20)>=10&&(x.food??20)>=9).slice(0,3)
    cands.forEach((b,k)=>{ const p=NOVEL_POS[(GEN+k)%NOVEL_POS.length]
      b.pathfinder.setGoal(new goals.GoalNear(p.x,p.y,p.z,1)); b.brain.lastAction='conviver_novidade' }) } }catch(e){}  // ate 3 bots COLADOS nos objetos -> da variancia a perto_X
  // ENSINO GUIADO POR TEORIA DA MENTE (H99/MiniMax): o TUTOR (proximo de prosperar) acha o APRENDIZ por perto cuja
  // FRONTEIRA ele consegue preencher (modela o que o outro NAO sabe mas esta pronto) e ensina EXATAMENTE isso.
  try{
    const ranked=alive.filter(b=>b.entity).map(b=>({b,score:(b.food??0)+(b.health??0)})).sort((x,y)=>y.score-x.score)
    if(ranked.length>=2){
      const tutor=ranked[0].b
      for(const {b:learner} of ranked.slice(1)){
        if(!learner.entity||!tutor.entity) continue
        if(tutor.entity.position.distanceTo(learner.entity.position)>14) continue   // precisa estar PERTO (embodido)
        const fr=mind.teachableFrontier(tutor.brain, learner.brain, COINED)
        if(fr){ const novo=mind.applyTeaching(learner.brain, fr); TOM.taught++
          if(novo) TOM.frontierHits++
          if(fr.kind==='threat') TOM.threatTaught++; else TOM.vocabTaught++
          // o tutor VOCALIZA o ensino (embodido): grita o simbolo da fronteira p/ o aprendiz
          try{ const sym=fr.symbol||fr.symbol; if(sym && (!tutor.brain._lastTeach||Date.now()-tutor.brain._lastTeach>3000)){ tutor.brain._lastTeach=Date.now(); tutor.chat((sym||'isso')+' (cuidado)') } }catch(e){}
          break }   // 1 ensino por janela (bounded)
      }
    }
  }catch(e){}
  try{ pforge.promote() }catch(e){}   // FORJA PERCEPTUAL: promove feature que divide classe irredutivel
  try{ REPLAYED += wm.prioritizedReplay(24) }catch(e){}   // REPLAY PRIORIZADO (Mattar&Daw): propaga valor nas transicoes de maior utilidade
  // OPCAO HIERARQUICA garantida: um bot calmo roda a rotina mais forte como UNIDADE, e a SR aprende a transicao
  // ABSTRATA (classe-inicio -> classe-fim da rotina inteira), nao passo-a-passo. Abstracao temporal (Sutton-Precup-Singh).
  ;(async()=>{ try{ const opt=skills.bestOption&&skills.bestOption(); if(!opt) return
    const b=alive.find(x=>x.entity&&(x.health??20)>=12&&(x.food??20)>=10); if(!b) return
    const c0=wm.classify(skills.makeAPI(b,goals), b)
    await skills.runSkill(opt, b, goals); opt.uses=(opt.uses||0)+1; OPT.ran++; b.brain.lastAction='opcao:'+opt.name
    const c1=wm.classify(skills.makeAPI(b,goals), b); wm.observeTransition(c0, 'opcao:'+opt.name.slice(0,12), c1)  // transicao ABSTRATA
  }catch(e){} })()
  const af=skills.affStats(); const wms=wm.stats()
  const semL=(S.minted_compose||0)+(S.minted_mutate||0)+(S.chunked||0), criadas=semL+comLLM, fracCriada=criadas>0?(semL/criadas):0
  const cks=skills.chunkStats(); const fg=forge.forgeStats()
  console.log(`[SKILLS] lib=${skills.libSize()} acoes-sem-LLM=${semLLM}(${(frac*100).toFixed(0)}%) | CRIADAS: comp=${S.minted_compose||0} mut=${S.minted_mutate||0} CHUNK=${S.chunked||0}(${cks.abstracted}/${cks.seen}) PRIMITIVAS-INVENTADAS=${fg.primitivas}(tried=${forge.STAT.tried}) llm=${comLLM} -> SEM-LLM=${(fracCriada*100).toFixed(0)}% | LOOT=${S.loot||0} RACIOCINIO=${S.diagnoses||0}`)
  const pf=pforge.stats(); const nv=pforge.novelStats()
  // ===== MOTOR ABERTO (PowerPlay+OMNI): a colonia INVENTA e persegue seus proprios desafios escalonados =====
  try{
    if(!SPAWN_REF && alive[0]&&alive[0].entity) SPAWN_REF=alive[0].entity.position.clone()
    let maxDist=0, resTotal=0
    for(const b of alive){ if(!b.entity) continue
      if(SPAWN_REF) maxDist=Math.max(maxDist, b.entity.position.distanceTo(SPAWN_REF))
      try{ resTotal += b.inventory.items().filter(i=>/log|planks|cobble|stone|coal|iron|beef|porkchop|bread/.test(i.name)).reduce((s,i)=>s+i.count,0) }catch(e){} }
    const vals={ explorar:maxDist, juntar:resTotal, vocab:Object.keys(COINED).length, skills:skills.libSize(),
                 tech:prog.stats().level, transferir:(wm.transferStats?wm.transferStats().applied:0),
                 abstrair:(pforge.stats?pforge.stats().invented:0) }
    ACTIVE_CHALLENGE = curriculum.step(vals)
  }catch(e){}
  const cs=curriculum.stats()
  console.log(`[MOTOR-ABERTO] desafios-DOMINADOS=${cs.dominados} | ativo=${ACTIVE_CHALLENGE?ACTIVE_CHALLENGE.dim+'(falta '+ACTIVE_CHALLENGE.gap.toFixed(0)+' p/ '+ACTIVE_CHALLENGE.bar.toFixed(0)+')':'-'} | fronteira=${JSON.stringify(cs.fronteira)} ult=${cs.ultimos.join(',')}`)
  const neu=wm.neuro?wm.neuro():{ach:0,ne:0}; const tr=wm.transferStats?wm.transferStats():{abs:0,applied:0}
  console.log(`[CEREBRO] wm:${wms.classes}cls/${wms.cells}cel relacoes=${wms.learned} trans=${wms.trans||0} SR=${wms.srRows||0}/${wms.srLinks||0} | PERCEPCOES=${pf.invented}(${pf.last||'-'}) | [IMAGINACAO-SR] planejou=${PLANN.acted} evitou=${PLANN.avoided} | replay-prior=${REPLAYED} OPCOES=${OPT.ran} | NEURO ach=${neu.ach} ne=${neu.ne} | [HIERARQUIA] abstrato=${tr.abs}cls TRANSFERIU=${tr.applied}`)
  console.log(`[PRESSAO-EXTERNA] novidades-vistas=${NOVELTY.detected} tipos-registrados=${NOVELTY.registered}(${nv.list.join(',')||'-'}) investigadas=${NOVELTY.investigated} | ultima=${NOVELTY.last||'-'}`)
  const ps=prog.stats(); console.log(`[PROGRESSAO] tech tree: ${ps.level}/${ps.of} marcos | conquistados: [${ps.achieved.join(', ')||'nenhum ainda'}] | ToM-modelos=${mind.tomStats()} | [ENSINO-ToM] ensinou=${TOM.taught} fronteira-NOVA=${TOM.frontierHits}(ameaca=${TOM.threatTaught} vocab=${TOM.vocabTaught})`)
  const hs=hippo.stats(); const compreendidos=bots.reduce((s,b)=>s+((b.brain&&b.brain._comprehended)||0),0)
  console.log(`[HIPOCAMPO] lugares=${hs.places} (comida=${hs.foodSpots} morte=${hs.deathSpots}) ep=${hs.episodes} | [LINGUAGEM] simbolos-ancorados=${language.stats()} compreensoes=${compreendidos} | [WORKSPACE] ${workspace.topWinners(4).join(' ')}`)
}, 10000)
