// transfer.js — TESTE DE TRANSFERENCIA CROSS-DOMINIO (o teste decisivo de GENERALIDADE rumo a AGI).
// Hipotese: o cerebro (skills_library + worldmodel + hippocampus) foi aprendido numa vida de FORRAGEIO/TECH.
// A AGI exige que o MESMO nucleo generalize para uma tarefa de TIPO DIFERENTE sem reajuste. Aqui a tarefa nova e
// SOBREVIVER A UMA NOITE HOSTIL (arena de combate) — algo que NAO foi o objetivo de treino. Comparamos:
//   TREINADO (carrega skills/worldmodel/hippocampus aprendidos)  vs  ZERADO (cerebro vazio).
// Metrica: tempo de sobrevivencia medio + n de mortes na janela. Se TREINADO > ZERADO de forma consistente ->
// PRIMEIRO SINAL HONESTO de transferencia. Se empatar -> o aprendizado e especifico-demais (achado honesto).
// Uso: node transfer.js loaded   |   node transfer.js wiped     (rode os dois e compare os numeros)
const mineflayer=require('mineflayer');
const { pathfinder, Movements, goals }=require('mineflayer-pathfinder');
const skills=require('./skills'); const wm=require('./worldmodel'); const hippo=require('./hippocampus');

const MODE=(process.argv[2]||'loaded').toLowerCase();   // 'loaded' | 'wiped'
const HOST='localhost', PORT=25565, N=3, DUR_MS=(parseInt(process.argv[3])||180)*1000; // 3 min de noite (override: argv[3] seg)
if(MODE==='loaded'){ skills.loadLib(); wm.load(); hippo.load(); }   // ZERADO: nao carrega nada (cerebro em branco)
console.log(`[TRANSFER] modo=${MODE} | skills=${skills.libSize()} | wm=${wm.stats().cells} celulas`);

const HOSTILE=new Set(['zombie','skeleton','spider','creeper','enderman']);
const bots=[]; const stats={ spawn:{}, deaths:0, survivedMs:[] };

// ARENA (op, reusa o nome 'Mundo' que ja esta opado): forca NOITE PERMANENTE + dificuldade dificil -> mobs spawnam
let arena;
function startArena(){
  arena=mineflayer.createBot({host:HOST,port:PORT,username:'Mundo',auth:'offline',version:'1.20.4'});
  arena.on('spawn',()=>{ console.log('[Arena] online'); try{
    arena.chat('/gamemode spectator'); arena.chat('/gamerule doDaylightCycle false'); arena.chat('/time set 18000'); arena.chat('/difficulty hard');
  }catch(e){} setTimeout(spawnBots,2000); });
  arena.on('end',()=>setTimeout(startArena,4000)); arena.on('error',()=>{});
}
// a cada 15s teleporta ate um Trans e invoca zumbis perto (garante pressao de combate)
setInterval(()=>{ if(!arena||!arena.entity) return; const t=Object.keys(arena.players||{}).find(n=>n.startsWith('Trans_')); if(!t) return;
  try{ arena.chat('/tp Mundo '+t); setTimeout(()=>{ for(let i=0;i<3;i++){ const dx=(Math.random()*6-3).toFixed(1), dz=(Math.random()*6-3).toFixed(1); arena.chat(`/summon minecraft:zombie ~${dx} ~ ~${dz}`); } },600); }catch(e){}
}, 15000);

function makeBot(i){
  const name='Trans_'+(i+1);
  const bot=mineflayer.createBot({host:HOST,port:PORT,username:name,auth:'offline',version:'1.20.4'});
  bot.loadPlugin(pathfinder);
  bot.once('spawn',()=>{ stats.spawn[name]=Date.now(); const mv=new Movements(bot); bot.pathfinder.setMovements(mv);
    bot.brain={ threat:{}, arch:{}, value:{} };   // estado afetivo minimo (TREINADO herda via skills/wm; ZERADO comeca seco)
    console.log(`[${name}] spawn (${MODE})`); loop(bot,name); });
  bot.on('death',()=>{ stats.deaths++; const t=Date.now()-(stats.spawn[name]||Date.now()); stats.survivedMs.push(t); console.log(`[${name}] MORREU apos ${(t/1000).toFixed(0)}s`); stats.spawn[name]=Date.now(); });
  bot.on('end',(r)=>{ console.log(`[${name}] end: ${r}`); }); bot.on('kicked',(r)=>{ console.log(`[${name}] KICKED: ${JSON.stringify(r).slice(0,120)}`); }); bot.on('error',(e)=>{ console.log(`[${name}] ERRO: ${e.message||e}`); });
  return bot;
}
function spawnBots(){ // ESCALONA logins (Paper throttla conexoes <4s do mesmo IP) -> 1 bot a cada 5s
  for(let i=0;i<N;i++){ setTimeout(()=>bots.push(makeBot(i)), i*5000); }
  setTimeout(report, DUR_MS + N*5000); }

// LOOP DE DECISAO: usa o MESMO mecanismo de skills/worldmodel. TREINADO tem biblioteca; ZERADO so reflexo cru.
async function loop(bot,name){
  while(bot.entity){
    try{
      const api=skills.makeAPI(bot, goals);
      const nh=api.nearestHostile();
      // tarefa nova = sobreviver ao combate. 1) se ha skill aprendida pra situacao, usa (TRANSFERENCIA). 2) senao reflexo.
      const fp=skills.situationFP(bot); const sk=skills.bestSkill(fp);
      if(sk){ await skills.runSkill(sk,bot,goals); }                  // <-- skill TRANSFERIDA da vida de forrageio/tech
      else if(nh && api.dist(nh)<8){ if(api.dist(nh)<3){ await api.attack(nh); } else { api.fleeFrom(nh.position); } }
      else { const p=api.pos; api.goto(p.x+(Math.random()*16-8), p.y, p.z+(Math.random()*16-8)); }   // baseline: vagar
      if((bot.food??20)<14) await api.eat();
    }catch(e){}
    await new Promise(r=>setTimeout(r, 900+Math.random()*400));
  }
}

function report(){
  const surv=stats.survivedMs; const mean=surv.length? (surv.reduce((a,b)=>a+b,0)/surv.length/1000):'(nenhuma morte=sobreviveu tudo)';
  const aliveNow=bots.filter(b=>b.entity).length;
  console.log(`\n===== TRANSFER [${MODE}] RESULTADO =====`);
  console.log(`vivos no fim: ${aliveNow}/${N} | mortes totais: ${stats.deaths} | sobrevivencia media por vida: ${typeof mean==='number'?mean.toFixed(0)+'s':mean}`);
  console.log(`skills disponiveis: ${skills.libSize()} | (TREINADO usa biblioteca aprendida; ZERADO=0)`);
  console.log(`INTERPRETACAO: compare 'loaded' vs 'wiped'. loaded sobreviver MAIS = transferencia real do cerebro.`);
  process.exit(0);
}
startArena();
