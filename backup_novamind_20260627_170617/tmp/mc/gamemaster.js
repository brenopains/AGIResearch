// gamemaster.js — "O MUNDO": bot OPADO que mantem a fauna renovavel perto da colonia.
// NAO alimenta os bots na mao — apenas faz o mundo ser VIVO (animais reaparecem), pra eles cacarem.
// Teleporta ate um Nova aleatorio e invoca alguns animais por perto, periodicamente.
const mineflayer = require('mineflayer');
const HOST='localhost', PORT=25565;
let bot, alive=false;

function connect(){
  bot = mineflayer.createBot({ host:HOST, port:PORT, username:'Mundo', auth:'offline', version:'1.20.4' });
  bot.on('spawn', ()=>{ alive=true; console.log('[Mundo] online (op). Mantendo fauna viva (bounded).');
    try{ bot.chat('/gamemode spectator'); }catch(e){}
    // CULL INICIAL: limpa o acumulo de animais de antes (que causava OOM)
    setTimeout(()=>{ for(const a of ['cow','pig','chicken','sheep']){ try{ bot.chat(`/kill @e[type=minecraft:${a}]`); }catch(e){} } console.log('[Mundo] cull inicial (limpou backlog de fauna)'); }, 2000);
  });
  bot.on('end', ()=>{ alive=false; setTimeout(connect, 5000); });
  bot.on('error', (e)=>{ /* silencioso */ });
  bot.on('kicked', ()=>{ alive=false; });
}
connect();

const ANIMALS=['cow','cow','pig','chicken','sheep'];
let cycle=0;
function tick(){
  if(!alive){ return; }
  try{
    cycle++;
    // CULL periodico (a cada ~2min): mata animais p/ BOUND a populacao (evitava OOM por acumulo infinito de entidades)
    if(cycle%4===0){ for(const a of ['cow','pig','chicken','sheep']){ try{ bot.chat(`/kill @e[type=minecraft:${a}]`); }catch(e){} }
      console.log('[Mundo] cull (reset da fauna p/ bound de memoria)'); }
    const novaNames = Object.keys(bot.players||{}).filter(n=>n.startsWith('Nova_'));
    if(!novaNames.length){ return; }
    const tgt = novaNames[Math.floor(Math.random()*novaNames.length)];
    bot.chat(`/tp Mundo ${tgt}`);
    setTimeout(()=>{
      const n = 2;                                // POUCOS por ciclo (bounded)
      for(let i=0;i<n;i++){ const a=ANIMALS[Math.floor(Math.random()*ANIMALS.length)];
        const dx=(Math.random()*5-2.5).toFixed(1), dz=(Math.random()*5-2.5).toFixed(1);
        try{ bot.chat(`/summon minecraft:${a} ~${dx} ~1 ~${dz}`); }catch(e){}
      }
      console.log(`[Mundo] +${n} animais perto de ${tgt}`);
      // DESLIGADO (pedido do usuario): NAO spawnar madeira/pedra na mao — estava quebrando o mapa (setblock/fill).
      // Eles vao FORRAGEAR de verdade numa seed rica em recursos (sem trapaca). So a fauna e mantida (mobs despawnam).
    }, 700);
  }catch(e){}
}
setInterval(tick, 30000);   // a cada 30s (mais lento, bounded)
setTimeout(tick, 4000);
