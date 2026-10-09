// vision.js — PERCEPCAO VISUAL HUMANA-LIKE (nao-wallhack, nao melhor que humano).
// Regras do usuario: (1) NAO wallhack -> oclusao por raycast (ja em canSee); (2) NAO melhor que humano ->
// CAMPO DE VISAO limitado (humano ~200deg periferico horizontal, ~120 vertical) + alcance finito + pode VIRAR
// a cabeca p/ escanear (nao tem visao de tras sem se virar). Antes os bots eram ONISCIENTES (sabiam tudo nos
// chunks = melhor que humano). Agora so percebem o que esta no campo visual, sem ocluido.
const HFOV = 100;   // meio-angulo horizontal (200deg total) — periferia humana
const VFOV = 65;    // meio-angulo vertical (130deg)
const RANGE = 32;   // alcance (mesmo do canSee)

// direcao p/ onde o bot olha (horizontal) a partir do yaw do mineflayer
function lookDir(bot){ const yaw=bot.entity.yaw; return { x:-Math.sin(yaw), z:Math.cos(yaw) }; }

// alvo esta dentro do CAMPO DE VISAO (cone) do bot?
function inFOV(bot, pos){
  try{ const e=bot.entity.position; const dx=pos.x-e.x, dz=pos.z-e.z, dy=pos.y-(e.y+1.6);
    const hd=Math.hypot(dx,dz); if(hd<1.2) return true;                 // muito perto = percebe (visao periferica/tato)
    const f=lookDir(bot); const dot=(f.x*dx+f.z*dz)/hd;                 // cos do angulo horizontal
    const hAng=Math.acos(Math.max(-1,Math.min(1,dot)))*180/Math.PI;
    if(hAng>HFOV) return false;
    const vAng=Math.atan2(Math.abs(dy),hd)*180/Math.PI;                 // angulo vertical
    if(vAng>VFOV) return false;
    return true;
  }catch(e){ return true; }
}

// ESCANEAR: vira a cabeca lentamente p/ ampliar o que percebe ao longo do tempo (sacadas/exploracao visual).
// chamado as vezes; muda o yaw alvo aleatoriamente dentro de um passo (olhar em volta).
async function scan(bot){
  try{ const yaw=bot.entity.yaw + (Math.random()*1.6-0.8); const pitch=(Math.random()*0.5-0.25);
    await bot.look(yaw, pitch, false); }catch(e){}
}

module.exports={ inFOV, scan, HFOV, VFOV, RANGE, lookDir };
