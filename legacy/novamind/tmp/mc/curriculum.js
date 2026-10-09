// curriculum.js — MOTOR ABERTO (PowerPlay + OMNI): a colonia INVENTA seus proprios proximos desafios.
// PowerPlay (Schmidhuber 2013): busca continuamente o desafio MAIS SIMPLES AINDA-NAO-DOMINADO na fronteira
// aprendivel; ao dominar, SOBE A BARRA -> nunca preso a um teto fixo (a jogada de Godel: adicionar o proximo).
// OMNI (Clune 2025): filtro de INTERESSANCIA = so persegue o que e APRENDIVEL (ha progresso) e NAO-trivial.
// NAO vence Solomonoff (impossivel/desnecessario) — implementa o que o humano faz: expandir o proprio fecho sem parar.
// Mede: quantos desafios auto-inventados a colonia DOMINOU e se a FRONTEIRA segue avancando (crescimento aberto).
const fs=require('fs'), path=require('path');
const CUR_PATH=path.join(__dirname,'curriculum.json');

// DIMENSOES de capacidade (cada uma com uma barra que SOBE quando dominada). Metricas mensuraveis da colonia.
const DIMS = {
  explorar:  { bar:24,  step:m=>m*1.4,  nudge:'explorar',      desc:'distancia max do spawn' },
  juntar:    { bar:8,   step:m=>m+6,    nudge:'cacar',         desc:'total de recurso no inventario' },
  vocab:     { bar:20,  step:m=>m+4,    nudge:null,            desc:'simbolos cunhados' },
  skills:    { bar:90,  step:m=>m+15,   nudge:null,            desc:'habilidades na biblioteca' },
  tech:      { bar:5,   step:m=>m+1,    nudge:'craftar',       desc:'marcos da tech tree' },
  transferir:{ bar:50,  step:m=>m*1.6,  nudge:null,            desc:'transferencias hierarquicas (generalizacao)' },
  abstrair:  { bar:3,   step:m=>m+2,    nudge:null,            desc:'percepcoes inventadas' },
};
let STATE = { bars:{}, mastered:0, history:[], lastVals:{}, lp:{} };  // lp = progresso por dimensao (interessancia)
function load(){ try{ const d=JSON.parse(fs.readFileSync(CUR_PATH,'utf8')); STATE=Object.assign(STATE,d); }catch(e){}
  for(const k in DIMS){ if(STATE.bars[k]==null) STATE.bars[k]=DIMS[k].bar; } return STATE; }
function save(){ try{ fs.writeFileSync(CUR_PATH, JSON.stringify(STATE)); }catch(e){} }

// recebe os valores ATUAIS das metricas e: (1) atualiza progresso (interessancia), (2) detecta DOMINIO -> sobe a barra,
// (3) PROPOE o proximo desafio = a dimensao APRENDIVEL (com progresso) mais perto da barra (PowerPlay: simplest unsolved).
function step(vals){
  const proposals=[];
  for(const k in DIMS){
    const v=vals[k]; if(v==null) continue;
    const prev=STATE.lastVals[k]; if(prev!=null) STATE.lp[k]=(STATE.lp[k]||0)*0.7 + Math.max(0,v-prev)*0.3;  // progresso de aprendizado
    STATE.lastVals[k]=v;
    if(v>=STATE.bars[k]){                              // DOMINOU -> sobe a barra (open-ended)
      const old=STATE.bars[k]; STATE.bars[k]=Math.ceil(DIMS[k].step(STATE.bars[k]));
      STATE.mastered++; STATE.history.push({dim:k,from:old,to:STATE.bars[k],t:STATE.mastered});
      if(STATE.history.length>200) STATE.history.shift();
    }
    // INTERESSANCIA (OMNI): aprendivel = ha progresso OU esta perto da barra; nao-trivial = ainda nao passou
    const gap = STATE.bars[k]-v;                        // quanto falta
    const learnable = (STATE.lp[k]||0) > 0.01 || gap < STATE.bars[k]*0.5;
    if(gap>0 && learnable) proposals.push({dim:k, gap, lp:STATE.lp[k]||0, bar:STATE.bars[k], val:v, nudge:DIMS[k].nudge});
  }
  // PowerPlay: o MAIS SIMPLES ainda-nao-dominado = menor gap relativo entre os aprendiveis
  proposals.sort((a,b)=> (a.gap/Math.max(1,a.bar)) - (b.gap/Math.max(1,b.bar)));
  save();
  return proposals[0]||null;   // desafio ativo (ou null se tudo dominado-momentaneo)
}
function stats(){ const fronteira=Object.fromEntries(Object.keys(DIMS).map(k=>[k,STATE.bars[k]]));
  return { dominados:STATE.mastered, fronteira, ultimos:STATE.history.slice(-3).map(h=>h.dim+'->'+h.to) }; }

module.exports={ load, save, step, stats, DIMS };
