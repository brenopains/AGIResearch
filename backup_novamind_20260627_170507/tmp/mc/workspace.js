// workspace.js — WORKSPACE GLOBAL (Global Neuronal Workspace, Dehaene/Baars).
// Neurociencia (pesquisada): o cerebro tem modulos especializados em PARALELO; a informacao mais SALIENTE
// "IGNITA" e e TRANSMITIDA (broadcast) a todos os modulos -> integracao numa decisao coerente unica.
// Aqui: os modulos cognitivos (sobrevivencia, memoria episodica, modelo de mundo/curiosidade, objetivos,
// teoria da mente, affordances) submetem LANCES (acao + saliencia); o vencedor IGNITA e e executado; o
// resultado e transmitido de volta (broadcast) p/ os modulos atualizarem. Integracao funcional (gap #4).
// Mantem um historico de qual modulo "venceu a consciencia" (p/ inspecao).
const WINS = {};   // modulo -> quantas vezes ignitou
function bid(name, action, salience, run){ return { name, action, salience, run }; }
// recebe lista de lances; escolhe o de maior saliencia (ignicao), com ruido pequeno; executa o run() do vencedor.
async function ignite(bids){
  const valid = bids.filter(b=>b && b.salience>0 && typeof b.run==='function');
  if(!valid.length) return null;
  let best=valid[0], bv=-1e9;
  for(const b of valid){ const s=b.salience + (Math.random()-0.5)*0.05; if(s>bv){ bv=s; best=b; } }
  WINS[best.name]=(WINS[best.name]||0)+1;
  try{ await best.run(); }catch(e){}
  return best;   // o vencedor (broadcast): quem ignitou e qual acao
}
function stats(){ return WINS; }
function topWinners(k=4){ return Object.entries(WINS).sort((a,b)=>b[1]-a[1]).slice(0,k).map(([n,c])=>n+':'+c); }
module.exports={ bid, ignite, stats, topWinners };
