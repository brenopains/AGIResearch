// forge.js — FORJA DE PRIMITIVAS ABERTAS (open-ended primitive discovery, SEM LLM).
// Hoje os skills recombinam 10 verbos do despachante. Aqui o agente INVENTA verbos NOVOS *abaixo* desse nivel:
// combina as chamadas CRUAS do corpo (mover/cavar/colocar/olhar/pular/atacar/equipar) com parametros amostrados,
// EXECUTA o candidato no mundo, e mede o EFEITO (delta de posicao/inventario/vida/fome). Mantem os que produzem
// um efeito CONSISTENTE e NOVO (assinatura inedita) num ARQUIVO de novidade -> generatividade de BAIXO pra cima.
// Fundamento: balbucio motor (bebes descobrem o proprio repertorio agindo), modelos internos do cerebelo,
// busca por NOVIDADE / qualidade-diversidade (Lehman & Stanley). NENHUMA chamada de LLM.

// ====== ALFABETO MOTOR (fragmentos de codigo cru que usam SO o 'api') ======
// cada gerador retorna um trecho JS curto; o forge concatena 2-4 em sequencia -> candidato a primitiva nova
const ALPHABET = [
  ()=>`api.sprint(true); await api.wait(${250+(_n()%400)});`,
  ()=>`api.jump(true); await api.wait(${150+(_n()%250)}); api.jump(false);`,
  ()=>`await api.digFront();`,
  ()=>`await api.digBelow();`,
  ()=>`await api.placeBelow();`,
  ()=>{ const a=(_n()%628)/100; return `{ const p=api.pos; api.goto(p.x+${(Math.cos(a)*3).toFixed(1)}, p.y, p.z+${(Math.sin(a)*3).toFixed(1)}); }`; },
  ()=>`{ const h=api.nearestHostile()||api.nearestPrey(); if(h){ await api.look(h.position); await api.attack(h); } }`,
  ()=>`{ const t=api.nearestPrey(); if(t) await api.look(t.position); }`,
  ()=>`await api.equipTool('pickaxe');`,
  ()=>`await api.equipTool('axe');`,
  ()=>`await api.eat();`,
];
let _seed=12345; function _n(){ _seed=(_seed*1103515245+12345)&0x7fffffff; return _seed; }   // PRNG deterministico (Math.random e proibido no harness de workflow; aqui no node ok, mas mantemos reprodutivel)

// gera o CORPO de uma primitiva candidata = sequencia de 2-4 fragmentos do alfabeto
function genCandidate(){
  const k=2+(_n()%3);                       // 2..4 passos
  const parts=[]; for(let i=0;i<k;i++){ parts.push(ALPHABET[_n()%ALPHABET.length]()); }
  return `async(api)=>{ ${parts.join(' ')} }`;
}

// ====== ASSINATURA DE EFEITO + ARQUIVO DE NOVIDADE ======
function snapshot(bot){
  const e=bot.entity; const pos=e?e.position.clone?e.position.clone():{x:e.position.x,y:e.position.y,z:e.position.z}:{x:0,y:0,z:0};
  let inv=0; try{ inv=bot.inventory.items().reduce((s,i)=>s+i.count,0); }catch(_){}
  return { x:pos.x, y:pos.y, z:pos.z, inv, hp:bot.health??20, food:bot.food??20 };
}
function sig(a,b){
  const dist=Math.hypot(b.x-a.x, b.z-a.z), dy=b.y-a.y;
  const parts=[];
  parts.push(dist>1.5?'andou':(dist>0.4?'passo':'parado'));
  parts.push(dy<-0.6?'desceu':(dy>0.6?'subiu':'plano'));
  parts.push(b.inv>a.inv?'ganhou':(b.inv<a.inv?'gastou':'inv0'));
  parts.push(b.hp<a.hp?'dano':(b.hp>a.hp?'curou':'hp0'));
  parts.push(b.food<a.food?'fome':(b.food>a.food?'comeu':'food0'));
  return parts.join('|');
}
function trivial(s){ return s==='parado|plano|inv0|hp0|food0'; }   // no-op puro

const ARCHIVE = {};   // assinatura -> {n, sample(code)}
function forgeStats(){ const ks=Object.keys(ARCHIVE); return { primitivas:ks.length, sigs:ks.slice(0,10) }; }
let STAT={ tried:0, discovered:0 };

// ====== BABBLING: gera 1 candidato, executa no mundo, mede efeito; se NOVO e nao-trivial -> nova primitiva ======
// runFn(codeString, bot) deve compilar+executar com timeout (reusa skills.compileSkillFn/runSkill via wrapper externo)
async function babble(bot, runCandidate){
  try{
    if(!bot||!bot.entity) return null;
    STAT.tried++;
    const code=genCandidate();
    const before=snapshot(bot);
    await runCandidate(code);                 // executa o candidato (motor babbling real no mundo)
    await new Promise(r=>setTimeout(r,200));
    const after=snapshot(bot);
    const s=sig(before,after);
    if(trivial(s)) return null;                // nao fez nada -> descarta
    const known=ARCHIVE[s];
    ARCHIVE[s]=ARCHIVE[s]||{n:0, code}; ARCHIVE[s].n++;
    if(!known){                                // EFEITO INEDITO -> abstrai como PRIMITIVA NOVA nomeada
      STAT.discovered++;
      return { name:'prim_'+s.replace(/\|/g,'_'), code, strength:0.5, uses:0, success:0, born:Date.now(), primitive:true, effect:s };
    }
    return null;
  }catch(e){ return null; }
}

module.exports = { genCandidate, babble, forgeStats, sig, snapshot, ARCHIVE, STAT, _reseed:(s)=>{_seed=s;} };
