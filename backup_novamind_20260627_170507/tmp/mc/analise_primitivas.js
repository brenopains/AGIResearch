// H-MC: os agentes criam PRIMITIVAS MENTAIS NOVAS a partir da experiencia no mundo? (medido)
const fs=require('fs')
const s=JSON.parse(fs.readFileSync('colony_state.json'))
const MANUAL=new Set(['comida','perigo','vem','cavar','fazer','novo','entendi','ajuda','amigo']) // o que DEMOS
const BODY=new Set(['agachar','pular','correr','colocar','cavar','girar'])
let coined=new Set(), bound={}, afford={}, archByVal={pos:new Set(),neg:new Set()}, selfs=[]
const B=s.brains||{}
for(const i in B){ const b=B[i]
  // 1) SIMBOLOS NOVOS cunhados (palavra que NAO demos, criada do mundo)
  for(const w in (b.bound||{})) if(!MANUAL.has(w)) coined.add(w)
  Object.keys(s.coined||{}).forEach(w=>coined.add(w))
  // 2) PALAVRAS SUAS aprendidas por contexto
  for(const w in (b.bound||{})) bound[w]=b.bound[w]
  // 3) AFFORDANCES aprendidas (acao corporal -> valor de sobrevivencia; primitiva PROCEDURAL nova)
  for(const k in (b.value||{})) if(k.startsWith('corpo:')||['cacar','lutar','hunt'].includes(k)){ afford[k]=(afford[k]||0)+1 }
  // 4) ARQUETIPOS com valencia (categoria-com-sentimento formada da experiencia)
  for(const t in (b.arch||{})){ const v=b.arch[t].val; if(v>0.2)archByVal.pos.add(t); if(v<-0.2)archByVal.neg.add(t) }
  if(b.self) selfs.push(b.self)
}
console.log('='.repeat(70))
console.log('TESTE: primitivas mentais NOVAS formadas pela experiencia no mundo')
console.log('-'.repeat(70))
console.log('1) SIMBOLOS cunhados do mundo (palavras que NAO demos):', [...coined].slice(0,12), '=> total', coined.size)
console.log('2) AFFORDANCES aprendidas (acao->sobrevivencia, primitiva procedural):', Object.keys(afford))
console.log('3) ARQUETIPOS-categoria com SENTIMENTO formados da experiencia:')
console.log('   positivos (atraem):', [...archByVal.pos].slice(0,8))
console.log('   negativos (repelem):', [...archByVal.neg].slice(0,8))
console.log('4) SELF (um "ser" emergente do sonho):', selfs.slice(0,5))
console.log('-'.repeat(70))
console.log('VEREDITO HONESTO:')
const novos = coined.size + Object.keys(afford).length
console.log('  Primitivas NOVAS (simbolos+affordances) formadas da experiencia:', novos)
console.log('  -> SIMBOLOS novos e AFFORDANCES SAO primitivas formadas do mundo (groundadas, nao dadas).')
console.log('  -> MAS as CATEGORIAS (bee, zombie) vem do rotulo do mundo, nao inventadas por eles.')
console.log('  -> Primitiva genuinamente nova = SUPERCATEGORIA emergente (agrupar coisas distintas por sentimento/affordance). Isso e o proximo teste.')
