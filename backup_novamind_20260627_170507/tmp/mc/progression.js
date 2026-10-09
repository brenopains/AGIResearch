// progression.js — CURRICULO AUTOMATICO / PROGRESSAO DE CAPACIDADE (a peca que faltava p/ AGI aberta).
// Voyager (pesquisado): inteligencia aberta = perseguir capacidades progressivamente mais dificeis,
// subindo a tech tree. Aqui: detecta MARCOS (do inventario), da RECOMPENSA INTRINSECA por marco NOVO
// (pico de dopamina = motivacao de dominio), e aponta o PROXIMO marco alcancavel (curriculo).
// Marcos sao COMPARTILHADOS (progresso da colonia) e persistentes -> a colonia evolui tecnologicamente.
const fs=require('fs'), path=require('path');
const PROG_PATH = path.join(__dirname,'progression.json');

// tech tree (ordem de dificuldade). cada marco: detector(inv) sobre nomes de itens do inventario.
const TREE = [
  ['madeira',           inv=> inv.some(n=>/_log|_planks/.test(n))],
  // mesa conta se tem a mesa OU qualquer ferramenta de madeira (ferramenta IMPLICA que usou uma mesa)
  ['mesa_trabalho',     inv=> inv.some(n=>n.includes('crafting_table')||/wooden_(pickaxe|axe|sword|shovel)/.test(n))],
  ['ferramenta_madeira',inv=> inv.some(n=>/wooden_(pickaxe|axe|sword|shovel)/.test(n))],
  ['pedra',             inv=> inv.some(n=>/cobblestone|^stone$/.test(n))],
  ['ferramenta_pedra',  inv=> inv.some(n=>/stone_(pickaxe|axe|sword|shovel)/.test(n))],
  ['forno',             inv=> inv.some(n=>n.includes('furnace'))],
  ['comida_cozida',     inv=> inv.some(n=>/cooked_|bread/.test(n))],
  ['carvao',            inv=> inv.some(n=>/coal|charcoal/.test(n))],
  ['ferro',             inv=> inv.some(n=>/iron_ingot|raw_iron/.test(n))],
  ['ferramenta_ferro',  inv=> inv.some(n=>/iron_(pickaxe|axe|sword)/.test(n))],
];

let PROG = { achieved:{}, firstBy:{}, count:0 };   // achieved[marco]=true (colonia); firstBy[marco]=bot
function load(){ try{ PROG=JSON.parse(fs.readFileSync(PROG_PATH,'utf8')); }catch(e){ PROG={achieved:{},firstBy:{},count:0}; } return PROG; }
function save(){ try{ fs.writeFileSync(PROG_PATH, JSON.stringify(PROG)); }catch(e){} }

// checa o inventario do bot -> retorna lista de marcos NOVOS conquistados agora (p/ recompensa)
function checkAchievements(invNames, botName){
  const novos=[];
  for(const [marco,det] of TREE){
    if(det(invNames)){
      if(!PROG.achieved[marco]){ PROG.achieved[marco]=true; PROG.firstBy[marco]=botName; PROG.count++; novos.push({marco, first:true}); }
      // marco pessoal do bot tambem conta p/ motivacao individual mesmo se a colonia ja tem
    }
  }
  if(novos.length) save();
  return novos;
}

// PROXIMO marco do curriculo: o primeiro NAO conquistado (pela colonia) — alvo de dificuldade certa
function nextMilestone(invNames){
  for(const [marco,det] of TREE){ if(!det(invNames) && !PROG.achieved[marco]) return marco; }
  for(const [marco,det] of TREE){ if(!det(invNames)) return marco; }   // ou o proximo pessoal
  return null;   // dominou a arvore (ate ferro)
}

function progressLevel(){ // quantos marcos a colonia ja dominou (0..TREE.length)
  return Object.keys(PROG.achieved).length;
}
function stats(){ return { level:progressLevel(), of:TREE.length, achieved:Object.keys(PROG.achieved) }; }

module.exports={ TREE, load, save, checkAchievements, nextMilestone, progressLevel, stats, getProg:()=>PROG };
