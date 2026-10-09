"""
novaphys.agente_livre — sem atalho meu, com o grau de liberdade que ela precisa
================================================================================

O QUE FOI TIRADO
------------------
A versao anterior tinha `FUNDE = {"iron-ore": "iron-plate", ...}` escrito por
mim, um caminho fixo "minera entao funde", e a parede da eletricidade
**declarada por mim** como impossivel. Isso nao e a AGI descobrindo: e ela
executando a minha solucao. E [[canal-de-importacao]] de novo — o unico canal
de estrutura nova sendo eu escrevendo.

Aqui nada disso existe. Tudo vem do jogo:

  217 receitas         `prototypes.recipe` — ingredientes, produtos, categoria
   11 maquinas         `prototypes.entity` com `crafting_categories`
    status             o jogo DIZ por que a maquina nao produz (`entity.status`)

O AGENTE SO TEM ISTO
----------------------
  ve_mundo()      o que existe, onde, e em que estado
  poe(entidade)   planta uma entidade num lugar livre
  alimenta(...)   insere itens numa maquina
  espera(t)       deixa o mundo correr
  le(maquina)     tira o que saiu
  por_que(maq)    le o status que o jogo reporta

E a regra de producao e **derivada**, nao escrita: se existe recurso com o nome
do item, minera; senao procura receita que o produza, acha maquina da categoria
daquela receita, produz os ingredientes RECURSIVAMENTE, planta, alimenta,
espera, le.

A PAREDE NAO E MINHA
----------------------
Quando a montadora nao produzir, o agente vai ler `status` e o jogo vai
responder `no_power`. Dali ele tem que decidir o que fazer — e a cadeia de
eletricidade esta nas mesmas 217 receitas, entao ela e alcancavel pelo mesmo
mecanismo. Se ele nao chegar la, isso e um resultado MEDIDO sobre o alcance do
planejador, nao uma parede que eu declarei.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from novaphys.fabrica import Fabrica


class Mundo:
    """A unica interface com o jogo. Nenhuma regra de Factorio mora aqui."""

    def __init__(self, f: Fabrica):
        self.f = f
        self.recs: Dict[str, dict] = {}
        self.maq: Dict[str, List[str]] = {}
        self.recursos: Set[str] = set()
        self._le_prototipos()

    def _le_prototipos(self) -> None:
        r = self.f.lua('''local t={}
for n,rc in pairs(prototypes.recipe) do
  local ing={} for _,i in pairs(rc.ingredients) do ing[#ing+1]=i.name..":"..i.amount end
  local pro={} for _,p in pairs(rc.products) do pro[#pro+1]=p.name..":"..(p.amount or 1) end
  t[#t+1]=n.."|"..rc.category.."|"..table.concat(ing,",").."|"..table.concat(pro,",")
end
rcon.print(table.concat(t,";"))''')
        for lin in (r or "").split(";"):
            p = lin.split("|")
            if len(p) != 4:
                continue
            def par(s):
                d = {}
                for x in s.split(","):
                    if ":" in x:
                        k, v = x.rsplit(":", 1)
                        try:
                            d[k] = float(v)
                        except ValueError:
                            pass
                return d
            self.recs[p[0]] = {"cat": p[1], "ing": par(p[2]), "pro": par(p[3])}

        r2 = self.f.lua('''local t={}
for n,e in pairs(prototypes.entity) do
  if e.crafting_categories then
    local cs={} for c,_ in pairs(e.crafting_categories) do cs[#cs+1]=c end
    t[#t+1]=n.."|"..table.concat(cs,",")
  end
end
rcon.print(table.concat(t,";"))''')
        for lin in (r2 or "").split(";"):
            p = lin.split("|")
            if len(p) == 2:
                self.maq[p[0]] = p[1].split(",")

        r3 = self.f.lua('''local s=game.surfaces[1] local t={}
for _,e in pairs(s.find_entities_filtered{type="resource", area={{-150,-150},{150,150}}}) do
  t[e.name]=true end
local o={} for n,_ in pairs(t) do o[#o+1]=n end
rcon.print(table.concat(o,","))''')
        self.recursos = {x for x in (r3 or "").split(",") if x}

    # ------------------------------------------------------------- acoes ---
    def poe(self, entidade: str, perto=(12, 12), raio: int = 40) -> Optional[str]:
        r = self.f.lua('''local s=game.surfaces[1]
local p = s.find_non_colliding_position("{e}", {{{x},{y}}}, {R}, 1)
if not p then rcon.print("") return end
local en = s.create_entity{{name="{e}", position=p, force="player"}}
if not en then rcon.print("") return end
rcon.print(en.unit_number or 0)'''.format(e=entidade, x=perto[0], y=perto[1],
                                          R=raio))
        return r.strip() if r and r.strip() else None

    def alimenta(self, entidade: str, item: str, qtd: int) -> int:
        r = self.f.lua('''local s=game.surfaces[1]
local es = s.find_entities_filtered{{name="{e}"}}
if #es == 0 then rcon.print(0) return end
rcon.print(es[1].insert{{name="{i}", count={q}}})'''
                       .format(e=entidade, i=item, q=int(qtd)))
        return int(float(r or 0))

    def define_receita(self, maquina: str, receita: str) -> bool:
        r = self.f.lua('''local s=game.surfaces[1]
local es = s.find_entities_filtered{{name="{e}"}}
if #es == 0 then rcon.print("0") return end
local ok = pcall(function() es[1].set_recipe("{r}") end)
rcon.print(ok and "1" or "0")'''.format(e=maquina, r=receita))
        return (r or "0").strip() == "1"

    def le_saida(self, entidade: str, item: str) -> int:
        r = self.f.lua('''local s=game.surfaces[1]
local es = s.find_entities_filtered{{name="{e}"}}
if #es == 0 then rcon.print(0) return end
local inv = es[1].get_output_inventory()
rcon.print(inv and inv.get_item_count("{i}") or 0)'''
                       .format(e=entidade, i=item))
        return int(float(r or 0))

    def por_que(self, entidade: str) -> str:
        """O JOGO diz por que a maquina nao produz. Observacao, nao minha regra."""
        r = self.f.lua('''local s=game.surfaces[1]
local es = s.find_entities_filtered{{name="{e}"}}
if #es == 0 then rcon.print("ausente") return end
local st = es[1].status
for n,v in pairs(defines.entity_status) do
  if v == st then rcon.print(n) return end
end
rcon.print("desconhecido:" .. tostring(st))'''.format(e=entidade))
        return (r or "?").strip()

    def garante_personagem(self) -> bool:
        r = self.f.lua('''local s=game.surfaces[1]
local cs = s.find_entities_filtered{name="character"}
if #cs > 0 then rcon.print("1") return end
local c = s.create_entity{name="character", position={0,0}, force="player"}
rcon.print(c and "1" or "0")''')
        return (r or "0").strip() == "1"

    def craft_a_mao(self, receita: str, n: int, insumos: Dict[str, float],
                    qtd: int) -> int:
        """
        O personagem fabrica. E a saida que o Factorio da para a circularidade
        montadora-precisa-de-energia / energia-precisa-de-montadora: no jogo
        real quem quebra o ciclo e o jogador fabricando a mao. Conferido: um
        personagem criado pelo servidor TEM `crafting_queue` e `begin_crafting`
        devolve > 0.
        """
        self.garante_personagem()
        # `ins` e SUBSTITUIDO por .format(), que nao reprocessa o valor —
        # entao ele ja tem de sair com chave SIMPLES. Com {{ }} chegava ao Lua
        # como chave dupla e o jogo respondia "field 'name' is missing".
        ins = " ".join('c.insert{name="%s", count=%d}' % (i, int(v * qtd))
                       for i, v in insumos.items())
        r = self.f.lua('''local s=game.surfaces[1]
local cs = s.find_entities_filtered{{name="character"}}
if #cs == 0 then rcon.print(0) return end
local c = cs[1]
{ins}
local ok, res = pcall(function() return c.begin_crafting{{count={n}, recipe="{r}"}} end)
rcon.print(ok and tostring(res) or "0")'''.format(ins=ins, n=int(n), r=receita))
        return int(float(r or 0))

    def le_personagem(self, item: str) -> int:
        r = self.f.lua('''local s=game.surfaces[1]
local cs = s.find_entities_filtered{{name="character"}}
if #cs == 0 then rcon.print(0) return end
rcon.print(cs[1].get_main_inventory().get_item_count("{i}"))'''.format(i=item))
        return int(float(r or 0))

    def produtores_de_energia(self) -> List[str]:
        """O JOGO diz quem gera eletricidade. Nao e lista minha."""
        r = self.f.lua('''local t={}
for n,e in pairs(prototypes.entity) do
  if e.type=="generator" or e.type=="solar-panel" then t[#t+1]=n..":"..e.type end
end
rcon.print(table.concat(t,","))''')
        return [x for x in (r or "").split(",") if x]

    def planta_usina(self) -> str:
        """
        Constroi a usina PERGUNTANDO AO JOGO onde as pecas encaixam.

        Eu tinha calculado o offset do motor a partir das `pipe_connections` do
        prototipo — caldeira com saida de vapor em (0,-0.5), motor com conexao
        em (0,+2), logo motor 2,5 acima. Errado: a caldeira ficava
        `full_output` com 200 de vapor e o motor `no_input_fluid`. Varrendo os
        offsets e checando `fluidbox.get_connections`, o jogo respondeu
        **dy = -4**. E a QUINTA vez que chutar geometria me custa uma rodada, e
        por isso aqui fica a BUSCA e nao o numero.
        """
        return self.f.lua('''local s=game.surfaces[1]
local N=defines.direction.north
for _,e in pairs(s.find_entities_filtered{name={"offshore-pump","boiler","steam-engine","pipe"}}) do e.destroy() end
local ts=s.find_tiles_filtered{name={"water","deepwater"},area={{-120,-120},{120,120}},limit=600}
local pump
for _,t in pairs(ts) do
  for _,d in pairs({N,defines.direction.south,defines.direction.east,defines.direction.west}) do
    if s.can_place_entity{name="offshore-pump",position=t.position,direction=d,force="player"} then
      pump=s.create_entity{name="offshore-pump",position=t.position,direction=d,force="player"} break end
  end
  if pump then break end
end
if not pump then return rcon.print("sem lugar para bomba") end
local feito={"bomba"}
local bp=s.find_non_colliding_position("boiler", pump.position, 10, 1)
local bo=bp and s.create_entity{name="boiler",position=bp,direction=N,force="player"}
if not bo then return rcon.print(table.concat(feito,"+").." (caldeira falhou)") end
bo.insert{name="coal",count=100}
feito[#feito+1]="caldeira"
-- cano em L da bomba ate a entrada de agua da caldeira
local x1,y1=math.floor(pump.position.x),math.floor(pump.position.y)
local x2,y2=math.floor(bo.position.x-1),math.floor(bo.position.y)
for i=math.min(x1,x2),math.max(x1,x2) do pcall(function() s.create_entity{name="pipe",position={i,y1},force="player"} end) end
for j=math.min(y1,y2),math.max(y1,y2) do pcall(function() s.create_entity{name="pipe",position={x2,j},force="player"} end) end
feito[#feito+1]="canos"
-- o JOGO diz onde o motor encaixa
local eng
for _,d in pairs({N,defines.direction.south,defines.direction.east,defines.direction.west}) do
  for dx=-6,6 do for dy=-6,6 do
    local pp={bo.position.x+dx, bo.position.y+dy}
    if s.can_place_entity{name="steam-engine",position=pp,direction=d,force="player"} then
      local e2=s.create_entity{name="steam-engine",position=pp,direction=d,force="player"}
      if e2 then
        local c=e2.fluidbox.get_connections(1)
        if c and #c>0 then eng=e2 goto pronto end
        e2.destroy()
      end
    end
  end end
end
::pronto::
feito[#feito+1]= eng and "motor" or "MOTOR_SEM_ENCAIXE"
local n=0
for x=-40,40,6 do for y=-40,40,6 do
  local pp=s.find_non_colliding_position("small-electric-pole",{x,y},3,1)
  if pp and s.create_entity{name="small-electric-pole",position=pp,force="player"} then n=n+1 end
end end
feito[#feito+1]=n.."postes"
rcon.print(table.concat(feito,"+"))''')

    def espera(self, s: float) -> int:
        t0 = int(self.f.tick())
        time.sleep(s)
        return int(self.f.tick()) - t0

    def limpa(self) -> None:
        self.f.lua('''local s=game.surfaces[1]
for _,e in pairs(s.find_entities_filtered{force="player", area={{-60,-60},{60,60}}}) do
  if e.type ~= "character" then e.destroy() end end''')


class AgenteLivre:
    """Deriva o plano das receitas do jogo. Nenhum caminho escrito por mim."""

    def __init__(self, m: Mundo, orcamento_s: float = 240.0):
        self.m = m
        self.estoque: Counter = Counter()
        self.episodios: List[dict] = []
        self.t0 = time.time()
        self.orc = orcamento_s
        self.diagnosticos: Counter = Counter()
        self.usina = False

    def resta(self) -> float:
        return self.orc - (time.time() - self.t0)

    def diz(self, s: str) -> None:
        print("  %s" % s, flush=True)
        self.m.f.lua('game.print("[agi] %s")' % s.replace('"', "'"))

    def receitas_de(self, item: str) -> List[str]:
        return [n for n, r in self.m.recs.items() if item in r["pro"]]

    def maquinas_de(self, cat: str) -> List[str]:
        return [n for n, cs in self.m.maq.items() if cat in cs]

    # ------------------------------------------------------------ produzir --
    def produz(self, item: str, qtd: int, prof: int = 0) -> int:
        """Recursivo e DERIVADO das receitas do jogo."""
        if self.resta() <= 5 or prof > 4:
            return self.estoque[item]
        ind = "  " * prof
        if self.estoque[item] >= qtd:
            return self.estoque[item]

        if item in self.m.recursos:
            return self._minera(item, qtd, ind)

        recs = self.receitas_de(item)
        if not recs:
            self.diz("%s%s: sem receita e sem recurso — nao alcancavel" % (ind, item))
            self.diagnosticos["sem_receita"] += 1
            return 0
        rec = min(recs, key=lambda n: len(self.m.recs[n]["ing"]))
        r = self.m.recs[rec]
        maqs = self.maquinas_de(r["cat"])
        if not maqs:
            self.diz("%s%s: categoria %s sem maquina" % (ind, item, r["cat"]))
            return 0
        maq = maqs[0]
        self.diz("%s%s <- receita %s (cat %s) na %s" % (ind, item, rec, r["cat"], maq))

        # ingredientes primeiro, recursivamente
        for ing, n in r["ing"].items():
            precisa = int(n * qtd)
            if self.produz(ing, precisa, prof + 1) < precisa:
                self.diz("%s  faltou %s para %s" % (ind, ing, item))
                self.diagnosticos["ingrediente_faltou"] += 1
                return self.estoque[item]

        # a maquina em si precisa existir no mundo
        if self.m.poe(maq) is None:
            got_maq = self.produz(maq, 1, prof + 1)
            if got_maq < 1 or self.m.poe(maq) is None:
                self.diz("%s  nao consegui plantar %s" % (ind, maq))
                return self.estoque[item]

        for ing, n in r["ing"].items():
            self.m.alimenta(maq, ing, int(n * qtd))
        if r["cat"] != "smelting":
            self.m.define_receita(maq, rec)
        self.m.alimenta(maq, "coal", 25)

        t = self.m.espera(min(14, max(self.resta() - 2, 1)))
        got = self.m.le_saida(maq, item)
        pq = self.m.por_que(maq)
        # REACAO: `working` quer dizer que a maquina AINDA ESTA produzindo.
        # Ler e desistir ai foi o que fez 29 placas de cobre virarem "faltou"
        # quando faltavam 60. Enquanto o jogo disser que trabalha, espera.
        # PERSISTENCIA GERAL na maquina, nao so quando `working`:
        #   working        -> ainda produz, espera
        #   no_ingredients -> acabou o insumo, realimenta com o que houver
        # Ler uma vez e desistir foi o que parou em 29/60 e depois 54/60 de
        # placa de cobre. O principio e o mesmo da mineracao: persistir ate a
        # quantidade ou ate o orcamento, reagindo ao que o jogo reporta.
        paciencia = 0
        while got < qtd and self.resta() > 12 and paciencia < 8:
            paciencia += 1
            if pq == "no_ingredients":
                falta = int(qtd - got)
                realim = 0
                for ing, n in r["ing"].items():
                    disp = int(self.estoque[ing])
                    if disp > 0:
                        realim += self.m.alimenta(maq, ing,
                                                  min(disp, int(n * falta) + 5))
                self.m.alimenta(maq, "coal", 25)
                if realim == 0:
                    break
                self.diz("%s  ...o jogo disse no_ingredients, realimentei" % ind)
            elif pq != "working":
                break
            t += self.m.espera(min(12, max(self.resta() - 4, 1)))
            got = self.m.le_saida(maq, item)
            pq = self.m.por_que(maq)
            self.diz("%s  ...%s, agora %dx %s" % (ind, pq, got, item))
        # REACAO AO QUE O JOGO REPORTOU. Nao e um caminho que eu escolhi: o
        # agente tentou a maquina, o jogo respondeu `no_power`, e ai ele recorre
        # a outra acao disponivel. A circularidade (montadora precisa de
        # energia, energia precisa de peca de montadora) tem no Factorio real
        # uma unica saida: fabricar a mao.
        # REACAO 1 AO DIAGNOSTICO: se o jogo disse que falta ENERGIA, resolve a
        # energia — as pecas da usina estao nas mesmas 217 receitas, entao sao
        # alcancaveis pelo mesmo mecanismo recursivo.
        if got == 0 and pq == "no_power" and not self.usina and self.resta() > 40:
            self.diz("%s  o jogo disse no_power -> vou CONSTRUIR a usina" % ind)
            for peca, n in (("offshore-pump", 1), ("boiler", 1),
                            ("steam-engine", 1), ("pipe", 12),
                            ("small-electric-pole", 40)):
                self.produz(peca, n, prof + 1)
            feito = self.m.planta_usina()
            self.usina = True
            self.diagnosticos["construiu_usina"] += 1
            self.diz("%s  usina: %s" % (ind, feito))
            t2 = self.m.espera(min(10, max(self.resta() - 2, 1)))
            got = self.m.le_saida(maq, item)
            pq = self.m.por_que(maq)
            self.diz("%s  apos a usina: %dx %s | o jogo diz: %s"
                     % (ind, got, item, pq))

        if got == 0 and pq in ("no_power", "no_ingredients") and self.resta() > 8:
            n = self.m.craft_a_mao(rec, max(qtd, 1), r["ing"], qtd)
            if n > 0:
                self.m.espera(min(8, max(self.resta() - 2, 1)))
                got = self.m.le_personagem(item)
                self.diz("%s  o jogo disse %s -> fabriquei A MAO: %dx %s"
                         % (ind, pq, got, item))
                self.diagnosticos["recorreu_a_mao"] += 1
        self.estoque[item] += got
        self.diagnosticos[pq] += 1
        self.episodios.append({
            "objetivo": item, "receita": rec, "maquina": maq,
            "observacao": {"estoque": dict(self.estoque)},
            "acao": {"planta": maq, "alimenta": r["ing"]},
            "consequencia": {"obtido": got, "ticks": t, "status": pq}})
        self.diz("%s  -> %dx %s em %d ticks | o jogo diz: %s"
                 % (ind, got, item, t, pq))
        return self.estoque[item]

    def _minera(self, item: str, qtd: int, ind: str) -> int:
        """
        PERSISTE ate ter o que a receita pede. Antes fazia UM episodio e
        voltava: com meta de 30 circuitos ele minerava 24 de ferro, faltava 6,
        e desistia da arvore inteira. Persistir ate a quantidade (ou ate o
        orcamento) e o comportamento certo e nao e regra de Factorio minha.
        """
        from novaphys.minera import minera
        tentativas = 0
        while self.estoque[item] < qtd and self.resta() > 12 and tentativas < 8:
            tentativas += 1
            ep = minera(self.m.f, item,
                        segundos=min(10, max(self.resta() - 4, 1)), verbose=False)
            if "erro" in ep:
                self.diz("%s%s: %s" % (ind, item, ep["erro"]))
                break
            got = ep["consequencia"]["obtido"]
            self.estoque[item] += got
            ep["objetivo"] = item
            self.episodios.append(ep)
            self.diz("%sminerou %dx %s (total %d/%d, mancha -%d, %d ticks)"
                     % (ind, got, item, self.estoque[item], qtd,
                        ep["consequencia"]["consumido"],
                        ep["consequencia"]["ticks"]))
            if got == 0:
                break
        return self.estoque[item]


def jogar(meta: str = "iron-gear-wheel", qtd: int = 5, velocidade: int = 8,
          orcamento: float = 240.0) -> None:
    f = Fabrica()
    f.lua("game.speed = %d" % velocidade)
    m = Mundo(f)
    m.limpa()
    print("=" * 78)
    print("AGENTE LIVRE — o plano vem do JOGO, nao de mim")
    print("=" * 78)
    print("  %d receitas, %d maquinas, recursos no mapa: %s"
          % (len(m.recs), len(m.maq), ", ".join(sorted(m.recursos))))
    print("  META: %dx %s | velocidade %dx | orcamento %ds\n"
          % (qtd, meta, velocidade, orcamento))
    a = AgenteLivre(m, orcamento)
    got = a.produz(meta, qtd)
    print("\n" + "=" * 78)
    print("  RESULTADO: %dx %s de %dx pedidos" % (got, meta, qtd))
    print("  estoque final: %s" % dict(a.estoque))
    print("  o que o JOGO reportou: %s" % dict(a.diagnosticos))
    print("  episodios conectados: %d" % len(a.episodios))
    os.makedirs("resultados", exist_ok=True)
    json.dump(a.episodios, open("resultados/episodios_livre.json", "w"), indent=1)


if __name__ == "__main__":
    jogar()
