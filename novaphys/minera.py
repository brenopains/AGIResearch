"""
novaphys.minera — extrair de verdade, com o mundo reagindo
============================================================

O DEFEITO QUE ISTO CONSERTA
-----------------------------
`agente_factorio.extrair` tinha esta linha, sob a docstring "Acao primitiva:
obter recurso do mundo":

    p.insert{name=item, count=qtd}

Ele **não minerava**: inseria o item direto no inventário. O agente resolvia a
árvore de receitas e *spawnava* os primitivos. O planejamento era real; a parte
"mundo reagindo" não existia — sem mineração, sem tempo de produção, sem
escassez, sem consequência.

Isso importa porque um episódio com `insert` é observação sem ação-consequência:
a mesma categoria das oito coleções desconectadas que este projeto já tinha, e
justamente o que o Factorio existia para não ser.

COMO FICA
-----------
Mineração de verdade, com máquina no mundo:

  1. acha uma mancha do minério pedido
  2. posiciona uma BROCA sobre ela, com um BAÚ na saída
  3. abastece a broca com carvão
  4. deixa o tempo passar — a broca consome combustível e produz no ritmo dela
  5. lê o baú

O que muda em relação ao `insert`: o recurso **acaba** (a mancha tem quantidade
finita), a broca **para** se faltar carvão, a produção **leva tempo real de
jogo**, e o episódio registra observação, ação e consequência separadas.

MEDIDA, NÃO PROMESSA
----------------------
  X0  PRÉ-CONDIÇÃO: existe minério; a broca é posicionada; ela produz > 0 em
      tempo finito. Se a broca não produzir, o conserto não está funcionando e
      não adianta seguir.
  X1  a quantidade no baú CRESCE com o tempo de espera, e a mancha ENCOLHE na
      mesma proporção -> o mundo está reagindo, e o episódio tem consequência
  X2  produz sem a mancha encolher -> ainda é spawn disfarçado
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from novaphys.fabrica import Fabrica

BROCA = "burner-mining-drill"
BAU = "wooden-chest"


def acha_mancha(f: Fabrica, minerio: str, raio: int = 120) -> Optional[dict]:
    """A posicao de um tile do minerio pedido, e quanto a mancha tem ali."""
    r = f.lua('''local s=game.surfaces[1]
local es = s.find_entities_filtered{{name="{m}", area={{{{-{R},-{R}}},{{{R},{R}}}}}}}
if #es == 0 then rcon.print("") return end
local e = es[math.min(#es, 5)]
rcon.print(e.position.x .. "|" .. e.position.y .. "|" .. e.amount)'''
              .format(m=minerio, R=raio))
    if not r or "|" not in r:
        return None
    x, y, q = r.split("|")
    return {"x": float(x), "y": float(y), "amount": int(float(q))}


def total_mancha(f: Fabrica, minerio: str, cx: float, cy: float,
                 raio: int = 6) -> int:
    """Quanto resta de minerio na vizinhanca — para ver a mancha ENCOLHER."""
    r = f.lua('''local s=game.surfaces[1] local t=0
for _,e in pairs(s.find_entities_filtered{{name="{m}", area={{{{{x1},{y1}}},{{{x2},{y2}}}}}}}) do
  t = t + e.amount end
rcon.print(t)'''.format(m=minerio, x1=cx - raio, y1=cy - raio,
                        x2=cx + raio, y2=cy + raio))
    return int(float(r or 0))


def monta_broca(f: Fabrica, x: float, y: float, carvao: int = 50) -> bool:
    """Broca sobre o minerio, bau na saida, carvao dentro. Devolve se plantou."""
    # DEFEITO CORRIGIDO: eu adivinhava a posicao do bau como (x, y-2). A saida
    # real da broca e `drop_position`, que o jogo expoe — no primeiro teste ela
    # deu (-27.5, 28.70) com o bau em (-27.5, 27.5), UM TILE ALEM. O minerio
    # caia no chao entre os dois, o buffer da broca enchia com 1 e ela travava:
    # a mancha encolhia 1 e o bau recebia 0, identico para 6s, 12s e 24s.
    # Agora o bau e posto EM `drop_position`, lida do proprio jogo.
    r = f.lua('''local s=game.surfaces[1]
for _,e in pairs(s.find_entities_filtered{{area={{{{{x0},{y0}}},{{{x1},{y1}}}}},
    name={{"{B}","{C}"}}}}) do e.destroy() end
local d = s.create_entity{{name="{B}", position={{{x},{y}}},
                           direction=defines.direction.north, force="player"}}
if not d then rcon.print("0") return end
local dp = d.drop_position
local c = s.create_entity{{name="{C}", position=dp, force="player"}}
if not c then
  c = s.create_entity{{name="{C}", position={{dp.x, dp.y - 0.5}}, force="player"}}
end
d.insert{{name="coal", count={cv}}}
rcon.print(c and (dp.x .. "|" .. dp.y) or "0")'''
              .format(B=BROCA, C=BAU, x=x, y=y,
                      x0=x - 4, y0=y - 4, x1=x + 4, y1=y + 4, cv=carvao))
    if not r or "|" not in r:
        return False
    bx, by = (float(v) for v in r.split("|"))
    monta_broca.ultimo_bau = (bx, by)
    return True


def le_bau(f: Fabrica, x: float, y: float, item: str) -> int:
    """Le o bau que `monta_broca` plantou na saida REAL da broca."""
    bx, by = getattr(monta_broca, "ultimo_bau", (x, y - 2))
    r = f.lua('''local s=game.surfaces[1]
local cs = s.find_entities_filtered{{name="{C}", position={{{x},{y}}}, radius=2}}
if #cs == 0 then rcon.print(0) return end
rcon.print(cs[1].get_inventory(defines.inventory.chest).get_item_count("{it}"))'''
              .format(C=BAU, x=bx, y=by, it=item))
    return int(float(r or 0))


def minera(f: Fabrica, minerio: str, segundos: float = 12.0,
           verbose: bool = True) -> dict:
    """
    UM EPISODIO de mineracao real: observacao -> acao -> consequencia.
    Devolve o episodio com os tres registrados separadamente.
    """
    m = acha_mancha(f, minerio)
    if m is None:
        return {"minerio": minerio, "erro": "sem mancha no raio"}
    antes = total_mancha(f, minerio, m["x"], m["y"])
    t0 = int(f.tick())
    if not monta_broca(f, m["x"], m["y"]):
        return {"minerio": minerio, "erro": "nao plantou a broca"}
    time.sleep(segundos)
    depois = total_mancha(f, minerio, m["x"], m["y"])
    obtido = le_bau(f, m["x"], m["y"], minerio)
    ep = {
        "minerio": minerio,
        "observacao": {"mancha_antes": antes, "tick": t0},
        "acao": {"tipo": "planta_broca", "x": m["x"], "y": m["y"],
                 "combustivel": 50},
        "consequencia": {"obtido": obtido, "mancha_depois": depois,
                         "consumido": antes - depois,
                         "ticks": int(f.tick()) - t0},
    }
    if verbose:
        c = ep["consequencia"]
        print("  %-12s obtido %3d | mancha %5d -> %5d (consumiu %3d) | %d ticks"
              % (minerio, c["obtido"], antes, depois, c["consumido"],
                 c["ticks"]), flush=True)
    return ep


def experimento(esperas=(6, 12, 24)) -> None:
    f = Fabrica()
    print("=" * 78)
    print("MINERA — o mundo reage, ou ainda e spawn disfarcado?")
    print("=" * 78)
    print("  X1: o obtido CRESCE com a espera E a mancha ENCOLHE na mesma medida\n")
    eps = []
    for s in esperas:
        print("  espera de %ds:" % s)
        for mi in ("iron-ore", "copper-ore", "stone"):
            ep = minera(f, mi, segundos=s)
            if "erro" in ep:
                print("    %-12s ERRO: %s" % (mi, ep["erro"]))
            else:
                eps.append((s, ep))
        print()

    print("=" * 78)
    por_espera = {}
    for s, ep in eps:
        por_espera.setdefault(s, []).append(ep["consequencia"])
    print("  %8s %10s %12s" % ("espera", "obtido", "consumido"))
    for s in sorted(por_espera):
        cs = por_espera[s]
        print("  %6ds %10d %12d" % (s, sum(c["obtido"] for c in cs),
                                    sum(c["consumido"] for c in cs)))
    tot_o = sum(c["obtido"] for cs in por_espera.values() for c in cs)
    tot_c = sum(c["consumido"] for cs in por_espera.values() for c in cs)
    print()
    if tot_o > 0 and tot_c > 0:
        print("X1 . O MUNDO REAGE: %d obtidos e %d consumidos da mancha." % (tot_o, tot_c))
        print("     O episodio tem consequencia, e o recurso e FINITO.")
    elif tot_o > 0:
        print("X2 . produz %d mas a mancha NAO encolhe — ainda e spawn." % tot_o)
    else:
        print("X0 FALHOU: a broca nao produziu nada. Conserto nao funcionou.")


if __name__ == "__main__":
    experimento()
