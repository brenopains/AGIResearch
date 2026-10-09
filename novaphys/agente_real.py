"""
novaphys.agente_real — a AGI jogando Factorio de verdade, no acelerado
========================================================================

O QUE MUDA EM RELACAO AO `agente_factorio`
--------------------------------------------
O agente antigo tinha isto sob a docstring "obter recurso do mundo":

    p.insert{name=item, count=qtd}

Ele nao jogava: resolvia a arvore de receitas e SPAWNAVA os primitivos. E na
primeira execucao deu 0/7 sem erro nenhum, porque num servidor sem jogador
conectado `game.players[1]` e nil e todo comando virava no-op silencioso.

Aqui tudo passa pelo mundo:

  MINERAR   broca a carvao sobre a mancha, bau na saida real (`drop_position`),
            producao no ritmo do jogo. Medido: obtido cresce com o tempo e a
            mancha encolhe na mesma medida (30 obtidos x 29 consumidos).
  FUNDIR    fornalha de pedra com minerio e carvao dentro. Sem eletricidade,
            mecanica real do jogo.

O QUE O AGENTE **NAO** CONSEGUE, E ISSO FICA DECLARADO
--------------------------------------------------------
Receitas acima de placa (engrenagem, circuito, montadora) exigem maquina de
montagem, que exige ELETRICIDADE: bomba + caldeira + motor a vapor + postes.
O agente reporta a parede em vez de fingir que passou — que era exatamente o
que o `insert` fazia.

CADA ACAO VIRA EPISODIO CONECTADO
-----------------------------------
    observacao   o que havia no mundo antes
    acao         o que foi feito, onde, com que insumo
    consequencia o que saiu, o que sumiu, quantos ticks

E isso e a categoria de dado que as oito colecoes do disco nao tinham: nenhuma
delas tem acao ligada a consequencia com relogio compartilhado.
"""

from __future__ import annotations

import json
import time
from typing import Dict, List, Optional, Tuple

from novaphys.fabrica import Fabrica
from novaphys.minera import acha_mancha, le_bau, minera, monta_broca, total_mancha

FORNALHA = "stone-furnace"
BAU = "wooden-chest"

FUNDE = {"iron-ore": "iron-plate", "copper-ore": "copper-plate",
         "stone": "stone-brick"}


class AgenteReal:
    def __init__(self, velocidade: int = 8):
        self.f = Fabrica()
        self.f.lua("game.speed = %d" % velocidade)
        self.episodios: List[dict] = []
        self.t0 = time.time()

    def diz(self, s: str) -> None:
        self.f.lua('game.print("[agi] %s")' % s.replace('"', "'"))
        print("  %s" % s, flush=True)

    # ------------------------------------------------------------ minerar ---
    def minerar(self, minerio: str, segundos: float = 10.0) -> int:
        ep = minera(self.f, minerio, segundos=segundos, verbose=False)
        if "erro" in ep:
            self.diz("minerar %s FALHOU: %s" % (minerio, ep["erro"]))
            return 0
        ep["objetivo"] = minerio
        self.episodios.append(ep)
        c = ep["consequencia"]
        self.diz("minerou %dx %s (mancha -%d, %d ticks)"
                 % (c["obtido"], minerio, c["consumido"], c["ticks"]))
        return c["obtido"]

    # ------------------------------------------------------------- fundir ---
    def fundir(self, minerio: str, qtd: int, segundos: float = 12.0) -> int:
        """Fornalha de pedra: mecanica real, sem eletricidade."""
        saida = FUNDE.get(minerio)
        if saida is None:
            self.diz("nao sei fundir %s" % minerio)
            return 0
        t0 = int(self.f.tick())
        # DEFEITO CORRIGIDO, achado rodando o Lua passo a passo em vez de
        # adivinhar pela terceira vez: **fornalha NAO tem `drop_position`** —
        # isso so existe em broca. A fornalha guarda o produto no proprio
        # inventario de saida. A fornalha ERA criada com sucesso em (8,8); o
        # que quebrava era o bau com posicao nil, e o erro subia como
        # "nao plantou", escondendo a causa.
        r = self.f.lua('''local s=game.surfaces[1]
for _,e in pairs(s.find_entities_filtered{{name="{F}"}}) do e.destroy() end
local pos = s.find_non_colliding_position("{F}", {{10,10}}, 40, 1)
if not pos then rcon.print("0") return end
local fn = s.create_entity{{name="{F}", position=pos, force="player"}}
if not fn then rcon.print("0") return end
fn.insert{{name="{m}", count={q}}}
fn.insert{{name="coal", count=25}}
rcon.print(pos.x .. "|" .. pos.y)'''
                       .format(F=FORNALHA, m=minerio, q=int(qtd)))
        if not r or "|" not in r:
            self.diz("fundir %s FALHOU: fornalha nao plantou" % minerio)
            return 0
        time.sleep(segundos)
        got = int(float(self.f.lua('''local s=game.surfaces[1]
local fs = s.find_entities_filtered{{name="{F}"}}
if #fs == 0 then rcon.print(0) return end
rcon.print(fs[1].get_output_inventory().get_item_count("{it}"))'''
                                   .format(F=FORNALHA, it=saida)) or 0))
        ep = {"objetivo": saida, "minerio": minerio,
              "observacao": {"insumo": int(qtd), "tick": t0},
              "acao": {"tipo": "planta_fornalha", "x": 10, "y": 10,
                       "combustivel": 25},
              "consequencia": {"obtido": got, "ticks": int(self.f.tick()) - t0}}
        self.episodios.append(ep)
        self.diz("fundiu %dx %s de %dx %s (%d ticks)"
                 % (got, saida, qtd, minerio, ep["consequencia"]["ticks"]))
        return got

    # ---------------------------------------------------------- perseguir ---
    def persegue(self, alvo: str, qtd: int = 10) -> dict:
        """Persegue a meta com o que o mundo permite. Declara a parede."""
        self.diz("META: %dx %s" % (qtd, alvo))
        if alvo in FUNDE.values():
            minerio = next(m for m, p in FUNDE.items() if p == alvo)
            bruto = 0
            for _ in range(3):
                bruto += self.minerar(minerio, segundos=10)
                if bruto >= qtd:
                    break
            if bruto == 0:
                return {"alvo": alvo, "obtido": 0, "parede": "sem minerio"}
            got = self.fundir(minerio, bruto, segundos=14)
            ok = got >= qtd
            self.diz("RESULTADO: %dx %s — %s" % (got, alvo,
                                                 "META BATIDA" if ok else "parcial"))
            return {"alvo": alvo, "obtido": got, "ok": ok, "bruto": bruto}
        if alvo in FUNDE:
            got = 0
            for _ in range(3):
                got += self.minerar(alvo, segundos=10)
                if got >= qtd:
                    break
            return {"alvo": alvo, "obtido": got, "ok": got >= qtd}
        return {"alvo": alvo, "obtido": 0,
                "parede": "exige montadora, que exige eletricidade — nao fingido"}


def jogar(metas=None, velocidade: int = 8) -> None:
    metas = metas or [("iron-ore", 20), ("iron-plate", 10),
                      ("copper-plate", 10), ("stone-brick", 8),
                      ("iron-gear-wheel", 5)]
    a = AgenteReal(velocidade)
    print("=" * 78)
    print("A AGI JOGANDO FACTORIO — velocidade %dx, mundo reagindo" % velocidade)
    print("=" * 78)
    print("  conecte em localhost para assistir\n")
    res = []
    for alvo, q in metas:
        r = a.persegue(alvo, q)
        res.append(r)
        print()
    print("=" * 78)
    print("  %-20s %8s  %s" % ("meta", "obtido", "veredito"))
    for r in res:
        v = ("META BATIDA" if r.get("ok") else
             ("PAREDE: " + r["parede"]) if "parede" in r else "parcial")
        print("  %-20s %8d  %s" % (r["alvo"], r["obtido"], v))
    print("\n  episodios conectados gravados: %d" % len(a.episodios))
    json.dump(a.episodios, open("resultados/episodios_factorio.json", "w"),
              indent=1)
    print("  -> resultados/episodios_factorio.json")


if __name__ == "__main__":
    jogar()
