"""
novaphys.arc2_nucleo — o primeiro numero do projeto no ARC-AGI-2
==================================================================

POR QUE ESTE CONJUNTO
-----------------------
O usuario registrou a ambicao em 24/09 ([[ambicao-do-usuario]]): "resolver
problemas que IAs atuais nao conseguem, que so humanos ate entao se mostraram
capazes". O ARC-AGI-2 foi construido exatamente com essa definicao: humanos a
66% na amostra de teste; o melhor placar privado do ARC Prize 2025 foi 24%, com
computacao enorme ([[literatura-arc-2025-2026]]).

Isto mede so o NUCLEO (Sistema 2, biblioteca de base) nas 120 tarefas publicas
de avaliacao, com o mesmo prazo de 60 s por tarefa usado em [[destila]]. E a
linha de base do dia 0 no conjunto que mede a ambicao — nao a mente inteira.

Roda na CPU, sem torch, um processo, gravando tarefa a tarefa (a maquina tem ~2
GB livres reais e ja travou uma vez: [[maquina-do-usuario]]).
"""

from __future__ import annotations

import json
import os
import sys
import time

import novaphys.descricao as D
from novaphys.arc import carregar

PASTA2 = r"D:\agi_dados\arc_agi_2\data"
SAIDA = "resultados/arc_agi_2/nucleo_%s.jsonl"
PRAZO_S = 60
_PRAZO = [float("inf")]
_ORIG = D.custo


def _custo_com_prazo(pilha, treino):
    if time.time() > _PRAZO[0]:
        raise TimeoutError("prazo")
    return _ORIG(pilha, treino)


D.custo = _custo_com_prazo


def rodar(conjunto: str = "evaluation", teto: int = 4000) -> None:
    arq = SAIDA % conjunto
    feito = {}
    if os.path.exists(arq):
        for linha in open(arq, encoding="utf-8"):
            d = json.loads(linha)
            feito[d["nome"]] = d["ok"]
    T = list(carregar(os.path.join(PASTA2, conjunto)))
    print("ARC-AGI-2 %s: %d tarefas | ja feitas %d" % (conjunto, len(T), len(feito)),
          flush=True)
    t0 = time.time()
    for k, (nome, tr, te) in enumerate(T):
        if nome in feito:
            continue
        _PRAZO[0] = time.time() + PRAZO_S
        ok, err = False, None
        try:
            fm, pilha, err = D.montar(tr, teto=teto)
            if err == 0 and pilha:
                ok = all(D.aplicar((fm, pilha), x) == y for x, y in te)
        except Exception:
            pass
        feito[nome] = ok
        with open(arq, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"nome": nome, "ok": ok, "err": err}) + "\n")
        if (k + 1) % 10 == 0:
            print("  %3d/%d | resolvidas %d | %.0f min"
                  % (k + 1, len(T), sum(feito.values()), (time.time() - t0) / 60),
                  flush=True)
    n = sum(feito.values())
    print("NUCLEO no ARC-AGI-2 %s: %d/%d (%.1f%%)" % (conjunto, n, len(T), 100 * n / len(T)),
          flush=True)


if __name__ == "__main__":
    rodar(sys.argv[1] if len(sys.argv) > 1 else "evaluation")
