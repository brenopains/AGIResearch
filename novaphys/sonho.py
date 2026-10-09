"""
novaphys.sonho — o Sistema 2 ensina o Sistema 1 sonhando
===========================================================

DE ONDE VEM
-------------
[[experiencia-acumula]] deu E3: pre-treinar o Sistema 1 em 130 tarefas ajuda,
mas pares EMBARALHADOS ajudam mais. O ganho era estatistica de grade, nao
experiencia de COMO transformacoes funcionam. O risco que eu tinha nomeado era
volume: a literatura pre-treina em dezenas de milhares de tarefas (re-ARC).

O nucleo simbolico tem uma coisa que nenhuma rede tem: REGRAS EXATAS. As 105
tarefas que ele resolve viram 105 geradores. Aplicando cada regra a grades de
OUTRAS tarefas, sai um numero ilimitado de pares (entrada, saida) cuja relacao e
verdadeira por construcao. E o sono do DreamCoder (fantasias da biblioteca
treinando o reconhecimento) e o replay do hipocampo (Schwartenbeck, Cell 2023:
replay monta hipoteses compostas; teoria dos sistemas complementares: o
hipocampo reencena e o cortex aprende). Aqui: o Sistema 2 ensina o Sistema 1.

SEM VAZAMENTO, POR CONSTRUCAO
-------------------------------
Os sonhos vem SO das regras das tarefas que o nucleo ja resolve. A avaliacao e
SO nas tarefas de mesma dimensao que o nucleo NAO resolve — cuja regra ninguem
conhece. Nenhuma regra avaliada entra no sonho.

O CONTROLE, FEITO CERTO DESDE O INICIO
----------------------------------------
Hoje quebrei dois placebos pelo mesmo motivo (pares verdadeiros vazando por
embaralhamento que devolve a identidade ou por grupo de tamanho 1). Aqui:
  SONHO          pares (g, regra(g))
  SONHO-PLACEBO  AS MESMAS entradas e AS MESMAS saidas, mas cada saida vem de
                 OUTRA entrada de mesma forma, e diferente da verdadeira
                 (desarranjo). Pares de forma unica saem dos DOIS bracos.
Mesma estatistica de pixel, mesmo volume, zero relacao. Isola exatamente o que
o E3 nao conseguiu isolar.

PRE-REGISTRO — escrito antes de rodar
---------------------------------------
  D0  PRE-CONDICAO. O sonho gera >= 2000 pares relacionais de >= 30 regras
      distintas. Menos que isso e o mesmo volume do E3 e nao testa nada novo.
  D1  O SISTEMA 2 ENSINA O SISTEMA 1: nas tarefas que o nucleo NAO resolve,
      SONHO resolve >= SONHO-PLACEBO + 4
      -> experiencia relacional compoe quando ha volume; e o canal pelo qual a
         mente transforma regra explicita em intuicao.
  D2  SONHO ~ SONHO-PLACEBO (dentro de +-3) -> nem com milhares de pares a
      relacao das regras conhecidas transfere para tarefa nova; o que o Sistema
      1 leva de uma tarefa para outra e estatistica de grade, ponto.
  D3  SONHO < SONHO-PLACEBO - 3 -> sonhar com as regras conhecidas ENVIESA o
      Sistema 1 para longe das tarefas que faltam (elas sao justamente as que
      as regras conhecidas nao cobrem).

PREVISAO: D2 a 0,40, D1 a 0,35, D3 a 0,25. Mecanismo de D2/D3, e por isso eu nao
aposto em D1: as tarefas que o nucleo NAO resolve sao, por definicao, as que as
105 regras conhecidas nao cobrem. Sonhar com o que ja se sabe pode ensinar
primitivas perceptuais uteis (D1) ou so reforcar o que ja se sabe (D2/D3).
"""

from __future__ import annotations

import json
import random
import sys
import time
from multiprocessing import Pool
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from novaphys import sistema1 as S
from novaphys.arc import carregar, dims
from novaphys.descricao import aplicar, montar
from novaphys.promessa import PASTA

N_GRADES_POR_REGRA = 60
PRE_PASSOS = 12000


def _regra(args):
    nome, tr = args
    try:
        fm, pilha, err = montar(tr, teto=4000)
    except Exception:
        return nome, None
    return nome, ((fm, pilha) if err == 0 and pilha else None)


def sonha(semente: int = 0) -> Dict:
    rnd = random.Random(semente)
    T = {n: (tr, te) for n, tr, te in carregar(PASTA)}
    nucleo = set(json.load(open("resultados/oficial.json")))
    with Pool(8) as pool:
        regras = dict(pool.map(_regra, [(n, T[n][0]) for n in sorted(nucleo)]))
    regras = {n: r for n, r in regras.items() if r is not None}
    grades = [np.array(a) for tr, te in T.values() for a, _b in tr + te]
    pares, por_regra = [], {}
    for nome, rg in regras.items():
        amostra = rnd.sample(grades, min(len(grades), N_GRADES_POR_REGRA * 3))
        k = 0
        for g in amostra:
            try:
                s = aplicar(rg, tuple(tuple(int(v) for v in l) for l in g))
            except Exception:
                continue
            if s is None:
                continue
            s = np.array(s)
            if s.shape != g.shape or np.array_equal(s, g):
                continue                      # so mesma dimensao, e nao-trivial
            pares.append((g, s, nome))
            k += 1
            if k >= N_GRADES_POR_REGRA:
                break
        if k:
            por_regra[nome] = k
    return {"pares": pares, "regras": por_regra, "n_regras_ok": len(regras)}


def placebo_de(pares, rnd):
    """Desarranjo por forma; pares de forma unica saem dos DOIS bracos."""
    por: Dict[tuple, List[int]] = {}
    for i, (g, _s, _n) in enumerate(pares):
        por.setdefault(g.shape, []).append(i)
    real, falso = [], []
    for idx in por.values():
        if len(idx) < 2:
            continue
        for _ in range(200):
            perm = idx[:]
            rnd.shuffle(perm)
            if all(p != i and not np.array_equal(pares[p][1], pares[i][1])
                   for i, p in zip(idx, perm)):
                break
        else:
            continue                          # sem desarranjo: grupo sai dos dois
        for i, p in zip(idx, perm):
            real.append((pares[i][0], pares[i][1]))
            falso.append((pares[i][0], pares[p][1]))
    return real, falso


def pretreina(pares, semente: int = 0):
    rnd = random.Random(semente)
    torch.manual_seed(semente)
    rede = S.Rede().to(S.DEV)
    opt = torch.optim.Adam(rede.parameters(), lr=1e-3)
    rede.train()
    for _ in range(PRE_PASSOS):
        x, y, m = S._amostra(pares, rnd, True, True)
        L = (F.cross_entropy(rede(x), y, reduction="none") * m).sum() / m.sum()
        opt.zero_grad()
        L.backward()
        opt.step()
    return rede.state_dict()


def experimento() -> Dict:
    from novaphys.experiencia import resolve_de
    t0 = time.time()
    print("=" * 78)
    print("SONHO — o Sistema 2 ensina o Sistema 1?")
    print("=" * 78, flush=True)
    s = sonha()
    real, falso = placebo_de(s["pares"], random.Random(1))
    print("  regras do nucleo reproduzidas: %d; geraram pares: %d"
          % (s["n_regras_ok"], len(s["regras"])))
    print("  pares de sonho: %d brutos -> %d em cada braco (forma unica removida)"
          % (len(s["pares"]), len(real)))
    print("  %.0f min\n" % ((time.time() - t0) / 60), flush=True)
    if len(real) < 2000 or len(s["regras"]) < 30:
        print("D0 FALHOU: volume insuficiente (%d pares, %d regras)."
              % (len(real), len(s["regras"])))
        return {"veredito": "D0 falhou"}

    nucleo = set(json.load(open("resultados/oficial.json")))
    alvo = [(n, tr, te) for n, tr, te in carregar(PASTA)
            if S._mesma(tr, te) and n not in nucleo]
    zero = set(json.load(open("resultados/sistema1.json"))["resolve"])
    grades = set(json.load(open("resultados/experiencia.json"))["placebo"])
    print("  alvo: %d tarefas de mesma dimensao que o nucleo NAO resolve" % len(alvo))
    print("  referencia medida antes: DO ZERO %d | PRE-TREINO EM GRADES %d"
          % (len(zero & {n for n, _, _ in alvo}), len(grades & {n for n, _, _ in alvo})),
          flush=True)
    res = {}
    for braco, dados in (("SONHO", real), ("SONHO-PLACEBO", falso)):
        pesos = pretreina(dados)
        ok = [n for n, tr, te in alvo if resolve_de(pesos, tr, te)[0]]
        res[braco] = ok
        print("    %-14s resolve %d | %.0f min"
              % (braco, len(ok), (time.time() - t0) / 60), flush=True)
    d, p = len(res["SONHO"]), len(res["SONHO-PLACEBO"])
    print("\n" + "=" * 78)
    print("  SONHO %d | SONHO-PLACEBO %d  (de %d que o nucleo nao resolve)"
          % (d, p, len(alvo)))
    print("  uniao nucleo + sonho: %d/400" % len(nucleo | set(res["SONHO"])))
    if d >= p + 4:
        ver = "D1"
        print("D1 . O SISTEMA 2 ENSINA O SISTEMA 1: %+d sobre o mesmo volume sem" % (d - p))
        print("     relacao. Regra explicita virou intuicao.")
    elif d < p - 3:
        ver = "D3"
        print("D3 . sonhar com o que ja se sabe ENVIESA para longe do que falta (%+d)."
              % (d - p))
    else:
        ver = "D2"
        print("D2 . a relacao das regras conhecidas nao transfere (%+d)." % (d - p))
    r = {"veredito": ver, "sonho": res["SONHO"], "placebo": res["SONHO-PLACEBO"],
         "pares": len(real), "regras": len(s["regras"])}
    json.dump(r, open("resultados/sonho.json", "w"), indent=1)
    return r


if __name__ == "__main__":
    experimento()
