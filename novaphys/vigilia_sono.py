"""
novaphys.vigilia_sono — vigilia e sono: a experiencia compoe volta apos volta?
================================================================================

A PECA QUE FALTA DEPOIS DO SONHO
----------------------------------
[[sonho-ensina]] mediu UMA volta: as regras do nucleo viram sonho, o Sistema 1
pre-treinado nelas resolve tarefas que nenhuma dessas regras resolve (+7 sobre
placebo). Mas uma volta nao e acumulo. Acumulo e o ciclo fechar sobre si mesmo:
o que o Sistema 1 aprende a resolver vira, ele tambem, gerador de sonho — e a
proxima volta sonha com mais do que a anterior.

  VIGILIA  o Sistema 1 tenta as tarefas de TREINO que o nucleo nao resolve.
           No treino ele confere a resposta (o gabarito do livro): e
           experiencia com correcao, que e como humanos aprendem.
  SONO     TUDO que foi resolvido vira gerador: as regras explicitas do
           nucleo E as funcoes que o Sistema 1 aprendeu. Uma rede que resolveu
           uma tarefa tambem e uma regra — so que sem nome. Aplicada a grades de
           outras tarefas, sonha pares novos.
  EXAME    a cada volta, o Sistema 1 renascido dos sonhos ACUMULADOS enfrenta o
           conjunto de AVALIACAO do ARC — 400 tarefas que nenhuma parte do ciclo
           jamais viu, nem as grades.

  PLACEBO  o MESMO ciclo inteiro, com os sonhos desarranjados (mesmas entradas,
           mesmas saidas, relacao quebrada, formas unicas fora dos dois bracos).
           O placebo tambem acorda, tambem resolve o que conseguir, tambem sonha
           com isso desarranjado — segue o mesmo caminho, sem a relacao.

PRE-REGISTRO — escrito antes de rodar
---------------------------------------
  C0  PRE-CONDICAO. Volta 0 (Sistema 1 do zero) e o nucleo sao MEDIDOS no
      exame. Sem essa referencia nada depois tem escala.
  C1  A EXPERIENCIA COMPOE: no exame, volta 3 >= volta 1 + 3 no braco real E
      volta 3 real >= volta 3 placebo + 5
      -> o ciclo acumula: cada volta sonha com mais e resolve mais do que
         nunca viu. E a propriedade que faltava para "resolver problemas".
  C2  a volta 1 ganha do placebo (>= +3, o D1 replica no conjunto de
      AVALIACAO) mas as voltas seguintes nao somam (volta 3 < volta 1 + 3)
      -> o sonho vale, mas uma vez; nao ha composicao nesta escala.
  C3  nem a volta 1 ganha do placebo no exame -> o D1 nao passa do treino
      para a avaliacao.

PREVISAO: C2 a 0,40, C1 a 0,30, C3 a 0,30. Mecanismo de C2: as funcoes que o
Sistema 1 aprende no treino sao poucas (~20-40) e mais ruidosas que regras
exatas; somar isso aos 3.600 pares do nucleo acrescenta pouco. Mecanismo de C1:
cada funcao aprendida cobre um conceito que NENHUMA regra do nucleo cobre — e
exatamente o material que faltava ao sonho da volta 1.
"""

from __future__ import annotations

import copy
import json
import os
import pickle
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from novaphys import sistema1 as S
from novaphys.arc import carregar
from novaphys.eter import PASTA_AVAL
from novaphys.promessa import PASTA
from novaphys.sonho import placebo_de, pretreina, sonha

RODADAS = 3
N_POR_REDE = 60
CACHE_REGRAS = "resultados/sonho_pares_nucleo.pkl"


def resolve_com_rede(pesos, tr, te, semente: int = 0):
    """Procedimento do Sistema 1; devolve (ok, rede, gira) para virar gerador."""
    rnd = random.Random(semente)
    torch.manual_seed(semente)
    pares = [(np.array(a), np.array(b)) for a, b in tr]
    for _nome, gira, cor in S.GRUPOS:
        rede = S.Rede().to(S.DEV)
        if pesos is not None:
            rede.load_state_dict(copy.deepcopy(pesos))
        opt = torch.optim.Adam(rede.parameters(), lr=2e-3)
        rede.train()
        for _ in range(S.PASSOS):
            x, y, m = S._amostra(pares, rnd, gira, cor)
            L = (F.cross_entropy(rede(x), y, reduction="none") * m).sum() / m.sum()
            opt.zero_grad()
            L.backward()
            opt.step()
        rede.eval()
        if all(np.array_equal(S._preve(rede, a, gira), b) for a, b in pares):
            ok = all(np.array_equal(S._preve(rede, np.array(a), gira), np.array(b))
                     for a, b in te)
            return ok, rede, gira
    return False, None, False


def sonha_de_redes(redes, grades, rnd) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """Cada funcao aprendida que resolveu uma tarefa sonha em grades alheias."""
    pares = []
    for nome, rede, gira in redes:
        k = 0
        for g in rnd.sample(grades, min(len(grades), N_POR_REDE * 3)):
            if g.shape[0] > S.LADO or g.shape[1] > S.LADO:
                continue
            s = S._preve(rede, g, gira)
            if np.array_equal(s, g):
                continue
            pares.append((g, s, "s1:" + nome))
            k += 1
            if k >= N_POR_REDE:
                break
    return pares


def exame(pesos, tarefas) -> List[str]:
    return [n for n, tr, te in tarefas if resolve_com_rede(pesos, tr, te)[0]]


def ciclo(placebo: bool, semente: int = 0) -> Dict:
    rnd = random.Random(semente)
    etq = "PLACEBO" if placebo else "REAL"
    t0 = time.time()
    treino = {n: (tr, te) for n, tr, te in carregar(PASTA)}
    aval = [(n, tr, te) for n, tr, te in carregar(PASTA_AVAL) if S._mesma(tr, te)]
    nucleo = set(json.load(open("resultados/oficial.json")))
    vigilia = [(n, tr, te) for n, (tr, te) in treino.items()
               if S._mesma(tr, te) and n not in nucleo]
    grades = [np.array(a) for tr, te in treino.values() for a, _b in tr + te]

    if os.path.exists(CACHE_REGRAS):
        pares_nucleo = pickle.load(open(CACHE_REGRAS, "rb"))
    else:
        pares_nucleo = sonha(semente=0)["pares"]
        pickle.dump(pares_nucleo, open(CACHE_REGRAS, "wb"))

    print("[%s] exame: %d tarefas de AVALIACAO (mesma dimensao) | vigilia: %d de treino"
          % (etq, len(aval), len(vigilia)), flush=True)
    hist = {}
    # A volta 0 (Sistema 1 do zero, sem sonho) e IDENTICA nos dois bracos — o
    # placebo so difere a partir da volta 1. Calcula-la duas vezes era metade
    # de um exame jogado fora numa GPU ja a 96%. O placebo usa a do braco real.
    if not placebo:
        sol0 = exame(None, aval)
        hist[0] = sol0
        print("[%s] volta 0 (do zero): exame %d | %.0f min"
              % (etq, len(sol0), (time.time() - t0) / 60), flush=True)

    redes: List = []                       # funcoes aprendidas que viraram gerador
    for r in range(1, RODADAS + 1):
        brutos = pares_nucleo + sonha_de_redes(redes, grades, rnd)
        real, falso = placebo_de(brutos, random.Random(semente + r))
        dados = falso if placebo else real
        pesos = pretreina(dados, semente=semente)
        sol = exame(pesos, aval)
        hist[r] = sol
        # VIGILIA: o Sistema 1 desta volta tenta o treino que o nucleo nao resolve
        novas = []
        ja = {n for n, _r, _g in redes}
        for n, tr, te in vigilia:
            if n in ja:
                continue
            ok, rede, gira = resolve_com_rede(pesos, tr, te)
            if ok:
                redes.append((n, rede, gira))
                novas.append(n)
        print("[%s] volta %d: sonhos %d (redes geradoras %d) | EXAME %d | vigilia +%d"
              " | %.0f min" % (etq, r, len(dados), len(redes) - len(novas),
                              len(sol), len(novas), (time.time() - t0) / 60), flush=True)
    r_ = {"braco": etq, "exame": {k: v for k, v in hist.items()},
          "contagem": {k: len(v) for k, v in hist.items()},
          "geradores_s1": [n for n, _r, _g in redes], "n_aval": len(aval)}
    json.dump(r_, open("resultados/vigilia_sono_%s.json" % etq.lower(), "w"), indent=1)
    return r_


if __name__ == "__main__":
    ciclo(placebo="--placebo" in sys.argv)
