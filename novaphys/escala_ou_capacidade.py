"""
novaphys.escala_ou_capacidade — o que falta e DADO ou e CAPACIDADE?
=====================================================================

A PERGUNTA DO USUARIO, NA FORMA EM QUE ELA SE MEDE
----------------------------------------------------
"O sistema demonstrou algumas capacidades, precisamos identificar se falta mais
alguma ou se o problema e escala de datasets."

Argumentar isso e facil nos dois sentidos. Medir tambem e: pega-se um problema
onde perdemos, fixa-se o teste, e cresce-se o TREINO. Se a DISTANCIA entre nos
e um rival trivial encolhe com dado, o que falta e escala. Se fica constante ou
cresce, mais dado nao fecha — falta capacidade.

O PROBLEMA ESCOLHIDO, E POR QUE
---------------------------------
HAR, porque e o unico lugar do projeto com ancora externa no mesmo dado e no
mesmo split ([[ancora-externa-har]]): publicado 0,967, SETE LINHAS de
estatistica por canal 0,752, nosso 0,560. Perdemos para sete linhas. Se isso
for falta de dado, a curva mostra; se for representacao, tambem mostra.

O CONFUNDIDOR QUE JA ME ENGANOU UMA VEZ
----------------------------------------
`carrega("train", N)` usa `max_rows` e pega as N PRIMEIRAS janelas — e isso muda
o NUMERO DE PESSOAS junto com o tamanho ([[limite-de-linhas-mentiu]]: 2400
janelas eram 8 voluntarios, nao 21). Aqui o treino inteiro (7352, 21 pessoas) e
carregado uma vez e SUBAMOSTRADO ao acaso, entao todo tamanho cobre as mesmas
pessoas e a unica coisa que varia e a quantidade. O teste e o oficial inteiro
(2947 janelas, 9 pessoas nunca vistas), fixo em todos os pontos.

PRE-REGISTRO — escrito antes de rodar
---------------------------------------
  D0  PRE-CONDICAO. No tamanho cheio o rival de 7 linhas fica perto do que a
      ancora mediu (>= 0,70). Se nao, o instrumento nao e o mesmo.
  D1  ESCALA: a distancia (rival - nosso) ENCOLHE com o treino, e de 1500 a
      7352 cai pelo menos um terco -> o que falta e dado; coletar mais paga.
  D2  CAPACIDADE: nossa curva e plana (ganho < 0,05 do menor ao maior treino)
      enquanto a distancia fica >= 0,10 -> mais dado nao fecha; o buraco e de
      REPRESENTACAO, e dataset nenhum resolve sozinho.
  D3  os dois sobem juntos, distancia estavel -> ha ganho de dado para todos,
      mas o buraco e de capacidade e anda em paralelo.

PREVISAO: D2 a 0,55. Mecanismo: nosso pipeline quantiza e faz histograma de
simbolos antes de qualquer coisa ([[fluxo-nao-tinha-tempo]] ja mostrou que ele
era invariante a permutacao do tempo); informacao jogada fora na entrada nao
volta com mais exemplos. D3 a 0,30, D1 a 0,15.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from typing import Dict, List

import numpy as np

from novaphys.har import carrega
from novaphys.har_pessoas import K_GRUPOS, SEMENTE, PipelineCongelado
from novaphys.ancora_externa import estatisticas
import novaphys.har_pessoas as HP

TAMANHOS = (300, 750, 1500, 3000, 7352)
N_SORTEIOS = 3


def _nosso(Xtr, ytr, Xte, yte, semente: int) -> float:
    e = HP.Espaco()
    p = PipelineCongelado(e)
    p.ajusta(Xtr, ytr, np.random.default_rng(semente), K_GRUPOS)
    return float(np.mean(p.aplica(Xte) == yte))


def _rival(Xtr, ytr, Xte, yte) -> float:
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=1)
    rf.fit(estatisticas(Xtr), ytr)
    return float(np.mean(rf.predict(estatisticas(Xte)) == yte))


def experimento() -> Dict:
    Xall, yall = carrega("train", None)
    Xte, yte = carrega("test", None)
    print("=" * 78)
    print("ESCALA OU CAPACIDADE — mais dado fecha a distancia?")
    print("=" * 78)
    print("  treino completo %d janelas; teste FIXO %d janelas (pessoas novas)"
          % (len(Xall), len(Xte)))
    print("  subamostra ALEATORIA do treino inteiro: todo tamanho cobre as")
    print("  mesmas 21 pessoas, so a quantidade varia\n")
    print("  %7s %16s %10s %12s" % ("treino", "NOSSO", "RIVAL 7L", "distancia"))
    print("  " + "-" * 50, flush=True)

    linhas = []
    for n in TAMANHOS:
        nos, riv = [], []
        for s in range(N_SORTEIOS if n < len(Xall) else 1):
            rng = np.random.default_rng(SEMENTE + 97 * s + n)
            idx = (np.arange(len(Xall)) if n >= len(Xall)
                   else rng.choice(len(Xall), size=n, replace=False))
            Xtr, ytr = Xall[idx], yall[idx]
            nos.append(_nosso(Xtr, ytr, Xte, yte, SEMENTE + s))
            riv.append(_rival(Xtr, ytr, Xte, yte))
        mn, mr = statistics.mean(nos), statistics.mean(riv)
        linhas.append((n, mn, mr, mr - mn))
        print("  %7d %9.3f +-%.3f %10.3f %+12.3f"
              % (n, mn, statistics.pstdev(nos), mr, mr - mn), flush=True)

    print("\n" + "=" * 78)
    n0, a0, r0, d0 = linhas[0]
    nf, af, rf_, df = linhas[-1]
    d15 = next(d for n, _a, _r, d in linhas if n == 1500)
    print("  nosso: %.3f -> %.3f (%+.3f)   rival: %.3f -> %.3f (%+.3f)"
          % (a0, af, af - a0, r0, rf_, rf_ - r0))
    print("  distancia: %.3f -> %.3f" % (d0, df))
    if rf_ < 0.70:
        ver = "D0 falhou"
        print("D0 FALHOU: o rival cheio deu %.3f; instrumento diferente." % rf_)
    elif df <= d15 * (2 / 3) and df < d0:
        ver = "D1"
        print("D1 . ESCALA: a distancia encolhe (%.3f em 1500 -> %.3f cheio)."
              % (d15, df))
        print("     O que falta e dado; coletar mais paga.")
    elif (af - a0) < 0.05 and df >= 0.10:
        ver = "D2"
        print("D2 . CAPACIDADE: nossa curva e plana (%+.3f em %dx mais dado)"
              % (af - a0, nf // n0))
        print("     e a distancia fica em %.3f. Mais dado NAO fecha: o buraco" % df)
        print("     e de representacao.")
    else:
        ver = "D3"
        print("D3 . os dois sobem e a distancia fica (%.3f -> %.3f): ha ganho"
              % (d0, df))
        print("     de dado para todos, mas o buraco e de capacidade.")
    r = {"veredito": ver, "linhas": linhas}
    json.dump(r, open("resultados/escala_ou_capacidade.json", "w"), indent=1)
    return r


if __name__ == "__main__":
    experimento()
