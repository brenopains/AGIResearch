"""
novaphys.har_lexico — o caminho que ENXERGA ORDEM, aplicado ao fluxo real
==========================================================================

DE ONDE VEM
-------------
Do usuario: "temos algo como um encoder ou transicao de fato em outras camadas
da AGI?"

Checado por grep, e a resposta reorganiza o projeto: ha DOIS caminhos de
codificacao com propriedades OPOSTAS.

  SENSIVEL A ORDEM (viterbi obrigatorio):  lingua, canto, acervo, digestor,
                                           sob_demanda
  CEGO A ORDEM (soma sobre o tempo):       o caminho do fluxo / de_fluxo

E os resultados solidos do projeto estao TODOS no primeiro: fala F1 0,557 vs
0,377 trivial; codigo 0,560 vs 0,358; chunks de canto empatando com trigrama
usando 8x menos parametros. Os resultados rasos estao todos no segundo —
[[fluxo-nao-tinha-tempo]] mediu que `de_fluxo` e histograma e empata com o nulo
de tempo embaralhado.

A IDEIA, ENTAO, E NAO CONSTRUIR NADA NOVO
-------------------------------------------
Aplicar ao fluxo real o MESMO mecanismo que ja funciona em tres dominios:
quantizar os 9 canais em simbolos e rodar `lingua.segmenta`, que descobre o
lexico por MDL e segmenta por viterbi — ordem estrutural, nao histograma.

  cada instante (vetor de 9 canais) -> UM simbolo, por quantizacao vetorial
  cada janela -> uma STRING de 128 simbolos
  `segmenta` acha os MOTIVOS recorrentes; a janela vira o conjunto deles
  agrupa-se pelos motivos, nao pela distribuicao

E o quarto dominio da mesma maquina: grade, fala, codigo, canto — agora sensor.

PRE-REGISTRO
--------------
  L0  PRE-CONDICAO. O lexico tem de crescer acima do alfabeto. Com V simbolos,
      se o lexico ficar em ~V entradas, nao ha motivo e nada abaixo vale.
      (E o limiar de [[limiar-de-dado-e-lei]]: abaixo dele o lexico E o
      alfabeto.)
  L1  pureza >= nulo_tempo + 0,15  -> o caminho da ORDEM enxerga o que o
      histograma nao enxergava, e a diferenca e temporal de verdade
  L2  pureza > nulo mas < +0,15    -> ve ordem, mas pouco
  L3  pureza ~ nulo                -> nem o viterbi acha forma temporal aqui,
                                      e o problema nao e o encoder

  NULO PAREADO identico ao de [[fluxo-nao-tinha-tempo]]: embaralhar o EIXO DO
  TEMPO dentro de cada janela ANTES de quantizar. Preserva a distribuicao de
  simbolos e destroi a ordem. 5 embaralhamentos.

  Referencia: o histograma deu real 0,528 / nulo 0,537 (empate) e a transicao
  deu 0,506 / 0,442 (+0,064). Esta rodada tem de bater +0,064 para valer.

PREVISAO: L1, 0,65. Mecanismo: andar e periodico e periodicidade produz motivos
RECORRENTES de simbolos, que e exatamente o que o MDL acha barato; parado nao
produz motivo nenhum alem da repeticao de um simbolo so. Risco nomeado: 128
instantes por janela pode estar ABAIXO do limiar de dado — na fala o lexico so
existiu acima de ~1000 unidades, e aqui cada janela tem 128. Se for isso, o
conserto e concatenar janelas da mesma atividade antes de segmentar.
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np

from novaphys.har import NOMES, agrupa, carrega, pureza
from novaphys.hdc_indice import Espaco, feixe
from novaphys.lingua import _viterbi, segmenta

V_SIMB = 20                     # tamanho do alfabeto da quantizacao
SEMENTE = 20260918
ALFA = "abcdefghijklmnopqrstuvwxyz"


def quantiza_vetorial(X: np.ndarray, v: int, rng) -> List[str]:
    """
    Cada instante (vetor de 9 canais) vira UM simbolo.

    Centroides por k-medias simples sobre uma amostra dos instantes de TODAS as
    janelas — o vocabulario e compartilhado, senao cada janela teria alfabeto
    proprio e nada seria comparavel.
    """
    n, T, K = X.shape
    todos = X.reshape(-1, K)
    mu, sd = todos.mean(0), todos.std(0)
    Z = (todos - mu) / np.where(sd > 1e-9, sd, 1.0)
    amostra = Z[rng.choice(len(Z), size=min(20000, len(Z)), replace=False)]
    cen = amostra[rng.choice(len(amostra), size=v, replace=False)]
    for _ in range(12):
        d = ((amostra[:, None, :] - cen[None, :, :]) ** 2).sum(-1)
        at = np.argmin(d, axis=1)
        novos = np.stack([amostra[at == c].mean(0) if (at == c).any()
                          else cen[c] for c in range(v)])
        if np.allclose(novos, cen):
            break
        cen = novos
    d = ((Z[:, None, :] - cen[None, :, :]) ** 2).sum(-1)
    simb = np.argmin(d, axis=1).reshape(n, T)
    return ["".join(ALFA[s] for s in linha) for linha in simb]


def vetor_de_motivos(e: Espaco, s: str, lex: set) -> np.ndarray:
    """A janela vira o feixe dos MOTIVOS que a segmentacao encontrou nela."""
    prob = {w: 1.0 / len(lex) for w in lex}
    ped = _viterbi(s, lex, prob)
    termos = [e.atomo(f"mot:{w}") for w in ped if len(w) > 1]
    return feixe(termos, e.dim) if termos else e.atomo("vazio")


def roda(X: np.ndarray, y: np.ndarray, e: Espaco, rng, k: int,
         rotulo: str) -> Tuple[float, int]:
    cadeias = quantiza_vetorial(X, V_SIMB, rng)
    corpus = "".join(cadeias[:120])          # so uma parte, por custo
    _toks, lex, _a, _b = segmenta(corpus, rodadas=400, cand=24)
    lex = set(lex) | set(ALFA[:V_SIMB])
    n_mot = len([w for w in lex if len(w) > 1])
    V = np.stack([vetor_de_motivos(e, s, lex) for s in cadeias])
    p = pureza(agrupa(V, k, np.random.default_rng(SEMENTE)), y)
    print(f"    {rotulo:26s} lexico {n_mot:4d} motivos   pureza {p:.3f}",
          flush=True)
    return p, n_mot


def experimento(n_max: int = 900, k: int = 12) -> None:
    e = Espaco()
    rng = np.random.default_rng(SEMENTE)
    X, y = carrega("train", n_max)

    print("=" * 78)
    print("HAR LEXICO — o caminho que ENXERGA ORDEM, no fluxo real")
    print("=" * 78)
    cc = Counter(y.tolist())
    acaso = max(cc.values()) / len(y)
    print(f"  {len(X)} janelas; quantizacao vetorial em {V_SIMB} simbolos")
    print(f"  acaso {acaso:.3f}; referencia do histograma: real 0,528 / "
          f"nulo 0,537 (empate)")
    print(f"  referencia da transicao: 0,506 / 0,442 (+0,064 a bater)\n")

    p, n_mot = roda(X, y, e, rng, k, "REAL")
    if n_mot <= 2:
        print(f"\nL0 FALHOU: lexico com {n_mot} motivos — nao passou do "
              "alfabeto. Abaixo do limiar de dado.")
        return

    ps_n = []
    for s in range(5):
        r2 = np.random.default_rng(1000 + s)
        Xe = np.stack([A[r2.permutation(A.shape[0])] for A in X])
        ps_n.append(roda(Xe, y, e, np.random.default_rng(SEMENTE), k,
                         f"NULO tempo embaralhado {s+1}")[0])
    pn, sn = statistics.mean(ps_n), statistics.pstdev(ps_n)

    print(f"\n  REAL {p:.3f}   NULO {pn:.3f} +- {sn:.3f}   "
          f"ganho {p-pn:+.3f}")
    print("\n" + "=" * 78)
    print(f"L0 . lexico com {n_mot} motivos, acima do alfabeto.")
    if p >= pn + 0.15:
        print(f"L1 . O CAMINHO DA ORDEM ENXERGA O QUE O HISTOGRAMA NAO VIA:")
        print(f"     ganho {p-pn:+.3f} sobre o nulo de tempo, contra +0,064 do")
        print("     encoder de transicao e 0,000 do histograma.")
    elif p > pn + 0.02:
        print(f"L2 . ve ordem, mas pouco ({p-pn:+.3f}).")
        if p - pn > 0.064:
            print("     Ainda assim BATE o encoder de transicao (+0,064).")
    else:
        print(f"L3 . nem o viterbi acha forma temporal aqui ({p-pn:+.3f}).")
        print("     O problema nao e o encoder.")


if __name__ == "__main__":
    experimento()
