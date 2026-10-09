"""
novaphys.har — dado REAL: o sistema descobre 'andar' sem que ninguem diga?
============================================================================

DE ONDE VEM
-------------
Do usuario, apontando os tres buracos que sobraram: "Significado com
profundidade, Escolher onde atacar, O dado ser seu."

Os tres convergem num so. Significado profundo E extensao — "caixa = modulo"
significa algo para o engenheiro porque ele viu centenas de sistemas modulares.
Imprimir o objeto no relatorio (a engrenagem encaixando na bacteria) e
recuperacao por similaridade, que exige acervo de objetos conhecidos. E "o dado
ser meu" e o item em si. Tudo pede a mesma coisa: DADO REAL, em volume.

O DADO
--------
UCI Human Activity Recognition (archive.ics.uci.edu, 61 MB). 30 voluntarios com
smartphone na cintura; 9 canais inerciais (aceleracao do corpo x/y/z, giroscopio
x/y/z, aceleracao total x/y/z) amostrados em janelas de 128 instantes.
10.299 janelas, 6 atividades NOMEADAS: WALKING, WALKING_UPSTAIRS,
WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING.

Por que este e nao outro: (a) e T x K nativo, sem encoder novo; (b) tem MILHARES
de instancias por conceito, que e a profundidade que faltava; (c) os conceitos
tem NOME NO MUNDO, entao "o sistema descobriu um objeto" deixa de ser afirmacao
sobre agrupamento e vira afirmacao CHECAVEL: descobriu, e o que descobriu e
`andar`. Nada nele foi desenhado por mim.

A PERGUNTA
------------
Sem nenhum rotulo, agrupando so pela estrutura do fluxo: os grupos descobertos
correspondem as atividades reais?

PRE-REGISTRO
--------------
  H0  PRE-CONDICAO E CONTA DE FOLGA (a licao de [[escolher-alvo-sem-folga]]:
      calcular o efeito possivel ANTES). Com 6 classes quase balanceadas o
      acaso de pureza e ~1/6 = 0,167 e o teto e 1,0 — folga de 0,83, muito
      acima do ruido esperado entre sementes (~0,05). O teste E capaz de
      distinguir.
  H1  pureza >= 0,50 e >= nulo + 0,15  -> o sistema descobre conceitos REAIS
      de dado que nao foi desenhado por mim
  H2  pureza entre acaso e 0,50        -> descobre estrutura fraca
  H3  pureza ~ acaso                   -> o encoder de fluxo nao captura
                                          conceito em dado real

  NULO PAREADO: as mesmas janelas com o EIXO DO TEMPO embaralhado dentro de
  cada uma. Preserva a distribuicao de cada canal e destroi a forma temporal —
  que e exatamente o que se afirma estar sendo capturado. 5 embaralhamentos.

PREVISAO: H1, 0,7. Mecanismo: LAYING/SITTING/STANDING tem assinatura de
aceleracao gravitacional muito distinta (o telefone muda de orientacao), e as
tres de andar tem periodicidade forte. O risco nomeado e que as tres variantes
de ANDAR se fundam entre si — sao a mesma forma com intensidade diferente, e
[[indice-ve-superficie]] mediu que o HDC ordena perto e perde resolucao longe.
Se fundirem, a pureza cai para ~4 grupos uteis de 6.
"""

from __future__ import annotations

import os
import statistics
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np

from novaphys.hdc_indice import Espaco, feixe, sim

RAIZ = r"D:\agi_dados\har\UCI HAR Dataset"
CANAIS = ("body_acc_x", "body_acc_y", "body_acc_z",
          "body_gyro_x", "body_gyro_y", "body_gyro_z",
          "total_acc_x", "total_acc_y", "total_acc_z")
NOMES = {1: "WALKING", 2: "WALK_UP", 3: "WALK_DOWN",
         4: "SITTING", 5: "STANDING", 6: "LAYING"}
ATRASOS = (0, 1, 4)
NIVEIS = 4                      # -4..+4  -> 9 faixas
SEMENTE = 20260918


def carrega(split: str = "train", n_max: int = 900) -> Tuple[np.ndarray, np.ndarray]:
    """Devolve (janelas T x K, rotulos). Os rotulos so avaliam."""
    base = os.path.join(RAIZ, split, "Inertial Signals")
    cols = []
    for c in CANAIS:
        arq = os.path.join(base, f"{c}_{split}.txt")
        cols.append(np.loadtxt(arq, max_rows=n_max, dtype=np.float32))
    X = np.stack(cols, axis=-1)                     # (n, 128, 9)
    y = np.loadtxt(os.path.join(RAIZ, split, f"y_{split}.txt"),
                   max_rows=n_max, dtype=int)
    return X, y


# ───────────────────────────── encoder rapido de fluxo
class EncFluxo:
    """
    `de_fluxo` vetorizado por CONTAGEM.

    O encoder original monta um vetor por (instante, canal, atraso): 128 x 9 x 3
    = 3.456 por janela, o que em 10 mil janelas vira dezenas de bilhoes de
    operacoes. Mas o numero de termos DISTINTOS e 9 canais x 9 niveis x 3
    atrasos = 243. Entao pre-computa-se os 243 e conta-se quantas vezes cada um
    ocorre; o feixe vira uma soma ponderada. Mesmo resultado, ~14x mais rapido.
    """

    def __init__(self, e: Espaco, modo: str = "transicao"):
        """
        `modo`:
          "nivel"     histograma de (canal, nivel) por atraso — a 1a versao
          "transicao" histograma de (canal, nivel_t, nivel_t+1)

        POR QUE A 1a VERSAO ESTAVA ERRADA, e o erro e do projeto inteiro e nao
        so deste arquivo: somar `bind(canal, nivel)` sobre TODOS os instantes e
        um HISTOGRAMA, e histograma e INVARIANTE A PERMUTACAO do tempo. Os
        atrasos nao salvam porque cada atraso vira seu proprio histograma
        independente. Medido em dado real: pureza 0,528 contra nulo de tempo
        embaralhado 0,537 — EMPATE. O `de_fluxo` de `hdc_indice` tem a mesma
        propriedade e ninguem tinha notado.

        "transicao" conta pares ADJACENTES de nivel: (canal, nivel_t,
        nivel_t+1). Sao 9 canais x 9 x 9 = 729 termos distintos, ainda
        contaveis, e sensivel a permutacao POR CONSTRUCAO.
        """
        self.e, self.modo = e, modo
        self.nn = 2 * NIVEIS + 1
        self.nk = len(CANAIS)
        if modo == "nivel":
            self.tab = np.stack([
                e.desloca(e.bind(e.atomo(f"c:{k}"), e.atomo(f"n:{n}")), d)
                for d in ATRASOS for k in range(self.nk)
                for n in range(-NIVEIS, NIVEIS + 1)
            ]).astype(np.int32)
        else:
            self.tab = np.stack([
                e.bind(e.atomo(f"c:{k}"),
                       e.bind(e.atomo(f"a:{a}"), e.desloca(e.atomo(f"b:{b}"), 3)))
                for k in range(self.nk)
                for a in range(self.nn) for b in range(self.nn)
            ]).astype(np.int32)                       # (9*9*9, dim)

    def _idx(self, A: np.ndarray) -> np.ndarray:
        mu, sd = A.mean(0), A.std(0)
        Z = (A - mu) / np.where(sd > 1e-9, sd, 1.0)
        Q = np.clip(np.round(Z * 2), -NIVEIS, NIVEIS).astype(int) + NIVEIS
        conta = np.zeros(len(self.tab), dtype=np.int32)
        if self.modo == "nivel":
            for di, d in enumerate(ATRASOS):
                Qd = Q[:len(Q) - d] if d else Q
                for k in range(self.nk):
                    b = np.bincount(Qd[:, k], minlength=self.nn)
                    base = di * self.nk * self.nn + k * self.nn
                    conta[base:base + self.nn] += b
        else:
            for k in range(self.nk):
                a, b = Q[:-1, k], Q[1:, k]
                par = a * self.nn + b                 # indice do par adjacente
                cnt = np.bincount(par, minlength=self.nn * self.nn)
                base = k * self.nn * self.nn
                conta[base:base + self.nn * self.nn] += cnt
        return conta

    def codifica(self, A: np.ndarray) -> np.ndarray:
        c = self._idx(A)
        s = (self.tab * c[:, None]).sum(axis=0)
        return np.where(s >= 0, 1, -1).astype(np.int8)


# ───────────────────────────────────── agrupamento
def agrupa(V: np.ndarray, k: int, rng, n_iter: int = 15) -> np.ndarray:
    """k-medoides sobre similaridade, vetorizado."""
    Vf = V.astype(np.float32)
    n = len(V)
    cen = rng.choice(n, size=k, replace=False)
    atrib = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        S = Vf @ Vf[cen].T                      # (n, k)
        atrib = np.argmax(S, axis=1)
        novos = []
        for c in range(k):
            m = np.where(atrib == c)[0]
            if len(m) == 0:
                novos.append(rng.integers(n)); continue
            Sm = Vf[m] @ Vf[m].T
            novos.append(int(m[np.argmax(Sm.sum(axis=1))]))
        novos = np.array(novos)
        if np.array_equal(np.sort(novos), np.sort(cen)):
            break
        cen = novos
    return atrib


def pureza(atrib: np.ndarray, y: np.ndarray) -> float:
    acerto = 0
    for c in set(atrib.tolist()):
        rot = y[atrib == c]
        if len(rot):
            acerto += Counter(rot.tolist()).most_common(1)[0][1]
    return acerto / len(y)


def experimento(n_max: int = 900, k: int = 12,
                modo: str = "transicao") -> None:
    e = Espaco()
    enc = EncFluxo(e, modo)
    rng = np.random.default_rng(SEMENTE)

    X, y = carrega("train", n_max)
    print("=" * 78)
    print("HAR — o sistema descobre 'andar' em dado REAL, sem ninguem dizer?")
    print("=" * 78)
    cc = Counter(y.tolist())
    print(f"  {len(X)} janelas de {X.shape[1]} instantes x {X.shape[2]} canais")
    print("  atividades: " + ", ".join(f"{NOMES[a]} {n}" for a, n in
                                       sorted(cc.items())))
    acaso = max(cc.values()) / len(y)
    print(f"  acaso de pureza (classe majoritaria): {acaso:.3f}")
    print(f"  folga ate o teto: {1.0-acaso:.3f}   (H0 exige folga >> ruido)\n")

    V = np.stack([enc.codifica(A) for A in X])
    at = agrupa(V, k, rng)
    p = pureza(at, y)

    # NULO: embaralha o EIXO DO TEMPO dentro de cada janela
    ps_n = []
    for s in range(5):
        r2 = np.random.default_rng(1000 + s)
        Xe = np.stack([A[r2.permutation(A.shape[0])] for A in X])
        Vn = np.stack([enc.codifica(A) for A in Xe])
        ps_n.append(pureza(agrupa(Vn, k, np.random.default_rng(SEMENTE)), y))
    pn, sn = statistics.mean(ps_n), statistics.pstdev(ps_n)

    print(f"  pureza REAL:  {p:.3f}   ({k} grupos, encoder={modo})")
    print(f"  pureza NULO (tempo embaralhado, 5x): {pn:.3f} +- {sn:.3f}\n")
    print("  composicao dos grupos:")
    for c in sorted(set(at.tolist())):
        rot = Counter(y[at == c].tolist())
        d = ", ".join(f"{NOMES[a]}:{n}" for a, n in rot.most_common(3))
        print(f"    grupo {c:2d} ({(at == c).sum():3d} janelas)  {d}")

    print("\n" + "=" * 78)
    if p >= 0.50 and p >= pn + 0.15:
        print(f"H1 . O SISTEMA DESCOBRE CONCEITOS REAIS: pureza {p:.3f} contra")
        print(f"     nulo {pn:.3f} e acaso {acaso:.3f}, em dado que nao foi")
        print("     desenhado por mim, sem ver um rotulo sequer.")
    elif p > acaso + 0.05:
        print(f"H2 . estrutura fraca: {p:.3f} contra acaso {acaso:.3f} e nulo "
              f"{pn:.3f}.")
    else:
        print(f"H3 . o encoder de fluxo nao captura conceito em dado real "
              f"({p:.3f} x {acaso:.3f}).")


if __name__ == "__main__":
    for m in ("nivel", "transicao"):
        experimento(modo=m)
        print("")
