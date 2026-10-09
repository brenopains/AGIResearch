"""
novaphys.har_pessoas — reconhece atividade em gente que NUNCA viu?
===================================================================

DE ONDE VEM
-------------
De uma revisao externa do projeto (18/09), que apontou — corretamente — que
`har_lexico.py` mede PUREZA TRANSDUTIVA: ajusta a quantizacao com as janelas
disponiveis, aprende o lexico em parte delas, agrupa o conjunto e mede no MESMO
conjunto. Pureza nao e acuracia em participantes novos.

A divisao oficial do UCI separa VOLUNTARIOS entre treino (21 pessoas) e teste
(9 pessoas) justamente para isso. Aqui tudo e ajustado so no treino, CONGELADO,
e aplicado a pessoas que o sistema nunca viu.

O QUE E CONGELADO NO TREINO
-----------------------------
  1. media e desvio por canal (normalizacao)
  2. centroides da quantizacao vetorial
  3. o lexico de motivos descoberto por MDL
  4. os medoides dos grupos
  5. o mapeamento grupo -> atividade (por maioria NO TREINO)

Nada disso ve o teste. O passo 5 usa rotulos do TREINO, o que e legitimo: e como
o sistema seria implantado. O que nao pode vazar sao os participantes finais.

PRE-REGISTRO
--------------
  P0  PRE-CONDICAO E CONTA DE FOLGA. No teste a classe majoritaria e ~18% e o
      teto e 100%: folga ~0,82 contra ruido pequeno (2.947 janelas). O teste
      CONSEGUE distinguir — a conta que [[escolher-alvo-sem-folga]] me ensinou
      a fazer ANTES.
  P1  acuracia em participantes novos >= 0,50 E >= nulo + 0,15
      -> o que foi descoberto TRANSFERE para pessoas que o sistema nunca viu
  P2  entre a maioria e 0,50   -> transfere pouco
  P3  ~ classe majoritaria     -> a estrutura era especifica daquelas pessoas

  NULO PAREADO: mesmo pipeline congelado, mas com o EIXO DO TEMPO embaralhado
  nas janelas de TESTE. Mede quanto da acuracia vem de forma temporal e quanto
  vem de distribuicao. 5 embaralhamentos.

  Referencia interna: `har_lexico` deu pureza transdutiva 0,778 e o nulo
  temporal 0,567 (+0,210). Aqui a pergunta e outra e mais dura.

PREVISAO: P1, 0,55. Mecanismo: o motivo de passada e a mesma FORMA entre
pessoas — andar e andar —, e a normalizacao por janela tira amplitude
individual. O risco nomeado e a quantizacao: os centroides foram ajustados nos
21 voluntarios de treino, e se um voluntario de teste tiver faixa de movimento
fora dessa cobertura, os simbolos dele caem todos no mesmo centroide e o motivo
some. Se P3 sair, testar primeiro se o problema e cobertura de centroide antes
de concluir qualquer coisa sobre transferencia.
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np

from novaphys.har import NOMES, carrega
from novaphys.hdc_indice import Espaco, feixe, sim
from novaphys.har_lexico import ALFA, V_SIMB
from novaphys.lingua import _viterbi, segmenta

SEMENTE = 20260918
K_GRUPOS = 12
N_TREINO = 2400
N_TESTE = 1200


class PipelineCongelado:
    """Tudo ajustado no treino; `aplica` nunca reajusta nada."""

    def __init__(self, e: Espaco):
        self.e = e

    def ajusta(self, X: np.ndarray, y: np.ndarray, rng, k: int) -> None:
        n, T, K = X.shape
        todos = X.reshape(-1, K)
        self.mu, self.sd = todos.mean(0), todos.std(0)          # (1)
        Z = (todos - self.mu) / np.where(self.sd > 1e-9, self.sd, 1.0)
        am = Z[rng.choice(len(Z), size=min(20000, len(Z)), replace=False)]
        cen = am[rng.choice(len(am), size=V_SIMB, replace=False)]
        for _ in range(12):                                      # (2)
            d = ((am[:, None, :] - cen[None, :, :]) ** 2).sum(-1)
            at = np.argmin(d, axis=1)
            novos = np.stack([am[at == c].mean(0) if (at == c).any() else cen[c]
                              for c in range(V_SIMB)])
            if np.allclose(novos, cen):
                break
            cen = novos
        self.cen = cen

        cad = self._cadeias(X)
        _t, lex, _a, _b = segmenta("".join(cad[:150]), rodadas=400, cand=24)
        self.lex = set(lex) | set(ALFA[:V_SIMB])                 # (3)

        V = np.stack([self._vetor(s) for s in cad])
        Vf = V.astype(np.float32)
        idx = rng.choice(len(V), size=k, replace=False)
        atrib = np.zeros(len(V), dtype=int)
        for _ in range(15):                                      # (4)
            atrib = np.argmax(Vf @ Vf[idx].T, axis=1)
            novos = []
            for c in range(k):
                m = np.where(atrib == c)[0]
                if len(m) == 0:
                    novos.append(rng.integers(len(V))); continue
                novos.append(int(m[np.argmax((Vf[m] @ Vf[m].T).sum(1))]))
            novos = np.array(novos)
            if np.array_equal(np.sort(novos), np.sort(idx)):
                break
            idx = novos
        self.medoides = V[idx]

        self.nome = {}                                           # (5)
        for c in range(k):
            rot = y[atrib == c]
            self.nome[c] = (Counter(rot.tolist()).most_common(1)[0][0]
                            if len(rot) else 1)

    def _cadeias(self, X: np.ndarray) -> List[str]:
        n, T, K = X.shape
        Z = (X.reshape(-1, K) - self.mu) / np.where(self.sd > 1e-9,
                                                    self.sd, 1.0)
        d = ((Z[:, None, :] - self.cen[None, :, :]) ** 2).sum(-1)
        s = np.argmin(d, axis=1).reshape(n, T)
        return ["".join(ALFA[c] for c in l) for l in s]

    def _vetor(self, s: str) -> np.ndarray:
        prob = {w: 1.0 / len(self.lex) for w in self.lex}
        ped = _viterbi(s, self.lex, prob)
        t = [self.e.atomo(f"mot:{w}") for w in ped if len(w) > 1]
        return feixe(t, self.e.dim) if t else self.e.atomo("vazio")

    def aplica(self, X: np.ndarray) -> np.ndarray:
        """Prevê a atividade de janelas de pessoas NOVAS. Nada e reajustado."""
        V = np.stack([self._vetor(s) for s in self._cadeias(X)])
        M = self.medoides.astype(np.float32)
        g = np.argmax(V.astype(np.float32) @ M.T, axis=1)
        return np.array([self.nome[int(c)] for c in g])


def experimento() -> None:
    e = Espaco()
    rng = np.random.default_rng(SEMENTE)
    Xtr, ytr = carrega("train", N_TREINO)
    Xte, yte = carrega("test", N_TESTE)

    print("=" * 78)
    print("HAR PESSOAS — reconhece atividade em gente que NUNCA viu?")
    print("=" * 78)
    print(f"  treino: {len(Xtr)} janelas (21 voluntarios)")
    print(f"  teste:  {len(Xte)} janelas (9 voluntarios DIFERENTES)")
    maioria = max(Counter(yte.tolist()).values()) / len(yte)
    print(f"  classe majoritaria no teste: {maioria:.3f}  (o acaso a bater)")
    print(f"  folga ate o teto: {1-maioria:.3f}\n")

    p = PipelineCongelado(e)
    p.ajusta(Xtr, ytr, rng, K_GRUPOS)
    print(f"  congelado: {len(p.lex)} entradas de lexico, {K_GRUPOS} grupos")
    print(f"  grupos nomeados: "
          f"{dict(Counter(NOMES[v] for v in p.nome.values()))}\n")

    pred = p.aplica(Xte)
    ac = float(np.mean(pred == yte))

    acs_n = []
    for s in range(5):
        r2 = np.random.default_rng(2000 + s)
        Xe = np.stack([A[r2.permutation(A.shape[0])] for A in Xte])
        acs_n.append(float(np.mean(p.aplica(Xe) == yte)))
    an, sn = statistics.mean(acs_n), statistics.pstdev(acs_n)

    print(f"  ACURACIA em pessoas novas:        {ac:.3f}")
    print(f"  nulo (tempo embaralhado, 5x):     {an:.3f} +- {sn:.3f}")
    print(f"  classe majoritaria:               {maioria:.3f}\n")

    print("  por atividade (revocacao):")
    for a in sorted(set(yte.tolist())):
        m = yte == a
        print(f"    {NOMES[a]:11s} {float(np.mean(pred[m] == a)):.3f}  "
              f"({m.sum()} janelas)")

    print("\n" + "=" * 78)
    if ac >= 0.50 and ac >= an + 0.15:
        print(f"P1 . TRANSFERE PARA PESSOAS NOVAS: {ac:.3f} contra maioria")
        print(f"     {maioria:.3f} e nulo temporal {an:.3f}. O que foi")
        print("     descoberto nao era especifico daqueles voluntarios.")
    elif ac > maioria + 0.05:
        print(f"P2 . transfere pouco: {ac:.3f} contra maioria {maioria:.3f} e")
        print(f"     nulo {an:.3f}.")
    else:
        print(f"P3 . NAO transfere ({ac:.3f} x maioria {maioria:.3f}).")
        print("     Antes de concluir: testar se e cobertura de centroide —")
        print("     o risco nomeado no pre-registro.")


if __name__ == "__main__":
    experimento()
