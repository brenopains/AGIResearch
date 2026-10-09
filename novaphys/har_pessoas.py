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

from novaphys.har import NOMES, RAIZ, carrega
from novaphys.hdc_indice import Espaco, feixe, sim
from novaphys.har_lexico import ALFA, V_SIMB
from novaphys.lingua import _viterbi, segmenta

SEMENTE = 20260918
K_GRUPOS = 12
N_TREINO = 2400
N_TESTE = 1200
N_SORTEIOS = 10


class PipelineCongelado:
    """Tudo ajustado no treino; `aplica` nunca reajusta nada."""

    def __init__(self, e: Espaco, sal: str = ""):
        self.e = e
        self.sal = sal          # so muda o hash do atomo: ZERO informacao

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
        self._finaliza(X, y, rng, k)

    def clona(self, sal: str, X: np.ndarray, y: np.ndarray, rng, k: int
              ) -> "PipelineCongelado":
        """
        Mesma normalizacao, mesmos centroides, MESMO lexico — so o sal muda.
        E uma bijecao dos rotulos, logo carrega informacao identica; serve para
        medir o piso de ruido do sorteio de atomos e para parear contra ele.
        Ver [[piso-de-ruido-e-pareamento]].
        """
        q = PipelineCongelado(self.e, sal)
        q.mu, q.sd, q.cen, q.lex = self.mu, self.sd, self.cen, self.lex
        q._finaliza(X, y, rng, k)
        return q

    def _finaliza(self, X: np.ndarray, y: np.ndarray, rng, k: int) -> None:
        cad = self._cadeias(X)
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
        t = [self.e.atomo(f"mot:{self.sal}{w}") for w in ped if len(w) > 1]
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
    # NUNCA MAIS afirmar contagem de pessoa em texto fixo: com max_rows, 2400
    # janelas cobrem 8 dos 21 voluntarios e 1200 cobrem 4 dos 9. O codigo
    # imprimiu "21" e "9" por meses e isso foi para a memoria e para o
    # ARQUITETURA. Agora e MEDIDO. Ver [[limite-de-linhas-mentiu]].
    import os as _os
    _st = np.loadtxt(_os.path.join(RAIZ, "train", "subject_train.txt"),
                     max_rows=N_TREINO, dtype=int)
    _se = np.loadtxt(_os.path.join(RAIZ, "test", "subject_test.txt"),
                     max_rows=N_TESTE, dtype=int)
    print(f"  treino: {len(Xtr)} janelas ({len(set(_st.tolist()))} voluntarios)")
    print(f"  teste:  {len(Xte)} janelas ({len(set(_se.tolist()))} voluntarios "
          f"DIFERENTES)")
    print(f"  (subamostra declarada: a base tem 7352/2947 e 21/9 pessoas)")
    maioria = max(Counter(yte.tolist()).values()) / len(yte)
    print(f"  classe majoritaria no teste: {maioria:.3f}  (o acaso a bater)")
    print(f"  folga ate o teto: {1-maioria:.3f}\n")

    p = PipelineCongelado(e)
    p.ajusta(Xtr, ytr, rng, K_GRUPOS)
    print(f"  congelado: {len(p.lex)} entradas de lexico, {K_GRUPOS} grupos")
    print(f"  grupos nomeados: "
          f"{dict(Counter(NOMES[v] for v in p.nome.values()))}\n")

    # Os dois embaralhamentos sao FIXOS: so o sal varia entre sorteios, para
    # que real e nulo caiam sempre no mesmo par — o ruido do sorteio de atomos
    # e modo comum e so assim ele cancela. Ver [[piso-de-ruido-e-pareamento]].
    Xemb = []
    for t in range(2):
        r2 = np.random.default_rng(2000 + t)
        Xemb.append(np.stack([A[r2.permutation(A.shape[0])] for A in Xte]))

    reais, difs, pred = [], [], None
    for t in range(N_SORTEIOS):
        q = p if t == 0 else p.clona(f"s{t}:", Xtr, ytr,
                                     np.random.default_rng(SEMENTE), K_GRUPOS)
        pt = q.aplica(Xte)
        a = float(np.mean(pt == yte))
        n = statistics.mean(float(np.mean(q.aplica(Xe) == yte)) for Xe in Xemb)
        if t == 0:
            pred = pt
        reais.append(a)
        difs.append(a - n)
    ac, sac = statistics.mean(reais), statistics.pstdev(reais)
    md, sd_d = statistics.mean(difs), statistics.pstdev(difs)
    pos = sum(1 for d in difs if d > 0)

    print(f"  ACURACIA em pessoas novas:        {ac:.3f} +- {sac:.3f}"
          f"   ({N_SORTEIOS} sorteios de atomo)")
    print(f"  GANHO sobre o tempo embaralhado:  {md:+.3f} +- {sd_d:.3f}"
          f"   (pareado, positivo {pos}/{N_SORTEIOS})")
    print(f"  classe majoritaria:               {maioria:.3f}")
    print(f"  o NIVEL oscila {sac/max(sd_d,1e-9):.0f}x mais que a DIFERENCA;")
    print( "  trocar so o nome dos atomos ja move o nivel, e nada mais muda.")
    print()

    print("  por atividade (revocacao):")
    for a in sorted(set(yte.tolist())):
        m = yte == a
        print(f"    {NOMES[a]:11s} {float(np.mean(pred[m] == a)):.3f}  "
              f"({m.sum()} janelas)")

    print("\n" + "=" * 78)
    # O 0,15 e o tamanho de efeito PRE-REGISTRADO e continua valendo; o
    # pareamento acrescenta estabilidade, nao substitui a barra. Eu troquei
    # uma pela outra ao instrumentar isto e o veredito virou P1 sem que o
    # efeito mudasse — trave movida, desfeita.
    if ac >= 0.50 and md >= 0.15 and pos == N_SORTEIOS:
        print(f"P1 . TRANSFERE PARA PESSOAS NOVAS: {ac:.3f} contra maioria")
        print(f"     {maioria:.3f}, com ganho temporal pareado {md:+.3f} +-")
        print(f"     {sd_d:.3f} estavel em todos os sorteios. O que foi")
        print("     descoberto nao era especifico daqueles voluntarios.")
    elif ac > maioria + 0.05:
        print(f"P2 . transfere pouco: {ac:.3f} contra maioria {maioria:.3f} e")
        print(f"     ganho temporal pareado {md:+.3f} +- {sd_d:.3f}.")
    else:
        print(f"P3 . NAO transfere ({ac:.3f} x maioria {maioria:.3f}).")
        print("     Antes de concluir: testar se e cobertura de centroide —")
        print("     o risco nomeado no pre-registro.")


if __name__ == "__main__":
    experimento()
