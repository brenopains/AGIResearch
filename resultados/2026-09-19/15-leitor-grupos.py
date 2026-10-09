"""
novaphys.leitor_grupos — o vao entre o que se descobre e o que se consegue ler
===============================================================================

DE ONDE VEM
-------------
[[ancora-externa-har]], 2x2 de codificacao x leitor:

    representacao (feixe HDC) + logistica      0,721
    representacao (feixe HDC) + medoides(12)   0,548

113% da perda esta no LEITOR. Mas os medoides nao sao detalhe trocavel: sao o que
faz os grupos se formarem SOZINHOS, sem rotulo, para so depois receberem nome.
Trocar por logistica compra 17 pontos e muda a afirmacao de DESCOBERTA para
classificador supervisionado. A pergunta certa e mais estreita:

    existe leitor que PRESERVE o auto-formar-se dos grupos e mesmo assim extraia
    mais que 0,548 da mesma representacao?

A PREVISAO VEM DA PROPRIA MEMORIA DO PROJETO
----------------------------------------------
[[motivo-e-pessoal]] mediu que o motivo de passada e assinatura INDIVIDUAL de
marcha. Com k=12 e 6 atividades, sao DOIS prototipos por atividade para cobrir 21
pessoas. Isso e falta de RESOLUCAO, nao de capacidade — e resolucao se conserta
sem tocar em nada que importe.

O LIMITE CONCEITUAL, NOMEADO ANTES
------------------------------------
Quando k -> n, cada janela vira seu proprio prototipo e o leitor degenera em
vizinho-mais-proximo. Ainda funciona, mas nao e mais "grupo que se forma": e
memoria de exemplares. Entao a varredura reporta a COMPRESSAO junto (n/k) e o
1-vizinho entra como referencia degenerada explicita. Um k moderado que capture a
maior parte do vao COM compressao real e o resultado bom; so o 1-vizinho
funcionar e resultado ruim disfarcado de bom.

O PROTOCOLO, E ELE E O PONTO
------------------------------
A revisao externa de 18/09 pegou selecao-no-teste em `primitiva_do_dado`. Varrer
k e pegar o melhor no teste seria o mesmo erro. Entao:

  1. os 21 voluntarios de TREINO sao partidos POR PESSOA em interno/validacao
  2. cada k e ajustado no interno, nomeado no interno, medido na validacao
  3. o k vencedor e escolhido AI, sem ver o teste
  4. reajusta nos 21 e mede UMA vez nos 9 nunca vistos

PRE-REGISTRO
--------------
  H0  PRE-CONDICAO: k=12 reproduz ~0,548 e a logistica ~0,721 na mesma
      representacao. Senao o instrumento mudou.
  H1  o k escolhido na validacao bate k=12 por >= 0,05 nos 9 do teste, em
      >= 4 dos 5 sorteios de atomo
      -> o agrupamento auto-formado estava SUBDIMENSIONADO, nao quebrado
  H2  nenhum k ajuda -> a perda nao e de resolucao; esta na atribuicao dura
      ou na nomeacao por maioria
  H3  so k perto de n ajuda -> degenerou em vizinho-mais-proximo e a historia
      de "grupo que se forma" nao sobrevive

PREVISAO: H1, 0,6, com o k vencedor entre 48 e 192. Mecanismo direto de
[[motivo-e-pessoal]]. Risco nomeado: se o ganho so aparecer em k >= 384
(compressao < 7:1), e H3 e eu devo dizer isso em vez de comemorar o numero.
"""

from __future__ import annotations

import os
import statistics
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

from novaphys.har import RAIZ, carrega
from novaphys.har_pessoas import (K_GRUPOS, N_TESTE, N_TREINO,
                                  PipelineCongelado, SEMENTE)
from novaphys.hdc_indice import Espaco
from novaphys.abstrai_motivo import analisa, vetores

KS = (12, 24, 48, 96, 192, 384)
N_SORT = 5


def sujeitos(split: str, n: int) -> np.ndarray:
    return np.loadtxt(os.path.join(RAIZ, split, "subject_" + split + ".txt"),
                      max_rows=n, dtype=int)


def medoides(V: np.ndarray, rng, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """k-medoides por produto interno. Devolve (prototipos, atribuicao)."""
    Vf = V.astype(np.float32)
    idx = rng.choice(len(V), size=min(k, len(V)), replace=False)
    atrib = np.zeros(len(V), dtype=int)
    for _ in range(15):
        atrib = np.argmax(Vf @ Vf[idx].T, axis=1)
        novos = []
        for c in range(len(idx)):
            m = np.where(atrib == c)[0]
            if len(m) == 0:
                novos.append(int(rng.integers(len(V))))
                continue
            if len(m) > 400:                      # medoide exato fica caro
                m = m[rng.choice(len(m), 400, replace=False)]
            novos.append(int(m[np.argmax((Vf[m] @ Vf[m].T).sum(1))]))
        novos = np.array(novos)
        if np.array_equal(np.sort(novos), np.sort(idx)):
            break
        idx = novos
    return V[idx], atrib


def le(Vtr, ytr, Vte, yte, rng, k: int) -> float:
    """Agrupa SEM rotulo, nomeia por maioria NO TREINO, le o teste."""
    med, atrib = medoides(Vtr, rng, k)
    nome = {}
    for c in range(len(med)):
        rot = ytr[atrib == c]
        nome[c] = Counter(rot.tolist()).most_common(1)[0][0] if len(rot) else 1
    g = np.argmax(Vte.astype(np.float32) @ med.astype(np.float32).T, axis=1)
    return float(np.mean(np.array([nome[int(c)] for c in g]) == yte))


def um_vizinho(Vtr, ytr, Vte, yte) -> float:
    g = np.argmax(Vte.astype(np.float32) @ Vtr.astype(np.float32).T, axis=1)
    return float(np.mean(ytr[g] == yte))


def experimento() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    e = Espaco()
    rng = np.random.default_rng(SEMENTE)
    Xtr, ytr = carrega("train", N_TREINO)
    Xte, yte = carrega("test", N_TESTE)
    s_tr = sujeitos("train", N_TREINO)
    p = PipelineCongelado(e)
    p.ajusta(Xtr, ytr, rng, K_GRUPOS)
    ped_tr, ped_te = analisa(p, Xtr), analisa(p, Xte)

    pessoas = sorted(set(s_tr.tolist()))
    corte = int(len(pessoas) * 2 / 3)
    p_int = set(pessoas[:corte])
    m_int = np.array([s in p_int for s in s_tr])
    m_val = ~m_int

    print("=" * 78)
    print("LEITOR DE GRUPOS — o vao entre descobrir e conseguir ler")
    print("=" * 78)
    print("  interno {} pessoas / validacao {} pessoas / teste 9 nunca vistas"
          .format(len(p_int), len(pessoas) - len(p_int)))
    print("  o k e escolhido na VALIDACAO; o teste e tocado uma vez.\n")

    escolhas: List[int] = []
    res: Dict[int, List[float]] = {k: [] for k in KS}
    base, teto, viz = [], [], []

    for t in range(N_SORT):
        rot = (lambda q: lambda w: ["mot" + str(q) + ":" + w])(t)
        Vtr, Vte = vetores(e, ped_tr, rot), vetores(e, ped_te, rot)

        melhor_k, melhor_v = KS[0], -1.0
        for k in KS:
            v = le(Vtr[m_int], ytr[m_int], Vtr[m_val], ytr[m_val],
                   np.random.default_rng(SEMENTE + t), k)
            if v > melhor_v:
                melhor_k, melhor_v = k, v
        escolhas.append(melhor_k)

        for k in KS:
            res[k].append(le(Vtr, ytr, Vte, yte,
                             np.random.default_rng(SEMENTE + t), k))
        base.append(res[12][-1])
        viz.append(um_vizinho(Vtr, ytr, Vte, yte))

        s = StandardScaler().fit(Vtr.astype(np.float32))
        m = LogisticRegression(max_iter=3000).fit(
            s.transform(Vtr.astype(np.float32)), ytr)
        teto.append(float(np.mean(
            m.predict(s.transform(Vte.astype(np.float32))) == yte)))
        print("  sorteio {}: k escolhido na validacao = {}"
              .format(t, melhor_k), flush=True)

    print("\n  H0 — k=12 {:.3f} (esperado ~0,548), logistica {:.3f} "
          "(esperado ~0,721)".format(statistics.mean(base),
                                     statistics.mean(teto)))

    print("\n      k   compressao   acuracia nos 9 (5 sorteios)")
    print("  ------   ----------   ---------------------------")
    for k in KS:
        print("  {:6d}   {:8.0f}:1   {:.3f} +- {:.3f}".format(
            k, N_TREINO / k, statistics.mean(res[k]),
            statistics.pstdev(res[k])))
    print("  {:>6}   {:8.0f}:1   {:.3f} +- {:.3f}   (degenerado, referencia)"
          .format("1-viz", 1, statistics.mean(viz), statistics.pstdev(viz)))

    k_mod = Counter(escolhas).most_common(1)[0][0]
    esc = [res[escolhas[t]][t] for t in range(N_SORT)]
    ganhos = [esc[t] - base[t] for t in range(N_SORT)]
    g, sg = statistics.mean(ganhos), statistics.pstdev(ganhos)
    n_ok = sum(1 for x in ganhos if x >= 0.05)
    vao = statistics.mean(teto) - statistics.mean(base)

    print("\n  k escolhido na validacao: {}".format(Counter(escolhas).most_common()))
    print("  acuracia com o k escolhido: {:.3f}".format(statistics.mean(esc)))
    print("  ganho sobre k=12 (pareado): {:+.3f} +- {:.3f}, >=0,05 em {}/{}"
          .format(g, sg, n_ok, N_SORT))
    print("  fatia do vao recuperada: {:.0f}%".format(100 * g / max(vao, 1e-9)))

    print("\n" + "=" * 78)
    if n_ok >= 4 and k_mod < 384:
        print("H1 . O AGRUPAMENTO ESTAVA SUBDIMENSIONADO. k={} ({:.0f}:1 de "
              "compressao)".format(k_mod, N_TREINO / k_mod))
        print("     recupera {:.0f}% do vao sem trocar nada que importe: os "
              "grupos".format(100 * g / max(vao, 1e-9)))
        print("     continuam se formando sozinhos e so depois recebem nome.")
    elif n_ok >= 4:
        print("H3 . so ajuda em k={} (compressao {:.0f}:1): degenerou em memoria"
              .format(k_mod, N_TREINO / k_mod))
        print("     de exemplares, nao e mais grupo que se forma. Ganho real,")
        print("     historia diferente.")
    else:
        print("H2 . resolucao nao era o problema ({:+.3f}). A perda esta na"
              .format(g))
        print("     atribuicao dura ou na nomeacao por maioria.")


if __name__ == "__main__":
    experimento()
