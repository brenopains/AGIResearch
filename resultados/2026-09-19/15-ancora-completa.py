"""
novaphys.ancora_completa — a mesma medida, agora com TODA a gente
==================================================================

O DEFEITO QUE ISTO CONSERTA
-----------------------------
`har_pessoas.py` usava `N_TREINO=2400` e `N_TESTE=1200` com `max_rows`, e
imprimia "21 voluntarios" e "9 voluntarios DIFERENTES". Conferido:

    train  7352 janelas / 21 pessoas   ->  usados 2400 /  8 pessoas
    test   2947 janelas /  9 pessoas   ->  usados 1200 /  4 pessoas

Toda a linha de transferencia entre pessoas rodou com **8 treino e 4 teste**, e
a contagem errada foi para a memoria e para o ARQUITETURA.

O que SOBREVIVE ao defeito: o split e legitimo (1,3,5,6,7,8,11,14 contra
2,4,9,10 — disjuntos), entao "transfere para gente nunca vista" continua de pe.
O que NAO sobrevive: a forca da afirmacao, e sobretudo o PAREAMENTO da ancora
externa — em [[ancora-externa-har]] eu rodei o braco PUBLICADO no split OFICIAL
COMPLETO (21/9) e os outros dois no truncado (8/4). O -0,407 contra a linha
publicada estava inflado por essa assimetria.

Aqui os tres bracos rodam no MESMO dado completo.

PRE-REGISTRO
--------------
  C0  PRE-CONDICAO: PUBLICADO reproduz ~0,96 (inalterado, ja rodava completo).
  C1  com 21 pessoas de treino em vez de 8, NOSSO sobe >= 0,05
      -> parte do buraco era falta de dado, nao de metodo, e a conclusao de
         [[ancora-externa-har]] precisa ser suavizada na mesma medida
  C2  NOSSO praticamente inalterado -> o buraco e de metodo, e a conclusao
      anterior vale como estava (so os numeros de pessoas mudam)
  C3  NOSSO cai -> mais gente e mais variedade individual, o que reforcaria
      [[motivo-e-pessoal]] ainda mais forte do que ja esta

PREVISAO: C2, 0,55. Mecanismo: [[limiar-de-dado-e-lei]] mediu que o lexico e
limitado por DADO abaixo de ~1000-1500 tokens, e aqui ja havia muito mais que
isso; e [[motivo-e-pessoal]] diz que o motivo e individual, entao mais pessoas
nao deveria ajudar o transporte. Nomeio C1 como possivel porque 8 pessoas e
pouco para os medoides cobrirem variedade de marcha, e o 2x2 do leitor mostrou
que cobertura de prototipo importa.
"""

from __future__ import annotations

import os
import statistics
from collections import Counter

import numpy as np

from novaphys.har import RAIZ, carrega
from novaphys.har_pessoas import K_GRUPOS, PipelineCongelado, SEMENTE
from novaphys.hdc_indice import Espaco
from novaphys.abstrai_motivo import analisa, vetores
from novaphys.ancora_externa import estatisticas, publicado
from novaphys.leitor_grupos import le, um_vizinho

TODO_TR, TODO_TE = 7352, 2947
N_SORT = 5


def experimento() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    e = Espaco()
    rng = np.random.default_rng(SEMENTE)
    Xtr, ytr = carrega("train", TODO_TR)
    Xte, yte = carrega("test", TODO_TE)
    s_tr = np.loadtxt(os.path.join(RAIZ, "train", "subject_train.txt"),
                      dtype=int)
    s_te = np.loadtxt(os.path.join(RAIZ, "test", "subject_test.txt"), dtype=int)

    print("=" * 78)
    print("ANCORA COMPLETA — os tres bracos no MESMO dado, 21 x 9 pessoas")
    print("=" * 78)
    print("  treino {} janelas / {} pessoas".format(
        len(Xtr), len(set(s_tr.tolist()))))
    print("  teste  {} janelas / {} pessoas".format(
        len(Xte), len(set(s_te.tolist()))))
    maioria = max(Counter(yte.tolist()).values()) / len(yte)
    print("  classe majoritaria: {:.3f}\n".format(maioria))

    p = PipelineCongelado(e)
    p.ajusta(Xtr, ytr, rng, K_GRUPOS)
    print("  congelado: {} entradas de lexico".format(len(p.lex)), flush=True)
    ped_tr, ped_te = analisa(p, Xtr), analisa(p, Xte)
    print("  Viterbi: {} + {} janelas analisadas\n".format(
        len(ped_tr), len(ped_te)), flush=True)

    n12, n24, n192, viz, teto = [], [], [], [], []
    for t in range(N_SORT):
        rot = (lambda q: lambda w: ["mot" + str(q) + ":" + w])(t)
        Vtr, Vte = vetores(e, ped_tr, rot), vetores(e, ped_te, rot)
        n12.append(le(Vtr, ytr, Vte, yte, np.random.default_rng(SEMENTE + t), 12))
        n24.append(le(Vtr, ytr, Vte, yte, np.random.default_rng(SEMENTE + t), 24))
        n192.append(le(Vtr, ytr, Vte, yte, np.random.default_rng(SEMENTE + t), 192))
        viz.append(um_vizinho(Vtr, ytr, Vte, yte))
        s = StandardScaler().fit(Vtr.astype(np.float32))
        m = LogisticRegression(max_iter=3000).fit(
            s.transform(Vtr.astype(np.float32)), ytr)
        teto.append(float(np.mean(
            m.predict(s.transform(Vte.astype(np.float32))) == yte)))
        print("  sorteio {} pronto".format(t), flush=True)

    Atr, Ate = estatisticas(Xtr), estatisticas(Xte)
    s = StandardScaler().fit(Atr)
    lr = LogisticRegression(max_iter=3000).fit(s.transform(Atr), ytr)
    est = float(np.mean(lr.predict(s.transform(Ate)) == yte))

    pub = publicado()

    def ms(v):
        return "{:.3f} +- {:.3f}".format(statistics.mean(v),
                                         statistics.pstdev(v))

    print("\n" + "=" * 78)
    print("  PUBLICADO  561 atributos dos autores + SVM      {:.3f}".format(pub))
    print("  PAREADO    7 linhas de estatistica por canal    {:.3f}".format(est))
    print("  NOSSO      medoides(12), como estava            {}".format(ms(n12)))
    print("  NOSSO      medoides(24)                         {}".format(ms(n24)))
    print("  NOSSO      medoides(192)                        {}".format(ms(n192)))
    print("  teto da representacao (logistica no feixe)      {}".format(ms(teto)))
    print("  1-vizinho (degenerado, referencia)              {}".format(ms(viz)))

    m12 = statistics.mean(n12)
    print("\n" + "=" * 78)
    print("  com 8 pessoas de treino, medoides(12) dava 0,560 +- 0,039")
    print("  com 21 pessoas de treino, da {:.3f}  ({:+.3f})".format(
        m12, m12 - 0.560))
    if m12 >= 0.560 + 0.05:
        print("\nC1 . parte do buraco era FALTA DE DADO. A conclusao de")
        print("     ancora-externa-har precisa ser suavizada na mesma medida.")
    elif m12 <= 0.560 - 0.05:
        print("\nC3 . CAIU com mais gente: mais variedade individual piora o")
        print("     transporte, e motivo-e-pessoal fica ainda mais forte.")
    else:
        print("\nC2 . praticamente inalterado: o buraco e de METODO, nao de")
        print("     dado. A conclusao anterior vale; so as contagens de pessoas")
        print("     estavam erradas.")


if __name__ == "__main__":
    experimento()
