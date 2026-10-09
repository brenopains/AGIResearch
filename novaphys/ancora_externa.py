"""
novaphys.ancora_externa — a ancora que este projeto nunca teve
===============================================================

POR QUE ESTE ARQUIVO EXISTE
-----------------------------
A revisao externa de 18/09 fechou com o apontamento que eu nao consegui derrubar:
*eu escolho o problema, a representacao, o gerador, os rivais e o criterio*. E o
usuario, em 19/09, afirmou que ja temos uma mente artificial geral.

Toda comparacao de HAR ate agora foi INTERNA: 0,560 contra maioria 0,187 e
contra o nulo de tempo embaralhado. Nenhuma contra gente de fora. Acontece que a
propria base traz os **561 atributos de engenharia** dos autores originais, no
MESMO split de voluntarios — entao da para RODAR a linha de base publicada em vez
de cita-la, e descobrir de quanto e o buraco.

E [[canto-chunk-x-markov]] gravou que o primeiro rival que eu escolho e sempre
fraco demais e a favor da hipotese. Entao aqui vao TRES rivais, nao um, e o
melhor deles e que conta.

OS BRACOS
-----------
  PUBLICADO   561 atributos dos autores + SVM linear, split oficial completo.
              Ancora externa: o artigo do UCI reporta ~96%.
  PAREADO     supervisionado sobre a MESMA entrada CRUA que o nosso pipeline ve
              (9 canais x 128 instantes), no MESMO subconjunto (2400/1200).
              Dois classificadores, vale o melhor — para nao montar rival fraco.
  NOSSO       o pipeline congelado de descoberta: 0,560 +- 0,039
              (10 sorteios de atomo, ver [[piso-de-ruido-e-pareamento]]).

A assimetria e declarada: PUBLICADO e PAREADO treinam com rotulo em tudo; NOSSO
descobre a estrutura sem rotulo e so usa rotulo no passo 5, para dar nome aos
grupos. Isso NAO desculpa o buraco — mede de que tamanho ele e.

PRE-REGISTRO
--------------
  E0  PRE-CONDICAO. O braco PUBLICADO reproduz ~0,96 do artigo. Se nao
      reproduzir, minha implementacao esta errada e nada mais vale.
  E1  NOSSO >= PAREADO - 0,05  -> o caminho da descoberta compete com
      aprendizado supervisionado na mesma entrada, e a afirmacao do usuario
      ganha sustentacao empirica
  E2  NOSSO bem abaixo do PAREADO -> o buraco e nosso, nao do dado, e o
      TAMANHO do buraco e a resposta honesta a "ja temos uma mente geral"

PREVISAO: E2, com folga grande. 0,25 de chance para E1. Mecanismo: nosso
pipeline joga fora quase tudo (quantiza 9 canais continuos em 16 simbolos, depois
resume a janela num feixe de motivos), e supervisionado sobre o cru nao joga nada
fora. Se o PAREADO ficar perto de 0,56, o problema e o DADO — 9 canais crus sao
pouco — e ai a comparacao inteira muda de sentido. Nomeio isso agora para nao
reinterpretar depois.
"""

from __future__ import annotations

import os
import numpy as np

from novaphys.har import RAIZ, carrega
from novaphys.har_pessoas import N_TESTE, N_TREINO

NOSSO, NOSSO_SD = 0.560, 0.039


def publicado() -> float:
    from sklearn.svm import LinearSVC
    Xtr = np.loadtxt(os.path.join(RAIZ, "train", "X_train.txt"))
    ytr = np.loadtxt(os.path.join(RAIZ, "train", "y_train.txt"), dtype=int)
    Xte = np.loadtxt(os.path.join(RAIZ, "test", "X_test.txt"))
    yte = np.loadtxt(os.path.join(RAIZ, "test", "y_test.txt"), dtype=int)
    m = LinearSVC(C=1.0, max_iter=5000).fit(Xtr, ytr)
    return float(np.mean(m.predict(Xte) == yte))


def estatisticas(X: np.ndarray) -> np.ndarray:
    """Resumo por canal da janela crua: o que um humano escreveria em 5 minutos."""
    d = np.diff(X, axis=1)
    return np.concatenate([
        X.mean(1), X.std(1), X.min(1), X.max(1),
        (X ** 2).mean(1), np.abs(d).mean(1), d.std(1),
    ], axis=1)


def pareado() -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Xtr, ytr = carrega("train", N_TREINO)
    Xte, yte = carrega("test", N_TESTE)
    out = {}

    Atr, Ate = estatisticas(Xtr), estatisticas(Xte)
    rf = RandomForestClassifier(n_estimators=300, random_state=0,
                                n_jobs=-1).fit(Atr, ytr)
    out["floresta sobre resumo por canal"] = float(np.mean(rf.predict(Ate) == yte))

    s = StandardScaler().fit(Atr)
    lr = LogisticRegression(max_iter=2000).fit(s.transform(Atr), ytr)
    out["logistica sobre resumo por canal"] = float(
        np.mean(lr.predict(s.transform(Ate)) == yte))

    Ftr = Xtr.reshape(len(Xtr), -1)
    Fte = Xte.reshape(len(Xte), -1)
    s2 = StandardScaler().fit(Ftr)
    lr2 = LogisticRegression(max_iter=1000).fit(s2.transform(Ftr), ytr)
    out["logistica sobre a janela crua"] = float(
        np.mean(lr2.predict(s2.transform(Fte)) == yte))
    return out


def experimento() -> None:
    print("=" * 78)
    print("ANCORA EXTERNA — de quanto e o buraco?")
    print("=" * 78)
    print()
    pub = publicado()
    print(f"  PUBLICADO  561 atributos dos autores + SVM linear   {pub:.3f}")
    print(f"             (o artigo do UCI reporta ~0,96)")
    if abs(pub - 0.96) > 0.05:
        print("             E0 FALHOU: nao reproduzi o artigo; pare aqui.")
        return
    print("             E0 ok.\n")

    riv = pareado()
    for k, v in sorted(riv.items(), key=lambda t: -t[1]):
        print(f"  PAREADO    {k:34s} {v:.3f}")
    melhor = max(riv.values())
    print(f"             melhor rival pareado: {melhor:.3f}\n")

    print(f"  NOSSO      descoberta congelada                    "
          f"{NOSSO:.3f} +- {NOSSO_SD:.3f}\n")

    print("=" * 78)
    d = NOSSO - melhor
    if d >= -0.05:
        print(f"E1 . o caminho da descoberta COMPETE: {d:+.3f} do melhor rival")
        print("     pareado, que treina com rotulo na mesma entrada crua.")
    else:
        print(f"E2 . o buraco e NOSSO: {d:+.3f} contra o melhor rival pareado,")
        print(f"     e {NOSSO - pub:+.3f} contra a linha publicada.")
        print("     Mesmo dado, mesmo split, mesma tarefa. Nao e o dado.")
