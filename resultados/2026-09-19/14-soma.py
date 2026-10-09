"""
O QUE A DESCOBERTA ACRESCENTA, SE ACRESCENTA ALGO.

`ancora_externa` mediu que 7 linhas de estatistica por canal (0,752) batem o
pipeline de descoberta (0,560) por 19 pontos na mesma entrada. Isso mede o
DESFECHO. A pergunta viva e outra: o que o MDL descobre — o lexico de motivos e
a segmentacao de Viterbi — carrega informacao que a estatistica trivial NAO da
de graca?

Testa-se somando, nao competindo. E usa-se a CONTAGEM DE MOTIVOS, nao o feixe
HDC, para tirar da conta o ruido de sorteio de atomos que
[[piso-de-ruido-e-pareamento]] mediu em +-0,040.

PRE-REGISTRO
  F0  PRE-CONDICAO: estatistica sozinha reproduz 0,752.
  F1  estatistica + motivos >= estatistica + 0,02
      -> a descoberta carrega informacao que ninguem ganha de graca, e o valor
         do caminho MDL nao depende de ele vencer sozinho
  F2  sem ganho -> a estrutura descoberta e REDUNDANTE com cinco minutos de
      estatistica, e isso e muito pior que perder por 19 pontos

PREVISAO: 0,45 para F1. A favor: os motivos sao de ORDEM e a estatistica so tem
diferenca de defasagem 1. Contra: WALK_UP sai em 0,292 no nosso pipeline, que e
justamente onde a ordem deveria pagar mais.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from novaphys.har import carrega
from novaphys.har_pessoas import (K_GRUPOS, N_TESTE, N_TREINO,
                                  PipelineCongelado, SEMENTE)
from novaphys.hdc_indice import Espaco
from novaphys.abstrai_motivo import analisa
from novaphys.ancora_externa import estatisticas

e = Espaco()
rng = np.random.default_rng(SEMENTE)
Xtr, ytr = carrega("train", N_TREINO)
Xte, yte = carrega("test", N_TESTE)
p = PipelineCongelado(e); p.ajusta(Xtr, ytr, rng, K_GRUPOS)
ped_tr, ped_te = analisa(p, Xtr), analisa(p, Xte)

lex = sorted(p.lex)
ix = {w: i for i, w in enumerate(lex)}
def contagem(peds):
    M = np.zeros((len(peds), len(lex)), dtype=np.float32)
    for r, pe in enumerate(peds):
        for w in pe:
            if w in ix:
                M[r, ix[w]] += 1
    return M

Ctr, Cte = contagem(ped_tr), contagem(ped_te)
Atr, Ate = estatisticas(Xtr), estatisticas(Xte)

def mede(Ftr, Fte):
    s = StandardScaler().fit(Ftr)
    m = LogisticRegression(max_iter=3000).fit(s.transform(Ftr), ytr)
    return float(np.mean(m.predict(s.transform(Fte)) == yte))

print("="*78)
print("A DESCOBERTA ACRESCENTA? — somando em vez de competir")
print("="*78)
print()
est = mede(Atr, Ate)
mot = mede(Ctr, Cte)
jun = mede(np.hstack([Atr, Ctr]), np.hstack([Ate, Cte]))
print("  estatistica por canal (63 col)      %.3f" % est)
print("  contagem de motivos  (%d col)      %.3f" % (len(lex), mot))
print("  AS DUAS JUNTAS                      %.3f" % jun)
print()
print("  ganho dos motivos sobre estatistica  %+.3f" % (jun - est))
print()
print("="*78)
if jun >= est + 0.02:
    print("F1 . A DESCOBERTA ACRESCENTA %+.3f sobre a estatistica trivial." % (jun-est))
    print("     Perder sozinho por 19 pontos nao significa nao achar nada:")
    print("     o lexico carrega informacao que ninguem ganha de graca.")
else:
    print("F2 . REDUNDANTE (%+.3f). A estrutura que o MDL descobre ja estava" % (jun-est))
    print("     em cinco minutos de estatistica por canal. Isso e pior que")
    print("     perder por 19 pontos, e e o resultado que importa.")
