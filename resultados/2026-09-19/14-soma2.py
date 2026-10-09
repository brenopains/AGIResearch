"""
F2b — SEGUNDA OLHADA, e declarada como tal.

F2 foi pre-registrado sobre a CONTAGEM de motivos e deu -0,012 (redundante). O
2x2 seguinte mostrou que a contagem nao e a melhor codificacao do que foi
descoberto: o feixe HDC rende mais (0,721 x 0,697 com logistica). Refazer o teste
com a codificacao melhor e legitimo e e SEGUNDA OLHADA — vale menos como
evidencia, e so conta se o efeito for claro.

  F2b  feixe + estatistica >= estatistica + 0,02, em 5/5 sorteios de atomo
       -> a descoberta acrescenta quando lida pela codificacao certa
  F2c  sem ganho -> redundancia confirmada nas DUAS codificacoes, e ai nao ha
       mais para onde apelar

PREVISAO: 0,35. Menor que os 0,45 de F2 — porque F2 ja saiu negativo e a unica
coisa que mudou foi trocar 0,697 por 0,721 numa comparacao contra 0,752.
"""
import statistics
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from novaphys.har import carrega
from novaphys.har_pessoas import (K_GRUPOS, N_TESTE, N_TREINO,
                                  PipelineCongelado, SEMENTE)
from novaphys.hdc_indice import Espaco
from novaphys.abstrai_motivo import analisa, vetores
from novaphys.ancora_externa import estatisticas

e = Espaco()
rng = np.random.default_rng(SEMENTE)
Xtr, ytr = carrega("train", N_TREINO)
Xte, yte = carrega("test", N_TESTE)
p = PipelineCongelado(e); p.ajusta(Xtr, ytr, rng, K_GRUPOS)
ped_tr, ped_te = analisa(p, Xtr), analisa(p, Xte)
Atr, Ate = estatisticas(Xtr), estatisticas(Xte)

def mede(Ftr, Fte):
    s = StandardScaler().fit(Ftr)
    m = LogisticRegression(max_iter=3000).fit(s.transform(Ftr), ytr)
    return float(np.mean(m.predict(s.transform(Fte)) == yte))

est = mede(Atr, Ate)
print("="*78)
print("F2b — SEGUNDA OLHADA: o feixe acrescenta sobre a estatistica?")
print("="*78)
print()
print("  estatistica sozinha                 %.3f" % est)
juns = []
for t in range(5):
    rot = (lambda k: lambda w: ["mot%d:%s" % (k, w)])(t)
    Vtr = vetores(e, ped_tr, rot).astype(np.float32)
    Vte = vetores(e, ped_te, rot).astype(np.float32)
    j = mede(np.hstack([Atr, Vtr]), np.hstack([Ate, Vte]))
    juns.append(j)
    print("  estatistica + feixe (sorteio %d)     %.3f   (%+.3f)" % (t, j, j-est),
          flush=True)
m, sd = statistics.mean(juns), statistics.pstdev(juns)
pos = sum(1 for j in juns if j >= est + 0.02)
print()
print("  media %.3f +- %.3f   ganho %+.3f   passa em %d/5" % (m, sd, m-est, pos))
print()
print("="*78)
if m >= est + 0.02 and pos == 5:
    print("F2b . acrescenta %+.3f com a codificacao certa (segunda olhada)." % (m-est))
else:
    print("F2c . REDUNDANCIA CONFIRMADA nas duas codificacoes (%+.3f)." % (m-est))
    print("      A estrutura que o MDL descobre neste dominio nao carrega nada")
    print("      que sete linhas de estatistica por canal ja nao deem.")
