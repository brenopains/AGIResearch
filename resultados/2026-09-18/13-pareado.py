"""
O +0,075 DE GANHO TEMPORAL SOBREVIVE AO PISO DE RUIDO?

`har_pessoas` reporta 0,545 real contra 0,470 +- 0,003 no nulo de tempo
embaralhado, e [[motivo-e-pessoal]] gravou esse +0,075 como "o ganho temporal
desaba mas nao zera". Aquele +-0,003 varia SO o embaralhamento, com o sorteio de
atomos FIXO. O piso de ruido do sorteio e +-0,040 — treze vezes maior.

Mas a comparacao e PAREADA: o mesmo sorteio de atomos serve os dois bracos, e o
ruido comum cancela. Entao a pergunta certa nao e se 0,545 e estavel (nao e), e
sim se a DIFERENCA e. Isso e o que este script mede: por sorteio, real menos
embaralhado.

PRE-REGISTRO
  P1  diferenca media >= 2 desvios dela mesma, e positiva em >= 8 dos 10
      -> o ganho temporal e real e [[motivo-e-pessoal]] esta certo
  P2  diferenca some no ruido -> o caminho da ordem nao entrega NADA entre
      pessoas, e a memoria precisa de correcao
"""
import statistics
import numpy as np
from novaphys.har import carrega
from novaphys.har_pessoas import (K_GRUPOS, N_TESTE, N_TREINO,
                                  PipelineCongelado, SEMENTE)
from novaphys.hdc_indice import Espaco
from novaphys.abstrai_motivo import analisa, braco

e = Espaco()
rng = np.random.default_rng(SEMENTE)
Xtr, ytr = carrega("train", N_TREINO)
Xte, yte = carrega("test", N_TESTE)
p = PipelineCongelado(e); p.ajusta(Xtr, ytr, rng, K_GRUPOS)
ped_tr = analisa(p, Xtr)
ped_te = analisa(p, Xte)

emb = []
for s in range(2):
    r = np.random.default_rng(2000+s)
    Xs = np.stack([A[r.permutation(A.shape[0])] for A in Xte])
    emb.append(analisa(p, Xs))

print("="*78)
print("PAREADO — o ganho temporal contra o piso de ruido de atomos")
print("="*78)
print()
print("  sorteio   real    embaralhado   diferenca")
print("  -------   -----   -----------   ---------")
difs, reais = [], []
for s in range(10):
    rot = (lambda t: lambda w: ["mot%d:%s" % (t, w)])(s)
    a = braco(e, ped_tr, ped_te, ytr, yte, rot, K_GRUPOS)
    ns = [braco(e, ped_tr, pe, ytr, yte, rot, K_GRUPOS) for pe in emb]
    n = statistics.mean(ns)
    difs.append(a-n); reais.append(a)
    print("  %5d     %.3f      %.3f        %+.3f" % (s, a, n, a-n), flush=True)

md, sdd = statistics.mean(difs), statistics.pstdev(difs)
pos = sum(1 for d in difs if d > 0)
print()
print("  real:      %.3f +- %.3f" % (statistics.mean(reais),
                                     statistics.pstdev(reais)))
print("  DIFERENCA: %+.3f +- %.3f   positiva em %d/10" % (md, sdd, pos))
print("  (har_pessoas anunciou +0,075 de um unico sorteio)")
print()
print("="*78)
if md >= 2*max(sdd, 1e-9) and pos >= 8:
    print("P1 . o ganho temporal SOBREVIVE ao piso: %+.3f +- %.3f, %d/10"
          % (md, sdd, pos))
    print("     pareado cancela o ruido de atomos e motivo-e-pessoal esta certo.")
else:
    print("P2 . o ganho temporal NAO SOBREVIVE: %+.3f +- %.3f, positiva em %d/10."
          % (md, sdd, pos))
    print("     O caminho da ORDEM nao entrega nada entre pessoas. A memoria")
    print("     motivo-e-pessoal precisa de correcao: nao e 'o ganho desaba de")
    print("     +0,210 para +0,075', e 'o ganho desaparece'.")
