"""
O denominador que faltou no A0b.

A0b mediu concordancia entre pessoas na MESMA atividade e viu +0,252. Nao mediu
concordancia entre atividades DIFERENTES. Se as duas subiram junto, a abstracao
borrou em vez de abstrair, e a diferenca (within - between) e a unica coisa que
um classificador consegue usar.
"""
import os, statistics
from collections import defaultdict, Counter
import numpy as np
from novaphys.har import RAIZ, NOMES, carrega
from novaphys.har_pessoas import K_GRUPOS, N_TREINO, PipelineCongelado, SEMENTE
from novaphys.hdc_indice import Espaco
from novaphys.abstrai_motivo import analisa, canon

e = Espaco()
rng = np.random.default_rng(SEMENTE)
Xtr, ytr = carrega("train", N_TREINO)
suj = np.loadtxt(os.path.join(RAIZ, "train", "subject_train.txt"),
                 max_rows=N_TREINO, dtype=int)
p = PipelineCongelado(e); p.ajusta(Xtr, ytr, rng, K_GRUPOS)
ped = analisa(p, Xtr)

def perfis(rotulo):
    d = defaultdict(Counter)
    for pe, a, s in zip(ped, ytr.tolist(), suj.tolist()):
        for w in pe:
            for r in rotulo(w):
                d[(a, s)][r] += 1
    return d

def cos(u, v):
    c = set(u) & set(v)
    num = sum(u[r]*v[r] for r in c)
    den = (sum(x*x for x in u.values())**0.5)*(sum(x*x for x in v.values())**0.5)
    return num/den if den > 0 else 0.0

print("="*78)
print("O DENOMINADOR QUE FALTOU — a abstracao discrimina ou so borra?")
print("="*78)
print()
print("  rotulo      mesma atividade   atividades DIFERENTES   margem")
print("  ---------   ---------------   ---------------------   ------")
for nome, rot in (("literal ", lambda w: ["mot:"+w]),
                  ("canonico", lambda w: ["pad:"+canon(w)])):
    d = perfis(rot)
    ch = sorted(d)
    dentro, fora = [], []
    for i in range(len(ch)):
        for j in range(i+1, len(ch)):
            (a1, s1), (a2, s2) = ch[i], ch[j]
            if s1 == s2:
                continue                      # so pares de PESSOAS diferentes
            (dentro if a1 == a2 else fora).append(cos(d[ch[i]], d[ch[j]]))
    md, mf = statistics.mean(dentro), statistics.mean(fora)
    print("  %s    %.3f (n=%d)     %.3f (n=%d)        %+.3f"
          % (nome, md, len(dentro), mf, len(fora), md-mf))
