"""
A MARGEM PREVE A ACURACIA? — uma triagem de abstracao sem treinar nada.

13 bracos ja medidos (literal, canonico, ambos, 10 placebos). Para cada um,
calcula-se SO a margem (concordancia entre pessoas na mesma atividade menos
entre atividades diferentes) sobre o TREINO, sem tocar no teste e sem agrupar
nada. Se a margem ordena os bracos como a acuracia em pessoas novas ordena,
a margem e uma triagem barata.

PRE-REGISTRO
  T1  Spearman >= 0,60 entre margem e acuracia -> triagem valida
  T2  abaixo disso -> a margem explica ESTE caso e nao serve de regra
"""
import os, statistics
from collections import defaultdict, Counter
import numpy as np
from novaphys.har import RAIZ, carrega
from novaphys.har_pessoas import K_GRUPOS, N_TREINO, PipelineCongelado, SEMENTE
from novaphys.hdc_indice import Espaco
from novaphys.abstrai_motivo import analisa, canon

ACC = {}   # preenchido abaixo, vindo da rodada anterior
ACC["literal"]  = 0.537
ACC["canonico"] = 0.438
ACC["ambos"]    = 0.467
for s, a in enumerate([0.563, 0.470, 0.393, 0.514, 0.448,
                       0.447, 0.512, 0.497, 0.472, 0.477]):
    ACC["placebo%d" % s] = a

e = Espaco()
rng = np.random.default_rng(SEMENTE)
Xtr, ytr = carrega("train", N_TREINO)
suj = np.loadtxt(os.path.join(RAIZ, "train", "subject_train.txt"),
                 max_rows=N_TREINO, dtype=int)
p = PipelineCongelado(e); p.ajusta(Xtr, ytr, rng, K_GRUPOS)
ped = analisa(p, Xtr)
motivos = sorted({w for pe in ped for w in pe})

def cos(u, v):
    c = set(u) & set(v)
    num = sum(u[r]*v[r] for r in c)
    den = (sum(x*x for x in u.values())**0.5)*(sum(x*x for x in v.values())**0.5)
    return num/den if den > 0 else 0.0

def margem(rotulo):
    d = defaultdict(Counter)
    for pe, a, s in zip(ped, ytr.tolist(), suj.tolist()):
        for w in pe:
            for r in rotulo(w):
                d[(a, s)][r] += 1
    ch = sorted(d); dentro, fora = [], []
    for i in range(len(ch)):
        for j in range(i+1, len(ch)):
            (a1, s1), (a2, s2) = ch[i], ch[j]
            if s1 == s2:
                continue
            (dentro if a1 == a2 else fora).append(cos(d[ch[i]], d[ch[j]]))
    return statistics.mean(dentro) - statistics.mean(fora)

arms = [("literal",  lambda w: ["mot:"+w]),
        ("canonico", lambda w: ["pad:"+canon(w)]),
        ("ambos",    lambda w: ["mot:"+w, "pad:"+canon(w)])]
for s in range(10):
    r2 = np.random.default_rng(7000+s)
    classes = [canon(w) for w in motivos]
    perm = r2.permutation(len(motivos))
    mapa = {motivos[i]: classes[perm[i]] for i in range(len(motivos))}
    arms.append(("placebo%d" % s, (lambda m: lambda w: ["pad:"+m.get(w, w)])(mapa)))

print("="*78)
print("TRIAGEM — a margem preve a acuracia, sem treinar?")
print("="*78)
print()
print("  braco        margem (so treino)   acuracia (pessoas novas)")
print("  ----------   ------------------   ------------------------")
xs, ys = [], []
for nome, rot in arms:
    m = margem(rot); a = ACC[nome]
    xs.append(m); ys.append(a)
    print("  %-10s       %+.3f                %.3f" % (nome, m, a), flush=True)

def rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0]*len(v)
    for pos, i in enumerate(o):
        r[i] = pos
    return r
rx, ry = rank(xs), rank(ys)
mx, my = statistics.mean(rx), statistics.mean(ry)
num = sum((a-mx)*(b-my) for a, b in zip(rx, ry))
den = (sum((a-mx)**2 for a in rx)**0.5)*(sum((b-my)**2 for b in ry)**0.5)
rho = num/den
mxp, myp = statistics.mean(xs), statistics.mean(ys)
nump = sum((a-mxp)*(b-myp) for a, b in zip(xs, ys))
denp = (sum((a-mxp)**2 for a in xs)**0.5)*(sum((b-myp)**2 for b in ys)**0.5)
print()
print("  Spearman %.3f   Pearson %.3f   (n=%d)" % (rho, nump/denp, len(xs)))
print()
print("="*78)
if rho >= 0.60:
    print("T1 . A MARGEM E TRIAGEM: ordena os bracos sem treinar nenhum.")
else:
    print("T2 . a margem explica o caso e nao serve de regra (rho %.3f)." % rho)
