"""
ONDE ESTAO OS 14 PONTOS — 2x2 de codificacao x leitor.

[[ancora-externa-har]] mediu que a contagem de motivos rende 0,697 e o pipeline
completo entrega 0,560: 14 pontos perdidos DEPOIS da descoberta. Ha dois
suspeitos e eles sao separaveis:

  CODIFICACAO  contagem (guarda multiplicidade) x feixe HDC (sinal-da-soma,
               descarta multiplicidade e satura)
  LEITOR       logistica x medoide mais proximo entre 12 grupos

PRE-REGISTRO
  G0  PRE-CONDICAO: contagem+logistica = 0,697 e feixe+medoide = 0,560 +- 0,04.
  G1  uma das duas dimensoes responde por >= 2/3 da perda -> o conserto e
      pontual e vale fazer agora
  G2  perda repartida -> os dois precisam mudar, e o feixe HDC deixa de ser
      escolha barata para virar decisao de arquitetura

PREVISAO: G1 no LEITOR, 0,6. Mecanismo: 12 medoides para 6 atividades e um
classificador de capacidade quase nula comparado a uma logistica com 120
entradas. Mas o feixe tambem perde por construcao, entao nomeio o risco de G2.
"""
import statistics
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from novaphys.har import carrega
from novaphys.har_pessoas import (K_GRUPOS, N_TESTE, N_TREINO,
                                  PipelineCongelado, SEMENTE)
from novaphys.hdc_indice import Espaco
from novaphys.abstrai_motivo import analisa, agrupa_e_nomeia, vetores

N_SORT = 5

e = Espaco()
rng = np.random.default_rng(SEMENTE)
Xtr, ytr = carrega("train", N_TREINO)
Xte, yte = carrega("test", N_TESTE)
p = PipelineCongelado(e); p.ajusta(Xtr, ytr, rng, K_GRUPOS)
ped_tr, ped_te = analisa(p, Xtr), analisa(p, Xte)
lex = sorted(p.lex); ix = {w: i for i, w in enumerate(lex)}

def contagem(peds):
    M = np.zeros((len(peds), len(lex)), dtype=np.float32)
    for r, pe in enumerate(peds):
        for w in pe:
            if w in ix:
                M[r, ix[w]] += 1
    return M

def por_logistica(Ftr, Fte):
    s = StandardScaler().fit(Ftr)
    m = LogisticRegression(max_iter=3000).fit(s.transform(Ftr), ytr)
    return float(np.mean(m.predict(s.transform(Fte)) == yte))

def por_medoide(Vtr, Vte, semente):
    med, nome = agrupa_e_nomeia(Vtr, ytr, np.random.default_rng(semente),
                                K_GRUPOS)
    g = np.argmax(Vte.astype(np.float32) @ med.astype(np.float32).T, axis=1)
    return float(np.mean(np.array([nome[int(c)] for c in g]) == yte))

Ctr, Cte = contagem(ped_tr), contagem(ped_te)
# medoide precisa de vetores comparaveis por produto interno: normaliza a contagem
Ntr = Ctr/np.maximum(np.linalg.norm(Ctr, axis=1, keepdims=True), 1e-9)
Nte = Cte/np.maximum(np.linalg.norm(Cte, axis=1, keepdims=True), 1e-9)

print("="*78)
print("2x2 — onde estao os 14 pontos")
print("="*78)
print()
cl = por_logistica(Ctr, Cte)
cm = statistics.mean(por_medoide(Ntr, Nte, SEMENTE+i) for i in range(N_SORT))

fl, fm = [], []
for t in range(N_SORT):
    rot = (lambda k: lambda w: ["mot%d:%s" % (k, w)])(t)
    Vtr, Vte = vetores(e, ped_tr, rot), vetores(e, ped_te, rot)
    fl.append(por_logistica(Vtr.astype(np.float32), Vte.astype(np.float32)))
    fm.append(por_medoide(Vtr, Vte, SEMENTE))
fl_m, fm_m = statistics.mean(fl), statistics.mean(fm)

print("                     LEITOR: logistica    LEITOR: medoide (12)")
print("  CODIF: contagem         %.3f                %.3f" % (cl, cm))
print("  CODIF: feixe HDC        %.3f                %.3f" % (fl_m, fm_m))
print()
perda = cl - fm_m
p_leitor = cl - cm
p_codif = cl - fl_m
print("  perda total                     %.3f" % perda)
print("  atribuivel ao LEITOR (medoide)  %.3f   (%.0f%%)"
      % (p_leitor, 100*p_leitor/max(perda, 1e-9)))
print("  atribuivel a CODIFICACAO (feixe)%.3f   (%.0f%%)"
      % (p_codif, 100*p_codif/max(perda, 1e-9)))
print()
print("="*78)
if max(p_leitor, p_codif) >= (2/3)*perda:
    quem = "LEITOR (medoide)" if p_leitor > p_codif else "CODIFICACAO (feixe)"
    print("G1 . a perda e do %s: %.0f%% dela." % (quem, 100*max(p_leitor,p_codif)/perda))
else:
    print("G2 . perda repartida (leitor %.0f%%, codificacao %.0f%%): o feixe HDC"
          % (100*p_leitor/perda, 100*p_codif/perda))
    print("     deixa de ser escolha barata e vira decisao de arquitetura.")
