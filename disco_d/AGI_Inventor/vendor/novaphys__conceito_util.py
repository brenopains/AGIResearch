"""
novaphys.conceito_util — julgar o conceito pelo que ele paga, nao pelo que comprime
=====================================================================================

DE ONDE VEM
-------------
`generalidade.py`, calibracao: no mundo magnetico (o conceito certo e a ENERGIA),
a invencao escolheu h - 0,04 r em 5/5 — a invariante que mais comprime as
trajetorias, e so APROXIMADA — e nao a energia, que e exata. Com E dado, a lei sai
10/10; com h, 0/10. Inventar o que mais comprime nao e inventar o que a lei precisa.

O CONSERTO
------------
Na crise, a observacao de trajetorias devolve ATE 3 invariantes DISTINTAS (as
melhores em bits, com |correlacao| < 0,95 entre si), nao so a primeira. Cada uma
e julgada A JUSANTE: com ela no caderno, a lei e redescoberta (olhos frescos) sobre
os dados de laboratorio guardados mais os da tarefa que surpreendeu, e mede-se o
comprimento de descricao total (erro + estrutura). Entra a que mais poupa — e so
se poupar em relacao a nao ter conceito novo. E o mesmo criterio que ja mostrou
que q pagava na lei seguinte (`inventa`), agora usado para ESCOLHER.

Tudo o mais e o laco de `autonomo.py` (surpresa contra o ruido, reanalise por
consenso, so observa de novo se a ultima rendeu). Os bracos rodam no MESMO codigo,
com a direcao n^ ligada nos dois universos (para comparar com o mesmo vocabulario).

PRE-REGISTRO, ANTES DE RODAR
------------------------------
Universo MAGNETICO -> oblato (semente 20261004), bracos fixo, consenso, util:
  U1  o braco util inventa a ENERGIA (corr >= 0,999) em >= 8/10
  U2  util - fixo >= +20 pontos na fase magnetica
Universo EINSTEIN -> oblato (semente 20261005), bracos consenso, util:
  U3  util >= consenso - 5 na fase einstein, e inventa h em >= 8/10 — o conserto
      nao quebra o caso original. Generalidade = o MESMO codigo nos dois mundos.
PREVISAO: U1 passa (a energia e distinta de h e a unica que faz a lei caber), U2
passa folgado (fixo inventa a variante de h e fica perto de 0%), U3 passa.

RESULTADO DA RODADA 1 (393 s):
  magnetico (0-19):   fixo 18%   consenso 19%   UTIL 100% (200/200)
                      util inventou a ENERGIA em 10/10
  einstein (0-19):    consenso 84%   UTIL 32%
                      util inventou h em 2/10 (o resto, uma quase-energia)
U1 PASSOU (10/10). U2 PASSOU (+82). U3 FALHOU (-51; h em 2/10).
DIAGNOSTICO: a primeira crise e na tarefa 0, e a utilidade e julgada com 20
medidas. No mundo einstein, com 20 corpos, ate o conceito certo so acerta a lei
metade das vezes; uma quase-energia (v^2/2 - 1/r - c/r^3, h^2 trocado pela media)
ganhou por acaso. Depois, a lei com o conceito errado ajusta dentro do ruido — a
crise nao volta, e o erro fica preso. E o ENTRINCHEIRAMENTO de novo, no nivel do
CONCEITO: comprometer-se cedo demais com pouca evidencia.

RODADA 2, PRE-REGISTRO — sementes 20261006 (magnetico) e 20261007 (einstein)
------------------------------------------------------------------------------
Braco UTIL2: os candidatos a conceito ficam numa SALA DE ESPERA; a escolha (um
conceito ativo, ou nenhum) e refeita pela mesma utilidade, sobre TODOS os dados
guardados, em toda crise E sempre que a evidencia DOBRA (1, 2, 4, 8, 16, 32 tarefas
guardadas). Se a escolha muda, troca o conceito e reanalisa por consenso. E a
licao da reanalise (+36) aplicada aos conceitos: nao fechar questao com pouco dado.
  W1  magnetico: util2 termina com a ENERGIA em >= 8/10 e util2 - fixo >= +20
  W2  einstein: util2 termina com h em >= 8/10 e util2 - consenso >= -5 (o mesmo
      limiar da U3)
PREVISAO: W1 passa. W2: o conceito final passa (com 160-320 medidas h vence), mas
a perda de acerto nas tarefas iniciais deve ficar perto de -8 — aposto que a parte
de acerto FALHA por pouco.

RESULTADO DA RODADA 2 (1202 s):
  magnetico (0-19): fixo 16%   UTIL2 99%  (+83)
  einstein  (0-19): consenso 88%   UTIL2 85%  (-4)
  conceito FINAL: energia no magnetico 4/10; h no einstein 0/10; trocas 3,2 / 2,5
W1 FALHOU e W2 FALHOU como registrados — pela parte do CONCEITO; a parte de
acerto passou nos dois (+83; -4 dentro de -5). Minha previsao de acerto errou
para BAIXO no einstein (previ -8).
ERRO DE DESENHO DO PORTAO (meu): o conceito foi medido no FIM da corrente, depois
da troca de lei; a ultima revisao (32 tarefas) junta dados dos dois mundos, e no
oblato a lei nao precisa de conceito. O conceito final nao diz nada sobre a fase
1. Diagnostico declarado (`--rodada 3`): reproduz as mesmas correntes e registra o
conceito em uso na tarefa 19 — nao reabre o portao.

DIAGNOSTICO (577 s, correntes 0-4 de cada universo, reproduzidas):
  fim da fase 1: magnetico -> ENERGIA (corr 1,0000) em 5/5; einstein -> h (corr
  1,0000) em 5/5. No fim da corrente, depois da troca, o conceito foi LARGADO na
  maioria ("nenhum") — no oblato ele nao paga. Plasticidade no nivel do conceito.

RODADA 4, CONFIRMACAO PRE-REGISTRADA — sementes 20261008 (magnetico) e 20261009
(einstein), 10 correntes, o portao no lugar certo (conceito em uso na tarefa 19)
---------------------------------------------------------------------------------
  W1' magnetico: util2 usa a ENERGIA (corr >= 0,999) na tarefa 19 em >= 8/10 e
      util2 - fixo >= +20 na fase magnetica
  W2' einstein: util2 usa h (corr >= 0,999) na tarefa 19 em >= 8/10 e
      util2 - consenso >= -5 na fase einstein
PREVISAO: as duas passam (o diagnostico deu 5/5 e 5/5; o acerto da rodada 2 deu
+83 e -4). O risco e o -5: -4 esta na borda.

RESULTADO DA RODADA 4 (1230 s):
  magnetico (0-19): fixo 13%   UTIL2 100% (200/200)   energia na tarefa 19: 10/10
  einstein  (0-19): consenso 86%   UTIL2 85% (-1)     h na tarefa 19:       10/10
W1' PASSOU (+87). W2' PASSOU (-1). Previsao certa.
O MESMO codigo, em dois mundos cujo conceito necessario e diferente, escolhe o
conceito certo em 10/10 e 10/10 — e larga o conceito quando o mundo muda e ele
deixa de pagar. Generalidade entre conceitos, dentro do dominio de orbitas.
"""

from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import itertools
import random
import time
from math import log2

import numpy as np
from scipy.linalg import eigh, LinAlgError

from novaphys import orbita as O
from novaphys import inventa as I
from novaphys import caderno as C
from novaphys import autonomo as A

SEMENTE_MAG = 20261004
SEMENTE_EIN = 20261005
K_CONC = 3


def liga():
    O.DIR_EXTRA = True


def distintas(Ttr, Tho, Sref, K=K_CONC):
    """Ate K invariantes aceitas (treino e fora da amostra), distintas entre si."""
    nomes, Ftr = I.alfabeto(Ttr)
    _, Fho = I.alfabeto(Tho)
    _, Fref = I.alfabeto(Sref)
    m = len(nomes)
    W, Tc = I._covs(Ftr)
    Ntr = Ftr.shape[0] * Ftr.shape[1]
    Nho = Fho.shape[0] * Fho.shape[1]
    cands = []
    for k in range(1, I.MAX_K + 1):
        for s in itertools.combinations(range(m), k):
            s = list(s)
            Ts = Tc[np.ix_(s, s)]
            ev = np.linalg.eigvalsh(Ts)
            if ev.min() <= 1e-9 * max(ev.max(), 1e-300):
                continue
            try:
                vals, vecs = eigh(W[np.ix_(s, s)], Ts)
            except LinAlgError:
                continue
            g = I.poupanca(max(vals[0], 1e-300), Ntr, Ftr.shape[0]) - I.estrutura(m, k, Ntr)
            if g > 0:
                cands.append((g, s, vecs[:, 0]))
    cands.sort(key=lambda c: -c[0])
    escolhidas = []
    for _g, s, w in cands:
        if I.poupanca(I._lam(Fho[..., s] @ w), Nho, Fho.shape[0]) <= 0:
            continue
        q = Fref[..., s] @ w
        if any(abs(np.corrcoef(q, e[2])[0, 1]) > 0.95 for e in escolhidas):
            continue
        escolhidas.append((s, w, q))
        if len(escolhidas) >= K:
            break
    return [(s, w) for s, w, _q in escolhidas]


def comprimento(conceitos, Sa, Da, base):
    """Bits da lei redescoberta (olhos frescos) com estes conceitos no caderno."""
    m2 = C.Memoria(0.7)
    m2.conceitos = conceitos
    ac, _ = C.descobre(Sa, Da, m2, base + 13)
    C._com_extras(m2)
    try:
        pred = O.preve(ac, Sa, False)
        P = len(O.caderno(Sa, False)[0])
    finally:
        O.EXTRAS.clear()
    sse = float(((Da - pred) ** 2).sum())
    est = sum(O.bits_termo(P, len(t[2]), Da.size) if t[0] == "lei"
              else O.bits_objeto(Da.size) for t in ac)
    return 0.5 * Da.size * log2(max(sse, 1e-300)) + est


def observa_util(mem, mundo, base, guardadas, S, D):
    rng = np.random.default_rng(base + 500)
    Ttr = I.trajetorias(mundo, rng, I.N_TRAJ)
    Tho = I.trajetorias(mundo, rng, I.N_TRAJ)
    Sh = O.estados(np.random.default_rng(base + 7), 500)
    novos = []
    for s1, w1 in distintas(Ttr, Tho, Sh):
        Q = I.grandeza(s1, w1, Ttr)
        if any(abs(np.corrcoef(Q(Sh), f(Sh))[0, 1]) > 0.99 for f in mem.conceitos.values()):
            continue
        novos.append(Q)
    if not novos:
        return False
    Sa = np.concatenate([g[1] for g in guardadas] + [S])
    Da = np.concatenate([g[2] for g in guardadas] + [D])
    L0 = comprimento(dict(mem.conceitos), Sa, Da, base)
    melhor = None
    nome = f"q{len(mem.conceitos)}"
    for Q in novos:
        c2 = dict(mem.conceitos)
        c2[nome] = Q
        L = comprimento(c2, Sa, Da, base)
        if melhor is None or L < melhor[0]:
            melhor = (L, Q)
    if melhor[0] < L0:
        mem.conceitos[nome] = melhor[1]
        return True
    return False


DOBRAS = (1, 2, 4, 8, 16, 32)


def _ids(mem, semente):
    """(corr com a energia, corr com h) de cada conceito em uso."""
    Sh = O.estados(np.random.default_rng(semente + 999), 500)
    En = 0.5 * (Sh[:, 2] ** 2 + Sh[:, 3] ** 2) - 1 / np.hypot(Sh[:, 0], Sh[:, 1])
    Hn = Sh[:, 0] * Sh[:, 3] - Sh[:, 1] * Sh[:, 2]
    return [(abs(np.corrcoef(f(Sh), En)[0, 1]), abs(np.corrcoef(f(Sh), Hn)[0, 1]))
            for f in mem.conceitos.values()]


def candidatos(pool, mundo, base):
    """Invariantes distintas das trajetorias, que ainda nao estao na sala de espera."""
    rng = np.random.default_rng(base + 500)
    Ttr = I.trajetorias(mundo, rng, I.N_TRAJ)
    Tho = I.trajetorias(mundo, rng, I.N_TRAJ)
    Sh = O.estados(np.random.default_rng(base + 7), 500)
    novos = []
    for s1, w1 in distintas(Ttr, Tho, Sh):
        Q = I.grandeza(s1, w1, Ttr)
        if any(abs(np.corrcoef(Q(Sh), f(Sh))[0, 1]) > 0.99 for f in pool + novos):
            continue
        novos.append(Q)
    return novos


def reavalia(pool, Sa, Da, base):
    """Qual conceito (ou nenhum) faz a lei mais curta sobre TODOS os dados."""
    opcoes = [None] + pool
    Ls = [comprimento({} if c is None else {"q0": c}, Sa, Da, base) for c in opcoes]
    return opcoes[int(np.argmin(Ls))]


def corrente(args):
    braco, s, semente, mundos = args
    pool, trocas = [], 0         # util2: sala de espera de conceitos
    fase1 = []                   # conceito em uso no FIM da fase 1 (tarefa 19)
    C.MUNDOS = mundos
    mem = C.Memoria(0.7)
    usadas, guardadas, primeiro, crises = [], [], {}, []
    lei = []
    limiar_chi = 1 + 3 * np.sqrt(2 / (2 * C.N_LAB))
    pode_observar = True
    for k in range(A.N_PASSOS):
        mundo = C.mundo_de(k)
        base = semente + 1000 * s + k
        rng = np.random.default_rng(base)
        S = O.estados(rng, C.N_LAB)
        D = O.mede(mundo, S, rng)
        if braco == "fixo" and k in A.FIXO and len(usadas) < A.ORCAMENTO:
            usadas.append(k)
            if A.observa(mem, mundo, base) and guardadas:
                lei = A.reanalisa(mem, guardadas, base)
        if braco in ("consenso", "util", "util2"):
            if A.surpresa(lei, S, D, mem) > limiar_chi:
                crises.append(k)
                if pode_observar and len(usadas) < A.ORCAMENTO:
                    usadas.append(k)
                    if braco == "consenso":
                        pode_observar = A.observa(mem, mundo, base)
                    elif braco == "util":
                        pode_observar = observa_util(mem, mundo, base, guardadas, S, D)
                    else:
                        novos = candidatos(pool, mundo, base)
                        pool += novos
                        pode_observar = bool(novos)
                if braco == "util2" and pool:
                    Sa = np.concatenate([g[1] for g in guardadas] + [S])
                    Da = np.concatenate([g[2] for g in guardadas] + [D])
                    esc = reavalia(pool, Sa, Da, base)
                    if esc is not mem.conceitos.get("q0"):
                        trocas += 1
                        mem.conceitos = {} if esc is None else {"q0": esc}
                lei = A.reanalisa_consenso(mem, guardadas, S, D, base, limiar_chi)
        ac, chaves = C.descobre(S, D, mem, base + 3)
        lei = ac
        primeiro[k] = C.acerta(ac, mem, mundo)
        guardadas.append((k, S, D, base + 3))
        mem.envelhece()
        for ch in chaves:
            mem.usa(ch)
        if braco == "util2" and pool and len(guardadas) in DOBRAS:
            # a evidencia dobrou: rever a escolha de conceito com TODOS os dados
            Sa = np.concatenate([g[1] for g in guardadas])
            Da = np.concatenate([g[2] for g in guardadas])
            esc = reavalia(pool, Sa, Da, base + 1)
            if esc is not mem.conceitos.get("q0"):
                trocas += 1
                mem.conceitos = {} if esc is None else {"q0": esc}
                lei = A.reanalisa_consenso(mem, guardadas[:-1], S, D, base, limiar_chi)
        if k == 19:
            fase1 = _ids(mem, semente)
    Sh = O.estados(np.random.default_rng(semente + 999), 500)
    En = 0.5 * (Sh[:, 2] ** 2 + Sh[:, 3] ** 2) - 1 / np.hypot(Sh[:, 0], Sh[:, 1])
    Hn = Sh[:, 0] * Sh[:, 3] - Sh[:, 1] * Sh[:, 2]
    ids = [(abs(np.corrcoef(f(Sh), En)[0, 1]), abs(np.corrcoef(f(Sh), Hn)[0, 1]))
           for f in mem.conceitos.values()]
    return braco, s, primeiro, usadas, crises, ids, mundos[0], trocas, fase1


def relatorio(res, universo, bracos, n):
    rs_u = [r for r in res if r[6] == universo]
    F1 = f"{universo} (0-19)"
    tab = {}
    print(f"\n  --- universo {universo} -> oblato ---")
    print("  acerto na primeira tentativa" + "".join(f"{b:>14s}" for b in bracos))
    for nome, ks in ((F1, range(0, 20)), ("logo apos a troca (20-24)", range(20, 25)),
                     ("oblato (25-39)", range(25, 40))):
        linha = ""
        for b in bracos:
            v = [r[2][k] for r in rs_u if r[0] == b for k in ks]
            tab[(nome, b)] = sum(v) / len(v)
            linha += f"{sum(v):6d}/{len(v):<3d}{100*sum(v)/len(v):3.0f}%"
        print(f"  {nome:28s}{linha}")
    ident = {}
    for b in bracos:
        rr = [r for r in rs_u if r[0] == b]
        e = sum(1 for r in rr if r[5] and r[5][0][0] >= 0.999)
        h = sum(1 for r in rr if r[5] and r[5][0][1] >= 0.999)
        ident[b] = (e, h)
        print(f"    {b:9s} primeiro conceito: energia {e}/{n}  momento {h}/{n}   "
              f"observacoes {np.mean([len(r[3]) for r in rr]):.1f}  ex. {[r[3] for r in rr][:4]}")
    return tab, ident, F1


def diagnostico(n, nucleos):
    """Reproduz as correntes 0..n-1 da rodada 2 (mesmas sementes, deterministico) e
    mostra QUAL conceito estava em uso no fim da fase 1 — nao e um novo teste."""
    from multiprocessing import Pool
    trab = ([("util2", s, 20261006, ("magnetico", "oblato")) for s in range(n)]
            + [("util2", s, 20261007, ("einstein", "oblato")) for s in range(n)])
    t0 = time.time()
    with Pool(nucleos, initializer=liga) as pool:
        res = pool.map(corrente, trab, chunksize=1)
    print(f"DIAGNOSTICO da rodada 2 ({time.time()-t0:.0f}s) — conceito em uso "
          f"(corr com E, corr com h)")
    for r in res:
        ac1 = sum(r[2][k] for k in range(20))
        f1 = "  ".join(f"({e:.4f}, {h:.4f})" for e, h in r[8]) or "nenhum"
        ff = "  ".join(f"({e:.4f}, {h:.4f})" for e, h in r[5]) or "nenhum"
        print(f"  {r[6]:9s} s{r[1]}  acerto fase 1 {ac1:2d}/20  trocas {r[7]}  "
              f"fim da fase 1: {f1:22s}  fim: {ff}")


def rodada2(n, nucleos, sem_mag=20261006, sem_ein=20261007, fase1=False):
    from multiprocessing import Pool
    trab = ([(b, s, sem_mag, ("magnetico", "oblato")) for s in range(n)
             for b in ("fixo", "util2")]
            + [(b, s, sem_ein, ("einstein", "oblato")) for s in range(n)
               for b in ("consenso", "util2")])
    print("=" * 76)
    print("CONCEITO UTIL 2 — sala de espera; rever a escolha quando a evidencia dobra")
    print("=" * 76)
    t0 = time.time()
    with Pool(nucleos, initializer=liga) as pool:
        res = pool.map(corrente, trab, chunksize=1)
    print(f"\n  ({time.time()-t0:.0f}s)")
    tm, _im, M = relatorio(res, "magnetico", ("fixo", "util2"), n)
    te, _ie, E = relatorio(res, "einstein", ("consenso", "util2"), n)

    def final(univ, b, j):
        rr = [r for r in res if r[6] == univ and r[0] == b]
        idx = 8 if fase1 else 5      # rodada 4: conceito em uso na tarefa 19
        return sum(1 for r in rr if r[idx] and r[idx][-1][j] >= 0.999), \
            np.mean([r[7] for r in rr])
    fe, tr_m = final("magnetico", "util2", 0)
    fh, tr_e = final("einstein", "util2", 1)
    print(f"\n  conceito FINAL do util2: energia no magnetico {fe}/{n} (trocas {tr_m:.1f});"
          f" h no einstein {fh}/{n} (trocas {tr_e:.1f})")
    print("")
    print("=" * 76)
    w1 = tm[(M, "util2")] - tm[(M, "fixo")]
    print(f"W1 {'.' if fe >= 8 and w1 >= 0.20 else 'FALHOU .'} magnetico: energia final "
          f"{fe}/{n}, util2 - fixo = {100*w1:+.0f} (limiar +20)")
    w2 = te[(E, "util2")] - te[(E, "consenso")]
    print(f"W2 {'.' if fh >= 8 and w2 >= -0.05 else 'FALHOU .'} einstein: h final {fh}/{n}, "
          f"util2 - consenso = {100*w2:+.0f} (limiar >= -5)")


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--correntes", type=int, default=10)
    ap.add_argument("--nucleos", type=int, default=3)
    ap.add_argument("--rodada", type=int, default=1)
    a = ap.parse_args()
    n = a.correntes
    liga()
    if a.rodada == 2:
        rodada2(n, a.nucleos)
        return
    if a.rodada == 3:
        diagnostico(n, a.nucleos)
        return
    if a.rodada == 4:
        print("RODADA 4 — confirmacao, conceito medido na tarefa 19")
        rodada2(n, a.nucleos, 20261008, 20261009, fase1=True)
        return
    trab = ([(b, s, SEMENTE_MAG, ("magnetico", "oblato")) for s in range(n)
             for b in ("fixo", "consenso", "util")]
            + [(b, s, SEMENTE_EIN, ("einstein", "oblato")) for s in range(n)
               for b in ("consenso", "util")])
    print("=" * 76)
    print("CONCEITO UTIL — julgar o conceito pelo que ele paga na lei em crise")
    print("=" * 76)
    t0 = time.time()
    with Pool(a.nucleos, initializer=liga) as pool:
        res = pool.map(corrente, trab, chunksize=1)
    print(f"\n  ({time.time()-t0:.0f}s)")
    tm, im, M = relatorio(res, "magnetico", ("fixo", "consenso", "util"), n)
    te, ie, E = relatorio(res, "einstein", ("consenso", "util"), n)
    print("")
    print("=" * 76)
    print(f"U1 {'.' if im['util'][0] >= 8 else 'FALHOU .'} util inventa a energia no mundo "
          f"magnetico: {im['util'][0]}/{n}")
    u2 = tm[(M, "util")] - tm[(M, "fixo")]
    print(f"U2 {'.' if u2 >= 0.20 else 'FALHOU .'} util - fixo (magnetico) = {100*u2:+.0f} (limiar +20)")
    u3 = te[(E, "util")] - te[(E, "consenso")]
    ok3 = u3 >= -0.05 and ie["util"][1] >= 8
    print(f"U3 {'.' if ok3 else 'FALHOU .'} einstein: util - consenso = {100*u3:+.0f} (>= -5), "
          f"util inventa h em {ie['util'][1]}/{n} (>= 8)")


if __name__ == "__main__":
    main()
