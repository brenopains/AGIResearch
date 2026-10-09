"""
novaphys.generalidade — o mesmo laco, outro mundo, OUTRO conceito
===================================================================

O GAP
-------
O laco autonomo fechou (`autonomo.py`, rodada 3: 83% contra 68% do meu
calendario). Mas em TODO experimento ate aqui o conceito que faltava era o mesmo:
o momento angular h. Um laco afinado num caso so pode estar decorando o caso. A
pergunta do usuario sobre generalizacao, na forma mais direta: o MESMO codigo, sem
mudar uma linha do laco, acha o conceito CERTO quando o conceito certo e outro?

O MUNDO MAGNETICO
-------------------
Newton mais uma forca PERPENDICULAR a velocidade (como a magnetica: nao realiza
trabalho), com intensidade proporcional a energia orbital:
    a = -r^/r^2 + A * E * n^,   E = v^2/2 - 1/r,   n^ = v girada de 90 graus
Fisicamente consistente: a forca extra nao faz trabalho, entao E se conserva
EXATAMENTE; e nao e central, entao h NAO se conserva. O conceito que faz a lei
caber e E: com ele a lei e UM termo (c q n^); sem ele sao dois (v^2 n^ e r^-1 n^).

O que eu mudo, dito antes: a gramatica ganha uma quarta direcao, n^ (v girada) —
generica, nao especifica deste mundo, mas ausente antes. E o universo passa a ser
magnetico ate a tarefa 20 e oblato depois. O laco (surpresa, invencao, consenso)
e o de `autonomo.py`, sem alteracao.

PROTOCOLO EM DUAS ETAPAS
--------------------------
Etapa 1 (calibracao declarada, semente 20261002): (a) a invencao acha E, e nao h,
nas trajetorias deste mundo? (b) com 20 e 40 corpos por tarefa, quanto acerta a
lei sem conceito, com E dado e com h dado (o conceito ERRADO)?
Etapa 2 (medida, semente 20261003): pre-registro escrito depois da calibracao.

CALIBRACAO (feita, semente 20261002)
--------------------------------------
(a) A invencao escolheu o conceito ERRADO em 5/5: q = -x vy + y vx - 0,04 r
    (corr com h 0,9997, com E 0,11), lambda ~1,5e-5. Neste mundo h NAO se conserva,
    mas h - c r se conserva dentro da precisao da medida no tempo observado — uma
    invariante APROXIMADA. A energia, conservada EXATAMENTE, perdeu. Por que: a
    escolha e por compressao, e h varia muito entre orbitas (ate de sinal, nas
    retrogradas), o que faz o lambda dele minusculo.
(b) Acerto da lei:      sem conceito   com E dado   com h dado
       N = 20              0/10          10/10         0/10
       N = 40              1/10          10/10         1/10
    O conceito certo decide tudo; o errado vale zero.
O GAP: a invencao escolhe o conceito que MAIS COMPRIME as observacoes, nao o que
a LEI PRECISA. (E o mesmo defeito que o `inventa` ja tinha mostrado: nao distingue
conservado de lentamente variavel.)

PRE-REGISTRO DA ETAPA 2 — semente 20261003, N = 20, 10 correntes, laco SEM MUDANCA
-----------------------------------------------------------------------------------
  G1  GENERALIZA: consenso inventa a energia em >= 8/10 E consenso - fixo >= +10
      na fase magnetica
PREVISAO: G1 FALHA — o consenso inventa a variante de h em ~10/10, a crise
persiste, a segunda observacao devolve a mesma invariante (duplicada), e ele
para de observar; consenso ~ nunca ~ 0% na fase magnetica. O conserto (rodada
seguinte): julgar o conceito pela UTILIDADE — quanto ele poupa em bits na lei que
esta em crise — e nao so pela compressao das trajetorias.

RESULTADO DA ETAPA 2 (269 s):
                      nunca   fixo   consenso
  magnetico (0-19)      1%     16%      12%
  logo apos a troca    84%     86%      88%
  primeiro conceito: energia 0/10 nos dois bracos; momento angular 10/10
  crises do consenso: [0,1,2,3,5,6,12,16] [0,1,3,4,5,9,11,12] ... — a crise NAO
  para: o detector sabe que a teoria nao da conta, mas a resposta nao tem remedio
G1 FALHOU, como previsto. O laco nao generaliza como estava: a surpresa e a
reanalise sao gerais; a INVENCAO nao — escolhe o que mais comprime.
"""

from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import time

import numpy as np

from novaphys import orbita as O
from novaphys import inventa as I
from novaphys import caderno as C
from novaphys import autonomo as A

SEMENTE_CAL = 20261002
SEMENTE = 20261003
BRACOS = ("nunca", "fixo", "consenso")


def liga():
    O.DIR_EXTRA = True
    C.MUNDOS = ("magnetico", "oblato")


def energia(S):
    return 0.5 * (S[..., 2] ** 2 + S[..., 3] ** 2) - 1 / np.hypot(S[..., 0], S[..., 1])


def momento(S):
    return S[..., 0] * S[..., 3] - S[..., 1] * S[..., 2]


def calibra(n_sem):
    liga()
    print("CALIBRACAO (semente 20261002)")
    print("  (a) a invencao nas trajetorias do mundo magnetico:")
    Sh = O.estados(np.random.default_rng(SEMENTE_CAL + 9), 500)
    for k in range(5):
        rng = np.random.default_rng(SEMENTE_CAL + 100 * k)
        Ttr = I.trajetorias("magnetico", rng, I.N_TRAJ)
        Tho = I.trajetorias("magnetico", rng, I.N_TRAJ)
        ac, top, nomes = I.inventa(Ttr, Tho)
        _g, s, w, lam, _lho, _gho = top[0]
        Q = I.grandeza(s, w, Ttr)
        cE = abs(np.corrcoef(Q(Sh), energia(Sh))[0, 1])
        cH = abs(np.corrcoef(Q(Sh), momento(Sh))[0, 1])
        print(f"    s{k} {'INVENTOU' if ac else 'nada':8s} {I.formula(s, w, nomes):40s} "
              f"lambda {lam:.1e}  corr E {cE:.4f}  corr h {cH:.4f}")
    print("  (b) acerto da lei (extrapola), sem conceito / com E dado / com h dado:")
    for n in (20, 40):
        ok = {"sem": 0, "com E": 0, "com h": 0}
        for k in range(n_sem):
            rng = np.random.default_rng(SEMENTE_CAL + 1000 * k + n)
            S = O.estados(rng, n)
            D = O.mede("magnetico", S, rng)
            for rot, conc in (("sem", None), ("com E", energia), ("com h", momento)):
                mem = None
                if conc is not None:
                    mem = C.Memoria(0.7)
                    mem.conceitos = {"q0": conc}
                ac, _ = C.descobre(S, D, mem, SEMENTE_CAL + k)
                ok[rot] += C.acerta(ac, mem, "magnetico")
        print(f"    N={n:3d}   " + "   ".join(f"{r} {v:2d}/{n_sem}" for r, v in ok.items()))


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibra", action="store_true")
    ap.add_argument("--correntes", type=int, default=10)
    ap.add_argument("--nucleos", type=int, default=3)
    ap.add_argument("--semente", type=int, default=SEMENTE)
    a = ap.parse_args()
    liga()
    if a.calibra:
        calibra(10)
        return
    n = a.correntes
    print("=" * 76)
    print("GENERALIDADE — o mesmo laco num mundo onde o conceito certo e a ENERGIA")
    print("=" * 76)
    t0 = time.time()
    with Pool(a.nucleos, initializer=liga) as pool:
        res = pool.map(A.corrente, [(b, s, a.semente) for s in range(n) for b in BRACOS],
                       chunksize=1)
    print(f"\n  semente {a.semente}   ({time.time()-t0:.0f}s)\n")
    fases = {"magnetico (0-19)": range(0, 20), "logo apos a troca (20-24)": range(20, 25),
             "oblato (25-39)": range(25, 40)}
    tab = {}
    print("  acerto na primeira tentativa" + "".join(f"{b:>14s}" for b in BRACOS))
    for nome, ks in fases.items():
        linha = ""
        for b in BRACOS:
            v = [r[2][k] for r in res if r[0] == b for k in ks]
            tab[(nome, b)] = sum(v) / len(v)
            linha += f"{sum(v):6d}/{len(v):<3d}{100*sum(v)/len(v):3.0f}%"
        print(f"  {nome:28s}{linha}")
    print("\n  QUE conceito cada braco inventou (primeiro conceito: corr com E / com h):")
    ident = {}
    for b in ("fixo", "consenso"):
        rs = [r for r in res if r[0] == b]
        eh_E = sum(1 for r in rs if r[6] and r[6][0][0] >= 0.999)
        eh_h = sum(1 for r in rs if r[6] and r[6][0][1] >= 0.999)
        ident[b] = eh_E
        print(f"    {b:9s} energia {eh_E}/{n}   momento angular {eh_h}/{n}   "
              f"observacoes {np.mean([len(r[3]) for r in rs]):.1f}")
    co = [r for r in res if r[0] == "consenso"]
    print("  consenso observou em: " + "  ".join(str(r[3]) for r in co[:6]))
    print("  crises do consenso:   " + "  ".join(str(r[5][:8]) for r in co[:6]))
    M = "magnetico (0-19)"
    T = "logo apos a troca (20-24)"
    print("")
    print("=" * 76)
    g1 = tab[(M, "consenso")] - tab[(M, "fixo")]
    g2 = tab[(M, "consenso")] - tab[(M, "nunca")]
    print(f"  consenso - fixo (magnetico) = {100*g1:+.0f}   consenso - nunca = {100*g2:+.0f}   "
          f"consenso - fixo (troca) = {100*(tab[(T, 'consenso')] - tab[(T, 'fixo')]):+.0f}")
    print(f"  consenso inventou a ENERGIA em {ident['consenso']}/{n}")


if __name__ == "__main__":
    main()
