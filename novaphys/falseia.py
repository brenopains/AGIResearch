"""
novaphys.falseia — saber que esta errado: ir medir onde as proprias hipoteses discordam
==========================================================================================

O GAP, VISTO EM TRES LUGARES
------------------------------
  orbitas      epiciclos passaram no R2 de 0,99 e desabaram fora da faixa medida
  caderno      com 5x mais dado, a lei velha + uma correcao ainda desviaram a
               analise conjunta em metade das vezes
  critica ARC  o nucleo so percebeu 12,5% das proprias respostas confiantes erradas
A causa comum: o agente so valida DENTRO do que ja viu — e la epiciclo e lei sao
iguais. Toda a validacao fora da amostra do projeto e fora da amostra, mas DENTRO
da faixa.

O MECANISMO: FALSIFICACAO ATIVA
---------------------------------
Duas intuicoes antigas do usuario juntas — simular mentalmente e escolher o
experimento. O agente forma um COMITE de hipoteses (a mesma descoberta em 4
reamostragens do que ja mediu), SIMULA o que cada uma preve num leque de estados
que ele NUNCA mediu, e vai medir onde elas mais discordam. E o que Le Verrier e
Einstein fizeram: procurar o lugar onde a teoria pode quebrar.

Na lente, escolher o experimento nao replicou (18 x 15 de 30). Aqui a teoria diz
que TEM de pagar: sem h, a lei de Einstein e (v^2 - vr^2)/r^2, e os expoentes
rivais do termo fraco (r^-1, r^-2, r^-3) so se separam longe e em orbitas radiais.

O DESENHO — orcamento casado de 400 medidas, mundo einstein, caderno SEM h
----------------------------------------------------------------------------
Todas comecam com as MESMAS 280 medidas sorteadas na faixa de sempre (r 0,7-3).
As outras 120:
    PASSIVO   sorteadas na mesma faixa (e a condicao da orbita3: lei exata 7/10)
    LARGO     sorteadas numa faixa larga (r 0,7-6, mais radial) — EU decido olhar
              longe; controle de "so amostrar mais longe"
    ATIVO     escolhidas PELO AGENTE, entre 2000 candidatas da faixa larga: as 120
              de maior discordancia do comite
Medida principal: a lei EXATA (os dois termos, coeficientes a 20%). Secundaria: R2
em r de 6 a 12 — mais longe do que QUALQUER braco mediu.

METACOGNICAO (a outra metade de "saber que esta errado")
----------------------------------------------------------
Depois do ajuste final, um comite novo sobre TODOS os dados; a DUVIDA e a
discordancia media dele fora da faixa. A duvida preve quando a lei final esta
errada? Mede-se a AUC nos 60 ajustes finais.

PRE-REGISTRO, ANTES DE RODAR — semente 20260927, 20 sementes por braco
-------------------------------------------------------------------------
  F0  PASSIVO: lei exata entre 40% e 85% (a orbita3 deu 70%)
  F1  ATIVO - PASSIVO >= +20 pontos em lei exata
  F2  ATIVO >= LARGO - 10 pontos: o agente, sozinho, escolhe tao bem quanto a
      faixa que eu teria de dar
  F3  a DUVIDA preve lei errada com AUC >= 0,75

PREVISAO: F0 passa. F1 passa (~+25). F2 passa — a discordancia vai mandar medir
longe, que e o que o largo faz por construcao. F3 passa (~0,8): nas falhas da
orbita3 o expoente errado venceu por pouco, entao reamostragens devem discordar.
Calibragem: das 7 previsoes desde ontem sobre magnitude, errei 5 para cima.

RESULTADO (511 s):
    passivo  lei exata 16/20   r medio das 120 extras 1,83
    largo    lei exata 19/20                          3,30
    ativo    lei exata 17/20                          1,20
F0 passou (80%). F1 FALHOU (+5). F2 passou no limite (-10). F3 FALHOU (AUC 0,63).
O DIAGNOSTICO: o agente foi medir PERTO DO CENTRO (r 1,20), nao longe. O comite de
reamostragens quase sempre concorda na FORMA da lei; so difere nos coeficientes —
e a variancia dos coeficientes e maior onde a aceleracao e maior, perto do centro.
Reamostrar os proprios dados mede RUIDO, nao IGNORANCIA ESTRUTURAL. E a mesma
duvida por reamostragem nao sabe quando a forma esta errada (AUC 0,63). O comite
da `lente_ativa` era do mesmo tipo — pode ser o mesmo defeito.
Conserto: `falseia2.py` — duvida ESTRUTURAL (rivais de forma diferente, pesados
por bits).
"""

from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import time

import numpy as np

from novaphys import orbita as O
from novaphys import orbita2 as O2
from novaphys import orbita3 as O3

SEMENTE = 20260927
N0, NA, K, POOL = 280, 120, 4, 2000
BRACOS = ("passivo", "largo", "ativo")


def estados_faixa(rng, n, rmin, rmax, ang):
    r = rng.uniform(rmin, rmax, n)
    th = rng.uniform(0, 2 * np.pi, n)
    v = rng.uniform(0.6, 1.3, n) / np.sqrt(r)
    a = rng.uniform(-ang, ang, n)
    sentido = np.where(rng.random(n) < 0.3, -1.0, 1.0)
    vt, vr = v * np.cos(a) * sentido, v * np.sin(a)
    c, s = np.cos(th), np.sin(th)
    return np.stack([r * c, r * s, vr * c - vt * s, vr * s + vt * c], 1)


def largo(rng, n):
    return estados_faixa(rng, n, 0.7, 6.0, 1.0)


S4 = estados_faixa(np.random.default_rng(SEMENTE + 4), 2000, 6.0, 12.0, 1.0)
PFORA = estados_faixa(np.random.default_rng(SEMENTE + 5), 500, 3.0, 6.0, 1.0)


def comite(S, D, rng):
    hips = []
    for _ in range(K):
        idx = rng.integers(0, len(S), len(S))
        sub = np.random.default_rng(int(rng.integers(1 << 30)))
        hips.append(O3.descobre(S[idx], D[idx], False, sub, "gauss"))
    return hips


def discordancia(hips, P):
    preds = np.stack([O.preve(ac, P, False) for ac in hips])
    return preds.var(0).sum(1)


def episodio(args):
    braco, k = args
    rng = np.random.default_rng(SEMENTE + 1000 * k)
    S0 = O.estados(rng, N0)
    D0 = O.mede("einstein", S0, rng)
    rb = np.random.default_rng(SEMENTE + 1000 * k + 1)
    if braco == "passivo":
        S1 = O.estados(rb, NA)
    elif braco == "largo":
        S1 = largo(rb, NA)
    else:
        hips = comite(S0, D0, rb)
        P = largo(rb, POOL)
        S1 = P[np.argsort(-discordancia(hips, P))[:NA]]
    D1 = O.mede("einstein", S1, rb)
    S, D = np.concatenate([S0, S1]), np.concatenate([D0, D1])
    ac = O3.descobre(S, D, False, np.random.default_rng(SEMENTE + 1000 * k + 2), "gauss")
    exata = O3.exata_sem_h(ac)
    r2 = O2._r2(O.anomalia("einstein", S4), O.preve(ac, S4, False))
    duvida = float(discordancia(comite(S, D, np.random.default_rng(SEMENTE + 1000 * k + 3)),
                                PFORA).mean())
    r1 = float(np.hypot(S1[:, 0], S1[:, 1]).mean())
    return braco, k, exata, r2, duvida, r1, O.escreve(ac)


def auc(pos, neg):
    """P(duvida de uma lei errada > duvida de uma lei certa)."""
    if not pos or not neg:
        return float("nan")
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--sementes", type=int, default=20)
    ap.add_argument("--nucleos", type=int, default=3)
    a = ap.parse_args()
    n = a.sementes
    print("=" * 76)
    print("FALSEIA — medir onde as proprias hipoteses discordam")
    print("=" * 76)
    t0 = time.time()
    with Pool(a.nucleos) as pool:
        res = pool.map(episodio, [(b, k) for k in range(n) for b in BRACOS], chunksize=1)
    print(f"\n  ({time.time()-t0:.0f}s)\n")
    tx = {}
    for b in BRACOS:
        rs = [r for r in res if r[0] == b]
        ex = sum(r[2] for r in rs)
        r2ok = sum(r[3] >= 0.98 for r in rs)
        tx[b] = ex / n
        print(f"  {b:8s} lei exata {ex:2d}/{n}   R2 em r 6-12 >= 0,98: {r2ok:2d}/{n}   "
              f"r medio das 120 medidas extras {np.mean([r[5] for r in rs]):.2f}")
    print("\n  exemplos (ativo):")
    for r in [r for r in res if r[0] == "ativo"][:3]:
        print(f"    s{r[1]}  {r[6]}")
    pos = [r[4] for r in res if not r[2]]
    neg = [r[4] for r in res if r[2]]
    A = auc(pos, neg)
    print("")
    print("=" * 76)
    p = tx["passivo"]
    print(f"F0 {'.' if 0.40 <= p <= 0.85 else 'FORA .'} passivo lei exata {100*p:.0f}% (40-85%)")
    f1 = tx["ativo"] - p
    print(f"F1 {'.' if f1 >= 0.20 else 'FALHOU .'} ativo - passivo = {100*f1:+.0f} pontos (limiar +20)")
    f2 = tx["ativo"] - tx["largo"]
    print(f"F2 {'.' if f2 >= -0.10 else 'FALHOU .'} ativo - largo = {100*f2:+.0f} pontos (limiar >= -10)")
    print(f"F3 {'.' if A >= 0.75 else 'FALHOU .'} a duvida preve lei errada: AUC = {A:.2f} "
          f"({len(pos)} erradas, {len(neg)} certas; limiar 0,75)")


if __name__ == "__main__":
    main()
