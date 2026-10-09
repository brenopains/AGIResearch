"""
novaphys.falseia2 — duvida ESTRUTURAL: os rivais que quase empataram
======================================================================

DE ONDE VEM
-------------
`falseia` falhou (ativo 17/20 contra passivo 16/20; duvida com AUC 0,63), e o
porque ficou visivel: o agente foi medir PERTO do centro (r medio 1,20 contra 3,30
do braco largo). A duvida era um comite de REAMOSTRAGENS — que quase sempre
concorda na FORMA da lei e so discorda nos coeficientes. Reamostrar os proprios
dados mede ruido, nao ignorancia.

A duvida certa e ESTRUTURAL: quais OUTRAS FORMAS de lei explicam quase tao bem o
que ja vi? Sem h, a lei tem um termo forte (v^2/r^2) e um fraco (vr^2/r^2), e o
expoente do fraco e o que fica em aberto — r^-1, r^-2 ou r^-3 empatam dentro da
faixa medida.

O MECANISMO
-------------
  RIVAIS: pega a lei aceita; troca o termo mais fraco (ou, se ha um termo so,
          acrescenta um segundo) pelos melhores candidatos alternativos da
          gramatica; inclui "sem o termo fraco". Cada rival e reajustado.
  PESO:   cada rival pesa 2^-(bits dele - bits do melhor) — comprimento de
          descricao completo (erro + estrutura). E a posterior do MDL.
  ESCOLHA: mede onde a variancia PONDERADA das previsoes dos rivais e maior.
  DUVIDA: 1 - peso da lei escolhida. Deve prever quando ela esta errada.

O DESENHO — semente 20260928, 20 sementes, orcamento 200 (140 + 60)
---------------------------------------------------------------------
Orcamento menor que o da `falseia` (400) porque la o passivo ja acertava 80% e
nao sobrava espaco para diferenca. Quatro bracos, mesmas 140 medidas iniciais:
    passivo            60 sorteadas na faixa de sempre
    largo              60 sorteadas na faixa larga (eu mando olhar longe)
    ativo_boot         o comite de reamostragem da `falseia` (replica do defeito)
    ativo_estrutural   o agente escolhe pelos rivais estruturais

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  G0  passivo: lei exata entre 25% e 75%
  G1  ativo_estrutural - passivo >= +20 pontos
  G2  ativo_estrutural - ativo_boot >= +15 pontos
  G3  a duvida ESTRUTURAL preve lei errada com AUC >= 0,75, e maior que a AUC da
      duvida por reamostragem

PREVISAO: G0 ~55%. G1 passa. G2 passa. G3 passa (~0,8 contra ~0,6) — e o
estrutural deve medir longe (r medio > 2,5) sem ninguem mandar.
Calibragem: das ultimas 9 previsoes de magnitude, 6 erradas para cima.

RESULTADO (327 s):
                      lei exata   r medio das 60 extras
  passivo               4/20          1,87
  largo                 6/20          3,39
  ativo_boot           16/20          1,07
  ativo_estrutural     11/20          1,10
  duvida estrutural preve lei errada: AUC 0,90;  reamostragem: 0,52
G0 FORA (passivo 20%, abaixo de 25%). G1 passou (+35). G2 FALHOU (-25: a
reamostragem escolheu MELHOR). G3 passou (0,90 x 0,52).
Tres coisas:
 1. ESCOLHER O EXPERIMENTO PAGOU, e muito, com dado escasso: 80% x 20% (a 400
    medidas estava no teto). Primeira vez no projeto com modelo APRENDIDO.
 2. Minha premissa estava errada: a informacao NAO esta longe. A anomalia cai
    como r^-4 — longe ela fica abaixo do ruido. Os dois ativos foram medir perto
    do centro sozinhos, e o "largo" quase nao ajudou. A reamostragem aponta onde o
    SINAL e grande — e isso e desenho otimo classico sob ruido constante.
 3. As duas duvidas servem para coisas DIFERENTES: reamostragem para escolher
    onde medir; ESTRUTURAL para saber que a lei esta errada (AUC 0,90 x 0,52).
Previsoes: G1 e G3 certas; G0, G2 e "o estrutural mede longe" erradas.
"""

from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import time
from math import log2

import numpy as np

from novaphys import orbita as O
from novaphys import orbita2 as O2
from novaphys import orbita3 as O3
from novaphys import falseia as F

SEMENTE = 20260928
N0, NA = 140, 60
M_RIVAIS = 6
BRACOS = ("passivo", "largo", "ativo_boot", "ativo_estrutural")


def _estr(ac):
    return [("lei", t[1], tuple(t[2]), tuple(int(x) for x in t[3])) if t[0] == "lei"
            else ("objeto", t[2], t[3]) for t in ac]


def _bases(estr, S):
    nomes, X = O.caderno(S, False)
    idx = {n: j for j, n in enumerate(nomes)}
    dirs = O.direcoes(S)
    out = []
    for t in estr:
        if t[0] == "lei":
            _, d, vs, es = t
            e = np.zeros(len(nomes))
            for v, x in zip(vs, es):
                e[idx[v]] = x
            out.append(O._F(e[None, :], X)[0][:, None] * dirs[d])
        else:
            out.append(O._obj_base(S, np.array([t[1], t[2]])))
    return out


def _ajusta(estr, S, D):
    if not estr:
        return [], float((D ** 2).sum())
    B = _bases(estr, S)
    M = np.stack([b.ravel() for b in B], 1)
    c, *_ = np.linalg.lstsq(M, D.ravel(), rcond=None)
    return c, float(((D.ravel() - M @ c) ** 2).sum())


def _como_ac(estr, c):
    out = []
    for t, ci in zip(estr, c):
        if t[0] == "lei":
            out.append(("lei", t[1], t[2], t[3], float(ci), 0.0))
        else:
            out.append(("objeto", float(ci), t[1], t[2], 0.0))
    return out


def rivais(S, D, ac):
    estr = _estr(ac)
    nomes, X = O.caderno(S, False)
    P = len(nomes)
    E, meta = O.candidatos(P)
    dirs = O.direcoes(S)
    fixos = estr if len(estr) <= 1 else estr[:-1]
    c, _ = _ajusta(fixos, S, D)
    R = D.copy()
    B = _bases(fixos, S)
    if B:
        M = np.stack([b.ravel() for b in B], 1)
        R = (D.ravel() - M @ c).reshape(D.shape)
    top = O2._top_termos(R, O2._ortonormal(B), X, dirs, P, D.size)
    hips, vistos = [estr], {tuple(estr)}
    for _g, i, d in top:
        _k, vs, es = meta[i]
        t = ("lei", d, tuple(nomes[j] for j in vs), tuple(int(x) for x in es))
        h = fixos + [t]
        if tuple(h) not in vistos:
            vistos.add(tuple(h))
            hips.append(h)
        if len(hips) >= M_RIVAIS:
            break
    if len(estr) >= 2 and tuple(fixos) not in vistos:
        hips.append(fixos)
    out = []
    for h in hips:
        ch, sse = _ajusta(h, S, D)
        est = sum(O.bits_termo(P, len(t[2]), D.size) if t[0] == "lei"
                  else O.bits_objeto(D.size) for t in h)
        out.append((h, ch, 0.5 * D.size * log2(max(sse, 1e-300)) + est))
    L = np.array([x[2] for x in out])
    w = 2.0 ** (-(L - L.min()))
    return out, w / w.sum()


def discordancia(rv, w, P):
    preds = np.stack([O.preve(_como_ac(h, c), P, False) if h else np.zeros((len(P), 2))
                      for h, c, _ in rv])
    media = (w[:, None, None] * preds).sum(0)
    return (w[:, None, None] * (preds - media) ** 2).sum(0).sum(1)


def episodio(args):
    braco, k = args
    rng = np.random.default_rng(SEMENTE + 1000 * k)
    S0 = O.estados(rng, N0)
    D0 = O.mede("einstein", S0, rng)
    rb = np.random.default_rng(SEMENTE + 1000 * k + 1)
    if braco == "passivo":
        S1 = O.estados(rb, NA)
    elif braco == "largo":
        S1 = F.largo(rb, NA)
    elif braco == "ativo_boot":
        hips = F.comite(S0, D0, rb)
        Pp = F.largo(rb, F.POOL)
        S1 = Pp[np.argsort(-F.discordancia(hips, Pp))[:NA]]
    else:
        ac0 = O3.descobre(S0, D0, False, np.random.default_rng(SEMENTE + 1000 * k + 5), "gauss")
        rv, w = rivais(S0, D0, ac0)
        Pp = F.largo(rb, F.POOL)
        S1 = Pp[np.argsort(-discordancia(rv, w, Pp))[:NA]]
    D1 = O.mede("einstein", S1, rb)
    S, D = np.concatenate([S0, S1]), np.concatenate([D0, D1])
    ac = O3.descobre(S, D, False, np.random.default_rng(SEMENTE + 1000 * k + 2), "gauss")
    exata = O3.exata_sem_h(ac)
    r2 = O2._r2(O.anomalia("einstein", F.S4), O.preve(ac, F.S4, False))
    _rv, w = rivais(S, D, ac)
    d_est = float(1 - w[0])
    d_boot = float(F.discordancia(F.comite(S, D, np.random.default_rng(SEMENTE + 1000 * k + 3)),
                                  F.PFORA).mean())
    r1 = float(np.hypot(S1[:, 0], S1[:, 1]).mean())
    return braco, k, exata, r2, d_est, d_boot, r1, O.escreve(ac)


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--sementes", type=int, default=20)
    ap.add_argument("--nucleos", type=int, default=3)
    a = ap.parse_args()
    n = a.sementes
    print("=" * 76)
    print("FALSEIA 2 — duvida estrutural: medir onde os rivais de FORMA discordam")
    print("=" * 76)
    t0 = time.time()
    with Pool(a.nucleos) as pool:
        res = pool.map(episodio, [(b, k) for k in range(n) for b in BRACOS], chunksize=1)
    print(f"\n  ({time.time()-t0:.0f}s)\n")
    tx = {}
    for b in BRACOS:
        rs = [r for r in res if r[0] == b]
        ex = sum(r[2] for r in rs)
        tx[b] = ex / n
        print(f"  {b:17s} lei exata {ex:2d}/{n}   R2 r 6-12 >= 0,98: "
              f"{sum(r[3] >= 0.98 for r in rs):2d}/{n}   r medio das extras "
              f"{np.mean([r[6] for r in rs]):.2f}")
    erradas = [r for r in res if not r[2]]
    certas = [r for r in res if r[2]]
    a_est = F.auc([r[4] for r in erradas], [r[4] for r in certas])
    a_boot = F.auc([r[5] for r in erradas], [r[5] for r in certas])
    print("")
    print("=" * 76)
    p = tx["passivo"]
    print(f"G0 {'.' if 0.25 <= p <= 0.75 else 'FORA .'} passivo {100*p:.0f}% (25-75%)")
    g1 = tx["ativo_estrutural"] - p
    print(f"G1 {'.' if g1 >= 0.20 else 'FALHOU .'} estrutural - passivo = {100*g1:+.0f} (limiar +20)")
    g2 = tx["ativo_estrutural"] - tx["ativo_boot"]
    print(f"G2 {'.' if g2 >= 0.15 else 'FALHOU .'} estrutural - reamostragem = {100*g2:+.0f} (limiar +15)")
    print(f"G3 {'.' if a_est >= 0.75 and a_est > a_boot else 'FALHOU .'} duvida estrutural "
          f"AUC {a_est:.2f}  x  reamostragem {a_boot:.2f}  ({len(erradas)} erradas, "
          f"{len(certas)} certas)")


if __name__ == "__main__":
    main()
