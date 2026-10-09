"""novaphys.zeta_fronteira — a relacao ATRAVESSA a fronteira, ou so descreve?

O QUE ISTO CONSERTA
---------------------
`busca()` divide treino/teste ALEATORIAMENTE (`rng.random(n) < 0.5`). Numa
sequencia ordenada isso INTERCALA treino e teste: o modelo nunca extrapola em t.
Toda validacao do zeta ate hoje foi INTERPOLACAO. Aqui o corte e PREFIXO ->
SUFIXO: treina nos primeiros 4.000 zeros, CONGELA, preve os ultimos ~4.000.

Mede a grandeza que hoje se mostrou decisiva em `alcance_total`: ACERTO NA
FRONTEIRA — nao "quanto comprime no visto", mas "acerta onde nunca esteve".

AS RELACOES CONGELADAS (nada de procurar coisa nova no sufixo)
----------------------------------------------------------------
Sobreviventes do nulo do duto (`nulo_do_duto.py` matou 15 de 18):
  R1  d ~ d_ant          anticorrelacao de espacamentos consecutivos (GUE)
  R2  N_menos_n ~ d      eu julguei ALGEBRICA a mao; o nulo do duto me corrigiu
  R3  N_menos_n ~ d_ant  idem
Controle positivo do INSTRUMENTO (tem de atravessar, e lei exata):
  C1  n ~ t              Riemann-von Mangoldt, +17.520 bits
Controle de referencia:
  B0  media do treino    o baseline trivial que toda relacao tem de bater

PRE-REGISTRO — previsoes escritas ANTES de rodar
--------------------------------------------------
  F0  C1 atravessa quase perfeito (R2 fronteira > 0,99). Se falhar, o TESTE
      esta quebrado e nada abaixo vale. Previsao: passa, 97%.
  F1  R1 atravessa: R2 fronteira >= 0,8 x R2 interpolado. Previsao: passa, 85%
      (anticorrelacao do GUE e estacionaria; nao depende de t).
  F2  R2 e R3 atravessam pelo mesmo criterio. Previsao: passa, 70% — menos
      confianca porque eu ja errei o julgamento sobre elas uma vez.
  F3  QUEDA DE EXTRAPOLACAO: para cada relacao, R2(prefixo) < R2(aleatorio).
      Se a queda for grande, a relacao estava em parte ajustando a TENDENCIA em
      t, nao o fenomeno. Previsao: queda pequena (<10%) em R1, maior em R2/R3.

Nenhuma relacao nova sera procurada no sufixo. O sufixo e so gabarito.
"""
from __future__ import annotations
import sys
from math import pi
import numpy as np, pandas as pd

sys.path.insert(0, ('C:/Users/usuario/Documents/AGI 2.0').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))
from novaphys.zeta_explora import grandezas
from novaphys.nulo_do_duto import substrato_aleatorio

ARQ = "D:/zeta/zeros.npy"
RELACOES = [("C0 dens~logt    (IDENTIDADE exata)", "dens", "logt"),
            ("C1 n~t          (Riemann-von Mangoldt)", "n", "t"),
            ("R1 d~d_ant      (GUE)", "d", "d_ant"),
            ("R2 N_menos_n~d", "N_menos_n", "d"),
            ("R3 N_menos_n~d_ant", "N_menos_n", "d_ant")]


def ajusta(x, y):
    A = np.vstack([np.ones_like(x), x]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return c


def r2(y, pred):
    ss = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - np.sum((y - pred) ** 2) / ss if ss > 0 else np.nan


def _familia(x, y, tr):
    """Escolhe linear ou log-log USANDO SO O TREINO — como o busca() faz.
    Congelar so a familia linear foi o defeito que o controle F0 pegou:
    n ~ t nao e linear (N(t) ~ t log t) e a reta extrapolada da R2 0,86."""
    cand = [("lin", lambda v: v, lambda v: v)]
    if (x[tr] > 0).all() and (y[tr] > 0).all():
        cand.append(("log", np.log, np.exp))
    melhor = None
    for nome, f, fi in cand:
        c = ajusta(f(x[tr]), f(y[tr]))
        pred = fi(c[0] + c[1] * f(x[tr]))
        r = r2(y[tr], pred)
        if melhor is None or r > melhor[0]:
            melhor = (r, nome, f, fi, c)
    return melhor[1:]


def avalia(df, nome, cy, cx, rng):
    y, x = df[cy].to_numpy(float), df[cx].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x); meio = n // 2
    # FRONTEIRA: treina no prefixo, congela, preve o sufixo
    tr0 = np.zeros(n, bool); tr0[:meio] = True
    fam, f, fi, c = _familia(x, y, tr0)
    with np.errstate(all="ignore"):
        r_fr = r2(y[meio:], fi(c[0] + c[1] * f(x[meio:])))
    # baseline trivial: media do TREINO aplicada ao sufixo
    r_b0 = r2(y[meio:], np.full(n - meio, y[:meio].mean()))
    # oraculo: reajusta NO sufixo (teto)
    te0 = ~tr0
    _, fo, fio, co = _familia(x, y, te0)
    with np.errstate(all="ignore"):
        r_or = r2(y[meio:], fio(co[0] + co[1] * fo(x[meio:])))
    # INTERPOLACAO: o split aleatorio que o busca() usava
    m = rng.random(n) < 0.5
    _, fm, fim, ci = _familia(x, y, m)
    with np.errstate(all="ignore"):
        r_in = r2(y[~m], fim(ci[0] + ci[1] * fm(x[~m])))
    return dict(nome=nome, n=n, fam=fam, fronteira=r_fr, interpol=r_in,
                oraculo=r_or, baseline=r_b0,
                queda=(r_in - r_fr) / abs(r_in) if abs(r_in) > 1e-9 else np.nan)


def main():
    rng = np.random.default_rng(20260916)
    t = np.load(ARQ)
    df = grandezas(t)
    print("=" * 84)
    print(f"ZETA FRONTEIRA — treina nos primeiros, CONGELA, preve os ultimos")
    print(f"{len(df):,} zeros derivados de {len(t):,} | corte prefixo/sufixo ao meio")
    print("=" * 84)
    print(f"\n{'relacao':<38}{'FRONTEIRA':>11}{'interpol':>10}{'oraculo':>9}"
          f"{'trivial':>9}{'queda':>8}")
    res = []
    for nome, cy, cx in RELACOES:
        r = avalia(df, nome, cy, cx, rng); res.append(r)
        print(f"  {nome:<36}{r['fronteira']:11.4f}{r['interpol']:10.4f}"
              f"{r['oraculo']:9.4f}{r['baseline']:9.4f}{r['queda']*100:7.1f}%  [{r['fam']}]")

    print("\n--- NULO DO DUTO: as mesmas relacoes sobre substrato aleatorio ---")
    tn = substrato_aleatorio(t, rng)
    dn = grandezas(tn)
    print(f"{'relacao':<38}{'FRONTEIRA':>11}{'interpol':>10}")
    nulo = {}
    for nome, cy, cx in RELACOES:
        r = avalia(dn, nome, cy, cx, rng); nulo[nome] = r
        print(f"  {nome:<36}{r['fronteira']:11.4f}{r['interpol']:10.4f}")

    print("\n" + "=" * 84)
    c0 = res[0]
    print(f"F0 {'.' if c0['fronteira'] > 0.999 else 'FALHOU .'} controle de "
          f"instrumento (identidade exata): R2 fronteira = {c0['fronteira']:.6f}")
    print(f"   (secundario, nao-linear) {res[1]['nome'][:20]}: {res[1]['fronteira']:.4f} "
          f"familia={res[1]['fam']}")
    for r in res[2:]:
        alvo = 0.8 * r['interpol']
        pas = r['fronteira'] >= alvo and r['fronteira'] > nulo[r['nome']]['fronteira'] + 0.01
        print(f"   {'.' if pas else 'FALHOU .'} {r['nome'][:22]:<24} fronteira "
              f"{r['fronteira']:7.4f} vs alvo {alvo:7.4f} vs nulo "
              f"{nulo[r['nome']]['fronteira']:7.4f}")


if __name__ == "__main__":
    main()
