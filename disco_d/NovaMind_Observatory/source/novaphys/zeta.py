"""
novaphys.zeta — a superficie empirica de Riemann, e um teste de FALSO POSITIVO
===============================================================================

A OBJECAO, PRIMEIRO
---------------------
Problemas do milenio sao problemas de PROVA. Esta maquina e um achador de
estrutura em dado: nao tem provador de teoremas, nem linguagem formal, nem
verificador de prova. Apontar para "prove Riemann" e erro de categoria.

O QUE E HONESTO FAZER
-----------------------
Riemann e o unico dos seis com superficie empirica rica, computavel nesta
maquina, e com lei conhecida que serve de CONTROLE POSITIVO: os zeros de
zeta(1/2 + it) tem espacamentos normalizados que seguem a estatistica GUE de
matrizes aleatorias — a lei de Montgomery-Odlyzko.

Entao este modulo faz duas coisas, e a segunda e a que vale mais:

  1. CONTROLE POSITIVO. A maquina reencontra sozinha, do dado, a estatistica GUE?
     Se nao reencontrar, o metodo esta quebrado — do mesmo jeito que K6
     (area = largura x altura) era o controle positivo no dado do Allen.

  2. TESTE DE FALSO POSITIVO DA MAQUINA EXPLORATORIA. Os zeros de zeta sao o
     dado mais estudado da matematica: Odlyzko computou zeros a altura 10^22, e
     a estatistica de espacamentos e verificada ha 50 anos. A probabilidade de
     haver relacao nova e essencialmente ZERO.
     Logo, se o lado exploratorio ANUNCIAR uma descoberta aqui, essa descoberta
     e um FALSO POSITIVO — e isso calibra o portao exploratorio num terreno onde
     se sabe a resposta, ANTES de apontar a maquina para o dado do cerebro, onde
     nao se sabe.

     Este e o uso real: nao descobrir Riemann, e sim medir a taxa de alarme
     falso da maquina de descoberta contra um gabarito perfeito.

O DADO
--------
Os primeiros N zeros nao triviais, calculados com mpmath (`zetazero`), 102 ms
cada, 8000 zeros em paralelo. A densidade media de zeros na altura t e
(1/2pi) log(t/2pi), entao o espacamento NORMALIZADO e

    d_n = (t_{n+1} - t_n) * (1/2pi) * log(t_n / 2pi)        media ~ 1

NULO: processo de Poisson com a mesma media (espacamentos exponenciais
independentes). Tudo o que se mede no dado real se mede no nulo.

PRE-REGISTRO, ANTES DE OLHAR
------------------------------
  M0  SANIDADE: media de d em [0,95; 1,05] (a normalizacao esta certa)
  M1  REPULSAO DE NIVEL: expoente beta de p(s) ~ s^beta para s pequeno fica em
      [1,4; 2,6] no dado real (GUE = 2), e em [-0,5; 0,5] no nulo de Poisson
      (Poisson = 0). E este expoente que distingue GUE de GOE (1) e de
      independencia (0)
  M2  VARIANCIA: Var(d) em [0,15; 0,23] no real (GUE ~ 0,180) e > 0,7 no nulo
      (Poisson = 1)
  M3  CORRELACAO DE PARES: no real, a forma de Montgomery
      R2(r) = 1 - (sin(pi r)/(pi r))^2 paga BITS contra o modelo plano de
      Poisson; e no nulo o sinal se inverte
  M4  FALSO POSITIVO: o lado exploratorio, rodando o mesmo achador de leis do
      `neuro_leis.py` sobre grandezas derivadas dos zeros, anuncia ZERO relacoes
      novas que sobrevivam a metade escondida + nulo permutado — descontadas as
      leis JA CONHECIDAS que ele deve reencontrar (contagem de
      Riemann-von Mangoldt, media 1 por construcao)
PREVISAO: M0 a M3 passam. Sao os fatos mais verificados da teoria analitica dos
numeros; se falharem, o erro e do meu codigo, e e exatamente para isso que o
controle positivo existe. M4 tambem passa, ou seja: nenhuma descoberta. Se M4
FALHAR — se a maquina anunciar relacao nova nos zeros de zeta — a leitura NAO e
"achamos algo", e "o portao exploratorio esta frouxo", e ele precisa apertar
antes de qualquer rodada no cerebro.
"""

from __future__ import annotations

import argparse
from math import log, pi, sin

import numpy as np

ARQ = "D:/zeta/zeros.npy"
SEMENTE = 20261025


def normaliza(t):
    """Espacamento normalizado pela densidade local de zeros."""
    d = np.diff(t) * (np.log(t[:-1] / (2 * pi)) / (2 * pi))
    return d


def nulo_poisson(n, media, rng):
    """Espacamentos independentes, exponenciais, mesma media."""
    return rng.exponential(media, n)


def repulsao(d, smax=0.6, nbin=8):
    """Expoente beta em p(s) ~ s^beta, ajustado nos bins de s pequeno.

    GUE (matrizes hermitianas) da beta = 2; GOE (simetricas reais) da 1;
    processo independente (Poisson) da 0. E este expoente, e nao a media nem a
    variancia, que identifica a CLASSE de simetria."""
    bins = np.linspace(0, smax, nbin + 1)
    h, _ = np.histogram(d, bins=bins, density=True)
    c = 0.5 * (bins[1:] + bins[:-1])
    m = h > 0
    if m.sum() < 3:
        return float("nan")
    p, _b = np.polyfit(np.log(c[m]), np.log(h[m]), 1)
    return float(p)


def r2_gue(r):
    x = pi * np.asarray(r, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(x == 0, 1.0, np.sin(x) / x)
    return 1.0 - s ** 2


def par_correlacao(t, rmax=3.0, nbin=30):
    """Correlacao de pares dos zeros normalizados, na forma de Montgomery."""
    u = np.cumsum(np.concatenate([[0.0], normaliza(t)]))   # posicoes unfolded
    bins = np.linspace(0, rmax, nbin + 1)
    conta = np.zeros(nbin)
    n = len(u)
    for k in range(1, 60):                      # vizinhos ate 60 de distancia
        dif = u[k:] - u[:-k]
        dif = dif[dif < rmax]
        if len(dif) == 0:
            break
        h, _ = np.histogram(dif, bins=bins)
        conta += h
    larg = bins[1] - bins[0]
    esperado = n * larg                         # densidade 1 por construcao
    return 0.5 * (bins[1:] + bins[:-1]), conta / max(esperado, 1e-9)


def bits_modelo(obs, prev, sigma):
    """Bits para codificar `obs` sob um modelo gaussiano em torno de `prev`."""
    obs, prev = np.asarray(obs, float), np.asarray(prev, float)
    return float(np.sum(np.log2(sigma * np.sqrt(2 * pi))
                        + (obs - prev) ** 2 / (2 * sigma ** 2 * np.log(2))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arquivo", default=ARQ)
    a = ap.parse_args()
    rng = np.random.default_rng(SEMENTE)

    t = np.load(a.arquivo)
    print("=" * 78)
    print(f"ZETA — superficie empirica de Riemann, {len(t)} zeros "
          f"(t de {t[0]:.3f} a {t[-1]:.1f})")
    print("=" * 78)

    d = normaliza(t)
    dn = nulo_poisson(len(d), float(d.mean()), rng)

    m_real, m_nulo = float(d.mean()), float(dn.mean())
    v_real, v_nulo = float(d.var()), float(dn.var())
    b_real, b_nulo = repulsao(d), repulsao(dn)

    print(f"  {'':22s} {'REAL':>10s} {'NULO Poisson':>14s} {'GUE':>8s}")
    print(f"  {'media':22s} {m_real:10.4f} {m_nulo:14.4f} {1.0:8.3f}")
    print(f"  {'variancia':22s} {v_real:10.4f} {v_nulo:14.4f} {0.180:8.3f}")
    print(f"  {'expoente de repulsao':22s} {b_real:10.4f} {b_nulo:14.4f} "
          f"{2.0:8.3f}")

    r, g_real = par_correlacao(t)
    u_nulo = np.cumsum(dn)
    t_nulo = u_nulo * 0 + 0                       # o nulo ja esta unfolded
    bins = np.linspace(0, 3.0, 31)
    conta = np.zeros(30)
    for k in range(1, 60):
        dif = u_nulo[k:] - u_nulo[:-k]
        dif = dif[dif < 3.0]
        if len(dif) == 0:
            break
        h, _ = np.histogram(dif, bins=bins)
        conta += h
    g_nulo = conta / max(len(u_nulo) * (bins[1] - bins[0]), 1e-9)

    sig = max(float(np.std(g_real - r2_gue(r))), 1e-3)
    br_gue = bits_modelo(g_real, r2_gue(r), sig)
    br_poi = bits_modelo(g_real, np.ones_like(r), sig)
    bn_gue = bits_modelo(g_nulo, r2_gue(r), sig)
    bn_poi = bits_modelo(g_nulo, np.ones_like(r), sig)
    print(f"\n  correlacao de pares, bits (menor = melhor):")
    print(f"    dado REAL:  GUE {br_gue:8.1f}   Poisson {br_poi:8.1f}   "
          f"GUE ganha por {br_poi-br_gue:+.1f}")
    print(f"    dado NULO:  GUE {bn_gue:8.1f}   Poisson {bn_poi:8.1f}   "
          f"GUE ganha por {bn_poi-bn_gue:+.1f}")
    print("\n    r     R2 medido   R2 GUE   (Poisson = 1)")
    for k in range(0, 12, 2):
        print(f"   {r[k]:4.2f}    {g_real[k]:8.3f} {r2_gue(r[k]):8.3f}")

    print("")
    print("=" * 78)
    ok0 = 0.95 <= m_real <= 1.05
    print(f"M0 {'.' if ok0 else 'FALHOU .'} normalizacao: media {m_real:.4f} "
          f"(0,95-1,05)")
    ok1 = 1.4 <= b_real <= 2.6 and -0.5 <= b_nulo <= 0.5
    print(f"M1 {'.' if ok1 else 'FALHOU .'} repulsao de nivel: real "
          f"{b_real:.2f} (1,4-2,6 / GUE=2), nulo {b_nulo:.2f} (-0,5..0,5)")
    ok2 = 0.15 <= v_real <= 0.23 and v_nulo > 0.7
    print(f"M2 {'.' if ok2 else 'FALHOU .'} variancia: real {v_real:.4f} "
          f"(0,15-0,23 / GUE=0,180), nulo {v_nulo:.3f} (> 0,7)")
    ok3 = (br_poi - br_gue) > 0 and (bn_poi - bn_gue) < 0
    print(f"M3 {'.' if ok3 else 'FALHOU .'} correlacao de pares: GUE paga "
          f"{br_poi-br_gue:+.1f} bits no real e {bn_poi-bn_gue:+.1f} no nulo")
    print("\n  M4 (falso positivo do lado exploratorio) roda em zeta_explora.py")


if __name__ == "__main__":
    main()
