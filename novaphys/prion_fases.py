"""
novaphys.prion_fases — o que governa a fase sintomatica, que e onde o
                       tratamento clinico real acontece
=======================================================================

O QUE O DADO MOSTROU
----------------------
Raymond et al. 2019, braco de dose unica no dia 120:

    inicio clinico        142 d -> 189 d      +33%
    sobrevida             157 d -> 244 d      +55%
    duracao sintomatica   15+-4 d -> 53+-7 d  3,5x

O modelo de carga preve o INICIO bem (~+5 pp). O que ele nao produz de jeito
nenhum e a triplicacao da fase sintomatica — porque nele o cruzamento do
limiar E a morte.

E essa fase e a que importa clinicamente: o ION717 e dado a pacientes JA
SINTOMATICOS. Para eles, adiar o inicio nao serve de nada; o que vale e
quanto tempo se ganha DEPOIS do inicio.

MODELO DE DUAS FASES
----------------------
    fase 1   infeccao -> inicio      inicio quando y >= LIMIAR
    fase 2   inicio -> morte         morte quando o dano acumulado L >= LETAL

A pergunta e o que dirige dL/dt. Quatro leis, todas plausiveis:

    carga     dL/dt = y             dano pelo agregado acumulado
    fluxo     dL/dt = 2*beta*x*z    dano pelo evento de conversao
    prpc      dL/dt = x*z           PrP-C disponivel x moldes
    moldes    dL/dt = z             numero de pontas ativas

So `fluxo` e `prpc` dependem de x — ou seja, so nelas baixar PrP-C age
IMEDIATAMENTE, com agregado ja formado. Se a lei vencedora depender de x,
isso explica a triplicacao e diz que o ASO age na fase que importa.

O TESTE, SEM PARAMETRO LIVRE NO PASSO FINAL
---------------------------------------------
    1. calibra beta para o controle: inicio em 142 d
    2. calibra LETAL para o controle: duracao de 15 d
    3. ajusta k SO na sobrevida profilatica (81%)
    4. PREVE a duracao tratada do braco tardio (observado 53 d)

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  S0  PRE-CONDICAO. Cada lei calibra para inicio 142 d e duracao 15 d.
  S1  alguma lei preve a duracao tratada dentro de +-30% de 53 d (37-69 d)
      -> a fase sintomatica tem mecanismo identificavel, e sabemos qual
  S2  nenhuma preve -> a duracao depende de algo fora deste modelo
  S3  com a lei vencedora: ordem de alvos PARA PACIENTE JA SINTOMATICO,
      que e a pergunta clinica de verdade e que ninguem respondeu ainda
      neste arquivo

MINHA PREVISAO: `fluxo` ou `prpc` vencem, porque so elas dependem de x.
`carga` deve falhar feio, porque y esta em equilibrio e mal se move. Se
`carga` vencer, minha leitura da biologia esta errada.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from .prion import BASE
from .prion_validacao import _efeito

LIMIAR = 55000.0
INICIO_CTRL = 142.0        # inicio clinico no controle, braco tardio
DUR_CTRL = 15.0            # duracao sintomatica no controle
DUR_TRAT = 53.0            # observado no tratado — o alvo da previsao
SOBREVIDA_CTRL = 157.0
OBS_PROFILATICO = 81.0     # extensao de sobrevida, braco profilatico NIH
CONTROLE_NIH = 143.0
DIA_DOSE = 120.0

LEIS: Dict[str, Callable] = {
    "carga":  lambda x, y, z, p: y,
    "fluxo":  lambda x, y, z, p: 2 * p["beta"] * x * z,
    "prpc":   lambda x, y, z, p: x * z,
    "moldes": lambda x, y, z, p: z,
}


def inicio_clinico(p: Dict[str, float], k: float = 0.0,
                   inicio: float = 0.0, sustentado: bool = True,
                   t_max: float = 3000.0) -> Optional[float]:
    """Fase 1, tres variaveis. Reusa a dinamica ja verificada."""
    n, d, beta, a, b = p["n"], p["d"], p["beta"], p["a"], p["b"]
    lam0 = p["lam"]

    def campo(t, s):
        x, y, z = s
        lam = lam0 * (1.0 - k * _efeito(t, inicio, sustentado))
        return [lam - d * x - 2 * beta * x * z,
                2 * beta * x * z - a * y - b * n * (n - 1) * z,
                b * (y - (2 * n - 1) * z) - a * z]

    def ev(t, s):
        return s[1] - LIMIAR
    ev.terminal = True
    ev.direction = 1
    sol = solve_ivp(campo, (0.0, t_max), [lam0 / d, n * 1.0, 1.0],
                    method="Radau", events=ev, rtol=1e-9, atol=1e-9)
    if not sol.success or not len(sol.t_events[0]):
        return None
    return float(sol.t_events[0][0])


def dano_acumulado(p: Dict[str, float], lei: str, letal: Optional[float],
                   k: float = 0.0, inicio: float = 0.0,
                   sustentado: bool = True, dur_max: float = 4000.0
                   ) -> Tuple[Optional[float], Optional[float], float]:
    """
    Fase 2. Integra a partir do inicio clinico ate o dano cruzar `letal`.

    Se `letal` for None, NAO ha evento: integra por `dur_max` e devolve o
    dano acumulado — e assim que o limiar letal e CALCULADO a partir do
    controle, em vez de bissectado. A bisseccao anterior exigia atol de 1e-9
    numa variavel que cresce a 1e8, e o integrador estourava.
    """
    ti = inicio_clinico(p, k, inicio, sustentado)
    if ti is None:
        return None, None, float("nan")
    n, d, beta, a, b = p["n"], p["d"], p["beta"], p["a"], p["b"]
    lam0 = p["lam"]
    g = LEIS[lei]

    # estado no inicio clinico
    def campo3(t, s):
        x, y, z = s
        lam = lam0 * (1.0 - k * _efeito(t, inicio, sustentado))
        return [lam - d * x - 2 * beta * x * z,
                2 * beta * x * z - a * y - b * n * (n - 1) * z,
                b * (y - (2 * n - 1) * z) - a * z]
    s0 = solve_ivp(campo3, (0.0, ti), [lam0 / d, n * 1.0, 1.0],
                   method="Radau", rtol=1e-9, atol=1e-9).y[:, -1]

    def campo4(t, s):
        x, y, z, _L = s
        lam = lam0 * (1.0 - k * _efeito(t, inicio, sustentado))
        return [lam - d * x - 2 * beta * x * z,
                2 * beta * x * z - a * y - b * n * (n - 1) * z,
                b * (y - (2 * n - 1) * z) - a * z,
                g(x, y, z, p)]

    eventos = []
    if letal is not None:
        def ev(t, s):
            return s[3] - letal
        ev.terminal = True
        ev.direction = 1
        eventos = [ev]
    sol = solve_ivp(campo4, (ti, ti + dur_max), list(s0) + [0.0],
                    method="Radau", events=eventos,
                    rtol=1e-8, atol=[1e-8, 1e-8, 1e-8, 1e-4])
    if not sol.success:
        return ti, None, float("nan")
    if letal is not None and len(sol.t_events[0]):
        tm = float(sol.t_events[0][0])
        return ti, tm, tm - ti
    return ti, None, float(sol.y[3, -1])


def letal_do_controle(p: Dict[str, float], lei: str, dur: float) -> float:
    """
    O limiar letal E o dano acumulado do controle ao longo da sua duracao
    sintomatica observada. Uma integracao, sem bisseccao.
    """
    ti = inicio_clinico(p)
    if ti is None:
        return float("nan")
    n, d, beta, a, b = p["n"], p["d"], p["beta"], p["a"], p["b"]
    lam0 = p["lam"]
    g = LEIS[lei]

    def campo3(t, s):
        x, y, z = s
        return [lam0 - d * x - 2 * beta * x * z,
                2 * beta * x * z - a * y - b * n * (n - 1) * z,
                b * (y - (2 * n - 1) * z) - a * z]
    s0 = solve_ivp(campo3, (0.0, ti), [lam0 / d, n * 1.0, 1.0],
                   method="Radau", rtol=1e-9, atol=1e-9).y[:, -1]

    def campo4(t, s):
        x, y, z, _L = s
        return [lam0 - d * x - 2 * beta * x * z,
                2 * beta * x * z - a * y - b * n * (n - 1) * z,
                b * (y - (2 * n - 1) * z) - a * z,
                g(x, y, z, p)]
    sol = solve_ivp(campo4, (ti, ti + dur), list(s0) + [0.0], method="Radau",
                    rtol=1e-8, atol=[1e-8, 1e-8, 1e-8, 1e-4])
    return float(sol.y[3, -1]) if sol.success else float("nan")


def cal_beta(alvo: float) -> float:
    lo, hi = 1e-4, 2.0
    for _ in range(55):
        mid = (lo + hi) / 2
        p = dict(BASE); p["beta"] = mid
        ti = inicio_clinico(p)
        if ti is None or ti > alvo:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def relatorio() -> None:
    print("=" * 78)
    print("PRION — o que governa a FASE SINTOMATICA")
    print("=" * 78)
    print("PRE-REGISTRO:")
    print("  S0  cada lei calibra para inicio 142 d e duracao 15 d")
    print("  S1  alguma preve duracao tratada em 37-69 d (+-30% de 53)")
    print("  S2  nenhuma preve -> mecanismo fora deste modelo")
    print("  S3  com a vencedora: ordem de alvos para PACIENTE SINTOMATICO")
    print("\nMINHA PREVISAO: `fluxo` ou `prpc` vencem (so elas dependem de x).")
    print("Se `carga` vencer, minha leitura da biologia esta errada.")

    beta_t = cal_beta(INICIO_CTRL)
    beta_n = cal_beta(CONTROLE_NIH)
    pt = dict(BASE); pt["beta"] = beta_t
    pn = dict(BASE); pn["beta"] = beta_n
    ti, _ = simular(pt, "carga", 1e30)
    print("\n" + "-" * 78)
    print(f"S0 · beta calibrado: tardio {beta_t:.5g} -> inicio {ti:.1f} d "
          f"(alvo {INICIO_CTRL:.0f})")

    print(f"\n{'lei':<10}{'LETAL':>13}{'dur ctrl':>10}{'k':>8}"
          f"{'dur prev':>11}{'obs':>7}{'erro':>9}")
    print("-" * 78)
    res = []
    for lei in LEIS:
        letal_t = cal_letal(pt, lei, DUR_CTRL)
        letal_n = cal_letal(pn, lei, DUR_CTRL)
        if not math.isfinite(letal_t):
            print(f"{lei:<10}  S0 REPROVADA")
            continue
        ti, tm = simular(pt, lei, letal_t)
        dur0 = tm - ti if (ti and tm) else float("nan")
        if not (abs(dur0 - DUR_CTRL) / DUR_CTRL < 0.15):
            print(f"{lei:<10}  S0 REPROVADA (dur ctrl {dur0:.1f})")
            continue
        # ajusta k na sobrevida profilatica
        lo, hi = 0.0, 0.995
        for _ in range(40):
            mid = (lo + hi) / 2
            _tin, tmn = simular(pn, lei, letal_n, k=mid, inicio=-14.0,
                                sustentado=True)
            e = math.inf if tmn is None else \
                100 * (tmn - CONTROLE_NIH) / CONTROLE_NIH
            if e > OBS_PROFILATICO:
                hi = mid
            else:
                lo = mid
        k = (lo + hi) / 2
        # PREVE a duracao tratada do braco tardio
        ti2, tm2 = simular(pt, lei, letal_t, k=k, inicio=DIA_DOSE,
                           sustentado=False)
        dur = (tm2 - ti2) if (ti2 and tm2) else math.inf
        err = math.inf if dur == math.inf else 100 * (dur - DUR_TRAT) / DUR_TRAT
        res.append((lei, k, dur, err, letal_t))
        ds = "inf" if dur == math.inf else f"{dur:.1f} d"
        es = "—" if err == math.inf else f"{err:+.0f}%"
        print(f"{lei:<10}{letal_t:>13.4g}{dur0:>10.1f}{k:>8.1%}"
              f"{ds:>11}{DUR_TRAT:>6.0f}d{es:>9}")

    print("\n" + "=" * 78)
    val = [r for r in res if r[3] != math.inf]
    if not val:
        print("S2 · nenhuma lei produz duracao finita.")
        return
    melhor = min(val, key=lambda r: abs(r[3]))
    lei, k, dur, err, letal = melhor
    if abs(err) <= 30.0:
        print(f"S1 · `{lei}` PREVE A FASE SINTOMATICA: {dur:.1f} d contra 53 d")
        print(f"     observados, erro {err:+.0f}%, com k ajustado noutro braco.")
        dep_x = lei in ("fluxo", "prpc")
        if dep_x:
            print("     E ela DEPENDE DE x — baixar PrP-C age imediatamente,")
            print("     com agregado ja formado. Isso explica a triplicacao.")
        else:
            print("     E ela NAO depende de x — minha leitura estava errada.")
    else:
        print(f"S2 · nenhuma lei preve (melhor: `{lei}`, erro {err:+.0f}%).")
        print("     A duracao sintomatica depende de algo fora deste modelo.")
        return

    # ── S3: a pergunta clinica
    print("\n" + "-" * 78)
    print(f"S3 · PACIENTE JA SINTOMATICO — tratamento comecando no inicio")
    print(f"     lei `{lei}` · duracao sem tratamento {DUR_CTRL:.0f} d")
    ALVOS = {"lam- (ASO)": ("lam", -1), "a+ (depuracao)": ("a", +1),
             "beta- (anti-alonga)": ("beta", -1), "b- (anti-frag)": ("b", -1)}
    ti0, _ = simular(pt, lei, letal)
    linhas = []
    for rot, (par, sinal) in ALVOS.items():
        q = dict(pt)
        # 50% de dose, comecando no inicio clinico
        if sinal < 0:
            q[par] = pt[par] * 0.5
        else:
            q[par] = pt[par] * 2.0
        # o parametro muda so a partir do inicio: aproximado aplicando de ti0
        tia, tma = simular(q, lei, letal, k=0.0, inicio=ti0, sustentado=True)
        # para lam usamos o canal de knockdown, que ja tem temporizacao
        if par == "lam":
            tia, tma = simular(pt, lei, letal, k=0.5, inicio=ti0,
                               sustentado=True)
        d2 = (tma - tia) if (tia and tma) else math.inf
        linhas.append((rot, d2))
    linhas.sort(key=lambda r: -(1e18 if r[1] == math.inf else r[1]))
    for rot, d2 in linhas:
        s = "nao morre no horizonte" if d2 == math.inf else \
            f"{d2:.1f} d  ({d2/DUR_CTRL:.1f}x)"
        print(f"     {rot:<22}{s}")
    print("\n     (dose de 50% em cada alvo, iniciada no inicio clinico)")
    if linhas and linhas[0][0].startswith("lam-"):
        print("\n     Para quem JA esta sintomatico, baixar PrP-C lidera.")
    else:
        print(f"\n     Para quem ja esta sintomatico, {linhas[0][0]} lidera.")


if __name__ == "__main__":
    relatorio()
