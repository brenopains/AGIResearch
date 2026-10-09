"""
novaphys.prion_espacial — a frente que avança, em vez do tanque que satura
===========================================================================

POR QUE ESTE MODELO EXISTE
----------------------------
`prion_mecanismo.py` testou quatro definicoes de dano e TODAS subestimaram o
beneficio de tratar tarde:

    carga        -15,3 pp        fluxo         -35,8 pp
    prpc         -35,8 pp        carga-tempo   -44,1 pp

Minha hipotese era que `fluxo` consertaria. Errei: ele piorou. Quando TODA
definicao de morte falha do mesmo jeito, o problema nao e a definicao.

O diagnostico: o modelo e BEM-MISTURADO. Um compartimento so, que satura por
volta do dia 100-200. No dia 120 o sistema ja esta no equilibrio endemico, e
nenhuma leitura de dano pode responder a tratamento depois disso.

A doenca real se espalha REGIAO POR REGIAO. Uma frente que avanca nao satura:
sempre ha tecido virgem adiante. E tratar tarde desacelera a frente — que e
exatamente o beneficio observado.

O MODELO
----------
N compartimentos em cadeia (grosseiramente: sitio de inoculacao -> tronco ->
cortex). Em cada um, a mesma cinetica de polimerizacao nucleada. Sementes
migram para o vizinho a uma taxa `mig`.

    morte quando a FRACAO DE COMPARTIMENTOS acima do limiar patologico
    passa de `FRACAO_LETAL`

Isso da progressao gradual sem saturar, porque a frente leva tempo para
atravessar a cadeia.

O TESTE E O MESMO DE ANTES, E CONTINUA SEM PARAMETRO LIVRE
------------------------------------------------------------
    1. calibra a migracao para o controle dar 143 dias
    2. ajusta k SO no braco NIH profilatico ASO1 (81%)
    3. PREVE a dose unica no dia 120 (observado 55%)

Mesma molecula nos dois bracos. O passo 3 nao tem parametro livre.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  E0  PRE-CONDICAO. Calibra para 143 dias, e a progressao tem que ser
      GRADUAL — a fracao de compartimentos doentes cresce ao longo do tempo,
      nao salta. Se saltar, e o tanque de novo com mais passos.

  E1  previsao da dose tardia com erro < 15 pp
      -> espalhamento espacial era o ingrediente que faltava

  E2  erro >= 15 pp mas MENOR que os -15,3 do bem-misturado
      -> ajuda mas nao basta

  E3  erro >= 15,3 pp
      -> espalhamento tambem nao e a peca. A hipotese cai e o util e ter
         eliminado duas explicacoes com o mesmo protocolo.

MINHA PREVISAO: E2. Espero que ajude — a frente da o gradualismo que falta —
mas que nao chegue nos 15 pp, porque um encadeamento linear e ainda uma
caricatura da rede neural real. Aposto contra E1.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from .prion import BASE
from .prion_validacao import _efeito

N_COMP = 12                  # compartimentos em cadeia
LIMIAR_LOCAL = 55000.0       # carga patologica por compartimento
FRACAO_LETAL = 0.5           # morte quando metade dos compartimentos passa
CONTROLE_NIH = 143.0
TARDIO_CTRL = 157.0
OBS_PROFILATICO = 81.0
OBS_TARDIO = 55.0
DIA_DOSE = 120.0


def simular(p: Dict[str, float], mig: float, k: float = 0.0,
            inicio: float = 0.0, sustentado: bool = True,
            t_max: float = 2500.0) -> Tuple[Optional[float], List[float]]:
    """
    Devolve (tempo ate morte, curva da fracao doente amostrada).
    Estado: [x_0..x_{N-1}, y_0.., z_0..].
    """
    n, d, beta, a, b = p["n"], p["d"], p["beta"], p["a"], p["b"]
    lam0 = p["lam"]
    N = N_COMP

    def campo(t, s):
        x = s[0:N]
        y = s[N:2 * N]
        z = s[2 * N:3 * N]
        lam = lam0 * (1.0 - k * _efeito(t, inicio, sustentado))
        conv = 2 * beta * x * z
        dx = lam - d * x - conv
        dy = conv - a * y - b * n * (n - 1) * z
        dz = b * (y - (2 * n - 1) * z) - a * z
        # migracao de sementes para o vizinho seguinte, em cadeia
        fluxo = mig * z
        dz = dz - fluxo
        dz[1:] = dz[1:] + fluxo[:-1]
        dy = dy - n * fluxo
        dy[1:] = dy[1:] + n * fluxo[:-1]
        return np.concatenate([dx, dy, dz])

    def cruza(t, s):
        y = s[N:2 * N]
        return float((y >= LIMIAR_LOCAL).sum()) / N - FRACAO_LETAL
    cruza.terminal = True
    cruza.direction = 1

    x0 = np.full(N, lam0 / d)
    y0 = np.zeros(N); y0[0] = n
    z0 = np.zeros(N); z0[0] = 1.0
    sol = solve_ivp(campo, (0.0, t_max), np.concatenate([x0, y0, z0]),
                    method="Radau", events=cruza, rtol=1e-8, atol=1e-8,
                    dense_output=True)
    if not sol.success:
        return None, []
    tf = sol.t_events[0][0] if (sol.t_events and len(sol.t_events[0])) else None
    curva = []
    if tf:
        for frac in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
            s = sol.sol(tf * frac)
            curva.append(float((s[N:2 * N] >= LIMIAR_LOCAL).sum()) / N)
    return (float(tf) if tf else None), curva


def calibrar_mig(p: Dict[str, float], alvo: float) -> float:
    lo, hi = 1e-6, 5.0
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        t, _ = simular(p, mid)
        if t is None or t > alvo:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def relatorio() -> None:
    print("=" * 78)
    print("PRION ESPACIAL — a frente que avanca, em vez do tanque que satura")
    print("=" * 78)
    print("PRE-REGISTRO:")
    print("  E0  calibra para 143 d E a progressao e GRADUAL")
    print("  E1  erro < 15 pp  -> espalhamento era a peca que faltava")
    print("  E2  erro < 15,3 pp mas >= 15 -> ajuda, nao basta")
    print("  E3  erro >= 15,3 pp -> espalhamento tambem nao e a peca")
    print("\nMINHA PREVISAO: E2. Aposto contra E1.")
    print(f"\n{N_COMP} compartimentos · morte a {FRACAO_LETAL:.0%} deles doentes")

    p = dict(BASE)
    p["beta"] = 0.0053403       # o mesmo calibrado no bem-misturado (NIH)
    mig_n = calibrar_mig(p, CONTROLE_NIH)
    t_n, curva = simular(p, mig_n)
    print("\n" + "-" * 78)
    print(f"E0 · migracao calibrada = {mig_n:.5g}  ->  controle {t_n:.1f} d "
          f"(alvo {CONTROLE_NIH:.0f})")
    print(f"     fracao doente em 10/25/50/75/90/100% do curso: "
          f"{['%.2f' % c for c in curva]}")
    gradual = len(curva) >= 4 and curva[0] < 0.3 and curva[-1] >= FRACAO_LETAL
    if t_n is None or abs(t_n - CONTROLE_NIH) / CONTROLE_NIH > 0.03:
        print("     E0 REPROVADA — nao calibra.")
        return
    if not gradual:
        print("     E0 REPROVADA — progressao nao e gradual; e o tanque de novo.")
        return
    print("     aprovada — progride gradualmente")

    p_t = dict(p)
    mig_t = calibrar_mig(p_t, TARDIO_CTRL)
    print(f"     coorte tardia: migracao {mig_t:.5g} -> "
          f"{simular(p_t, mig_t)[0]:.1f} d (alvo {TARDIO_CTRL:.0f})")

    # ajusta k no profilatico
    lo, hi = 0.0, 0.995
    for _ in range(40):
        mid = (lo + hi) / 2
        t, _ = simular(p, mig_n, k=mid, inicio=-14.0, sustentado=True)
        e = math.inf if t is None else 100 * (t - CONTROLE_NIH) / CONTROLE_NIH
        if e > OBS_PROFILATICO:
            hi = mid
        else:
            lo = mid
    k = (lo + hi) / 2
    t, _ = simular(p, mig_n, k=k, inicio=-14.0, sustentado=True)
    e_prof = 100 * (t - CONTROLE_NIH) / CONTROLE_NIH if t else math.inf
    print("\n" + "-" * 78)
    print(f"ajuste · k = {k:.1%} reproduz o profilatico ({e_prof:.1f}% vs "
          f"{OBS_PROFILATICO:.0f}%)")

    # PREVE o tardio
    t, _ = simular(p_t, mig_t, k=k, inicio=DIA_DOSE, sustentado=False)
    prev = math.inf if t is None else 100 * (t - TARDIO_CTRL) / TARDIO_CTRL
    err = math.inf if prev == math.inf else prev - OBS_TARDIO
    print(f"PREVISAO · dose unica no dia {DIA_DOSE:.0f}: "
          f"{'nao progride' if prev == math.inf else f'{prev:.1f}%'}  "
          f"(observado {OBS_TARDIO:.0f}%)  erro "
          f"{'—' if err == math.inf else f'{err:+.1f} pp'}")

    print("\n" + "=" * 78)
    if err != math.inf and abs(err) < 15.0:
        print(f"E1 · O ESPALHAMENTO ERA A PECA. Erro {err:+.1f} pp, contra -15,3 do")
        print("     bem-misturado, sem nenhum parametro livre a mais.")
    elif err != math.inf and abs(err) < 15.3:
        print(f"E2 · AJUDA MAS NAO BASTA. Erro {err:+.1f} pp contra -15,3.")
        print("     Espalhamento e parte da resposta; falta mais.")
    else:
        v = "nao progride" if err == math.inf else f"{err:+.1f} pp"
        print(f"E3 · ESPALHAMENTO TAMBEM NAO E A PECA (erro {v}).")
        print("     Duas explicacoes eliminadas com o mesmo protocolo:")
        print("     mecanismo de dano e geometria de propagacao.")


if __name__ == "__main__":
    relatorio()
