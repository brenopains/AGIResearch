"""
novaphys.prion_validacao — o modelo prevê dado que ele nunca viu?
===================================================================

A FRAQUEZA QUE ISTO ATACA
---------------------------
`prion.py` calibra beta contra um periodo de incubacao que EU escolhi. Um
modelo calibrado contra si mesmo nao pode errar, e por isso nao vale nada.
Aqui ele passa a poder errar.

DADO REAL, de Raymond et al., JCI Insight 2019 (PMC6777807)
-------------------------------------------------------------
    regime                       controle   tratado   extensao
    NIH profilatico ASO1          143 d      259 d      81%
    NIH profilatico ASO2          143 d      283 d      98%
    Broad profilatico ASO1        170 d      274 d      61%
    Broad profilatico ASO2        170 d      300 d      76%
    dose unica no dia 120         157 d      244 d      55%

    Farmacocinetica: reducao de Prnp mRNA persiste ~84 dias e volta ao basal
    por volta de 112 dias apos a injecao.

O DESENHO — ajusta em UM, prevê QUATRO
----------------------------------------
    1. calibra `beta` para a incubacao do controle (por coorte)
    2. ajusta UM parametro livre — a fracao de knockdown `k` — para
       reproduzir SO o braco NIH ASO1 (81%)
    3. com esse MESMO k, PREVÊ os outros quatro bracos
    4. compara previsao com observado

O passo 4 nao tem parametro livre. E previsao fora da amostra.

O BRACO QUE DECIDE
--------------------
A dose unica no dia 120 e dada a ~76% do curso da doenca e ainda compra 55%.
Se o modelo disser que tratamento tardio e quase inutil, ele esta errado sobre
a biologia que importa — que e justamente a situacao clinica do ION717, dado a
pacientes ja sintomaticos.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  V0  PRE-CONDICAO. A calibracao tem que reproduzir 143/170/157 dias de
      controle. Se nao reproduz, nada abaixo se le.

  V1  existe k plausivel (entre 20% e 98%) que reproduz o braco de ajuste

  V2  TESTE PRINCIPAL. Com o mesmo k, a previsao da DOSE UNICA TARDIA cai
      dentro de +-15 pontos percentuais dos 55% observados
      -> o modelo tem conteudo preditivo fora da amostra

  V3  se a previsao errar por mais de 15 pontos, o modelo NAO serve para
      responder perguntas de temporizacao, e a analise de dose minima de
      `prion.py` fica restrita ao regime profilatico

  V4  os tres bracos profilaticos restantes servem de checagem adicional

MINHA PREVISAO, escrita antes de rodar: espero V3. Suspeito que o modelo
subestime muito o beneficio tardio, porque nele a carga cresce ate um
equilibrio e reduzir o suprimento tarde deveria mudar pouco o tempo ate o
limiar. Se eu estiver certo, o resultado util e saber ONDE o modelo falha.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from scipy.integrate import solve_ivp

from .prion import BASE, LIMIAR, T_MAX

# ── dado observado
COORTES = {
    "NIH": 143.0,
    "Broad": 170.0,
    "tardio": 157.0,
}
BRACOS = [
    # rotulo,            coorte,   inicio,  sustentado, observado(%)
    ("NIH profil. ASO1", "NIH",     -14.0,  True,   81.0),   # <- braco de AJUSTE
    ("NIH profil. ASO2", "NIH",     -14.0,  True,   98.0),
    ("Broad prof. ASO1", "Broad",   -14.0,  True,   61.0),
    ("Broad prof. ASO2", "Broad",   -14.0,  True,   76.0),
    ("dose unica d120",  "tardio",  120.0,  False,  55.0),
]
AJUSTE = 0                      # indice do braco usado para ajustar k
PERSIST = 84.0                  # dias de efeito pleno apos uma injecao
REBOTE = 112.0                  # dias ate o basal


def _efeito(t: float, inicio: float, sustentado: bool) -> float:
    """Fracao do knockdown ativa no instante t."""
    if t < inicio:
        return 0.0
    if sustentado:
        return 1.0
    dt = t - inicio
    if dt <= PERSIST:
        return 1.0
    if dt >= REBOTE:
        return 0.0
    return 1.0 - (dt - PERSIST) / (REBOTE - PERSIST)


def simular_tratado(p: Dict[str, float], k: float, inicio: float,
                    sustentado: bool, limiar: float,
                    t_max: float = 2000.0) -> Optional[float]:
    n, d, beta, a, b = p["n"], p["d"], p["beta"], p["a"], p["b"]
    lam0 = p["lam"]

    def campo(t, s):
        x, y, z = s
        lam = lam0 * (1.0 - k * _efeito(t, inicio, sustentado))
        return [lam - d * x - 2 * beta * x * z,
                2 * beta * x * z - a * y - b * n * (n - 1) * z,
                b * (y - (2 * n - 1) * z) - a * z]

    def cruza(t, s):
        return s[1] - limiar
    cruza.terminal = True
    cruza.direction = 1

    sol = solve_ivp(campo, (0.0, t_max), [lam0 / d, n * 1.0, 1.0],
                    method="Radau", events=cruza, rtol=1e-9, atol=1e-9)
    if not sol.success:
        return None
    if sol.t_events and len(sol.t_events[0]) > 0:
        return float(sol.t_events[0][0])
    return None


def calibrar_coorte(alvo: float, limiar: float,
                    iters: int = 60) -> Dict[str, float]:
    """Bisseccao em beta para a incubacao do controle daquela coorte."""
    lo, hi = 1e-5, 1.0
    q = dict(BASE)
    for _ in range(iters):
        mid = (lo + hi) / 2
        q["beta"] = mid
        t = simular_tratado(q, 0.0, 0.0, True, limiar)
        if t is None or t > alvo:
            lo = mid
        else:
            hi = mid
    q["beta"] = (lo + hi) / 2
    return q


def extensao(p: Dict[str, float], k: float, inicio: float, sustentado: bool,
             base: float, limiar: float) -> Optional[float]:
    t = simular_tratado(p, k, inicio, sustentado, limiar)
    if t is None:
        return math.inf
    return 100.0 * (t - base) / base


def relatorio() -> None:
    limiar = LIMIAR
    print("=" * 78)
    print("PRION — VALIDACAO: ajusta em um braco, PREVE os outros quatro")
    print("=" * 78)
    print("dado: Raymond et al., JCI Insight 2019 (PMC6777807)")
    print("PRE-REGISTRO:")
    print("  V0  a calibracao reproduz 143/170/157 dias de controle")
    print("  V1  existe k plausivel (20-98%) que reproduz o braco de ajuste")
    print("  V2  com o MESMO k, a dose unica tardia cai a +-15 pp dos 55%")
    print("  V3  se errar por mais de 15 pp, o modelo nao serve para")
    print("      temporizacao e a analise anterior fica restrita ao profilatico")
    print("\nMINHA PREVISAO: V3 — espero que o modelo subestime o beneficio")
    print("tardio. Se eu estiver certo, o util e saber ONDE ele falha.")

    # ── V0
    print("\n" + "-" * 78)
    print("V0 · calibracao por coorte")
    params: Dict[str, Dict[str, float]] = {}
    ok0 = True
    for nome, alvo in COORTES.items():
        q = calibrar_coorte(alvo, limiar)
        t = simular_tratado(q, 0.0, 0.0, True, limiar)
        erro = abs(t - alvo) / alvo if t else 1.0
        params[nome] = q
        print(f"  {nome:<10} alvo {alvo:.0f} d  ->  {t:.1f} d  "
              f"(beta={q['beta']:.5g})  erro {erro:.2%}")
        if erro > 0.02:
            ok0 = False
    if not ok0:
        print("\n  V0 REPROVADA — a calibracao nao reproduz os controles.")
        return
    print("  aprovada")

    # ── V1: ajusta k no braco escolhido
    rot_a, co_a, ini_a, sus_a, obs_a = BRACOS[AJUSTE]
    print("\n" + "-" * 78)
    print(f"V1 · ajustar k SO no braco '{rot_a}' (observado {obs_a:.0f}%)")
    lo, hi = 0.0, 0.995
    p_a, base_a = params[co_a], COORTES[co_a]
    for _ in range(50):
        mid = (lo + hi) / 2
        e = extensao(p_a, mid, ini_a, sus_a, base_a, limiar)
        if e is None or e == math.inf or e > obs_a:
            hi = mid
        else:
            lo = mid
    k = (lo + hi) / 2
    e_a = extensao(p_a, k, ini_a, sus_a, base_a, limiar)
    print(f"  k ajustado = {k:.1%}  ->  extensao {e_a:.1f}% "
          f"(observado {obs_a:.0f}%)")
    if not (0.20 <= k <= 0.98):
        print(f"\n  V1 REPROVADA — k={k:.1%} fora da faixa plausivel 20-98%.")
        print("  O modelo so reproduz o dado com knockdown implausivel.")
        return
    print("  aprovada — knockdown na faixa que ASO de fato atinge")

    # ── V2 / V4: previsao fora da amostra, SEM parametro livre
    print("\n" + "-" * 78)
    print(f"V2/V4 · PREVISAO com o mesmo k = {k:.1%}, zero parametro livre")
    print(f"  {'braco':<20}{'observado':>12}{'previsto':>12}{'erro':>10}")
    print("  " + "-" * 54)
    erros = []
    for i, (rot, co, ini, sus, obs) in enumerate(BRACOS):
        e = extensao(params[co], k, ini, sus, COORTES[co], limiar)
        marca = "  (ajuste)" if i == AJUSTE else ""
        if e == math.inf:
            print(f"  {rot:<20}{obs:>11.0f}%{'nao progride':>12}"
                  f"{'—':>10}{marca}")
            if i != AJUSTE:
                erros.append((rot, math.inf))
            continue
        d = e - obs
        print(f"  {rot:<20}{obs:>11.0f}%{e:>11.1f}%{d:>+9.1f}{marca}")
        if i != AJUSTE:
            erros.append((rot, d))

    # ── leitura
    tardio = [d for r, d in erros if "unica" in r]
    print("\n" + "=" * 78)
    if tardio and tardio[0] != math.inf and abs(tardio[0]) <= 15.0:
        print(f"V2 · O MODELO PREVE FORA DA AMOSTRA. Dose unica tardia errou por")
        print(f"     {tardio[0]:+.1f} pontos percentuais, dentro dos +-15 declarados,")
        print("     com o k ajustado em outro braco e nenhum parametro livre.")
        print("     A analise de dose minima vale tambem para tratamento tardio.")
    else:
        v = "nao progride" if (tardio and tardio[0] == math.inf) else \
            f"{tardio[0]:+.1f} pp" if tardio else "sem dado"
        print(f"V3 · O MODELO NAO PREVE O BRACO TARDIO (erro {v}).")
        print("     Ele nao serve para perguntas de temporizacao, e a analise de")
        print("     dose minima de `prion.py` fica RESTRITA ao regime profilatico.")
        print("     Saber onde ele falha vale mais que a tabela que ele produziu.")
    finitos = [abs(d) for _r, d in erros if d != math.inf]
    if finitos:
        print(f"\n  erro absoluto medio nos {len(finitos)} bracos previstos: "
              f"{sum(finitos)/len(finitos):.1f} pontos percentuais")


if __name__ == "__main__":
    relatorio()
