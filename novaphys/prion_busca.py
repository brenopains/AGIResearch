"""
novaphys.prion_busca — a maquina simbolica procura a lei de dano
==================================================================

POR QUE ISTO USA A NOSSA ARQUITETURA DE VERDADE
-------------------------------------------------
Ate aqui eu escolhi QUATRO leis de dano a mao (carga, fluxo, prpc, moldes) e
as quatro falharam. Escolher a mao e exatamente o que o `novaphys` existe para
substituir: gramatica, busca com orcamento fixo, verificador que nao pode ser
convencido.

Aqui a busca e sobre o espaco de leis `dL/dt = g(x, y, z)`, e o verificador e
o dado publicado: a duracao sintomatica tratada, 53 dias.

O TRUQUE QUE TORNA A BUSCA EXAUSTIVA VIAVEL
---------------------------------------------
Avaliar uma lei parecia exigir integrar as EDOs de novo. Nao exige.

    a trajetoria (x, y, z) NAO depende da lei de dano
    a lei so entra na QUARTA equacao, que nao realimenta as outras

Entao integra-se DUAS vezes — controle e tratado — e depois cada candidata e
so uma quadratura sobre trajetorias ja armazenadas. Milhares de leis em
segundos, em vez de horas.

O k tambem nao precisa ser reajustado por lei: ele e fixado pelo atraso de
INICIO, que e fase 1, independente do dano.

O PROTOCOLO, IDENTICO AO DAS LEIS ESCRITAS A MAO
--------------------------------------------------
    1. beta calibrado para inicio do controle = 142 d
    2. k calibrado para inicio do tratado = 189 d   (uma vez, para todas)
    3. por lei: LETAL = integral de g no controle ao longo de 15 d
    4. por lei: PREVE a duracao tratada; compara com 53 d observados

Nada e ajustado no passo 4. As leis escritas a mao entram na mesma corrida,
para que o resultado seja comparavel.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  B0  PRE-CONDICAO. A quadratura tem que reproduzir os numeros das quatro
      leis escritas a mao (carga 16,1 d · fluxo 19,3 d · prpc 19,3 d ·
      moldes 13,0 d, com o k daquele ajuste). Se nao reproduz, a otimizacao
      esta errada e nada abaixo se le.

  B1  alguma lei descoberta preve a duracao dentro de +-30% (37-69 d)
      -> a busca achou o que eu nao achei a mao, e temos candidata a
         mecanismo

  B2  NENHUMA lei da gramatica inteira preve
      -> resultado muito mais forte que quatro tentativas: o mecanismo que
         falta NAO e funcao do estado (x, y, z). Exige uma variavel que o
         modelo nao acompanha — inflamacao, contagem de neuronios, ou uma
         modificacao persistente. Isso e eliminacao exaustiva, nao palpite.

MINHA PREVISAO: B2. As quatro leis a mao ja cobriam as formas obvias, e o
problema me parece de ESTADO FALTANDO, nao de forma funcional. Mas se a busca
achar algo, e exatamente o tipo de coisa que eu nao acharia a mao.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from .prion import BASE
from .prion_fases import (DUR_CTRL, DUR_TRAT, INICIO_CTRL, LIMIAR,
                          cal_beta, inicio_clinico)
from .prion_validacao import _efeito

INICIO_TRAT = 189.0
DIA_DOSE = 120.0
N_AMOSTRAS = 6000          # densidade da trajetoria armazenada
DUR_MAX = 600.0            # horizonte da fase sintomatica tratada
PROF = 2                   # profundidade da gramatica
OPS = ("+", "-", "*", "/")
FOLHAS = ("x", "y", "z", "1")


def trajetoria(p: Dict[str, float], k: float, inicio: float,
               sustentado: bool, dur: float
               ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Devolve (tempos, estados) da fase sintomatica: do inicio clinico ate
    inicio + dur, amostrado denso. (x, y, z) nao dependem da lei de dano.
    """
    ti = inicio_clinico(p, k, inicio, sustentado)
    if ti is None:
        return None
    n, d, beta, a, b = p["n"], p["d"], p["beta"], p["a"], p["b"]
    lam0 = p["lam"]

    def campo(t, s):
        x, y, z = s
        lam = lam0 * (1.0 - k * _efeito(t, inicio, sustentado))
        return [lam - d * x - 2 * beta * x * z,
                2 * beta * x * z - a * y - b * n * (n - 1) * z,
                b * (y - (2 * n - 1) * z) - a * z]

    pre = solve_ivp(campo, (0.0, ti), [lam0 / d, n * 1.0, 1.0],
                    method="Radau", rtol=1e-9, atol=1e-9)
    sol = solve_ivp(campo, (ti, ti + dur), pre.y[:, -1], method="Radau",
                    rtol=1e-9, atol=1e-9, dense_output=True)
    if not sol.success:
        return None
    ts = np.linspace(ti, ti + dur, N_AMOSTRAS)
    return ts, sol.sol(ts)


# ══════════════════════════════════════════════ gramatica
def expressoes(prof: int) -> List[str]:
    """Todas as formas ate `prof`, deduplicadas por string canonica."""
    nivel = list(FOLHAS)
    todas = set(nivel)
    atual = nivel
    for _ in range(prof):
        novo = []
        for a in atual:
            for b in nivel:
                for o in OPS:
                    novo.append(f"({a}{o}{b})")
        atual = novo
        todas.update(novo)
    return sorted(todas)


def avaliar(expr: str, S: np.ndarray) -> Optional[np.ndarray]:
    """Avalia a expressao sobre a trajetoria inteira, vetorizado."""
    x, y, z = S[0], S[1], S[2]
    um = np.ones_like(x)
    try:
        with np.errstate(all="ignore"):
            v = eval(expr, {"__builtins__": {}},
                     {"x": x, "y": y, "z": z, "1": um})
    except Exception:
        return None
    v = np.asarray(v, dtype=float)
    if v.shape != x.shape or not np.all(np.isfinite(v)):
        return None
    if np.any(v < 0):
        return None                     # dano nao pode ser negativo
    if v.max() <= 0:
        return None                     # dano nulo nao produz morte
    return v


def duracao_prevista(g_ctrl: np.ndarray, t_ctrl: np.ndarray,
                     g_trat: np.ndarray, t_trat: np.ndarray,
                     dur_ctrl: float) -> Optional[float]:
    """
    LETAL = integral no controle ao longo de `dur_ctrl`.
    Duracao prevista = quando a integral no tratado alcanca LETAL.
    """
    dt_c = t_ctrl[1] - t_ctrl[0]
    acum_c = np.cumsum(g_ctrl) * dt_c
    i = int(dur_ctrl / dt_c)
    if i >= len(acum_c):
        return None
    letal = acum_c[i]
    if not (letal > 0 and math.isfinite(letal)):
        return None
    dt_t = t_trat[1] - t_trat[0]
    acum_t = np.cumsum(g_trat) * dt_t
    idx = np.searchsorted(acum_t, letal)
    if idx >= len(acum_t):
        return None                     # nao morre no horizonte
    return float(idx * dt_t)


def relatorio() -> None:
    print("=" * 78)
    print("PRION — a maquina simbolica procura a lei de dano")
    print("=" * 78)
    print("PRE-REGISTRO:")
    print("  B0  a quadratura reproduz as quatro leis escritas a mao")
    print("  B1  alguma lei descoberta preve 53 d dentro de +-30% (37-69 d)")
    print("  B2  NENHUMA preve -> o mecanismo que falta nao e funcao de")
    print("      (x,y,z); exige uma variavel que o modelo nao acompanha")
    print("\nMINHA PREVISAO: B2. Aposto que falta ESTADO, nao forma funcional.")

    b = cal_beta(INICIO_CTRL)
    p = dict(BASE); p["beta"] = b
    # k pelo atraso de inicio — uma vez, vale para todas as leis
    lo, hi = 0.0, 0.995
    for _ in range(45):
        m = (lo + hi) / 2
        ti = inicio_clinico(p, k=m, inicio=DIA_DOSE, sustentado=False)
        if ti is None or ti > INICIO_TRAT:
            hi = m
        else:
            lo = m
    k = (lo + hi) / 2
    print(f"\nbeta {b:.6g} · inicio controle "
          f"{inicio_clinico(p):.1f} d · k {k:.1%} -> inicio tratado "
          f"{inicio_clinico(p, k=k, inicio=DIA_DOSE, sustentado=False):.1f} d")

    tc = trajetoria(p, 0.0, 0.0, True, DUR_CTRL * 3)
    tt = trajetoria(p, k, DIA_DOSE, False, DUR_MAX)
    if tc is None or tt is None:
        print("trajetorias falharam")
        return
    t_c, S_c = tc
    t_t, S_t = tt

    # ── B0
    MAO = {"carga": "y", "fluxo": "(x*z)", "prpc": "(x*z)", "moldes": "z"}
    ESPERADO = {"carga": 16.1, "fluxo": 19.3, "prpc": 19.3, "moldes": 13.0}
    print("\n" + "-" * 78)
    print("B0 · reproducao das leis escritas a mao")
    ok0 = True
    for nome, ex in MAO.items():
        gc = avaliar(ex, S_c); gt = avaliar(ex, S_t)
        d = duracao_prevista(gc, t_c, gt, t_t, DUR_CTRL) if (
            gc is not None and gt is not None) else None
        alvo = ESPERADO[nome]
        bate = d is not None and abs(d - alvo) <= 1.5
        print(f"     {nome:<8} {ex:<8} -> {d if d is None else round(d, 1)} d "
              f"(esperado {alvo}) {'ok' if bate else 'DIVERGE'}")
        if not bate:
            ok0 = False
    if not ok0:
        print("\n     B0 REPROVADA — a quadratura nao reproduz. Nao leio o resto.")
        return
    print("     aprovada")

    # ── busca exaustiva
    exprs = expressoes(PROF)
    print("\n" + "-" * 78)
    print(f"BUSCA · {len(exprs)} leis na gramatica, profundidade {PROF}")
    print(f"        folhas {FOLHAS} · operadores {OPS}")
    achados = []
    validas = 0
    for ex in exprs:
        gc = avaliar(ex, S_c)
        if gc is None:
            continue
        gt = avaliar(ex, S_t)
        if gt is None:
            continue
        d = duracao_prevista(gc, t_c, gt, t_t, DUR_CTRL)
        if d is None:
            continue
        validas += 1
        achados.append((abs(d - DUR_TRAT), d, ex))
    achados.sort()
    print(f"        {validas} produzem morte no horizonte e sao avaliaveis")

    print("\n" + "-" * 78)
    print("AS 12 MELHORES DE TODA A GRAMATICA")
    print(f"{'lei':<26}{'duracao prevista':>18}{'erro vs 53 d':>15}")
    print("-" * 78)
    for _e, d, ex in achados[:12]:
        err = 100 * (d - DUR_TRAT) / DUR_TRAT
        print(f"{ex:<26}{d:>15.1f} d{err:>+14.0f}%")

    print("\n" + "=" * 78)
    if achados and abs(achados[0][1] - DUR_TRAT) / DUR_TRAT <= 0.30:
        _e, d, ex = achados[0]
        print(f"B1 · A BUSCA ACHOU. `{ex}` preve {d:.1f} d contra 53 d")
        print(f"     observados, erro {100*(d-DUR_TRAT)/DUR_TRAT:+.0f}%, com k")
        print("     ajustado no atraso de inicio e nada ajustado aqui.")
    else:
        melhor = achados[0] if achados else None
        m = f"{melhor[1]:.1f} d ({100*(melhor[1]-DUR_TRAT)/DUR_TRAT:+.0f}%)" \
            if melhor else "nenhuma"
        print(f"B2 · NENHUMA LEI DA GRAMATICA PREVE. Melhor: {m}.")
        print(f"     {validas} leis avaliadas exaustivamente sobre (x, y, z).")
        print()
        print("     Isso e mais forte que quatro tentativas a mao: o mecanismo")
        print("     que falta NAO e funcao do estado que o modelo acompanha.")
        print("     Ele exige uma variavel nova — algo que MUDE de forma")
        print("     persistente com o tratamento e nao volte quando o farmaco")
        print("     sai. Contagem de neuronios, inflamacao, ou uma modificacao")
        print("     duradoura do tecido.")


if __name__ == "__main__":
    relatorio()
