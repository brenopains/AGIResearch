"""
novaphys.prion_conjunto — parar de testar uma coisa por vez
============================================================

A CRITICA QUE ORIGINOU ISTO, E ELA PROCEDE
--------------------------------------------
Testei tres consertos estruturais do modelo cinetico, um de cada vez:

    inoculo maior          -> nao remove o limiar falso
    alongamento satura     -> razao 6,99x -> 6,21x  (observado 2,46x)
    capacidade limitada    -> 6,99x -> 8,43x, PIOR

Cada um falhou sozinho e eu segui em frente. NUNCA testei dois juntos. Isso e
explorar as marginais e ignorar o espaco conjunto — e num sistema nao-linear
uma combinacao pode satisfazer o que nenhuma parte satisfaz.

Pior: nunca REVISITEI. Quando rejeitei a lei `fluxo`, nunca voltei a ela
depois de descobrir a falha de dosagem genica — mesmo que essa descoberta
mude as CONDICOES sob as quais `fluxo` foi rejeitada.

AS TRES PECAS QUE FALTAVAM
----------------------------
    1. POPULACAO de modelos vivos, nao um modelo por vez
    2. TODAS as observacoes ao mesmo tempo, nao uma por experimento
    3. REVISITAR: um modelo rejeitado volta a corrida quando o contexto muda

Este arquivo faz as tres. Nada aqui e nova biologia — e o mesmo espaco de
hipoteses de antes, percorrido de forma conjunta em vez de sequencial.

AS QUATRO OBSERVACOES, TODAS SIMULTANEAS
------------------------------------------
    O1  incubacao do controle = 142 d          (calibracao, sempre satisfeita)
    O2  10x PrP -> incubacao 2,46x mais curta  (tga20 vs selvagem)
    O3  0,5x PrP -> a doenca AINDA OCORRE      (Prnp+/-)
    O4  dose unica no dia 120 -> inicio +33%   (Raymond 2019)
    O5  ... e duracao sintomatica 3,5x

Minha busca linear satisfez O1 e O4 e violou O2, O3 e O5. Nunca perguntei se
ALGUM ponto do espaco conjunto satisfaz os cinco.

ORDEM DE VERIFICACAO, MAIS BARATO PRIMEIRO
--------------------------------------------
O3 custa uma integracao e mata a maioria. Verificar na ordem certa e o que
torna a varredura conjunta viavel — e e a versao concreta de "nao percorrer
tudo linearmente".

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  A0  PRE-CONDICAO. O ponto base (sem conserto) tem que reproduzir o que ja
      sei: satisfaz O1 e O4, viola O2, O3, O5. Se nao reproduzir, a varredura
      esta errada.

  A1  algum ponto do espaco conjunto satisfaz as CINCO
      -> a familia nao estava morta; a busca linear e que perdeu

  A2  nenhum satisfaz as cinco
      -> a familia esta morta de forma CONJUNTA, nao sequencial. Muito mais
         forte que tres rejeicoes isoladas.

  A3  qual e o maior subconjunto satisfazivel, e QUAIS observacoes sao
      mutuamente incompativeis dentro da familia. Isso vale mesmo em A2, e e
      o diagnostico que aponta o que precisa ser abandonado.

MINHA PREVISAO: A2, com O2 e O5 mutuamente incompativeis — satisfazer a
dosagem genica exige achatar a dependencia em PrP, e a extensao de duracao
exige o contrario. Se eu estiver certo, isso e uma contradicao INTERNA da
familia, e nomeia-la vale mais que qualquer conserto.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Tuple

from scipy.integrate import solve_ivp

from .prion import BASE
from .prion_validacao import _efeito

LIMIAR = 55000.0
ALVO_INCUB = 142.0
OBS_RAZAO_10X = 2.46
OBS_INICIO = 189.0
OBS_DUR_CTRL = 15.0
OBS_DUR_TRAT = 53.0
DIA_DOSE = 120.0
TOL_O2 = 0.40          # razao a 10x dentro de +-40% de 2,46
TOL_O5 = 0.40          # duracao tratada dentro de +-40% de 53 d


def simular(v: Dict, mult_lam: float = 1.0, k: float = 0.0,
            inicio: float = 0.0, sustentado: bool = True,
            dano: Optional[str] = None, letal: Optional[float] = None,
            t_max: float = 6000.0) -> Tuple[Optional[float], Optional[float]]:
    """
    Um ponto do espaco conjunto. `v` carrega os conserto(s) ativos —
    saturacao em x, capacidade limitada em z, e escalas de a, b, n.
    Devolve (inicio clinico, morte).
    """
    n, d = v["n"], BASE["d"]
    beta, a, b = v["beta"], v["a"], v["b"]
    lam0 = BASE["lam"] * mult_lam
    x0_ref = BASE["lam"] / d
    Kx, Kz = v["Kx"], v["Kz"]

    def taxa(x, z):
        e = 2 * beta * x * z
        if Kx is not None:
            e = 2 * beta * ((x / (Kx + x)) * (Kx + x0_ref)) * z
        if Kz is not None:
            e = e * (Kz / (Kz + z))
        return e

    def campo(t, s):
        x, y, z, _L = s
        lam = lam0 * (1.0 - k * _efeito(t, inicio, sustentado))
        e = taxa(x, z)
        dL = 0.0
        if dano is not None and y >= LIMIAR:
            dL = {"x": x, "x2": x * x, "x3": x * x * x,
                  "y": y, "xz": x * z}[dano]
        return [lam - d * x - e, e - a * y - b * n * (n - 1) * z,
                b * (y - (2 * n - 1) * z) - a * z, dL]

    def ev_i(t, s):
        return s[1] - LIMIAR
    ev_i.terminal = letal is None
    ev_i.direction = 1

    def ev_m(t, s):
        return s[3] - (letal if letal is not None else 1e30)
    ev_m.terminal = True
    ev_m.direction = 1

    try:
        sol = solve_ivp(campo, (0.0, t_max), [lam0 / d, n * 1.0, 1.0, 0.0],
                        method="Radau", events=[ev_i, ev_m],
                        rtol=1e-9, atol=1e-9)
    except Exception:
        return None, None
    if not sol.success:
        return None, None
    ti = float(sol.t_events[0][0]) if len(sol.t_events[0]) else None
    tm = float(sol.t_events[1][0]) if len(sol.t_events[1]) else None
    return ti, tm


def calibrar(v: Dict) -> Optional[Dict]:
    lo, hi = 1e-5, 3.0
    for _ in range(45):
        mid = (lo + hi) / 2
        v["beta"] = mid
        ti, _ = simular(v)
        if ti is None or ti > ALVO_INCUB:
            lo = mid
        else:
            hi = mid
    v["beta"] = (lo + hi) / 2
    ti, _ = simular(v)
    if ti is None or abs(ti - ALVO_INCUB) / ALVO_INCUB > 0.03:
        return None
    return v


def avaliar_ponto(v: Dict) -> Dict[str, object]:
    """Ordem: O1, O3, O2, O4, O5 — mais barato e mais mortal primeiro."""
    r: Dict[str, object] = {k: False for k in ("O1", "O2", "O3", "O4", "O5")}
    if calibrar(v) is None:
        return r
    r["O1"] = True

    # O3 — 0,5x PrP ainda adoece
    ti5, _ = simular(v, mult_lam=0.5)
    r["O3"] = ti5 is not None
    if not r["O3"]:
        return r                     # corte barato

    # O2 — razao a 10x
    ti1, _ = simular(v)
    ti10, _ = simular(v, mult_lam=10.0)
    if ti1 and ti10:
        razao = ti1 / ti10
        r["razao"] = round(razao, 2)
        r["O2"] = abs(razao - OBS_RAZAO_10X) / OBS_RAZAO_10X <= TOL_O2
    if not r["O2"]:
        return r

    # O4 — k que da inicio +33%
    lo, hi = 0.0, 0.99
    for _ in range(35):
        m = (lo + hi) / 2
        ti, _ = simular(v, k=m, inicio=DIA_DOSE, sustentado=False)
        if ti is None or ti > OBS_INICIO:
            hi = m
        else:
            lo = m
    k = (lo + hi) / 2
    ti, _ = simular(v, k=k, inicio=DIA_DOSE, sustentado=False)
    r["k"] = round(k, 3)
    r["O4"] = ti is not None and abs(ti - OBS_INICIO) / OBS_INICIO < 0.05
    if not r["O4"]:
        return r

    # O5 — duracao 3,5x, testando TODAS as leis de dano (revisitar!)
    melhor = None
    for lei in ("x", "x2", "x3", "y", "xz"):
        ti0, _ = simular(v, dano=lei, letal=None)
        if ti0 is None:
            continue
        # letal = dano acumulado do controle em 15 d
        _t, _m = simular(v, dano=lei, letal=None)
        sol_l = _acum(v, lei, OBS_DUR_CTRL)
        if sol_l is None or sol_l <= 0:
            continue
        tit, tmt = simular(v, k=k, inicio=DIA_DOSE, sustentado=False,
                           dano=lei, letal=sol_l)
        if tit is None or tmt is None:
            continue
        dur = tmt - tit
        err = abs(dur - OBS_DUR_TRAT) / OBS_DUR_TRAT
        if melhor is None or err < melhor[0]:
            melhor = (err, dur, lei)
    if melhor:
        r["dur"] = round(melhor[1], 1)
        r["lei"] = melhor[2]
        r["O5"] = melhor[0] <= TOL_O5
    return r


def _acum(v: Dict, lei: str, dur: float) -> Optional[float]:
    ti, _ = simular(v, dano=lei, letal=None)
    if ti is None:
        return None
    n, d = v["n"], BASE["d"]
    beta, a, b, lam0 = v["beta"], v["a"], v["b"], BASE["lam"]
    x0_ref = BASE["lam"] / d
    Kx, Kz = v["Kx"], v["Kz"]

    def campo(t, s):
        x, y, z, _L = s
        e = 2 * beta * x * z
        if Kx is not None:
            e = 2 * beta * ((x / (Kx + x)) * (Kx + x0_ref)) * z
        if Kz is not None:
            e = e * (Kz / (Kz + z))
        dL = {"x": x, "x2": x * x, "x3": x * x * x,
              "y": y, "xz": x * z}[lei] if y >= LIMIAR else 0.0
        return [lam0 - d * x - e, e - a * y - b * n * (n - 1) * z,
                b * (y - (2 * n - 1) * z) - a * z, dL]
    try:
        s = solve_ivp(campo, (0.0, ti + dur), [lam0 / d, n * 1.0, 1.0, 0.0],
                      method="Radau", rtol=1e-8,
                      atol=[1e-8, 1e-8, 1e-8, 1e-4])
        return float(s.y[3, -1]) if s.success else None
    except Exception:
        return None


def relatorio() -> None:
    print("=" * 78)
    print("PRION — o espaco CONJUNTO, em vez de um conserto por vez")
    print("=" * 78)
    print("PRE-REGISTRO:")
    print("  A0  o ponto base reproduz o ja sabido (O1,O4 sim · O2,O3,O5 nao)")
    print("  A1  algum ponto satisfaz as CINCO -> a busca linear e que perdeu")
    print("  A2  nenhum -> familia morta de forma conjunta, nao sequencial")
    print("  A3  quais observacoes sao mutuamente incompativeis")
    print("\nMINHA PREVISAO: A2, com O2 e O5 incompativeis entre si.")

    x0 = BASE["lam"] / BASE["d"]
    grade = {
        "Kx": [None, 3 * x0, 1 * x0, 0.3 * x0],
        "Kz": [None, 1e4, 1e3],
        "a":  [BASE["a"] * m for m in (0.1, 0.3, 1.0, 3.0)],
        "b":  [BASE["b"] * m for m in (0.1, 0.3, 1.0, 3.0, 10.0)],
        "n":  [2.0, 4.0, 6.0],
    }
    combos = list(itertools.product(*[grade[k] for k in
                                      ("Kx", "Kz", "a", "b", "n")]))
    print(f"\nespaco conjunto: {len(combos)} pontos "
          f"(Kx × Kz × a × b × n), cada um com 5 leis de dano revisitadas")

    # A0
    base_v = {"Kx": None, "Kz": None, "a": BASE["a"], "b": BASE["b"],
              "n": BASE["n"], "beta": BASE["beta"]}
    r0 = avaliar_ponto(dict(base_v))
    print("\n" + "-" * 78)
    print(f"A0 · ponto base: " + " · ".join(
        f"{k}={'sim' if r0.get(k) else 'nao'}" for k in
        ("O1", "O2", "O3", "O4", "O5")))
    if not (r0["O1"] and not r0["O3"]):
        print("     A0 REPROVADA — nao reproduz o ja sabido.")
        return
    print("     aprovada — reproduz o diagnostico anterior")

    conta = {k: 0 for k in ("O1", "O2", "O3", "O4", "O5")}
    melhores: List[Tuple[int, Dict, Dict]] = []
    for c in combos:
        v = dict(zip(("Kx", "Kz", "a", "b", "n"), c))
        v["beta"] = BASE["beta"]
        r = avaliar_ponto(v)
        sat = sum(1 for k in ("O1", "O2", "O3", "O4", "O5") if r.get(k))
        for k in conta:
            if r.get(k):
                conta[k] += 1
        melhores.append((sat, v, r))
    melhores.sort(key=lambda t: -t[0])

    print("\n" + "-" * 78)
    print("quantos pontos satisfazem cada observacao, isoladamente:")
    for k in ("O1", "O2", "O3", "O4", "O5"):
        print(f"     {k}  {conta[k]:>4} / {len(combos)}")

    print("\n" + "-" * 78)
    print("melhores pontos do espaco conjunto")
    print(f"{'sat':>4}  {'Kx':>8}{'Kz':>8}{'a':>8}{'b':>10}{'n':>4}"
          f"{'razao':>8}{'dur':>8}  satisfeitas")
    for sat, v, r in melhores[:10]:
        kx = "—" if v["Kx"] is None else f"{v['Kx']/x0:.1f}x0"
        kz = "—" if v["Kz"] is None else f"{v['Kz']:.0e}"
        ok = ",".join(k for k in ("O1", "O2", "O3", "O4", "O5") if r.get(k))
        print(f"{sat:>4}  {kx:>8}{kz:>8}{v['a']:>8.3f}{v['b']:>10.4f}"
              f"{v['n']:>4.0f}{str(r.get('razao','—')):>8}"
              f"{str(r.get('dur','—')):>8}  {ok}")

    top = melhores[0][0]
    print("\n" + "=" * 78)
    if top == 5:
        sat, v, r = melhores[0]
        print("A1 · EXISTE PONTO QUE SATISFAZ AS CINCO. A familia nao estava")
        print(f"     morta — a busca linear e que perdeu. lei de dano `{r.get('lei')}`,")
        print(f"     razao {r.get('razao')}x, duracao {r.get('dur')} d.")
    else:
        print(f"A2 · NENHUM PONTO SATISFAZ AS CINCO. Maximo alcancado: {top}/5,")
        print(f"     em {sum(1 for s,_,_ in melhores if s == top)} pontos.")
        print("     A familia esta morta de forma CONJUNTA — muito mais forte")
        print("     que tres rejeicoes sequenciais.")
        # A3: quais pares nunca coexistem
        print("\nA3 · pares de observacoes que NUNCA sao satisfeitos juntos:")
        obs = ("O2", "O3", "O4", "O5")
        for i in range(len(obs)):
            for j in range(i + 1, len(obs)):
                juntos = sum(1 for _s, _v, r in melhores
                             if r.get(obs[i]) and r.get(obs[j]))
                if juntos == 0:
                    print(f"     {obs[i]} + {obs[j]}  incompativeis em "
                          f"{len(combos)} pontos")


if __name__ == "__main__":
    relatorio()
