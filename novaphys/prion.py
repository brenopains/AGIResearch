"""
novaphys.prion — onde esta a alavanca terapeutica na replicacao de prions
==========================================================================

O QUE ISTO E, DITO ANTES DE QUALQUER NUMERO
---------------------------------------------
Isto NAO cura Creutzfeldt-Jakob. Isto e uma analise de sensibilidade e de
COMBINACAO sobre um modelo cinetico publicado, executada com a disciplina que
construimos hoje: intervencao em vez de observacao, orcamento casado, portao
de pre-condicao, e verificacao antes de leitura.

O que ela pode produzir de util e uma coisa so, e ela e checavel:

    a ORDEM de alavancagem entre alvos terapeuticos, e se existe combinacao
    que rende mais que a soma das partes, a DOSE TOTAL CASADA

Ordem de alvo unico e analise padrao. O que nao e padrao — e e exatamente o
que a nossa maquina faz — e varrer exaustivamente as COMBINACOES com
orcamento fixo, porque o sistema e nao-linear e sinergia nao e obvia.

O MODELO
---------
Polimerizacao nucleada, Masel, Jansen & Nowak (1999), forma reduzida em
momentos. PrP-sen (monomero) alonga polimeros de PrP-res; polimeros se
fragmentam criando novas pontas; abaixo de n subunidades o polimero e
instavel e dissolve.

    dx/dt = lam - d*x - 2*beta*x*z
    dy/dt = 2*beta*x*z - a*y - b*n*(n-1)*z
    dz/dt = b*(y - (2*n-1)*z) - a*z

    x  PrP-sen (monomero normal)
    y  massa total em polimeros (PrP-res)
    z  numero de polimeros (pontas de crescimento)

    lam   producao de PrP-C          <- alvo dos ASO (ION717)
    d     degradacao de PrP-C
    beta  alongamento               <- alvo de anti-agregantes
    a     depuracao de polimeros    <- alvo de indutores de autofagia
    b     fragmentacao              <- nao ha droga obvia; incluido porque o
                                       efeito e NAO-MONOTONICO e isso importa
    n     tamanho minimo estavel

OS ALVOS, E A CORRESPONDENCIA CLINICA
---------------------------------------
    reduzir lam    ASO anti-PRNP. ION717 (Ionis) esta em fase 1/2a
                   (NCT06153966), iniciada no fim de 2024
    reduzir beta   compostos que bloqueiam a conversao/alongamento
    aumentar a     aumentar depuracao de agregados
    reduzir b      reduzir fragmentacao — cuidado, pode PIORAR

DOSE CASADA
------------
Comparar "reduzir lam em 50%" com "reduzir beta em 50%" so e honesto se a
"dose" for a mesma. Aqui dose = mesma fracao de mudanca no parametro. Nao e
dose farmacologica — e a unica normalizacao defensavel sem dados de PK, e
esta declarada como limitacao, nao escondida.

Para COMBINACOES o orcamento e repartido: dois alvos com meia dose cada,
contra um alvo com dose inteira. Sem isso, combinar sempre "vence" por usar
mais intervencao — e o mesmo erro que `gargalo.py` pegou quando campo maior
vencia so por ter mais busca.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  P0  PRE-CONDICAO A. Sem tratamento, o modelo tem que atingir o limiar
      patologico em tempo finito. Se nao atinge, nao ha o que atrasar.

  P1  PRE-CONDICAO B. O efeito de `b` (fragmentacao) tem que ser
      NAO-MONOTONICO — e uma propriedade conhecida do modelo. Se o meu codigo
      nao a reproduz, ele esta errado e nenhum numero abaixo se le.

  P2  ordem de alavancagem entre alvos unicos, dose casada

  P3  existe combinacao que, a DOSE TOTAL CASADA, atrasa mais que o melhor
      alvo unico? Sinergia real, nao soma de doses.

  P4  ROBUSTEZ. A ordem se mantem quando os parametros variam dentro da
      incerteza publicada? Se a ordem inverte dentro da incerteza, ela nao
      e um resultado — e uma escolha de parametro.

MINHA PREVISAO, escrita antes: espero que reduzir `lam` domine, porque e o
que a literatura persegue e porque o substrato limita a autocatalise. Espero
que `b` mostre a nao-monotonicidade. Sobre P3 nao tenho intuicao, e e por
isso que vale rodar.
"""

from __future__ import annotations

import argparse
import itertools
import math
from typing import Dict, List, Optional, Sequence, Tuple

# PARAMETROS CALIBRADOS, NAO CITADOS.
# Minha primeira tentativa escreveu numeros "da literatura" que sao
# internamente inconsistentes: davam e*b = 4,4e-9 contra a^2 = 1,6e-3, seis
# ordens de grandeza abaixo do limiar de decolagem, e o portao P0 reprovou.
# Erro meu de unidade, e o portao existia para pegar isso.
#
# A correcao honesta nao e chutar de novo. E fixar o que e bem restringido
# pela biologia e CALIBRAR o resto contra um observavel:
#
#   a    0,04/dia   depuracao de PrP-res, lenta (meia-vida ~17 dias)
#   n    6          tamanho minimo estavel
#   x*   lam/d      PrP-C em equilibrio, escala arbitraria
#   b    livre      fragmentacao, varrida em P1 e P4
#   beta CALIBRADO  para o tempo ate o limiar bater o alvo de incubacao
#
# A escala de concentracao e arbitraria, entao so as combinacoes
# adimensionais importam — e sao essas que a calibracao fixa.
INCUBACAO_ALVO = 200.0     # dias sem tratamento, ordem de grandeza murina

BASE = {
    "lam": 4400.0,
    "d": 4.0,
    "beta": 1.0e-3,    # sobrescrito por calibrar()
    "a": 0.04,
    "b": 1.0e-3,
    "n": 6.0,
}


def calibrar(p, alvo=INCUBACAO_ALVO, lo=1e-6, hi=1.0, iters=60):
    """
    Bissecao em beta ate o tempo sem tratamento bater o alvo. Nao ha ajuste
    para a direcao de nenhuma conclusao: o alvo e o periodo de incubacao, que
    e observavel, e e fixado ANTES de qualquer analise de intervencao.
    """
    q = dict(p)
    for _ in range(iters):
        mid = (lo + hi) / 2
        q["beta"] = mid
        t = simular(q)
        if t is None or t > alvo:
            lo = mid            # devagar demais: aumentar alongamento
        else:
            hi = mid
    q["beta"] = (lo + hi) / 2
    return q
# A massa de polimero e limitada por producao/depuracao: y_max ~ lam/a.
# Com os parametros base isso da 110.000. Meu primeiro LIMIAR era 2e6 — 18x
# ACIMA do maximo atingivel, e por isso "nunca atinge" com qualquer beta.
# O limiar patologico agora e fracao do atingivel, e fica FIXO sob tratamento
# (a carga que o tecido tolera nao muda porque se deu a droga).
LIMIAR = 0.5 * 4400.0 / 0.04      # = 55.000
T_MAX = 3000.0         # dias
DT = 0.02

ALVOS = {
    "lam-  (ASO anti-PRNP)":      ("lam", -1),
    "beta- (anti-alongamento)":   ("beta", -1),
    "a+    (depuracao)":          ("a", +1),
    "b-    (anti-fragmentacao)":  ("b", -1),
    "d+    (degradar PrP-C)":     ("d", +1),
}


def _campo(s: Tuple[float, float, float], p: Dict[str, float]):
    x, y, z = s
    n = p["n"]
    dx = p["lam"] - p["d"] * x - 2 * p["beta"] * x * z
    dy = 2 * p["beta"] * x * z - p["a"] * y - p["b"] * n * (n - 1) * z
    dz = p["b"] * (y - (2 * n - 1) * z) - p["a"] * z
    return dx, dy, dz


def simular(p: Dict[str, float], t_max: float = T_MAX, dt: float = DT,
            limiar: Optional[float] = None, rtol: float = 1e-9,
            atol: float = 1e-9) -> Optional[float]:
    """
    Integrador RIGIDO com controle de erro (Radau), e evento terminal no
    cruzamento do limiar.

    RETRATACAO: a primeira versao usava RK4 de passo fixo com dt=0.02. O
    sistema e rigido — `d`=4/dia contra `a`=0,04/dia, 100x de separacao de
    escalas — e a resposta NAO estava convergida:

        dt=0.02  -> 200 dias        dt=0.01  -> nunca atinge
        dt=0.05  ->  57 dias        dt=0.005 -> nunca atinge
        dt=0.1   ->  42 dias        dt=0.001 -> nunca atinge

    O valor de 200 dias era erro de integracao puro, e a calibracao bissectou
    sobre esse artefato. Toda a analise construida em cima foi retirada.

    `dt` fica na assinatura so para compatibilidade das chamadas antigas; o
    passo agora e adaptativo e `dt` e ignorado.
    """
    from scipy.integrate import solve_ivp

    # BUG CORRIGIDO: `limiar=LIMIAR` como default era avaliado na DEFINICAO
    # da funcao. Reatribuir o global do modulo nao mudava o default, e a
    # funcao seguia usando o valor antigo. Resolvido em tempo de chamada.
    if limiar is None:
        limiar = LIMIAR

    n = p["n"]
    lam, d, beta, a, b = p["lam"], p["d"], p["beta"], p["a"], p["b"]

    def campo(_t, s):
        x, y, z = s
        return [lam - d * x - 2 * beta * x * z,
                2 * beta * x * z - a * y - b * n * (n - 1) * z,
                b * (y - (2 * n - 1) * z) - a * z]

    def cruza(_t, s):
        return s[1] - limiar
    cruza.terminal = True
    cruza.direction = 1

    s0 = [lam / d, n * 1.0, 1.0]
    try:
        sol = solve_ivp(campo, (0.0, t_max), s0, method="Radau",
                        events=cruza, rtol=rtol, atol=atol, dense_output=False)
    except Exception:
        return None
    if not sol.success:
        return None
    if sol.t_events and len(sol.t_events[0]) > 0:
        return float(sol.t_events[0][0])
    return None


def aplicar(p: Dict[str, float], alvo: str, sinal: int,
            dose: float) -> Dict[str, float]:
    """
    dose em [0,1). sinal -1 reduz o parametro por essa fracao; +1 aumenta
    pelo fator equivalente, para que "meia dose" signifique a mesma coisa
    nos dois sentidos.
    """
    q = dict(p)
    if sinal < 0:
        q[alvo] = p[alvo] * (1.0 - dose)
    else:
        q[alvo] = p[alvo] / max(1e-9, (1.0 - dose))
    return q


def atraso(p: Dict[str, float], base_t: float) -> float:
    """Dias ganhos. Infinito -> a doenca nao progride no horizonte."""
    t = simular(p)
    return math.inf if t is None else t - base_t


def relatorio(dose: float = 0.5) -> None:
    print("=" * 78)
    print("PRION — onde esta a alavanca, em modelo cinetico publicado")
    print("=" * 78)
    print("Isto NAO e uma cura. E analise de sensibilidade e combinacao sobre")
    print("polimerizacao nucleada (Masel, Jansen & Nowak 1999), com dose")
    print("casada e portoes de pre-condicao.")
    print()
    print("PRE-REGISTRO:")
    print("  P0  sem tratamento o limiar e atingido em tempo finito")
    print("  P1  o efeito de `b` e NAO-MONOTONICO (propriedade conhecida)")
    print("  P2  ordem entre alvos unicos, dose casada")
    print("  P3  ha combinacao que vence o melhor alvo unico a dose TOTAL igual?")
    print("  P4  a ordem sobrevive a incerteza dos parametros?")
    print(f"\ndose = {dose:.0%} de mudanca fracional · limiar = {LIMIAR:.0e}")

    # ── calibracao, antes de qualquer analise
    global BASE
    BASE = calibrar(BASE)
    print("")
    print(f"calibracao: beta = {BASE['beta']:.4g} para incubacao "
          f"~{INCUBACAO_ALVO:.0f} dias sem tratamento")

    # ── P0
    t0 = simular(BASE)
    print("\n" + "-" * 78)
    if t0 is None:
        print("P0 · REPROVADA. Sem tratamento o limiar nunca e atingido.")
        print("     O modelo nao progride; nao ha o que atrasar. Nada abaixo vale.")
        return
    print(f"P0 · aprovada — sem tratamento, limiar em {t0:.0f} dias")

    # ── P1: nao-monotonicidade da fragmentacao
    print("\nP1 · varredura de `b` (fragmentacao), tudo mais fixo")
    fs, ts = [], []
    for mult in (0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0):
        q = dict(BASE)
        q["b"] = BASE["b"] * mult
        t = simular(q)
        fs.append(mult)
        ts.append(t)
        print(f"     b x{mult:<6g}  ->  "
              f"{'nunca atinge' if t is None else f'{t:.0f} dias'}")
    # `None` = nunca atinge o limiar = infinitamente lento. E o EXTREMO da
    # ordenacao, nao um dado ausente. A primeira versao deste teste filtrava
    # os None e olhava so os finitos — o que apagava exatamente a
    # nao-monotonicidade que ele deveria detectar, e reprovou um modelo que
    # estava correto.
    vals = [math.inf if t is None else t for t in ts]
    cresc = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
    decr = all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
    nao_mono = not (cresc or decr)
    if not nao_mono:
        print("\n     P1 REPROVADA — o efeito de `b` saiu monotonico, e o modelo")
        print("     publicado diz que nao e. Meu codigo esta errado; nao leio o")
        print("     resto.")
        return
    print("\n     aprovada — efeito nao-monotonico reproduzido")

    # ── P2: alvos unicos, dose casada
    print("\n" + "-" * 78)
    print(f"P2 · ALVO UNICO, dose {dose:.0%}")
    unicos: List[Tuple[str, float]] = []
    for rot, (par, sinal) in ALVOS.items():
        g = atraso(aplicar(BASE, par, sinal, dose), t0)
        unicos.append((rot, g))
        s = "NAO PROGRIDE" if g == math.inf else f"{g:+.0f} dias"
        print(f"     {rot:<28} {s}")
    unicos.sort(key=lambda r: -(1e18 if r[1] == math.inf else r[1]))
    melhor_rot, melhor_g = unicos[0]

    # ── P3: combinacoes, dose TOTAL casada
    print("\n" + "-" * 78)
    print(f"P3 · COMBINACOES DE DOIS, {dose/2:.0%} em cada — dose TOTAL igual")
    combos: List[Tuple[str, float]] = []
    for (r1, (p1, s1)), (r2, (p2, s2)) in itertools.combinations(ALVOS.items(), 2):
        q = aplicar(aplicar(BASE, p1, s1, dose / 2), p2, s2, dose / 2)
        g = atraso(q, t0)
        combos.append((f"{r1.split()[0]} + {r2.split()[0]}", g))
    combos.sort(key=lambda r: -(1e18 if r[1] == math.inf else r[1]))
    for rot, g in combos:
        s = "NAO PROGRIDE" if g == math.inf else f"{g:+.0f} dias"
        print(f"     {rot:<28} {s}")

    print("\n" + "=" * 78)
    mc_rot, mc_g = combos[0]
    print(f"melhor alvo unico   {melhor_rot:<28} "
          f"{'NAO PROGRIDE' if melhor_g==math.inf else f'{melhor_g:+.0f} dias'}")
    print(f"melhor combinacao   {mc_rot:<28} "
          f"{'NAO PROGRIDE' if mc_g==math.inf else f'{mc_g:+.0f} dias'}")
    if mc_g == math.inf and melhor_g != math.inf:
        print("\nP3 · SINERGIA. A dose total identica, a combinacao interrompe a")
        print("     progressao e o melhor alvo unico nao.")
    elif mc_g > melhor_g * 1.1:
        print(f"\nP3 · SINERGIA. {mc_g/max(melhor_g,1e-9):.2f}x o melhor alvo unico,")
        print("     com a MESMA dose total repartida.")
    elif melhor_g > mc_g * 1.1:
        print("\nP3 · SEM SINERGIA. Concentrar a dose num alvo unico rende mais")
        print("     que reparti-la. Combinar dilui.")
    else:
        print("\nP3 · empate dentro de 10%. A reparticao nao muda o resultado.")

    # ── P4: robustez
    print("\n" + "-" * 78)
    print("P4 · ROBUSTEZ — a ordem sobrevive a variacao dos parametros?")
    topo: Dict[str, int] = {}
    n_ok = 0
    for fb in (0.3, 1.0, 3.0):
        for fbeta in (0.5, 1.0, 2.0):
            for fa in (0.5, 1.0, 2.0):
                q = dict(BASE)
                q["b"] *= fb
                q["beta"] *= fbeta
                q["a"] *= fa
                tb = simular(q)
                if tb is None:
                    continue
                n_ok += 1
                ranked = []
                for rot, (par, sinal) in ALVOS.items():
                    g = atraso(aplicar(q, par, sinal, dose), tb)
                    ranked.append((rot, 1e18 if g == math.inf else g))
                ranked.sort(key=lambda r: -r[1])
                topo[ranked[0][0]] = topo.get(ranked[0][0], 0) + 1
    print(f"     {n_ok} configuracoes progridem; alvo no topo em cada uma:")
    for rot, c in sorted(topo.items(), key=lambda r: -r[1]):
        print(f"       {rot:<28} {c}/{n_ok}")
    if topo and max(topo.values()) / max(1, n_ok) >= 0.8:
        vencedor = max(topo, key=lambda k: topo[k])
        print(f"\n     ROBUSTO — {vencedor.strip()} lidera em "
              f"{max(topo.values())}/{n_ok} das configuracoes.")
    else:
        print("\n     NAO ROBUSTO — o alvo no topo muda com os parametros. A")
        print("     ordem e escolha de parametro, nao resultado. Reportar assim.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dose", type=float, default=0.5)
    a = ap.parse_args()
    relatorio(dose=a.dose)


# ══════════════════════════════════════════════ dose minima eficaz
#
# A analise a dose fixa de 50% SATUROU: quase todo alvo interrompia a
# progressao e nada discriminava. Saturacao nao e resultado, e falta de
# resolucao — a mesma familia do piso de medicao de `intervencao_ativa.py`.
#
# A metrica que a saturacao indica e melhor, e e a clinicamente relevante:
#
#     DOSE MINIMA EFICAZ — a menor reducao fracional que interrompe a
#     progressao no horizonte simulado
#
# Um ASO entrega uma fracao de knockdown, nao um efeito abstrato. "Precisa de
# 60% de reducao" e uma afirmacao checavel e acionavel; "ganha 26 dias a 50%"
# nao e.
#
# Para COMBINACAO a dose total e repartida igualmente, entao "dose total
# minima" e comparavel entre 1 e 2 alvos. Sinergia real = a dupla interrompe
# com dose TOTAL menor que o melhor alvo sozinho.

def _halta(p: Dict[str, float]) -> bool:
    return simular(p) is None


def dose_minima(p: Dict[str, float], alvos: Sequence[Tuple[str, int]],
                iters: int = 40) -> Optional[float]:
    """
    Bisseccao na dose TOTAL, repartida igualmente entre os alvos.
    Devolve None se nem a dose maxima interrompe.

    PORTAO: verifica monotonicidade antes de bissectar. `b` tem efeito
    nao-monotonico e uma bisseccao cega sobre ele daria numero sem sentido.
    """
    def tenta(total: float) -> bool:
        q = dict(p)
        cada = total / len(alvos)
        for par, sinal in alvos:
            q = aplicar(q, par, sinal, cada)
        return _halta(q)

    if not tenta(0.98):
        return None
    # monotonicidade: uma vez que halta, doses maiores tambem haltam
    for teste in (0.98, 0.9, 0.8):
        if tenta(teste) and not tenta(min(0.98, teste + 0.05)):
            return None                      # nao monotonico, nao bissecta
    lo, hi = 0.0, 0.98
    for _ in range(iters):
        mid = (lo + hi) / 2
        if tenta(mid):
            hi = mid
        else:
            lo = mid
    return hi


def relatorio_med() -> None:
    global BASE
    BASE = calibrar(BASE)
    t0 = simular(BASE)
    print("=" * 78)
    print("PRION — DOSE MINIMA EFICAZ por alvo e por combinacao")
    print("=" * 78)
    print(f"sem tratamento: limiar em {t0:.0f} dias")
    print("metrica: menor reducao fracional que INTERROMPE a progressao")
    print("combinacoes repartem a dose TOTAL igualmente entre os dois alvos")

    print("\n" + "-" * 78)
    print("ALVO UNICO")
    unicos = []
    for rot, (par, sinal) in ALVOS.items():
        dm = dose_minima(BASE, [(par, sinal)])
        unicos.append((rot, dm))
        print(f"  {rot:<28} "
              f"{'nao interrompe' if dm is None else f'{dm:.1%}'}")

    print("\n" + "-" * 78)
    print("COMBINACOES DE DOIS, dose TOTAL repartida")
    combos = []
    for (r1, a1), (r2, a2) in itertools.combinations(ALVOS.items(), 2):
        dm = dose_minima(BASE, [a1, a2])
        combos.append((f"{r1.split()[0]} + {r2.split()[0]}", dm))
    combos.sort(key=lambda r: (r[1] is None, r[1] if r[1] else 9))
    for rot, dm in combos:
        print(f"  {rot:<28} "
              f"{'nao interrompe' if dm is None else f'{dm:.1%}'}")

    validos_u = [(r, d) for r, d in unicos if d is not None]
    validos_c = [(r, d) for r, d in combos if d is not None]
    print("\n" + "=" * 78)
    if not validos_u or not validos_c:
        print("sem comparacao possivel")
        return
    mu_r, mu_d = min(validos_u, key=lambda r: r[1])
    mc_r, mc_d = min(validos_c, key=lambda r: r[1])
    print(f"melhor alvo unico   {mu_r:<28} {mu_d:.1%}")
    print(f"melhor combinacao   {mc_r:<28} {mc_d:.1%}")
    print()
    if mc_d < mu_d * 0.9:
        print(f"SINERGIA: a dupla interrompe com dose TOTAL {1-mc_d/mu_d:.0%}")
        print("menor que o melhor alvo sozinho. Repartir vence concentrar.")
    elif mu_d < mc_d * 0.9:
        print("SEM SINERGIA: concentrar a dose num alvo unico exige dose total")
        print("MENOR que reparti-la. Combinar dilui.")
    else:
        print("EMPATE dentro de 10%: repartir ou concentrar da no mesmo.")

    print("\n" + "-" * 78)
    print("ROBUSTEZ — a ordem sobrevive a variacao dos parametros?")
    topo: Dict[str, int] = {}
    n = 0
    for fb in (0.5, 1.0, 2.0):
        for fbeta in (0.7, 1.0, 1.5):
            for fa in (0.5, 1.0, 2.0):
                q = dict(BASE)
                q["b"] *= fb
                q["beta"] *= fbeta
                q["a"] *= fa
                if simular(q) is None:
                    continue
                n += 1
                r = []
                for rot, (par, sinal) in ALVOS.items():
                    dm = dose_minima(q, [(par, sinal)])
                    if dm is not None:
                        r.append((rot, dm))
                if r:
                    topo[min(r, key=lambda z: z[1])[0]] = \
                        topo.get(min(r, key=lambda z: z[1])[0], 0) + 1
    print(f"  {n} configuracoes progridem; alvo de MENOR dose em cada:")
    for rot, c in sorted(topo.items(), key=lambda r: -r[1]):
        print(f"    {rot:<28} {c}/{n}")
    if topo and max(topo.values()) / max(1, n) >= 0.8:
        print(f"\n  ROBUSTO — {max(topo, key=lambda k: topo[k]).strip()} lidera")
        print(f"  em {max(topo.values())}/{n} das configuracoes.")
    else:
        print("\n  NAO ROBUSTO — o alvo de menor dose muda com os parametros.")
