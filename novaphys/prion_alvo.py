"""
novaphys.prion_alvo — a pergunta clinica, agora com um modelo que tem chao
===========================================================================

POR QUE ESTA ANALISE PODE SER REFEITA
---------------------------------------
A primeira versao disto (`prion.py`) produziu uma tabela de alvos terapeuticos
e eu a RETIREI, porque o modelo por tras falhava em tres eixos independentes:

    duracao sintomatica sob tratamento   15,0 d invariante   vs 53 d observados
    incubacao a 10x PrP                  6,99x               vs 2,46x
    doenca a 0,5x PrP                    nao ocorre          vs ocorre

A varredura conjunta (`prion_conjunto.py`) achou a regiao que satisfaz as
CINCO observacoes ao mesmo tempo, e ela e definida por duas restricoes:

    conversao SATURA em PrP-C,  Kx entre ~1 e ~10 vezes o nivel fisiologico
    depuracao de agregado ao menos 3x mais lenta que o valor convencional

e e insensivel a `b` (faixa de 30x) e a `n` (2 a 6) — assinatura de restricao
estrutural, nao de ajuste. A lei de dano que funciona e LINEAR em PrP-C.

A PERGUNTA, E ELA E A DO ENSAIO REAL
--------------------------------------
O ION717 e dado a pacientes JA SINTOMATICOS. Para eles adiar o inicio nao
serve de nada. A pergunta e:

    comecando o tratamento NO INICIO CLINICO, quanto tempo cada alvo compra,
    e quanta reducao e necessaria?

ROBUSTEZ ANTES DE RECOMENDACAO
--------------------------------
Nao se responde isso num ponto. A resposta e calculada em TODA a regiao
sobrevivente (16 combinacoes de Kx e a), e so vale se a ordem for a mesma em
todas. Se a ordem mudar dentro da regiao, a resposta depende de um parametro
que ninguem mediu — e ai o entregavel deixa de ser uma recomendacao e passa a
ser QUAL MEDICAO resolveria.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  T0  PRE-CONDICAO. Todo ponto da regiao reproduz duracao de controle ~15 d.
      Se nao reproduz, nao e a regiao que eu penso que e.

  T1  um alvo lidera em TODA a regiao -> recomendacao robusta
  T2  a ordem muda dentro da regiao   -> nao ha recomendacao; o entregavel e
      a medicao que resolveria a ambiguidade
  T3  o numero que importa: dias ganhos por nivel de reducao, para o alvo
      lider, com faixa sobre a regiao

MINHA PREVISAO: T2. A saturacao muda quanto baixar PrP rende, e a depuracao
lenta muda quanto acelerar depuracao rende — os dois parametros livres da
regiao mexem justamente nos dois alvos principais, em direcoes opostas.
Espero que a ordem entre `lam-` e `a+` se inverta dentro da regiao.
"""

from __future__ import annotations

import itertools
import warnings
from typing import Dict, List, Optional, Tuple

from scipy.integrate import solve_ivp

from .prion import BASE

warnings.filterwarnings("ignore")

LIMIAR = 55000.0
ALVO_INCUB = 142.0
DUR_CTRL = 15.0

# a regiao sobrevivente de `prion_conjunto.py`
REGIAO_KX = (2.0, 3.0, 6.0, 10.0)          # multiplos de x0
REGIAO_A = (0.03, 0.1, 0.2, 0.3)           # multiplos do `a` convencional
B_FIXO, N_FIXO = 1.0, 2.0                  # insensiveis na regiao

ALVOS = {
    "lam-  ASO anti-PRNP": "lam",
    "a+    depuracao": "a",
    "beta- anti-conversao": "beta",
    "b-    anti-fragmentacao": "b",
}
DOSES = (0.25, 0.50, 0.75, 0.90)


def _campo(v: Dict, lam0: float, a: float, beta: float, b: float,
           t_ini: Optional[float], dano: bool, escala: float = 1.0):
    """
    `escala` normaliza a variavel de dano. Sem isso ela cresce ate ~5e5 com
    atol de 1e-4 — nove ordens de grandeza de diferenca — e o Radau falha no
    PRIMEIRO passo da fase 2. Pior: a falha voltava como `None`, e a tabela
    de lider convertia `None` em `inf`, pontuando FALHA NUMERICA COMO CURA.
    """
    n, d = v["n"], BASE["d"]
    x0 = BASE["lam"] / BASE["d"]
    Kx = v["Kx"]

    def f(t, s):
        x, y, z, _L = s
        e = 2 * beta * ((x / (Kx + x)) * (Kx + x0)) * z
        dL = (x / escala) if (dano and y >= LIMIAR) else 0.0
        return [lam0 - d * x - e, e - a * y - b * n * (n - 1) * z,
                b * (y - (2 * n - 1) * z) - a * z, dL]
    return f


def rodar(v: Dict, alvo: Optional[str] = None, dose: float = 0.0,
          letal: Optional[float] = None, dur_max: float = 4000.0
          ) -> Tuple[Optional[float], Optional[float]]:
    """
    Fase 1 sem tratamento (paciente chega sintomatico), fase 2 COM o alvo
    modificado a partir do inicio clinico e mantido.
    """
    lam0, a, beta, b = BASE["lam"], v["a"], v["beta"], v["b"]
    n = v["n"]

    # fase 1: sempre sem tratamento
    def ev_i(t, s):
        return s[1] - LIMIAR
    ev_i.terminal = True
    ev_i.direction = 1
    s1 = solve_ivp(_campo(v, lam0, a, beta, b, None, False),
                   (0.0, 4000.0), [lam0 / BASE["d"], n * 1.0, 1.0, 0.0],
                   method="Radau", events=ev_i, rtol=1e-9, atol=1e-9)
    if not s1.success or not len(s1.t_events[0]):
        return None, None
    ti = float(s1.t_events[0][0])
    est = s1.y[:, -1].copy()
    est[3] = 0.0

    # fase 2: alvo modificado
    L2, a2, be2, b2 = lam0, a, beta, b
    if alvo == "lam":
        L2 = lam0 * (1 - dose)
    elif alvo == "a":
        a2 = a / max(1e-9, 1 - dose)
    elif alvo == "beta":
        be2 = beta * (1 - dose)
    elif alvo == "b":
        b2 = b * (1 - dose)

    # ovo e galinha: preciso do letal para normalizar, mas o letal e
    # calculado nesta mesma funcao. Na passada que o calcula uso LIMIAR como
    # escala de referencia — mesma ordem de grandeza do acumulo em 15 d.
    escala = letal if letal else LIMIAR
    eventos = []
    if letal is not None:
        def ev_m(t, s):
            return s[3] - 1.0                # dano normalizado: letal = 1
        ev_m.terminal = True
        ev_m.direction = 1
        eventos = [ev_m]
    s2 = solve_ivp(_campo(v, L2, a2, be2, b2, ti, True, escala),
                   (ti, ti + dur_max), est, method="Radau", events=eventos,
                   rtol=1e-8, atol=[1e-8, 1e-8, 1e-8, 1e-6])
    if not s2.success:
        return ti, "FALHA"                   # NUNCA confundir com sobrevida
    if letal is not None:
        if len(s2.t_events[0]):
            return ti, float(s2.t_events[0][0]) - ti
        return ti, None                      # nao morre no horizonte, de fato
    return ti, float(s2.y[3, -1]) * escala


def calibrar(Kx_f: float, a_f: float) -> Optional[Dict]:
    x0 = BASE["lam"] / BASE["d"]
    v = {"Kx": Kx_f * x0, "a": BASE["a"] * a_f, "b": BASE["b"] * B_FIXO,
         "n": N_FIXO, "beta": BASE["beta"]}
    lo, hi = 1e-5, 3.0
    for _ in range(45):
        mid = (lo + hi) / 2
        v["beta"] = mid
        ti, _ = rodar(v)
        if ti is None or ti > ALVO_INCUB:
            lo = mid
        else:
            hi = mid
    v["beta"] = (lo + hi) / 2
    ti, _ = rodar(v)
    if ti is None or abs(ti - ALVO_INCUB) / ALVO_INCUB > 0.03:
        return None
    return v


def relatorio() -> None:
    print("=" * 78)
    print("PRION — quanto tempo cada alvo compra para PACIENTE JA SINTOMATICO")
    print("=" * 78)
    print("PRE-REGISTRO:")
    print("  T0  todo ponto da regiao reproduz duracao de controle ~15 d")
    print("  T1  um alvo lidera em TODA a regiao -> recomendacao robusta")
    print("  T2  a ordem muda -> nao ha recomendacao; entregavel e a medicao")
    print("  T3  dias ganhos por nivel de reducao, com faixa")
    print("\nMINHA PREVISAO: T2 — espero que `lam-` e `a+` troquem de lugar")
    print("dentro da regiao, porque os dois parametros livres mexem")
    print("justamente neles, em direcoes opostas.")

    pontos: List[Tuple[float, float, Dict, float]] = []
    print("\n" + "-" * 78)
    print("T0 · calibracao da regiao")
    for kx, af in itertools.product(REGIAO_KX, REGIAO_A):
        v = calibrar(kx, af)
        if v is None:
            print(f"     Kx={kx:>4}x0  a={af:>4}x   nao calibra")
            continue
        _ti, dano = rodar(v)
        if dano is None or dano <= 0:
            print(f"     Kx={kx:>4}x0  a={af:>4}x   sem dano acumulavel")
            continue
        # letal = dano do controle em 15 d
        _ti2, letal = rodar(v, dur_max=DUR_CTRL)
        _t3, dur0 = rodar(v, letal=letal)
        if dur0 is None or abs(dur0 - DUR_CTRL) / DUR_CTRL > 0.10:
            print(f"     Kx={kx:>4}x0  a={af:>4}x   duracao ctrl {dur0}")
            continue
        pontos.append((kx, af, v, letal))
    print(f"     {len(pontos)} de {len(REGIAO_KX)*len(REGIAO_A)} pontos "
          f"calibram e reproduzem 15 d")
    if len(pontos) < 4:
        print("     T0 REPROVADA — regiao pequena demais para falar de robustez")
        return
    print("     aprovada")

    print("\n" + "-" * 78)
    print("DURACAO DA FASE SINTOMATICA, tratamento iniciado no inicio clinico")
    print("(sem tratamento: 15 d · valores em dias, mediana da regiao "
          "[min-max])")
    print(f"\n{'alvo':<24}" + "".join(f"{int(d*100):>13}%" for d in DOSES))
    print("-" * 78)
    lideres = []
    for rot, par in ALVOS.items():
        linha = f"{rot:<24}"
        for dose in DOSES:
            vals = []
            for _kx, _af, v, letal in pontos:
                _ti, dur = rodar(v, alvo=par, dose=dose, letal=letal)
                vals.append(dur)
            falhas = sum(1 for x in vals if x == "FALHA")
            fin = [x for x in vals if isinstance(x, float)]
            infs = sum(1 for x in vals if x is None)
            if falhas:
                linha += f"{'FALHA x%d' % falhas:>14}"
            elif not fin:
                linha += f"{'nao morre':>14}"
            else:
                fin.sort()
                med = fin[len(fin) // 2]
                linha += f"{med:>7.0f} [{min(fin):.0f}-{max(fin):.0f}]"
        print(linha)

    # ordem por ponto, na dose de 50%
    print("\n" + "-" * 78)
    print("T1/T2 · qual alvo lidera em cada ponto da regiao (dose 50%)")
    print(f"{'Kx/x0':>7}{'a/base':>9}   lider")
    conta: Dict[str, int] = {}
    for kx, af, v, letal in pontos:
        melhor, qual = -1.0, ""
        for rot, par in ALVOS.items():
            _ti, dur = rodar(v, alvo=par, dose=0.5, letal=letal)
            if dur == "FALHA":
                continue                     # falha nao compete
            d = float("inf") if dur is None else dur
            if d > melhor:
                melhor, qual = d, rot
        conta[qual] = conta.get(qual, 0) + 1
        lideres.append(qual)
        print(f"{kx:>7}{af:>9}   {qual}  ({melhor:.0f} d)")

    print("\n" + "=" * 78)
    if len(set(lideres)) == 1:
        print(f"T1 · RECOMENDACAO ROBUSTA. `{lideres[0].strip()}` lidera nos")
        print(f"     {len(pontos)} pontos da regiao sobrevivente.")
    else:
        print("T2 · A ORDEM MUDA DENTRO DA REGIAO. Nao ha recomendacao.")
        for k, c in sorted(conta.items(), key=lambda r: -r[1]):
            print(f"     {k.strip():<26} lidera em {c}/{len(pontos)}")
        print()
        print("     O entregavel deixa de ser um alvo e passa a ser uma")
        print("     MEDICAO: qual dos dois parametros livres da regiao —")
        print("     a constante de saturacao da conversao, ou a taxa de")
        print("     depuracao de agregado — decide a ordem. Medir esse")
        print("     parametro resolve a escolha terapeutica.")


if __name__ == "__main__":
    relatorio()
