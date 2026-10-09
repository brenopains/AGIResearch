"""
novaphys.campo_cumulativo — o campo cresce com o que descobre?
===============================================================

`campo_generativo.py` mostrou síntese composicional: com cinco terminais e uma
pressão de descrição, emergiram `v/v` e `|v−v|` sem que "dividir" ou "módulo"
fossem nomeados como candidatos. I = 2,455 de 2,585 bits.

E a ressalva correta, que eu tinha enunciado fraco demais: essas estruturas
estavam fora do CARDÁPIO NOMEADO, não fora do ESPAÇO GENERATIVO. Os cinco
terminais já determinavam que elas eram alcançáveis. Isso é síntese
composicional dentro de um espaço que eu projetei — não autoexpansão.

    demonstrado       compor estruturas não fornecidas como operadores
    NÃO demonstrado   aumentar o espaço do que é alcançável

A DIFERENÇA, TORNADA MEDÍVEL
------------------------------
Com orçamento de busca FIXO, uma estrutura profunda é inalcançável: ela não
cabe na profundidade permitida. A pergunta deixa de ser filosófica:

    uma estrutura descoberta, ao voltar para o campo como terminal, torna
    alcançável o que antes não era — COM O MESMO ORÇAMENTO?

Se sim, o espaço do que o sistema consegue descobrir cresce com o que ele já
descobriu. É acumulação, e é diferente de busca mais funda.

    rodada 1   campo = {v, +, −, ·, /, |·|}          profundidade 1 de composição
       ↓       resolve tarefas rasas, descobre estruturas
    CONSOLIDAÇÃO   o que recorre E comprime volta como TERMINAL
       ↓
    rodada 2   campo = terminais + estruturas consolidadas   mesma profundidade
       ↓       alcança o que exigia composição de composição

A consolidação não me consulta: entra o que reaparece em tarefas distintas e
paga o próprio custo em descrição. As estruturas consolidadas recebem nome por
ÍNDICE (`E1`, `E2`), nunca semântico — se eu as chamasse de "razão" teria
recolocado o cardápio pela porta dos fundos.

PRÉ-REGISTRO — escrito antes de rodar
--------------------------------------
  R1  na rodada 1, as tarefas PROFUNDAS falham: a estrutura certa não cabe no
      orçamento. Falha aqui é o esperado e é o que dá sentido ao resto.
  R2  depois da consolidação, as MESMAS tarefas, com o MESMO orçamento,
      passam a ser resolvidas.
  R3  o controle raso NÃO usa terminal consolidado — se usar, a consolidação
      está inflando a busca em vez de ampliar o alcance.
  R4  a mesma coisa contra a FÍSICA: `f_Q = P_fus/P_aux` é uma razão, e o
      mecanismo com cardápio nomeado se absteve ali. Sem cardápio, e com o
      campo cumulativo, ele deveria alcançá-la.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics as st
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

Obs = Tuple[List[float], int]
N_VARS = 4
LAM = 1.0
EPS = 1e-6


# ══════════════════════════════════════════════ expressões
@dataclass(frozen=True)
class Expr:
    op: str                 # 'V' | 'C' (consolidada) | '+' '-' '*' '/' 'abs'
    a: Any = None
    b: Any = None
    rot: str = ''           # rótulo por índice, para consolidadas

    def val(self, v: Sequence[float]) -> float:
        o = self.op
        if o == 'V':
            return v[self.a]
        if o == 'C':
            return self.a.val(v)
        x = self.a.val(v)
        if o == 'abs':
            return abs(x)
        y = self.b.val(v)
        if o == '+':
            return x + y
        if o == '-':
            return x - y
        if o == '*':
            return x * y
        if o == '/':
            return x / y if abs(y) > EPS else math.copysign(1e6, x or 1.0)
        raise ValueError(o)

    def tam(self) -> int:
        """Consolidada custa 1 — é esse o ganho de tê-la no campo."""
        if self.op in ('V', 'C'):
            return 1
        if self.op == 'abs':
            return 1 + self.a.tam()
        return 1 + self.a.tam() + self.b.tam()

    def forma(self) -> str:
        if self.op == 'V':
            return 'v'
        if self.op == 'C':
            return self.rot
        if self.op == 'abs':
            return f"|{self.a.forma()}|"
        return f"({self.a.forma()}{self.op}{self.b.forma()})"

    def expandida(self) -> str:
        """A forma com as consolidadas abertas — para auditar o que foi achado."""
        if self.op == 'V':
            return 'v'
        if self.op == 'C':
            return self.a.expandida()
        if self.op == 'abs':
            return f"|{self.a.expandida()}|"
        return f"({self.a.expandida()}{self.op}{self.b.expandida()})"

    def __str__(self) -> str:
        if self.op == 'V':
            return f"v{self.a}"
        if self.op == 'C':
            return self.rot
        if self.op == 'abs':
            return f"|{self.a}|"
        return f"({self.a}{self.op}{self.b})"


def compor(campo: Sequence[Expr]) -> List[Expr]:
    """UM nível de composição sobre o campo atual. O orçamento é este."""
    out = list(campo)
    for x in campo:
        for y in campo:
            if x is y:
                continue
            for o in ('+', '-', '*', '/'):
                out.append(Expr(o, x, y))
        out.append(Expr('abs', x))
    return out


@dataclass(frozen=True)
class Teste:
    e: Expr
    k: Optional[float] = None
    e2: Optional[Expr] = None

    def aval(self, v: Sequence[float]) -> bool:
        return self.e.val(v) < (self.e2.val(v) if self.e2 is not None else self.k)

    def tam(self) -> int:
        return 1 + self.e.tam() + (self.e2.tam() if self.e2 is not None else 1)

    def forma(self) -> str:
        return (f"{self.e.forma()}<{self.e2.forma()}" if self.e2 is not None
                else f"{self.e.forma()}<k")


@dataclass(frozen=True)
class Prog:
    testes: Tuple[Tuple[Teste, int], ...] = ()
    padrao: int = 1

    def aval(self, v: Sequence[float]) -> int:
        for t, s in self.testes:
            if t.aval(v):
                return s
        return self.padrao

    def tam(self) -> int:
        return 1 + sum(t.tam() + 1 for t, _ in self.testes)

    def custo(self, obs: Sequence[Obs]) -> float:
        return sum(1 for (v, s) in obs if self.aval(v) != s) + LAM * self.tam()

    def usa_consolidada(self) -> bool:
        def tem(e: Expr) -> bool:
            if e.op == 'C':
                return True
            return any(tem(x) for x in (e.a, e.b) if isinstance(x, Expr))
        return any(tem(t.e) or (t.e2 is not None and tem(t.e2))
                   for t, _ in self.testes)


def montar_testes(obs: Sequence[Obs], campo: Sequence[Expr],
                  n_cortes: int = 5) -> List[Teste]:
    exprs = compor(campo)
    out: List[Teste] = []
    for e in exprs:
        vals = sorted(e.val(v) for v, _ in obs)
        for k in range(n_cortes):
            out.append(Teste(e, k=vals[int((k + 1) * len(vals) / (n_cortes + 1))]))
    for x in campo:
        for y in campo:
            if x is not y:
                out.append(Teste(x, e2=y))
    return out


def melhor(obs: Sequence[Obs], testes: Sequence[Teste],
           profundidade: int = 2) -> Prog:
    best, best_c = None, math.inf
    for padrao in (1, -1):
        atual, c_at = Prog(padrao=padrao), Prog(padrao=padrao).custo(obs)
        for _ in range(profundidade):
            cand, cand_c = atual, c_at
            for t in testes:
                for s in (1, -1):
                    p = Prog(atual.testes + ((t, s),), padrao)
                    c = p.custo(obs)
                    if c < cand_c:
                        cand, cand_c = p, c
            if cand is atual:
                break
            atual, c_at = cand, cand_c
        if c_at < best_c:
            best, best_c = atual, c_at
    return best


# ══════════════════════════════════════════════ tarefas
RASAS = ('razao', 'distancia', 'produto')
PROFUNDAS = ('razao_desloc', 'dist_de_razao')


def tarefa(tipo: str, rng: random.Random, n: int = 350) -> List[Obs]:
    """
    RASAS      solução cabe em UM nível de composição
    PROFUNDAS  solução exige composição DE composição:
               razao_desloc    (v0/v1) − v2  <  k
               dist_de_razao   |(v0/v1) − v2| <  k
    """
    c = {'razao': rng.uniform(0.7, 1.5), 'distancia': rng.uniform(0.2, 0.45),
         'produto': rng.uniform(0.15, 0.4), 'razao_desloc': rng.uniform(-0.3, 0.5),
         'dist_de_razao': rng.uniform(0.25, 0.6),
         'alinhada': rng.uniform(0.3, 0.7)}[tipo]
    obs: List[Obs] = []
    for _ in range(n):
        v = [rng.uniform(0.2, 1.0) for _ in range(N_VARS)]
        if tipo == 'razao':
            s = 1 if v[0] / v[1] < c else -1
        elif tipo == 'distancia':
            s = 1 if abs(v[0] - v[1]) < c else -1
        elif tipo == 'produto':
            s = 1 if v[0] * v[1] < c else -1
        elif tipo == 'razao_desloc':
            s = 1 if (v[0] / v[1]) - v[2] < c else -1
        elif tipo == 'dist_de_razao':
            s = 1 if abs((v[0] / v[1]) - v[2]) < c else -1
        elif tipo == 'alinhada':
            s = 1 if v[0] < c else -1
        else:
            raise ValueError(tipo)
        obs.append((v, s))
    return obs


# ══════════════════════════════════════════════ consolidação
def consolidar(descobertas: Sequence[Expr], min_vezes: int = 2,
               max_novas: int = 2, ja: int = 0) -> List[Expr]:
    """
    O que reaparece em tarefas distintas e paga o próprio custo volta ao campo.

    Rótulo por ÍNDICE (`E1`, `E2`). Nunca semântico: chamar de "razão"
    recolocaria o cardápio pela porta dos fundos.
    """
    por_forma: Dict[str, List[Expr]] = {}
    for e in descobertas:
        if e.op in ('V', 'C'):
            continue
        por_forma.setdefault(e.forma(), []).append(e)
    novas: List[Expr] = []
    for forma, exs in sorted(por_forma.items(), key=lambda kv: -len(kv[1])):
        if len(exs) < min_vezes:
            continue
        modelo = exs[0]
        if modelo.tam() <= 2:                 # não compensa nomear
            continue
        novas.append(Expr('C', modelo, rot=f"E{ja + len(novas) + 1}"))
        if len(novas) >= max_novas:
            break
    return novas


def _subexprs(e: Expr) -> List[Expr]:
    out = [e]
    for x in (e.a, e.b):
        if isinstance(x, Expr):
            out.extend(_subexprs(x))
    return out


# ══════════════════════════════════════════════ experimento
def experimento(sementes: Sequence[int] = (0, 1, 2, 3, 4), n: int = 350,
                verboso: bool = True) -> Dict[str, Any]:
    base = [Expr('V', i) for i in range(N_VARS)]

    # ── rodada 1: campo base, tarefas rasas ─────────────────────────────
    descobertas: List[Expr] = []
    aj_rasas = []
    for fam in RASAS:
        for s in sementes:
            rng = random.Random(s * 811 + hash(fam) % 71)
            obs = tarefa(fam, rng, n=n)
            p = melhor(obs, montar_testes(obs, base))
            aj_rasas.append(1 - sum(1 for (v, y) in obs if p.aval(v) != y) / len(obs))
            for t, _ in p.testes:
                descobertas.extend(_subexprs(t.e))

    # ── rodada 1 nas PROFUNDAS: previsão R1 é que falhem ────────────────
    aj_prof_antes = {}
    for fam in PROFUNDAS:
        ajs = []
        for s in sementes:
            rng = random.Random(s * 811 + hash(fam) % 71)
            obs = tarefa(fam, rng, n=n)
            p = melhor(obs, montar_testes(obs, base))
            ajs.append(1 - sum(1 for (v, y) in obs if p.aval(v) != y) / len(obs))
        aj_prof_antes[fam] = st.mean(ajs)

    # ── consolidação ────────────────────────────────────────────────────
    novas = consolidar(descobertas)
    campo2 = base + novas
    if verboso:
        print("CONSOLIDADAS (rótulo por índice; o conteúdo é auditoria, não nome):")
        for e in novas:
            print(f"  {e.rot} = {e.a.forma()}")
        if not novas:
            print("  nenhuma")

    # ── rodada 2: mesmo orçamento, campo enriquecido ────────────────────
    aj_prof_depois, usou_prof = {}, {}
    for fam in PROFUNDAS:
        ajs, usos = [], 0
        for s in sementes:
            rng = random.Random(s * 811 + hash(fam) % 71)
            obs = tarefa(fam, rng, n=n)
            p = melhor(obs, montar_testes(obs, campo2))
            ajs.append(1 - sum(1 for (v, y) in obs if p.aval(v) != y) / len(obs))
            usos += int(p.usa_consolidada())
        aj_prof_depois[fam] = st.mean(ajs)
        usou_prof[fam] = usos / len(sementes)

    # ── controle raso: consolidada NÃO deve ser usada ───────────────────
    usou_ctrl, aj_ctrl = 0, []
    for s in sementes:
        rng = random.Random(s * 811 + 41)
        obs = tarefa('alinhada', rng, n=n)
        p = melhor(obs, montar_testes(obs, campo2))
        aj_ctrl.append(1 - sum(1 for (v, y) in obs if p.aval(v) != y) / len(obs))
        usou_ctrl += int(p.usa_consolidada())

    return {'novas': novas, 'aj_rasas': st.mean(aj_rasas),
            'antes': aj_prof_antes, 'depois': aj_prof_depois,
            'usou': usou_prof, 'usou_ctrl': usou_ctrl / len(sementes),
            'aj_ctrl': st.mean(aj_ctrl)}


def relatorio(n: int = 350) -> Dict[str, Any]:
    print("═" * 78)
    print("CAMPO CUMULATIVO — o descoberto volta ao campo e amplia o alcançável?")
    print("orçamento de busca FIXO nas duas rodadas: UM nível de composição")
    print("═" * 78)
    print("PRÉ-REGISTRO:")
    print("  R1  tarefas PROFUNDAS falham na rodada 1 (não cabem no orçamento)")
    print("  R2  após consolidação, as MESMAS tarefas com o MESMO orçamento passam")
    print("  R3  o controle raso NÃO usa terminal consolidado")
    print()

    r = experimento(n=n)
    print()
    print(f"ajuste nas tarefas rasas (rodada 1) .......... {r['aj_rasas']:.0%}")
    print()
    print(f"{'tarefa profunda':<18}{'antes':>9}{'depois':>9}{'ganho':>9}{'usou E':>9}")
    for fam in PROFUNDAS:
        a, d = r['antes'][fam], r['depois'][fam]
        print(f"{fam:<18}{a:>8.0%}{d:>9.0%}{d - a:>+8.0%}{r['usou'][fam]:>9.0%}")
    print()
    print(f"controle raso: ajuste {r['aj_ctrl']:.0%} · "
          f"usou consolidada em {r['usou_ctrl']:.0%}   (ideal 0%)")

    print("─" * 78)
    ganhos = [r['depois'][f] - r['antes'][f] for f in PROFUNDAS]
    usos = [r['usou'][f] for f in PROFUNDAS]
    if not r['novas']:
        print("Nada foi consolidado — sem terminal novo não há o que testar.")
    elif r['usou_ctrl'] > 0.2:
        print(f"CONTROLE REPROVADO: a consolidada é usada em {r['usou_ctrl']:.0%} do")
        print("controle raso, onde não ajuda. A consolidação está inflando a busca,")
        print("não ampliando o alcance — e os ganhos acima ficam ilegíveis.")
    elif st.mean(ganhos) > 0.08 and st.mean(usos) > 0.6:
        print("O CAMPO CRESCEU. Com o MESMO orçamento de busca, estruturas que não")
        print("cabiam passaram a caber, porque uma descoberta anterior virou terminal.")
        print("Isso é acumulação: o espaço do que o sistema consegue descobrir depende")
        print("do que ele já descobriu — e não de busca mais funda.")
        print()
        print("A ressalva: os terminais INICIAIS continuam sendo escolha minha. O que")
        print("cresceu foi o alcançável a partir deles, não o espaço em si.")
    else:
        print(f"Parcial: ganho médio {st.mean(ganhos):+.0%}, uso {st.mean(usos):.0%}.")
        print("Não fecha o critério de acumulação.")
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=350)
    a = ap.parse_args()
    relatorio(n=a.n)
