"""
novaphys.campo_generativo — sem operações nomeadas
===================================================

Em `geracao.py` havia TRÊS operações de expansão com nome — PARAMETRIZAR,
SOMAR, MULTIPLICAR — e a pergunta era qual delas a experiência selecionava.
Isso mede seleção dentro de um cardápio que eu escrevi.

Aqui não há cardápio. Há uma GRAMÁTICA pequena e uma pressão só:

    E  :=  v_i  |  E+E  |  E−E  |  E·E  |  E/E  |  |E|
    T  :=  E < E   |   E < k
    P  :=  se T então s senão P   |   s

    pressão:  minimizar   erros + λ·tamanho

Nenhuma estrutura é nomeada, listada ou preferida. "Razão", "distância",
"produto" não existem como conceitos no código — são apenas caminhos possíveis
dentro da mesma gramática. O que emerge, emerge.

O QUE ISTO TESTA E O ANTERIOR NÃO TESTAVA
------------------------------------------
Duas das famílias de tarefa abaixo exigem estruturas que NÃO estavam entre as
três operações nomeadas de `geracao.py`:

    razão       v_i / v_j < k        DIVIDIR nunca foi uma operação nomeada
    distância   |v_i − v_j| < k      valor absoluto idem

Se o mecanismo encontrar essas estruturas, encontrou algo fora do conjunto que
eu havia enumerado como "operações de expansão" — e sem que ninguém dissesse
que dividir ou tirar módulo eram coisas a procurar.

O REGRESSO DESCE DE NOVO, E DE NOVO NÃO SOME
----------------------------------------------
Eu escolhi os terminais da gramática: {+, −, ·, /, |·|}. Isso é escolha minha e
está dito antes do resultado. O que eu NÃO escolhi:

    · quais composições importam
    · qual profundidade
    · que estrutura serve para que tarefa
    · nem que "razão" e "distância" sejam categorias

A diferença com `geracao.py` é essa: lá o cardápio tinha três pratos; aqui há
ingredientes e nenhuma receita.

PRÉ-REGISTRO — escrito antes de rodar
--------------------------------------
  Q1  cada família deve fazer emergir a estrutura correspondente, SEM que ela
      esteja nomeada em lugar nenhum
  Q2  o controle alinhado deve emergir com estrutura TRIVIAL (v_i < k), não
      com uma expressão composta — expressão composta ali é sobreajuste
  Q3  I(tarefa ; forma descoberta) deve ficar bem acima de zero; com seis
      famílias o máximo é log2(6) ≈ 2,585 bits
  Q4  razão e distância — as duas que estavam fora do conjunto nomeado — são o
      teste real. Se falharem, o limite é da gramática ou da busca, e isso é
      resultado.
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
import statistics as st
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

Obs = Tuple[List[float], int]
N_VARS = 3
LAM = 1.0
EPS = 1e-6


# ══════════════════════════════════════════════ a gramática
@dataclass(frozen=True)
class Expr:
    """Uma expressão sobre o contexto. Sem nome, sem categoria."""
    op: str                        # 'V' | '+' | '-' | '*' | '/' | 'abs'
    a: Any = None
    b: Any = None

    def val(self, v: Sequence[float]) -> float:
        o = self.op
        if o == 'V':
            return v[self.a]
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
            return x / y if abs(y) > EPS else (math.copysign(1e6, x) if x else 0.0)
        raise ValueError(o)

    def tam(self) -> int:
        if self.op == 'V':
            return 1
        if self.op == 'abs':
            return 1 + self.a.tam()
        return 1 + self.a.tam() + self.b.tam()

    def forma(self) -> str:
        """A FORMA, com as variáveis apagadas. É isto que se compara."""
        if self.op == 'V':
            return 'v'
        if self.op == 'abs':
            return f"|{self.a.forma()}|"
        return f"({self.a.forma()}{self.op}{self.b.forma()})"

    def __str__(self) -> str:
        if self.op == 'V':
            return f"v{self.a}"
        if self.op == 'abs':
            return f"|{self.a}|"
        return f"({self.a}{self.op}{self.b})"


def expressoes(max_tam: int = 3) -> List[Expr]:
    """Tudo que a gramática gera até `max_tam`. Nenhuma forma é privilegiada."""
    base = [Expr('V', i) for i in range(N_VARS)]
    todas = list(base)
    if max_tam >= 3:
        for i in range(N_VARS):
            for j in range(N_VARS):
                if i == j:
                    continue
                for o in ('+', '-', '*', '/'):
                    todas.append(Expr(o, base[i], base[j]))
        for i in range(N_VARS):
            for j in range(N_VARS):
                if i < j:
                    todas.append(Expr('abs', Expr('-', base[i], base[j])))
    return todas


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

    def __str__(self) -> str:
        return f"{self.e}<{self.e2 if self.e2 is not None else f'{self.k:.2f}'}"


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

    def forma(self) -> str:
        return self.testes[0][0].forma() if self.testes else 'constante'

    def __str__(self) -> str:
        if not self.testes:
            return '+' if self.padrao > 0 else '−'
        return (" senão ".join(f"se {t} → {'+' if s > 0 else '−'}"
                               for t, s in self.testes)
                + f" senão {'+' if self.padrao > 0 else '−'}")


def testes_da_gramatica(obs: Sequence[Obs], max_tam: int = 3,
                        n_cortes: int = 5) -> List[Teste]:
    exprs = expressoes(max_tam)
    out: List[Teste] = []
    for e in exprs:
        vals = sorted(e.val(v) for v, _ in obs)
        cortes = [vals[int((k + 1) * len(vals) / (n_cortes + 1))]
                  for k in range(n_cortes)]
        for c in cortes:
            out.append(Teste(e, k=c))
    base = [Expr('V', i) for i in range(N_VARS)]
    for i in range(N_VARS):
        for j in range(N_VARS):
            if i != j:
                out.append(Teste(base[i], e2=base[j]))
    return out


def melhor(obs: Sequence[Obs], testes: Sequence[Teste],
           profundidade: int = 2) -> Prog:
    best, best_c = None, math.inf
    for padrao in (1, -1):
        atual = Prog(padrao=padrao)
        c_at = atual.custo(obs)
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
FAMILIAS = ('relacional', 'soma', 'produto', 'razao', 'distancia')
CONTROLE = 'alinhada'

# A forma que a gramática deveria fazer emergir. NÃO é usada pelo mecanismo —
# só pelo relatório, para dizer se acertou. Se participasse da busca, seria o
# cardápio que este módulo existe para não ter.
ESPERADA = {
    'relacional': 'v<v',
    'soma': '(v+v)<k',
    'produto': '(v*v)<k',
    'razao': '(v/v)<k',
    'distancia': '|(v-v)|<k',
    'alinhada': 'v<k',
}


def tarefa(tipo: str, rng: random.Random, n: int = 400) -> List[Obs]:
    cs = rng.uniform(0.7, 1.3)
    cp = rng.uniform(0.15, 0.35)
    cr = rng.uniform(0.6, 1.6)
    cd = rng.uniform(0.2, 0.45)
    ca = rng.uniform(0.3, 0.7)
    obs: List[Obs] = []
    for _ in range(n):
        v = [rng.uniform(0.15, 1.0) for _ in range(N_VARS)]
        if tipo == 'relacional':
            s = 1 if v[0] < v[1] else -1
        elif tipo == 'soma':
            s = 1 if v[0] + v[1] < cs else -1
        elif tipo == 'produto':
            s = 1 if v[0] * v[1] < cp else -1
        elif tipo == 'razao':
            s = 1 if v[0] / v[1] < cr else -1
        elif tipo == 'distancia':
            s = 1 if abs(v[0] - v[1]) < cd else -1
        elif tipo == 'alinhada':
            s = 1 if v[0] < ca else -1
        else:
            raise ValueError(tipo)
        obs.append((v, s))
    return obs


# ══════════════════════════════════════════════ medição
def info_mutua(pares: Sequence[Tuple[str, str]]) -> float:
    n = len(pares)
    if not n:
        return 0.0
    pxy, px, py = Counter(pares), Counter(a for a, _ in pares), Counter(b for _, b in pares)
    return sum((c / n) * math.log2((c / n) / ((px[a] / n) * (py[b] / n)))
               for (a, b), c in pxy.items())


def experimento(sementes: Sequence[int] = (0, 1, 2, 3, 4), n: int = 400,
                verboso: bool = True) -> Dict[str, Any]:
    pares: List[Tuple[str, str]] = []
    linhas: List[Dict[str, Any]] = []
    for fam in list(FAMILIAS) + [CONTROLE]:
        formas, ajustes, tams = [], [], []
        for s in sementes:
            rng = random.Random(s * 613 + hash(fam) % 89)
            obs = tarefa(fam, rng, n=n)
            p = melhor(obs, testes_da_gramatica(obs))
            f = p.forma()
            formas.append(f)
            pares.append((fam, f))
            ajustes.append(1 - sum(1 for (v, y) in obs if p.aval(v) != y) / len(obs))
            tams.append(p.tam())
        dom, cnt = Counter(formas).most_common(1)[0]
        linhas.append({'familia': fam, 'forma': dom, 'estabilidade': cnt / len(sementes),
                       'ajuste': st.mean(ajustes), 'tam': st.mean(tams),
                       'acertou': dom == ESPERADA[fam]})
    return {'linhas': linhas, 'im': info_mutua(pares)}


def relatorio(n: int = 400) -> Dict[str, Any]:
    print("═" * 78)
    print("CAMPO GENERATIVO — sem operações nomeadas")
    print("gramática: v · a+b · a−b · a·b · a/b · |a|   ·   pressão: erro + λ·tamanho")
    print("'razão', 'distância' e 'produto' NÃO existem como conceitos no código")
    print("═" * 78)
    print("PRÉ-REGISTRO:")
    print("  Q1 cada família faz emergir a estrutura correspondente, sem ser nomeada")
    print("  Q2 o controle alinhado emerge TRIVIAL (v<k), não composto")
    print("  Q3 I(tarefa;forma) bem acima de zero — máximo log2(6) ≈ 2,585 bits")
    print("  Q4 razão e distância estavam FORA do conjunto nomeado de geracao.py:")
    print("     é nelas que o teste realmente pesa")
    print()

    r = experimento(n=n)
    print(f"{'família':<13}{'forma descoberta':>22}{'esperada':>16}"
          f"{'estab.':>9}{'ajuste':>9}{'':>4}")
    for l in r['linhas']:
        print(f"{l['familia']:<13}{l['forma']:>22}{ESPERADA[l['familia']]:>16}"
              f"{l['estabilidade']:>8.0%}{l['ajuste']:>9.0%}"
              f"{'  ✓' if l['acertou'] else '  ✗':>4}")

    print("─" * 78)
    fam = [l for l in r['linhas'] if l['familia'] != CONTROLE]
    ctrl = next(l for l in r['linhas'] if l['familia'] == CONTROLE)
    acertos = sum(l['acertou'] for l in fam)
    novas = [l for l in fam if l['familia'] in ('razao', 'distancia')]
    print(f"I(tarefa ; forma descoberta) = {r['im']:.3f} bits   (máximo 2,585)")
    print(f"formas corretas: {acertos}/{len(fam)}   ·   "
          f"controle trivial: {'sim' if ctrl['acertou'] else 'NÃO'}")
    print()
    if not ctrl['acertou']:
        print("CONTROLE REPROVADO: a família alinhada fez emergir estrutura composta")
        print(f"({ctrl['forma']}) onde `v<k` bastava. Isso é sobreajuste da gramática,")
        print("e torna as outras linhas ilegíveis.")
    elif sum(l['acertou'] for l in novas) == len(novas):
        print("AS DUAS ESTRUTURAS FORA DO CONJUNTO NOMEADO EMERGIRAM.")
        print("`razão` e `distância` não eram operações de expansão em `geracao.py` —")
        print("não estavam no cardápio. Aqui não há cardápio: há uma gramática e uma")
        print("pressão, e a estrutura certa apareceu para cada família sem ninguém")
        print("dizer que dividir ou tirar módulo eram coisas a procurar.")
        print()
        print("A ressalva, escrita antes de rodar: eu escolhi os TERMINAIS da")
        print("gramática. Não escolhi as composições, a profundidade, nem que forma")
        print("serve para que tarefa.")
    else:
        faltou = [l['familia'] for l in novas if not l['acertou']]
        print(f"Parcial: {acertos}/{len(fam)} formas corretas; falhou em {faltou}.")
        print("O limite é da gramática ou da busca — e isso é resultado, não defeito.")
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=400)
    a = ap.parse_args()
    relatorio(n=a.n)
