"""
novaphys.descoberta — a experiência SELECIONA a abstração, ou o substrato?
==========================================================================

`abstracao.py` mediu três coisas e elas se sustentam: extração, transferência
cega e escopo (0% de uso no controle onde não ajuda).

Mas uma auditoria posterior derrubou a leitura mais forte. Variando a
experiência entre tarefas rasas e tarefas aninhadas, a primitiva promovida foi
sempre a mesma:

    só rasas ............. SE(MENOR(□,□),s,s)  5/5
    metade/metade ........ SE(MENOR(□,□),s,s)  5/5
    só aninhadas ......... SE(MENOR(□,□),s,s)  2/5 · nenhuma 3/5

O motivo é topológico, não experiencial: `SE(MENOR(□,□),s,s)` é **subárvore de
praticamente todo programa não-trivial** daquele substrato. Minerar por
frequência de subárvore garante que ela vença. O espaço de primitivas
descobríveis tinha tamanho efetivo 1, e "descoberta" sem alternativas não
informa nada sobre aprendizado.

    demonstrado      extração · reutilização · transferência · escopo
    NÃO demonstrado  que a experiência seleciona ENTRE abstrações possíveis

DUAS MUDANÇAS DE MECANISMO, EXIGIDAS PELO DIAGNÓSTICO
-------------------------------------------------------
**1. Minerar o esqueleto-RAIZ, não toda subárvore.** Cada tarefa vota uma vez,
na forma que a resolveu. Isso remove a vantagem automática de quem é subárvore
de todo mundo — que era o confound inteiro.

**2. Famílias com oportunidade estrutural comparável.** Três formas compostas
que não são subárvore uma da outra:

    LIMIAR      SE(MENOR(v,c), s, s)                compara valor a constante
    DIFERENÇA   SE(MENOR(MENOS(v,v),c), s, s)       compara DIFERENÇA a constante
    CONJUNÇÃO   SE(E(t,t), s, s)                    exige duas condições juntas

O DESENHO: MATRIZ EXPERIÊNCIA × SUBSTRATO
------------------------------------------
                    exp. LIMIAR   exp. DIFERENÇA   exp. CONJUNÇÃO
    substrato S1        ?              ?                ?          (só MENOR)
    substrato S2        ?              ?                ?          (+ MENOS)
    substrato S3        ?              ?                ?          (+ E)

A pergunta deixa de ser "qual primitiva apareceu" e passa a ser:

    I(experiência ; primitiva promovida) > 0 ?

Se a abstração acompanha a EXPERIÊNCIA, a informação mútua é alta e o
resultado muda de categoria. Se acompanha o SUBSTRATO — mesma primitiva em
toda coluna — o que existe é compressão estrutural, não descoberta orientada
por experiência. Em S1 a informação mútua DEVE ser baixa por construção: sem
`MENOS` e sem `E` não há o que escolher. S1 é o controle que mostra como se
parece a ausência do fenômeno.

TRÊS CRITÉRIOS QUE PODEM DISCORDAR
-----------------------------------
Frequência não é relevância. Uma forma rara pode comprimir muito; uma forma
frequente pode não economizar nada. O relatório imprime os três lado a lado —
frequência, compressão e utilidade em tarefas novas — justamente para que a
discordância entre eles fique visível em vez de ser mediada por um escore só.
"""

from __future__ import annotations

import itertools
import math
import random
import statistics as st
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

Obs = Tuple[List[float], int]        # (vetor de contexto, sinal)
N_VARS = 3


# ══════════════════════════════════════════════ programas
@dataclass(frozen=True)
class No:
    op: str
    args: Tuple[Any, ...] = ()

    def aval(self, v: Sequence[float]) -> Any:
        o = self.op
        if o == 'VAR':
            return v[self.args[0]]
        if o == 'NUM':
            return self.args[0]
        if o == 'SINAL':
            return self.args[0]
        if o == 'MENOS':
            return self.args[0].aval(v) - self.args[1].aval(v)
        if o == 'MENOR':
            return self.args[0].aval(v) < self.args[1].aval(v)
        if o == 'E':
            return self.args[0].aval(v) and self.args[1].aval(v)
        if o == 'SE':
            return (self.args[1] if self.args[0].aval(v) else self.args[2]).aval(v)
        raise ValueError(o)

    def tamanho(self) -> int:
        return 1 + sum(a.tamanho() for a in self.args if isinstance(a, No))

    def esqueleto(self) -> str:
        o = self.op
        if o in ('VAR', 'NUM'):
            return '□'
        if o == 'SINAL':
            return 's'
        return f"{o}({','.join(a.esqueleto() for a in self.args if isinstance(a, No))})"

    def familia(self) -> str:
        """
        A que família de forma composta este programa pertence.

        Nomeado só para o relatório — o mecanismo nunca consulta esta função.
        Se ela participasse da busca ou da promoção, seria a lista de formas
        que este módulo existe para não ter.
        """
        e = self.esqueleto()
        if 'E(' in e:
            return 'CONJUNÇÃO'
        if 'MENOS(' in e:
            return 'DIFERENÇA'
        if 'MENOR(' in e:
            return 'LIMIAR'
        return 'constante'

    def __str__(self) -> str:
        o = self.op
        if o == 'VAR':
            return f"v{self.args[0]}"
        if o == 'NUM':
            return f"{self.args[0]:.2f}"
        if o == 'SINAL':
            return '+' if self.args[0] > 0 else '−'
        if o == 'MENOS':
            return f"({self.args[0]}−{self.args[1]})"
        if o == 'MENOR':
            return f"{self.args[0]}<{self.args[1]}"
        if o == 'E':
            return f"({self.args[0]} e {self.args[1]})"
        return f"se({self.args[0]}){{{self.args[1]}}}senão{{{self.args[2]}}}"


MAIS, MENOS_S = No('SINAL', (1,)), No('SINAL', (-1,))


# ══════════════════════════════════════════════ substratos
SUBSTRATOS = {
    'S1 (só MENOR)': {'menos': False, 'e': False},
    'S2 (+ MENOS)': {'menos': True, 'e': False},
    'S3 (+ MENOS, E)': {'menos': True, 'e': True},
}


def cortes(obs: Sequence[Obs], n: int = 4) -> List[List[float]]:
    out = []
    for i in range(N_VARS):
        vals = sorted(o[0][i] for o in obs)
        out.append([vals[int((k + 1) * len(vals) / (n + 1))] for k in range(n)])
    return out


def programas(obs: Sequence[Obs], sub: Dict[str, bool]) -> List[No]:
    ct = cortes(obs)
    testes: List[No] = []
    for i in range(N_VARS):
        for c in ct[i]:
            testes.append(No('MENOR', (No('VAR', (i,)), No('NUM', (c,)))))
    if sub['menos']:
        difs = sorted({round(a - b, 3) for i in range(N_VARS) for j in range(N_VARS)
                       if i != j for a in ct[i][:2] for b in ct[j][:2]})[:5]
        for i in range(N_VARS):
            for j in range(N_VARS):
                if i == j:
                    continue
                for c in difs:
                    testes.append(No('MENOR', (
                        No('MENOS', (No('VAR', (i,)), No('VAR', (j,)))),
                        No('NUM', (c,)))))
    conj: List[No] = []
    if sub['e']:
        simples = [t for t in testes if t.args[0].op == 'VAR']
        for a in simples:
            for b in simples:
                if a.args[0].args[0] >= b.args[0].args[0]:
                    continue
                conj.append(No('E', (a, b)))

    progs: List[No] = [MAIS, MENOS_S]
    for t in testes + conj:
        progs.append(No('SE', (t, MAIS, MENOS_S)))
        progs.append(No('SE', (t, MENOS_S, MAIS)))
    # profundidade 2 só com testes simples, para que S1 tenha como aproximar
    simples = [t for t in testes if t.args[0].op == 'VAR']
    for t in simples:
        for t2 in simples:
            if t2 is t:
                continue
            progs.append(No('SE', (t, No('SE', (t2, MAIS, MENOS_S)), MENOS_S)))
    return progs


def resolver(obs: Sequence[Obs], sub: Dict[str, bool],
             lam: float = 1.0) -> Dict[str, Any]:
    melhor, melhor_c = None, math.inf
    for p in programas(obs, sub):
        e = sum(1 for (v, s) in obs if p.aval(v) != s)
        c = e + lam * p.tamanho()
        if c < melhor_c:
            melhor, melhor_c = p, c
    ac = sum(1 for (v, s) in obs if melhor.aval(v) == s) / max(1, len(obs))
    return {'programa': melhor, 'custo': melhor_c, 'ajuste': ac,
            'tamanho': melhor.tamanho()}


# ══════════════════════════════════════════════ verificação de capacidade
def geraveis(obs: Sequence[Obs], sub: Dict[str, bool]) -> set:
    """Os esqueletos que este substrato CONSEGUE construir."""
    return {p.esqueleto() for p in programas(obs, sub)}


class FalhaDeInstrumento(RuntimeError):
    """A medição alegou algo que o substrato não pode produzir."""


def verificar_capacidade(esq: str, obs: Sequence[Obs], sub: Dict[str, bool],
                         rotulo: str) -> None:
    """
    REGRA PERMANENTE:  promovida(p) ⟹ p ∈ G(substrato)

    Se o pipeline reporta uma primitiva que o substrato não consegue gerar, o
    experimento não produziu descoberta falsa — produziu FALHA DO INSTRUMENTO,
    e a diferença entre as duas coisas decide se o número deve ser consertado
    ou descartado.

    Esta trava existe por um caso concreto: a primeira rodada reportou
    CONJUNÇÃO nas nove células, inclusive em S1, que não tem o nó `E`. A causa
    era `'E(' in esqueleto`, e todo esqueleto começa com `"SE("`. O resultado
    só foi pego porque era IMPOSSÍVEL — com famílias de nomes sem substring em
    comum, a tabela sairia plausível e errada.
    """
    if esq not in geraveis(obs, sub):
        raise FalhaDeInstrumento(
            f"{rotulo}: promoveu '{esq}', que este substrato não gera. "
            f"Isto é defeito de medição, não resultado.")


# ══════════════════════════════════════════════ as experiências
# Cada tipo tem VÁRIAS superfícies: variáveis diferentes carregam o sinal, com
# escalas, deslocamentos e direções diferentes, e há distratores. A operação
# exigida é a mesma; o que muda é a aparência. Se a família promovida se
# mantiver, o que foi capturado é regularidade OPERACIONAL — e não correlação
# entre um tipo de tarefa e um rótulo.
SUPERFICIES = 4


def tarefa(tipo: str, rng: random.Random, n: int = 70,
           superficie: int = 0) -> List[Obs]:
    # a superfície decide QUAIS variáveis carregam o sinal, e como
    pares = [(0, 1), (1, 2), (2, 0), (0, 2)]
    i, j = pares[superficie % len(pares)]
    escala = [1.0, 3.0, 0.4, 2.0][superficie % 4]
    desloc = [0.0, -1.0, 2.5, 0.0][superficie % 4]
    inverte = superficie % 2 == 1
    c0, c1 = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)
    cd = rng.uniform(-0.3, 0.3)
    obs: List[Obs] = []
    for _ in range(n):
        base = [rng.random() for _ in range(N_VARS)]
        if tipo == 'LIMIAR':
            s = 1 if base[i] < c0 else -1
        elif tipo == 'DIFERENÇA':
            s = 1 if (base[i] - base[j]) < cd else -1
        elif tipo == 'CONJUNÇÃO':
            s = 1 if (base[i] < c0 and base[j] < c1) else -1
        else:
            raise ValueError(tipo)
        if inverte:
            s = -s
        v = list(base)
        v[i] = base[i] * escala + desloc
        v[j] = base[j] * escala + desloc
        obs.append((v, s))
    return obs


# ══════════════════════════════════════════════ mineração por raiz
def promover(solucoes: Sequence[No], min_votos: int = 3):
    """
    Cada tarefa vota UMA vez, no esqueleto que a resolveu.

    Minerar por subárvore foi o que produziu a inevitabilidade da rodada
    anterior: `SE(MENOR(□,□),s,s)` é subárvore de quase todo programa, então
    vencia por topologia. Contando raízes, cada experiência pesa igual e a
    forma promovida pode, em princípio, mudar com a experiência — que é
    exatamente o que este módulo quer medir.
    """
    votos = Counter(s.esqueleto() for s in solucoes if s.op == 'SE')
    if not votos:
        return None
    esq, n = votos.most_common(1)[0]
    if n < min_votos:
        return None
    exemplar = next(s for s in solucoes if s.op == 'SE' and s.esqueleto() == esq)
    return (esq, exemplar)


def _ops(n: No) -> set:
    s = {n.op}
    for a in n.args:
        if isinstance(a, No):
            s |= _ops(a)
    return s


def _familia_do_no(n: No) -> str:
    """
    Família pela ÁRVORE, não pelo texto do esqueleto.

    A primeira versão testava `'E(' in esqueleto` — e todo esqueleto começa com
    `"SE("`, que contém `"E("`. Resultado: a matriz inteira reportou CONJUNÇÃO,
    inclusive no substrato S1, que não tem o nó `E` e é incapaz de construí-la.
    Um resultado impossível é sorte: se as três famílias tivessem nomes sem
    substring em comum, a tabela sairia plausível e errada.
    """
    ops = _ops(n)
    if 'E' in ops:
        return 'CONJUNÇÃO'
    if 'MENOS' in ops:
        return 'DIFERENÇA'
    if 'MENOR' in ops:
        return 'LIMIAR'
    return '—'


# ══════════════════════════════════════════════ informação mútua
def info_mutua(pares: Sequence[Tuple[str, str]]) -> float:
    """I(experiência ; primitiva promovida), em bits."""
    n = len(pares)
    if n == 0:
        return 0.0
    pxy = Counter(pares)
    px = Counter(a for a, _ in pares)
    py = Counter(b for _, b in pares)
    i = 0.0
    for (a, b), c in pxy.items():
        p_xy, p_x, p_y = c / n, px[a] / n, py[b] / n
        i += p_xy * math.log2(p_xy / (p_x * p_y))
    return i


# ══════════════════════════════════════════════ o experimento
def experimento(n_tarefas: int = 7, sementes: Sequence[int] = (0, 1, 2, 3, 4),
                n_novas: int = 5) -> Dict[str, Any]:
    exps = ['LIMIAR', 'DIFERENÇA', 'CONJUNÇÃO']
    matriz: Dict[str, Dict[str, str]] = {}
    pares_por_sub: Dict[str, List[Tuple[str, str]]] = {}
    criterios: Dict[str, Dict[str, Any]] = {}

    for nome_sub, sub in SUBSTRATOS.items():
        matriz[nome_sub] = {}
        pares: List[Tuple[str, str]] = []
        for exp in exps:
            promovidas: List[str] = []
            freq_sol, comp_sol, util_sol = [], [], []
            for sem in sementes:
                rng = random.Random(sem * 101 + hash(exp) % 97)
                sols, ult_obs = [], None
                for k in range(n_tarefas):
                    ob = tarefa(exp, rng, superficie=k % SUPERFICIES)
                    sols.append(resolver(ob, sub)['programa'])
                    ult_obs = ob
                pr = promover(sols)
                if pr is not None:
                    verificar_capacidade(pr[0], ult_obs, sub, f"{nome_sub}|{exp}")
                esq = pr[0] if pr else None
                promovidas.append(_familia_do_no(pr[1]) if pr else 'nenhuma')
                if esq:
                    votos = Counter(s.esqueleto() for s in sols if s.op == 'SE')
                    freq_sol.append(votos[esq] / n_tarefas)
                    tam = next(s.tamanho() for s in sols if s.esqueleto() == esq)
                    comp_sol.append((tam - 1) * votos[esq])
                    novas = [resolver(tarefa(exp, rng, superficie=k % SUPERFICIES), sub)
                             for k in range(n_novas)]
                    util_sol.append(st.mean(x['ajuste'] for x in novas))
            dom = Counter(promovidas).most_common(1)[0][0]
            matriz[nome_sub][exp] = dom
            for p in promovidas:
                pares.append((exp, p))
            criterios[f"{nome_sub}|{exp}"] = {
                'frequencia': st.mean(freq_sol) if freq_sol else 0.0,
                'compressao': st.mean(comp_sol) if comp_sol else 0.0,
                'utilidade': st.mean(util_sol) if util_sol else 0.0,
            }
        pares_por_sub[nome_sub] = pares
    return {'matriz': matriz, 'pares': pares_por_sub, 'criterios': criterios}


def relatorio() -> Dict[str, Any]:
    print("═" * 78)
    print("A EXPERIÊNCIA SELECIONA A ABSTRAÇÃO, OU O SUBSTRATO?")
    print("três famílias compostas que não são subárvore uma da outra;")
    print("mineração pelo esqueleto-RAIZ, para que cada tarefa vote uma vez")
    print("═" * 78)

    r = experimento()
    exps = ['LIMIAR', 'DIFERENÇA', 'CONJUNÇÃO']

    print(f"{'substrato':<18}" + "".join(f"{'exp ' + e:>16}" for e in exps)
          + f"{'I(exp;prim)':>13}")
    ims = {}
    for sub in SUBSTRATOS:
        linha = f"{sub:<18}"
        for e in exps:
            linha += f"{r['matriz'][sub][e]:>16}"
        im = info_mutua(r['pares'][sub])
        ims[sub] = im
        print(linha + f"{im:>12.3f}b")

    print("─" * 78)
    print("TRÊS CRITÉRIOS, lado a lado — para a discordância entre eles aparecer")
    print(f"{'':<18}{'':<14}{'frequência':>12}{'compressão':>12}{'utilidade':>11}")
    for sub in SUBSTRATOS:
        for e in exps:
            c = r['criterios'][f"{sub}|{e}"]
            print(f"{sub:<18}{e:<14}{c['frequencia']:>11.0%}"
                  f"{c['compressao']:>12.1f}{c['utilidade']:>10.0%}")

    print("─" * 78)
    im_s1 = ims['S1 (só MENOR)']
    im_s3 = ims['S3 (+ MENOS, E)']
    print(f"controle S1 (sem alternativas): I = {im_s1:.3f} bits")
    print(f"substrato completo S3 ........: I = {im_s3:.3f} bits")
    print()
    if im_s3 > 0.5 and im_s3 > im_s1 + 0.3:
        print("A EXPERIÊNCIA SELECIONA. Com famílias de oportunidade comparável, a")
        print("primitiva promovida muda conforme a distribuição de tarefas, e a")
        print("informação mútua sobe muito acima do controle sem alternativas.")
        print("Isso é diferente de compressão estrutural: há escolha, e ela é")
        print("determinada pelo que o sistema viu.")
    elif im_s3 < 0.2:
        print("A EXPERIÊNCIA NÃO SELECIONA. Mesmo com três famílias disponíveis, a")
        print("primitiva promovida praticamente não depende da experiência — o que")
        print("existe é compressão estrutural, e a auditoria de `abstracao.py` se")
        print("confirma no caso geral, não só naquele substrato.")
    else:
        print(f"Parcial: I = {im_s3:.3f} bits em S3 contra {im_s1:.3f} no controle.")
        print("Há dependência, mas fraca demais para trocar de categoria.")
    return r


if __name__ == "__main__":
    relatorio()
