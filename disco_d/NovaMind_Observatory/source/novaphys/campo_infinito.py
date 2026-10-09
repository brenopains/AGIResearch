"""
novaphys.campo_infinito — consultar um espaço que não cabe numa lista
======================================================================

`campo_cumulativo.py` tinha um campo FINITO: terminais mais consolidadas, tudo
alcançável por enumeração exaustiva. O ganho vinha de encurtar caminhos dentro
de um espaço que, em princípio, dava para listar.

Aqui o campo é INFINITO por construção. Não há lista. Árvores de expressão são
geradas proceduralmente com profundidade ilimitada — a cada nó, uma moeda
decide entre expandir num operador ou fechar num terminal. O espaço não é
enumerável na prática, então ele não pode ser percorrido: tem de ser
CONSULTADO.

    campo finito     enumera tudo, escolhe o melhor
    campo infinito   amostra, avalia, REFINA a própria amostra, repete

E é o refinamento que corresponde a "desenvolver a forma própria": a estrutura
que sobrevive não é a que foi sorteada — é a que o sistema modificou a partir
do que sorteou, trocando subárvores por outras recém-geradas, guiado pelo erro.

O TESTE, COM ORÇAMENTO PAREADO
--------------------------------
Sem parear o orçamento, "o campo infinito é melhor" é vazio: claro que avaliar
mais candidatos acha mais. Então as duas estratégias recebem EXATAMENTE o mesmo
número de avaliações de candidato:

    ENUMERAÇÃO   gera sistematicamente por tamanho crescente, avalia B
    CAMPO        amostra do gerador procedural, avalia, refina, avalia... até B

A pergunta fica objetiva: com o mesmo custo, consultar-e-refinar um espaço
infinito alcança o que percorrer um espaço finito não alcança?

A PRÉ-CONDIÇÃO É PORTÃO, DE NOVO
----------------------------------
Antes de qualquer mecanismo, força bruta responde: a estrutura-alvo está
realmente fora do alcance da enumeração no orçamento B? Se estiver dentro, a
comparação não mede nada e o experimento não roda. Foi assim que uma bancada
inválida foi barrada na estreia, e é a única trava desta sessão que pegou o
problema ANTES de virar número.

O LIMITE, DITO ANTES DO RESULTADO
-----------------------------------
Um espaço infinito gerado por uma gramática finita continua sendo a MINHA
gramática. "Infinito" aqui quer dizer que não cabe numa lista e que a
profundidade não tem teto — não quer dizer que contenha qualquer estrutura
concebível. O que se testa é se geração ilimitada + busca guiada + automodificação
alcançam o que enumeração limitada não alcança. Isso é uma pergunta real e é
menos do que a palavra "infinito" sugere.

PRÉ-REGISTRO
-------------
  S1  a enumeração, no orçamento B, NÃO resolve a tarefa profunda
  S2  o campo, no MESMO orçamento B, resolve
  S3  no controle raso, o campo devolve estrutura TRIVIAL — a pressão de
      descrição impede que profundidade ilimitada vire barroco
  S4  a estrutura vencedora do campo deve ter sido REFINADA, não sorteada
      pronta: se as amostras iniciais já resolvem, o refinamento não é o que
      está funcionando e o mecanismo não é o que eu digo que é
"""

from __future__ import annotations

import argparse
import math
import random
import statistics as st
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

Obs = Tuple[List[float], int]
N_VARS = 4
LAM = 0.6
EPS = 1e-6
OPS2 = ('+', '-', '*', '/')


# ══════════════════════════════════════════════ expressões
@dataclass(frozen=True)
class E:
    op: str
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
        return x / y if abs(y) > EPS else math.copysign(1e6, x or 1.0)

    def tam(self) -> int:
        if self.op == 'V':
            return 1
        if self.op == 'abs':
            return 1 + self.a.tam()
        return 1 + self.a.tam() + self.b.tam()

    def forma(self) -> str:
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


# ══════════════════════════════════════════════ o campo: geração procedural
def amostrar(rng: random.Random, p_expandir: float = 0.62,
             prof: int = 0, max_prof: int = 6) -> E:
    """
    Uma forma tirada do campo. Sem lista, sem catálogo: a cada nó, uma moeda
    decide entre abrir num operador ou fechar num terminal. O espaço que isto
    gera não é enumerável — só amostrável.
    """
    if prof >= max_prof or rng.random() > p_expandir:
        return E('V', rng.randrange(N_VARS))
    if rng.random() < 0.18:
        return E('abs', amostrar(rng, p_expandir, prof + 1, max_prof))
    return E(rng.choice(OPS2),
             amostrar(rng, p_expandir, prof + 1, max_prof),
             amostrar(rng, p_expandir, prof + 1, max_prof))


def _nos(e: E) -> List[Tuple[E, Tuple[int, ...]]]:
    out = [(e, ())]
    for i, x in enumerate((e.a, e.b)):
        if isinstance(x, E):
            out.extend((s, (i,) + c) for s, c in _nos(x))
    return out


def _trocar(e: E, caminho: Tuple[int, ...], novo: E) -> E:
    if not caminho:
        return novo
    i, resto = caminho[0], caminho[1:]
    args = [e.a, e.b]
    args[i] = _trocar(args[i], resto, novo)
    return E(e.op, args[0], args[1])


def refinar(e: E, rng: random.Random, max_prof: int = 3) -> E:
    """
    A forma PRÓPRIA: troca uma subárvore por outra recém-tirada do campo.

    É isto que distingue consultar de copiar. O que sobrevive não é o que foi
    sorteado — é o que o sistema modificou a partir do sorteado, guiado pelo
    erro. Sem este passo o mecanismo seria só amostragem com muitas tentativas.
    """
    nos = _nos(e)
    _, cam = nos[rng.randrange(len(nos))]
    return _trocar(e, cam, amostrar(rng, prof=0, max_prof=max_prof)) if cam else \
        amostrar(rng, prof=0, max_prof=max_prof)


# ══════════════════════════════════════════════ programas
@dataclass(frozen=True)
class Prog:
    e: E
    k: float
    s: int

    def aval(self, v: Sequence[float]) -> int:
        return self.s if self.e.val(v) < self.k else -self.s

    def custo(self, obs: Sequence[Obs]) -> float:
        err = sum(1 for (v, y) in obs if self.aval(v) != y)
        return err + LAM * self.e.tam()

    def ajuste(self, obs: Sequence[Obs]) -> float:
        return 1 - sum(1 for (v, y) in obs if self.aval(v) != y) / len(obs)


def melhor_corte(e: E, obs: Sequence[Obs], n_cortes: int = 7) -> Prog:
    vals = sorted(e.val(v) for v, _ in obs)
    best, bc = None, math.inf
    for i in range(n_cortes):
        k = vals[int((i + 1) * len(vals) / (n_cortes + 1))]
        for s in (1, -1):
            p = Prog(e, k, s)
            c = p.custo(obs)
            if c < bc:
                best, bc = p, c
    return best


# ══════════════════════════════════════════════ as duas estratégias
def _enumerar_ate(orcamento: int) -> List[E]:
    """Geração sistemática por tamanho crescente, até esgotar o orçamento."""
    base = [E('V', i) for i in range(N_VARS)]
    todas, fronteira = list(base), list(base)
    while len(todas) < orcamento:
        nova: List[E] = []
        for x in fronteira:
            for y in base:
                for o in OPS2:
                    nova.append(E(o, x, y))
            nova.append(E('abs', x))
            if len(todas) + len(nova) >= orcamento:
                break
        if not nova:
            break
        todas.extend(nova)
        fronteira = nova[:60]
    return todas[:orcamento]


def por_enumeracao(obs: Sequence[Obs], orcamento: int) -> Dict[str, Any]:
    best, bc = None, math.inf
    for e in _enumerar_ate(orcamento):
        p = melhor_corte(e, obs)
        if p.custo(obs) < bc:
            best, bc = p, p.custo(obs)
    return {'prog': best, 'ajuste': best.ajuste(obs), 'forma': best.e.forma()}


def por_campo(obs: Sequence[Obs], orcamento: int, semente: int,
              largura: int = 24) -> Dict[str, Any]:
    """
    Consulta e refina. O orçamento conta AVALIAÇÕES, igual à enumeração.
    """
    rng = random.Random(semente)
    gastos = 0
    pop: List[Tuple[float, Prog]] = []
    aj_inicial = 0.0

    while gastos < orcamento and len(pop) < largura:      # consulta inicial
        e = amostrar(rng)
        p = melhor_corte(e, obs)
        pop.append((p.custo(obs), p)); gastos += 1
    pop.sort(key=lambda t: t[0])
    aj_inicial = pop[0][1].ajuste(obs) if pop else 0.0

    while gastos < orcamento:                              # refinamento guiado
        _, pai = pop[rng.randrange(min(len(pop), 8))]
        e = refinar(pai.e, rng)
        p = melhor_corte(e, obs)
        pop.append((p.custo(obs), p)); gastos += 1
        pop.sort(key=lambda t: t[0])
        pop = pop[:largura]

    best = pop[0][1]
    return {'prog': best, 'ajuste': best.ajuste(obs), 'forma': best.e.forma(),
            'ajuste_inicial': aj_inicial, 'refinou': best.ajuste(obs) > aj_inicial}


# ══════════════════════════════════════════════ tarefas
def tarefa(tipo: str, rng: random.Random, n: int = 300) -> List[Obs]:
    c = {'profunda': rng.uniform(0.25, 0.6), 'rasa': rng.uniform(0.3, 0.7)}[tipo]
    obs: List[Obs] = []
    for _ in range(n):
        v = [rng.uniform(0.2, 1.0) for _ in range(N_VARS)]
        if tipo == 'profunda':
            s = 1 if abs(v[0] / v[1] - v[2] * v[3]) < c else -1
        else:
            s = 1 if v[0] < c else -1
        obs.append((v, s))
    return obs


ALVO = '|((v/v)-(v*v))|'


# ══════════════════════════════════════════════ pré-condição
def precondicao(orcamento: int, sementes: Sequence[int],
                verboso: bool = True) -> Dict[str, Any]:
    """A estrutura-alvo está fora do alcance da enumeração neste orçamento?"""
    formas = {e.forma() for e in _enumerar_ate(orcamento)}
    dentro = ALVO in formas
    ajs = []
    for s in sementes:
        obs = tarefa('profunda', random.Random(s * 97 + 5))
        ajs.append(por_enumeracao(obs, orcamento)['ajuste'])
    if verboso:
        print(f"PRÉ-CONDIÇÃO — orçamento {orcamento} avaliações")
        print(f"  formas geradas pela enumeração ....... {len(formas)}")
        print(f"  a forma-alvo {ALVO} está entre elas? {'SIM' if dentro else 'não'}")
        print(f"  ajuste da enumeração na tarefa ....... {st.mean(ajs):.0%}")
    return {'fora_do_alcance': not dentro, 'ajuste_enum': st.mean(ajs)}


# ══════════════════════════════════════════════ experimento
def relatorio(orcamento: int = 1400, n: int = 300,
              sementes: Sequence[int] = (0, 1, 2, 3, 4)) -> Dict[str, Any]:
    print("═" * 78)
    print("CAMPO INFINITO — consultar e refinar, contra enumerar")
    print("orçamento PAREADO: as duas estratégias avaliam o mesmo nº de candidatos")
    print("═" * 78)
    print("PRÉ-REGISTRO:")
    print("  S1  a enumeração NÃO resolve a tarefa profunda no orçamento")
    print("  S2  o campo resolve, com o MESMO orçamento")
    print("  S3  no controle raso o campo devolve estrutura TRIVIAL")
    print("  S4  a forma vencedora foi REFINADA, não sorteada pronta")
    print()

    pc = precondicao(orcamento, sementes)
    print("─" * 78)
    if not pc['fora_do_alcance']:
        print("PRÉ-CONDIÇÃO REPROVADA: a forma-alvo está dentro do alcance da")
        print("enumeração neste orçamento. A comparação não mede nada — não rodo.")
        return {'precondicao': pc}
    print("Pré-condição APROVADA: a forma-alvo está fora do alcance da enumeração.")
    print()

    linhas = []
    for s in sementes:
        obs = tarefa('profunda', random.Random(s * 97 + 5), n=n)
        en = por_enumeracao(obs, orcamento)
        ca = por_campo(obs, orcamento, semente=s)
        linhas.append({'s': s, 'enum': en['ajuste'], 'campo': ca['ajuste'],
                       'forma': ca['forma'], 'inicial': ca['ajuste_inicial'],
                       'refinou': ca['refinou']})

    print(f"{'semente':>8}{'enumeração':>13}{'campo':>9}{'amostra inicial':>18}"
          f"{'refinou':>9}   forma vencedora")
    for l in linhas:
        f = l['forma'] if len(l['forma']) <= 26 else l['forma'][:23] + '…'
        print(f"{l['s']:>8}{l['enum']:>12.0%}{l['campo']:>9.0%}"
              f"{l['inicial']:>17.0%}{'  sim' if l['refinou'] else '  não':>9}   {f}")

    ctrl_c, ctrl_tam = [], []
    for s in sementes:
        obs = tarefa('rasa', random.Random(s * 41 + 3), n=n)
        ca = por_campo(obs, orcamento, semente=s + 100)
        ctrl_c.append(ca['ajuste']); ctrl_tam.append(ca['prog'].e.tam())

    print()
    print(f"controle raso: ajuste {st.mean(ctrl_c):.0%} · "
          f"tamanho médio da forma {st.mean(ctrl_tam):.1f}   (trivial = 1)")

    print("─" * 78)
    m_en, m_ca = st.mean(l['enum'] for l in linhas), st.mean(l['campo'] for l in linhas)
    refinou = sum(l['refinou'] for l in linhas) / len(linhas)
    print(f"ajuste médio — enumeração {m_en:.0%}  ·  campo {m_ca:.0%}  "
          f"({m_ca - m_en:+.0%})")
    if st.mean(ctrl_tam) > 3:
        print()
        print("CONTROLE REPROVADO: no problema raso o campo devolveu estrutura")
        print(f"grande (tamanho {st.mean(ctrl_tam):.1f}). Profundidade ilimitada virou")
        print("barroco, e o ganho na tarefa profunda fica ilegível.")
    elif m_ca > m_en + 0.05 and refinou > 0.6:
        print()
        print("O CAMPO ALCANÇA O QUE A ENUMERAÇÃO NÃO ALCANÇA, com o mesmo custo.")
        print(f"E a forma vencedora foi REFINADA em {refinou:.0%} das execuções — não")
        print("sorteada pronta. Consultar e desenvolver bate percorrer.")
        print()
        print("A ressalva escrita antes: a gramática do campo é minha. 'Infinito'")
        print("quer dizer que não cabe numa lista e não tem teto de profundidade —")
        print("não que contenha qualquer estrutura concebível.")
    else:
        print()
        print(f"Parcial: campo {m_ca - m_en:+.0%} sobre a enumeração, refinou em "
              f"{refinou:.0%}.")
        print("Não fecha o critério.")
    return {'linhas': linhas}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--orcamento', type=int, default=1400)
    ap.add_argument('--n', type=int, default=300)
    a = ap.parse_args()
    relatorio(orcamento=a.orcamento, n=a.n)
