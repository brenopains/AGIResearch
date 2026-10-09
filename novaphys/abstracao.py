"""
novaphys.abstracao — de onde veio o PARTIR?
============================================

A sequência de bancadas deixou uma pergunta em pé, e ela é a única que sobrou:

    operadores.py    qual dos operadores existentes eu uso?          resolvido
    hipoteses.py     como componho um operador recursivamente?       resolvido
    ESTA             de onde veio o próprio operador?                aberto

Porque PARTIR fui eu que escrevi. O sistema compõe estruturas que não estavam
em lugar nenhum — dentro do espaço que aquela escolha define.

A ARMADILHA, DITA ANTES DO DESENHO
-----------------------------------
Escrever um módulo que "descobre PARTIR" é o mesmo regresso um nível acima: eu
escolheria a representação em que a descoberta acontece. Trocar lista-de-formas
por lista-de-operadores foi progresso; trocar lista-de-operadores por
lista-de-descobridores-de-operadores não seria.

A versão honesta exige duas coisas verificáveis:

  1. O SUBSTRATO É ESTRITAMENTE ABAIXO DO NÍVEL DE OPERADOR. Nada aqui se chama
     PARTIR, nem tem a forma de PARTIR. São peças genéricas de programa:
     ler um campo, comparar, ramificar, devolver sinal. `IF` é controle de
     fluxo, não estrutura de domínio — a diferença é que `IF` sozinho não sabe
     em QUE comparar nem ONDE cortar.

  2. O SUBSTRATO EXPRESSA COISAS QUE NÃO SÃO PARTIR. Se só programas do tipo
     PARTIR fossem escrevíveis, "descobrir PARTIR" seria vacuidade. Por isso as
     tarefas incluem mundos resolvidos por programas de outra forma — a
     abstração tem de ESCOLHER o que promover, e pode escolher errado.

O MECANISMO: BIBLIOTECA POR COMPRESSÃO
---------------------------------------
Nenhuma etapa se chama `descobrir_operador()`.

    resolve N tarefas com programas crus
              ↓
    minera os TRECHOS que se repetem entre soluções
              ↓
    promove a primitiva nomeada quem paga o próprio custo em compressão
              ↓
    tarefas novas podem usar a primitiva promovida

"Separar em regimes" não é dado: é o que sobra quando o mesmo esqueleto aparece
em domínios diferentes e compensa nomear. Um esqueleto que aparece uma vez só
não paga o custo e não vira primitiva — que é o análogo computacional de não
confundir uma coincidência com uma ferramenta.

O TESTE QUE VALE
-----------------
Não "ele descobriu PARTIR?" — isso eu poderia arranjar. O teste é
TRANSFERÊNCIA CEGA:

    Fase 1   tarefas do domínio A, onde o corte é numa variável
    Fase 2   mineração e promoção, sem supervisão
    Fase 3   tarefas do domínio B, onde o corte é em OUTRA variável e nenhum
             exemplo de B participou da fase 1
    Fase 4   controle: tarefas monotônicas, onde a primitiva NÃO pode ajudar

Se a primitiva promovida em A for reusada em B e reduzir o custo de descrição,
o operador não foi colocado pelo pesquisador. Se ela for usada também no
controle monotônico, não houve abstração — houve sobreajuste da biblioteca, e
aí o número de transferência não vale nada.
"""

from __future__ import annotations

import itertools
import math
import random
import statistics as st
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Uma observação: (campos, sinal). `campos` é o contexto lido pelo programa.
CAMPOS = ('x', 'z', 'w')
Obs = Tuple[Dict[str, float], int]


# ══════════════════════════════════════════════ o substrato
@dataclass(frozen=True)
class No:
    """
    Um nó de programa. As peças são deliberadamente genéricas:

        VAR(c)        lê o campo c do contexto
        NUM(v)        constante
        MENOR(a,b)    comparação
        SE(p,A,B)     ramificação
        SINAL(s)      devolve +1 ou −1

    Nenhuma delas é "separar em regimes". SE é controle de fluxo: sozinho ele
    não sabe em que comparar nem onde cortar. É a COMBINAÇÃO
    SE(MENOR(VAR(·), NUM(·)), SINAL, SINAL) que tem esse significado — e é
    justamente ela que precisa emergir da mineração, não da minha mão.
    """
    op: str
    args: Tuple[Any, ...] = ()

    def aval(self, ctx: Dict[str, float]) -> Any:
        o = self.op
        if o == 'VAR':
            return ctx.get(self.args[0], 0.0)
        if o == 'NUM':
            return self.args[0]
        if o == 'SINAL':
            return self.args[0]
        if o == 'MENOR':
            return self.args[0].aval(ctx) < self.args[1].aval(ctx)
        if o == 'SE':
            return (self.args[1] if self.args[0].aval(ctx) else self.args[2]).aval(ctx)
        if o == 'LIB':
            return self.args[0].instanciar(self.args[1:]).aval(ctx)
        raise ValueError(o)

    def tamanho(self) -> int:
        if self.op == 'LIB':
            # o ganho da biblioteca: a primitiva promovida custa 1 nó, e não o
            # tamanho do esqueleto que ela substituiu
            return 1 + sum(a.tamanho() for a in self.args[1:] if isinstance(a, No))
        return 1 + sum(a.tamanho() for a in self.args if isinstance(a, No))

    def esqueleto(self) -> str:
        """
        A forma do programa com constantes e nomes de campo trocados por
        buracos. É isto que se compara entre soluções: dois programas que
        cortam em variáveis diferentes e em pontos diferentes têm o MESMO
        esqueleto, e é essa igualdade que faz "separar em regimes" ser uma
        coisa só em vez de mil coincidências.
        """
        o = self.op
        if o in ('VAR', 'NUM'):
            return '□'
        if o == 'SINAL':
            return 's'
        if o == 'LIB':
            return f"@{self.args[0].nome}"
        return f"{o}({','.join(a.esqueleto() for a in self.args if isinstance(a, No))})"

    def buracos(self) -> List[Any]:
        """Os valores que preenchem os buracos, na ordem em que aparecem."""
        o = self.op
        if o in ('VAR', 'NUM'):
            return [self.args[0]]
        if o == 'SINAL':
            return []
        out: List[Any] = []
        for a in self.args:
            if isinstance(a, No):
                out.extend(a.buracos())
        return out

    def tipos_dos_buracos(self) -> List[str]:
        """
        'campo' ou 'numero', na ordem em que os buracos aparecem.

        Necessário porque uma primitiva profunda — duas comparações encadeadas,
        por exemplo — tem quatro buracos, e não dois. A primeira versão gerava
        sempre dois valores e estourava a instanciação em qualquer primitiva que
        não fosse a mais rasa.
        """
        o = self.op
        if o == 'VAR':
            return ['campo']
        if o == 'NUM':
            return ['numero']
        if o == 'SINAL':
            return []
        out: List[str] = []
        for a in self.args:
            if isinstance(a, No):
                out.extend(a.tipos_dos_buracos())
        return out

    def __str__(self) -> str:
        o = self.op
        if o == 'VAR':
            return self.args[0]
        if o == 'NUM':
            return f"{self.args[0]:.3g}"
        if o == 'SINAL':
            return '+' if self.args[0] > 0 else '−'
        if o == 'MENOR':
            return f"{self.args[0]}<{self.args[1]}"
        if o == 'SE':
            return f"se({self.args[0]}){{{self.args[1]}}}senão{{{self.args[2]}}}"
        if o == 'LIB':
            return f"{self.args[0].nome}({','.join(str(a) for a in self.args[1:])})"
        return o


@dataclass
class Primitiva:
    """Um esqueleto promovido: vira uma peça de uma unidade de custo."""
    nome: str
    esqueleto: str
    modelo: No                     # um exemplar, para instanciar
    n_buracos: int
    vezes_vista: int = 0

    def instanciar(self, valores: Sequence[Any]) -> No:
        it = iter(valores)

        def rec(n: No) -> No:
            if n.op in ('VAR', 'NUM'):
                v = next(it)
                return No('VAR', (v,)) if isinstance(v, str) else No('NUM', (v,))
            if n.op == 'SINAL':
                return n
            return No(n.op, tuple(rec(a) if isinstance(a, No) else a for a in n.args))

        return rec(self.modelo)


# ══════════════════════════════════════════════ busca de programas
MAIS, MENOS = No('SINAL', (1,)), No('SINAL', (-1,))


def _folhas(ctx_campos: Sequence[str], cortes: Sequence[float]) -> List[No]:
    return ([No('VAR', (c,)) for c in ctx_campos] + [No('NUM', (v,)) for v in cortes])


def enumerar(campos: Sequence[str], cortes: Sequence[float],
             biblioteca: Sequence[Primitiva]) -> List[No]:
    """
    Todos os programas até a profundidade permitida. Enumeração pequena de
    propósito: o interesse é a ABSTRAÇÃO, não a busca. Uma busca poderosa
    esconderia a diferença que a biblioteca faz.
    """
    progs: List[No] = [MAIS, MENOS]
    testes = [No('MENOR', (No('VAR', (c,)), No('NUM', (v,))))
              for c in campos for v in cortes]
    for t in testes:
        progs.append(No('SE', (t, MAIS, MENOS)))
        progs.append(No('SE', (t, MENOS, MAIS)))
    # profundidade 2: um dos ramos pode ser outro teste
    for t in testes:
        for t2 in testes:
            if t2 is t:
                continue
            progs.append(No('SE', (t, No('SE', (t2, MAIS, MENOS)), MENOS)))
            progs.append(No('SE', (t, MAIS, No('SE', (t2, MAIS, MENOS)))))
    # o que a biblioteca acrescenta: a primitiva promovida, com seus buracos
    # preenchidos conforme o TIPO de cada um
    for p in biblioteca:
        tipos = p.modelo.tipos_dos_buracos()
        if len(tipos) > 4:                      # teto: mantém a enumeração viável
            continue
        opcoes = [list(campos) if t == 'campo' else list(cortes) for t in tipos]
        for combo in itertools.product(*opcoes):
            progs.append(No('LIB', (p,) + combo))
    return progs


def resolver(obs: Sequence[Obs], campos: Sequence[str], cortes: Sequence[float],
             biblioteca: Sequence[Primitiva],
             lam: float = 1.5) -> Dict[str, Any]:
    """
    Melhor programa por comprimento de descrição: erros + λ·tamanho.

    O mesmo critério de sempre — explicar melhor tem de pagar a complexidade.
    Aqui ele também é o que dá vantagem à biblioteca: uma primitiva promovida
    custa 1 nó em vez dos 5 do esqueleto que ela resume.
    """
    progs = enumerar(campos, cortes, biblioteca)
    melhor, melhor_c = None, math.inf
    for p in progs:
        erros = sum(1 for (ctx, s) in obs if p.aval(ctx) != s)
        c = erros + lam * p.tamanho()
        if c < melhor_c:
            melhor, melhor_c = p, c
    acertos = sum(1 for (ctx, s) in obs if melhor.aval(ctx) == s)
    return {'programa': melhor, 'custo': melhor_c, 'tamanho': melhor.tamanho(),
            'ajuste': acertos / max(1, len(obs)), 'n_avaliados': len(progs)}


# ══════════════════════════════════════════════ a abstração
def minerar(solucoes: Sequence[No], min_vezes: int = 3,
            lam: float = 1.5) -> List[Primitiva]:
    """
    Promove a primitiva o esqueleto que se repete E que compensa nomear.

    Duas condições, e a segunda é a que impede a biblioteca de inchar:

      repete-se      aparece em pelo menos `min_vezes` soluções distintas
      compensa       o que se economiza (tamanho−1, por uso) supera o custo de
                     carregar mais uma primitiva

    Um esqueleto visto uma vez é coincidência. Um esqueleto visto muitas vezes
    e que não economiza nada é ruído com frequência alta. Só o que passa nas
    duas vira ferramenta.
    """
    freq: Dict[str, List[No]] = {}
    for s in solucoes:
        for sub in _subarvores(s):
            if sub.op in ('SINAL', 'VAR', 'NUM'):
                continue
            freq.setdefault(sub.esqueleto(), []).append(sub)
    out: List[Primitiva] = []
    for esq, exemplares in sorted(freq.items(), key=lambda kv: -len(kv[1])):
        if len(exemplares) < min_vezes:
            continue
        modelo = exemplares[0]
        # A primitiva tem de ser um PROGRAMA, não um pedaço solto. `MENOR(□,□)`
        # aparece 7× e devolve um booleano — usá-la como programa compararia
        # True/False com ±1 e daria erro em toda observação. É restrição de
        # tipo, não ajuste: uma peça só entra na biblioteca se puder ocupar o
        # lugar de uma solução inteira.
        if modelo.op not in ('SE', 'SINAL', 'LIB'):
            continue
        economia = (modelo.tamanho() - 1) * len(exemplares) * lam
        if economia <= lam * modelo.tamanho():      # não paga o próprio custo
            continue
        out.append(Primitiva(nome=f"P{len(out)+1}", esqueleto=esq, modelo=modelo,
                             n_buracos=len(modelo.buracos()),
                             vezes_vista=len(exemplares)))
        if len(out) >= 2:
            break
    return out


def _subarvores(n: No) -> List[No]:
    out = [n]
    for a in n.args:
        if isinstance(a, No):
            out.extend(_subarvores(a))
    return out


# ══════════════════════════════════════════════ os mundos
def tarefa(dominio: str, rng: random.Random, n: int = 60) -> List[Obs]:
    """
    Cada domínio decide o sinal por uma regra diferente. Nenhum deles menciona
    "regime": a regra é só uma função do contexto.
    """
    obs: List[Obs] = []
    cx = rng.uniform(3.0, 7.0)
    cz = rng.uniform(0.3, 0.7)
    cw = rng.uniform(0.3, 0.7)
    for _ in range(n):
        ctx = {'x': rng.uniform(0, 10), 'z': rng.random(), 'w': rng.random()}
        if dominio == 'A':          # corte em x
            s = 1 if ctx['x'] < cx else -1
        elif dominio == 'B':        # corte em w — nunca visto na fase 1
            s = 1 if ctx['w'] < cw else -1
        elif dominio == 'mono':     # controle: nenhum regime
            s = 1
        elif dominio == 'duplo':    # dois testes encadeados: NÃO é o esqueleto de A
            s = 1 if (ctx['x'] < cx and ctx['z'] < cz) else -1
        else:
            raise ValueError(dominio)
        obs.append((ctx, s))
    return obs


# ══════════════════════════════════════════════ o experimento
def experimento(n_fase1: int = 12, n_teste: int = 12, semente: int = 0,
                verboso: bool = True) -> Dict[str, Any]:
    rng = random.Random(semente)
    cortes = [1.0, 3.0, 5.0, 7.0, 9.0, 0.25, 0.5, 0.75]
    campos = list(CAMPOS)

    # ── Fase 1: experiência. Só domínio A e `duplo`, para a mineração ter de
    #    ESCOLHER entre dois esqueletos concorrentes.
    solucoes: List[No] = []
    for i in range(n_fase1):
        dom = 'A' if i % 2 == 0 else 'duplo'
        r = resolver(tarefa(dom, rng), campos, cortes, [])
        solucoes.append(r['programa'])

    # ── Fase 2: mineração e promoção, sem supervisão
    lib = minerar(solucoes)
    if verboso:
        print("primitivas promovidas (ninguém disse que deviam existir):")
        for p in lib:
            print(f"  {p.nome}  esqueleto {p.esqueleto}  vista {p.vezes_vista}×  "
                  f"exemplar: {p.modelo}")
        if not lib:
            print("  nenhuma — nada se repetiu o suficiente para compensar")

    # ── Fases 3 e 4: transferência cega e controle
    res: Dict[str, Any] = {}
    for dom in ('B', 'mono'):
        sem_lib, com_lib, usou = [], [], 0
        for _ in range(n_teste):
            obs = tarefa(dom, rng)
            a = resolver(obs, campos, cortes, [])
            b = resolver(obs, campos, cortes, lib)
            sem_lib.append(a['custo']); com_lib.append(b['custo'])
            usou += int('LIB' in [n.op for n in _subarvores(b['programa'])])
        res[dom] = {'custo_sem': st.mean(sem_lib), 'custo_com': st.mean(com_lib),
                    'usou_primitiva': usou / n_teste}
    return {'biblioteca': lib, 'resultados': res}


def relatorio(sementes: Tuple[int, ...] = (0, 1, 2, 3, 4)) -> Dict[str, Any]:
    print("═" * 78)
    print("DE ONDE VEIO O OPERADOR — abstração por compressão")
    print("substrato: ler campo · constante · comparar · ramificar · sinal")
    print("nada aqui se chama PARTIR, nem tem a forma de PARTIR")
    print("═" * 78)

    linhas = []
    for i, s in enumerate(sementes):
        r = experimento(semente=s, verboso=(i == 0))
        linhas.append(r)
    print()

    n_lib = st.mean([len(l['biblioteca']) for l in linhas])
    print(f"{'domínio':<10}{'custo sem lib':>15}{'custo com lib':>15}"
          f"{'usou a primitiva':>19}")
    for dom, rot in (('B', 'transferência cega'), ('mono', 'controle monotônico')):
        cs = st.mean([l['resultados'][dom]['custo_sem'] for l in linhas])
        cc = st.mean([l['resultados'][dom]['custo_com'] for l in linhas])
        us = st.mean([l['resultados'][dom]['usou_primitiva'] for l in linhas])
        print(f"{dom:<10}{cs:>15.2f}{cc:>15.2f}{us:>18.0%}   {rot}")

    print("─" * 78)
    usoB = st.mean([l['resultados']['B']['usou_primitiva'] for l in linhas])
    usoM = st.mean([l['resultados']['mono']['usou_primitiva'] for l in linhas])
    ganhoB = (st.mean([l['resultados']['B']['custo_sem'] for l in linhas])
              - st.mean([l['resultados']['B']['custo_com'] for l in linhas]))
    print(f"primitivas promovidas por rodada: {n_lib:.1f}")
    if not n_lib:
        print("Nenhuma abstração emergiu. Sem primitiva promovida não há o que")
        print("transferir, e o experimento não responde à pergunta.")
    elif usoB > 0.8 and usoM < 0.2 and ganhoB > 0:
        print("TRANSFERÊNCIA CEGA: a primitiva minerada no domínio A é reusada no")
        print("domínio B, cujo corte é em outra variável e do qual nenhum exemplo")
        print("participou da mineração — e reduz o comprimento de descrição.")
        print("No controle monotônico ela NÃO é usada, que é o que separa abstração")
        print("de biblioteca inchada.")
        print()
        print("O operador não foi colocado por mim. Saiu de resolver tarefas com")
        print("peças genéricas e notar o que se repetia.")
    elif usoM >= 0.2:
        print(f"A primitiva é usada em {usoM:.0%} das tarefas monotônicas, onde ela não")
        print("pode ajudar. Isso é sobreajuste da biblioteca, e invalida a leitura")
        print("da transferência.")
    else:
        print(f"Transferência {usoB:.0%}, ganho {ganhoB:+.2f}. Não é o suficiente para")
        print("dizer que a abstração pegou.")
    return {'linhas': linhas}


if __name__ == "__main__":
    relatorio()
