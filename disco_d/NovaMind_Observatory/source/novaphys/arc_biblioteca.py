"""
novaphys.arc_biblioteca — as abstracoes crescem a partir do que ja resolveu?
=============================================================================

A DSL escrita a mao e o teto. Ampliei de 17 para 24 primitivas e o ARC subiu
de 1,0% para 2,2% — mas escrever mais primitivas de grade e especifico do ARC
e nao ensina nada transferivel.

A pergunta que separa mecanismo de gambiarra e a mesma que `campo_cumulativo`,
`campo_recursivo` e `gargalo` mediram em fisica:

    as abstracoes extraidas do que o sistema JA RESOLVEU aumentam quantas
    tarefas NOVAS ele resolve com o MESMO orcamento?

CURRICULO COM SEPARACAO REAL
------------------------------
    TREINO      400 tarefas — e daqui que as macros sao extraidas
    AVALIACAO   400 tarefas — nunca vistas, e onde tudo e medido

Nenhuma macro e minerada da avaliacao.

O ORCAMENTO E A PECA CRITICA
------------------------------
`gargalo.py` mostrou que acrescentar conteudo ao campo AUMENTA o espaco de
busca, e que sem teto isso vence por construcao (+0pp ao dobrar o espaco sem
conteudo novo; mas 12pp de PENALIDADE por molde ruim quando o orcamento e
fixo). Aqui a enumeracao e por profundidade crescente e todos os bracos
avaliam exatamente K programas.

Isso da uma definicao medivel de abstracao util:

    uma macro se paga se move uma solucao da profundidade 3 para a 1

Ela ocupa lugar na profundidade 1, empurrando outros candidatos para depois.
Se nao comprimir profundidade, ela so dilui — e isso aparece no numero.

QUATRO BRACOS, ORCAMENTO CASADO
---------------------------------
    G0        base, 24 primitivas
    G1_freq   macros pelos fragmentos MAIS FREQUENTES nas solucoes do treino
    G1_prof   macros pelos fragmentos que mais COMPRIMEM PROFUNDIDADE
    G1_plac   macros ALEATORIAS, pareadas em numero e comprimento

O braco `freq` esta aqui por um motivo: `politicas.py` mostrou que consolidar
por frequencia na origem e SOBREAJUSTE AO DOMINIO DE DESCOBERTA — `(v*v)`
vencia so em `fusao`, o modulo onde nasceu. Se aquele achado for geral e nao
um acidente da fisica, `freq` deve perder para `prof` aqui tambem. E a
primeira chance de testar aquele mecanismo FORA do dominio onde foi medido.

SEMENTE DO PLACEBO, PRE-REGISTRADA: 20260812

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  L0  PRE-CONDICAO. Se as solucoes do treino forem quase todas de
      profundidade <= 1, nao ha o que comprimir e o experimento e vazio.
      Verificar ANTES. Se falhar, nao rodar o resto.

  L1  o melhor G1 > G0 em >= 1pp E > placebo em >= 1pp
      -> abstrair do proprio trabalho resolvido funciona, e a identidade do
         que se abstrai importa

  L2  G1 ~ placebo, ambos > G0
      -> qualquer macro ajuda; e vantagem de vocabulario, nao acumulacao
         especifica. Mesma leitura de `placebo.py` em fisica

  L3  G1 ~ G0
      -> as abstracoes do treino nao transferem para a avaliacao

  L4  prof > freq
      -> confirma FORA da fisica que frequencia na origem sobreajusta

MINHA PREVISAO, escrita antes de rodar: espero L4 (prof > freq) porque o
mecanismo de `politicas.py` deveria ser geral. Sobre L1 x L2 estou dividido,
e e por isso que o placebo esta aqui. Espero ganho pequeno em qualquer caso —
com 2,2% de base, ha pouquissimas solucoes das quais minerar.
"""

from __future__ import annotations

import argparse
import itertools
import random
import time
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from .arc import (Grid, PRIMITIVAS, DICIONARIO, Programa, ajustar_mapa,
                  aplicar_mapa,
                  carregar, dims, valida)

SEMENTE_PLACEBO = 20260812
PROF = 3
N_MACROS = 6

TREINO = (('C:/Users/usuario/Documents/AGI 2.0/backup_novamind_20260627_170507/data/ARC/data/training').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))
AVALIACAO = (('C:/Users/usuario/Documents/AGI 2.0/backup_novamind_20260627_170507/data/ARC/data/evaluation').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))


# ══════════════════════════════════════════════ busca com orcamento casado
def _aplicar_ops(g: Grid, ops: Sequence[str],
                 macros: Dict[str, Tuple[str, ...]]) -> Optional[Grid]:
    for nome in ops:
        for base in macros.get(nome, (nome,)):
            g = DICIONARIO[base](g)          # type: ignore[assignment,arg-type]
            if not valida(g):
                return None
    return g


def enumerar(nomes: Sequence[str], prof: int):
    """Profundidade crescente — ordem de Occam. Macros se pagam vindo antes."""
    yield ()
    for d in range(1, prof + 1):
        for ops in itertools.product(nomes, repeat=d):
            yield ops


def resolver(treino: Sequence[Tuple[Grid, Grid]], nomes: Sequence[str],
             macros: Dict[str, Tuple[str, ...]], orcamento: int,
             prof: int = PROF) -> Optional[Tuple[Programa, int, int]]:
    """
    Devolve (programa, profundidade_efetiva, programas_avaliados).
    Verificador exato: reproduz TODOS os pares de treino ou nao vale.
    """
    n = 0
    for ops in enumerar(nomes, prof):
        n += 1
        if n > orcamento:
            return None
        saidas, ok = [], True
        for a, b in treino:
            r = _aplicar_ops(a, ops, macros)
            if r is None:
                ok = False
                break
            saidas.append(r)
        if not ok:
            continue
        if all(r == b for r, (_, b) in zip(saidas, treino)):
            return Programa(ops), len(ops), n
        mapa = ajustar_mapa(list(zip(saidas, [b for _, b in treino])))
        if mapa is not None:
            if all(aplicar_mapa(r, mapa) == b
                   for r, (_, b) in zip(saidas, treino)):
                return Programa(ops, mapa), len(ops), n
    return None


def expandir(ops: Sequence[str],
             macros: Dict[str, Tuple[str, ...]]) -> Tuple[str, ...]:
    fora: List[str] = []
    for o in ops:
        fora.extend(macros.get(o, (o,)))
    return tuple(fora)


# ══════════════════════════════════════════════ mineracao de macros
def minerar(solucoes: List[Tuple[str, Tuple[str, ...], int]]
            ) -> Tuple[List[Tuple[str, ...]], List[Tuple[str, ...]]]:
    """
    Dois criterios sobre os MESMOS fragmentos:
      freq  quantas tarefas distintas usam o fragmento
      prof  quanta profundidade o fragmento pouparia (len-1 por uso)

    `politicas.py` prevê que `freq` sobreajusta ao dominio de origem.
    """
    freq: Counter = Counter()
    econ: Counter = Counter()
    for _nome, ops, _d in solucoes:
        vistos = set()
        for tam in (2, 3):
            for i in range(len(ops) - tam + 1):
                frag = tuple(ops[i:i + tam])
                if frag in vistos:
                    continue
                vistos.add(frag)
                freq[frag] += 1
                econ[frag] += tam - 1
    por_freq = [f for f, c in freq.most_common() if c >= 2][:N_MACROS]
    por_prof = [f for f, _ in econ.most_common()
                if freq[f] >= 2][:N_MACROS]
    return por_freq, por_prof


def macros_aleatorias(rng: random.Random,
                      gabarito: Sequence[Tuple[str, ...]]
                      ) -> List[Tuple[str, ...]]:
    """Mesmo numero, mesmos comprimentos, conteudo do gerador pre-registrado."""
    base = [n for n, _ in PRIMITIVAS]
    return [tuple(rng.choice(base) for _ in range(len(g))) for g in gabarito]


# ══════════════════════════════════════════════ avaliacao de um braco
def rodar_braco(tarefas, nome_braco: str, novas: Sequence[Tuple[str, ...]],
                orcamento: int) -> Dict[str, object]:
    base = [n for n, _ in PRIMITIVAS]
    macros: Dict[str, Tuple[str, ...]] = {}
    for i, frag in enumerate(novas):
        macros[f"M{i+1}"] = frag
    nomes = base + list(macros)
    ok, profs, avaliados = 0, [], []
    for _nome, tr, te in tarefas:
        r = resolver(tr, nomes, macros, orcamento)
        if r is None:
            continue
        p, d, n = r
        if all(Programa(expandir(p.ops, macros), p.mapa).aplicar(a) == b
               for a, b in te):
            ok += 1
            profs.append(d)
            avaliados.append(n)
    return {'braco': nome_braco, 'resolvidas': ok,
            'prof_media': (sum(profs) / len(profs)) if profs else 0.0,
            'aval_medio': (sum(avaliados) / len(avaliados)) if avaliados else 0.0,
            'macros': [".".join(m) for m in novas]}


def relatorio() -> None:
    base = [n for n, _ in PRIMITIVAS]
    orcamento = sum(len(base) ** d for d in range(PROF + 1))
    print("=" * 78)
    print("ARC — as abstracoes crescem a partir do que ja resolveu?")
    print("=" * 78)
    print("PRE-REGISTRO:")
    print("  L0  se as solucoes do treino forem quase todas prof<=1, vazio")
    print("  L1  melhor G1 > G0 e > placebo em >=1pp -> identidade importa")
    print("  L2  G1 ~ placebo > G0 -> vantagem de vocabulario")
    print("  L3  G1 ~ G0 -> nao transfere")
    print("  L4  prof > freq -> frequencia na origem sobreajusta, FORA da fisica")
    print()
    print(f"orcamento casado: {orcamento} programas por tarefa, em todos os bracos")
    print(f"semente do placebo, pre-registrada: {SEMENTE_PLACEBO}")

    # ── FASE 1: resolver o TREINO, de onde as macros saem
    print("\n" + "-" * 78)
    print("FASE 1 · resolver o conjunto de TREINO com a base")
    tr_tarefas = carregar(TREINO)
    t0 = time.time()
    solucoes: List[Tuple[str, Tuple[str, ...], int]] = []
    for nome, tr, _te in tr_tarefas:
        r = resolver(tr, base, {}, orcamento)
        if r is not None:
            solucoes.append((nome, r[0].ops, r[1]))
    print(f"  resolvidas no treino: {len(solucoes)}/{len(tr_tarefas)} "
          f"({len(solucoes)/len(tr_tarefas):.1%})  ·  {time.time()-t0:.0f}s")
    dist = Counter(d for _n, _o, d in solucoes)
    print(f"  profundidades: {dict(sorted(dist.items()))}")

    # ── L0: a pre-condicao
    profundas = sum(c for d, c in dist.items() if d >= 2)
    print(f"\nL0 · solucoes com profundidade >=2: {profundas}")
    if profundas < 5:
        print("     PRE-CONDICAO REPROVADA. Quase nao ha solucao composta no")
        print("     treino, entao nao ha fragmento recorrente para abstrair e o")
        print("     experimento nao pode mostrar nada. Nao rodo o resto.")
        return
    print("     aprovada — ha estrutura composta para minerar")

    por_freq, por_prof = minerar(solucoes)
    rng = random.Random(SEMENTE_PLACEBO)
    gabarito = por_prof if por_prof else por_freq
    placebo = macros_aleatorias(rng, gabarito)
    print(f"\n  macros por FREQUENCIA: {[ '.'.join(m) for m in por_freq ]}")
    print(f"  macros por PROFUNDIDADE: {[ '.'.join(m) for m in por_prof ]}")
    print(f"  macros PLACEBO: {[ '.'.join(m) for m in placebo ]}")

    if not por_freq and not por_prof:
        print("\n  nenhum fragmento recorre em 2+ tarefas. Nada a abstrair.")
        return

    # ── FASE 2: avaliar os quatro bracos na AVALIACAO, nunca vista
    print("\n" + "-" * 78)
    print("FASE 2 · AVALIACAO (400 tarefas nunca vistas), orcamento casado")
    av = carregar(AVALIACAO)
    res = []
    for nome, novas in (('G0 base', []), ('G1 freq', por_freq),
                        ('G1 prof', por_prof), ('G1 placebo', placebo)):
        t0 = time.time()
        r = rodar_braco(av, nome, novas, orcamento)
        r['segundos'] = round(time.time() - t0)
        res.append(r)
        print(f"  {nome:<12} {r['resolvidas']:>3}/{len(av)} = "
              f"{r['resolvidas']/len(av):>5.1%}   prof media "
              f"{r['prof_media']:.2f}   {r['segundos']}s")

    print("\n" + "=" * 78)
    d = {r['braco']: r['resolvidas'] / len(av) for r in res}
    g0, plac = d['G0 base'], d['G1 placebo']
    melhor = max(d['G1 freq'], d['G1 prof'])
    print(f"{'braco':<14}{'resolvidas':>12}{'vs G0':>10}")
    print("-" * 78)
    for r in res:
        p = r['resolvidas'] / len(av)
        print(f"{r['braco']:<14}{p:>11.1%}{p-g0:>+10.1%}")
    print()
    if melhor >= g0 + 0.01 and melhor >= plac + 0.01:
        print("L1 · ABSTRAIR DO PROPRIO TRABALHO FUNCIONA, e a identidade importa.")
    elif melhor >= g0 + 0.01 and abs(melhor - plac) < 0.01:
        print("L2 · VANTAGEM DE VOCABULARIO. Macro aleatoria pareada empata com")
        print("     macro minerada — o ganho vem de ter macro, nao de qual.")
    else:
        print("L3 · NAO TRANSFERE. As abstracoes do treino nao ajudam na avaliacao.")
    print()
    if d['G1 prof'] > d['G1 freq']:
        print("L4 · prof > freq. Confirma FORA da fisica que minerar por")
        print("     frequencia na origem sobreajusta ao dominio de descoberta.")
    elif d['G1 freq'] > d['G1 prof']:
        print("L4 · freq > prof. CONTRARIA o mecanismo medido em `politicas.py`.")
    else:
        print("L4 · empate entre freq e prof — nao discrimina.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.parse_args()
    relatorio()
