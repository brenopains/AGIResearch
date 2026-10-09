"""
novaphys.arc_inducao — induzir do delta, em vez de adivinhar o programa
========================================================================

O PADRAO DE TODOS OS FRACASSOS DE HOJE
----------------------------------------
    biblioteca      recombina primitivas velhas   -> comprime, nao estende
    guia            reordena primitivas velhas    -> +1 tarefa em 400
    combinadores    reestrutura primitivas velhas -> empate 14 x 14

Tudo opera DENTRO de um conjunto fixo de primitivas. Macro de primitiva velha
nao alcanca o que primitiva velha nao alcanca — isso e quase tautologico, e os
tres experimentos mediram exatamente essa tautologia.

A INVERSAO
------------
Toda primitiva ate agora e `Grade -> Grade`: o sistema ADIVINHA a
transformacao inteira e depois testa. Sao 46 mil chutes por tarefa.

Mas a diferenca entre entrada e saida E a definicao do problema. Entao:

    em vez de   gerar programa -> testar
    fazer       computar o delta -> INDUZIR a regra

Para cada celula que mudou, olhar o contexto local e ajustar
`contexto -> nova cor`, consistente em TODOS os pares de treino. Isso nao e
busca, e construcao. Custo O(celulas), nao exponencial.

O RISCO E MEMORIZACAO, E E TRATADO COM OCCAM
----------------------------------------------
Com vizinhanca de 8 e 11 cores, o espaco de contextos e 11^9. Quase toda
celula teria contexto unico, a tabela memorizaria, o treino bateria 100% e o
teste seria lixo.

O controle e a mesma disciplina de MDL do `novaphys`: tentar conjuntos de
atributos em ordem CRESCENTE de complexidade e ficar com o PRIMEIRO que for
consistente.

    1  cor propria                          recoloracao pura
    2  cor + esta na borda
    3  cor + quantos vizinhos nao-fundo
    4  cor + multiconjunto dos 4 vizinhos
    5  cor + os 4 vizinhos ordenados
    6  cor + os 8 vizinhos ordenados

Mais um guarda explicito: se a tabela tiver mais regras que uma fracao das
celulas, ela memorizou e e rejeitada.

ESCOPO
-------
So se aplica a tarefas em que TODOS os pares tem entrada e saida da mesma
forma — 63% dos pares do ARC. E a maior familia que a busca por cadeias nao
toca: preencher, recolorir por vizinhanca, propagar, crescer. ZERO das 14
solucoes atuais sao dessas.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  I0  PRE-CONDICAO. O nivel 1 (cor propria) tem que resolver as tarefas de
      recoloracao pura. Se nem isso, a inducao esta quebrada e o resto nao
      se le.

  I1  >= 5 tarefas que a busca por programa NAO resolve
      -> a familia local existe e e alcancavel por construcao, e o mecanismo
         estende alcance, coisa que biblioteca/guia/combinadores nao fizeram

  I2  0 tarefas novas
      -> regras locais nao cobrem a familia de mesma forma; o que falta e
         estrutura GLOBAL, e a inversao nao e o caminho

  I3  SOBREAJUSTE, medido explicitamente: quantas tarefas a regra ajusta no
      treino e ERRA no teste. Se essa taxa for alta, a ordenacao por Occam
      nao esta segurando e o numero de I1 nao vale.

MINHA PREVISAO, escrita antes de rodar: espero I1 fraco, 3 a 8 tarefas, e uma
taxa de sobreajuste NAO trivial — entre 20% e 40% — porque tabela de consulta
generaliza mal por natureza. Aposto contra o mecanismo. Se I1 passar de 20 vou
procurar vazamento antes de comemorar.
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Optional, Sequence, Tuple

from .arc import Grid, carregar, dims, fundo

AVALIACAO = (('C:/Users/usuario/Documents/AGI 2.0/backup_novamind_20260627_170507/data/ARC/data/evaluation').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))

# tarefas que a busca por programa ja resolve — para separar o que e NOVO
JA_RESOLVIDAS = {
    "0c786b71", "0c9aba6e", "18419cfa", "31d5ba1a", "34b99a2b", "506d28a5",
    "59341089", "60c09cac", "66f2d22f", "833dafe3", "903d1b4a", "be03b35f",
    "d19f7514", "e133d23d", "e345f17b",
}

BORDA = -1
NIVEIS = (1, 2, 3, 4, 5, 6)
FRACAO_MAX = 0.25          # guarda contra memorizacao


def _viz(g: Grid, i: int, j: int, d: Sequence[Tuple[int, int]]) -> List[int]:
    h, w = dims(g)
    fora = []
    for di, dj in d:
        a, b = i + di, j + dj
        fora.append(g[a][b] if 0 <= a < h and 0 <= b < w else BORDA)
    return fora


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def contexto(g: Grid, i: int, j: int, nivel: int, f: int) -> tuple:
    """Atributos locais da celula, em complexidade crescente com o nivel."""
    c = g[i][j]
    if nivel == 1:
        return (c,)
    h, w = dims(g)
    if nivel == 2:
        return (c, i == 0 or j == 0 or i == h - 1 or j == w - 1)
    v4 = _viz(g, i, j, D4)
    if nivel == 3:
        return (c, sum(1 for x in v4 if x != f and x != BORDA))
    if nivel == 4:
        return (c, tuple(sorted(v4)))
    if nivel == 5:
        return (c, tuple(v4))
    return (c, tuple(_viz(g, i, j, D8)))


def induzir(pares: Sequence[Tuple[Grid, Grid]],
            nivel: int) -> Optional[Dict[tuple, int]]:
    """
    Ajusta `contexto -> cor de saida` sobre TODOS os pares. Devolve None se
    algum contexto exigir duas cores diferentes — inconsistencia mata a regra,
    nao ha voto majoritario nem nota parcial.
    """
    regra: Dict[tuple, int] = {}
    celulas = 0
    for a, b in pares:
        if dims(a) != dims(b):
            return None
        f = fundo(a)
        h, w = dims(a)
        for i in range(h):
            for j in range(w):
                celulas += 1
                k = contexto(a, i, j, nivel, f)
                v = b[i][j]
                if k in regra:
                    if regra[k] != v:
                        return None
                else:
                    regra[k] = v
    if celulas and len(regra) > FRACAO_MAX * celulas:
        return None                       # memorizou
    return regra


def aplicar_regra(g: Grid, regra: Dict[tuple, int], nivel: int) -> Grid:
    f = fundo(g)
    h, w = dims(g)
    return tuple(tuple(regra.get(contexto(g, i, j, nivel, f), g[i][j])
                       for j in range(w))
                 for i in range(h))


def resolver(treino: Sequence[Tuple[Grid, Grid]]
             ) -> Optional[Tuple[int, Dict[tuple, int]]]:
    """Occam: o PRIMEIRO nivel consistente, do mais simples ao mais complexo."""
    if any(dims(a) != dims(b) for a, b in treino):
        return None
    for nivel in NIVEIS:
        regra = induzir(treino, nivel)
        if regra is None:
            continue
        if all(aplicar_regra(a, regra, nivel) == b for a, b in treino):
            return nivel, regra
    return None


def relatorio() -> None:
    print("=" * 78)
    print("ARC — induzir do delta, em vez de adivinhar o programa")
    print("=" * 78)
    print("PRE-REGISTRO:")
    print("  I0  nivel 1 resolve recoloracao pura, senao a inducao esta quebrada")
    print("  I1  >= 5 tarefas NOVAS -> o mecanismo estende alcance")
    print("  I2  0 novas -> falta estrutura global, nao regra local")
    print("  I3  taxa de sobreajuste medida explicitamente")
    print()
    print("niveis: 1 cor · 2 borda · 3 grau · 4 viz4 multiset · "
          "5 viz4 ordem · 6 viz8")
    print(f"guarda contra memorizacao: regras <= {FRACAO_MAX:.0%} das celulas")

    av = carregar(AVALIACAO)
    aplicaveis = [t for t in av if all(dims(a) == dims(b) for a, b in t[1])]
    print(f"\ntarefas: {len(av)} · aplicaveis (mesma forma em todos os pares): "
          f"{len(aplicaveis)}")

    t0 = time.time()
    ajustou, acertou, novas = 0, 0, []
    sobreajuste = []
    por_nivel: Dict[int, int] = {}
    for nome, tr, te in aplicaveis:
        r = resolver(tr)
        if r is None:
            continue
        nivel, regra = r
        ajustou += 1
        ok = all(dims(a) == dims(b) and aplicar_regra(a, regra, nivel) == b
                 for a, b in te)
        if ok:
            acertou += 1
            por_nivel[nivel] = por_nivel.get(nivel, 0) + 1
            if nome not in JA_RESOLVIDAS:
                novas.append((nome, nivel, len(regra)))
        else:
            sobreajuste.append((nome, nivel, len(regra)))
    dt = time.time() - t0

    print("\n" + "=" * 78)
    print(f"ajustou o treino          {ajustou}")
    print(f"ACERTOU O TESTE           {acertou}/400 = {acertou/400:.1%}")
    print(f"  destas, NOVAS           {len(novas)}  (busca por programa nao acha)")
    print(f"sobreajuste               {len(sobreajuste)}"
          f"  ({len(sobreajuste)/max(1,ajustou):.0%} das que ajustaram)")
    print(f"tempo                     {dt:.0f}s")
    print(f"\nnivel usado nas que acertaram: "
          f"{dict(sorted(por_nivel.items()))}")
    print(f"\nreferencia — busca por programa, 50k orcamento: 14/400")

    if novas:
        print("\nNOVAS (nome · nivel · tamanho da tabela):")
        for n, nv, t in novas[:40]:
            print(f"   {n}   nivel {nv}   {t} regras")

    print("\n" + "=" * 78)
    if len(novas) >= 5:
        print(f"I1 · O MECANISMO ESTENDE ALCANCE. {len(novas)} tarefas que a busca")
        print("     por programa nao alcanca, construidas do delta em vez de")
        print("     adivinhadas. E a primeira coisa hoje que estende em vez de")
        print("     comprimir ou reordenar.")
    elif not novas:
        print("I2 · NENHUMA TAREFA NOVA. Regras locais nao cobrem a familia de")
        print("     mesma forma. O que falta e estrutura GLOBAL — a inversao nao")
        print("     e o caminho, e a hipotese cai.")
    else:
        print(f"Parcial: {len(novas)} novas, abaixo do limiar de 5.")
    if ajustou:
        tx = len(sobreajuste) / ajustou
        print()
        if tx > 0.4:
            print(f"I3 · SOBREAJUSTE ALTO ({tx:.0%}). A ordenacao por Occam nao")
            print("     esta segurando; o numero de I1 nao vale como esta.")
        else:
            print(f"I3 · sobreajuste {tx:.0%} — dentro do que eu previ.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.parse_args()
    relatorio()
