"""
novaphys.censofino — que ESTRUTURA tem o que ainda falta desenhar?
====================================================================

POR QUE ESTE ARQUIVO EXISTE
-----------------------------
O censo grosso (`inventario.py`) disse: 199 nao resolvidas produzem formas que
nao existiam na entrada. Isso apontou para EMISSAO (+2) e para o COMPOSITOR
(+3), e as duas entregaram.

Depois eu tentei adivinhar o que vinha a seguir — "composicao com estrutura
interna" — construi `arranjo.py`, e deu ZERO.

E esse e o padrao mais forte da sessao, agora em vinte tentativas:

    construcao que veio de MEDIDA        entregou (+2, +3, +3)
    construcao que veio de PALPITE meu   zero, dezessete vezes

Entao, em vez de adivinhar de novo, mede-se de novo — e mais fino, so sobre o
subconjunto que interessa: as que produzem formas novas E que continuam nao
resolvidas.

O QUE SE MEDE
---------------
So nos pares de TREINO, nunca no teste. Assinatura so vale se valer em TODOS os
pares da tarefa.

Sobre tarefas de MESMA DIMENSAO (onde a diferenca celula a celula existe):

    preserva        nenhuma celula nao-fundo da entrada muda -> construcao
                    puramente ADITIVA, so acrescenta
    novas_uma_cor   todas as celulas novas tem a mesma cor
    novas_colam     toda celula nova e 4-adjacente a algo que ja existia ->
                    CRESCIMENTO a partir do existente
    novas_linha     as celulas novas ocupam linhas ou colunas inteiras ->
                    familia de LINHA/RAIO
    novas_enchem    as celulas novas sao exatamente o fundo CERCADO ->
                    PREENCHIMENTO
    novas_replicam  a forma das celulas novas coincide com a forma de algum
                    objeto que ja existia -> CARIMBO / copia
    simetriza       a saida e simetrica e a entrada nao -> COMPLETAR SIMETRIA
    apaga           alguma celula nao-fundo da entrada vira fundo

Sobre as que MUDAM de dimensao, as assinaturas de celula nao se aplicam; conta-
se separado.

PRE-REGISTRO
--------------
  F0  PRE-CONDICAO. Menos de 30% do subconjunto pode ficar sem assinatura
      nenhuma, senao o vocabulario e cego de novo.

  F1  uma assinatura concentra >= 50 tarefas -> alvo nomeado e grande
  F2  a maior fica entre 20 e 49
  F3  nenhuma passa de 20 -> cauda longa dentro da propria familia de
      criacao, e nao ha proxima construcao obvia

MINHA PREVISAO: F2, e aposto em `preserva` e `novas_colam` como as maiores —
construcao aditiva a partir do que ja esta la. Aposto que `simetriza` e
pequena, apesar de ser a familia mais citada na literatura do ARC.

E registro: se der F3, a leitura e que a familia de criacao tambem e cauda
longa, e a estrategia de "achar a proxima representacao" tem retorno
decrescente ate dentro do maior balde que o censo grosso encontrou.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from typing import Dict, List, Set, Tuple

from novaphys.arc import Grid, carregar, dims
from novaphys.eter import PASTA_AVAL
from novaphys.inventario import RESOLVIDAS, _fundo, _forma, _objs

EXTRA = {"070dd51e", "aa18de87", "3194b014", "ccd554ac", "cd3c21df"}
FEITAS = RESOLVIDAS | EXTRA


def _cercado(g: Grid, bg: int) -> Set[Tuple[int, int]]:
    h, w = dims(g)
    vis = [[False] * w for _ in range(h)]
    fila = deque()
    for i in range(h):
        for j in (0, w - 1):
            if g[i][j] == bg and not vis[i][j]:
                vis[i][j] = True
                fila.append((i, j))
    for j in range(w):
        for i in (0, h - 1):
            if g[i][j] == bg and not vis[i][j]:
                vis[i][j] = True
                fila.append((i, j))
    while fila:
        y, x = fila.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if (0 <= ny < h and 0 <= nx < w and not vis[ny][nx]
                    and g[ny][nx] == bg):
                vis[ny][nx] = True
                fila.append((ny, nx))
    return {(i, j) for i in range(h) for j in range(w)
            if g[i][j] == bg and not vis[i][j]}


def _sim(g: Grid) -> bool:
    return (g == tuple(tuple(l[::-1]) for l in g)) or (g == g[::-1])


def assinaturas_par(a: Grid, b: Grid) -> Set[str]:
    bg = _fundo(a)
    h, w = dims(a)
    novas = {(i, j) for i in range(h) for j in range(w)
             if a[i][j] == bg and b[i][j] != bg}
    apagadas = {(i, j) for i in range(h) for j in range(w)
                if a[i][j] != bg and b[i][j] == bg}
    mudou_cor = any(a[i][j] != bg and b[i][j] != bg and a[i][j] != b[i][j]
                    for i in range(h) for j in range(w))
    s = set()
    if not apagadas and not mudou_cor:
        s.add("preserva")
    if apagadas:
        s.add("apaga")
    if not novas:
        return s
    cn = {b[i][j] for i, j in novas}
    if len(cn) == 1:
        s.add("novas_uma_cor")
    if all(any(0 <= i + dy < h and 0 <= j + dx < w and a[i + dy][j + dx] != bg
               for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)))
           for i, j in novas):
        s.add("novas_colam")
    linhas = {i for i, _j in novas}
    cols = {j for _i, j in novas}
    if all(sum(1 for j in range(w) if (i, j) in novas) == w for i in linhas) \
            or all(sum(1 for i in range(h) if (i, j) in novas) == h
                   for j in cols):
        s.add("novas_linha")
    if novas == _cercado(a, bg):
        s.add("novas_enchem")
    formas = {_forma(c) for c in _objs(a, bg)}
    if _forma(frozenset(novas)) in formas:
        s.add("novas_replicam")
    if _sim(b) and not _sim(a):
        s.add("simetriza")
    return s


def main() -> None:
    argparse.ArgumentParser().parse_args()
    print("=" * 74)
    print("CENSO FINO — que estrutura tem o que ainda falta desenhar?")
    print("=" * 74)
    tarefas = carregar(PASTA_AVAL)

    alvo, muda_dim = [], []
    for n, tr, _te in tarefas:
        if n in FEITAS:
            continue
        try:
            bg = _fundo(tr[0][0])
            tem_novas = all(
                bool({_forma(c) for c in _objs(b, bg)} -
                     {_forma(c) for c in _objs(a, bg)}) for a, b in tr)
        except Exception:
            continue
        if not tem_novas:
            continue
        if any(dims(a) != dims(b) for a, b in tr):
            muda_dim.append(n)
        else:
            alvo.append((n, tr))

    print(f"\nsubconjunto: formas novas E nao resolvidas")
    print(f"  mesma dimensao : {len(alvo)}")
    print(f"  muda dimensao  : {len(muda_dim)}  (assinaturas de celula nao se")
    print(f"                    aplicam; ficam de fora do censo abaixo)\n")

    por: Dict[str, Set[str]] = {}
    for n, tr in alvo:
        comuns = None
        for a, b in tr:
            try:
                s = assinaturas_par(a, b)
            except Exception:
                s = set()
            comuns = s if comuns is None else (comuns & s)
        por[n] = comuns or set()

    c = Counter(x for s in por.values() for x in s)
    sem = sum(1 for s in por.values() if not s)
    print(f"{'assinatura':16s} {'tarefas':>8s} {'%':>7s}")
    print("-" * 34)
    for k, v in c.most_common():
        print(f"{k:16s} {v:>8d} {v/max(len(alvo),1):>7.1%}")
    print("-" * 34)
    print(f"{'NENHUMA':16s} {sem:>8d} {sem/max(len(alvo),1):>7.1%}")

    # combinacoes mais frequentes: e o perfil da tarefa, nao a marca isolada
    perfis = Counter(tuple(sorted(s)) for s in por.values() if s)
    print(f"\nperfis mais comuns (combinacao de assinaturas):")
    for p, v in perfis.most_common(8):
        print(f"  {v:3d}  {'+'.join(p)}")

    maior = c.most_common(1)[0] if c else ("", 0)
    print("\n" + "=" * 74)
    if sem > len(alvo) * 0.30:
        print("F0 FALHOU: mais de 30% sem assinatura. Vocabulario cego.")
        return
    if maior[1] >= 50:
        print(f"F1 · `{maior[0]}` concentra {maior[1]}. Alvo nomeado e grande.")
    elif maior[1] >= 20:
        print(f"F2 · a maior e `{maior[0]}` com {maior[1]}.")
    else:
        print("F3 · cauda longa DENTRO da familia de criacao. Nao ha proxima")
        print("     construcao obvia, e procurar a proxima representacao tem")
        print("     retorno decrescente ate aqui.")


if __name__ == "__main__":
    main()
