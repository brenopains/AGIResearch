"""
novaphys.gap_generativo — o que exatamente APARECE no fundo?
==============================================================

DE ONDE VEIO ESTA PERGUNTA
----------------------------
`porque_falha.py` mediu a divisao das 80 primeiras tarefas de avaliacao:

    D. erra no FUNDO (o que devia aparecer do nada)   26   <- maior grupo
    C. dimensao muda e a pilha erra o conteudo        16
    E. erra DENTRO de objeto                          14
    RESOLVIDA                                         12
    A. nao montou pilha nenhuma                        8

    residuo no FUNDO: 34/64 das falhas (53%)
    pilha mais comum na falha: `copia` sozinha, 11 vezes

`copia` sozinha significa: o sistema descreve "a saida repete a entrada", acerta
o fundo todo, e NAO TEM COMO DIZER O QUE APARECE. Para ali.

E isso explica por que as 85 propriedades inventadas nunca entraram em pilha
vencedora: `cor_por_prop` recolore objeto que existe, `apaga_prop` apaga objeto
que existe. **Todo o vocabulario de propriedade opera sobre objeto existente** e
e estruturalmente incapaz de tocar o grupo dominante.

A LICAO DE METODO, APLICADA AGORA
-----------------------------------
Da ultima vez eu pulei da medida ("40 de 60 nao descrevem") para uma causa por
ANALOGIA com a fisica ("falta a grandeza"), construi 85 propriedades, e o dado
rejeitou. Entao aqui nao se propoe nada: mede-se O QUE aparece, e so depois se
escreve o codec da classe majoritaria.

O QUE SE MEDE — assinaturas do conjunto APARECEU
--------------------------------------------------
Para cada par (a, b) de mesma dimensao, com bg = fundo(a):

    APARECEU = {(i,j) : a[i][j] == bg  e  b[i][j] != bg}

e testa-se qual assinatura cobre esse conjunto:

  raio          cada celula que apareceu esta na mesma LINHA ou COLUNA de uma
                celula de objeto, e com a cor dela — projecao, feixe, rastro
  preenche      APARECEU sao exatamente regioes de fundo CERCADAS por objeto
  halo          APARECEU sao as celulas de fundo VIZINHAS de objeto (contorno
                para fora)
  simetria      b e a completada por reflexao/rotacao
  copia_desl    algum componente de APARECEU tem a MESMA FORMA de um objeto de a
                (copia transladada)
  linha_liga    APARECEU liga pares de celulas de mesma cor (segmentos retos
                entre eles)
  molde         APARECEU repete um bloco de a em outra posicao (ladrilho)

Uma tarefa pode ter mais de uma assinatura; o que interessa e a distribuicao,
para saber qual classe de codec compra mais tarefas por linha de codigo escrita.

SEM PORTAO, DE PROPOSITO: isto e descricao, nao hipotese. O portao vem depois,
no codec.
"""

from __future__ import annotations

import argparse
from collections import Counter

from novaphys.arc import carregar, dims
from novaphys.eter import PASTA_AVAL
from novaphys import descricao as D
from novaphys.descricao import objetos, fundo, cercadas


def _apareceu(a, b, bg):
    h, w = dims(b)
    if dims(a) != (h, w):
        return None
    return {(i, j) for i in range(h) for j in range(w)
            if a[i][j] == bg and b[i][j] != bg}


def _cels_objeto(a, bg):
    h, w = dims(a)
    return {(i, j) for i in range(h) for j in range(w) if a[i][j] != bg}


def assinaturas(a, b):
    """Quais assinaturas cobrem >= 80% do que apareceu."""
    bg = fundo(a)
    ap = _apareceu(a, b, bg)
    if not ap:
        return set()
    h, w = dims(a)
    obj = _cels_objeto(a, bg)
    fora = set()
    n = len(ap)

    # raio: na mesma linha ou coluna de um objeto, com a COR dele
    linhas = {}
    colunas = {}
    for i, j in obj:
        linhas.setdefault(i, set()).add(a[i][j])
        colunas.setdefault(j, set()).add(a[i][j])
    ok = sum(1 for i, j in ap
             if b[i][j] in linhas.get(i, set()) or b[i][j] in colunas.get(j, set()))
    if ok >= 0.8 * n:
        fora.add("raio")

    # preenche: regiao de fundo cercada
    try:
        cerc = set(cercadas(a, bg))
    except Exception:
        cerc = set()
    if cerc and len(ap & cerc) >= 0.8 * n:
        fora.add("preenche")

    # halo: celula de fundo vizinha de objeto
    halo = set()
    for i, j in obj:
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)):
            p = (i + di, j + dj)
            if 0 <= p[0] < h and 0 <= p[1] < w and a[p[0]][p[1]] == bg:
                halo.add(p)
    if halo and len(ap & halo) >= 0.8 * n:
        fora.add("halo")

    # simetria: b e a refletida/rodada, unida
    for nome, f in (("h", lambda g: [l[::-1] for l in g]),
                    ("v", lambda g: g[::-1]),
                    ("t", lambda g: [list(r) for r in zip(*g)])):
        try:
            s = f(a)
            if dims(s) != (h, w):
                continue
            if all(b[i][j] == (a[i][j] if a[i][j] != bg else s[i][j])
                   for i, j in ap):
                fora.add("simetria")
                break
        except Exception:
            pass

    # copia_desl: componente de APARECEU com a forma de um objeto de a
    formas = set()
    for c, _k in objetos(a, bg):
        ys = [y for y, _x in c]
        xs = [x for _y, x in c]
        formas.add(frozenset((y - min(ys), x - min(xs)) for y, x in c))
    vis, comps = set(), []
    for p in ap:
        if p in vis:
            continue
        pil, comp = [p], []
        vis.add(p)
        while pil:
            q = pil.pop()
            comp.append(q)
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                r = (q[0] + di, q[1] + dj)
                if r in ap and r not in vis:
                    vis.add(r)
                    pil.append(r)
        comps.append(comp)
    if comps and any(frozenset((y - min(y2 for y2, _ in c),
                                x - min(x2 for _, x2 in c)) for y, x in c) in formas
                     for c in comps):
        fora.add("copia_desl")

    # linha_liga: segmento reto entre duas celulas de mesma cor
    liga = set()
    porcor = {}
    for i, j in obj:
        porcor.setdefault(a[i][j], []).append((i, j))
    for cor, ps in porcor.items():
        if len(ps) > 40:
            continue
        for x in range(len(ps)):
            for y in range(x + 1, len(ps)):
                (i1, j1), (i2, j2) = ps[x], ps[y]
                if i1 == i2:
                    for j in range(min(j1, j2) + 1, max(j1, j2)):
                        if a[i1][j] == bg:
                            liga.add((i1, j))
                elif j1 == j2:
                    for i in range(min(i1, i2) + 1, max(i1, i2)):
                        if a[i][j1] == bg:
                            liga.add((i, j1))
    if liga and len(ap & liga) >= 0.8 * n:
        fora.add("linha_liga")

    # molde: b repete um bloco de a
    for ph in (range(1, h + 1) if h * w <= 400 else []):
        for pw in range(1, w + 1):
            if ph == h and pw == w:
                continue
            if all(b[i][j] == b[i % ph][j % pw] for i in range(h)
                   for j in range(w)):
                fora.add("molde")
                break
        if "molde" in fora:
            break
    return fora


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--tarefas", type=int, default=80)
    ap_.add_argument("--teto", type=int, default=400)
    a_ = ap_.parse_args()
    tarefas = carregar(PASTA_AVAL)[: a_.tarefas]
    print("=" * 78)
    print(f"GAP GENERATIVO — o que APARECE no fundo, em {len(tarefas)} tarefas")
    print("=" * 78)

    conta = Counter()
    sem = []
    com_ap = 0
    por_tarefa = {}
    for k, (tarefa, tr, te) in enumerate(tarefas):
        try:
            fm, pilha, err = D.montar(tr, teto=a_.teto)
            resolvida = (pilha and err == 0
                         and all(D.aplicar((fm, pilha), x) == y for x, y in te))
        except Exception:
            resolvida = False
        if resolvida:
            continue
        # assinatura que vale em TODOS os pares de treino
        comum = None
        tem = False
        for a, b in tr:
            bg = fundo(a)
            x = _apareceu(a, b, bg)
            if x is None or not x:
                continue
            tem = True
            s = assinaturas(a, b)
            comum = s if comum is None else (comum & s)
        if not tem:
            continue
        com_ap += 1
        comum = comum or set()
        por_tarefa[tarefa] = comum
        if comum:
            for s in comum:
                conta[s] += 1
        else:
            sem.append(tarefa)
        if (k + 1) % 25 == 0:
            print(f"    {k+1} tarefas...", flush=True)

    print("")
    print("=" * 78)
    print(f"  tarefas NAO resolvidas em que algo APARECE no fundo: {com_ap}")
    print(f"\n  assinatura valida em TODOS os pares de treino:")
    for s, n in conta.most_common():
        print(f"    {s:14s} {n:3d}/{com_ap}  ({100*n/max(com_ap,1):.0f}%)")
    print(f"\n  sem nenhuma assinatura conhecida: {len(sem)}/{com_ap}")
    print(f"    {sem[:12]}")
    print(f"\n  tarefas com assinatura UNICA (as mais faceis de comprar):")
    uni = Counter()
    for t, s in por_tarefa.items():
        if len(s) == 1:
            uni[next(iter(s))] += 1
    for s, n in uni.most_common():
        print(f"    {s:14s} {n:3d}")


if __name__ == "__main__":
    main()
