"""
novaphys.estrutura_arc — o conteudo da saida como soma sobre os objetos da entrada
==================================================================================

DE ONDE VEM
-------------
`relacoes_arc` (etapa 1): relacoes GLOBAIS entrada->saida acertam as dimensoes da
saida de teste em 79% das 213 tarefas de formas novas, mas a contagem completa de
cores so em 4,7%. O diagnostico: celulas OCUPADAS certas em 23%, n de OBJETOS em
30%; na tarefa tipica erra so a cor DESENHADA e o fundo. O que falta e o
CONTEUDO — e conteudo depende de estrutura: quais objetos, e quanto cada um vira.

O MECANISMO (ETAPA 2)
-----------------------
Uma grandeza da saida Y (ocupadas, n de objetos, contagem de cada cor) passa a poder
ser uma SOMA SOBRE OS OBJETOS DA ENTRADA:
    Y = SOMA_{o : filtro(o)} ( a * f(o) + b )
    f em {1, tamanho, altura, largura, area da caixa, lado maior}
    filtro: todos os objetos, ou so os da cor c
    a em {1, 2, 3, 4, 1/2}, b em {-2..4}
Objetos = componentes 4-conexos de mesma cor fora do fundo. As familias sao minhas.
Aceita se vale EXATO em todos os pares de treino e poupa bits (como na etapa 1); a
relacao estrutural concorre com as globais da etapa 1 pela mesma regua.

PRE-REGISTRO, ANTES DE RODAR — mesmas 213 tarefas formas_novas, nulo 20261015
-------------------------------------------------------------------------------
  S1  celulas OCUPADAS certas no teste: >= 40% (etapa 1: 23%), nulo <= 5%
  S2  contagem completa (dimensoes + 10 cores) certa: >= 12% (etapa 1: 4,7%),
      nulo <= 2%
PREVISAO: as duas FALHAM por pouco — S1 ~35%, S2 ~9%. Soma sobre objetos cobre
"cada objeto vira um quadro/uma linha", mas boa parte do desenho do ARC depende de
RELACAO entre objetos (ligar dois pontos, preencher entre eles), que soma de
termos independentes nao pega.

RESULTADO:
  formas_novas n=213   ocupadas certas 23,9% (etapa 1: 23,0%)   contagem completa 4,2%
                       (etapa 1: 4,7%)   nulo 0% / 0%
S1 FALHOU. S2 FALHOU. As duas falhas previstas, e piores que a previsao (35% / 9%).
A soma por objeto nao acrescentou NADA — e a contagem completa perdeu uma tarefa:
com mais relacoes candidatas, uma estrutural mais barata ajustou o treino, errou o
teste e tirou o lugar de uma global certa (mais hipoteses, mais ajuste espurio).
O conteudo das tarefas de criacao nao e soma de contribuicoes independentes por
objeto: depende de RELACOES ENTRE objetos. E o mesmo muro que o laco na grade achou
na lente: a gramatica de testes e de profundidade 1 (sem negar teste derivado, sem
combinar deslocamentos). Proximo: gramatica propria no substrato de grade.
CORRECAO: esse "muro" nao existia — a lei da lente e exprimivel em profundidade 1
por contagem (`viz8 = 1`), e o nucleo a acha com 16 cenas (20/20 novas). Ligar
profundidade 2 (negar testes derivados) PIOROU (14/20). A licao que fica e a
mesma desta etapa: mais hipoteses, mais ajuste espurio.
"""

from __future__ import annotations

import argparse
import itertools
import random
from fractions import Fraction
from math import log2

from novaphys.arc import carregar, dims, fundo
from novaphys.eter import PASTA_AVAL
from novaphys.inventario import assinaturas
from novaphys import invariante_arc as IA
from novaphys import relacoes_arc as R

SEMENTE_NULO = 20261015
AS = (Fraction(1), Fraction(2), Fraction(3), Fraction(4), Fraction(1, 2))
BS = range(-2, 5)
FEATS = ("um", "tam", "alt", "larg", "caixa", "maior")
OCUP = IA.NOMES.index("ocup")
NOBJ = IA.NOMES.index("nobj")


def objetos(g):
    """(cor, tamanho, altura, largura) de cada componente 4-conexo fora do fundo."""
    h, w = dims(g)
    bg = fundo(g)
    visto = [[False] * w for _ in range(h)]
    out = []
    for i in range(h):
        for j in range(w):
            if visto[i][j] or g[i][j] == bg:
                continue
            c = g[i][j]
            pilha, cel = [(i, j)], []
            visto[i][j] = True
            while pilha:
                y, x = pilha.pop()
                cel.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and not visto[yy][xx] \
                            and g[yy][xx] == c:
                        visto[yy][xx] = True
                        pilha.append((yy, xx))
            ys = [p[0] for p in cel]
            xs = [p[1] for p in cel]
            out.append((c, len(cel), max(ys) - min(ys) + 1, max(xs) - min(xs) + 1))
    return out


def _f(o, nome):
    c, t, a, l = o
    return {"um": 1, "tam": t, "alt": a, "larg": l, "caixa": a * l, "maior": max(a, l)}[nome]


def _soma(objs, filtro, feat, a, b):
    return sum(a * _f(o, feat) + b for o in objs if filtro is None or o[0] == filtro)


def estruturais(tr, j):
    """Relacoes por objeto para a grandeza j da saida, validas em todo o treino."""
    Os = [objetos(a) for a, _b in tr]
    alvo = [IA.grandezas(b)[j] for _a, b in tr]
    filtros = [None] + sorted({o[0] for objs in Os for o in objs})
    custo = log2(len(filtros)) + log2(len(FEATS)) + log2(len(AS)) + log2(len(BS)) + 3
    out = []
    for filtro, feat, a, b in itertools.product(filtros, FEATS, AS, BS):
        vs = [_soma(o, filtro, feat, a, b) for o in Os]
        if all(v == y for v, y in zip(vs, alvo)):
            out.append((custo, ("obj", filtro, feat, a, b)))
    return out


def preve(tr, te):
    """Etapa 1 (globais) + etapa 2 (por objeto), escolhidas pela mesma regua."""
    Fe = [[Fraction(x) for x in IA.grandezas(a)] for a, _b in tr]
    Fs = [[Fraction(x) for x in IA.grandezas(b)] for _a, b in tr]
    ganho_sem = len(tr) * R.BITS_VAL
    certas = {}
    for j in range(R.NQ):
        cs = R._candidatas(Fe, Fs, j)
        if j in (OCUP, NOBJ) or j < 10:
            cs = cs + estruturais(tr, j)
        if not cs:
            certas[j] = False
            continue
        custo, rel = min(cs, key=lambda c: c[0])
        if ganho_sem - custo <= 0:
            certas[j] = False
            continue
        ok = True
        for a, b in te:
            y = IA.grandezas(b)[j]
            if rel[0] == "obj":
                _t, filtro, feat, aa, bb = rel
                v = _soma(objetos(a), filtro, feat, aa, bb)
            else:
                v = R._aplica(rel, [Fraction(x) for x in IA.grandezas(a)])
            ok &= v == y
        certas[j] = ok
    return certas


def mede(tarefas):
    n = ocup = comp = 0
    for _nome, tr, te in tarefas:
        c = preve(tr, te)
        n += 1
        ocup += c[OCUP]
        comp += all(c[j] for j in R.DIM) and all(c[j] for j in range(10))
    return n, ocup, comp


def main() -> None:
    argparse.ArgumentParser().parse_args()
    todas = carregar(PASTA_AVAL)
    tarefas = [t for t in todas if "formas_novas" in assinaturas(t[1])]
    rng = random.Random(SEMENTE_NULO)
    saidas = [b for _n, tr, te in todas for _a, b in tr + te]
    nul = [(nome, [(a, rng.choice(saidas)) for a, _b in tr],
            [(a, rng.choice(saidas)) for a, _b in te]) for nome, tr, te in tarefas]
    print("=" * 76)
    print("ESTRUTURA ARC — o conteudo como soma sobre os objetos da entrada")
    print("=" * 76)
    n, o, c = mede(tarefas)
    nn, on, cn = mede(nul)
    print(f"  formas_novas n={n}   ocupadas certas {o} ({100*o/n:.1f}%)   "
          f"contagem completa {c} ({100*c/n:.1f}%)")
    print(f"  nulo               ocupadas {100*on/nn:.1f}%   contagem completa {100*cn/nn:.1f}%")
    print("")
    print("=" * 76)
    print(f"S1 {'.' if o / n >= 0.40 and on / nn <= 0.05 else 'FALHOU .'} ocupadas "
          f"{100*o/n:.1f}% (>= 40%; etapa 1: 23%), nulo {100*on/nn:.1f}%")
    print(f"S2 {'.' if c / n >= 0.12 and cn / nn <= 0.02 else 'FALHOU .'} contagem completa "
          f"{100*c/n:.1f}% (>= 12%; etapa 1: 4,7%), nulo {100*cn/nn:.1f}%")


if __name__ == "__main__":
    main()
