"""
novaphys.arc — sintese de programas com verificador, aplicada ao ARC
=====================================================================

POR QUE TROCAR DE ABORDAGEM, E NAO AJUSTAR A ANTERIOR
-------------------------------------------------------
A auditoria do `run_arc_integrated.py` deu, com controle pareado e 5 sementes:

    copiar a entrada        76,7% pixel   ·  6/2000 exact
    arquitetura sem treino  15,1% pixel   ·  0/2000 exact
    modelo treinado         79,9% pixel   ·  7/2000 exact

O treino faz trabalho real (+64,7pp sobre init aleatorio, controle negativo
limpo) e o ganho sobre copiar sobrevive (+3,15pp pareado, 95% dos lotes).
Mas exact match e 7/2000 contra 6/2000 de copiar: UMA tarefa a mais em 2000.

Um transformer de 128 dims prevendo pixels nao vira solucionador de tarefas
por ajuste de hiperparametro. O alvo do ARC nao e a distribuicao de pixels,
e o PROGRAMA que leva entrada em saida.

O QUE ISTO USA, E POR QUE NAO E ESPECIFICO DO ARC
---------------------------------------------------
O ARC tem a mesma estrutura que o `novaphys` explora em fisica:

    fisica   intervir, medir, e o `custo()` diz se a expressao explica
    ARC      aplicar programa aos pares de treino, e a igualdade exata diz
             se o programa explica

Nos dois casos existe um VERIFICADOR que nao pode ser convencido por
narrativa. Foi a peca decisiva no `intervencao.py` (margem 0 -> ~3000) e e a
mesma peca que o Geoint-R1 obtem do Lean4.

O mecanismo geral sob teste e:

    gramatica de operacoes  ->  busca com orcamento fixo  ->  verificacao
    ->  abstrair o que resolveu em molde  ->  reusar em tarefas novas

A parte generalizavel NAO e a lista de primitivas de grade. E a pergunta
`Reach(G_k)`: acumular abstracoes do que ja resolveu aumenta quantas tarefas
NOVAS sao resolviveis com o MESMO orcamento? Essa pergunta e identica a que
`campo_cumulativo.py`, `campo_recursivo.py` e `gargalo.py` mediram em fisica.

ESTE ARQUIVO E O PISO
-----------------------
Aqui so se mede a base: quantas tarefas de avaliacao a busca resolve com
biblioteca VAZIA. Sem isso nao existe linha de base contra a qual medir o
ganho de biblioteca, e o experimento de acumulacao nao teria sentido.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  B1  a base resolve >= 2% das tarefas de avaliacao (>= 8 de 400)
      -> a DSL alcanca uma fatia real; vale medir acumulacao em cima
  B2  base < 2%
      -> a DSL e fraca demais para servir de substrato; o experimento de
         biblioteca nao teria margem para mostrar nada, e o certo e
         ampliar a gramatica ANTES, nao interpretar o ganho
  B3  comparar sempre com o piso trivial `identidade` (copiar a entrada),
      que na auditoria resolveu 6/2000. Qualquer numero abaixo disso e
      regressao.

MINHA PREVISAO, escrita antes de rodar: espero de 2% a 6%. DSLs pequenas
feitas a mao costumam pegar as tarefas de simetria, recorte e escala, que
sao uma fatia pequena mas nao trivial. Se passar de 10% eu vou procurar
vazamento antes de comemorar.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

Grid = Tuple[Tuple[int, ...], ...]

MAX_LADO = 30


# ══════════════════════════════════════════════ utilidades
def para_grid(g: List[List[int]]) -> Grid:
    return tuple(tuple(int(x) for x in linha) for linha in g)


def dims(g: Grid) -> Tuple[int, int]:
    return len(g), len(g[0]) if g else 0


def valida(g: Optional[Grid]) -> bool:
    if not g or not g[0]:
        return False
    h, w = dims(g)
    if h > MAX_LADO or w > MAX_LADO:
        return False
    return all(len(l) == w for l in g)


def fundo(g: Grid) -> int:
    """A cor mais frequente — convencao usual de fundo no ARC."""
    cont: Dict[int, int] = {}
    for linha in g:
        for c in linha:
            cont[c] = cont.get(c, 0) + 1
    return max(cont, key=lambda k: cont[k])


# ══════════════════════════════════════════════ primitivas
def p_identidade(g: Grid) -> Grid:
    return g


def p_rot90(g: Grid) -> Grid:
    return tuple(zip(*g[::-1]))


def p_rot180(g: Grid) -> Grid:
    return tuple(tuple(l[::-1]) for l in g[::-1])


def p_rot270(g: Grid) -> Grid:
    return tuple(zip(*g))[::-1]


def p_espelha_h(g: Grid) -> Grid:
    return tuple(tuple(l[::-1]) for l in g)


def p_espelha_v(g: Grid) -> Grid:
    return g[::-1]


def p_transpoe(g: Grid) -> Grid:
    return tuple(zip(*g))


def p_recorta(g: Grid) -> Optional[Grid]:
    """Recorta na caixa delimitadora do que nao e fundo."""
    f = fundo(g)
    h, w = dims(g)
    ls = [i for i in range(h) if any(g[i][j] != f for j in range(w))]
    cs = [j for j in range(w) if any(g[i][j] != f for i in range(h))]
    if not ls or not cs:
        return None
    return tuple(tuple(g[i][j] for j in range(cs[0], cs[-1] + 1))
                 for i in range(ls[0], ls[-1] + 1))


def _amplia(g: Grid, k: int) -> Optional[Grid]:
    h, w = dims(g)
    if h * k > MAX_LADO or w * k > MAX_LADO:
        return None
    return tuple(tuple(g[i // k][j // k] for j in range(w * k))
                 for i in range(h * k))


def p_amplia2(g: Grid) -> Optional[Grid]:
    return _amplia(g, 2)


def p_amplia3(g: Grid) -> Optional[Grid]:
    return _amplia(g, 3)


def p_ladrilha_h(g: Grid) -> Optional[Grid]:
    h, w = dims(g)
    if w * 2 > MAX_LADO:
        return None
    return tuple(l + l for l in g)


def p_ladrilha_v(g: Grid) -> Optional[Grid]:
    h, _ = dims(g)
    if h * 2 > MAX_LADO:
        return None
    return g + g


def p_simetriza_h(g: Grid) -> Optional[Grid]:
    h, w = dims(g)
    if w * 2 > MAX_LADO:
        return None
    return tuple(l + l[::-1] for l in g)


def p_simetriza_v(g: Grid) -> Optional[Grid]:
    h, _ = dims(g)
    if h * 2 > MAX_LADO:
        return None
    return g + g[::-1]


def p_fractal(g: Grid) -> Optional[Grid]:
    """
    Cada celula nao-fundo vira uma copia da grade; celulas de fundo viram
    blocos de fundo. E o padrao de auto-similaridade classico do ARC.
    """
    h, w = dims(g)
    if h * h > MAX_LADO or w * w > MAX_LADO:
        return None
    f = fundo(g)
    saida = []
    for i in range(h):
        for bi in range(h):
            linha: List[int] = []
            for j in range(w):
                if g[i][j] != f:
                    linha.extend(g[bi])
                else:
                    linha.extend([f] * w)
            saida.append(tuple(linha))
    return tuple(saida)


def p_bloco_reduz(g: Grid) -> Optional[Grid]:
    """Reduz blocos uniformes k×k a uma celula, se a grade for uniforme em blocos."""
    h, w = dims(g)
    for k in (3, 2):
        if h % k or w % k:
            continue
        ok = True
        for i in range(0, h, k):
            for j in range(0, w, k):
                v = g[i][j]
                if any(g[i + a][j + b] != v for a in range(k) for b in range(k)):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return tuple(tuple(g[i][j] for j in range(0, w, k))
                         for i in range(0, h, k))
    return None


def p_maior_objeto(g: Grid) -> Optional[Grid]:
    """Recorta o maior componente conexo que nao e fundo (4-vizinhanca)."""
    f = fundo(g)
    h, w = dims(g)
    visto = [[False] * w for _ in range(h)]
    melhor: List[Tuple[int, int]] = []
    for i in range(h):
        for j in range(w):
            if g[i][j] == f or visto[i][j]:
                continue
            pilha, comp = [(i, j)], []
            visto[i][j] = True
            while pilha:
                a, b = pilha.pop()
                comp.append((a, b))
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    x, y = a + da, b + db
                    if 0 <= x < h and 0 <= y < w and not visto[x][y] \
                            and g[x][y] != f:
                        visto[x][y] = True
                        pilha.append((x, y))
            if len(comp) > len(melhor):
                melhor = comp
    if not melhor:
        return None
    li = [a for a, _ in melhor]
    lj = [b for _, b in melhor]
    return tuple(tuple(g[i][j] for j in range(min(lj), max(lj) + 1))
                 for i in range(min(li), max(li) + 1))


PRIMITIVAS: List[Tuple[str, Callable[[Grid], Optional[Grid]]]] = [
    ("id", p_identidade),
    ("rot90", p_rot90), ("rot180", p_rot180), ("rot270", p_rot270),
    ("espH", p_espelha_h), ("espV", p_espelha_v), ("transp", p_transpoe),
    ("recorta", p_recorta), ("maiorObj", p_maior_objeto),
    ("amplia2", p_amplia2), ("amplia3", p_amplia3),
    ("ladriH", p_ladrilha_h), ("ladriV", p_ladrilha_v),
    ("simH", p_simetriza_h), ("simV", p_simetriza_v),
    ("fractal", p_fractal), ("blocoReduz", p_bloco_reduz),
]



# ══════════════════════════════════════════════ nivel 2 da gramatica
# B2 disparou: a gramatica de 17 primitivas resolveu 4/400 (1,0%), abaixo do
# piso de 2% que eu havia declarado. O pre-registro manda AMPLIAR antes de
# medir acumulacao de biblioteca — nao interpretar ganho sobre piso ruim.
# As familias abaixo sao as maiores fatias do ARC que uma DSL alcanca e que
# a v1 nao cobria: logica entre paineis, e reparo por simetria.

def _dividir(g: Grid) -> Optional[Tuple[Grid, Grid]]:
    """
    Parte a grade em dois paineis: por uma linha/coluna separadora inteira de
    cor unica, ou exatamente ao meio. Muitas tarefas do ARC sao "combine os
    dois lados".
    """
    h, w = dims(g)
    for j in range(1, w - 1):                      # separador vertical
        col = {g[i][j] for i in range(h)}
        if len(col) == 1 and j == w - 1 - j:
            a = tuple(l[:j] for l in g)
            b = tuple(l[j + 1:] for l in g)
            if dims(a) == dims(b) and a[0]:
                return a, b
    for i in range(1, h - 1):                      # separador horizontal
        if len(set(g[i])) == 1 and i == h - 1 - i:
            a, b = g[:i], g[i + 1:]
            if dims(a) == dims(b) and a:
                return a, b
    if w % 2 == 0:                                 # ao meio, vertical
        a = tuple(l[:w // 2] for l in g)
        b = tuple(l[w // 2:] for l in g)
        return a, b
    if h % 2 == 0:                                 # ao meio, horizontal
        return g[:h // 2], g[h // 2:]
    return None


def _logica(g: Grid, regra: str) -> Optional[Grid]:
    """
    Combina os dois paineis celula a celula e devolve 0/1. A cor final fica
    para `ajustar_mapa` estimar do dado — mesma disciplina das constantes
    ajustadas em `novaphys`, em vez de mais uma primitiva na busca.
    """
    d = _dividir(g)
    if d is None:
        return None
    a, b = d
    f = fundo(g)
    fora = []
    for la, lb in zip(a, b):
        linha = []
        for x, y in zip(la, lb):
            xa, xb = x != f, y != f
            if regra == 'e':
                v = xa and xb
            elif regra == 'ou':
                v = xa or xb
            elif regra == 'xor':
                v = xa != xb
            else:
                v = xa and not xb
            linha.append(1 if v else 0)
        fora.append(tuple(linha))
    return tuple(fora) if fora and fora[0] else None


def p_pE(g: Grid) -> Optional[Grid]:
    return _logica(g, 'e')


def p_pOU(g: Grid) -> Optional[Grid]:
    return _logica(g, 'ou')


def p_pXOR(g: Grid) -> Optional[Grid]:
    return _logica(g, 'xor')


def p_pDIF(g: Grid) -> Optional[Grid]:
    return _logica(g, 'dif')


def p_repara_sim(g: Grid) -> Optional[Grid]:
    """
    Repara buracos usando simetria horizontal, vertical e 180. A cor do
    buraco e a menos frequente que forma um bloco — heuristica usual: a cor
    cujo apagamento torna a grade simetrica.
    """
    h, w = dims(g)
    cores = {c for l in g for c in l}
    for buraco in sorted(cores):
        m = [list(l) for l in g]
        mudou = False
        for _ in range(3):
            for i in range(h):
                for j in range(w):
                    if m[i][j] != buraco:
                        continue
                    for a, b in ((i, w - 1 - j), (h - 1 - i, j),
                                 (h - 1 - i, w - 1 - j)):
                        if m[a][b] != buraco:
                            m[i][j] = m[a][b]
                            mudou = True
                            break
        if mudou and not any(c == buraco for l in m for c in l):
            return tuple(tuple(l) for l in m)
    return None


def p_painel_a(g: Grid) -> Optional[Grid]:
    d = _dividir(g)
    return d[0] if d else None


def p_painel_b(g: Grid) -> Optional[Grid]:
    d = _dividir(g)
    return d[1] if d else None


PRIMITIVAS.extend([
    ("pE", p_pE), ("pOU", p_pOU), ("pXOR", p_pXOR), ("pDIF", p_pDIF),
    ("painelA", p_painel_a), ("painelB", p_painel_b),
    ("reparaSim", p_repara_sim),
])
DICIONARIO = dict(PRIMITIVAS)

# ══════════════════════════════════════════════ mapa de cores ajustado
def ajustar_mapa(pares: Sequence[Tuple[Grid, Grid]]) -> Optional[Dict[int, int]]:
    """
    Ajusta uma recoloracao consistente em TODOS os pares. E um parametro
    estimado do dado, como as constantes ajustadas em `novaphys` — nao uma
    primitiva a mais na busca.
    """
    mapa: Dict[int, int] = {}
    for a, b in pares:
        if dims(a) != dims(b):
            return None
        for la, lb in zip(a, b):
            for x, y in zip(la, lb):
                if x in mapa:
                    if mapa[x] != y:
                        return None
                else:
                    mapa[x] = y
    return mapa


def aplicar_mapa(g: Grid, mapa: Dict[int, int]) -> Grid:
    return tuple(tuple(mapa.get(c, c) for c in l) for l in g)


# ══════════════════════════════════════════════ o programa e a busca
class Programa:
    __slots__ = ('ops', 'mapa')

    def __init__(self, ops: Sequence[str], mapa: Optional[Dict[int, int]] = None):
        self.ops = tuple(ops)
        self.mapa = mapa

    def aplicar(self, g: Grid) -> Optional[Grid]:
        for nome in self.ops:
            fn = DICIONARIO[nome]
            g = fn(g)          # type: ignore[assignment]
            if not valida(g):
                return None
        if self.mapa is not None:
            g = aplicar_mapa(g, self.mapa)
        return g

    def __str__(self) -> str:
        s = " . ".join(self.ops) if self.ops else "id"
        return s + (" +recolore" if self.mapa else "")


DICIONARIO = dict(PRIMITIVAS)


def sequencias(prof: int, extras: Sequence[str] = ()) -> List[Tuple[str, ...]]:
    """Todas as composicoes ate `prof`, incluindo moldes da biblioteca."""
    nomes = [n for n, _ in PRIMITIVAS] + list(extras)
    fora: List[Tuple[str, ...]] = [()]
    for d in range(1, prof + 1):
        fora.extend(itertools.product(nomes, repeat=d))
    return fora


def resolver(treino: Sequence[Tuple[Grid, Grid]], prof: int = 3,
             orcamento: int = 10 ** 9,
             extras: Sequence[str] = ()) -> Optional[Programa]:
    """
    Busca um programa que reproduza EXATAMENTE todos os pares de treino.
    O verificador e igualdade de grade — nao ha nota parcial, nao ha como
    convencer.
    """
    avaliados = 0
    for ops in sequencias(prof, extras):
        avaliados += 1
        if avaliados > orcamento:
            break
        saidas = []
        ok = True
        for a, b in treino:
            r: Optional[Grid] = a
            for nome in ops:
                r = DICIONARIO[nome](r)     # type: ignore[arg-type]
                if not valida(r):
                    ok = False
                    break
            if not ok:
                break
            saidas.append(r)
        if not ok:
            continue
        if all(r == b for r, (_, b) in zip(saidas, treino)):
            return Programa(ops)
        mapa = ajustar_mapa(list(zip(saidas, [b for _, b in treino])))
        if mapa is not None:
            p = Programa(ops, mapa)
            if all(p.aplicar(a) == b for a, b in treino):
                return p
    return None


# ══════════════════════════════════════════════ avaliacao
def carregar(pasta: str) -> List[Tuple[str, List[Tuple[Grid, Grid]],
                                       List[Tuple[Grid, Grid]]]]:
    fora = []
    for nome in sorted(os.listdir(pasta)):
        if not nome.endswith(".json"):
            continue
        with open(os.path.join(pasta, nome)) as f:
            t = json.load(f)
        tr = [(para_grid(p["input"]), para_grid(p["output"])) for p in t["train"]]
        te = [(para_grid(p["input"]), para_grid(p["output"])) for p in t["test"]]
        fora.append((nome[:-5], tr, te))
    return fora


def avaliar(pasta: str, prof: int = 3, verboso: bool = True) -> Dict[str, object]:
    tarefas = carregar(pasta)
    t0 = time.time()
    resolvidas, acertos_teste, identidade = 0, 0, 0
    nomes: List[str] = []
    for nome, tr, te in tarefas:
        # piso trivial declarado em B3
        if all(a == b for a, b in tr) and all(a == b for a, b in te):
            identidade += 1
        p = resolver(tr, prof=prof)
        if p is None:
            continue
        resolvidas += 1
        if all(p.aplicar(a) == b for a, b in te):
            acertos_teste += 1
            nomes.append(f"{nome}  [{p}]")
    dt = time.time() - t0
    n = len(tarefas)
    if verboso:
        print("=" * 74)
        print(f"ARC — sintese de programas com verificador  ·  {pasta}")
        print("=" * 74)
        print(f"tarefas                      {n}")
        print(f"programa acha o treino       {resolvidas}  ({resolvidas/n:.1%})")
        print(f"ACERTA O TESTE (exact match) {acertos_teste}  "
              f"({acertos_teste/n:.1%})")
        print(f"piso trivial (id resolve)    {identidade}  ({identidade/n:.1%})")
        print(f"profundidade {prof} · {len(sequencias(prof))} programas por tarefa"
              f" · {dt:.0f}s")
        if nomes:
            print("\nresolvidas:")
            for x in nomes[:40]:
                print("   " + x)
    return {'n': n, 'treino': resolvidas, 'teste': acertos_teste,
            'identidade': identidade, 'segundos': dt}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", default=('C:/Users/usuario/Documents/AGI 2.0/backup_novamind_20260627_170507/data/ARC/data/evaluation').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))
    ap.add_argument("--prof", type=int, default=3)
    a = ap.parse_args()
    avaliar(a.pasta, prof=a.prof)
