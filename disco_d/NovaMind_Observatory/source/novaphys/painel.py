"""
novaphys.painel — a particao ou a regra de combinacao?
=======================================================

DE ONDE VEM
-------------
`fisica.py` mostrou que 9 leis fisicas extraidas de um mundo valem +1 tarefa
no ARC, contra +0 do controle de permutacao. E `priors.py` explicou por que e
so +1: o ARC nao e majoritariamente fisica. PAINEL carrega 9 das 15 tarefas
do motor tipado. Estavamos extraindo do lugar certo com o mundo errado.

A pergunta do usuario foi: e se o mundo tiver fisica E logica de painel?

A armadilha e obvia. Acrescentar `pOU` a lista de dinamicas nao e emergencia
— e escrever o operador com passos extras. Entao a pergunta honesta e QUE
PROCESSO FISICO produz logica de painel, e a resposta e DOBRAR. Dobre uma
folha ao meio: onde as duas camadas tem tinta, ha sobreposicao. Com camadas
opacas, a de cima tapa a de baixo e o par some — isso e XOR, e ninguem
escreveu XOR. A logica e a sombra da fisica de camadas.

Mas a operacao de painel do ARC tem DUAS metades, e a dobra so entrega uma:

    PARTICAO     achar a costura — a parede, o separador, onde cortar
    COMBINACAO   o que acontece quando as duas camadas se sobrepoem

Este arquivo separa as duas por fatorial, porque so assim da para saber qual
delas o mundo esta ensinando — e para nao repetir o erro de `prior.py`, onde
o controle nao tinha a resposta e a comparacao virou "tem contra nao tem".

O DESENHO
-----------
Mundo: uma PAREDE (coluna ou linha uniforme, posicao sorteada por cena) com
objetos independentes dos dois lados. A dinamica dobra a folha na costura.

    A  PAREDE+CAMADA     acha a parede, dobra, combina por opacidade
    B  PAREDE+ARBITRARIO acha a parede, dobra, combina por TABELA SORTEADA
                         sobre pares de cor (100 entradas, aleatoria)
    C  MEIO+CAMADA       corta no meio fixo, dobra, combina por opacidade
    BASE                 nada

    A - B  isola a REGRA DE COMBINACAO (ambos acham a parede)
    A - C  isola a PARTICAO            (ambos combinam por opacidade)

O controle B e a peca que faz este experimento valer: tem a mesma maquinaria
de achar parede, a mesma dobra, o mesmo numero de primitivas, e e igualmente
consistente e aprendivel. So a regra de combinacao e arbitraria em vez de
fisica. Uma tabela aleatoria sobre 100 pares de cor contem as funcoes logicas
com probabilidade desprezivel.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  N0  PRE-CONDICAO. Os tres bracos extraem o mesmo numero de primitivas e
      rodam no mesmo orcamento. Se B secar, nao ha controle.

  N1  A - B >= +3  e  A - C >= +2
      as duas metades importam e o mundo entrega as duas. Leitura forte:
      prior nao precisa ser escrito nem comprado a peso de dado.

  N2  A - B >= +3  e  A - C <= +1
      o que vale e a COMBINACAO; achar a parede nao acrescenta. O ARC usa
      metades justapostas mais do que separador.

  N3  A - B <= +1  e  A - C >= +2
      o que vale e a PARTICAO. Qualquer combinacao serve desde que o corte
      esteja certo — e ai o prior de painel e geometrico, nao logico.

  N4  A - B <= +1  e  A - C <= +1
      nem uma nem outra transfere. Dobra nao e o processo certo.

MINHA PREVISAO, escrita antes de rodar: N2, e com folga menor do que gostaria
— algo como A-B = +3 a +5, A-C = 0 a +1. Razao: as tarefas de painel que o
motor tipado resolve usam `pE`, `pOU`, `pXOR` sobre metades justapostas, e o
corte no meio ja cobre isso. A parede deve render pouco porque exige que a
coluna separadora seja uniforme E que o corte a descarte, e nem toda tarefa
de separador do ARC tem esse formato.

Se der N3 eu estava lendo o problema ao contrario a sessao inteira: seria a
GEOMETRIA DO CORTE, e nao a operacao logica, o que carrega o PAINEL.
"""

from __future__ import annotations

import argparse
import random
from typing import Callable, List, Optional, Tuple

from .fisica import N_CORES, PASTA_AVAL, avaliar, carregar

Grid = Tuple[Tuple[int, ...], ...]
Mat = List[List[int]]


# ══════════════════════════════════════════════ o mundo: parede e camadas
def cena_parede(rng: random.Random, com_parede: bool) -> Mat:
    """
    Duas regioes com objetos independentes.

    ERRO DE GEOMETRIA na primeira versao, pego pelo portao N0: eu exigia que
    a parede deixasse os dois lados iguais, mas sorteava a posicao dela numa
    grade de largura PAR. Condicao insatisfazivel — `acha_costura` devolvia
    None em toda cena e os dois bracos com parede secaram (0 primitivas).

    Correcao: com parede, a grade tem lado IMPAR e a parede fica no centro,
    entao os dois lados tem o mesmo tamanho e da para sobrepor. Para achar a
    costura continuar sendo trabalho de verdade, o TAMANHO da grade varia por
    cena — o indice do centro muda e nao da para decorar.
    """
    if com_parede:
        k = rng.randint(5, 12)
        w = 2 * k + 1
        h = rng.randint(6, 18)
    else:
        k = rng.randint(5, 12)
        w, h = 2 * k, rng.randint(6, 18)
    vert = rng.random() < 0.5
    if not vert:
        w, h = h, w
    g = [[0] * w for _ in range(h)]
    cor_parede = rng.randint(1, N_CORES - 1)

    def regiao(y0: int, y1: int, x0: int, x1: int) -> None:
        if y1 - y0 < 1 or x1 - x0 < 1:
            return
        for _ in range(rng.randint(2, 5)):
            cor = rng.randint(1, N_CORES - 1)
            if com_parede and cor == cor_parede:
                cor = 1 + (cor % (N_CORES - 1))
            a = rng.randint(1, min(3, y1 - y0))
            b = rng.randint(1, min(2, x1 - x0))
            y = rng.randrange(y0, y1 - a + 1)
            x = rng.randrange(x0, x1 - b + 1)
            for u in range(a):
                for v in range(b):
                    g[y + u][x + v] = cor

    if vert:
        corte = k
        regiao(0, h, 0, corte)
        regiao(0, h, corte + (1 if com_parede else 0), w)
        if com_parede:
            for i in range(h):
                g[i][corte] = cor_parede
    else:
        corte = k
        regiao(0, corte, 0, w)
        regiao(corte + (1 if com_parede else 0), h, 0, w)
        if com_parede:
            g[corte] = [cor_parede] * w
    return g


def acha_costura(g: Mat) -> Optional[Tuple[bool, int]]:
    """
    Acha a parede: linha ou coluna inteira de uma cor so, que nao seja fundo,
    e que deixe os dois lados do mesmo tamanho. Nada e passado de fora — se
    a costura nao estiver visivel na grade, a dobra nao acontece.
    """
    h, w = len(g), len(g[0])
    for j in range(1, w - 1):
        if j != w - 1 - j:
            continue
        col = {g[i][j] for i in range(h)}
        if len(col) == 1 and 0 not in col:
            return (True, j)
    for i in range(1, h - 1):
        if i != h - 1 - i:
            continue
        if len(set(g[i])) == 1 and g[i][0] != 0:
            return (False, i)
    return None


def _partes(g: Mat, com_parede: bool):
    """Devolve as duas camadas a sobrepor, ou None se nao der para dobrar."""
    h, w = len(g), len(g[0])
    if com_parede:
        c = acha_costura(g)
        if c is None:
            return None
        vert, k = c
        if vert:
            a = [linha[:k] for linha in g]
            b = [linha[k + 1:] for linha in g]
        else:
            a = [linha[:] for linha in g[:k]]
            b = [linha[:] for linha in g[k + 1:]]
    else:
        # CONFUNDIMENTO CORRIGIDO. A primeira versao exigia largura PAR no
        # braco do meio, entao ele recusava 195ba7dc (13) e 5d2a5c43 (9) por
        # geometria, nao por falta de particao — e `A - C` media "achar
        # parede MAIS aceitar largura impar" em vez de so a particao.
        #
        # A dobra correta e a de `fisica.py`: l[:m] e l[w-m:] com m = w//2.
        # Em largura impar isso descarta exatamente a coluna do vinco. Nao e
        # remocao de separador escrita a mao; e o que dobrar uma folha impar
        # faz. Agora C dobra igualzinho ao A e a UNICA diferenca entre os
        # dois e se a costura foi PROCURADA ou assumida no centro.
        if w >= 4:
            m = w // 2
            a = [linha[:m] for linha in g]
            b = [linha[w - m:] for linha in g]
        elif h >= 4:
            m = h // 2
            a = [linha[:] for linha in g[:m]]
            b = [linha[:] for linha in g[h - m:]]
        else:
            return None
    if not a or not b or not a[0] or not b[0]:
        return None
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        return None
    return a, b


# ── combinacao por FISICA DE CAMADA: opacidade, oclusao, sobreposicao
def _camada(modo: str, com_parede: bool):
    def fn(g: Mat) -> Mat:
        p = _partes(g, com_parede)
        if p is None:
            return [linha[:] for linha in g]
        a, b = p
        fora = []
        for i in range(len(a)):
            linha = []
            for j in range(len(a[0])):
                x, y = a[i][j], b[i][j]
                if modo == "sobrepoe":       # as duas camadas tem tinta
                    linha.append(1 if (x and y) else 0)
                elif modo == "uniao":        # qualquer camada tem tinta
                    linha.append(1 if (x or y) else 0)
                elif modo == "opaca":        # opacas: uma tapa a outra
                    linha.append(1 if (bool(x) != bool(y)) else 0)
                elif modo == "frente":       # a de cima oclui a de baixo
                    linha.append(x if x else y)
                else:                        # a de baixo aparece no vazio
                    linha.append(y if y else x)
            fora.append(linha)
        return fora
    return fn


# ── combinacao ARBITRARIA: tabela sorteada sobre pares de cor (o controle)
def _tabela(rng: random.Random, com_parede: bool):
    tab = {(x, y): rng.randrange(N_CORES)
           for x in range(N_CORES) for y in range(N_CORES)}

    def fn(g: Mat) -> Mat:
        p = _partes(g, com_parede)
        if p is None:
            return [linha[:] for linha in g]
        a, b = p
        return [[tab[(a[i][j], b[i][j])] for j in range(len(a[0]))]
                for i in range(len(a))]
    return fn


MODOS = ["sobrepoe", "uniao", "opaca", "frente", "fundo"]


def dinamicas(braco: str, semente: int) -> List[Tuple[str, Callable]]:
    rng = random.Random(semente)
    if braco == "A":
        return [(f"p_{m}", _camada(m, True)) for m in MODOS]
    if braco == "C":
        return [(f"m_{m}", _camada(m, False)) for m in MODOS]
    return [(f"t_{k}", _tabela(rng, True)) for k in range(len(MODOS))]


# ══════════════════════════════════════════════ extracao
def extrair(braco: str, n_cenas: int, semente: int
            ) -> List[Tuple[str, Callable]]:
    """
    Mesma disciplina de `fisica.py`: so entra a lei que reproduz TODOS os
    pares observados. A familia continua sendo escolha minha — o que muda
    entre os bracos e a ESTRUTURA do mundo e a regra que ele obedece.
    """
    rng = random.Random(semente)
    com_parede = braco in ("A", "B")
    fora: List[Tuple[str, Callable]] = []
    for nome, fn in dinamicas(braco, semente + 99):
        pares = []
        for _ in range(n_cenas):
            a = cena_parede(rng, com_parede)
            b = fn(a)
            if a != b:
                pares.append((a, b))
        if len(pares) < 3 or not all(fn(a) == b for a, b in pares):
            continue

        def como_grid(g: Grid, fn=fn) -> Optional[Grid]:
            try:
                r = fn([list(linha) for linha in g])
            except Exception:
                return None
            if not r or not r[0]:
                return None
            return tuple(tuple(linha) for linha in r)
        fora.append((nome, como_grid))
    return fora


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", default=PASTA_AVAL)
    ap.add_argument("--cenas", type=int, default=40)
    ap.add_argument("--orcamento", type=int, default=14424)
    ap.add_argument("--semente", type=int, default=20260907)
    a = ap.parse_args()

    print("=" * 74)
    print("PAINEL — a particao ou a regra de combinacao?")
    print("=" * 74)

    ext = {b: extrair(b, a.cenas, a.semente + i)
           for i, b in enumerate(("A", "B", "C"))}
    for b, rot in (("A", "PAREDE+CAMADA"), ("B", "PAREDE+ARBITRARIO"),
                   ("C", "MEIO+CAMADA")):
        print(f"N0 · {rot:18s} {len(ext[b])} extraidas  "
              f"{[n for n, _ in ext[b]]}")
    k = min(len(v) for v in ext.values())
    if k == 0:
        print("\n     N0 · um braco secou. Sem controle a comparacao nao vale.")
        return
    ext = {b: v[:k] for b, v in ext.items()}
    print(f"     casados em {k} primitivas cada\n")

    tarefas = carregar(a.pasta)
    print(f"orcamento casado {a.orcamento} candidatos/tarefa, prof 2\n")
    base = avaliar(tarefas, [], a.orcamento, "BASE")
    rb = avaliar(tarefas, ext["B"], a.orcamento, "B parede+arb")
    rc = avaliar(tarefas, ext["C"], a.orcamento, "C meio+camada")
    ra = avaliar(tarefas, ext["A"], a.orcamento, "A parede+cam")

    ab, ac = len(ra) - len(rb), len(ra) - len(rc)
    print(f"\nA - BASE = {len(ra) - len(base):+d}")
    print(f"A - B    = {ab:+d}   <- valor da REGRA DE COMBINACAO")
    print(f"A - C    = {ac:+d}   <- valor da PARTICAO (achar a parede)")
    print(f"\nso A resolve : {sorted(ra - base - rb - rc)}")
    print(f"so B resolve : {sorted(rb - base - ra - rc)}")
    print(f"so C resolve : {sorted(rc - base - ra - rb)}")
    print(f"C - BASE = {len(rc) - len(base):+d}  (camada sem achar parede)")

    print("\n" + "=" * 74)
    if ab >= 3 and ac >= 2:
        print("N1 · as duas metades importam e o mundo entrega as duas.")
        print("     Prior de painel extraido de fisica de camada, com")
        print("     controle arbitrario casado. Leitura forte.")
    elif ab >= 3:
        print("N2 · o que vale e a COMBINACAO. Achar a parede nao acrescenta:")
        print("     o ARC usa metades justapostas mais do que separador.")
    elif ac >= 2:
        print("N3 · o que vale e a PARTICAO. Qualquer combinacao serve desde")
        print("     que o corte esteja certo — o prior de painel e")
        print("     GEOMETRICO, nao logico. Eu estava lendo ao contrario.")
    else:
        print("N4 · nem particao nem combinacao transferem.")


if __name__ == "__main__":
    main()
