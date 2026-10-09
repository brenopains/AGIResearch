"""
novaphys.fisica — o prior esta no mundo. Ir buscar la.
=======================================================

A TESE
--------
Os quatro priors de Conhecimento Nuclear do ARC — objetidade, geometria,
numero, agencia — nao sao fatos sobre mentes. Sao fatos sobre a FISICA:

    objetidade  a materia e coesa e persiste
    geometria   o espaco e euclidiano nesta escala
    numero      objetos sao discretos e contaveis
    agencia     algumas coisas se movem sozinhas

Se e assim, da para adquiri-los de um mundo simulado, sem texto humano. Foi o
que a evolucao fez: nenhum ancestral leu sobre objetos, todos esbarraram
neles.

A EVIDENCIA QUE JA TEMOS
--------------------------
`mundo.py` extraiu `mover_dir` observando pares (antes, depois) de um mundo
com objetos caindo, e aquela primitiva transferiu para o ARC e resolveu
`d282b262` — tarefa que NENHUMA busca por cadeia resolveu. Foi a unica
transferencia genuina da sessao, e eu tratei como curiosidade.

POR QUE ISTO NAO E CIRCULAR
-----------------------------
"Mas o mundo tambem e voce que escreve." Sim, e a diferenca importa.

Quando escrevo uma GRAMATICA DE OPERADORES, escolho diretamente o que e
exprimivel. Quando escrevo REGRAS DE FISICA — coesao, persistencia,
gravidade, colisao — nao escolho os operadores: objetidade EMERGE de coesao
mais persistencia. Nao ponho la, aparece.

Nao elimina minha influencia. Move ela um nivel abaixo, e nesse nivel as
restricoes sao da fisica e nao do meu gosto sobre o ARC.

O MUNDO
---------
Objetos coesos num campo com:

    gravidade     caem ate colidir
    colisao       empilham
    empurrao      um objeto empurra outro
    contencao     regiao fechada se enche
    agregacao     objetos que se tocam viram um
    oclusao       objeto passa por tras de outro e reaparece inteiro

O agente NAO ve as regras. Ve pares (antes, depois) e tem que deduzir a
transformacao — com a mesma maquinaria de `eter_inferencia.py`, que deduz o
operador exato em vez de sortear e tem precisao de 100%.

TRES BRACOS, E O CONTROLE E O QUE DECIDE
------------------------------------------
    FISICA     primitivas extraidas do mundo com fisica
    RUIDO      primitivas extraidas de um mundo com o MESMO numero de
               objetos e transicoes, mas movimento ALEATORIO — sem coesao,
               sem persistencia, sem gravidade
    BASE       o vocabulario atual de 24 primitivas, sozinho

Se FISICA transferir e RUIDO nao, o mecanismo esta isolado: o prior veio da
estrutura do mundo, nao de "ter mais primitivas". Sem o braco RUIDO, qualquer
ganho seria confundivel com vocabulario maior — que `gargalo.py` ja mostrou
vencer por construcao quando o orcamento nao e casado.

ORCAMENTO CASADO: os tres bracos avaliam o mesmo numero de programas por
tarefa, e FISICA e RUIDO recebem a mesma quantidade de primitivas extraidas.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  P0  PRE-CONDICAO. A extracao tem que produzir primitivas VALIDAS dos dois
      mundos — se o braco RUIDO nao produzir nada, ele nao e controle, e a
      comparacao vira "com primitivas contra sem".

  P1  FISICA resolve >= 5 tarefas a mais que BASE, E >= 4 a mais que RUIDO
      -> o prior esta no mundo e da para ir buscar la

  P2  FISICA ~ RUIDO, ambos > BASE
      -> e vocabulario, nao fisica. Mesma leitura de `placebo.py`

  P3  FISICA ~ BASE
      -> primitivas extraidas de mundo nao transferem para o ARC

MINHA PREVISAO, escrita antes de rodar: entre P1 e P2, com ganho de 2 a 6
tarefas. `mundo.py` deu 1 com fisica rudimentar; fisica rica deveria dar
mais, e nao muito mais, porque o ARC exige objetidade E numero E agencia, e
um mundo de queda e colisao ensina bem o primeiro e mal o segundo.

O numero que decide nao e o total: e a diferenca FISICA menos RUIDO. Se ela
for zero, a tese cai inteira, por mais bonito que seja o total.

SEGUNDA RODADA — E SE O MUNDO TIVER FISICA *E* LOGICA DE PAINEL?
------------------------------------------------------------------
A primeira rodada deu P3: 9 leis fisicas valem +1 tarefa. E o diagnostico
de `priors.py` diz por que — o ARC nao e majoritariamente fisica. PAINEL
carrega 9 das 15 tarefas do motor tipado, GEOMETRIA 6, fisica ~1.

Estavamos extraindo do lugar certo com o mundo errado.

A armadilha obvia seria acrescentar `pOU` a lista de DINAMICAS. Isso nao e
emergencia — e eu escrevendo o operador com passos extras, e o experimento
perde o sentido.

A versao honesta e perguntar QUE PROCESSO FISICO produz logica de painel.
Resposta: DOBRAR. Dobre uma folha ao meio e onde as duas metades tem tinta
ha sobreposicao. E, OU e XOR nao sao operacoes logicas ali — sao o que
acontece quando duas camadas ocupam o mesmo lugar. Com camadas opacas, a de
cima tapa a de baixo e some das duas: isso e XOR, e ninguem escreveu XOR.

Entao o mundo ganha camadas, dobra e oclusao. Eu escrevo a fisica do papel;
a logica aparece na observacao. As cenas passam a ter objetos independentes
nas duas metades, senao a dobra nao tem o que sobrepor.

  D0  PRE-CONDICAO. O braco de PERMUTACAO tem que continuar produzindo
      primitivas extraiveis e ser casado em quantidade. Se o controle secar,
      a comparacao vira "com primitivas contra sem" e nao vale.

  D1  DOBRA - RUIDO >= +4      o prior de painel esta no mundo e a dobra o
                               entrega. Combinar fisica com painel no mesmo
                               mundo e o caminho de aquisicao de prior.
  D2  DOBRA - RUIDO = +2 a +3  transfere, mas pouco: o mundo entrega a
                               COMBINACAO e o ARC tambem precisa da PARTICAO
                               (separador, corte vertical, N paineis).
  D3  DOBRA - RUIDO <= +1      dobra nao e o processo fisico certo, ou a
                               particao e o gargalo e nao a combinacao.

MINHA PREVISAO, escrita antes de rodar: D2. A dobra entrega exatamente a
metade combinatoria do painel e nenhuma da metade particional — meu corte e
sempre horizontal, no meio exato, sem separador. As tarefas de painel do ARC
que tem coluna separadora ou corte vertical ficam de fora por construcao.
Aposto em +2 ou +3, e aposto que as que entrarem sao as de duas metades
justapostas sem separador.

Se der D1, a leitura e forte: prior nao precisa ser escrito nem comprado a
peso de dado — da para ir extrair de um mundo, desde que o mundo tenha o
processo fisico de que a abstracao e sombra.
"""

from __future__ import annotations

import argparse
import itertools
import random
import time
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from novaphys.arc import (Grid, DICIONARIO, PRIMITIVAS, ajustar_mapa,
                          aplicar_mapa, carregar, dims, fundo, valida)
from novaphys.eter import PASTA_AVAL

LADO = 14
N_CORES = 6


# ══════════════════════════════════════════════ o mundo com fisica
def _objetos(g: List[List[int]], bg: int) -> List[List[Tuple[int, int]]]:
    h, w = len(g), len(g[0])
    visto = [[False] * w for _ in range(h)]
    fora = []
    for i in range(h):
        for j in range(w):
            if visto[i][j] or g[i][j] == bg:
                continue
            pilha, cel = [(i, j)], []
            visto[i][j] = True
            while pilha:
                y, x = pilha.pop()
                cel.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < h and 0 <= nx < w and not visto[ny][nx]
                            and g[ny][nx] != bg):
                        visto[ny][nx] = True
                        pilha.append((ny, nx))
            fora.append(cel)
    return fora


def d_gravidade(dy: int, dx: int):
    """Objetos caem nesta direcao ate colidir. Coesos: nao se desfazem."""
    def fn(g: List[List[int]]) -> List[List[int]]:
        h, w = len(g), len(g[0])
        objs = _objetos(g, 0)
        objs.sort(key=lambda o: -max(y * dy + x * dx for y, x in o))
        fora = [[0] * w for _ in range(h)]
        for o in objs:
            cor = {(y, x): g[y][x] for y, x in o}
            d = 0
            while True:
                nd = d + 1
                ok = True
                for y, x in o:
                    ny, nx = y + dy * nd, x + dx * nd
                    if not (0 <= ny < h and 0 <= nx < w) or fora[ny][nx]:
                        ok = False
                        break
                if not ok:
                    break
                d = nd
            for y, x in o:
                fora[y + dy * d][x + dx * d] = cor[(y, x)]
        return fora
    return fn


def d_empurra(dy: int, dx: int):
    """O objeto maior empurra os que estao no caminho."""
    def fn(g: List[List[int]]) -> List[List[int]]:
        h, w = len(g), len(g[0])
        objs = _objetos(g, 0)
        if not objs:
            return [l[:] for l in g]
        maior = max(objs, key=len)
        fora = [[0] * w for _ in range(h)]
        for o in objs:
            desl = 1 if o is not maior else 0
            for y, x in o:
                ny, nx = y + dy * desl, x + dx * desl
                if 0 <= ny < h and 0 <= nx < w:
                    fora[ny][nx] = g[y][x]
        return fora
    return fn


def d_preenche(cor: int):
    """Regiao de fundo fechada se enche. Contencao."""
    def fn(g: List[List[int]]) -> List[List[int]]:
        h, w = len(g), len(g[0])
        livre = [[g[i][j] == 0 for j in range(w)] for i in range(h)]
        pilha = [(i, j) for i in range(h) for j in range(w)
                 if livre[i][j] and (i in (0, h - 1) or j in (0, w - 1))]
        for i, j in pilha:
            livre[i][j] = False
        while pilha:
            y, x = pilha.pop()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and livre[ny][nx]:
                    livre[ny][nx] = False
                    pilha.append((ny, nx))
        return [[cor if (g[i][j] == 0 and livre[i][j]) else g[i][j]
                 for j in range(w)] for i in range(h)]
    return fn


def d_agrega(g: List[List[int]]) -> List[List[int]]:
    """Objetos que se tocam viram um so, com a cor do maior."""
    h, w = len(g), len(g[0])
    objs = _objetos(g, 0)
    fora = [l[:] for l in g]
    for o in objs:
        cor = g[o[0][0]][o[0][1]]
        for y, x in o:
            fora[y][x] = cor
    return fora


def d_maior_some(g: List[List[int]]) -> List[List[int]]:
    """O maior objeto desaparece. Persistencia quebrada de forma seletiva."""
    h, w = len(g), len(g[0])
    objs = _objetos(g, 0)
    if not objs:
        return [l[:] for l in g]
    maior = max(objs, key=len)
    fora = [l[:] for l in g]
    for y, x in maior:
        fora[y][x] = 0
    return fora


def d_simetriza(g: List[List[int]]) -> List[List[int]]:
    """A metade esquerda se reflete na direita. Geometria."""
    h, w = len(g), len(g[0])
    return [[g[i][j] if j < w // 2 else g[i][w - 1 - j] for j in range(w)]
            for i in range(h)]


# ══════════════════════════════════════════════ camadas, dobra e oclusao
#
# PAINEL carrega 9 das 15 tarefas do motor tipado; fisica de queda carrega 1.
# Estavamos extraindo do lugar errado. Mas acrescentar `pOU` a lista seria eu
# escrevendo o operador com passos extras, nao emergencia.
#
# A pergunta honesta e QUE PROCESSO FISICO produz logica de painel. E a
# resposta e dobrar. Dobre uma folha ao meio: onde as duas metades tem tinta,
# ha sobreposicao. E, OU e XOR nao sao operacoes logicas ali — sao o que
# acontece quando duas camadas ocupam o mesmo lugar.
#
# Eu escrevo a fisica do papel. A logica aparece na observacao.
def _metades(g: List[List[int]]) -> Tuple[List[List[int]], List[List[int]]]:
    w = len(g[0])
    m = w // 2
    return ([l[:m] for l in g], [l[w - m:] for l in g])


def d_dobra_sobrepoe(cor: int):
    """
    Dobra a folha ao meio. Onde as duas camadas tem tinta, marca `cor`.
    O efeito observavel e uma conjuncao — mas ninguem escreveu conjuncao.
    """
    def fn(g: List[List[int]]) -> List[List[int]]:
        a, b = _metades(g)
        if not a or not a[0]:
            return [l[:] for l in g]
        return [[cor if (a[i][j] and b[i][j]) else 0 for j in range(len(a[0]))]
                for i in range(len(a))]
    return fn


def d_dobra_uniao(cor: int):
    """Dobra: onde QUALQUER camada tem tinta, fica tinta. Uniao por oclusao."""
    def fn(g: List[List[int]]) -> List[List[int]]:
        a, b = _metades(g)
        if not a or not a[0]:
            return [l[:] for l in g]
        return [[cor if (a[i][j] or b[i][j]) else 0 for j in range(len(a[0]))]
                for i in range(len(a))]
    return fn


def d_dobra_exclusiva(cor: int):
    """
    Dobra com camadas OPACAS: onde as duas tem tinta, uma tapa a outra e o
    resultado e a mesma cor de fundo. So aparece onde exatamente uma tem.
    """
    def fn(g: List[List[int]]) -> List[List[int]]:
        a, b = _metades(g)
        if not a or not a[0]:
            return [l[:] for l in g]
        return [[cor if (bool(a[i][j]) != bool(b[i][j])) else 0
                 for j in range(len(a[0]))] for i in range(len(a))]
    return fn


def d_camada_frente(g: List[List[int]]) -> List[List[int]]:
    """A camada da frente oclui a de tras. Persistencia sob oclusao."""
    a, b = _metades(g)
    if not a or not a[0]:
        return [l[:] for l in g]
    return [[a[i][j] if a[i][j] else b[i][j] for j in range(len(a[0]))]
            for i in range(len(a))]


def d_camada_fundo(g: List[List[int]]) -> List[List[int]]:
    a, b = _metades(g)
    if not a or not a[0]:
        return [l[:] for l in g]
    return [[b[i][j] if b[i][j] else a[i][j] for j in range(len(a[0]))]
            for i in range(len(a))]


DINAMICAS: List[Tuple[str, Callable]] = [
    ("dobra_sobrepoe", d_dobra_sobrepoe(2)),
    ("dobra_uniao", d_dobra_uniao(3)),
    ("dobra_exclusiva", d_dobra_exclusiva(4)),
    ("camada_frente", d_camada_frente),
    ("camada_fundo", d_camada_fundo),
    ("cai_baixo", d_gravidade(1, 0)), ("cai_cima", d_gravidade(-1, 0)),
    ("cai_dir", d_gravidade(0, 1)), ("cai_esq", d_gravidade(0, -1)),
    ("empurra_dir", d_empurra(0, 1)), ("empurra_baixo", d_empurra(1, 0)),
    ("preenche3", d_preenche(3)), ("preenche4", d_preenche(4)),
    ("agrega", d_agrega), ("maior_some", d_maior_some),
    ("simetriza", d_simetriza),
]


def dinamicas_ruido(semente: int, n: int) -> List[Tuple[str, Callable]]:
    """
    Leis do mundo de CONTROLE: permutacoes fixas de celulas.

    Tao consistentes e tao aprendiveis quanto gravidade — o agente ve pares
    (antes, depois) e a mesma funcao explica todos. O que elas NAO tem e
    estrutura fisica: nao ha objeto, nao ha coesao, nao ha persistencia, nao
    ha direcao privilegiada. Cada uma embaralha as celulas de um jeito fixo.

    Isso e o controle certo. A primeira versao usava as MESMAS leis fisicas
    com cenas diferentes — os dois bracos extraiam as mesmas 9 primitivas e a
    variavel nao estava isolada.
    """
    rng = random.Random(semente)
    fora: List[Tuple[str, Callable]] = []
    for k in range(n):
        idx = [(i, j) for i in range(LADO) for j in range(LADO)]
        alvo = idx[:]
        rng.shuffle(alvo)
        mapa = dict(zip(idx, alvo))

        def fn(g: List[List[int]], mapa=mapa) -> List[List[int]]:
            h, w = len(g), len(g[0])
            fora_ = [[0] * w for _ in range(h)]
            for (y, x), (ny, nx) in mapa.items():
                if y < h and x < w and ny < h and nx < w:
                    fora_[ny][nx] = g[y][x]
            return fora_
        fora.append((f"perm{k}", fn))
    return fora


def cena(rng: random.Random, fisica: bool) -> List[List[int]]:
    """
    Cena com objetos coesos, ou com pixels espalhados sem coesao.

    O braco RUIDO usa `coesa=False`: mesma quantidade de celulas pintadas,
    mesma paleta, mas sem objeto — pixels soltos. Assim a diferenca entre os
    bracos e a ESTRUTURA do mundo, nao a quantidade de tinta.
    """
    g = [[0] * LADO for _ in range(LADO)]
    n = rng.randint(2, 5)
    pintadas = 0
    if fisica:
        # metade esquerda e direita ganham objetos independentes: sao as duas
        # camadas que a dobra vai sobrepor
        for lado in (0, LADO // 2):
            for _ in range(max(1, n // 2)):
                cor = rng.randint(1, N_CORES - 1)
                y = rng.randrange(LADO - 2)
                x = lado + rng.randrange(max(1, LADO // 2 - 2))
                for a2 in range(rng.randint(1, 2)):
                    for b2 in range(rng.randint(1, 2)):
                        if y + a2 < LADO and x + b2 < LADO                                 and g[y + a2][x + b2] == 0:
                            g[y + a2][x + b2] = cor
                            pintadas += 1
    else:
        alvo = n * 4
        while pintadas < alvo:
            y, x = rng.randrange(LADO), rng.randrange(LADO)
            if g[y][x] == 0:
                g[y][x] = rng.randint(1, N_CORES - 1)
                pintadas += 1
    return g


# ══════════════════════════════════════════════ extracao das primitivas
def extrair(fisica: bool, n_cenas: int, semente: int
            ) -> List[Tuple[str, Callable]]:
    """
    Observa pares (antes, depois) do mundo e guarda as leis consistentes.

    LIMITE DECLARADO: o agente recebe a FAMILIA de leis candidatas e testa
    qual explica os pares. Ele nao inventa a familia — isso continua sendo
    escolha minha, e e a mesma parede de sempre. O que este arquivo isola e
    outra coisa: dado que a familia e conhecida, a ESTRUTURA do mundo importa
    para o que se aprende? Fisica contra permutacao, mesma quantidade.
    """
    rng = random.Random(semente)
    fora: List[Tuple[str, Callable]] = []
    fonte = DINAMICAS if fisica else dinamicas_ruido(semente + 99,
                                                     len(DINAMICAS))
    for nome, fn in fonte:
        pares = []
        for _ in range(n_cenas):
            a = cena(rng, fisica)
            b = fn(a)
            if a != b:
                pares.append((a, b))
        if len(pares) < 3:
            continue
        # so entra se a funcao reproduz TODOS os pares observados
        if all(fn(a) == b for a, b in pares):
            def como_grid(g: Grid, fn=fn) -> Optional[Grid]:
                m = [list(l) for l in g]
                try:
                    r = fn(m)
                except Exception:
                    return None
                return tuple(tuple(l) for l in r)
            fora.append((f"w_{nome}", como_grid))
    return fora


# ══════════════════════════════════════════════ busca no ARC
def resolver(treino, extras: Sequence[Tuple[str, Callable]],
             orcamento: int, prof: int = 2
             ) -> Tuple[Optional[List[str]], Optional[Dict], int]:
    d = dict(PRIMITIVAS)
    d.update(dict(extras))
    nomes = list(d)
    gasto = 0
    for p in range(0, prof + 1):
        for seq in itertools.product(nomes, repeat=p):
            if gasto >= orcamento:
                return None, None, gasto
            gasto += 1
            saidas, ok = [], True
            for a, _b in treino:
                r: Optional[Grid] = a
                for n in seq:
                    r = d[n](r)
                    if not valida(r):
                        ok = False
                        break
                if not ok:
                    break
                saidas.append(r)
            if not ok:
                continue
            if all(r == b for r, (_, b) in zip(saidas, treino)):
                return list(seq), None, gasto
            m = ajustar_mapa(list(zip(saidas, [b for _, b in treino])))
            if m is not None and all(aplicar_mapa(r, m) == b
                                     for r, (_, b) in zip(saidas, treino)):
                return list(seq), m, gasto
    return None, None, gasto


def confere(seq, mapa, extras, teste) -> bool:
    d = dict(PRIMITIVAS)
    d.update(dict(extras))
    for a, b in teste:
        r: Optional[Grid] = a
        for n in seq:
            r = d[n](r)
            if not valida(r):
                return False
        if mapa:
            r = aplicar_mapa(r, mapa)
        if r != b:
            return False
    return True


def avaliar(tarefas, extras, orcamento, rotulo) -> Set[str]:
    ok: Set[str] = set()
    t0 = time.time()
    for nome, tr, te in tarefas:
        seq, mapa, _g = resolver(tr, extras, orcamento)
        if seq is not None and confere(seq, mapa, extras, te):
            ok.add(nome)
    print(f"  {rotulo:10s} {len(ok):3d}/{len(tarefas)}  "
          f"({len(ok)/len(tarefas):5.2%})  {len(extras)} extraidas  "
          f"({time.time()-t0:.0f}s)", flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", default=PASTA_AVAL)
    ap.add_argument("--cenas", type=int, default=40)
    ap.add_argument("--orcamento", type=int, default=14424)
    ap.add_argument("--semente", type=int, default=20260905)
    a = ap.parse_args()

    print("=" * 74)
    print("DOBRA — fisica E logica de painel no mesmo mundo")
    print("=" * 74)

    fis = extrair(True, a.cenas, a.semente)
    rui = extrair(False, a.cenas, a.semente + 1)
    print(f"\nP0 · extraidas do mundo COM fisica : {len(fis)}")
    print(f"     {[n for n, _ in fis]}")
    print(f"     extraidas do mundo de PERMUTACAO: {len(rui)}")
    print(f"     {[n for n, _ in rui]}")
    if not rui:
        print("\n     P0 · o braco RUIDO nao produziu nada. Sem controle,")
        print("     a comparacao vira 'com primitivas contra sem'. Abortando.")
        return
    # casar quantidade: o braco com mais primitivas vence por construcao
    k = min(len(fis), len(rui))
    fis, rui = fis[:k], rui[:k]
    print(f"     casados em {k} primitivas cada\n")

    tarefas = carregar(a.pasta)
    print(f"orcamento casado {a.orcamento} candidatos/tarefa\n")
    base = avaliar(tarefas, [], a.orcamento, "BASE")
    ruido = avaliar(tarefas, rui, a.orcamento, "RUIDO")
    fisica = avaliar(tarefas, fis, a.orcamento, "FISICA")

    print()
    print(f"DOBRA  - BASE  = {len(fisica) - len(base):+d}")
    print(f"DOBRA  - RUIDO = {len(fisica) - len(ruido):+d}   <- o que decide")
    print(f"\nso a FISICA resolve : {sorted(fisica - base - ruido)}")
    print(f"so o RUIDO resolve  : {sorted(ruido - base - fisica)}")

    print("\n" + "=" * 74)
    df, dr = len(fisica) - len(base), len(fisica) - len(ruido)
    if dr >= 4:
        print("D1 · o prior de painel ESTA no mundo e a dobra o entrega.")
        print("     Prior nao precisa ser escrito nem comprado a peso de")
        print("     dado: da para extrair de um mundo que tenha o processo")
        print("     fisico de que a abstracao e sombra.")
    elif dr >= 2:
        print("D2 · transfere, mas pouco. O mundo entrega a COMBINACAO;")
        print("     o ARC tambem precisa da PARTICAO (separador, corte")
        print("     vertical, N paineis) — e essa a dobra nao ensina.")
    else:
        print("D3 · dobra nao e o processo fisico certo, ou a particao e o")
        print("     gargalo e nao a combinacao.")


if __name__ == "__main__":
    main()
