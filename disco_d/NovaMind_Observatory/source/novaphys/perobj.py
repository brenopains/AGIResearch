"""
novaphys.perobj — o quantificador UNIVERSAL sobre objetos, com mapeamento
==========================================================================

O ERRO DE DESENHO QUE ESTE ARQUIVO CONSERTA
---------------------------------------------
`ordem1.py` deu O3: leituras quantificadas renderam +1 tarefa (3 -> 4). Eu
tinha previsto 12 a 20. A previsao quebrou, e a autopsia achou um erro de
desenho especifico, nao azar.

Os quantificadores que eu implementei sao todos SUPERLATIVOS — `o maior`, `o
menor`, `o unico`, `o de cor rara`. Sao `argmax`. E `cor_obj(maior)` transmite
UMA cor para a grade inteira: e uma constante.

Mas a forma dominante no ARC nao e essa. E:

    para CADA objeto, pinte-o conforme o seu tamanho
    para CADA objeto, o que tem buraco vira X e o sem buraco vira Y
    o objeto que aparece N vezes vira uma cor, o que aparece M vezes vira outra

Isso e QUANTIFICACAO UNIVERSAL com uma funcao por objeto — nao `argmax`. Eu
medi o quantificador errado.

O MECANISMO, E ELE CONTINUA SENDO DEDUCAO
-------------------------------------------
Uma leitura por-objeto precisa de um mapa propriedade -> cor, e esse mapa nao
pode ser escrito por mim: depende da tarefa. Entao ele e AJUSTADO aos pares de
treino, resolvendo a restricao — cada objeto com propriedade p tem que virar a
cor que ele vira nos exemplos. Nao ha amostragem em lugar nenhum.

E o mesmo principio que ja existe no motor (`ajustar_mapa` deduz o mapa de
cores da saida) e o mesmo que rendeu 3.000x em `eter_inferencia`: quando a
resposta esta determinada pelos dados, resolva; nao sorteie.

Propriedades por objeto, todas computaveis sem eu saber a tarefa:

    tam         numero de celulas
    altura      largura     forma da caixa
    n_buracos   componentes de fundo cercadas — classico do ARC
    rank_tam    posicao no ranking por tamanho
    n_mesmo_tam quantos objetos tem esse mesmo tamanho
    cor         a propria cor (vira recoloracao por objeto)
    e_maior     se e o maior (booleano, nao superlativo selecionado)

RISCO DECLARADO, e ele e serio: uma leitura ajustada tem MUITO mais liberdade
que uma leitura fixa. Ela pode decorar. A protecao e o verificador, que exige
igualdade EXATA nas grades de TESTE — nunca vistas no ajuste. Decoreba nao
passa nele. E por isso o numero que vale e o de teste, nunca o de treino.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  P0  PRE-CONDICAO. O braco sem as leituras por-objeto tem que reproduzir
      `ordem1.py`: 4/400 num passo. Se nao reproduzir, divergiu.

  P1  +6 ou mais sobre `ordem1`   o quantificador universal e o que faltava,
                                  e o superlativo era o quantificador errado
  P2  +2 a +5                     ajuda na ordem dos outros mecanismos
  P3  +0 ou +1                    objeto continua nao sendo o que falta, e ai
                                  sao DUAS formas de quantificacao sobre
                                  objeto testadas e refutadas

MINHA PREVISAO: P2, e desta vez modesta de proposito, porque acabei de errar
uma aposta alta no mesmo terreno. Espero +2 a +5 num passo. A razao para nao
apostar alto de novo: um passo continua sendo um teto duro — a mesma linguagem
por celula da 3 num passo e 8 em dois — entao mesmo o vocabulario certo tem
pouco espaco para aparecer nesse regime. O numero que vale sera o do braco de
DOIS passos.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from typing import Callable, Dict, List, Optional, Set, Tuple

from novaphys.arc import Grid, carregar, dims, fundo, valida
from novaphys.eter import PASTA_AVAL
from novaphys.eter_inferencia import todas_leituras, todos_testes
from novaphys.ordem1 import (aplicar, inferir, leituras_quantificadas,
                             objetos, testes_quantificados)


def _buracos(cel: Set[Tuple[int, int]], g: Grid) -> int:
    """Componentes de fundo cercadas pelo objeto: classico do ARC."""
    if not cel:
        return 0
    ys = [y for y, _ in cel]
    xs = [x for _, x in cel]
    y0, y1, x0, x1 = min(ys), max(ys), min(xs), max(xs)
    h, w = y1 - y0 + 3, x1 - x0 + 3
    livre = [[True] * w for _ in range(h)]
    for (y, x) in cel:
        livre[y - y0 + 1][x - x0 + 1] = False
    vis = [[False] * w for _ in range(h)]
    fila = deque([(0, 0)])
    vis[0][0] = True
    while fila:
        y, x = fila.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not vis[ny][nx] and livre[ny][nx]:
                vis[ny][nx] = True
                fila.append((ny, nx))
    n = 0
    for i in range(h):
        for j in range(w):
            if livre[i][j] and not vis[i][j]:
                n += 1
                fila = deque([(i, j)])
                vis[i][j] = True
                while fila:
                    y, x = fila.popleft()
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < h and 0 <= nx < w and not vis[ny][nx]
                                and livre[ny][nx]):
                            vis[ny][nx] = True
                            fila.append((ny, nx))
    return n


def _props(objs, g: Grid) -> List[Dict[str, int]]:
    tams = sorted({len(c) for c, _k in objs}, reverse=True)
    cont_tam = Counter(len(c) for c, _k in objs)
    cont_cor = Counter(k for _c, k in objs)
    fora = []
    for cel, cor in objs:
        ys = [y for y, _ in cel]
        xs = [x for _, x in cel]
        fora.append({
            "tam": len(cel),
            "altura": max(ys) - min(ys) + 1,
            "largura": max(xs) - min(xs) + 1,
            "n_buracos": _buracos(cel, g),
            "rank_tam": tams.index(len(cel)) + 1,
            "n_mesmo_tam": cont_tam[len(cel)],
            "n_mesma_cor": cont_cor[cor],
            "cor": cor,
            "e_maior": int(len(cel) == tams[0]),
        })
    return fora


PROPS = ("tam", "altura", "largura", "n_buracos", "rank_tam",
         "n_mesmo_tam", "n_mesma_cor", "cor", "e_maior")


class LeituraObj:
    """
    Valor de cada celula = mapa[propriedade do objeto a que ela pertence].
    O mapa e DEDUZIDO dos pares de treino, nunca sorteado.
    """

    __slots__ = ("prop", "diag", "mapa", "const")

    def __init__(self, prop: str, diag: bool, mapa: Dict[int, int]):
        self.prop, self.diag, self.mapa = prop, diag, mapa
        self.const = False

    def valores(self, g: Grid, bg: int) -> List[List[int]]:
        h, w = dims(g)
        fora = [[bg] * w for _ in range(h)]
        objs = objetos(g, bg, self.diag)
        for (cel, _cor), p in zip(objs, _props(objs, g)):
            v = self.mapa.get(p[self.prop], -1)
            for (i, j) in cel:
                fora[i][j] = v
        return fora

    def __str__(self) -> str:
        return f"porObj({self.prop}){'8' if self.diag else '4'}"


def ajustar(treino, prop: str, diag: bool) -> Optional[LeituraObj]:
    """
    Resolve o mapa propriedade -> cor a partir dos exemplos. Voto de maioria
    quando um valor de propriedade aparece com saidas diferentes; a cobertura
    depois mede o quanto isso explica, e a cobertura parcial cuida do resto.
    """
    votos: Dict[int, Counter] = {}
    for a, b in treino:
        if dims(a) != dims(b):
            return None
        bg = fundo(a)
        objs = objetos(a, bg, diag)
        if not objs or len(objs) > 60:
            continue
        for (cel, _cor), p in zip(objs, _props(objs, a)):
            c = Counter(b[i][j] for (i, j) in cel)
            votos.setdefault(p[prop], Counter()).update(c)
    if not votos:
        return None
    return LeituraObj(prop, diag,
                      {k: v.most_common(1)[0][0] for k, v in votos.items()})


def leituras_por_objeto(treino) -> List[LeituraObj]:
    fora = []
    for prop in PROPS:
        for diag in (False, True):
            L = ajustar(treino, prop, diag)
            if L is not None:
                fora.append(L)
    return fora


# ══════════════════════════════════════════════ avaliacao
def vocabulario(treino, quant: bool, porobj: bool):
    cores = sorted({c for a, b in treino for l in (a + b) for c in l})
    L = list(todas_leituras(cores))
    T = list(todos_testes(cores))
    if quant:
        L += leituras_quantificadas()
        T += testes_quantificados()
    if porobj:
        L += leituras_por_objeto(treino)
    return L, T


def uma(args):
    nome, tr, te, quant, porobj = args
    try:
        L, T = vocabulario(tr, quant, porobj)
        op, g = inferir(tr, L, T)
    except Exception:
        return nome, False, None
    if op is None:
        return nome, False, None
    try:
        ok = all(aplicar(op, x) == y for x, y in te)
    except Exception:
        return nome, False, None
    return nome, ok, f"[{op[0]} ? {op[1]} : {op[2]}]"


def avaliar(tarefas, quant, porobj, rotulo, nucleos):
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(nucleos) as pool:
        res = pool.map(uma, [(n, tr, te, quant, porobj)
                             for n, tr, te in tarefas], chunksize=2)
    ok = {n for n, o, _t in res if o}
    ex = {n: t for n, o, t in res if o}
    n = len(tarefas)
    print(f"  {rotulo:24s} {len(ok):3d}/{n}  ({len(ok)/n:5.2%})  "
          f"({time.time()-t0:.0f}s)", flush=True)
    return ok, ex


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", default=PASTA_AVAL)
    ap.add_argument("--nucleos", type=int, default=10)
    a = ap.parse_args()

    print("=" * 78)
    print("PEROBJ — quantificador UNIVERSAL sobre objetos, com mapeamento")
    print("=" * 78)
    tarefas = carregar(a.pasta)
    print(f"\n{len(PROPS)} propriedades x 2 conectividades, mapa DEDUZIDO")
    print("verificador: igualdade EXATA no teste, nunca visto no ajuste\n")

    base, _ = avaliar(tarefas, True, False, "so superlativo (ordem1)",
                      a.nucleos)
    cheio, ex = avaliar(tarefas, True, True, "+ universal por objeto",
                        a.nucleos)

    d = len(cheio) - len(base)
    print(f"\nuniversal - superlativo = {d:+d}")
    print(f"so o universal resolve : {sorted(cheio - base)}")
    print(f"so o superlativo resolve: {sorted(base - cheio)}")
    print("\nleis novas:")
    for n in sorted(cheio - base)[:14]:
        print(f"  {n}  {ex[n]}")

    print("\n" + "=" * 78)
    if len(base) != 4:
        print(f"P0 · o braco base deu {len(base)}/400 e `ordem1` deu 4.")
        print("     Divergiu — ler o resto com ressalva.")
    if d >= 6:
        print("P1 · o quantificador UNIVERSAL era o que faltava. O")
        print("     superlativo (argmax) era o quantificador errado: o ARC")
        print("     mapeia TODOS os objetos, nao seleciona um.")
    elif d >= 2:
        print("P2 · ajuda, na ordem dos outros mecanismos da sessao.")
    else:
        print("P3 · DUAS formas de quantificacao sobre objeto testadas e")
        print("     refutadas num passo. Ou objeto nao e o que falta, ou um")
        print("     passo e um teto que nenhum vocabulario atravessa.")


if __name__ == "__main__":
    main()
