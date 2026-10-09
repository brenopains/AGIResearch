"""
novaphys.feixe — sustentar N pensamentos parciais SEM ficar viciada
=====================================================================

DE ONDE VEM
-------------
A ideia do usuario: sustentar N estados mentais ao mesmo tempo, compor um em
seguida do outro como um video mental, e aprender com isso "sem ficar viciada".

A primeira metade — sustentar pensamento PARCIAL — ja foi medida e e um dos
tres mecanismos que sobreviveram a controle nesta sessao. `relaxado.py`: exigir
cobertura 100% do operador da 3 tarefas e ZERO composicoes de 2 passos, mesmo
com 2,4x o orcamento; exigir 85% da 8 tarefas e 5 composicoes com 41% do
orcamento. Curva em U — 0,50 e pior que 0,85.

A segunda metade — nao ficar viciada — nunca foi testada, e ela tem endereco
exato no codigo. Em `relaxado.py`:

    cob.sort(key=lambda x: -x[2])      # pega os K melhores por cobertura

Os K melhores por cobertura tendem a ser QUASE-COPIAS. E ha uma medida deste
projeto que diz o tamanho do problema: 3.486 hipoteses da gramatica colapsam em
apenas 324 comportamentos distintos — 10,8x de redundancia por equivalencia
observacional. Um feixe cheio de clones tem largura nominal K e largura
EFETIVA K/10.

Isso e vicio no sentido literal e mensuravel: a politica de retencao nao
distingue um pensamento novo de mais uma variacao do mesmo pensamento.

O TESTE
---------
Mesmo motor, mesmo limiar, mesmo teto de operadores carregados, mesmo teto de
pares varridos. Muda so a POLITICA DE RETENCAO:

    VICIADO   aceita os `teto` primeiros que passam do limiar (ordem por
              cobertura) — exatamente o `relaxado.py` de hoje
    DIVERSO   aceita um operador so se o CONJUNTO DE CELULAS que ele explica
              for suficientemente diferente do que ja esta no feixe
              (Jaccard <= `sim`), e continua varrendo ate encher

O braco DIVERSO paga por isso: varre mais pares para encher as mesmas vagas.
Por isso o teto de pares varridos e igual nos dois e entra no relatorio — se
ele gastar mais orcamento, o ganho nao vale.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  F0  PRE-CONDICAO. O braco VICIADO tem que reproduzir `relaxado.py` no mesmo
      limiar (8 tarefas em 0,85). Se nao reproduzir, este arquivo diverge do
      anterior e a comparacao nao vale.

  F1  DIVERSO > VICIADO no mesmo orcamento
      -> a politica de retencao e o mecanismo. Sustentar N pensamentos so
         vale se os N forem diferentes, e a redundancia de 10,8x estava
         comendo a largura do feixe.

  F2  DIVERSO = VICIADO
      -> a redundancia existe mas nao custa: os clones sao descartados de
         graca mais adiante e a diversidade nao compra nada.

  F3  DIVERSO < VICIADO
      -> cobertura alta e informativa e forcar diversidade joga fora os
         melhores candidatos. O vicio seria, na verdade, foco.

MINHA PREVISAO, escrita antes de rodar: eu comecei esta linha esperando NULO,
porque profundidade ja foi refutada nesta sessao (o feixe chegou a
profundidade 10 e nada apareceu alem da 4) e porque guiar a busca deu tres
nulos. Mudei de ideia ao localizar o `sort` e lembrar da medida de 10,8x: os
tres nulos anteriores eram de RANQUEAMENTO, e este e de RETENCAO — a unica
familia que ja moveu numero aqui.

Entao aposto F1, modesto: +1 a +3 tarefas sobre as 8, e aposto que o braco
DIVERSO varre mais pares para encher o feixe (custo real, tem que aparecer no
relatorio). Se der F3, a leitura tambem e util: significa que cobertura alta
carrega informacao e que "focar" bate "diversificar" neste espaco.
"""

from __future__ import annotations

import argparse
import time
from typing import List, Optional, Tuple

from novaphys.arc import Grid, carregar, dims, fundo
from novaphys.eter import PASTA_AVAL
from novaphys.eter_inferencia import (Leitura, Teste, aplicar, inferir,
                                      todas_leituras, todos_testes)
from novaphys.relaxado import cobertura_bits


def parciais(treino, limiar: float, teto: int, teto_gasto: int,
             diverso: bool, sim: float
             ) -> Tuple[List[Tuple[Teste, Leitura, Leitura]], int]:
    """
    Operadores que explicam >= `limiar` das celulas.

    `diverso=False` reproduz `relaxado.py`: aceita os `teto` primeiros na
    ordem por cobertura. `diverso=True` so aceita se o conjunto de celulas
    explicadas for novo — Jaccard <= `sim` contra tudo que ja esta no feixe.

    Os dois param no mesmo `teto_gasto` de pares varridos, entao o custo da
    diversidade aparece: ela precisa varrer mais para encher as mesmas vagas.
    """
    if any(dims(a) != dims(b) for a, b in treino):
        return [], 0
    bg = fundo(treino[0][0])
    cores = sorted({c for a, b in treino for l in (a + b) for c in l})
    leituras = todas_leituras(cores)
    cob, TODAS, desloc = cobertura_bits(treino, leituras, bg)
    if not cob:
        return [], 0
    alvo = int(TODAS.bit_count() * limiar)

    testes = todos_testes(cores)
    mt = []
    for T in testes:
        bits = 0
        for k, (a, _b) in enumerate(treino):
            mk = T.mascara(a, bg)
            h, w = dims(a)
            off = desloc[k]
            for i in range(h):
                for j in range(w):
                    if mk[i][j]:
                        bits |= 1 << (off + i * w + j)
        mt.append(bits)

    cob.sort(key=lambda x: -x[2])
    fora: List[Tuple[Teste, Leitura, Leitura]] = []
    aceitos: List[int] = []          # padrao de celulas explicadas por cada um
    gasto = 0
    for Lv, bv, _nv in cob:
        if len(fora) >= teto or gasto >= teto_gasto:
            break
        for Lf, bf, _nf in cob:
            gasto += 1
            if len(fora) >= teto or gasto >= teto_gasto:
                break
            juntos = bv | bf
            if juntos.bit_count() < alvo:
                continue
            ev, ef = bv & ~bf, bf & ~bv
            for T, bt in zip(testes, mt):
                if (ev & ~bt) == 0 and (ef & bt) == 0:
                    if diverso and aceitos:
                        # VICIO = mais uma variacao do mesmo pensamento.
                        # Jaccard sobre as celulas explicadas mede isso
                        # direto: 1,0 e clone perfeito.
                        pior = max(
                            (juntos & p).bit_count()
                            / max((juntos | p).bit_count(), 1)
                            for p in aceitos)
                        if pior > sim:
                            break        # ja penso assim; procura outro
                    fora.append((T, Lv, Lf))
                    aceitos.append(juntos)
                    break
    return fora, gasto


def resolver(treino, teste, limiar: float, teto: int, teto_gasto: int,
             diverso: bool, sim: float) -> Tuple[Optional[str], int]:
    """Um passo rigido; se falhar, um parcial seguido de inferencia no residuo."""
    op, gasto = inferir(treino)
    if op is not None:
        if all(aplicar(op, x) == y for x, y in teste):
            return "1 passo", gasto
        return None, gasto

    cand, g2 = parciais(treino, limiar, teto, teto_gasto, diverso, sim)
    gasto += g2
    for p in cand:
        meio, bom = [], True
        for a, b in treino:
            r = aplicar(p, a)
            if r is None or dims(r) != dims(b):
                bom = False
                break
            meio.append((r, b))
        if not bom:
            continue
        op2, g3 = inferir(meio)
        gasto += g3
        if op2 is None:
            continue
        ok = True
        for x, y in teste:
            r = aplicar(p, x)
            if r is None or aplicar(op2, r) != y:
                ok = False
                break
        if ok:
            return "2 passos", gasto
    return None, gasto


def uma(args):
    nome, tr, te, limiar, teto, teto_gasto, diverso, sim = args
    r, g = resolver(tr, te, limiar, teto, teto_gasto, diverso, sim)
    return nome, r, g


def avaliar(tarefas, limiar, teto, teto_gasto, diverso, sim, rotulo,
            nucleos=10):
    from multiprocessing import Pool
    ok, dois, gastos = set(), 0, 0
    t0 = time.time()
    with Pool(nucleos) as pool:
        res = pool.map(uma, [(n, tr, te, limiar, teto, teto_gasto, diverso,
                              sim) for n, tr, te in tarefas], chunksize=4)
    for nome, r, g in res:
        gastos += g
        if r is not None:
            ok.add(nome)
            dois += int(r == "2 passos")
    print(f"  {rotulo:12s} {len(ok):3d}/{len(tarefas)}  "
          f"{dois} de 2 passos   {gastos // len(tarefas):>7d} pares/tarefa   "
          f"({time.time() - t0:.0f}s)", flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", default=PASTA_AVAL)
    ap.add_argument("--limiar", type=float, default=0.85)
    ap.add_argument("--teto", type=int, default=40)
    ap.add_argument("--gasto", type=int, default=60000)
    ap.add_argument("--nucleos", type=int, default=10)
    ap.add_argument("--sim", type=float, default=0.9)
    a = ap.parse_args()

    print("=" * 74)
    print("FEIXE — sustentar N pensamentos parciais SEM ficar viciada")
    print("=" * 74)
    tarefas = carregar(a.pasta)
    print(f"\nlimiar {a.limiar}, teto {a.teto} operadores no feixe, "
          f"teto {a.gasto} pares varridos")
    print(f"DIVERSO rejeita candidato com Jaccard > {a.sim} contra o feixe\n")

    vic = avaliar(tarefas, a.limiar, a.teto, a.gasto, False, 1.0,
                  "VICIADO", a.nucleos)
    div = avaliar(tarefas, a.limiar, a.teto, a.gasto, True, a.sim,
                  "DIVERSO", a.nucleos)

    print(f"\nDIVERSO - VICIADO = {len(div) - len(vic):+d}")
    print(f"so o DIVERSO resolve : {sorted(div - vic)}")
    print(f"so o VICIADO resolve : {sorted(vic - div)}")

    print("\n" + "=" * 74)
    if len(vic) < 6:
        print(f"F0 · o braco VICIADO deu {len(vic)}, e `relaxado.py` deu 8 no")
        print("     mesmo limiar. Divergiu do anterior — nao comparar.")
    if len(div) > len(vic):
        print("F1 · a POLITICA DE RETENCAO e o mecanismo. Sustentar N")
        print("     pensamentos so vale se os N forem diferentes: a")
        print("     redundancia de 10,8x estava comendo a largura do feixe.")
    elif len(div) == len(vic):
        print("F2 · a redundancia existe e nao custa. Os clones sao")
        print("     descartados de graca adiante; diversidade nao compra.")
    else:
        print("F3 · cobertura alta E informativa e forcar diversidade joga")
        print("     fora os melhores. O 'vicio' era foco.")


if __name__ == "__main__":
    main()
