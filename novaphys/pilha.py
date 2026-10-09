"""
novaphys.pilha — vocabulario completo COM composicao de dois passos
=====================================================================

O QUE FALTA MEDIR
-------------------
Todos os numeros de vocabulario desta sessao foram medidos com UM PASSO, e um
passo e um teto duro. A prova esta na propria linguagem por celula:

    por celula, 1 passo                       3/400
    por celula, 2 passos + cobertura 0,85     8/400

Quase 3x, so por deixar uma estrutura parcial explicar quase tudo e inferir o
residuo depois. Entao todo vocabulario testado num passo foi testado no regime
em que o teto domina:

    + superlativo (argmax sobre objetos)      4/400   +1
    + universal por objeto (∀ com mapa)       6/400   +2

Este arquivo mede o que sobra quando os dois se encontram.

    B   por celula          2 passos    reproduz `relaxado.py`, ~8
    E   vocabulario cheio   2 passos    o que nunca foi rodado

    E - B   o ganho do vocabulario quantificado NO REGIME onde composicao
            existe. Comparar com (6 - 3) = +3, que e o mesmo ganho no regime
            de um passo.

Se (E - B) > 3, vocabulario e composicao se MULTIPLICAM: cada estrutura que o
quantificador torna expressavel abre composicoes que antes nao existiam. Se
(E - B) ~ 3, apenas somam.

Essa distincao decide a estrategia inteira. Somar da ~0,4 tarefa por mecanismo
e nao sai do lugar em 400. Multiplicar e a unica forma que a aritmetica desta
sessao ja mostrou ter chance de mudar de patamar.

POR QUE ISSO PODE MULTIPLICAR, mecanicamente
----------------------------------------------
O passo parcial funciona quando existe uma estrutura que explica QUASE tudo e
deixa um residuo simples. Na linguagem por celula poucas leis tem essa forma —
por isso 3 -> 8 e nao mais.

As leituras quantificadas tem essa forma por construcao:

    porObj(tam)       explica todos os objetos e erra so o fundo
    porObj(n_buracos) explica a classificacao e erra so a posicao
    empilha(baixo)    explica a coluna inteira menos as celulas recoloridas
    dentro            explica o preenchimento menos a borda

Sao exatamente estruturas de cobertura alta com residuo pequeno — o insumo de
que a composicao de dois passos precisa e que a linguagem por celula quase
nunca produz.

RISCO DECLARADO
-----------------
A leitura por objeto e AJUSTADA aos pares de treino (o mapa propriedade -> cor
e deduzido, nao escrito). Isso da muito mais liberdade e pode decorar. A
protecao e o verificador: igualdade EXATA nas grades de TESTE, nunca vistas no
ajuste. E o mesmo verificador dos outros bracos, sem afrouxamento.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  Z0  PRE-CONDICAO. B tem que dar ~8/400 (reproduz `relaxado.py`). Se nao
      reproduzir, este arquivo divergiu e nada aqui vale.

  Z1  E - B >= 6    multiplicam
  Z2  E - B = 3 a 5 somam
  Z3  E - B <= 2    o vocabulario quantificado nao sobrevive a composicao

MINHA PREVISAO: Z2, entre +3 e +5, levando E para algo entre 11 e 13. Estou
sendo modesto de proposito porque errei uma aposta alta hoje no mesmo terreno
(previ 12-20 para o quantificador num passo e deu 4). O argumento mecanico
acima favorece Z1, mas argumento mecanico foi exatamente o que me traiu da
ultima vez.

Se der Z1, e o primeiro efeito MULTIPLICATIVO medido no projeto e muda a
estrategia: para de valer a pena procurar mecanismo novo e passa a valer
compor os que ja existem.
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Optional, Set, Tuple

from novaphys.arc import Grid, carregar, dims, fundo
from novaphys.eter import PASTA_AVAL
from novaphys.eter_inferencia import todas_leituras, todos_testes
from novaphys.ordem1 import (aplicar, inferir, leituras_quantificadas,
                             testes_quantificados)
from novaphys.ordem2 import parciais
from novaphys.perobj import leituras_por_objeto


def vocabulario(treino, cheio: bool):
    cores = sorted({c for a, b in treino for l in (a + b) for c in l})
    L = list(todas_leituras(cores))
    T = list(todos_testes(cores))
    if cheio:
        L += leituras_quantificadas()
        T += testes_quantificados()
        try:
            L += leituras_por_objeto(treino)
        except Exception:
            pass
    return L, T


def resolver(treino, teste, cheio: bool, limiar: float, teto: int):
    L, T = vocabulario(treino, cheio)
    op, _g = inferir(treino, L, T)
    if op is not None:
        if all(aplicar(op, x) == y for x, y in teste):
            return "1 passo", f"[{op[0]} ? {op[1]} : {op[2]}]"
        return None, None

    cand, _g2 = parciais(treino, L, T, limiar, teto)
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
        L2, T2 = vocabulario(meio, cheio)
        op2, _g3 = inferir(meio, L2, T2)
        if op2 is None:
            continue
        ok = True
        for x, y in teste:
            r = aplicar(p, x)
            if r is None or aplicar(op2, r) != y:
                ok = False
                break
        if ok:
            return ("2 passos",
                    f"[{p[0]} ? {p[1]} : {p[2]}] -> "
                    f"[{op2[0]} ? {op2[1]} : {op2[2]}]")
    return None, None


def uma(args):
    nome, tr, te, cheio, limiar, teto = args
    try:
        r, txt = resolver(tr, te, cheio, limiar, teto)
    except Exception:
        return nome, None, None
    return nome, r, txt


def avaliar(tarefas, cheio, limiar, teto, rotulo, nucleos):
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(nucleos) as pool:
        res = pool.map(uma, [(n, tr, te, cheio, limiar, teto)
                             for n, tr, te in tarefas], chunksize=2)
    ok = {n for n, r, _t in res if r}
    ex = {n: t for n, r, t in res if r}
    p2 = sum(1 for _n, r, _t in res if r == "2 passos")
    n = len(tarefas)
    print(f"  {rotulo:26s} {len(ok):3d}/{n}  ({len(ok)/n:5.2%})  "
          f"{p2:2d} de 2 passos  ({time.time()-t0:.0f}s)", flush=True)
    return ok, ex


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", default=PASTA_AVAL)
    ap.add_argument("--limiar", type=float, default=0.85)
    ap.add_argument("--teto", type=int, default=40)
    ap.add_argument("--nucleos", type=int, default=10)
    a = ap.parse_args()

    print("=" * 80)
    print("PILHA — vocabulario quantificado ENCONTRA composicao de 2 passos")
    print("=" * 80)
    tarefas = carregar(a.pasta)
    print(f"\nlimiar {a.limiar}, teto {a.teto}, verificador exato no teste\n")

    B, _ = avaliar(tarefas, False, a.limiar, a.teto,
                   "B por celula, 2 passos", a.nucleos)
    E, eE = avaliar(tarefas, True, a.limiar, a.teto,
                    "E vocabulario cheio, 2 passos", a.nucleos)

    d = len(E) - len(B)
    print(f"\nE - B = {d:+d}   (o mesmo ganho num passo foi +3)")
    print(f"so o E resolve : {sorted(E - B)}")
    print(f"so o B resolve : {sorted(B - E)}")
    print("\nleis do braco E:")
    for n in sorted(E - B)[:16]:
        print(f"  {n}  {eE[n]}")

    print("\n" + "=" * 80)
    if len(B) < 6:
        print(f"Z0 · B deu {len(B)}/400 e `relaxado.py` deu 8. Divergiu —")
        print("     ler o resto com ressalva.")
    if d >= 6:
        print("Z1 · vocabulario e composicao MULTIPLICAM. Primeiro efeito")
        print("     multiplicativo medido no projeto: para de valer a pena")
        print("     procurar mecanismo novo e passa a valer compor os que ha.")
    elif d >= 3:
        print("Z2 · somam. Cada mecanismo entrega o seu e nada mais.")
    else:
        print("Z3 · o vocabulario quantificado nao sobrevive a composicao.")


if __name__ == "__main__":
    main()
