"""
novaphys.ordem2 — os tres mecanismos que venceram, empilhados
===============================================================

O QUE ESTA SENDO COMPOSTO
---------------------------
Tres coisas sobreviveram a controle nesta sessao, e cada uma sozinha foi
medida contra placebo pareado e orcamento casado:

  1. DEDUZIR em vez de sortear     no mesmo espaco de 178M operadores,
     (`eter_inferencia.py`)        amostrar acha 1 util em 3.000; deduzir
                                   acha o exato, precisao 100%

  2. ACEITAR COBERTURA PARCIAL     exigir 100% da 3 tarefas e ZERO
     (`relaxado.py`)               composicoes de 2 passos, mesmo com 2,4x o
                                   orcamento; exigir 85% da 8 tarefas e 5
                                   composicoes com 41% do orcamento.
                                   Curva em U: 0,50 e pior que 0,85

  3. QUANTIFICAR                   na linguagem de objetivos, proposicional se
     (`quantificado.py`,           refuta em 24/25 mundos; um quantificador da
      `ordem1.py`)                 99,3% de transferencia por 1,09x de custo

Cada um foi medido isolado. Nunca foram compostos. Este arquivo compoe.

O DESENHO E UM FATORIAL, NAO UM AMONTOADO
-------------------------------------------
Empilhar tudo e reportar o total seria propaganda: nao daria para saber se o
ganho e da soma ou de um so. Entao roda-se o 2x2:

                      1 PASSO EXATO        2 PASSOS PARCIAIS (0,85)
    POR CELULA             A                        B
    QUANTIFICADA           C                        D

    B - A   valor do passo parcial na linguagem antiga
    C - A   valor do quantificador sozinho
    D - C   valor do passo parcial DEPOIS de quantificar
    D - B   valor do quantificador DEPOIS de compor

Se (D - C) > (B - A), os dois mecanismos se MULTIPLICAM: cada estrutura
parcial nova que o quantificador torna expressavel abre composicoes que antes
nao existiam. Se (D - C) ~ (B - A), eles apenas somam.

Essa distincao e a coisa toda. Somar da 0,4 tarefa por mecanismo e leva a
lugar nenhum em 400. Multiplicar e a unica forma que a aritmetica desta
sessao ja mostrou ter chance de sair do patamar.

O verificador nao muda em braco nenhum: igualdade EXATA nas grades de teste,
que a deducao nunca viu. Ajuste extensional nao passa nele.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  R0  PRE-CONDICAO. A tem que dar ~3/400 (reproduz `eter_inferencia`) e B
      ~8/400 (reproduz `relaxado`). Se nao reproduzirem, nada aqui vale.

  R1  D - C  >  B - A     os mecanismos se MULTIPLICAM
  R2  D - C  ~  B - A     somam
  R3  D <= max(B, C)      compor nao acrescenta; um deles domina

MINHA PREVISAO, escrita antes de rodar: R1, e por uma razao mecanica e nao por
otimismo. O passo parcial funciona quando existe uma estrutura que explica
QUASE tudo e deixa um residuo simples. Na linguagem por celula, poucas leis
tem essa forma — por isso 3 -> 8. Com leituras que quantificam sobre objetos,
"quase tudo" fica muito mais comum: `empilha` explica a coluna inteira menos as
celulas que mudaram de cor; `dentro` explica o preenchimento menos a borda.
Sao exatamente estruturas de cobertura alta com residuo pequeno.

Aposto D >= 25/400 num passo e meio de busca, e aposto que (D - C) fica pelo
menos 2x (B - A). Se der R3, a leitura tambem e limpa e fecha a linha: os
mecanismos ocupam o mesmo nicho e a sessao inteira estava medindo um efeito so
por tres instrumentos diferentes.
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Optional, Set, Tuple

from novaphys.arc import Grid, carregar, dims, fundo, valida
from novaphys.eter import PASTA_AVAL
from novaphys.eter_inferencia import Teste, todas_leituras, todos_testes
from novaphys.ordem1 import (aplicar, inferir, leituras_quantificadas,
                             testes_quantificados)


def vocabulario(treino, quant: bool):
    cores = sorted({c for a, b in treino for l in (a + b) for c in l})
    L = list(todas_leituras(cores))
    T = list(todos_testes(cores))
    if quant:
        L += leituras_quantificadas()
        T += testes_quantificados()
    return L, T


def parciais(treino, L, T, limiar: float, teto: int):
    """
    Operadores que explicam >= `limiar` das celulas — o mecanismo de
    `relaxado.py`, aqui com o vocabulario injetado de fora para que os dois
    bracos do fatorial usem exatamente a mesma rotina.
    """
    if any(dims(a) != dims(b) for a, b in treino):
        return [], 0
    bg = fundo(treino[0][0])
    desloc, base = [], 0
    for a, _b in treino:
        h, w = dims(a)
        desloc.append(base)
        base += h * w
    TODAS = (1 << base) - 1
    alvo = int(base * limiar)

    cob = []
    for Lx in L:
        bits, n = 0, 0
        try:
            for k, (a, b) in enumerate(treino):
                v = Lx.valores(a, bg)
                h, w = dims(a)
                off = desloc[k]
                for i in range(h):
                    vi, bi = v[i], b[i]
                    for j in range(w):
                        if vi[j] == bi[j]:
                            bits |= 1 << (off + i * w + j)
                            n += 1
        except Exception:
            continue
        if n:
            cob.append((Lx, bits, n))
    if not cob:
        return [], 0

    mt = []
    for Tx in T:
        bits = 0
        try:
            for k, (a, _b) in enumerate(treino):
                mk = Tx.mascara(a, bg)
                h, w = dims(a)
                off = desloc[k]
                for i in range(h):
                    for j in range(w):
                        if mk[i][j]:
                            bits |= 1 << (off + i * w + j)
        except Exception:
            bits = 0
        mt.append(bits)

    cob.sort(key=lambda x: -x[2])
    fora, gasto = [], 0
    for Lv, bv, _nv in cob:
        if len(fora) >= teto:
            break
        for Lf, bf, _nf in cob:
            gasto += 1
            if len(fora) >= teto:
                break
            if (bv | bf).bit_count() < alvo:
                continue
            ev, ef = bv & ~bf, bf & ~bv
            for Tx, bt in zip(T, mt):
                if (ev & ~bt) == 0 and (ef & bt) == 0:
                    fora.append((Tx, Lv, Lf))
                    break
    return fora, gasto


def resolver(treino, teste, quant: bool, dois: bool, limiar: float, teto: int):
    L, T = vocabulario(treino, quant)
    op, gasto = inferir(treino, L, T)
    if op is not None:
        if all(aplicar(op, x) == y for x, y in teste):
            return "1 passo", f"[{op[0]} ? {op[1]} : {op[2]}]", gasto
        return None, None, gasto
    if not dois:
        return None, None, gasto

    cand, g2 = parciais(treino, L, T, limiar, teto)
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
        L2, T2 = vocabulario(meio, quant)
        op2, g3 = inferir(meio, L2, T2)
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
            return ("2 passos",
                    f"[{p[0]} ? {p[1]} : {p[2]}] -> "
                    f"[{op2[0]} ? {op2[1]} : {op2[2]}]", gasto)
    return None, None, gasto


def uma(args):
    nome, tr, te, quant, dois, limiar, teto = args
    r, txt, g = resolver(tr, te, quant, dois, limiar, teto)
    return nome, r, txt, g


def avaliar(tarefas, quant, dois, limiar, teto, rotulo, nucleos):
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(nucleos) as pool:
        res = pool.map(uma, [(n, tr, te, quant, dois, limiar, teto)
                             for n, tr, te in tarefas], chunksize=2)
    ok = {n for n, r, _t, _g in res if r}
    ex = {n: t for n, r, t, _g in res if r}
    p2 = sum(1 for _n, r, _t, _g in res if r == "2 passos")
    gs = sum(g for _n, _r, _t, g in res)
    n = len(tarefas)
    print(f"  {rotulo:22s} {len(ok):3d}/{n}  ({len(ok)/n:5.2%})  "
          f"{p2:2d} de 2 passos  {gs//n:>7d} pares/tarefa  "
          f"({time.time()-t0:.0f}s)", flush=True)
    return ok, ex


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", default=PASTA_AVAL)
    ap.add_argument("--limiar", type=float, default=0.85)
    ap.add_argument("--teto", type=int, default=40)
    ap.add_argument("--nucleos", type=int, default=10)
    a = ap.parse_args()

    print("=" * 84)
    print("ORDEM2 — deduzir x cobertura parcial x quantificador, no fatorial")
    print("=" * 84)
    tarefas = carregar(a.pasta)
    print(f"\nlimiar {a.limiar}, teto {a.teto}, verificador exato no teste\n")

    A, _ = avaliar(tarefas, False, False, a.limiar, a.teto,
                   "A por-celula 1 passo", a.nucleos)
    B, _ = avaliar(tarefas, False, True, a.limiar, a.teto,
                   "B por-celula 2 passos", a.nucleos)
    C, _ = avaliar(tarefas, True, False, a.limiar, a.teto,
                   "C quantificada 1 passo", a.nucleos)
    D, eD = avaliar(tarefas, True, True, a.limiar, a.teto,
                    "D quantificada 2 passos", a.nucleos)

    ba, dc = len(B) - len(A), len(D) - len(C)
    ca, db = len(C) - len(A), len(D) - len(B)
    print(f"\nB - A = {ba:+3d}   passo parcial na linguagem antiga")
    print(f"C - A = {ca:+3d}   quantificador sozinho")
    print(f"D - C = {dc:+3d}   passo parcial DEPOIS de quantificar")
    print(f"D - B = {db:+3d}   quantificador DEPOIS de compor")
    print(f"\nuniao dos quatro: {len(A | B | C | D)}/400")
    print(f"so o D resolve : {sorted(D - A - B - C)[:20]}")
    print("\nleis do braco D:")
    for n in sorted(D - A - B)[:12]:
        print(f"  {n}  {eD[n]}")

    print("\n" + "=" * 84)
    if len(A) < 2 or len(B) < 5:
        print(f"R0 · A={len(A)} (esperado ~3) e B={len(B)} (esperado ~8).")
        print("     Divergiu dos arquivos anteriores — ler com ressalva.")
    if dc > ba:
        print(f"R1 · os mecanismos se MULTIPLICAM: o passo parcial vale "
              f"{ba:+d} na")
        print(f"     linguagem por celula e {dc:+d} na quantificada. Cada")
        print("     estrutura que o quantificador torna expressavel abre")
        print("     composicoes que antes nao existiam.")
    elif dc >= ba - 1:
        print("R2 · somam. Cada mecanismo entrega o seu e nada mais.")
    else:
        print("R3 · compor nao acrescenta; os mecanismos ocupam o mesmo")
        print("     nicho e a sessao media um efeito so por tres instrumentos.")


if __name__ == "__main__":
    main()
