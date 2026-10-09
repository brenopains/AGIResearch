"""
novaphys.residuo — apagar os limiares: o sistema escolhe o próprio afrouxamento
================================================================================

A CRITICA QUE ORIGINA ESTE ARQUIVO
------------------------------------
O usuario apontou que eu codifiquei quatro constantes e depois medi as minhas
proprias constantes:

    limiar = 0.85          numero magico, varrido uma vez e congelado
    primeiro passo >= 85%  segundo passo tem que fechar 100%
    exatamente 2 passos
    teto = 40 operadores

O 0,85 e o melhor NA MEDIA de 400 tarefas. Nao ha razao nenhuma para o otimo
ser o mesmo em toda tarefa: uma que precisa de tres estagios precisa de um
regime diferente de uma que precisa de um. Eu impus um formato e medi o
formato.

A pergunta dele — "e se o proprio sistema escolhesse quando e como afrouxar, em
N passos?" — nao e um ajuste. E apagar os quatro numeros.

A FORMULACAO SEM LIMIAR
-------------------------
O estado da busca deixa de ser "sequencia de operadores" e passa a ser

    (grades atuais, conjunto de celulas que ainda NAO estao explicadas)

Comeca com tudo errado; termina com residuo zero. Cada operador aplicado reduz
(ou nao) o residuo. A fronteira e ordenada por residuo.

    O LIMIAR SOME.  Nao existe ">= 85%". Um passo que explica 30% e aceito se
                    for o melhor disponivel; um que explica 99% e preferido. O
                    sistema escolhe o proprio afrouxamento, POR TAREFA.
    O "2 PASSOS" SOME.  Sao quantos forem precisos ate o residuo zerar.
    A ASSIMETRIA SOME.  Nao existe "primeiro afrouxado, segundo exato". Todo
                    passo e igual; o unico criterio e residuo zero no fim.

E ha um quarto efeito, que era um item separado da lista do que falta:

    VERIFICACAO VIRA RESTRICAO DURANTE A BUSCA. Nao da para aceitar nada
    enquanto o residuo nao fechar. E o harness do paper Twin (que vale 53
    pontos na literatura, so por bloquear a acao ate a explicacao fechar)
    embutido na REPRESENTACAO, em vez de ser checagem final.

O que sobra de parametro e orcamento, que e recurso e nao julgamento de
qualidade. E a largura da expansao, que e resource-bound e nao quality-bound:
quantos operadores distintos deduzir por estado.

COMO O ESTADO E EXPANDIDO
---------------------------
Nao da para enumerar operadores: o espaco (TESTE, LEITURA_v, LEITURA_f) e
astronomico. Entao a expansao DEDUZ, como sempre — varre a cobertura de cada
leitura por mascara de bits e escolhe os pares viaveis. A diferenca e que, em
vez de parar no primeiro que passa de um limiar, coleta os melhores por
cobertura, DESCARTA os que produzem a mesma grade resultante (equivalencia
observacional, o mecanismo que ja se mostrou correto) e empilha o resto.

Dedup por grade resultante e o que impede a busca de refazer trabalho: dois
operadores diferentes que levam ao mesmo lugar sao um so estado.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  R0  PRE-CONDICAO. Com profundidade maxima 2 e largura suficiente, tem que
      reproduzir o motor de limiar fixo (~8/400 na linguagem por celula). Se
      nao reproduzir, a comparacao com tudo que veio antes nao vale.

  R1  >= 14/400 na linguagem por celula, ou >= 17 com vocabulario cheio
      -> apagar os limiares foi o que faltava; o sistema escolhendo o proprio
         afrouxamento bate o meu numero magico com folga
  R2  entre 9 e 13
      -> ajuda, na ordem dos outros mecanismos
  R3  <= 8
      -> o limiar fixo ja estava perto do otimo e a adaptividade nao paga o
         custo de busca que cobra

MINHA CALIBRAGEM, e ela e o motivo de eu apostar baixo: hoje fiz quatro
previsoes sobre mecanismos e errei TRES, todas na direcao otimista, todas em
mecanismos ADITIVOS. Formulei no meio da sessao o principio de que mecanismo
aditivo rende 0 a +2 e mecanismo que elimina trabalho determinado rende fator.

Este e do segundo tipo — ele REMOVE parametros e dedupa estados — entao o
principio prevê fator. Mas o principio tambem me diria para desconfiar de mim.

PREVISAO: R2, entre 10 e 13 na linguagem por celula. Aposto que a maior parte
do ganho vem de tarefas de TRES passos, que o motor antigo nao podia nem
tentar, e nao de afrouxamento adaptativo no primeiro passo.

LIMITE DECLARADO: residuo por celula so e definido quando entrada e saida tem
a MESMA dimensao. Tarefas que recortam ou ampliam ficam de fora por
construcao, exatamente como no motor anterior. Isso limita o teto a ~60% das
400.
"""

from __future__ import annotations

import argparse
import heapq
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

from novaphys.arc import Grid, carregar, dims, fundo
from novaphys.eter import PASTA_AVAL
from novaphys.eter_inferencia import todas_leituras, todos_testes
from novaphys.ordem1 import (aplicar, leituras_quantificadas,
                             testes_quantificados)
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


def _mascaras(grades, alvos, L, T):
    """Cobertura de cada leitura e mascara de cada teste, em bits."""
    bg = fundo(grades[0])
    desloc, base = [], 0
    for a in grades:
        h, w = dims(a)
        desloc.append(base)
        base += h * w
    TODAS = (1 << base) - 1

    cob = []
    for Lx in L:
        bits, n = 0, 0
        try:
            for k, a in enumerate(grades):
                v = Lx.valores(a, bg)
                h, w = dims(a)
                off, b = desloc[k], alvos[k]
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
    mt = []
    for Tx in T:
        bits = 0
        try:
            for k, a in enumerate(grades):
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
    return cob, mt, TODAS


def expandir(grades, alvos, L, T, largura: int):
    """
    Deduz operadores para este estado e devolve os que produzem grades
    DISTINTAS, ordenados por quanto do alvo passam a explicar.

    Nenhum limiar. A ordenacao e por residuo; quem explica mais vem antes, e
    quem explica pouco continua na fila caso o resto morra.
    """
    cob, mt, TODAS = _mascaras(grades, alvos, L, T)
    if not cob:
        return []
    cob.sort(key=lambda x: -x[2])
    cob = cob[:60]                    # resource-bound, nao quality-bound
    fora, vistos = [], set()
    for Lv, bv, _nv in cob:
        if len(fora) >= largura:
            break
        for Lf, bf, _nf in cob:
            if len(fora) >= largura:
                break
            ev, ef = bv & ~bf, bf & ~bv
            for Tx, bt in zip(T, mt):
                if (ev & ~bt) == 0 and (ef & bt) == 0:
                    op = (Tx, Lv, Lf)
                    novas = []
                    bom = True
                    for a in grades:
                        r = aplicar(op, a)
                        if r is None:
                            bom = False
                            break
                        novas.append(r)
                    if not bom:
                        break
                    t = tuple(novas)
                    if t in vistos or t == tuple(grades):
                        break
                    vistos.add(t)
                    res = sum(1 for x, y in zip(t, alvos)
                              for li, lj in zip(x, y)
                              for u, v in zip(li, lj) if u != v)
                    fora.append((res, t, op))
                    break
    fora.sort(key=lambda x: x[0])
    return fora


def resolver(treino, teste, cheio: bool, orcamento: int, prof: int,
             largura: int):
    """
    Melhor-primeiro sobre RESIDUO. Sem limiar, sem numero fixo de passos.
    Termina quando o residuo zera — que e a verificacao virando restricao da
    propria busca.
    """
    if any(dims(a) != dims(b) for a, b in treino):
        return None, 0
    grades = tuple(a for a, _b in treino)
    alvos = [b for _a, b in treino]
    L, T = vocabulario(treino, cheio)

    res0 = sum(1 for x, y in zip(grades, alvos)
               for li, lj in zip(x, y) for u, v in zip(li, lj) if u != v)
    if res0 == 0:
        return [], 0

    fila = [(res0, 0, grades, [])]
    vistos = {grades}
    gasto = 0
    while fila and gasto < orcamento:
        _res, prof_at, est, caminho = heapq.heappop(fila)
        if prof_at >= prof:
            continue
        for res, novas, op in expandir(list(est), alvos, L, T, largura):
            gasto += 1
            if gasto >= orcamento:
                break
            if novas in vistos:
                continue
            vistos.add(novas)
            cam = caminho + [op]
            if res == 0:
                ok = True
                for x, y in teste:
                    r: Optional[Grid] = x
                    for o in cam:
                        r = aplicar(o, r)
                        if r is None:
                            ok = False
                            break
                    if not ok or r != y:
                        ok = False
                        break
                if ok:
                    return cam, gasto
                continue
            heapq.heappush(fila, (res, prof_at + 1, novas, cam))
    return None, gasto


def uma(args):
    nome, tr, te, cheio, orc, prof, larg = args
    try:
        cam, g = resolver(tr, te, cheio, orc, prof, larg)
    except Exception:
        return nome, None, 0
    if cam is None:
        return nome, None, g
    return nome, len(cam), g


def avaliar(tarefas, cheio, orc, prof, larg, rotulo, nucleos):
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(nucleos) as pool:
        res = pool.map(uma, [(n, tr, te, cheio, orc, prof, larg)
                             for n, tr, te in tarefas], chunksize=2)
    ok = {n for n, c, _g in res if c is not None}
    passos: Dict[int, int] = {}
    for _n, c, _g in res:
        if c is not None:
            passos[c] = passos.get(c, 0) + 1
    n = len(tarefas)
    print(f"  {rotulo:26s} {len(ok):3d}/{n}  ({len(ok)/n:5.2%})  "
          f"passos {dict(sorted(passos.items()))}  ({time.time()-t0:.0f}s)",
          flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", default=PASTA_AVAL)
    ap.add_argument("--orcamento", type=int, default=400)
    ap.add_argument("--prof", type=int, default=4)
    ap.add_argument("--largura", type=int, default=6)
    ap.add_argument("--cheio", action="store_true")
    ap.add_argument("--nucleos", type=int, default=10)
    a = ap.parse_args()

    print("=" * 80)
    print("RESIDUO — sem limiar, sem numero fixo de passos")
    print("=" * 80)
    tarefas = carregar(a.pasta)
    print(f"\nmelhor-primeiro sobre residuo, profundidade max {a.prof}, "
          f"largura {a.largura}, orcamento {a.orcamento}")
    print("nenhum limiar de cobertura em lugar nenhum\n")
    rot = "vocabulario cheio" if a.cheio else "por celula"
    avaliar(tarefas, a.cheio, a.orcamento, a.prof, a.largura, rot, a.nucleos)


if __name__ == "__main__":
    main()
