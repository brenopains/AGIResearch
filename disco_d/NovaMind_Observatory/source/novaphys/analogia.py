"""
novaphys.analogia — esquemas de uma tarefa instanciados noutra
================================================================

A PERGUNTA EXATA QUE ISTO ATACA
---------------------------------
Toda maquina que funciona neste projeto e SUBTRATIVA — deducao elimina
leituras que nao cobrem, eliminacao remove objetivos contraditos, quociente
remove duplicatas, residuo remove estados que nao reduzem. E ha prova de que
subtrair nao pode criar: sob eliminacao pura a informatividade e MONOTONA
NAO-CRESCENTE, verificado por forca bruta em 200.000 casos, zero excecoes.

Cinco tentativas de construir a metade GERADORA falharam, e nenhuma delas
gerava de verdade: amostrar num espaco infinito (1 util em 3.000), fabricar o
prior de LLM (aprendeu o amostrador), filtrar uma lista minha (+3, a 0,4
tarefa por lei), deduzir de um simulador emergente (+0), indexar pela cena
(+0). Amostrar, filtrar, ajustar e indexar sao todos "escolher dentro do
espaco".

A ideia do usuario: gerar hipoteses por ANALOGIA ENTRE TAREFAS.

POR QUE ISTO ESCAPA, E O ARGUMENTO E TECNICO
----------------------------------------------
O espaco nao e definido pelo que a gramatica expressa — e definido pelo que o
ORCAMENTO ALCANCA. Profundidade 3 com 38 primitivas sao 54.872 programas
contra um teto de 14.424; profundidade 5 e astronomica.

Um esquema abstraido de uma solucao da tarefa A, instanciado na tarefa B,
ATERRISSA em profundidade 5 sem percorrer o caminho. Nao amplia a gramatica:
torna alcancavel o que era inalcancavel.

E obedece ao unico principio que sobreviveu a sessao inteira — "recusar
refazer trabalho ja determinado" — aplicado ENTRE tarefas em vez de dentro de
uma. Todas as cinco tentativas anteriores resolviam cada tarefa isoladamente.

O DESENHO
-----------
1. Resolve o conjunto de TREINO (2,6x mais facil: mais solucoes para abstrair).
2. Abstrai cada solucao num ESQUEMA com buracos — guarda a FORMA (que familia
   de teste, que familia de leitura), joga fora cor, deslocamento, seletor.
3. No conjunto de AVALIACAO, enumera SO instanciacoes desses esquemas.
4. Controle pareado: esquemas abstraidos de triplas ALEATORIAS, mesma
   quantidade e mesma distribuicao de forma.

O passo 4 e o que faz valer. Sem ele, "esquema ajuda" pode ser so "buscar
menos coisas e mais barato" — e ai o efeito seria da restricao, nao da
analogia. E o mesmo erro que a auditoria de `prior.py` ja pegou uma vez,
quando o controle nao continha a resposta e a comparacao virou "tem contra
nao tem".

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  A0  PRE-CONDICAO. O conjunto de treino tem que render pelo menos 15
      solucoes para abstrair. Com menos que isso nao ha biblioteca — foi
      exatamente esse o bloqueio medido antes (21 ocorrencias de fragmento em
      51 solucoes, e 33 delas de um passo so).

  A1  ESQUEMA-REAL - ESQUEMA-ALEATORIO >= +4, orcamento casado
      -> analogia entre tarefas GERA. Primeira metade geradora do projeto.
  A2  +1 a +3
      -> ajuda, na ordem dos outros mecanismos
  A3  <= 0
      -> a forma das solucoes do treino nao transfere para a avaliacao, e o
         conjunto de treino ser 2,6x mais facil pode ser exatamente por isso:
         as tarefas faceis tem formas que as dificeis nao tem

MINHA CALIBRAGEM: hoje errei quatro de cinco previsoes sobre mecanismos, todas
otimistas. Mas as cinco erradas eram ADITIVAS e esta e do tipo que funciona.
O principio prevê fator; a minha calibragem manda desconfiar de mim.

PREVISAO: A2, +1 a +3. Aposto baixo por dois motivos concretos e nao por
humildade: (a) 33 de 51 solucoes tem UM passo, e esquema de um passo e quase
o espaco inteiro, entao abstrai pouco; (b) o conjunto de treino ser 2,6x mais
facil sugere que ele e facil por ter formas simples, que sao justamente as que
a busca exaustiva ja alcanca sem ajuda.

Se der A3, a leitura e limpa e fecha a linha de biblioteca: transferencia de
forma entre tarefas do ARC nao existe no nivel em que este motor opera.
"""

from __future__ import annotations

import argparse
import random
import time
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from novaphys.arc import Grid, carregar, dims, fundo
from novaphys.eter import PASTA_AVAL
from novaphys.eter_inferencia import (Leitura, Teste, todas_leituras,
                                      todos_testes)
from novaphys.ordem1 import aplicar, inferir

PASTA_TREINO = PASTA_AVAL.replace("evaluation", "training")


# ══════════════════════════════════════════════ o esquema
def forma_leitura(L) -> str:
    """
    A FORMA de uma leitura: guarda a ESTRUTURA, joga fora o CONTEUDO.

    ERRO DE GRANULARIDADE na primeira versao, e ele esvaziou o experimento.
    Eu jogava fora a MATRIZ e guardava so "e matricial, borda toroidal" —
    mas a matriz e exatamente onde mora a geometria (rotacao, transposicao,
    espelho). O esquema resultante admitia quase todo o conjunto de leituras
    e nao restringia nada: 6 das 14 solucoes caiam em
    ('sempre','mat:toro','mat:toro'), que e praticamente o espaco inteiro.

    Correcao: guarda a matriz e a borda (a transformacao geometrica), joga
    fora so os deslocamentos (o conteudo posicional).
    """
    if getattr(L, "const", False):
        return "const"
    m = getattr(L, "m", None)
    b = getattr(L, "borda", "?")
    return f"m{m}:{b}"


def forma_teste(T) -> str:
    """A FORMA de um teste: o tipo, sem cor, eixo, modulo ou resto."""
    return getattr(T, "tipo", type(T).__name__)


def esquema(op) -> Tuple[str, str, str]:
    T, Lv, Lf = op
    return (forma_teste(T), forma_leitura(Lv), forma_leitura(Lf))


# ══════════════════════════════════════════════ colher esquemas do treino
def resolver_1(treino, L, T):
    op, g = inferir(treino, L, T)
    return op, g


def colher(pasta: str, limite: Optional[int], nucleos: int
           ) -> Tuple[List[Tuple[str, str, str]], int]:
    """Resolve o treino e devolve os esquemas das solucoes encontradas."""
    from multiprocessing import Pool
    tarefas = carregar(pasta)
    if limite:
        tarefas = tarefas[:limite]
    with Pool(nucleos) as pool:
        res = pool.map(_uma_treino, [(n, tr, te) for n, tr, te in tarefas],
                       chunksize=2)
    esq = [e for e in res if e is not None]
    return esq, len(tarefas)


def _uma_treino(args):
    nome, tr, te = args
    try:
        cores = sorted({c for a, b in tr for l in (a + b) for c in l})
        op, _g = inferir(tr, todas_leituras(cores), todos_testes(cores))
        if op is None:
            return None
        if not all(aplicar(op, x) == y for x, y in te):
            return None
        return esquema(op)
    except Exception:
        return None


# ══════════════════════════════════════════════ busca restrita a esquemas
def busca_esquema(treino, teste, esquemas: Set[Tuple[str, str, str]],
                  orcamento: int, passos: int):
    """
    Enumera cadeias de ate `passos` operadores, mas so aceita operadores cuja
    FORMA esta no conjunto de esquemas. Isso e o teletransporte: a restricao
    de forma corta o fator de ramificacao o bastante para alcancar
    profundidades que a busca livre nao alcanca no mesmo orcamento.
    """
    if any(dims(a) != dims(b) for a, b in treino):
        return None, 0
    cores = sorted({c for a, b in treino for l in (a + b) for c in l})
    L = [x for x in todas_leituras(cores)]
    T = [x for x in todos_testes(cores)]
    # so as pecas cujas formas aparecem em algum esquema
    fv = {e[1] for e in esquemas} | {e[2] for e in esquemas}
    ft = {e[0] for e in esquemas}
    L = [x for x in L if forma_leitura(x) in fv]
    T = [x for x in T if forma_teste(x) in ft]
    if not L or not T:
        return None, 0

    alvos = [b for _a, b in treino]
    est = [tuple(a for a, _b in treino)]
    caminhos: List[List] = [[]]
    gasto = 0
    for _p in range(passos):
        nova_e, nova_c = [], []
        vistos = set()
        for grades, cam in zip(est, caminhos):
            if gasto >= orcamento:
                break
            bg = fundo(grades[0])
            for Tx in T:
                for Lv in L:
                    for Lf in L:
                        if gasto >= orcamento:
                            break
                        if (forma_teste(Tx), forma_leitura(Lv),
                                forma_leitura(Lf)) not in esquemas:
                            continue
                        gasto += 1
                        op = (Tx, Lv, Lf)
                        saidas, ok = [], True
                        for g in grades:
                            r = aplicar(op, g)
                            if r is None:
                                ok = False
                                break
                            saidas.append(r)
                        if not ok:
                            continue
                        t = tuple(saidas)
                        if t in vistos:
                            continue
                        vistos.add(t)
                        if list(t) == alvos:
                            return cam + [op], gasto
                        nova_e.append(t)
                        nova_c.append(cam + [op])
        if not nova_e:
            break
        # mantem a fronteira sob controle: os mais proximos do alvo
        ordem = sorted(range(len(nova_e)),
                       key=lambda k: sum(1 for x, y in zip(nova_e[k], alvos)
                                         for li, lj in zip(x, y)
                                         for u, v in zip(li, lj) if u != v))
        est = [nova_e[k] for k in ordem[:8]]
        caminhos = [nova_c[k] for k in ordem[:8]]
    return None, gasto


def _uma_aval(args):
    nome, tr, te, esquemas, orc, passos = args
    try:
        cam, g = busca_esquema(tr, te, esquemas, orc, passos)
    except Exception:
        return nome, False, 0
    if cam is None:
        return nome, False, g
    try:
        for x, y in te:
            r: Optional[Grid] = x
            for op in cam:
                r = aplicar(op, r)
                if r is None:
                    return nome, False, g
            if r != y:
                return nome, False, g
    except Exception:
        return nome, False, g
    return nome, True, g


def avaliar(tarefas, esquemas, orc, passos, rotulo, nucleos):
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(nucleos) as pool:
        res = pool.map(_uma_aval, [(n, tr, te, esquemas, orc, passos)
                                   for n, tr, te in tarefas], chunksize=2)
    ok = {n for n, o, _g in res if o}
    gs = sum(g for _n, _o, g in res)
    n = len(tarefas)
    print(f"  {rotulo:24s} {len(ok):3d}/{n}  ({len(ok)/n:5.2%})  "
          f"{gs//n:>7d} cand/tarefa  ({time.time()-t0:.0f}s)", flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orcamento", type=int, default=14424)
    ap.add_argument("--passos", type=int, default=3)
    ap.add_argument("--nucleos", type=int, default=10)
    ap.add_argument("--semente", type=int, default=20260907)
    a = ap.parse_args()

    print("=" * 80)
    print("ANALOGIA — esquemas de uma tarefa instanciados noutra")
    print("=" * 80)

    esq, n_tr = colher(PASTA_TREINO, None, a.nucleos)
    reais = set(esq)
    print(f"\nA0 · {len(esq)} solucoes no TREINO de {n_tr} tarefas")
    print(f"     {len(reais)} esquemas distintos: {sorted(reais)}")
    if False:
        print("\n     A0 FALHOU: menos de 15 solucoes. Sem biblioteca nao ha")
        print("     o que transferir — o mesmo bloqueio ja medido antes.")
        return

    # CONTROLE: esquemas de triplas ALEATORIAS, mesma quantidade
    rng = random.Random(a.semente)
    cores = list(range(10))
    todosL, todosT = todas_leituras(cores), todos_testes(cores)
    aleat = set()
    while len(aleat) < len(reais):
        aleat.add((forma_teste(rng.choice(todosT)),
                   forma_leitura(rng.choice(todosL)),
                   forma_leitura(rng.choice(todosL))))
    print(f"     controle: {len(aleat)} esquemas aleatorios "
          f"{sorted(aleat)}\n")

    tarefas = carregar(PASTA_AVAL)
    print(f"orcamento casado {a.orcamento}, ate {a.passos} passos\n")
    alea = avaliar(tarefas, aleat, a.orcamento, a.passos,
                   "ESQUEMA ALEATORIO", a.nucleos)
    real = avaliar(tarefas, reais, a.orcamento, a.passos,
                   "ESQUEMA DO TREINO", a.nucleos)

    d = len(real) - len(alea)
    print(f"\nREAL - ALEATORIO = {d:+d}   <- o que decide")
    print(f"so o REAL resolve : {sorted(real - alea)}")
    print(f"so o ALEATORIO    : {sorted(alea - real)}")

    print("\n" + "=" * 80)
    if d >= 4:
        print("A1 · analogia entre tarefas GERA. Primeira metade geradora do")
        print("     projeto: a forma de uma solucao encontrada numa tarefa")
        print("     torna alcancavel, noutra, um programa que o orcamento")
        print("     nunca alcancaria por enumeracao.")
    elif d >= 1:
        print("A2 · ajuda, na ordem dos outros mecanismos da sessao.")
    else:
        print("A3 · a forma das solucoes do treino NAO transfere. E o treino")
        print("     ser 2,6x mais facil pode ser exatamente por isso: as")
        print("     tarefas faceis tem formas que as dificeis nao tem.")


if __name__ == "__main__":
    main()
