"""
novaphys.acumula — resolver a tarefa A torna a tarefa B mais facil?
====================================================================

O PORTAO DE TUDO QUE VEM DEPOIS
---------------------------------
Curriculo pressupoe ACUMULACAO. Aprender no facil e levar para o dificil so
significa alguma coisa se o que foi aprendido FICA.

E o nosso recipiente parece vazar: a biblioteca de esquemas tem reuso 1,42 —
quase um esquema novo por solucao. Se cada tarefa traz estrutura propria e nao
compartilhada, nao ha o que acumular e curriculo nenhum ajuda.

Nunca medimos isso. Medimos TRANSFERENCIA EM BLOCO (colher 33 esquemas do
treino inteiro, aplicar na avaliacao: 4 -> 7). Nunca medimos CUMULATIVO:
resolver em sequencia, cada solucao alimentando a biblioteca, e ver se as
tarefas do fim se beneficiam das do comeco.

O DESENHO, e ele evita o controle degenerado
----------------------------------------------
O controle obvio seria "sem acumulacao", mas com biblioteca vazia o motor de
esquema nao tem o que instanciar e resolve ZERO. Comparar contra zero nao mede
nada.

Entao a medida e INTERNA a curva: a biblioteca comeca vazia e cresce com cada
solucao. Se houver acumulacao, a taxa de acerto no ULTIMO quartil supera a do
PRIMEIRO — porque as ultimas tarefas enfrentam uma biblioteca maior.

E o controle para "as ultimas eram mais faceis por acaso" e a ORDEM: repete-se
com varias ordens aleatorias. Se o efeito for real, aparece em todas.

E O TESTE DE CURRICULO PROPRIAMENTE DITO
------------------------------------------
Alem das ordens aleatorias, roda-se uma ordem FACIL-PRIMEIRO (por area das
grades, que e proxy neutro de dificuldade e nao usa o resultado).

    se facil-primeiro > aleatorio   -> curriculo importa, a ORDEM ensina
    se iguais                       -> ha acumulacao, mas nao curriculo
    se nenhum acumula               -> fecha a linha inteira

Essa separacao importa: acumulacao e curriculo sao coisas diferentes e a
sessao vem tratando como se fossem uma.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  A0  PRE-CONDICAO. A biblioteca tem que crescer: pelo menos 5 esquemas
      colhidos ao longo da passada. Se nao crescer, nao ha o que medir.

  A1  Q4 > Q1 em TODAS as ordens aleatorias
      -> acumulacao existe
  A2  facil-primeiro bate a media das aleatorias por >= 2 tarefas
      -> curriculo importa alem da acumulacao
  A3  Q4 ~ Q1
      -> nao acumula; curriculo infantil nao vai funcionar por falta de
         recipiente, e nao por ser ma ideia

MINHA CALIBRAGEM: dezessete previsoes na sessao, SEIS certas, as onze erradas
todas otimistas.

PREVISAO: A3. O reuso de 1,42 e o argumento aritmetico: se cada solucao traz um
esquema quase exclusivo, a biblioteca cresce em TAMANHO e nao em COBERTURA, e
tarefa nova quase nunca cai num esquema ja visto. Espero Q4 ~ Q1 e espero
facil-primeiro empatar com aleatorio.

Se der A3, a leitura e util e fecha uma porta: o problema do curriculo nao e
pedagogico, e de RETENCAO. Antes de graduar o ensino, e preciso um mecanismo
que faca a solucao de uma tarefa valer para outra — e o unico que temos rende
1,42.
"""

from __future__ import annotations

import argparse
import random
import time
from typing import Dict, List, Optional, Set, Tuple

from novaphys.analogia import busca_esquema, esquema
from novaphys.arc import carregar, dims
from novaphys.eter import PASTA_AVAL
from novaphys.eter_inferencia import todas_leituras, todos_testes
from novaphys.ordem1 import aplicar, inferir


def resolver_livre(tr, te):
    """Deducao direta, sem biblioteca. E de onde vem o esquema novo."""
    cores = sorted({c for a, b in tr for l in (a + b) for c in l})
    op, _g = inferir(tr, todas_leituras(cores), todos_testes(cores))
    if op is None:
        return None
    try:
        if all(aplicar(op, x) == y for x, y in te):
            return esquema(op)
    except Exception:
        return None
    return None


def resolver_com(tr, te, lib, orcamento, passos):
    """Busca restrita aos esquemas da biblioteca atual."""
    if not lib:
        return None
    try:
        cam, _g = busca_esquema(tr, te, lib, orcamento, passos)
    except Exception:
        return None
    if cam is None:
        return None
    try:
        for x, y in te:
            r = x
            for op in cam:
                r = aplicar(op, r)
                if r is None:
                    return None
            if r != y:
                return None
    except Exception:
        return None
    return esquema(cam[0])


def _uma(args):
    """Uma tarefa contra a biblioteca CONGELADA daquele instante."""
    nome, tr, te, lib, orc, passos = args
    e = resolver_livre(tr, te)
    if e is not None:
        return nome, True, e, "livre"
    e = resolver_com(tr, te, lib, orc, passos)
    if e is not None:
        return nome, True, e, "biblioteca"
    return nome, False, None, None


def passada(tarefas, orc, passos, nucleos, rotulo):
    """
    Percorre em sequencia. A biblioteca comeca VAZIA e cresce a cada solucao.

    O paralelismo e por BLOCO: as tarefas de um bloco veem a mesma biblioteca,
    e ela so cresce entre blocos. Isso preserva a causalidade (nenhuma tarefa
    ve um esquema colhido depois dela) e mantem o custo tratavel.
    """
    from multiprocessing import Pool
    lib: Set = set()
    quart = [0, 0, 0, 0]
    n = len(tarefas)
    bloco = 20
    resolvidas, por_lib = 0, 0
    t0 = time.time()
    with Pool(nucleos) as pool:
        for ini in range(0, n, bloco):
            sub = tarefas[ini:ini + bloco]
            congelada = frozenset(lib)
            res = pool.map(_uma, [(nm, tr, te, congelada, orc, passos)
                                  for nm, tr, te in sub], chunksize=2)
            for k, (nm, ok, e, via) in enumerate(res):
                if ok:
                    resolvidas += 1
                    quart[min(3, (ini + k) * 4 // n)] += 1
                    if via == "biblioteca":
                        por_lib += 1
                    if e is not None:
                        lib.add(e)
    print(f"  {rotulo:18s} total {resolvidas:3d}  quartis {quart}  "
          f"biblioteca {len(lib):2d} esquemas  via-lib {por_lib:2d}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    return resolvidas, quart, len(lib), por_lib


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orcamento", type=int, default=14424)
    ap.add_argument("--passos", type=int, default=2)
    ap.add_argument("--nucleos", type=int, default=8)
    ap.add_argument("--sementes", type=int, default=3)
    a = ap.parse_args()

    print("=" * 84)
    print("ACUMULA — resolver a tarefa A torna a tarefa B mais facil?")
    print("=" * 84)
    base = carregar(PASTA_AVAL)
    print(f"\n{len(base)} tarefas, biblioteca comeca VAZIA e cresce a cada "
          f"solucao\n")

    curvas = []
    for s in range(a.sementes):
        t = list(base)
        random.Random(20260908 + s).shuffle(t)
        curvas.append(passada(t, a.orcamento, a.passos, a.nucleos,
                              f"aleatoria {s+1}"))

    # dificuldade por AREA das grades: proxy neutro, nao usa o resultado
    facil = sorted(base, key=lambda x: sum(
        dims(g)[0] * dims(g)[1] for p in x[1] for g in p))
    cur = passada(facil, a.orcamento, a.passos, a.nucleos, "facil-primeiro")

    print()
    q1 = sum(c[1][0] for c in curvas) / len(curvas)
    q4 = sum(c[1][3] for c in curvas) / len(curvas)
    med = sum(c[0] for c in curvas) / len(curvas)
    libm = sum(c[2] for c in curvas) / len(curvas)
    print(f"A0 · biblioteca media ao fim: {libm:.1f} esquemas")
    print(f"\naleatorias  Q1={q1:.1f}  Q4={q4:.1f}  total medio {med:.1f}")
    print(f"facil-1o    Q1={cur[1][0]}  Q4={cur[1][3]}  total {cur[0]}")
    sobe = all(c[1][3] > c[1][0] for c in curvas)

    print("\n" + "=" * 84)
    if libm < 5:
        print("A0 FALHOU: a biblioteca nao cresceu. Nada a medir.")
    elif sobe and q4 > q1:
        print("A1 · ACUMULA: o ultimo quartil bate o primeiro em todas as")
        print("     ordens. O que e aprendido fica e ajuda adiante.")
        if cur[0] >= med + 2:
            print("A2 · e o CURRICULO importa alem disso: ordem graduada bate")
            print("     a media das aleatorias.")
        else:
            print("     Mas curriculo NAO importa: ordem graduada empata com")
            print("     aleatoria. Ha acumulacao, nao ha curriculo.")
    else:
        print("A3 · NAO ACUMULA. Curriculo infantil nao vai funcionar por")
        print("     falta de RECIPIENTE, nao por ser ma ideia. Antes de")
        print("     graduar o ensino e preciso um mecanismo que faca a")
        print("     solucao de uma tarefa valer para outra — e o unico que")
        print("     temos rende 1,42 esquemas por solucao.")


if __name__ == "__main__":
    main()
