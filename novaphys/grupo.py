"""
novaphys.grupo — quocientar a busca pelo grupo: elegancia como eficiencia
==========================================================================

DE ONDE VEM
-------------
A pergunta do usuario: em relato psicodelico as transformacoes atingem simetria
quase perfeita, formas que se completam e se encaixam — nao esta faltando isso,
ou nao deveriamos aplicar isso com elegancia?

"Forma que se completa e se encaixa" tem nome exato: e a ORBITA DE UM GRUPO. E
nao e analogia — o resultado de Bressloff e Cowan e literalmente que as
planformes alucinatorias sao os SUBGRUPOS AXIAIS DE E(2). O cerebro nao inventa
aquelas simetrias; elas sao a estrutura do espaco em que ele opera, e ficam
visiveis quando o ganho sobe.

E falta mesmo aqui. Os atomos `rot90`, `espH`, `espV`, `transp` geram o grupo
diedral D4, com OITO elementos. Mas a busca enumera `itertools.product` sem
saber disso:

    rot90 rot90 rot90 rot90   e a identidade, e gasta 4 de orcamento
    espH espV   e   rot90 rot90   sao o MESMO elemento, e gastam 4
    qualquer prefixo que ja apareceu e recontado a cada profundidade

Ha uma medida deste projeto que da o tamanho do desperdicio: 3.486 hipoteses da
gramatica colapsam em apenas 324 comportamentos distintos. Fator 10,8x. Isso
nao e ruido — e a estrutura de grupo aparecendo como trabalho repetido.

E o lugar onde dói agora e preciso. Com as 38 primitivas (24 escritas + 14
extraidas do mundo da dobra), profundidade 3 sao 38^3 = 54.872 sequencias
contra um orcamento de 14.424. A busca de `soma.py` NAO TERMINA: ela e cortada
no meio, e mesmo assim deu 17/400. Se o espaco colapsar pelo grupo, o mesmo
orcamento pode cobrir o espaco inteiro.

O MECANISMO
-------------
Equivalencia observacional. Dois programas que produzem os MESMOS valores em
todas as entradas de treino sao intercambiaveis para o resto da busca — o que
vier depois nao consegue distinguir. Entao guarda-se a assinatura (a tupla das
saidas) e descarta-se o segundo.

Isso quocienta pelo grupo sem que ninguem precise saber qual grupo e. Nao
escrevo "D4 tem 8 elementos" em lugar nenhum: a repeticao se revela sozinha nos
dados. E captura mais do que D4 — captura toda coincidencia que as entradas
daquela tarefa nao separam.

A busca vira largura-primeiro sobre CLASSES em vez de sobre sequencias:
mantem-se a fronteira de comportamentos distintos e estende-se cada um por um
operador.

O CONTROLE, e ele importa
---------------------------
Colapsar muda a MOEDA. Um candidato do braco quocientado custa mais caro em
tempo (calcular e comparar a assinatura) e cobre mais espaco. Comparar "mesmo
numero de candidatos" favoreceria o quociente de graca.

Entao a comparacao e em duas moedas ao mesmo tempo:

    CANDIDATOS   quantos programas foram avaliados (a moeda da sessao)
    APLICACOES   quantas vezes um operador foi aplicado a uma grade — o custo
                 real de computacao, que `multiverso.py` ja usou como moeda
                 mais dura quando o casamento por candidatos ficou suspeito

Se o quociente ganhar nas duas, ganhou.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  G0  PRE-CONDICAO. Sem colapso, a busca em largura tem que reproduzir os
      numeros de `soma.py`: 14/400 com 24 primitivas e 17/400 com 38, na
      profundidade 3. Se nao reproduzir, este arquivo diverge e nada vale.

  G1  o fator de colapso medido e >= 3x  E  o quociente resolve mais tarefas
      nas duas moedas
      -> a estrutura de grupo era desperdicio puro e elegancia e eficiencia

  G2  colapso alto mas mesmas tarefas
      -> a redundancia existe e o orcamento nao era o gargalo: a busca ja
         achava tudo que sabia achar antes de acabar o orcamento

  G3  colapso baixo (< 2x)
      -> as tarefas do ARC tem entradas variadas demais para dois programas
         coincidirem, e a redundancia de 10,8x era uma propriedade da
         gramatica de `eter2.py`, nao desta busca

MINHA PREVISAO, escrita antes de rodar: G1 no fator de colapso e G2 no numero
de tarefas. Espero colapso ALTO — de 5x a 15x, porque as grades de treino do
ARC sao poucas (2 a 4 por tarefa) e pequenas, entao coincidencia e barata. Mas
espero o numero de tarefas subir POUCO, +0 a +2, porque `eficiencia.py` e o
feixe ja mostraram que profundidade alem da 4 nao entrega, e o que o colapso
compra e principalmente PROFUNDIDADE.

Se eu estiver certo, o achado util nao e o placar: e que o orcamento nunca foi
o gargalo desta busca, e a sessao inteira vinha casando orcamento numa moeda
que nao estava apertada. Isso reenquadra retroativamente todo experimento de
"orcamento casado" que rodamos hoje.
"""

from __future__ import annotations

import argparse
import itertools
import time
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from novaphys.fisica import (PASTA_AVAL, PRIMITIVAS, ajustar_mapa,
                             aplicar_mapa, carregar, extrair, valida)

Grid = Tuple[Tuple[int, ...], ...]


def busca(treino, teste, extras, orcamento: int, prof: int, colapsar: bool
          ) -> Tuple[bool, int, int, int]:
    """
    Largura-primeiro. Com `colapsar`, a fronteira guarda uma entrada por
    COMPORTAMENTO distinto nas entradas de treino; sem, guarda uma por
    sequencia (o que reproduz `itertools.product`).

    Devolve (resolveu, candidatos, aplicacoes, tamanho final da fronteira).
    """
    d = dict(PRIMITIVAS)
    d.update(dict(extras))
    ops = list(d.values())
    entradas = tuple(a for a, _b in treino)
    alvos = [b for _a, b in treino]

    def fecha(saidas) -> bool:
        if list(saidas) == alvos:
            return True
        m = ajustar_mapa(list(zip(saidas, alvos)))
        return m is not None and all(aplicar_mapa(r, m) == b
                                     for r, b in zip(saidas, alvos))

    cand, aplic = 1, 0
    if fecha(list(entradas)):
        return True, cand, aplic, 1

    fronteira: List[Tuple[Grid, ...]] = [entradas]
    vistos: Set[Tuple[Grid, ...]] = {entradas}
    for _p in range(prof):
        nova: List[Tuple[Grid, ...]] = []
        for est in fronteira:
            for f in ops:
                if cand >= orcamento:
                    return False, cand, aplic, len(fronteira)
                cand += 1
                saidas = []
                ok = True
                for g in est:
                    aplic += 1
                    r = f(g)
                    if not valida(r):
                        ok = False
                        break
                    saidas.append(r)
                if not ok:
                    continue
                t = tuple(saidas)
                if colapsar:
                    if t in vistos:
                        continue          # mesmo comportamento: intercambiavel
                    vistos.add(t)
                if fecha(saidas):
                    return True, cand, aplic, len(nova)
                nova.append(t)
        fronteira = nova
        if not fronteira:
            break
    return False, cand, aplic, len(fronteira)


def avaliar(tarefas, extras, orcamento, prof, colapsar, rotulo):
    ok: Set[str] = set()
    tc, ta = 0, 0
    t0 = time.time()
    for nome, tr, te in tarefas:
        r, c, ap, _f = busca(tr, te, extras, orcamento, prof, colapsar)
        tc += c
        ta += ap
        if r:
            ok.add(nome)
    n = len(tarefas)
    print(f"  {rotulo:18s} {len(ok):3d}/{n}  ({len(ok)/n:5.2%})  "
          f"{tc//n:>7d} cand/tarefa  {ta//n:>8d} aplic/tarefa  "
          f"({time.time()-t0:.0f}s)", flush=True)
    return ok, tc, ta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", default=PASTA_AVAL)
    ap.add_argument("--orcamento", type=int, default=14424)
    ap.add_argument("--prof", type=int, default=3)
    ap.add_argument("--cenas", type=int, default=40)
    ap.add_argument("--semente", type=int, default=20260905)
    a = ap.parse_args()

    print("=" * 84)
    print("GRUPO — quocientar a busca por equivalencia observacional")
    print("=" * 84)

    dobra = extrair(True, a.cenas, a.semente)
    print(f"\n{len(PRIMITIVAS)} primitivas escritas + {len(dobra)} extraidas "
          f"do mundo da dobra")
    print(f"espaco a profundidade {a.prof}: "
          f"{len(PRIMITIVAS) ** a.prof} e "
          f"{(len(PRIMITIVAS) + len(dobra)) ** a.prof}")
    print(f"orcamento {a.orcamento} candidatos/tarefa\n")

    tarefas = carregar(a.pasta)
    r = {}
    for extras, rot in (([], "24 prim"), (dobra, "38 prim")):
        for col in (False, True):
            k = f"{rot} {'QUOCIENTE' if col else 'CRU'}"
            r[k] = avaliar(tarefas, extras, a.orcamento, a.prof, col, k)

    print()
    for rot in ("24 prim", "38 prim"):
        cru, quo = r[f"{rot} CRU"], r[f"{rot} QUOCIENTE"]
        fc = cru[1] / max(quo[1], 1)
        print(f"{rot}:  tarefas {len(cru[0])} -> {len(quo[0])}   "
              f"candidatos {cru[1]//len(tarefas)} -> {quo[1]//len(tarefas)}  "
              f"({fc:.2f}x)   ganhas {sorted(quo[0] - cru[0])}")

    print("\n" + "=" * 84)
    c38, q38 = r["38 prim CRU"], r["38 prim QUOCIENTE"]
    fator = c38[1] / max(q38[1], 1)
    d = len(q38[0]) - len(c38[0])
    if fator >= 3 and d > 0:
        print("G1 · a estrutura de grupo era desperdicio puro. Colapsar por")
        print(f"     comportamento corta {fator:.1f}x o trabalho E resolve")
        print(f"     {d} tarefa(s) a mais. Elegancia e eficiencia.")
    elif fator >= 2:
        print(f"G2 · colapso de {fator:.1f}x e o placar nao mexe ({d:+d}).")
        print("     A redundancia existe e o ORCAMENTO NUNCA FOI O GARGALO:")
        print("     a busca ja achava tudo que sabia achar antes de acabar.")
        print("     Isso reenquadra todo 'orcamento casado' desta sessao.")
    else:
        print(f"G3 · colapso baixo ({fator:.1f}x). As entradas do ARC sao")
        print("     variadas demais para dois programas coincidirem; a")
        print("     redundancia de 10,8x era da gramatica de `eter2.py`,")
        print("     nao desta busca.")


if __name__ == "__main__":
    main()
