"""
novaphys.poda — descartar dimensao antes de procurar
======================================================

DE ONDE VEM
-------------
Da gravacao do usuario resolvendo ARC em voz alta, aos 02:04:

    "a ordem e completamente irrelevante aqui"

Ele DECIDIU que um eixo do dado nao carrega informacao e parou de olhar. A
partir dali o espaco de busca dele colapsou e ele resolveu a parte principal
da tarefa em segundos.

Nossos motores nunca fazem isso. Eles carregam as dez propriedades ate o fim,
testam uma a uma e ficam com a primeira consistente.

POR QUE ISSO DEVERIA FUNCIONAR, E A PREVISAO E DERIVADA DE MEDIDA
-------------------------------------------------------------------
A sessao mediu, em dois pontos independentes, que PRECISAO E FUNCAO DA
LIBERDADE DA BUSCA:

    deducao exata   1 resposta determinada        precisao 100%
    multi-lente     80 familias                   precisao   4,7%
    ladrilho        48 hipoteses exclusivas       precisao 100%

Se precisao cai com liberdade, entao reduzir o espaco de forma PRINCIPIADA —
tirando o que comprovadamente nao informa — tem que subir precisao sem custar
cobertura. Essa e a previsao e ela e falsificavel.

O QUE E PODADO, e o criterio nao pode ser "o que da o melhor numero"
----------------------------------------------------------------------
Duas podas, ambas lidas dos dados de treino e nenhuma olhando o teste:

  1. CONSTANTE. Se a propriedade tem o mesmo valor em TODOS os objetos de
     TODOS os pares, ela nao distingue nada. Uma regra baseada nela e
     indistinguivel de `global`, e so ocupa espaco.

  2. REDUNDANTE. Se duas propriedades induzem a MESMA PARTICAO dos objetos,
     elas dizem a mesma coisa. Fica a mais simples pela ordem declarada, a
     outra sai. E o mesmo principio do quociente por equivalencia
     observacional, que ja cortou 2,06x de trabalho na busca monolitica.

Nenhuma das duas usa o resultado para decidir o que podar. Podar pelo que da
o melhor placar seria escolher a hipotese pelo teste, que e o vazamento que
`propositor.py` cometeu e custou uma conclusao inteira.

O QUE SE MEDE
---------------
    hipoteses efetivas   antes e depois da poda (o tamanho do espaco)
    ajustou o treino     quantas acham alguma regra
    passou no teste
    PRECISAO             teste/ajustou, antes e depois

Se a poda cortar espaco E subir precisao E manter cobertura, o mecanismo esta
validado. Se cortar cobertura, ela esta jogando fora informacao real.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  P0  PRE-CONDICAO. As 4 tarefas de `ockham.py` precisam sobreviver. Podar so
      pode TIRAR hipotese que nao informa; se perder tarefa, o criterio de
      poda esta errado.

  P1  precisao sobe E cobertura se mantem
      -> descartar dimensao e o mecanismo, e a introspeccao humana entregou o
         segundo mecanismo util da sessao
  P2  espaco cai mas precisao nao sobe
      -> a poda economiza trabalho e nao melhora escolha; e um ganho de
         eficiencia, nao de qualidade
  P3  cobertura cai
      -> as propriedades "constantes" ou "redundantes" carregavam informacao
         que eu nao vi

MINHA CALIBRAGEM: quinze previsoes na sessao, SEIS certas, as nove erradas
todas otimistas.

PREVISAO: P2. Espero o espaco cair bastante (varias propriedades sao
redundantes entre si — `tam`, `rank_tam` e `e_maior` induzem particoes
parecidas em cena com poucos objetos) e espero a precisao NAO subir, porque
`ockham` ja esta em 100% e nao ha para onde subir. O ganho visivel deve ser
tempo.

E registro o que tornaria isto interessante apesar disso: se a poda permitir
AFROUXAR o portao imunologico sem perder precisao. Espaco menor precisa de
menos validacao — foi exatamente o que `ladrilho.py` mostrou com 48 hipoteses
e precisao 100% SEM portao. Se a poda derrubar o espaco o bastante, tarefas de
2 pares (hoje rejeitadas por construcao) voltam a ser alcancaveis.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from novaphys.arc import carregar, dims
from novaphys.eter import PASTA_AVAL
from novaphys.ockham import ORDEM_LENTE, ORDEM_PROP
from novaphys.segmenta import LENTES, PROPS, aplicar, deduzir, fundo_de, props


def podar(treino, lente) -> Tuple[List[str], List[str]]:
    """
    Devolve (propriedades que sobram, motivos da poda). Le so o treino.
    """
    valores: Dict[str, List[Any]] = {p: [] for p in PROPS}
    particoes: Dict[str, List[Tuple]] = {p: [] for p in PROPS}
    for a, _b in treino:
        bg = fundo_de(a)
        c = lente(a, bg)
        if not c.objs:
            return list(PROPS), []
        pr = props(c)
        ids = sorted(pr)
        for p in PROPS:
            vs = [pr[i][p] for i in ids]
            valores[p].extend(vs)
            # a PARTICAO que a propriedade induz, sem depender dos rotulos
            grupos: Dict[Any, int] = {}
            assin = []
            for v in vs:
                if v not in grupos:
                    grupos[v] = len(grupos)
                assin.append(grupos[v])
            particoes[p].append(tuple(assin))

    fora, motivos = [], []
    vistas: Dict[Tuple, str] = {}
    for p in PROPS:
        if len(set(valores[p])) <= 1:
            motivos.append(f"{p}: constante")
            continue
        chave = tuple(particoes[p])
        if chave in vistas:
            motivos.append(f"{p}: mesma particao que {vistas[chave]}")
            continue
        vistas[chave] = p
        fora.append(p)
    return fora, motivos


def deduzir_podado(treino, lente, permitidas):
    """
    Igual a `segmenta.deduzir`, mas so considera as propriedades que
    sobreviveram a poda. Nada mais muda.
    """
    r = deduzir(treino, lente)
    if r is None:
        return None
    if r[0] not in permitidas and r[0] != "global":
        return None
    return r


def imune(treino, lente, permitidas, minimo: int) -> bool:
    if len(treino) < minimo:
        return False
    for i in range(len(treino)):
        resto = treino[:i] + treino[i + 1:]
        r = deduzir_podado(resto, lente, permitidas)
        if r is None:
            return False
        try:
            if aplicar(r, treino[i][0], lente) != treino[i][1]:
                return False
        except Exception:
            return False
    return True


def _uma(args):
    nome, tr, te, usar_poda, minimo = args
    cands, ajustou, espaco = [], False, 0
    for ln, lente in LENTES:
        try:
            if usar_poda:
                perm, _m = podar(tr, lente)
            else:
                perm = list(PROPS)
            espaco += len(perm)
            r = deduzir_podado(tr, lente, perm)
            if r is None:
                continue
            if any(aplicar(r, a, lente) != b for a, b in tr):
                continue
            ajustou = True
            if not imune(tr, lente, perm, minimo):
                continue
            cands.append((ORDEM_LENTE[ln], ORDEM_PROP.get(r[0], 99), ln, r,
                          lente))
        except Exception:
            continue
    if not cands:
        return nome, ajustou, False, None, espaco
    cands.sort(key=lambda z: (z[0], z[1]))
    _a, _b, ln, r, lente = cands[0]
    try:
        ok = all(aplicar(r, x, lente) == y for x, y in te)
    except Exception:
        ok = False
    return nome, ajustou, True, (ln if ok else None), espaco


def rodar(tarefas, usar_poda, minimo, rotulo, nucleos):
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(nucleos) as pool:
        res = pool.map(_uma, [(n, tr, te, usar_poda, minimo)
                              for n, tr, te in tarefas], chunksize=4)
    ajust = [n for n, aj, _v, _o, _e in res if aj]
    sobrev = [n for n, _aj, v, _o, _e in res if v]
    ok = {n for n, _aj, _v, o, _e in res if o}
    esp = sum(e for _n, _aj, _v, _o, e in res) / max(len(res), 1)
    print(f"  {rotulo:22s} espaco {esp:5.1f}  ajustou {len(ajust):3d}  "
          f"imune {len(sobrev):3d}  teste {len(ok):3d}  "
          f"precisao {len(ok)/max(len(ajust),1):6.1%}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    return ok, len(ajust), esp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=8)
    a = ap.parse_args()

    print("=" * 84)
    print("PODA — descartar dimensao antes de procurar")
    print("=" * 84)
    tarefas = carregar(PASTA_AVAL)
    print("\nexemplo do que e podado, na primeira tarefa:")
    _n, tr, _te = tarefas[0]
    perm, mot = podar(tr, dict(LENTES)["cc4cor"])
    print(f"  sobram {len(perm)} de {len(PROPS)}: {perm}")
    for m in mot[:6]:
        print(f"    podada  {m}")

    print()
    sem, a_sem, e_sem = rodar(tarefas, False, 3, "SEM poda", a.nucleos)
    com, a_com, e_com = rodar(tarefas, True, 3, "COM poda", a.nucleos)
    # espaco menor precisa de menos validacao: com poda, aceitar 2 pares
    com2, a_com2, _e = rodar(tarefas, True, 2, "COM poda, min 2 pares",
                             a.nucleos)

    antes = {"37d3e8b2", "72a961c9", "b1fc8b8e", "f45f5ca7"}
    print(f"\nP0 · das 4 de `ockham`, mantidas COM poda: "
          f"{sorted(antes & com)}")
    print(f"     perdidas: {sorted(antes - com)}")
    print(f"\nespaco medio por tarefa: {e_sem:.1f} -> {e_com:.1f} "
          f"({e_sem/max(e_com,1e-9):.2f}x menor)")
    print(f"tarefas: sem poda {len(sem)}, com poda {len(com)}, "
          f"com poda e 2 pares {len(com2)}")
    print(f"so a poda resolve : {sorted(com - sem)}")
    print(f"so o min-2 resolve: {sorted(com2 - com)}")

    p_sem = len(sem) / max(a_sem, 1)
    p_com = len(com) / max(a_com, 1)
    print("\n" + "=" * 84)
    if not antes <= com:
        print("P3 · a poda PERDEU tarefa: as propriedades cortadas carregavam")
        print("     informacao que eu nao vi.")
    elif p_com > p_sem + 1e-9 and len(com) >= len(sem):
        print("P1 · descartar dimensao e o mecanismo. A introspeccao humana")
        print("     entregou o segundo mecanismo util da sessao.")
    elif e_com < e_sem:
        print("P2 · a poda economiza trabalho e nao melhora a escolha. Ganho")
        print("     de eficiencia, nao de qualidade.")
    else:
        print("P2 · sem efeito mensuravel.")
    if len(com2) > len(com):
        print(f"\n     E o que interessa: com o espaco menor, o portao pode")
        print(f"     ser afrouxado para 2 pares sem perder precisao, e isso")
        print(f"     recupera {len(com2)-len(com)} tarefa(s) antes inalcancavel")
        print("     por construcao. Espaco menor precisa de menos validacao —")
        print("     o mesmo que `ladrilho.py` mostrou com 48 hipoteses.")


if __name__ == "__main__":
    main()
