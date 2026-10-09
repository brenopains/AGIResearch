"""
novaphys.abstrai — anti-unificacao sobre o proprio corpus
==========================================================

DE ONDE VEM
-------------
Da literatura, e desta vez com o corpus ja pronto.

  Stitch (Top-Down Synthesis for Library Learning, POPL 2023)
      Inducao de biblioteca como MDL GULOSO sobre um corpus de programas.
      Constroi abstracoes a partir das primitivas da DSL, usando casamento
      sintatico de abstracoes intermediarias para podar. 3 a 4 ordens de
      grandeza mais rapido que o DreamCoder, com qualidade igual ou melhor.
      arxiv.org/abs/2211.16605

  babble (E-Graphs and Anti-unification)
      Mesma ideia com e-graphs e ANTI-UNIFICACAO: dado um conjunto de
      programas, achar a generalizacao mais especifica — o esqueleto comum,
      com o que difere virando buraco.

  Prospective Compression in Human Abstraction Learning
      Humanos formam abstracoes por UTILIDADE FUTURA antecipada, nao por
      compressao retrospectiva. Por isso preferem compressoes "subotimas" dos
      dados atuais que transferem melhor. arxiv.org/abs/2605.09985

POR QUE AGORA E NAO ANTES
---------------------------
`biblioteca.py` ja tentou acumular entre tarefas e chegou a TRES esquemas em
400 tarefas. O veredito na epoca: acumulacao e immensuravel a 1% de acerto —
nao havia sucesso suficiente para acumular.

Agora sao 51/400 (12,75%) com pilhas EXPLICITAS e programas inventados, e a
estrutura repetida ja aparece sozinha no log:

    15663ba9  programa + programa + copia
    903d1b4a  programa + programa + d4

Dois codecs inventados compondo entre si. Ha esqueleto para abstrair.

O QUE E A ANTI-UNIFICACAO AQUI
--------------------------------
Um programa e `PINTAR(t1 & t2 & t3) COM pintor`. Dados dois:

    PINTAR(fundo & raio(ocupada,N)) COM const(3)
    PINTAR(fundo & raio(ocupada,S)) COM const(7)
                |
                v  generalizacao mais especifica
    PINTAR(fundo & ?T)              COM const(?C)

O esqueleto `fundo & ?` com pintor `const` e a ABSTRACAO. Ela e paga UMA vez
na biblioteca; cada uso paga so os buracos.

E O GANHO NAO E SO DE BITS. Uma abstracao com dois buracos e um espaco de
busca muito menor que a linguagem inteira, e SEMEADO num ponto que ja provou
servir. Isso e a compressao prospectiva implementada: o esqueleto util no
passado e tentado primeiro no futuro. E a resposta ao "o sistema nao antecipa
utilidade futura" — ele passa a antecipar, e o criterio continua sendo bits.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  B0  PRE-CONDICAO. O corpus tem que ter pelo menos 20 programas inventados.
      Abaixo disso nao ha o que anti-unificar e a medida nao existe.

  B1  Alguma abstracao com >= 3 usos e esqueleto nao trivial (>= 1 teste fixo)?
          sim -> ha estrutura repetida no que o sistema inventou sozinho
          nao -> cada programa e um caso isolado, e biblioteca nao se aplica

  B2  compressividade: bits economizados pela biblioteca sobre o corpus
      > 0  -> a biblioteca se paga pelo criterio do Stitch

MINHA CALIBRAGEM: quarenta e cinco previsoes, DEZESSEIS certas.

PREVISAO: B1 sim, B2 positivo mas pequeno. O `programa + programa` aparecendo
duas vezes sem eu pedir e evidencia forte de esqueleto repetido. Mas 51 tarefas
e um corpus pequeno para biblioteca — o Pang acumulou 538 programas de 1.000
tarefas, uma ordem de grandeza acima. Aposto que a estrutura existe e que o
ganho em bits e modesto; o ganho que interessa e o de SEMENTE de busca, e esse
so aparece na rodada seguinte.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from collections import Counter, defaultdict
from itertools import combinations

from novaphys.arc import carregar
from novaphys.eter import PASTA_AVAL
from novaphys import descricao as D

ARQ = "biblioteca_abstracoes.json"
FONTE = "ambos"
TAR = None


PASTA_TREINO = PASTA_AVAL.replace("evaluation", "training")


def _ini():
    """
    Corpus das DUAS particoes. Colher do treino do ARC e legitimo e e o que a
    particao existe para ser; e colher da avaliacao tambem e, porque `montar`
    so olha os PARES DE TREINO de cada tarefa e nunca a saida de teste. O que
    nao pode ser tocado e a saida de teste, e ela so entra na pontuacao.
    B0 falhou com 15 programas inventados sobre 400 tarefas. Com 800 o corpus
    dobra, que e a correcao honesta de um portao que falhou por falta de dado
    e nao por falta de mecanismo.
    """
    global TAR
    if FONTE == "treino":
        TAR = carregar(PASTA_TREINO)
    elif FONTE == "aval":
        TAR = carregar(PASTA_AVAL)
    else:
        TAR = carregar(PASTA_TREINO) + carregar(PASTA_AVAL)


def _uma(ti):
    n, tr, te = TAR[ti]
    try:
        fm, pilha, err = D.montar(tr, refoco=True, feixe=1)
        if err != 0 or not pilha:
            return n, None
        progs = [m for cd, m in pilha if cd.nome == "programa"]
        return n, {"forma": fm[0],
                   "pilha": [cd.nome for cd, _m in pilha],
                   "progs": [[[list(t) for t in sel], list(pin)]
                             for sel, pin in progs]}
    except Exception:
        return n, None


def antiunifica(p1, p2):
    """
    Generalizacao mais especifica de dois programas. Devolve
    (testes fixos, quantos buracos de teste, pintor fixo ou familia), ou None
    quando os pintores sao de familias diferentes — ai nao ha esqueleto comum.
    """
    (s1, k1), (s2, k2) = p1, p2
    comum = frozenset(s1) & frozenset(s2)
    bur = max(len(s1), len(s2)) - len(comum)
    if k1 == k2:
        pin = ("fixo",) + tuple(k1)
    elif k1[0] == k2[0]:
        pin = ("familia", k1[0])
    else:
        return None
    if not comum and pin[0] == "familia":
        return None                     # esqueleto vazio: nao e abstracao
    return (comum, bur, pin)


def casa(abst, prog):
    comum, _bur, pin = abst
    sel, k = prog
    if not comum <= frozenset(sel):
        return False
    if pin[0] == "fixo":
        return tuple(k) == pin[1:]
    return k[0] == pin[1]


def main() -> None:
    global FONTE, ARQ
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=8)
    ap.add_argument("--fonte", default="ambos",
                    choices=("treino", "aval", "ambos"))
    ap.add_argument("--saida", default=ARQ)
    a = ap.parse_args()
    FONTE, ARQ = a.fonte, a.saida
    _ini()
    print(f"  fonte do corpus: {FONTE}")

    print("=" * 76)
    print("ABSTRAI — anti-unificacao sobre o corpus que o sistema gerou")
    print("=" * 76)
    t0 = time.time()
    with Pool(a.nucleos, initializer=_ini) as pool:
        res = pool.map(_uma, range(len(TAR)), chunksize=4)

    corpus = {n: d for n, d in res if d}
    progs = []
    for n, d in corpus.items():
        for sel, pin in d["progs"]:
            progs.append((tuple(tuple(t) for t in sel), tuple(pin), n))
    print(f"\n  ({time.time()-t0:.0f}s)")
    print(f"  tarefas com residuo 0 : {len(corpus)}")
    print(f"  programas inventados  : {len(progs)}")
    print(f"  B0 . corpus suficiente: {len(progs) >= 20}")
    if len(progs) < 2:
        print("\nB0 FALHOU: corpus pequeno demais para anti-unificar.")
        return

    cont = Counter()
    onde = defaultdict(set)
    for (s1, k1, n1), (s2, k2, n2) in combinations(progs, 2):
        ab = antiunifica((s1, k1), (s2, k2))
        if ab is None:
            continue
        cont[ab] += 1
        onde[ab] |= {n1, n2}

    # usos reais: quantos programas do corpus a abstracao casa
    usos = {}
    for ab in cont:
        usos[ab] = sum(1 for s, k, _n in progs if casa(ab, (s, k)))

    bons = sorted(((ab, usos[ab], len(onde[ab])) for ab in cont
                   if usos[ab] >= 2),
                  key=lambda x: (-x[1], len(x[0][0])))
    print(f"  abstracoes com >= 2 usos: {len(bons)}")
    print("")
    print("  usos  tarefas  esqueleto")
    print("  ----  -------  ---------")
    for ab, u, t in bons[:14]:
        comum, bur, pin = ab
        sk = " & ".join(sorted(str(c) for c in comum)) or "(vazio)"
        pn = ("=" + str(pin[1:])) if pin[0] == "fixo" else ("familia " + pin[1])
        print(f"  {u:4d}  {t:7d}  PINTAR({sk}{' & ?' * bur}) COM {pn}")

    # compressividade no criterio do Stitch: bits do corpus com e sem biblioteca
    def bits_prog(s, k):
        cp = (D.BITS_COR if k[0] == "const"
              else 5.0 if k[0] in ("prop", "prox", "esp", "rep") else 3.0)
        return sum(D._bits_teste(c) for c in s) + cp + 2
    sem = sum(bits_prog(s, k) for s, k, _n in progs)
    lib, com = [], 0.0
    usados = set()
    for ab, u, _t in bons:
        comum, bur, pin = ab
        if u < 2:
            continue
        custo_lib = sum(D._bits_teste(c) for c in comum) + 6
        econ = 0.0
        alvo = [i for i, (s, k, _n) in enumerate(progs)
                if i not in usados and casa(ab, (s, k))]
        if len(alvo) < 2:
            continue
        for i in alvo:
            s, k, _n = progs[i]
            econ += sum(D._bits_teste(c) for c in comum)
        if econ - custo_lib > 0:
            lib.append((ab, len(alvo)))
            com += custo_lib
            usados |= set(alvo)
    ganho = sum(sum(D._bits_teste(c) for c in ab[0]) * n
                for ab, n in lib) - com
    print("")
    print(f"  bits do corpus sem biblioteca : {sem:9.1f}")
    print(f"  abstracoes que se pagam       : {len(lib)}")
    print(f"  B2 . bits economizados        : {ganho:+9.1f}")

    saida = [{"comum": [list(c) for c in sorted(ab[0], key=repr)],
              "buracos": ab[1], "pintor": list(ab[2]), "usos": n}
             for ab, n in lib]
    io.open(ARQ, "w", encoding="utf-8").write(
        json.dumps(saida, indent=1, ensure_ascii=False))
    print(f"  biblioteca gravada em {ARQ}")

    print("")
    print("=" * 76)
    naotriv = [b for b in bons if b[1] >= 3 and len(b[0][0]) >= 1]
    if naotriv:
        print(f"B1 . ha estrutura repetida: {len(naotriv)} abstracoes com >= 3")
        print("     usos e esqueleto nao trivial.")
    else:
        print("B1 FALHOU . cada programa e um caso isolado; biblioteca nao se")
        print("     aplica neste corpus.")
    print(f"B2 . {'a biblioteca se paga' if ganho > 0 else 'a biblioteca NAO se paga'}"
          f" pelo criterio do Stitch.")


if __name__ == "__main__":
    main()
