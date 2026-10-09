"""
novaphys.transducao — prever sem extrair regra
===============================================

DE ONDE VEM
-------------
`Combining Induction and Transduction for Abstract Reasoning`
(Li, Ellis et al., arxiv.org/abs/2411.02272) mede que INDUZIR o programa e
PREVER a saida direto resolvem conjuntos DISJUNTOS, e que a divisao e estavel
entre sementes. Inducao vence em computacao precisa e composicao; transducao
vence em conceito perceptual difuso.

E o survey do campo (arxiv.org/html/2603.13372v1) poe numero:

    inducao pura <= 79%   transducao pura ~40%   HIBRIDOS 80-94%

Este projeto e 100% inducao. Toda a arquitetura — MDL, pilhas, linguagem de
codecs, biblioteca — extrai uma REGRA e depois aplica. Nunca prevemos direto.

O QUE E TRANSDUCAO AQUI, SEM PRE-TREINO E SEM REDE
----------------------------------------------------
Nao ha regra, nao ha programa, nao ha bits de modelo. So o par (entrada, saida)
do treino e a entrada de teste.

    cada CELULA vira um vetor de features sobre os ~250 testes da gramatica
    (`fundo`, `cor 3`, `viz8=2`, `raio(ocupada,N)`, `desl(N,ocupada)`, ...)

    dos pares de treino, monta-se o mapa   assinatura -> cor de saida
    para uma celula de teste: casamento EXATO se existir e for unanime,
    senao o vizinho mais proximo por diferenca simetrica das assinaturas

E transducao no sentido estrito do paper: a predicao de cada celula sai de
celulas parecidas ja vistas, nao de uma funcao latente inferida. Nenhuma regra
e escrita, e nada e reutilizavel noutra tarefa.

E a gramatica de testes, construida para a INDUCAO, e o que torna isto possivel
de graca — a mesma representacao servindo os dois modos de inferencia. Se os
conjuntos forem disjuntos como o paper preve, ensemblar e ganho de graca.

O CONTROLE DA MEMORIZACAO
---------------------------
Transducao decora por construcao: com features suficientes, cada celula tem
assinatura unica e o mapa vira uma copia da saida. Isso ajusta o treino
perfeitamente e nao vale nada — foi o que matou o `segmenta.py` (127
ajustavam, 6 passavam).

Por isso o portao e o imunologico do projeto, e nao a reproducao do treino:
LEAVE-ONE-OUT sobre os pares. O mapa e montado com n-1 pares e tem que
reproduzir o par retirado EXATAMENTE. Uma tabela decorada falha nisso sempre.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  T0  PRE-CONDICAO. Tem que resolver pelo menos 1 tarefa. Zero significa que a
      transducao nao e viavel nesta representacao, e nao que ela e ruim.

  T1  DISJUNCAO (a hipotese do paper). Das que a transducao resolve, quantas a
      inducao NAO resolve?
          >= metade -> disjunto como o paper preve; ensemblar e ganho de graca
          < metade  -> a transducao so repete o que a inducao ja fazia

  T2  quantas INEDITAS entram na uniao do projeto

MINHA CALIBRAGEM: quarenta e oito previsoes, DEZESSETE certas, as trinta e uma
erradas quase todas otimistas.

PREVISAO: T0 passa, T1 disjunto, T2 baixo (1 a 3). O paper mediu disjuncao com
modelos NEURAIS treinados em muito dado; a nossa transducao e um vizinho mais
proximo sobre features feitas a mao, muito mais fraca. Mas o MODO de inferencia
e genuinamente outro, e disjuncao e uma afirmacao sobre o modo, nao sobre a
potencia. Aposto que resolve pouco e que quase tudo que resolve e inedito.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from typing import Dict, FrozenSet, List, Optional, Tuple

from novaphys.arc import Grid, carregar, dims
from novaphys.compositor import fundo
from novaphys.eter import PASTA_AVAL
from novaphys import descricao as D
from novaphys import forma as F


def assinaturas(g: Grid) -> Dict[Tuple[int, int], FrozenSet]:
    """Cada celula -> o conjunto de testes verdadeiros nela."""
    bg = fundo(g)
    T = D.CondicaoAprendida._testes(g, bg)
    h, w = dims(g)
    fora: Dict[Tuple[int, int], set] = {(i, j): set()
                                        for i in range(h) for j in range(w)}
    for c, S in T.items():
        for p in S:
            if p in fora:
                fora[p].add(c)
    return {p: frozenset(v) for p, v in fora.items()}


def mapa(pares) -> Dict[FrozenSet, Counter]:
    m: Dict[FrozenSet, Counter] = {}
    for a, b in pares:
        A = assinaturas(a)
        for p, s in A.items():
            m.setdefault(s, Counter())[b[p[0]][p[1]]] += 1
    return m


def prever(m, a: Grid) -> Optional[Grid]:
    A = assinaturas(a)
    h, w = dims(a)
    chaves = list(m)
    fora = []
    for i in range(h):
        lin = []
        for j in range(w):
            s = A[(i, j)]
            c = m.get(s)
            if c is None:
                # vizinho mais proximo por diferenca simetrica
                melhor, md = None, None
                for k in chaves:
                    d = len(s ^ k)
                    if md is None or d < md:
                        melhor, md = k, d
                if melhor is None:
                    return None
                c = m[melhor]
            lin.append(c.most_common(1)[0][0])
        fora.append(tuple(lin))
    return tuple(fora)


def imune(tr) -> bool:
    """
    Leave-one-out. Uma tabela decorada reproduz o treino e falha aqui sempre,
    porque a assinatura do par retirado nunca esteve no mapa.
    """
    if len(tr) < 3:
        return False
    for i in range(len(tr)):
        resto = tr[:i] + tr[i + 1:]
        try:
            if prever(mapa(resto), tr[i][0]) != tr[i][1]:
                return False
        except Exception:
            return False
    return True


TAR = None


def _ini():
    global TAR
    TAR = carregar(PASTA_AVAL)


def _uma(ti):
    n, tr, te = TAR[ti]
    try:
        fms = F.propor(tr)
        if not fms:
            return n, 0, 0
        fm = fms[0]
        reb = []
        for a, b in tr:
            g = F.base(fm, a)
            if g is None or dims(g) != dims(b):
                return n, 0, 0
            reb.append((g, b))
        if not imune(reb):
            return n, 0, 0
        m = mapa(reb)
        ok = True
        for x, y in te:
            gx = F.base(fm, x)
            if gx is None or prever(m, gx) != y:
                ok = False
                break
        return n, 1, 1 if ok else 0
    except Exception:
        return n, 0, 0


# o que a INDUCAO resolve hoje, do melhor braco medido
INDUCAO = {"00576224", "070dd51e", "0a2355a6", "0c786b71", "12eac192",
           "140c817e", "15663ba9", "17b80ad2", "1acc24af", "1c0d0a4b",
           "32e9702f", "37d3e8b2", "45bbe264", "59341089", "5ffb2104",
           "60c09cac", "66e6c45b", "7953d61e", "7d1f7ee8", "833dafe3",
           "84f2aca1", "903d1b4a", "9ddd00f0", "aa18de87", "aa4ec2a5",
           "ae58858e", "be03b35f", "c48954c1", "c7d4e6ad", "cd3c21df",
           "e0fb7511", "ed98d772", "fc754716"}


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=8)
    a = ap.parse_args()
    _ini()

    print("=" * 76)
    print("TRANSDUCAO — prever sem extrair regra")
    print("=" * 76)
    t0 = time.time()
    with Pool(a.nucleos, initializer=_ini) as pool:
        res = pool.map(_uma, range(len(TAR)), chunksize=4)

    imunes = {n for n, i, _o in res if i}
    passou = {n for n, _i, o in res if o}
    print(f"\n  ({time.time()-t0:.0f}s)")
    print(f"  sobreviveu ao leave-one-out : {len(imunes):3d}/400")
    print(f"  passou no TESTE             : {len(passou):3d}/400   "
          f"precisao {len(passou)/max(len(imunes),1):.1%}")
    print(f"\n  tarefas: {sorted(passou)}")
    fora = passou - INDUCAO
    print(f"\n  T1 . a inducao NAO resolve  : {len(fora)}/{len(passou)}  "
          f"{sorted(fora)}")
    print(f"  T2 . INEDITAS no projeto    : {len(passou - D.JA)}  "
          f"{sorted(passou - D.JA)}")
    print(f"\n  uniao inducao + transducao  : {len(INDUCAO | passou)}/400")

    print("")
    print("=" * 76)
    if not passou:
        print("T0 FALHOU: transducao nao e viavel nesta representacao.")
    elif len(fora) * 2 >= len(passou):
        print("T1 . DISJUNTO, como Li & Ellis preveem. Ensemblar e ganho de")
        print("     graca, e os dois modos de inferencia sao complementares.")
    else:
        print("T1 . nao disjunto: a transducao repete o que a inducao ja fazia.")


if __name__ == "__main__":
    main()
