"""
novaphys.fatorial — o erro como dado, separado da tolerancia a errar
=====================================================================

DE ONDE VEM
-------------
Do usuario: "a mente humana erra grosseira e as vezes sutilmente em tempo real
enquanto pensa. As IAs padrao sao rigidas e nao flexiveis a erros, sempre
forcando o acerto."

O diagnostico esta certo e e literal no codigo. Esta linha jogava tudo fora:

    if err != 0 or not pilha:
        return nome, 0, False, (), None

Uma pilha que derrubou o residuo de 700 celulas para 3 era descartada
exatamente como uma que nao fez nada. Medido nas 400:

    residuo final    0 (resolvida)  26      celulas de saida    229.117
                     1-3             5      ainda erradas        31.110
                     4-10           13      -> a pilha ja explica 86,4%
                     11-30          66         das celulas E VALE ZERO
                     31-100        149
                     101+          101
                     falhou         40

A tarefa `e1d2900e`, que eu analisei a mao e cuja regra eu tinha certa em 21/21
pecas, aparece com residuo 2.

A TESE TEM DOIS MECANISMOS DENTRO, E ELES SAO DIFERENTES
----------------------------------------------------------
  M1 REFOCO — o erro vira DADO.
     Hoje `propor(treino)` ve a tarefa crua. O `mapa_cor` precisa ser
     consistente na grade INTEIRA, falha, e nunca chega a ser proposto — mesmo
     quando seria perfeitamente consistente nas 10% de celulas que sobraram.
     Passando o residuo para `propor`, cada codec deduz so sobre o que a pilha
     ainda erra. O erro deixa de ser penalidade e vira o RECORTE do problema.

  M2 FEIXE — tolerancia a PIORAR.
     O laco pega sempre o melhor passo imediato. Guloso nao atravessa vale: um
     codec que economiza zero bits agora e habilita economia enorme no passo
     seguinte jamais e escolhido. E a rigidez de exigir acerto a cada passo — o
     analogo do top-k, que o usuario apontou como o que salva as LLM do argmax.

Sao coisas distintas e podem ter valores muito distintos, entao vao num
FATORIAL 2x2 em vez de irem juntos. Ir junto e o erro classico que esconde qual
metade pagou.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  R0  PRE-CONDICAO. O braco (refoco=nao, feixe=1) e o sistema de ontem e tem
      que dar ~20/400. Baixei `fundo_max` de 6 para 5 pelo custo do feixe,
      entao aceito 19 ou 20; abaixo disso e erro de implementacao.

  R1  refoco sozinho  ((sim,1) - (nao,1))
  R2  feixe sozinho   ((nao,4) - (nao,1))
  R3  interacao       ((sim,4) - o melhor dos dois isolados)

MINHA CALIBRAGEM: trinta e nove previsoes, CATORZE certas.

PREVISAO, e com mecanismo por tras em vez de torcida:
    R1 POSITIVO, R2 perto de ZERO.
Motivo: a historia medida deste projeto e "mudar a BUSCA rende zero" (doze
tentativas) e "mudar a REPRESENTACAO rende +3" (oito). O feixe e mudanca de
busca pura. O refoco NAO e busca: ele muda o dado que o codec ve na hora de
deduzir, que e mudanca de representacao. Se R2 vier grande eu estou errado
sobre a lei mais estavel que este projeto ja mediu.
Aposto R1 entre +3 e +8, R2 entre 0 e +1.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

from novaphys.arc import carregar
from novaphys.eter import PASTA_AVAL
from novaphys import descricao as D

BRACOS = [("nao", 1), ("sim", 1), ("nao", 4), ("sim", 4)]
TAR = None


def _ini():
    global TAR
    TAR = carregar(PASTA_AVAL)


def _uma(args):
    bi, ti = args
    rf, fx = BRACOS[bi]
    n, tr, te = TAR[ti]
    try:
        fm, pilha, err = D.montar(tr, refoco=(rf == "sim"), feixe=fx)
        if err != 0 or not pilha:
            return bi, n, 0, 0, None
        ok = all(D.aplicar((fm, pilha), x) == y for x, y in te)
        return bi, n, 1 if ok else 0, len(pilha), fm[0]
    except Exception:
        return bi, n, 0, 0, None


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=8)
    a = ap.parse_args()
    _ini()

    print("=" * 76)
    print("FATORIAL — erro como dado (refoco) x tolerancia a piorar (feixe)")
    print("=" * 76)
    trab = [(bi, ti) for bi in range(4) for ti in range(len(TAR))]
    t0 = time.time()
    with Pool(a.nucleos, initializer=_ini) as pool:
        res = pool.map(_uma, trab, chunksize=8)

    por = {bi: set() for bi in range(4)}
    prof = {bi: Counter() for bi in range(4)}
    fmm = {bi: Counter() for bi in range(4)}
    for bi, n, ok, k, fm in res:
        if ok:
            por[bi].add(n)
            prof[bi][k] += 1
            fmm[bi][fm] += 1

    print(f"\n  ({time.time()-t0:.0f}s)\n")
    print("  refoco  feixe   passou   pilhas 2+   forma != igual")
    print("  ------  -----   ------   ---------   --------------")
    for bi, (rf, fx) in enumerate(BRACOS):
        p2 = sum(v for k, v in prof[bi].items() if k >= 2)
        fd = sum(v for k, v in fmm[bi].items() if k and k != "igual")
        print(f"  {rf:6s}  {fx:5d}   {len(por[bi]):6d}   {p2:9d}   {fd:14d}")

    b, r, f, rf_ = (len(por[0]), len(por[1]), len(por[2]), len(por[3]))
    print("")
    print(f"  R0 . base (nao,1)        : {b}/400")
    print(f"  R1 . refoco sozinho      : {r-b:+d}")
    print(f"  R2 . feixe sozinho       : {f-b:+d}")
    print(f"  R3 . os dois             : {rf_-b:+d}   "
          f"(interacao {rf_-b-(r-b)-(f-b):+d})")

    uniao = por[0] | por[1] | por[2] | por[3]
    print(f"\n  uniao dos quatro bracos  : {len(uniao)}/400")
    for bi, (rfn, fx) in enumerate(BRACOS):
        so = por[bi] - set().union(*[por[k] for k in range(4) if k != bi])
        if so:
            print(f"    so ({rfn},{fx}) resolve: {sorted(so)}")

    print("")
    print("=" * 76)
    if b < 19:
        print("R0 FALHOU: a base nao reproduz o sistema de ontem.")
    if (r - b) > (f - b) and (r - b) > 0:
        print("R1 > R2 . o erro como DADO vale mais que tolerar piorar.")
        print("     A lei 'representacao rende, busca nao' sobrevive.")
    elif (f - b) >= (r - b) and (f - b) > 0:
        print("R2 >= R1 . tolerar piorar vale tanto ou mais. A lei mais estavel")
        print("     do projeto acaba de ganhar excecao.")
    else:
        print("Nenhum dos dois pagou.")


if __name__ == "__main__":
    main()
