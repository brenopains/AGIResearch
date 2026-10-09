"""
novaphys.orcamento_arc — ninguem pensa menos que a tarefa mediana
===================================================================

DE ONDE VEM
-------------
No laco na grade, a mesma lente com 24 cenas caiu de 20/20 para 5/20 so pelo teto
de avaliacoes do nucleo — TETO_CEL // celulas: com MAIS dado, MENOS busca — e voltou
a 20/20 sem ele. A mesma regra vale no ARC. A distribuicao nas 400 tarefas de
avaliacao (TETO_CEL = 1.500.000):

    quantil 0,00     416 avaliacoes   (47996f11, 3600 celulas de treino)
    quantil 0,10   1.033
    quantil 0,25   1.901
    quantil 0,50   3.750             (mediana)
    quantil 0,75   9.740
    quantil 0,90  23.437
    quantil 1,00 500.000             (642d658d, 3 celulas)
    mediana nas resolvidas 4.643; nas nao resolvidas 3.605

As tarefas com mais dado pensam ate 9x menos que a mediana. O teto veio de um
motivo real — uma tarefa 30x30 levava 537 s e travava a linha —, mas a forma dele
pune justamente quem tem mais evidencia.

O TESTE
---------
PISO na mediana: teto = max(TETO_CEL // celulas, 3.750). Ninguem pensa menos que a
tarefa tipica; as pequenas continuam com o que ja tem. So as tarefas ABAIXO da
mediana mudam de orcamento — as outras dao resultado identico por construcao
(busca deterministica, mesmo teto) — entao so elas rodam, e cada uma e comparada
com o resultado da rodada atual (nucleo 44/400, `descricao` com completar-com-copia).

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  B1  ganho liquido no nucleo >= +1 (entre as tarefas afetadas)
  B2  no maximo 1 perda (mais busca tambem pode achar uma pilha mais barata que
      fecha o treino e erra o teste — o ajuste espurio de sempre)
PREVISAO: +1 a +3, zero ou uma perda. O efeito da lente foi grande porque la o dado
era de UM mundo; no ARC, a maioria das tarefas grandes nao resolvidas deve estar
fora do alcance da linguagem, nao do orcamento.
"""

from __future__ import annotations

import argparse
import time

from novaphys.arc import carregar, dims
from novaphys.eter import PASTA_AVAL
from novaphys import descricao as D

PISO = 3750
# as 44 do nucleo na rodada com completar-com-copia (2026-09-11, 3325 s)
ATUAL = {"00576224", "00dbd492", "070dd51e", "0a2355a6", "0c786b71", "0f63c0b9",
         "12eac192", "140c817e", "15663ba9", "17b80ad2", "1c0d0a4b", "21f83797",
         "32e9702f", "332efdb3", "37d3e8b2", "45737921", "45bbe264", "48131b3c",
         "4f537728", "55059096", "59341089", "5ffb2104", "60c09cac", "66e6c45b",
         "7953d61e", "7d1f7ee8", "833dafe3", "84f2aca1", "903d1b4a", "9ddd00f0",
         "aa18de87", "aa4ec2a5", "ae58858e", "be03b35f", "bf32578f", "bf699163",
         "c48954c1", "cd3c21df", "ce039d91", "d37a1ef5", "e0fb7511", "ed98d772",
         "f0df5ff0", "fc754716"}


def _uma(args):
    nome, tr, te, teto = args
    try:
        fm, pilha, err = D.montar(tr, teto=teto)
        if err != 0 or not pilha:
            return nome, False
        return nome, all(D.aplicar((fm, pilha), x) == y for x, y in te)
    except Exception:
        return nome, False


def _ini():
    D.usar_biblioteca(None) if False else None


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=3)
    a = ap.parse_args()
    tarefas = carregar(PASTA_AVAL)
    afetadas = []
    for nome, tr, te in tarefas:
        cel = max(1, sum(dims(b)[0] * dims(b)[1] for _a, b in tr))
        lim = max(D.TETO_CEL // cel, 40)
        if lim < PISO:
            afetadas.append((nome, tr, te, PISO))
    print("=" * 76)
    print(f"ORCAMENTO ARC — piso de {PISO} avaliacoes nas {len(afetadas)} tarefas abaixo da mediana")
    print("=" * 76)
    t0 = time.time()
    with Pool(a.nucleos) as pool:
        res = dict(pool.map(_uma, afetadas, chunksize=1))
    print(f"\n  ({time.time()-t0:.0f}s)")
    antes = {n for n, *_r in afetadas if n in ATUAL}
    depois = {n for n, ok in res.items() if ok}
    ganhos, perdas = sorted(depois - antes), sorted(antes - depois)
    print(f"  afetadas resolvidas antes {len(antes)}, com o piso {len(depois)}")
    print(f"  GANHOS {len(ganhos)}: {ganhos}")
    print(f"  PERDAS {len(perdas)}: {perdas}")
    print(f"  nucleo: 44 -> {44 + len(ganhos) - len(perdas)}/400")
    novas = sorted(set(ganhos) - D.JA)
    print(f"  ineditas (fora da uniao): {len(novas)} {novas}")
    print("")
    print("=" * 76)
    liq = len(ganhos) - len(perdas)
    print(f"B1 {'.' if liq >= 1 else 'FALHOU .'} ganho liquido {liq:+d} (>= +1)")
    print(f"B2 {'.' if len(perdas) <= 1 else 'FALHOU .'} perdas {len(perdas)} (<= 1)")


if __name__ == "__main__":
    main()
