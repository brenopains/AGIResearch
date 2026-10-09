"""
novaphys.quantosex — quantos exemplos fazem "ajustou" voltar a ser evidencia?
===============================================================================

A AFIRMACAO QUE ESTA SENDO TESTADA
------------------------------------
`ockham.py` mediu: com 8 lentes x 10 propriedades = 80 familias de hipotese,
127 de 400 tarefas acham uma regra que reproduz TODO o treino, e so 6 passam no
teste. Precisao 4,7%. Validacao cruzada corta para 4 com precisao 100%, mas
perde 2 verdadeiras e exclui 20 tarefas que tem menos de 3 pares.

Dai nasceu a leitura de que o gargalo passou a ser o BENCHMARK: o ARC da 2 a 4
exemplos, e deixar um de fora com 3 e deduzir com 2.

Essa leitura e plausivel e nao estava medida. Este arquivo mede.

RESSALVA QUE PRECISA VIR ANTES DO NUMERO
------------------------------------------
A limitacao de exemplos explica uma FATIA, nao o resultado todo. Os 4/400 saem
de UM motor estreito — deltas de objeto previstos por uma propriedade. A uniao
dos motores do projeto e 25/400, e as outras 21 falham por INEXPRESSIVIDADE,
nao por falta de dado. Confundir as duas coisas transforma um limite meu num
limite do mundo.

A MEDIDA, e ela nao sintetiza nada
------------------------------------
Para cada tarefa real do ARC, com K = 1, 2, 3, ... pares de treino:

    quantas das 80 familias reproduzem os K primeiros pares?

Com K=1 muitas familias passam por acidente. Conforme K cresce, as espurias
morrem. A curva diz quantos exemplos sao necessarios para sobrar ~1 — e e ai
que "ajustou" volta a ser evidencia.

Nao ha nada sintetico: sao as tarefas reais, e a contagem e exata.

E ha um segundo numero, que e o que decide de verdade: entre as familias
sobreviventes com K pares, qual fracao acerta o TESTE? Essa e a PRECISAO em
funcao de K. Se ela ja esta alta em K=3, mais exemplos nao resolveriam nada e a
parede nao e o benchmark. Se ela sobe forte de K=2 para K=4, a extrapolacao
para K=8 tem base.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  Q0  PRE-CONDICAO. A contagem em K = todos os pares tem que reproduzir
      `segmenta.py`: ~127 tarefas com pelo menos uma familia sobrevivente.

  Q1  a mediana de familias sobreviventes cai para <= 2 em K <= 4, E a
      precisao sobe monotonicamente com K
      -> a leitura esta certa: mais exemplos fariam "ajustou" virar evidencia,
         e o gargalo desta linha e mesmo o benchmark
  Q2  a mediana cai mas a precisao NAO sobe com K
      -> exemplos eliminam candidatos e nao melhoram a escolha; o problema nao
         e quantidade de dado, e a familia estar errada de saida
  Q3  a mediana nao cai
      -> as familias sao redundantes demais entre si; nenhum numero de
         exemplos separa, porque elas concordam nos dados e discordam so fora

MINHA CALIBRAGEM: onze previsoes hoje, TRES certas, e as oito erradas todas
otimistas.

PREVISAO: Q2. Espero a mediana cair rapido (de ~5 em K=1 para 1 ou 2 em K=3) e
espero a precisao NAO subir muito, ficando abaixo de 30% mesmo com todos os
pares. Razao concreta: as 80 familias nao sao independentes — `cc4cor` e
`cc8cor` dao a mesma segmentacao na maioria das grades, e `tam`/`rank_tam`/
`e_maior` sao quase a mesma propriedade. Familias redundantes morrem juntas ou
sobrevivem juntas, e exemplo nenhum separa o que concorda.

Se der Q2, a leitura corrige a narrativa: o mundo NAO esta negando dado. A
familia certa nao esta no conjunto, e mais exemplos nao colocam ela la.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from typing import Dict, List, Tuple

from novaphys.arc import carregar
from novaphys.eter import PASTA_AVAL
from novaphys.segmenta import LENTES, PROPS, aplicar, deduzir


def familias_que_ajustam(tr, k: int) -> List[Tuple[str, object, object]]:
    """
    Quais (lente, regra) reproduzem os K PRIMEIROS pares. A deducao usa so
    esses K — nada dos pares seguintes vaza.
    """
    sub = tr[:k]
    fora = []
    for ln, lente in LENTES:
        try:
            r = deduzir(sub, lente)
            if r is None:
                continue
            if any(aplicar(r, a, lente) != b for a, b in sub):
                continue
            fora.append((f"{ln}|{r[0]}", r, lente))
        except Exception:
            continue
    return fora


def _uma(args):
    nome, tr, te = args
    linha = []
    for k in range(1, len(tr) + 1):
        fam = familias_que_ajustam(tr, k)
        certas = 0
        for _n, r, lente in fam:
            try:
                if all(aplicar(r, x, lente) == y for x, y in te):
                    certas += 1
            except Exception:
                pass
        linha.append((k, len(fam), certas))
    return nome, len(tr), linha


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=10)
    a = ap.parse_args()

    print("=" * 78)
    print("QUANTOSEX — quantos exemplos fazem 'ajustou' voltar a ser evidencia")
    print("=" * 78)
    tarefas = carregar(PASTA_AVAL)
    t0 = time.time()
    with Pool(a.nucleos) as pool:
        res = pool.map(_uma, [(n, tr, te) for n, tr, te in tarefas],
                       chunksize=4)

    por_k: Dict[int, List[Tuple[int, int]]] = {}
    for _n, _np, linha in res:
        for k, nf, nc in linha:
            if nf:
                por_k.setdefault(k, []).append((nf, nc))

    print(f"\n{len(tarefas)} tarefas, {len(LENTES)}x{len(PROPS)} = "
          f"{len(LENTES)*len(PROPS)} familias  ({time.time()-t0:.0f}s)")
    print(f"\n{'K':>2} {'tarefas com':>12} {'familias':>10} {'familias':>10} "
          f"{'PRECISAO':>10}")
    print(f"{'':>2} {'alguma fam':>12} {'medianas':>10} {'certas':>10} "
          f"{'certa/viva':>10}")
    print("-" * 50)
    for k in sorted(por_k):
        v = por_k[k]
        med = sorted(nf for nf, _nc in v)[len(v) // 2]
        tot_f = sum(nf for nf, _nc in v)
        tot_c = sum(nc for _nf, nc in v)
        print(f"{k:>2} {len(v):>12} {med:>10} {tot_c:>10} "
              f"{tot_c/max(tot_f,1):>9.1%}")
    print("-" * 50)

    ks = sorted(por_k)
    prec = {k: sum(nc for _nf, nc in por_k[k])
            / max(sum(nf for nf, _nc in por_k[k]), 1) for k in ks}
    med = {k: sorted(nf for nf, _nc in por_k[k])[len(por_k[k]) // 2]
           for k in ks}
    ate4 = [k for k in ks if k <= 4]
    cai = med[max(ate4)] <= 2 if ate4 else False
    sobe = all(prec[ks[i]] <= prec[ks[i + 1]] + 1e-9
               for i in range(len(ks) - 1)) and len(ks) > 1

    print("\n" + "=" * 78)
    if cai and sobe:
        print("Q1 · mais exemplos fariam 'ajustou' voltar a ser evidencia.")
        print("     O gargalo DESTA LINHA e mesmo o benchmark.")
    elif cai:
        print("Q2 · exemplos ELIMINAM candidatos e NAO melhoram a escolha.")
        print("     O problema nao e quantidade de dado — e a familia certa")
        print("     nao estar no conjunto. Mais exemplos nao colocam ela la.")
        print("     O mundo NAO esta negando dado.")
    else:
        print("Q3 · a mediana nao cai: as familias sao redundantes entre si.")
        print("     Concordam nos dados e discordam so fora deles, entao")
        print("     numero nenhum de exemplos separa.")


if __name__ == "__main__":
    main()
