"""
novaphys.conj — HDC contra o obvio, na busca que o residuo abriu
=================================================================

DE ONDE VEM
-------------
Do usuario perguntando se HDC nao seria a camada que falta. Fui verificar antes
de opinar e descobri que este projeto ja tem os dois lados MEDIDOS, e eles
dizem coisas opostas conforme o PAPEL:

    HDC como ORDENADOR de espaco que outro enumerou
        `prion_hdc.py`, orcamento casado, controle aleatorio pareado
        -> erro 2,6 d contra 11,3 d do sorteio; achou em profundidade 3 o que
           o exaustivo de profundidade 2 nunca alcancou.  POSITIVO.

    HDC/recombinacao como GERADOR de estrutura nova
        `analogia.py`, controle pareado
        -> restritivo 7/400, generativo 1.092 triplas 2/400, aleatorio 0/400.
           Recombinar partes DESTROI sinal.  NEGATIVO.

O motivo do negativo e estrutural: `bind`/`bundle` tem perda, e recuperar
estrutura exata exige cleanup contra um codebook — entao so se recupera
combinacao de atomos JA presentes. HDC nunca propoe primitiva nova.

Entao o teste certo e no papel em que ele ganhou: ORDENAR.

O ESPACO
----------
`CondicaoAprendida` coleta os testes NECESSARIOS (verdadeiros em todas as
celulas-alvo) e procura uma conjuncao cuja intersecao seja exatamente o alvo.
Como todo teste necessario contem o alvo, toda intersecao tambem contem: o erro
e so FALSO POSITIVO, e adicionar teste so encolhe.

Antes eu truncava em profundidade 3 por custo. Com ordem boa da para ir a 5 —
que e exatamente a situacao do prion, onde o exaustivo morria na 2.

E POR ISSO EU INCLUO O BRACO OBVIO
------------------------------------
Se o erro e so falso positivo e adicionar teste so encolhe, isto e COBERTURA, e
cobertura tem um algoritmo obvio e forte: guloso, pegando a cada passo o teste
que mata mais falso positivo. Se eu rodasse so exaustivo x aleatorio x HDC, o
HDC ganharia de dois espantalhos e eu chamaria isso de vitoria.

Os quatro bracos avaliam no MAXIMO o mesmo numero de conjuncoes (ORC_CONJ), do
mesmo conjunto de candidatos. So a ORDEM muda.

    exaustivo   ordem canonica, tamanho crescente
    aleatorio   ordem sorteada, semente pre-registrada 20260909
    hdc         semeia 1/4 do orcamento, aprende prototipos BOM e RUIM,
                ordena o resto por sim(v,BOM) - sim(v,RUIM)
    guloso      cobertura gulosa, sem pool

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  P0  PRE-CONDICAO. Algum braco tem que dar >= 22/400, que e o sistema com a
      busca truncada em profundidade 3. Se todos cairem, e erro de
      implementacao na troca da busca.

  P1  HDC > aleatorio E HDC >= guloso -> a representacao ordena melhor que a
      heuristica obvia, e a camada associativa do usuario esta certa
  P2  HDC > aleatorio mas guloso >= HDC -> HDC ordena, mas o obvio ordena
      igual ou melhor; nao vale a complexidade
  P3  HDC ~ aleatorio -> num espaco de cobertura o HDC nao acrescenta nada

MINHA CALIBRAGEM: quarenta e duas previsoes, DEZESSEIS certas, e as vinte e
seis erradas quase todas otimistas.

PREVISAO: P2. O guloso deve ganhar, porque o erro aqui e monotono e aditivo —
cada teste mata um subconjunto de falsos positivos, e isso e cobertura pura,
onde guloso tem garantia logaritmica. O prion era diferente: la o sinal era um
gradiente suave e nao aditivo ("mais x e melhor"), sem heuristica gulosa obvia,
e foi por isso que o HDC teve espaco. Aposto que o HDC bate o sorteio e perde
para o guloso — o que ainda seria informativo, porque diria que o valor do HDC
depende do sinal ser nao-aditivo, e nao de ele ser "associativo".
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

from novaphys.arc import carregar
from novaphys.eter import PASTA_AVAL
from novaphys import descricao as D

MODOS = ["exaustivo", "aleatorio", "hdc", "guloso"]
TAR = None


def _ini():
    global TAR
    TAR = carregar(PASTA_AVAL)


def _uma(args):
    mi, ti = args
    D.MODO_CONJ = MODOS[mi]
    n, tr, te = TAR[ti]
    try:
        fm, pilha, err = D.montar(tr, refoco=True, feixe=1)
        if err != 0 or not pilha:
            return mi, n, 0, ()
        ok = all(D.aplicar((fm, pilha), x) == y for x, y in te)
        return mi, n, 1 if ok else 0, tuple(cd.nome for cd, _m in pilha)
    except Exception:
        return mi, n, 0, ()


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=8)
    a = ap.parse_args()
    _ini()

    print("=" * 76)
    print("CONJUNCAO — HDC contra o obvio, orcamento casado")
    print("=" * 76)
    print(f"  orcamento {D.ORC_CONJ} conjuncoes por tarefa, profundidade ate "
          f"{D.PROF_CONJ}, dim HDC {D.DIM_HDC}")
    trab = [(mi, ti) for mi in range(len(MODOS)) for ti in range(len(TAR))]
    t0 = time.time()
    with Pool(a.nucleos, initializer=_ini) as pool:
        res = pool.map(_uma, trab, chunksize=8)

    por = {mi: set() for mi in range(len(MODOS))}
    usa = {mi: 0 for mi in range(len(MODOS))}
    for mi, n, ok, f in res:
        if ok:
            por[mi].add(n)
            if "condicao" in f:
                usa[mi] += 1

    print(f"\n  ({time.time()-t0:.0f}s)\n")
    print("  ordem        passou   usando condicao")
    print("  ---------    ------   ---------------")
    for mi, m in enumerate(MODOS):
        print(f"  {m:9s}    {len(por[mi]):6d}   {usa[mi]:15d}")

    ex, al, hd, gu = [len(por[i]) for i in range(4)]
    print("")
    print(f"  P . hdc - aleatorio : {hd-al:+d}")
    print(f"  P . hdc - guloso    : {hd-gu:+d}")
    print(f"  P . guloso - exaust : {gu-ex:+d}")
    uni = set().union(*por.values())
    print(f"\n  uniao dos quatro    : {len(uni)}/400")
    for mi, m in enumerate(MODOS):
        so = por[mi] - set().union(*[por[k] for k in range(4) if k != mi])
        if so:
            print(f"    so `{m}` resolve: {sorted(so)}")

    print("")
    print("=" * 76)
    if max(ex, al, hd, gu) < 22:
        print("P0 FALHOU: nenhum braco alcanca o sistema truncado em prof 3.")
    if hd > al and hd >= gu:
        print("P1 . a representacao ordena melhor que a heuristica obvia.")
    elif hd > al:
        print("P2 . HDC ordena, mas o obvio ordena igual ou melhor.")
    else:
        print("P3 . num espaco de cobertura o HDC nao acrescenta nada.")


if __name__ == "__main__":
    main()
