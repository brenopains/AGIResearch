"""
novaphys.holdout — a biblioteca ANTECIPA, ou so comprime o que ja viu?
=======================================================================

A PERGUNTA
------------
`abstrai.py` extraiu 6 abstracoes que se pagam pelo criterio do Stitch e as
ligou na busca. Mas o corpus veio das DUAS particoes, e a medida foi feita na
particao de avaliacao. Isso ainda e compressao RETROSPECTIVA — sobre o corpus
em vez da tarefa, mas retrospectiva.

A afirmacao forte, a de `Prospective Compression in Human Abstraction Learning`
(arxiv.org/abs/2605.09985), e outra: humanos formam abstracoes por utilidade
FUTURA antecipada, e por isso preferem compressoes subotimas dos dados atuais
que transferem melhor.

Testar isso exige apostar em tarefas que o corpus nunca viu.

O DESENHO
-----------
Extrair a biblioteca SO da particao de TREINO do ARC. Medir na de AVALIACAO.
Zero vazamento: nenhuma das 400 tarefas pontuadas contribuiu com uma unica
abstracao.

    A0  sem biblioteca                     a base
    A1  biblioteca do TREINO               a hipotese
    A2  biblioteca ALEATORIA, pareada      o placebo

O braco A2 existe porque semear a busca com QUALQUER coisa perturba o guloso, e
perturbar guloso as vezes ajuda por acidente. Sem ele, um ganho de A1 nao
distingue "a abstracao certa" de "qualquer abstracao". E o mesmo controle que
`analogia.py` usou para descobrir que recombinar esquema destroi sinal — sem
ele, o braco generativo pareceria funcionar.

O placebo e pareado em TUDO menos no conteudo: mesmo numero de abstracoes,
mesmo tamanho de esqueleto, mesmo numero de buracos, mesma familia de pintor.
So os TESTES sao sorteados do vocabulario, com semente pre-registrada.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  H0  PRE-CONDICAO. A0 tem que dar >= 32, que e o melhor braco medido sem
      biblioteca. Se cair, e erro na troca de biblioteca.

  H1  A1 > A2  -> a biblioteca ANTECIPA. Abstracoes formadas em tarefas que o
                  sistema nunca vera de novo ajudam em tarefas novas, e o
                  conteudo delas importa. E compressao prospectiva medida.
  H2  A1 ~ A2  -> semear ajuda, mas o conteudo nao importa; e perturbacao de
                  busca, nao transferencia de conhecimento.
  H3  A1 <= A0 -> a biblioteca atrapalha; o esqueleto do passado enviesa a
                  busca para longe da resposta da tarefa nova.

SEMENTE PRE-REGISTRADA: 20260909

MINHA CALIBRAGEM: quarenta e sete previsoes, DEZESSETE certas, e as trinta
erradas quase todas otimistas.

PREVISAO: H2. Duas razoes, e as duas sao medidas desta sessao. Primeira: o
proprio `abstrai.py` mostrou que os seletores sao especificos por tarefa e so
os PINTORES se repetem — as abstracoes de topo tinham esqueleto VAZIO. Se o que
transfere e so o pintor, entao o esqueleto sorteado do placebo perde pouco.
Segunda: quinze mudancas de busca deram zero neste projeto, e semear e mudanca
de busca. Aposto que A1 e A2 empatam perto de A0, e que a diferenca real, se
existir, vem do preco em BITS e nao da semente.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import time
from collections import Counter

from novaphys.arc import carregar
from novaphys.compositor import fundo
from novaphys.eter import PASTA_AVAL
from novaphys import descricao as D

SEMENTE = 20260909
LIB_TREINO = "biblioteca_treino.json"
LIB_PLACEBO = "biblioteca_placebo.json"
BRACOS = [("A0 sem biblioteca", None),
          ("A1 do treino", LIB_TREINO),
          ("A2 placebo pareado", LIB_PLACEBO)]
TAR = None


def _ini():
    global TAR
    TAR = carregar(PASTA_AVAL)


def gerar_placebo():
    """
    Mesmo formato da biblioteca real, conteudo sorteado. Pareado em numero de
    abstracoes, tamanho de esqueleto, numero de buracos e familia de pintor.
    """
    real = json.loads(io.open(LIB_TREINO, encoding="utf-8").read())
    ts = carregar(PASTA_AVAL)
    voc = set()
    for _n, tr, _te in ts[:40]:
        a = tr[0][0]
        voc |= set(D.CondicaoAprendida._testes(a, fundo(a)))
    voc = sorted(voc, key=repr)
    rnd = random.Random(SEMENTE)
    fora = []
    for ab in real:
        k = len(ab["comum"])
        fora.append({"comum": [list(x) for x in rnd.sample(voc, k)],
                     "buracos": ab["buracos"],
                     "pintor": ab["pintor"],
                     "usos": ab["usos"]})
    io.open(LIB_PLACEBO, "w", encoding="utf-8").write(
        json.dumps(fora, indent=1, ensure_ascii=False))
    return len(fora), len(voc)


def _uma(args):
    bi, ti = args
    D.usar_biblioteca(BRACOS[bi][1])
    n, tr, te = TAR[ti]
    try:
        fm, pilha, err = D.montar(tr, refoco=True, feixe=1)
        if err != 0 or not pilha:
            return bi, n, 0, 0
        ok = all(D.aplicar((fm, pilha), x) == y for x, y in te)
        return bi, n, 1 if ok else 0, sum(
            1 for cd, _m in pilha if cd.nome == "programa")
    except Exception:
        return bi, n, 0, 0


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=8)
    a = ap.parse_args()
    _ini()

    print("=" * 76)
    print("HOLDOUT — a biblioteca antecipa, ou so comprime o que ja viu?")
    print("=" * 76)
    nab, nvoc = gerar_placebo()
    print(f"  placebo pareado: {nab} abstracoes sorteadas de {nvoc} testes, "
          f"semente {SEMENTE}")
    print("  biblioteca A1 extraida SO do treino; medida SO na avaliacao")

    trab = [(bi, ti) for bi in range(3) for ti in range(len(TAR))]
    t0 = time.time()
    with Pool(a.nucleos, initializer=_ini) as pool:
        res = pool.map(_uma, trab, chunksize=8)

    por = {bi: set() for bi in range(3)}
    prog = {bi: 0 for bi in range(3)}
    for bi, n, ok, np_ in res:
        if ok:
            por[bi].add(n)
            prog[bi] += np_

    print(f"\n  ({time.time()-t0:.0f}s)\n")
    print("  braco                  passou   programas usados")
    print("  --------------------   ------   ----------------")
    for bi, (nome, _f) in enumerate(BRACOS):
        print(f"  {nome:20s}   {len(por[bi]):6d}   {prog[bi]:16d}")

    b, r, pl = len(por[0]), len(por[1]), len(por[2])
    print("")
    print(f"  H . A1 - A0 (biblioteca vs nada)    : {r-b:+d}")
    print(f"  H . A2 - A0 (placebo   vs nada)     : {pl-b:+d}")
    print(f"  H . A1 - A2 (conteudo importa?)     : {r-pl:+d}")
    print(f"\n  uniao dos tres: {len(por[0] | por[1] | por[2])}/400")
    so1 = por[1] - por[0] - por[2]
    if so1:
        print(f"  so a biblioteca do treino resolve: {sorted(so1)}")

    print("")
    print("=" * 76)
    if b < 32:
        print("H0 FALHOU: a base caiu abaixo do melhor braco sem biblioteca.")
    if r > pl and r > b:
        print("H1 . a biblioteca ANTECIPA. Abstracoes de tarefas que o sistema")
        print("     nunca vera de novo ajudam em tarefas novas, e o CONTEUDO")
        print("     delas importa. Compressao prospectiva medida.")
    elif r <= b:
        print("H3 . a biblioteca atrapalha: o esqueleto do passado enviesa a")
        print("     busca para longe da resposta da tarefa nova.")
    else:
        print("H2 . semear ajuda, mas o conteudo nao importa. E perturbacao de")
        print("     busca, nao transferencia de conhecimento.")


if __name__ == "__main__":
    main()
