"""
novaphys.profundidade — quanto a COMPOSICAO compra, com o vocabulario congelado
================================================================================

POR QUE ESTE EXPERIMENTO EXISTE
---------------------------------
Um revisor hostil derrubou hoje a afirmacao central do projeto:

    "100% do rendimento e composicional" era TAUTOLOGICO.
    Medido: 0 de 22 codecs fecham uma grade sozinhos.

Se nenhum codec fecha sozinho, pilha profunda e obrigatoria POR CONSTRUCAO, e
dizer que o rendimento e composicional nao diz nada sobre o mundo — diz sobre a
definicao de codec.

E a segunda derrubada: a uniao 33 -> 54 e CO-INTERVENCAO. Entre os dois numeros
entraram codec de forma, condicao aprendida, linguagem de codecs, gramatica de
testes e pintores. O efeito do portao nunca foi medido isolado.

O QUE DA PARA MEDIR HONESTAMENTE
----------------------------------
A permutacao da ordem de insercao NAO conserta o confundidor real: o codec 22
foi escrito com mais informacao sobre as falhas que o codec 1, e permutar a
ordem de AVALIACAO nao desfaz a ordem de CRIACAO. Isso e irreparavel neste
conjunto de dados; so um pre-registro com codecs congelados e mundos sorteados
por terceiro resolveria.

Mas ha uma medida limpa que ninguem fez, e ela responde a pergunta que importa:

    com o conjunto de codecs CONGELADO, quanto a profundidade compra?

Rodar a uniao com `fundo_max` em {1, 2, 3, 5}. A diferenca e o valor da
composicao, medido sem trocar nada alem do limite de profundidade. Nao e
tautologico: se profundidade 1 ja desse quase tudo, "composicao" seria fachada.

E ha um segundo braco que separa forma de pilha, que e a metrica que eu declarei
vigiar ha horas e nunca medi: quantas tarefas precisam de forma != igual E
pilha >= 2 ao mesmo tempo.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  D0  PRE-CONDICAO. `fundo_max=5` tem que dar ~54, que e o numero corrente. Se
      nao der, algo mudou desde a ultima medida e o resto nao vale.

  D1  A CURVA DE PROFUNDIDADE
      prof 1 >= 80% do total -> composicao e fachada; o ganho e dos codecs
      prof 1 <= 40% do total -> composicao compra de verdade
      entre 40% e 80%        -> compra, e menos do que eu vinha dizendo

  D2  tarefas com forma != igual E pilha >= 2: o multiplicador que eu previ
      quando construi o codec de forma e nunca verifiquei.

MINHA CALIBRAGEM: cinquenta e oito previsoes, VINTE E DUAS certas, e hoje eu ja
tive uma afirmacao central retirada por tautologia.

PREVISAO: D1 entre 40% e 80% — provavelmente perto de 55%. Razao: a distribuicao
de profundidade das pilhas vencedoras e {1:13, 2:13, 3:5, 4:6, 5:1}, entao 13 de
38 fecham com um codec so. Mas isso conta o codec DEPOIS do rebase da forma, e
`forma` faz trabalho pesado. Aposto que prof 1 da entre 20 e 30 tarefas.
E aposto D2 BAIXO, entre 0 e 3 — as duas mecanicas nunca se cruzaram nas medidas
anteriores, e nao ha razao para terem comecado.
"""

from __future__ import annotations

import argparse
import time

from novaphys.arc import carregar
from novaphys.eter import PASTA_AVAL
from novaphys import descricao as D

PROFS = (1, 2, 3, 5)
TAR = None


def _ini():
    global TAR
    TAR = carregar(PASTA_AVAL)
    D.usar_biblioteca(None)


def _uma(args):
    ti, prof = args
    n, tr, te = TAR[ti]
    try:
        fm, pilha, err = D.montar(tr, refoco=True, feixe=4, fundo_max=prof)
        if err != 0 or not pilha:
            return prof, n, 0, 0, None
        ok = all(D.aplicar((fm, pilha), x) == y for x, y in te)
        return prof, n, 1 if ok else 0, len(pilha), fm[0]
    except Exception:
        return prof, n, 0, 0, None


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=4)
    a = ap.parse_args()
    _ini()

    print("=" * 76)
    print("PROFUNDIDADE — quanto a composicao compra, vocabulario congelado")
    print("=" * 76)
    trab = [(ti, p) for p in PROFS for ti in range(len(TAR))]
    t0 = time.time()
    with Pool(a.nucleos, initializer=_ini) as pool:
        res = pool.map(_uma, trab, chunksize=4)

    print(f"\n  ({time.time()-t0:.0f}s)\n")
    print("  prof   passou   ganho sobre a anterior")
    print("  ----   ------   ----------------------")
    por = {}
    ant = 0
    for p in PROFS:
        s = {n for pp, n, o, _k, _f in res if pp == p and o}
        por[p] = s
        print(f"  {p:4d}   {len(s):6d}   {len(s)-ant:+22d}")
        ant = len(s)

    tot = len(por[PROFS[-1]])
    p1 = len(por[PROFS[0]])
    frac = p1 / max(tot, 1)
    print(f"\n  D1 . prof 1 = {p1}/{tot} = {frac:.0%} do total")

    cruz = [(n, k, f) for pp, n, o, k, f in res
            if pp == PROFS[-1] and o and k >= 2 and f and f != "igual"]
    print(f"\n  D2 . forma != igual E pilha >= 2: {len(cruz)}")
    for n, k, f in sorted(cruz):
        print(f"        {n}  forma={f}  pilha={k}")

    novas = por[PROFS[-1]] - por[PROFS[0]]
    print(f"\n  so a composicao resolve ({len(novas)}): {sorted(novas)}")

    print("")
    print("=" * 76)
    if frac >= 0.80:
        print("D1 . COMPOSICAO E FACHADA: profundidade 1 ja da quase tudo, e o")
        print("     ganho vem dos codecs, nao de compor.")
    elif frac <= 0.40:
        print("D1 . composicao compra de verdade.")
    else:
        print(f"D1 . composicao compra, e MENOS do que eu vinha dizendo:")
        print(f"     {frac:.0%} do total ja sai com um codec so.")


if __name__ == "__main__":
    main()
