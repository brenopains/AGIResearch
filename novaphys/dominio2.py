"""
novaphys.dominio2 — as mesmas leituras valem num mundo que nao e o ARC?
=======================================================================

POR QUE ESTE EXPERIMENTO EXISTE
---------------------------------
A sessao inteira o usuario defendeu quatro coisas: transferencia, ancoragem de
simbolo, curriculo e vocabulario acumulado. As quatro empataram com placebo ou
deram zero — e ha uma razao estrutural para isso que so ficou clara agora:

    ancoragem semantica e IMENSURAVEL num dominio so.

Dentro do ARC, todo simbolo ja esta disponivel em toda tarefa. Por isso a
biblioteca deu 30/30/30 contra placebo sorteado, e por isso o professor perfeito
do `dica.py` comprou so +3. Nao ha o que transferir quando nao ha um segundo
lugar para onde transferir.

Entao o teste de "os simbolos sao gerais" e obrigar as MESMAS leituras a valerem
noutro mundo.

O MUNDO ESCOLHIDO, E POR QUE NAO E TRAPACA
--------------------------------------------
O simulador de particulas do `deducao.py`. Tres razoes para ele e nao para um
dominio que eu inventasse agora:

  1. Ele e ANTERIOR a tudo que foi construido hoje. Nenhuma leitura, nenhum
     pintor, nenhum operador foi desenhado olhando para ele. Inventar o segundo
     dominio agora seria construir algo que casa com o vocabulario por
     construcao — o erro classico.
  2. O mapa grade->grade e EMERGENTE. O proprio arquivo diz: "eu escrevo apenas
     regras LOCAIS... esse mapa grade->grade eu NAO escrevi e nao saberia
     escrever". Ninguem, nem eu, conhece a funcao alvo.
  3. Ele ja tem LINHA DE BASE: com a arquitetura antiga, `deducao.py` deu +0.

E ele traz o controle pareado de graca: `pares(fisico=False, ...)` gera dinamica
ALEATORIA com o mesmo custo, mesmo numero de cenas, mesmo numero de passos. Se o
nucleo resolver fisica e aleatorio igualmente, o vocabulario nao esta capturando
estrutura — esta ajustando qualquer coisa, que e o modo de falha que matou o
`segmenta.py` (127 ajustavam o treino, 6 passavam).

E HA UM GRADIENTE EMBUTIDO
----------------------------
Cada episodio observa a grade depois de K passos de simulacao. K=1 e uma
aplicacao da regra local; K=5 e a regra iterada cinco vezes. O nucleo tem `ATE`
(ponto fixo) mas nao tem "exatamente K vezes" — entao a taxa de acerto por K
mede ate onde a composicao alcanca, sem eu precisar desenhar nada.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  X0  PRE-CONDICAO. Algum episodio tem que fechar. Zero nos dois bracos
      significa que o dominio esta fora de alcance e a medida nao existe.

  X1  fisico > aleatorio -> o vocabulario captura ESTRUTURA REAL, e as leituras
      construidas para o ARC valem num mundo que nao e o ARC. E a primeira
      evidencia de transferencia do projeto.
  X2  empate -> o vocabulario ajusta qualquer coisa; nao e generalidade, e
      capacidade de sobreajuste.
  X3  fisico < aleatorio -> algo esta invertido e a medida esta errada.

  X4  QUAIS testes e pintores aparecem. E a medida que o usuario pediu: se as
      leituras topologicas (`sovaco`, `ponta`, `anomalia`) aparecerem aqui, elas
      nao sao conceitos de grade — sao conceitos.

SEMENTE PRE-REGISTRADA: 20260909

MINHA CALIBRAGEM: cinquenta previsoes, DEZOITO certas, e as trinta e duas
erradas quase todas otimistas.

PREVISAO: X1, com taxa baixa e concentrada em K=1. Fisica de particula e local e
iterada, que e exatamente `vizinhanca` + `prop` + `ATE`; entao espero que o
braco fisico ganhe. Mas `ATE` roda ate o ponto fixo e o mundo roda K vezes, e
essa diferenca deve matar quase tudo com K>=2. Aposto em 5 a 15% em K=1 e perto
de zero acima disso — e o aleatorio perto de zero em todos.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

from novaphys.arc import dims
from novaphys import descricao as D
from novaphys import deducao as DD

SEMENTE = 20260909
EPS = None


def _ini():
    global EPS
    EPS = []
    for s_ in range(6):
        for fisico in (True, False):
            for k, obs in enumerate(DD.pares(fisico, 8, 4, SEMENTE + s_)):
                EPS.append((f"{'fis' if fisico else 'ale'}_k{k+1}_s{s_}",
                            fisico, k + 1, obs))


def _uma(ti):
    nome, fisico, k, obs = EPS[ti]
    try:
        tr, te = obs[:-2], obs[-2:]
        if len(tr) < 3:
            return nome, fisico, k, 0, ()
        fm, pilha, err = D.montar(tr, refoco=True, feixe=4)
        if err != 0 or not pilha:
            return nome, fisico, k, 0, ()
        ok = all(D.aplicar((fm, pilha), x) == y for x, y in te)
        pecas = []
        for x in pilha:
            if x in D.MARCAS:
                pecas.append(x)
            else:
                pecas.append(x[0].nome)
                if x[0].nome == "programa":
                    sel, pin = x[1]
                    pecas += [f"T:{c[0]}" for c in sel] + [f"P:{pin[0]}"]
        return nome, fisico, k, 1 if ok else 0, tuple(pecas)
    except Exception:
        return nome, fisico, k, 0, ()


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=3)
    a = ap.parse_args()
    _ini()

    print("=" * 76)
    print("DOMINIO 2 — as mesmas leituras valem fora do ARC?")
    print("=" * 76)
    print(f"  simulador de particulas do `deducao.py`, semente {SEMENTE}")
    print(f"  {len(EPS)} episodios; controle pareado = dinamica aleatoria")
    t0 = time.time()
    with Pool(a.nucleos, initializer=_ini) as pool:
        res = pool.map(_uma, range(len(EPS)), chunksize=1)

    fis = [(k, ok, p) for _n, f, k, ok, p in res if f]
    ale = [(k, ok, p) for _n, f, k, ok, p in res if not f]
    print(f"\n  ({time.time()-t0:.0f}s)\n")
    print("    K   fisico   aleatorio")
    print("   ---  ------   ---------")
    for k in sorted({k for k, _o, _p in fis}):
        nf = sum(o for kk, o, _p in fis if kk == k)
        tf = sum(1 for kk, _o, _p in fis if kk == k)
        na = sum(o for kk, o, _p in ale if kk == k)
        ta = sum(1 for kk, _o, _p in ale if kk == k)
        print(f"   {k:3d}  {nf:2d}/{tf:<3d}   {na:2d}/{ta:<3d}")
    F = sum(o for _k, o, _p in fis)
    A = sum(o for _k, o, _p in ale)
    print(f"\n  TOTAL  fisico {F}/{len(fis)}   aleatorio {A}/{len(ale)}")

    uso = Counter(x for _k, o, p in fis if o for x in p)
    print(f"\n  X4 . o que apareceu nas solucoes fisicas:")
    for x, n in uso.most_common(14):
        print(f"       {x:22s} {n}")

    print("")
    print("=" * 76)
    if F == 0 and A == 0:
        print("X0 FALHOU: dominio fora de alcance; a medida nao existe.")
    elif F > A:
        print("X1 . o vocabulario captura ESTRUTURA REAL. As leituras feitas")
        print("     para o ARC valem num mundo que nao e o ARC — primeira")
        print("     evidencia de transferencia do projeto.")
    elif F == A:
        print("X2 . empate: o vocabulario ajusta qualquer coisa. Nao e")
        print("     generalidade, e capacidade de sobreajuste.")
    else:
        print("X3 . invertido; a medida esta errada.")


if __name__ == "__main__":
    main()
