"""
novaphys.biblioteca — o ganho da analogia escala com o tamanho da biblioteca?
==============================================================================

DE ONDE VEM
-------------
`analogia.py` mostrou o controle mais limpo da sessao: esquemas abstraidos de
solucoes REAIS resolvem 4/400, esquemas ALEATORIOS de mesma forma e quantidade
resolvem ZERO, gastando MAIS orcamento. Transferencia de estrutura entre
tarefas e real.

Mas a biblioteca tinha 14 solucoes, colhidas com o motor mais fraco do projeto
(deducao de um passo por celula, 3/400 na avaliacao). E o portao A0 falhou por
isso — 14 abaixo do minimo pre-registrado de 15.

A ideia do usuario: alimentar a analogia com mais solucoes.

A PERGUNTA CERTA NAO E "MAIS AJUDA?"
--------------------------------------
E "o ganho ESCALA com o tamanho da biblioteca, ou satura?" — e ha uma razao
medida para desconfiar que satura ou piora.

`segmenta.py` mediu o custo de ampliar o espaco de hipoteses: 8 lentes x 10
propriedades = 80 familias, 127 tarefas ajustando o treino e 6 passando no
teste. Precisao 4,7%, sobreajuste 21x. **Ampliar o espaco compra ajuste.**

E os esquemas tem reuso de 1,17 (12 esquemas distintos para 14 solucoes). Se
cada solucao nova traz um esquema novo, entao dobrar a biblioteca dobra o
espaco de busca — que e exatamente o movimento que rendeu 21x de sobreajuste.

Entao mede-se a CURVA, nao um ponto: bibliotecas de tamanho 4, 8, 16, 32 e
todas, cada uma com seu controle aleatorio de mesmo tamanho.

DE ONDE VEM MAIS SOLUCAO
--------------------------
Do motor mais forte que existe na representacao por celula: vocabulario
quantificado (leituras sobre objetos e raios) com composicao de DOIS passos e
cobertura parcial em 0,85 — o `pilha.py`, que faz 11/400 na avaliacao contra
3/400 do motor de um passo usado antes.

E dois passos rendem esquemas de PAR, que e onde o argumento de teletransporte
morde de verdade: um par de esquemas instanciado aterrissa em profundidade 4
sem percorrer o caminho.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  B0  PRE-CONDICAO. A colheita tem que render MAIS que as 14 anteriores. Se
      nao render, nao ha o que medir e o portao A0 continua falhando.

  B1  o ganho CRESCE monotonicamente com o tamanho da biblioteca, e a
      biblioteca cheia bate +4
      -> alimentar a analogia e o caminho; escala
  B2  o ganho SATURA (cresce e para)
      -> a analogia entrega o que tem para entregar com poucas solucoes;
         alimentar mais nao adianta
  B3  o ganho CAI com biblioteca grande
      -> ampliar compra sobreajuste, exatamente como as 8 lentes; e a
         analogia sofre da mesma doenca

MINHA CALIBRAGEM: doze previsoes hoje, QUATRO certas, e as oito erradas todas
otimistas.

PREVISAO: B3, e desta vez o argumento nao e otimista nem pessimista — e
aritmetico. Reuso de 1,17 significa que biblioteca maior e quase inteiramente
espaco de busca maior, e espaco maior contra 2-4 pares de treino foi medido
como 21x de sobreajuste. Espero o ganho subir de 4 para 6-8 ate uns 16
esquemas e depois CAIR, com a precisao (certas/ajustadas) caindo junto.

Se der B3, a leitura fecha a linha de biblioteca com uma condicao util: a
analogia so vale enquanto a biblioteca for PEQUENA E CERTA. Alimentar mais nao
e a resposta — a resposta seria alimentar MELHOR, e para isso e preciso saber
qual esquema vale, que e o mesmo problema de selecao do `ockham.py` uma volta
acima.
"""

from __future__ import annotations

import argparse
import io
import random
import time
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from novaphys.analogia import (PASTA_TREINO, busca_esquema, esquema,
                               forma_leitura, forma_teste)
from novaphys.arc import carregar, dims
from novaphys.eter import PASTA_AVAL
from novaphys.eter_inferencia import todas_leituras, todos_testes
from novaphys.ordem1 import aplicar, inferir
from novaphys.ordem2 import parciais
from novaphys.pilha import vocabulario


def _colher(args):
    """
    Colhe do motor FORTE: vocabulario quantificado, dois passos, cobertura
    parcial. Devolve os esquemas de cada passo da solucao.
    """
    nome, tr, te = args
    try:
        L, T = vocabulario(tr, True)
        op, _g = inferir(tr, L, T)
        if op is not None:
            if all(aplicar(op, x) == y for x, y in te):
                return [esquema(op)]
            return []
        cand, _g2 = parciais(tr, L, T, 0.85, 40)
        for p in cand:
            meio, bom = [], True
            for a, b in tr:
                r = aplicar(p, a)
                if r is None or dims(r) != dims(b):
                    bom = False
                    break
                meio.append((r, b))
            if not bom:
                continue
            L2, T2 = vocabulario(meio, True)
            op2, _g3 = inferir(meio, L2, T2)
            if op2 is None:
                continue
            ok = True
            for x, y in te:
                r = aplicar(p, x)
                if r is None or aplicar(op2, r) != y:
                    ok = False
                    break
            if ok:
                return [esquema(p), esquema(op2)]
    except Exception:
        return []
    return []


def _uma_aval(args):
    nome, tr, te, esq, orc, passos = args
    try:
        cam, _g = busca_esquema(tr, te, esq, orc, passos)
    except Exception:
        return nome, False
    if cam is None:
        return nome, False
    try:
        for x, y in te:
            r = x
            for op in cam:
                r = aplicar(op, r)
                if r is None:
                    return nome, False
            if r != y:
                return nome, False
    except Exception:
        return nome, False
    return nome, True


def avaliar(tarefas, esq, orc, passos, nucleos) -> Set[str]:
    from multiprocessing import Pool
    with Pool(nucleos) as pool:
        res = pool.map(_uma_aval, [(n, tr, te, esq, orc, passos)
                                   for n, tr, te in tarefas], chunksize=2)
    return {n for n, o in res if o}


CACHE = "novaphys/_esquemas.json"


def colher(nucleos: int):
    """
    Colhe e salva INCREMENTALMENTE.

    Duas execucoes anteriores foram perdidas: a primeira estourou o Pool
    seguinte (WinError 5 — nesta maquina so ha UM Pool por processo), a segunda
    morreu com o processo. Cada perda custou 20 minutos de computacao ja feita.

    Com `imap_unordered` e gravacao a cada resultado, uma interrupcao custa no
    maximo a tarefa em andamento. E o mesmo principio da sessao inteira, agora
    aplicado a mim: nao refazer trabalho ja determinado.
    """
    import json
    from multiprocessing import Pool
    treino = carregar(PASTA_TREINO)
    t0 = time.time()
    esq, nsol, feitas = [], 0, 0
    with Pool(nucleos) as pool:
        it = pool.imap_unordered(_colher,
                                 [(n, tr, te) for n, tr, te in treino],
                                 chunksize=1)
        for lst in it:
            feitas += 1
            if lst:
                nsol += 1
                esq.extend(list(e) for e in lst)
            if feitas % 25 == 0 or feitas == len(treino):
                with io.open(CACHE, "w", encoding="utf-8") as f:
                    json.dump({"esq": esq, "nsol": nsol, "feitas": feitas}, f)
                print(f"  {feitas}/{len(treino)} tarefas, {nsol} solucoes, "
                      f"{len(esq)} esquemas ({time.time()-t0:.0f}s)",
                      flush=True)
    d = {tuple(e) for e in esq}
    print(f"\nB0 · {nsol} solucoes (antes: 14), {len(esq)} esquemas, "
          f"{len(d)} distintos, reuso {len(esq)/max(len(d),1):.2f}")


def main() -> None:
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--colher", action="store_true")
    ap.add_argument("--tam", type=int, default=0)
    ap.add_argument("--braco", default="restr")
    ap.add_argument("--orcamento", type=int, default=14424)
    ap.add_argument("--passos", type=int, default=3)
    ap.add_argument("--nucleos", type=int, default=10)
    ap.add_argument("--semente", type=int, default=20260908)
    a = ap.parse_args()

    if a.colher:
        colher(a.nucleos)
        return

    with io.open(CACHE, encoding="utf-8") as f:
        dados = json.load(f)
    reais = sorted({tuple(e) for e in dados["esq"]})
    rng = random.Random(a.semente)
    t = a.tam if a.tam else len(reais)
    sub = set(rng.sample(reais, t)) if t < len(reais) else set(reais)

    if a.braco == "gener":
        # ANALOGIA GENERATIVA: as pecas RECOMBINAM entre esquemas. O teste de
        # um esquema com as leituras de outro produz uma tripla que NENHUMA
        # tarefa exibiu. A restritiva so instancia formas ja observadas.
        alvo = {(x[0], y[1], z[2]) for x in sub for y in sub for z in sub}
    elif a.braco == "aleat":
        cores = list(range(10))
        tl, tt = todas_leituras(cores), todos_testes(cores)
        alvo = set()
        while len(alvo) < len(sub):
            alvo.add((forma_teste(rng.choice(tt)),
                      forma_leitura(rng.choice(tl)),
                      forma_leitura(rng.choice(tl))))
    else:
        alvo = sub

    tarefas = carregar(PASTA_AVAL)
    t0 = time.time()
    ok = avaliar(tarefas, alvo, a.orcamento, a.passos, a.nucleos)
    print(f"{a.braco:6s} tam={t:>3} esquemas={len(alvo):>6} "
          f"{len(ok):>3}/400 ({time.time()-t0:.0f}s) {sorted(ok)}")


if __name__ == "__main__":
    main()
