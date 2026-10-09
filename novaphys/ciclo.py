"""
novaphys.ciclo — descobrir o objetivo e depois AGIR com ele
=============================================================

A SOLDA QUE FALTAVA
---------------------
A sessao mediu duas metades e nunca as ligou:

  METADE 1 — descobrir o alvo. `quantificado.py`: com quantificador na
  linguagem de objetivos, a regra descoberta num labirinto CLASSIFICA a
  99,3% num labirinto nunca visto; sem quantificador a linguagem se refuta
  sozinha em 24 de 25 mundos.

  METADE 2 — achar a regra dado o alvo. `residuo.py` e todo o motor de
  deducao: busca guiada pelo que ainda nao esta explicado.

Mas a metade 2 sempre recebeu o alvo DE GRACA — no ARC os pares de treino sao
a especificacao. Num laboratorio, ou num jogo, ninguem entrega o alvo celula a
celula: ha uma consequencia esparsa que chega depois.

Este arquivo liga as duas. O agente descobre o que conta como vitoria num
mundo, e usa essa regra descoberta como ESPECIFICACAO para agir noutro mundo.

O QUE ISTO TESTA E NINGUEM TESTOU
-----------------------------------
Que a regra descoberta CLASSIFICA bem ja esta medido. Que ela seja UTILIZAVEL
para agir, nao. Sao coisas diferentes e a diferenca importa:

    uma regra pode classificar 99,3% e ser inutil para planejar, se o agente
    nao consegue dizer qual acao chega mais perto dela

Um classificador diz "isto e vitoria" quando voce ja chegou. Uma
especificacao util diz para onde ir antes de chegar. `alcance.py` ja mostrou
que essa distincao morde: com modelo APRENDIDO (em vez de dado de graca) o
planejador nao conseguia expandir a busca e virava identico a novidade —
razao 1,0x.

O DESENHO
-----------
Mundo A: descobre-se o objetivo por eliminacao com exploracao por novidade,
na linguagem COM quantificador (`saidas==k`, avaliado pelo modelo APRENDIDO,
indefinido enquanto houver lacuna).

Mundo B, novo, mesmas regras: mede-se quantas ACOES ate a primeira vitoria
real, com tres agentes:

    CEGO         novidade pura sobre o estado completo — o mecanismo que
                 venceu por 128 a 5 em `objetivo.py`
    DESCOBERTO   usa a hipotese sobrevivente do mundo A como alvo: faz busca
                 em largura no MODELO QUE APRENDEU e, se alcancar um estado
                 que a hipotese diz ser vitorioso, vai ate la; senao, novidade
    VERDADEIRO   o mesmo, com o objetivo VERDADEIRO — o teto

O teto e a peca que faz o experimento valer. Sem ele, "o descoberto ajuda" nao
diz se ajuda pouco ou se ja e tudo que da para extrair da informacao.

A moeda e a mesma dos outros arquivos: ACOES TOMADAS. E o modelo do mundo B
comeca VAZIO nos tres bracos — nada de `_prox()` de graca, que foi o erro que
invalidou a leitura forte de `distante.py`.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  L0  PRE-CONDICAO. Os tres bracos precisam vencer em pelo menos 80% dos
      mundos dentro do orcamento, senao a mediana nao diz nada. E o braco
      VERDADEIRO tem que ser o mais rapido dos tres; se nao for, o
      planejamento esta quebrado e nada mais vale.

  L1  DESCOBERTO fica a menos de 1,5x do VERDADEIRO e bate o CEGO
      -> o ciclo fecha: descobrir -> especificar -> agir. A regra descoberta
         nao e so um classificador, e uma especificacao utilizavel.
  L2  DESCOBERTO entre o CEGO e o VERDADEIRO
      -> ajuda parcialmente; parte da informacao se perde na traducao de
         classificador para alvo
  L3  DESCOBERTO ~ CEGO
      -> a regra classifica e nao guia. Saber reconhecer a vitoria nao ajuda
         a alcanca-la, e a solda nao existe.

MINHA PREVISAO: L1, e desta vez com um argumento que nao e otimismo — a regra
descoberta e a classe de equivalencia FUNCIONAL do objetivo verdadeiro (foi
assim que a convergencia foi medida, por assinatura sobre todos os estados
alcancaveis). Se ela responde igual ao verdadeiro em todo estado alcancavel,
entao guiar por ela e guiar pelo verdadeiro, e a unica diferenca possivel vem
do custo de AVALIAR (o atomo quantificado exige as quatro vizinhas conhecidas).

Entao aposto L1 com DESCOBERTO entre 1,0x e 1,3x do VERDADEIRO.

E se der L3, e o achado mais importante da linha de agencia: significa que
transferencia de CLASSIFICACAO (os 99,3%) nao implica transferencia de
CONTROLE, e que a sessao inteira mediu a metade fácil do problema de objetivo.
"""

from __future__ import annotations

import argparse
import random
from collections import deque
from typing import Callable, Dict, List, Optional, Set, Tuple

from novaphys.objetivo import ACOES, LADO, Modelo, acao_novidade
from novaphys.objetivo import atomos, atomos_quantificados
from novaphys.quantificado import (MundoBeco, ModeloPleno, _prop, linguagem,
                                   treinar)


def partes(nome: str, atomos) -> List[Callable]:
    """
    Um objetivo desta gramatica e um atomo ou uma CONJUNCAO de dois, e o nome
    guarda isso: "saidas==1 & bolso>=2". Recuperar as partes permite pontuar
    satisfacao PARCIAL.
    """
    d = dict(atomos)
    return [d[p.strip()] for p in nome.split("&") if p.strip() in d]


def _plano(mod: Modelo, e0: tuple, alvo: Callable, prof: int
           ) -> Optional[int]:
    """
    Busca em largura no MODELO APRENDIDO. Devolve a primeira acao de um
    caminho ate um estado que `alvo` diz ser vitorioso, ou None se nenhum for
    alcancavel com o que se sabe hoje.

    Lacuna no modelo NAO e chutada: `prever` devolve None e o ramo morre.
    """
    fila = deque([(e0, None, 0)])
    visto = {e0}
    while fila:
        e, prim, d = fila.popleft()
        if d >= prof:
            continue
        for a in range(4):
            p = mod.prever(e, a)
            if p is None or p in visto:
                continue
            visto.add(p)
            pa = a if prim is None else prim
            try:
                if alvo(p, mod) is True:
                    return pa
            except Exception:
                pass
            fila.append((p, pa, d + 1))
    return None


def _gradiente(mod: Modelo, e0: tuple, conj: List[Callable], prof: int,
               cont: Dict[tuple, int]) -> Optional[int]:
    """
    CONSERTO, e ele e o achado do arquivo. A primeira versao perguntava "ha um
    estado VITORIOSO ao meu alcance?" — predicado binario. Com modelo cheio de
    lacunas quase nunca ha: o planejador disparou 5 vezes em 43 acoes com 2
    chaves, e ZERO vezes em 87 com 3. Classificador nao guia.

    Um objetivo em forma de conjuncao pode ser pontuado PARCIALMENTE: um
    estado que satisfaz um dos dois conjuntos esta mais perto que um que nao
    satisfaz nenhum. Coletar uma chave da credito mesmo longe do beco.

    E o mesmo principio de `relaxado.py` (contar celulas que o operador ainda
    nao explica) e de `residuo.py` (contar celulas que ainda diferem do alvo),
    aplicado ao objetivo: contar CONJUNTOS ainda nao satisfeitos. Exigir
    satisfacao total antes de agir e o mesmo erro que exigir cobertura 100%,
    e aquele custou de 3 para 8 tarefas.
    """
    def nota(e) -> int:
        n = 0
        for f in conj:
            try:
                if f(e, mod) is True:
                    n += 1
            except Exception:
                pass
        return n

    # DESCONTO POR VISITA, e sem ele o mecanismo se auto-sabota. Medido: com
    # `bolso>=3 & saidas==1`, o agente ganha um ponto indo ao beco mais
    # proximo (`saidas==1` vale em QUALQUER beco, sem chave), chega la, nada
    # melhora, cai em novidade, sai, e ve o beco de novo. OSCILA, e falhou em
    # 6 de 6 mundos em 2.000 acoes — pior que cego.
    #
    # Credito parcial sem desconto por repeticao vira vicio. E o mesmo erro ja
    # medido em `distante.py`: o planejador sem desconto repetia o MESMO
    # experimento, 33 estados distintos contra 198, e terminava com MAIS
    # hipoteses vivas. Mesma correcao: dividir pela contagem de visitas.
    def pont(e) -> float:
        return nota(e) / (1.0 + cont.get(e, 0))

    base = pont(e0)
    fila = deque([(e0, None, 0)])
    visto = {e0}
    melhor, melhor_a = base, None
    while fila:
        e, prim, d = fila.popleft()
        if d >= prof:
            continue
        for a in range(4):
            p = mod.prever(e, a)
            if p is None or p in visto:
                continue
            visto.add(p)
            pa = a if prim is None else prim
            k = pont(p)
            if k > melhor:
                melhor, melhor_a = k, pa
            fila.append((p, pa, d + 1))
    return melhor_a


def episodio(m: MundoBeco, alvo: Optional[Callable], max_acoes: int,
             semente: int, prof: int, conj: Optional[List] = None
             ) -> Optional[int]:
    """
    Acoes ate a PRIMEIRA vitoria real. `alvo=None` e o braco cego (so
    novidade). O modelo comeca VAZIO — nada e dado de graca.
    """
    rng = random.Random(semente)
    mod = Modelo()
    cont: Dict[tuple, int] = {}
    m.reiniciar()
    for t in range(1, max_acoes + 1):
        e0 = m.estado()
        a = None
        if alvo is not None:
            a = _plano(mod, e0, alvo, prof)
        if a is None and conj:
            a = _gradiente(mod, e0, conj, prof, cont)
        if a is None:
            a = acao_novidade(mod, e0, cont, rng)
        e1 = m.passo(a)
        mod.aprender(e0, a, e1)
        cont[e1] = cont.get(e1, 0) + 1
        if m.venceu(e1):
            return t
    return None


def lista_atomos(chaves: int):
    return [(n, _prop(f)) for n, f in atomos(chaves)] + atomos_quantificados()


def descobrir(semente: int, chaves: int, acoes: int, hip):
    """
    Mundo A: descobre o objetivo por eliminacao. Devolve (funcao, conjuntos) —
    a funcao para reconhecer, e as PARTES para pontuar satisfacao parcial.
    """
    tre = MundoBeco(semente, n_chaves=chaves)
    if not any(tre.venceu(e) for e in tre.estados_alcancaveis()):
        return None, None, None
    vivos, _c, _v = treinar(tre, hip, acoes, semente, ModeloPleno(tre))
    if not vivos:
        return None, None, None
    nome, fn = hip[vivos[0]]
    return fn, partes(nome, lista_atomos(chaves)), nome


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mundos", type=int, default=20)
    ap.add_argument("--treino", type=int, default=3000)
    ap.add_argument("--acoes", type=int, default=3000)
    ap.add_argument("--prof", type=int, default=8)
    ap.add_argument("--chaves", type=int, default=2)
    ap.add_argument("--semente", type=int, default=20260907)
    a = ap.parse_args()

    print("=" * 76)
    print("CICLO — descobrir o objetivo e depois AGIR com ele")
    print("=" * 76)
    hip = linguagem(True, a.chaves)
    print(f"\nlinguagem COM quantificador: {len(hip)} hipoteses")
    print(f"mundo A descobre; mundo B (novo) mede acoes ate a 1a vitoria")
    print("modelo do mundo B comeca VAZIO nos tres bracos\n")

    BRACOS = ["CEGO", "DESCOB.parcial", "DESCOB.binario", "VERDADEIRO"]
    res: Dict[str, List[int]] = {b: [] for b in BRACOS}
    falhas = {b: 0 for b in BRACOS}
    usados, s = 0, a.semente
    while usados < a.mundos and s < a.semente + a.mundos * 30:
        s += 1
        try:
            tes = MundoBeco(s + 9973, n_chaves=a.chaves)
        except ValueError:
            continue
        if not any(tes.venceu(e) for e in tes.estados_alcancaveis()):
            continue
        f, conj, nome = descobrir(s, a.chaves, a.treino, hip)
        if f is None:
            continue
        if usados == 0:
            print(f'    exemplo de objetivo descoberto: {nome}')
        usados += 1

        def verdadeiro(e, _mod, w=tes):
            return w.venceu(e)

        # QUATRO bracos. O DESCOBERTO entra em duas formas porque a
        # instrumentacao mostrou que a diferenca entre elas e o achado:
        # com gradiente parcial o agente coletou UMA chave em 2.000 acoes e
        # visitou 17 celulas de ~49 — o conjunto barato (`saidas==1`, vale em
        # qualquer beco) rouba metade das acoes e sufoca a exploracao, que e
        # justamente o que coleta chaves.
        vconj = [lambda e, m, w=tes: w.venceu(e)]
        for rot, alvo, cj in (("CEGO", None, None),
                              ("DESCOB.parcial", f, conj),
                              ("DESCOB.binario", f, None),
                              ("VERDADEIRO", verdadeiro, vconj)):
            r = episodio(tes, alvo, a.acoes, s, a.prof, cj)
            if r is None:
                falhas[rot] += 1
            else:
                res[rot].append(r)

    if usados == 0:
        print("L0 FALHOU: nenhum par de mundos utilizavel.")
        return
    print(f"L0 · {usados} pares de mundos\n")
    print(f"{'braco':16s} {'mediana':>9s} {'media':>9s} {'falhou':>8s}")
    print("-" * 42)
    med = {}
    for rot in BRACOS:
        v = sorted(res[rot])
        if not v:
            print(f"{rot:16s} {'—':>9s} {'—':>9s} "
                  f"{falhas[rot]:>5d}/{usados}")
            med[rot] = a.acoes
            continue
        med[rot] = v[len(v) // 2]
        print(f"{rot:16s} {med[rot]:>9d} {sum(v)/len(v):>9.0f} "
              f"{falhas[rot]:>5d}/{usados}")
    print("-" * 42)

    c, d, vd = (med["CEGO"], med["DESCOB.binario"], med["VERDADEIRO"])
    print(f"\nCEGO / DESCOBERTO   = {c/max(d,1):.2f}x")
    print(f"DESCOBERTO / VERDADEIRO = {d/max(vd,1):.2f}x   <- perda na traducao")

    print("\n" + "=" * 76)
    if vd > c:
        print("L0 FALHOU: o braco VERDADEIRO nao e o mais rapido. O")
        print("     planejamento esta quebrado; nada mais aqui vale.")
        return
    if d <= vd * 1.5 and d < c:
        print("L1 · o ciclo FECHA. A regra descoberta nao e so um")
        print("     classificador — e uma especificacao utilizavel para agir.")
        print("     Descobrir -> especificar -> agir, com o objetivo vindo de")
        print("     dentro em vez de ser entregue.")
    elif d < c:
        print("L2 · ajuda parcialmente. Parte da informacao se perde na")
        print("     traducao de classificador para alvo.")
    else:
        print("L3 · a regra CLASSIFICA e nao GUIA. Transferencia de")
        print("     classificacao (99,3%) NAO implica transferencia de")
        print("     controle — e a sessao mediu a metade facil do problema")
        print("     de objetivo.")


if __name__ == "__main__":
    main()
