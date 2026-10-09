"""
novaphys.quantificado — o quantificador no objetivo, e o que ele custa
========================================================================

DE ONDE VEM
-------------
`xadrez.py` mediu o degrau: nenhuma funcao de 13 features estaticas passa de
95,06% ao definir xeque-mate, porque 66 de 668 posicoes tem vetor identico e
rotulo diferente. Um atomo quantificado — "para todo movimento, e ilegal" —
zera as colisoes, chega a 100% e TRANSFERE perfeito para Rei+Dama, um jogo
nunca visto, enquanto a melhor regra proposicional cai 9,4 pontos.

Aqui a lacuna e de outra natureza, e dizer isso com precisao e o que separa
medida de propaganda: os atomos de `objetivo.py` incluem `pos=(i,j)`, entao o
estado ja esta totalmente determinado e NAO existe colisao. Neste mundo o
quantificador nao compra expressividade.

Compra TRANSFERENCIA. "Beco sem saida" vale em qualquer labirinto. "pos=(3,5)"
nao vale em nenhum outro. E transferencia e exatamente o que separa um sistema
que decorou de um que entendeu.

O MUNDO
---------
Labirinto 7x7 com paredes sorteadas, dinamica escondida, objetivo escondido.
O objetivo verdadeiro e BECO COM TODAS AS CHAVES:

    saidas(pos) == 1   E   bolso >= n_chaves

Mesma forma do mate: um conjunto quantificado sobre acoes mais um proposicional
sobre o estado. Natural e fisico — a escotilha de fuga so abre com as chaves.

    L1  atomos proposicionais de `objetivo.py` + conjuncoes de 2
    L2  os mesmos + `saidas==k` e `saidas<=1` + conjuncoes de 2

O CUSTO, que e a parte honesta
--------------------------------
O agente avalia `saidas` pelo MODELO QUE APRENDEU, nunca pelo labirinto
verdadeiro. Enquanto uma das quatro vizinhas for lacuna, o atomo devolve None
e a hipotese nao pode ser eliminada nem confirmada. O quantificador so fica
avaliavel depois de o agente ter tentado as quatro acoes daquela celula.

Isso e o oposto do erro de `distante.py`, onde o planejador recebia `_prox()`,
a funcao de transicao verdadeira, de graca — e eu descrevi como "modelo que
aprendeu". Aqui nao ha nada de graca: o quantificador se paga em exploracao, e
o preco entra na mesma moeda, acoes tomadas.

A MEDIDA
----------
Treina em N labirintos. Para cada um, pega as hipoteses sobreviventes e as
aplica num labirinto NOVO, com paredes diferentes, contra o objetivo
verdadeiro de la. Acuracia BALANCEADA (media de acerto nos vitoriosos e nos
nao-vitoriosos), porque beco e raro e "sempre falso" ja daria 90% na acuracia
crua.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  Q0  PRE-CONDICAO. O objetivo tem que ser alcancavel: cada labirinto precisa
      ter pelo menos um estado vitorioso alcancavel, e o agente precisa vencer
      pelo menos uma vez. Objetivo nunca atingido nao testa inferencia —
      foi o que a auditoria pegou em `objetivo.py` com a chave murada.

  Q1  L2 transfere >= 90% balanceado  e  L1 transfere <= 65%
      -> o quantificador e o que faz o objetivo generalizar. Decorar a
         posicao ajusta o mundo de treino e nao sobrevive a troca.

  Q2  os dois transferem alto
      -> a gramatica proposicional generaliza sozinha; o quantificador e
         conveniencia. Improvavel, mas se der, e o resultado.

  Q3  os dois transferem baixo
      -> o problema nao e a ordem logica da linguagem, e a inferencia nao
         chegou perto do objetivo verdadeiro em nenhum dos dois.

MINHA PREVISAO, escrita antes de rodar: Q1. Espero L2 acima de 95% e L1 perto
de 50%, que e o acaso na balanceada. Espero tambem que L1 muitas vezes fique
com ZERO sobreviventes — o objetivo e verdadeiro em varios becos espalhados e
nenhuma conjuncao de DOIS atomos posicionais cobre tres celulas dispersas. Se
isso acontecer, a linguagem se refuta sozinha, o que e um sinal limpo e nao um
bug: eliminacao pura chegando ao conjunto vazio significa "nao esta aqui".

E espero que L2 gaste MAIS acoes ate convergir, porque precisa sondar as
quatro direcoes de cada celula candidata antes de o atomo virar avaliavel.
Se L2 nao custar mais caro, desconfiar: e sinal de que o atomo esta sendo
avaliado com informacao que o agente nao tinha.
"""

from __future__ import annotations

import argparse
import random
from typing import Callable, Dict, List, Optional, Set, Tuple

from .objetivo import (ACOES, LADO, Modelo, Mundo, acao_novidade, atomos,
                       atomos_quantificados)


# ══════════════════════════════════════════════ o mundo com objetivo quantificado
class MundoBeco(Mundo):
    """
    Vitoria = beco sem saida COM todas as chaves. A contagem de saidas usa as
    paredes VERDADEIRAS — e a regra do mundo, nao o que o agente sabe.
    """

    def saidas_reais(self, pos) -> int:
        n = 0
        for dy, dx in ACOES:
            a = (pos[0] + dy, pos[1] + dx)
            if not (0 <= a[0] < LADO and 0 <= a[1] < LADO):
                continue
            if a in self.paredes:
                continue
            n += 1
        return n

    def venceu(self, e: Optional[tuple] = None) -> bool:
        pos, _ch, bolso = e if e else self.estado()
        return self.saidas_reais(pos) == 1 and bolso >= self.n_chaves


class ModeloPleno(Modelo):
    """Modelo completo de um labirinto — so para AVALIAR transferencia."""

    def __init__(self, m: MundoBeco) -> None:
        super().__init__()
        for i in range(LADO):
            for j in range(LADO):
                (self.parede if (i, j) in m.paredes else self.livre).add((i, j))


# ══════════════════════════════════════════════ as duas linguagens
def _prop(f) -> Callable:
    return lambda e, _mod, f=f: f(e)


def linguagem(quant: bool, n_chaves: int) -> List[Tuple[str, Callable]]:
    at = [(n, _prop(f)) for n, f in atomos(n_chaves)]
    if quant:
        at = at + atomos_quantificados()
    fora = list(at)
    for i in range(len(at)):
        for j in range(i + 1, len(at)):
            fa, fb = at[i][1], at[j][1]

            def conj(e, mod, fa=fa, fb=fb):
                a = fa(e, mod)
                if a is False:
                    return False
                b = fb(e, mod)
                if b is False:
                    return False
                if a is None or b is None:
                    return None
                return True
            fora.append((f"{at[i][0]} & {at[j][0]}", conj))
    return fora


# ══════════════════════════════════════════════ treino e transferencia
def treinar(m, hip, max_acoes: int, semente: int, mod_pleno
            ) -> Tuple[List[int], Optional[int], int]:
    """
    Explora por novidade sobre o estado completo (o mecanismo que sobreviveu a
    controle em `objetivo.py`: 128 vitorias contra 5, p = 0,00001) e elimina
    hipoteses. Hipotese INDEFINIDA nao e eliminada — nao se refuta o que nao se
    consegue avaliar.

    REGUA DE CONVERGENCIA. A primeira versao contava sobreviventes ("<= 3") e
    nunca convergia. E o mesmo erro que a auditoria de `objetivo.py` ja tinha
    pego uma vez: a classe de hipoteses LOGICAMENTE IGUAIS ao objetivo tem
    muitos membros, o agente encontra a classe inteira, e contar diz "nao
    convergiu". O criterio certo e por ASSINATURA — todas as sobreviventes
    respondendo igual em todo estado alcancavel. Calculado a cada 25 acoes,
    so quando o conjunto ja esta pequeno, para nao dominar o custo.
    """
    rng = random.Random(semente)
    mod = Modelo()
    cont: Dict[tuple, int] = {}
    vivos = list(range(len(hip)))
    convergiu: Optional[int] = None
    vitorias = 0
    est = m.estados_alcancaveis()
    m.reiniciar()
    for t in range(1, max_acoes + 1):
        e0 = m.estado()
        a = acao_novidade(mod, e0, cont, rng)
        e1 = m.passo(a)
        mod.aprender(e0, a, e1)
        cont[e1] = cont.get(e1, 0) + 1
        v = m.venceu(e1)
        vitorias += int(v)
        vivos = [k for k in vivos
                 if (r := hip[k][1](e1, mod)) is None or r == v]
        if convergiu is None and vivos and len(vivos) <= 60 and t % 25 == 0:
            assin = {tuple(hip[k][1](e, mod_pleno) for e in est)
                     for k in vivos}
            if len(assin) == 1:
                convergiu = t
        if v:
            m.reiniciar()
    return vivos, convergiu, vitorias


def transferir(hip, vivos: List[int], novo: MundoBeco) -> Optional[float]:
    """
    Acuracia BALANCEADA das sobreviventes num labirinto que elas nunca viram,
    com modelo completo do novo mundo — aqui se testa a REGRA, nao a
    exploracao. Balanceada porque beco e raro e "sempre falso" daria ~90% na
    acuracia crua.
    """
    if not vivos:
        return None
    mod = ModeloPleno(novo)
    est = novo.estados_alcancaveis()
    ver = [novo.venceu(e) for e in est]
    pos = [i for i, y in enumerate(ver) if y]
    neg = [i for i, y in enumerate(ver) if not y]
    if not pos or not neg:
        return None
    notas = []
    for k in vivos:
        f = hip[k][1]
        tp = sum(1 for i in pos if f(est[i], mod) is True) / len(pos)
        tn = sum(1 for i in neg if f(est[i], mod) is not True) / len(neg)
        notas.append((tp + tn) / 2)
    return sum(notas) / len(notas)


class MundoPorta(Mundo):
    """
    Objetivo PROPOSICIONAL: chegar na porta com todas as chaves. As duas
    linguagens expressam este. Serve para isolar o CUSTO de carregar o
    quantificador — se L2 converge mais devagar aqui, o preco e real.
    """

    def venceu(self, e=None) -> bool:
        pos, _ch, bolso = e if e else self.estado()
        return pos == self.porta and bolso >= self.n_chaves


def condicao(rotulo: str, classe, l1, l2, a) -> None:
    print("\n" + "=" * 74)
    print(rotulo)
    print("-" * 74)
    res: Dict[str, list] = {"L1": [], "L2": []}
    conv: Dict[str, list] = {"L1": [], "L2": []}
    vazios = {"L1": 0, "L2": 0}
    nconv = {"L1": 0, "L2": 0}
    usados, s = 0, a.semente
    while usados < a.mundos and s < a.semente + a.mundos * 30:
        s += 1
        try:
            tre = classe(s, n_chaves=a.chaves)
            tes = classe(s + 9973, n_chaves=a.chaves)
        except ValueError:
            continue
        if not any(tre.venceu(e) for e in tre.estados_alcancaveis()):
            continue
        if not any(tes.venceu(e) for e in tes.estados_alcancaveis()):
            continue
        usados += 1
        pleno = ModeloPleno(tre)
        for rot, hip in (("L1", l1), ("L2", l2)):
            vivos, c, _v = treinar(tre, hip, a.acoes, s, pleno)
            if not vivos:
                vazios[rot] += 1
                continue
            if c is None:
                nconv[rot] += 1
            else:
                conv[rot].append(c)
            t = transferir(hip, vivos, tes)
            if t is not None:
                res[rot].append(t)
    if usados == 0:
        print("Q0 FALHOU: nenhum par com vitoria alcancavel.")
        return
    print(f"Q0 · {usados} pares com vitoria alcancavel nos dois lados\n")
    print(f"{'':4s} {'transf.balanceada':>18s} {'refutou':>9s} "
          f"{'nao convergiu':>14s} {'acoes ate convergir':>21s}")
    print("-" * 70)
    for rot in ("L1", "L2"):
        tr = f"{sum(res[rot]) / len(res[rot]):.1%}" if res[rot] else "—"
        cc = f"{sum(conv[rot]) / len(conv[rot]):.0f}" if conv[rot] else "—"
        print(f"{rot:4s} {tr:>18s} {vazios[rot]:>6d}/{usados} "
              f"{nconv[rot]:>11d}/{usados} {cc:>21s}")
    print("-" * 70)
    m1 = sum(res["L1"]) / len(res["L1"]) if res["L1"] else 0.0
    m2 = sum(res["L2"]) / len(res["L2"]) if res["L2"] else 0.0
    if m2 >= 0.90 and m1 <= 0.65:
        print("Q1 · o quantificador e o que faz o objetivo GENERALIZAR.")
        print(f"     L2 transfere {m2:.1%} para um labirinto nunca visto;")
        print(f"     L1 fica em {m1:.1%}.")
    elif m1 > 0.65 and m2 >= 0.90:
        print("Q2 · os dois generalizam. Aqui o quantificador nao e o que")
        print("     decide — e o esperado neste objetivo, que e proposicional.")
    else:
        print("Q3 · nenhum dos dois chegou perto.")
    if vazios["L1"] > vazios["L2"]:
        print(f"     L1 se REFUTOU sozinha em {vazios['L1']}/{usados} mundos:")
        print("     eliminacao pura chegou ao vazio. A linguagem diz, sem")
        print("     ambiguidade, 'o objetivo nao esta em mim'.")
    if conv["L1"] and conv["L2"]:
        c1 = sum(conv["L1"]) / len(conv["L1"])
        c2 = sum(conv["L2"]) / len(conv["L2"])
        print(f"     CUSTO do quantificador: {c2 / c1:.2f}x acoes ate "
              f"convergir ({c1:.0f} -> {c2:.0f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mundos", type=int, default=25)
    ap.add_argument("--acoes", type=int, default=4000)
    ap.add_argument("--chaves", type=int, default=2)
    ap.add_argument("--semente", type=int, default=20260907)
    a = ap.parse_args()

    print("=" * 74)
    print("QUANTIFICADO — o quantificador no objetivo: o que compra e quanto custa")
    print("=" * 74)
    l1 = linguagem(False, a.chaves)
    l2 = linguagem(True, a.chaves)
    print(f"\nL1 proposicional   {len(l1)} hipoteses")
    print(f"L2 + quantificador {len(l2)} hipoteses")
    print(f"{a.mundos} pares de labirintos, {a.acoes} acoes, "
          f"novidade nos dois bracos (mesma moeda)")

    condicao(f"OBJETIVO QUANTIFICADO   saidas==1 E bolso>={a.chaves}",
             MundoBeco, l1, l2, a)
    condicao(f"OBJETIVO PROPOSICIONAL  pos==porta E bolso>={a.chaves}"
             "   (as duas expressam)", MundoPorta, l1, l2, a)


if __name__ == "__main__":
    main()
