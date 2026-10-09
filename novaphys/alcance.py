"""
novaphys.alcance — a variavel que explica novidade E discriminacao
====================================================================

O QUE ESTE ARQUIVO TESTA
--------------------------
Dois resultados da sessao pareciam contraditorios:

    `distante.py`   discriminacao bate novidade por 30 a 54x
    `objetivo.py`   discriminacao empata com novidade (55 contra 57, p=0,61)

E ha um fato demonstrado que estreita as explicacoes possiveis: a
informatividade de um estado fixo, medida como min(v, n) entre as hipoteses
sobreviventes, e MONOTONA NAO-CRESCENTE sob eliminacao. Eliminar so remove
hipoteses, entao v e n so caem, e min(v,n) nunca sobe. Verificado por forca
bruta em 200.000 casos: zero aumentos.

Consequencia: um estado hoje sem informacao nunca passa a ter. Valor
informacional prospectivo NAO pode vir de hipoteses ficarem mais diferentes.

Sobra uma unica fonte possivel: estados ficarem ALCANCAVEIS.

    NOVIDADE        expande o alcancavel       -> cria informacao futura
    DISCRIMINACAO   explora o alcancavel atual -> extrai informacao presente

E isso preve a diferenca entre os dois mundos:

    `distante.py`   grade aberta, tudo alcancavel desde o inicio
                    -> novidade nao tem o que expandir, discriminacao ganha
    `objetivo.py`   alcancabilidade cresce com as chaves
                    -> novidade e essencial, discriminacao sozinha empata

O TESTE
---------
Mesmo mundo de `distante.py`, com UMA mudanca: uma porta que so abre com a
chave. A celula decisiva fica atras dela. Tudo o mais identico — mesma grade,
mesmas hipoteses, mesma distancia, mesmos bracos.

Se a explicacao estiver certa, a vantagem do discriminador desaba, porque a
celula decisiva deixa de existir no modelo dele ate a chave ser pega — e ele
nao tem motivo para pegar a chave, ja que pegar chave nao separa hipotese
nenhuma.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  A0  PRE-CONDICAO. Sem a porta, os numeros tem que reproduzir `distante.py`
      (discriminador ~10x mais rapido que novidade). Se nao reproduzir, este
      arquivo diverge do anterior e a comparacao nao vale.

  A1  COM porta, a vantagem do discriminador cai pelo menos pela metade
      -> a alcancabilidade e a variavel que explica os dois mundos

  A2  a vantagem se mantem
      -> a explicacao esta errada; a diferenca entre os mundos e outra coisa

  A3  o discriminador melhora com a porta
      -> nao tenho hipotese para isso

MINHA PREVISAO, escrita antes de rodar: A1, e forte — espero que o
discriminador chegue a NAO CONVERGIR com a porta, porque pegar a chave tem
valor informacional zero para ele e ele nunca vai ter motivo de ir la. A
novidade deve sofrer bem menos, porque a chave abre estados novos e e isso
que ela persegue.

Se A1 der certo, a sintese e que os dois mecanismos nao competem: um cria o
espaco que o outro explora. E ai a pergunta seguinte deixa de ser "qual e
melhor" e passa a ser "quando trocar de um para o outro".
"""

from __future__ import annotations

import argparse
import random
from collections import deque
from typing import Callable, Dict, List, Optional, Tuple

LADO = 30
ACOES = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class Mundo:
    """
    Coluna esquerda vence. A celula (d,0) e o unico ponto que discrimina.

    Com `porta=True`, a metade esquerda so e acessivel depois de pegar a
    chave — que fica no meio do caminho e nao separa hipotese nenhuma.
    """

    def __init__(self, d: int, porta: bool):
        self.d = d
        self.porta = porta
        self.chave = (LADO // 2, LADO // 2)
        self.col_porta = LADO // 3
        self.inicio = (0, LADO - 1)
        self.reiniciar()

    def reiniciar(self) -> None:
        self.pos = self.inicio
        self.tem = 0

    def estado(self) -> tuple:
        return (self.pos, self.tem)

    def passo(self, a: int) -> tuple:
        dy, dx = ACOES[a]
        n = (self.pos[0] + dy, self.pos[1] + dx)
        if not (0 <= n[0] < LADO and 0 <= n[1] < LADO):
            return self.estado()
        if self.porta and n[1] < self.col_porta and not self.tem:
            return self.estado()             # bloqueado sem a chave
        self.pos = n
        if self.porta and self.pos == self.chave:
            self.tem = 1
        return self.estado()

    def venceu(self, e: Optional[tuple] = None) -> bool:
        p = (e if e else self.estado())[0]
        return p[1] == 0


def hipoteses(d: int) -> List[Tuple[str, Callable]]:
    """Verdadeira, uma rival que so (d,0) separa, e engodos."""
    fora: List[Tuple[str, Callable]] = [
        ("coluna==0", lambda e: e[0][1] == 0),
        (f"coluna==0 e linha!={d}",
         lambda e, d=d: e[0][1] == 0 and e[0][0] != d),
    ]
    for c in range(1, LADO):
        fora.append((f"coluna=={c}", lambda e, c=c: e[0][1] == c))
        fora.append((f"linha=={c}", lambda e, c=c: e[0][0] == c))
    fora.append(("borda", lambda e: e[0][0] in (0, LADO - 1)
                 or e[0][1] in (0, LADO - 1)))
    return fora


class Modelo:
    """Transicoes deduzidas. Nao inventa: estado nao visto e lacuna."""

    def __init__(self) -> None:
        self.tab: Dict[Tuple[tuple, int], tuple] = {}

    def aprender(self, e0, a, e1) -> None:
        self.tab[(e0, a)] = e1

    def prever(self, e0, a) -> Optional[tuple]:
        return self.tab.get((e0, a))


def esc_acaso(e, cont, vivos, hip, rng, mod) -> int:
    return rng.randrange(4)


def esc_novidade(e, cont, vivos, hip, rng, mod) -> int:
    melhor, c = None, []
    for a in range(4):
        p = mod.prever(e, a)
        pont = -1 if p is None else cont.get(p, 0)
        if melhor is None or pont < melhor:
            melhor, c = pont, [a]
        elif pont == melhor:
            c.append(a)
    return rng.choice(c)


def esc_planejada(e, cont, vivos, hip, rng, mod) -> int:
    if len(vivos) < 2:
        return esc_novidade(e, cont, vivos, hip, rng, mod)
    fila = deque([(e, None, 0)])
    visto = {e}
    melhor, melhor_a = 0.0, None
    while fila:
        est, prim, prof = fila.popleft()
        if prof > LADO * 2 or len(visto) > 6000:
            break
        if prim is not None:
            v = sum(1 for k in vivos if hip[k][1](est))
            corte = min(v, len(vivos) - v) / (1 + cont.get(est, 0))
            if corte > melhor:
                melhor, melhor_a = corte, prim
        for a in range(4):
            p = mod.prever(est, a)
            if p is None or p in visto:
                continue
            visto.add(p)
            fila.append((p, a if prim is None else prim, prof + 1))
    if melhor_a is None:
        return esc_novidade(e, cont, vivos, hip, rng, mod)
    return melhor_a


EXPLORADORES = {"acaso": esc_acaso, "novidade": esc_novidade,
                "planejada": esc_planejada}


def rodar(d: int, porta: bool, explorador: str, max_acoes: int,
          semente: int) -> Optional[int]:
    rng = random.Random(semente)
    m = Mundo(d, porta)
    hip = hipoteses(d)
    est = [((i, j), t) for i in range(LADO) for j in range(LADO)
           for t in (0, 1)]
    assin = [tuple(f(e) for e in est) for _n, f in hip]
    vivos = list(range(len(hip)))
    cont: Dict[tuple, int] = {}
    mod = Modelo()
    esc = EXPLORADORES[explorador]

    for t in range(1, max_acoes + 1):
        e0 = m.estado()
        a = esc(e0, cont, vivos, hip, rng, mod)
        e1 = m.passo(a)
        mod.aprender(e0, a, e1)
        cont[e1] = cont.get(e1, 0) + 1
        v = m.venceu(e1)
        vivos = [k for k in vivos if hip[k][1](e1) == v]
        if vivos and len({assin[k] for k in vivos}) == 1:
            return t
        if v:
            m.reiniciar()
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sementes", type=int, default=15)
    ap.add_argument("--acoes", type=int, default=20000)
    ap.add_argument("--semente", type=int, default=20260905)
    a = ap.parse_args()

    print("=" * 68)
    print("ALCANCE — a alcancabilidade explica novidade E discriminacao?")
    print("=" * 68)
    print(f"grade {LADO}x{LADO}. Com porta, a metade esquerda so abre com a")
    print("chave — que fica no meio e nao separa hipotese nenhuma.\n")

    for porta in (False, True):
        rot = "COM PORTA (alcance cresce)" if porta else \
              "SEM PORTA (alcance fixo)"
        print(f"{rot}")
        print(f"{'d':>4} {'ACASO':>9} {'NOVIDADE':>10} {'PLANEJADA':>11} "
              f"{'nov/pla':>8}")
        print("-" * 46)
        razoes = []
        for d in (9, 27):
            med = {}
            for ex in ("acaso", "novidade", "planejada"):
                r = []
                for k in range(a.sementes):
                    x = rodar(d, porta, ex, a.acoes, a.semente + k)
                    r.append(x if x is not None else a.acoes)
                r.sort()
                med[ex] = r[len(r) // 2]
            rz = med["novidade"] / max(med["planejada"], 1)
            razoes.append(rz)
            print(f"{d:>4} {med['acaso']:>9} {med['novidade']:>10} "
                  f"{med['planejada']:>11} {rz:>7.1f}x", flush=True)
        print("-" * 46)
        media = sum(razoes) / len(razoes)
        print(f"vantagem media do planejador sobre novidade: {media:.1f}x\n")
        if not porta:
            sem_porta = media
        else:
            com_porta = media

    print("=" * 68)
    print(f"A0 · sem porta o planejador ganha {sem_porta:.1f}x "
          f"{'· reproduz distante.py' if sem_porta >= 5 else '· NAO reproduz'}")
    if sem_porta < 5:
        print("     Sem reproduzir o anterior, a comparacao nao vale.")
        return
    print(f"     com porta ganha {com_porta:.1f}x")
    if com_porta <= sem_porta / 2:
        print("\nA1 · a vantagem DESABA quando o alcance passa a crescer.")
        print("     A alcancabilidade e a variavel que explica os dois mundos:")
        print("     novidade cria o espaco, discriminacao o explora.")
    elif com_porta > sem_porta:
        print("\nA3 · o planejador MELHORA com a porta. Sem hipotese.")
    else:
        print("\nA2 · a vantagem se mantem. A explicacao esta errada.")


if __name__ == "__main__":
    main()
