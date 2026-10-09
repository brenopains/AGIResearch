"""
novaphys.descritor2 — o sistema inventa as proprias reguas
============================================================

DE ONDE VEM ESTE ARQUIVO
--------------------------
`descricao.py` mediu que uma regua ESTRUTURAL separa prefixo certo de prefixo
aleatorio (AUC 0,860) enquanto a regua de PIXEL fica no acaso (0,511). Se isso
sobreviver a ablacao, muda o quadro: os dois nulos de guia da sessao podem ter
sido bussola quebrada, nao paisagem plana.

Mas as 9 medidas daquele descritor fui EU que escolhi — contagem de objetos,
cores, massa, simetria. E a mesma critica de sempre: o gargalo so subiu um
andar. Um sistema que so enxerga pelas reguas que eu entreguei nao esta
descobrindo nada sobre o mundo, esta confirmando o meu gosto.

Aqui as reguas sao GERADAS e SELECIONADAS.

A META-GRAMATICA DE REGUAS
----------------------------
Uma regua e um numero extraido da grade:

    regua(g) = AGREGADOR( celulas que satisfazem PREDICADO )

    PREDICADO   cor==k | cor!=fundo | na borda | paridade de (i,j)
                | igual ao vizinho em D | diferente do vizinho em D
                | igual a celula refletida por M
    AGREGADOR   contagem | fracao | componentes conexos | linhas distintas
                | maior sequencia continua | linha/coluna com mais
                | valores distintos

Disso saem, sem eu nomear: contagem de objetos, massa, simetria horizontal,
periodicidade, densidade de borda, numero de cores — e centenas de medidas sem
nome, que e o ponto.

O SINAL QUE SELECIONA — e por que ele e o "expectativa x realidade"
---------------------------------------------------------------------
Uma regua sobrevive se ela SEPARA: prefixos do programa certo recebem valor
proximo do alvo, prefixos aleatorios nao. Isso e medido por AUC.

Esse criterio nao e estetico e nao e meu: e a diferenca entre o que a regua
prediz e o que o verificador confirma. Reguas que preveem ficam; reguas que
nao preveem morrem. E exatamente a forma computacional do que se descreve
como dopamina — erro de previsao sobre valor esperado —, so que sem a
palavra, e medida.

A ARMADILHA, E ELA E GRANDE AQUI
----------------------------------
So existem 9 tarefas do ARC cuja solucao tem 2 passos ou mais, ou seja 11
prefixos proprios. Escolher a melhor de ~1000 reguas usando 11 pontos e
sobreajuste de manual: a melhor regua vai parecer otima por acaso.

Por isso NADA e reportado dentro da amostra de selecao. A medicao e
validacao cruzada deixando uma tarefa de fora:

    seleciona as reguas nas 8 tarefas restantes
    mede a AUC na tarefa que ficou de fora, nunca usada na selecao
    repete para cada tarefa e reporta a media

E a unica leitura honesta possivel com esta amostra, e ela pode muito bem
mostrar que a selecao nao generaliza nada.

CONTROLE
----------
Um braco de reguas SORTEADAS sem selecao, em mesma quantidade. Se as
selecionadas nao baterem as sorteadas fora da amostra, a selecao nao esta
aprendendo — esta decorando.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  C0  PRE-CONDICAO. As reguas geradas tem que variar entre grades. Reguas
      constantes nao separam nada e inflariam o conjunto sem informar.
      Descartar as constantes ANTES de selecionar e reportar quantas sobraram.

  C1  AUC fora-da-amostra das reguas SELECIONADAS >= AUC das minhas 9 reguas
      escritas a mao, E > AUC das sorteadas sem selecao
      -> o sistema acha reguas melhores que as minhas sozinho

  C2  selecionadas ~ minhas 9
      -> empata; a geracao alcanca a escolha manual mas nao a supera

  C3  selecionadas < sorteadas, ou < minhas
      -> a selecao sobreajustou; com 11 pontos nao da para escolher regua

MINHA PREVISAO, escrita antes de rodar: espero C3. Onze pontos para escolher
entre centenas de candidatas e o cenario classico de sobreajuste, e a
validacao cruzada vai expor isso. Se C1 aparecer com esta amostra, minha
primeira reacao vai ser procurar vazamento entre selecao e teste, nao
comemorar.

O valor de rodar mesmo prevendo C3 e que o NUMERO diz quanta amostra faltaria
— e isso transforma "nao deu" em "precisa de tantas tarefas resolvidas", que
e uma meta, nao um beco.
"""

from __future__ import annotations

import argparse
import math
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from novaphys.arc import PRIMITIVAS, carregar, dims, fundo, resolver, valida
from novaphys.descricao import CHAVES, aplicar, auc, descrever

PASTA_TREINO = (('C:/Users/usuario/Documents/AGI 2.0/backup_novamind_20260627_170507/data/ARC/data/training').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))

Grid = Tuple[Tuple[int, ...], ...]
VIZ = [(0, 1), (1, 0), (1, 1), (1, -1)]
MATS = [(1, 0, 0, -1), (-1, 0, 0, 1), (0, 1, 1, 0), (-1, 0, 0, -1)]


# ══════════════════════════════════════════════ meta-gramatica de reguas
class Regua:
    """Um numero extraido da grade. Nenhum nome dado por mim."""

    def __init__(self, rng: random.Random):
        self.pred = rng.choice(["cor", "naofundo", "borda", "paridade",
                                "vizinho_ig", "vizinho_dif", "espelho"])
        if self.pred == "cor":
            self.k = rng.randrange(10)
        elif self.pred == "paridade":
            self.eixo = rng.choice(["i", "j", "i+j"])
            self.mod = rng.choice([2, 2, 3])
            self.r = rng.randrange(self.mod)
        elif self.pred in ("vizinho_ig", "vizinho_dif"):
            self.d = rng.choice(VIZ)
        elif self.pred == "espelho":
            self.m = rng.choice(MATS)
        self.agr = rng.choice(["contagem", "fracao", "componentes",
                               "linhas_dist", "maior_seq", "max_linha",
                               "valores_dist"])

    def _marca(self, g: Grid, h: int, w: int, bg: int) -> List[List[bool]]:
        p = self.pred
        m = [[False] * w for _ in range(h)]
        for i in range(h):
            for j in range(w):
                if p == "cor":
                    m[i][j] = g[i][j] == self.k
                elif p == "naofundo":
                    m[i][j] = g[i][j] != bg
                elif p == "borda":
                    m[i][j] = i in (0, h - 1) or j in (0, w - 1)
                elif p == "paridade":
                    v = {"i": i, "j": j, "i+j": i + j}[self.eixo]
                    m[i][j] = v % self.mod == self.r
                elif p in ("vizinho_ig", "vizinho_dif"):
                    dy, dx = self.d
                    ny, nx = i + dy, j + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        ig = g[i][j] == g[ny][nx]
                        m[i][j] = ig if p == "vizinho_ig" else not ig
                else:
                    a, b, c, d = self.m
                    ny = a * i + b * j + (h - 1 if a < 0 or b < 0 else 0)
                    nx = c * i + d * j + (w - 1 if c < 0 or d < 0 else 0)
                    if 0 <= ny < h and 0 <= nx < w:
                        m[i][j] = g[i][j] == g[ny][nx]
        return m

    def valor(self, g: Grid) -> float:
        h, w = dims(g)
        if not h or not w:
            return 0.0
        bg = fundo(g)
        m = self._marca(g, h, w, bg)
        a = self.agr
        if a == "contagem":
            return float(sum(1 for l in m for x in l if x))
        if a == "fracao":
            return sum(1 for l in m for x in l if x) / (h * w)
        if a == "componentes":
            visto = [[False] * w for _ in range(h)]
            n = 0
            for i in range(h):
                for j in range(w):
                    if m[i][j] and not visto[i][j]:
                        n += 1
                        pilha = [(i, j)]
                        visto[i][j] = True
                        while pilha:
                            y, x = pilha.pop()
                            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                                ny, nx = y + dy, x + dx
                                if (0 <= ny < h and 0 <= nx < w
                                        and m[ny][nx] and not visto[ny][nx]):
                                    visto[ny][nx] = True
                                    pilha.append((ny, nx))
            return float(n)
        if a == "linhas_dist":
            return float(len({tuple(l) for l in m}))
        if a == "maior_seq":
            mx = 0
            for l in m:
                c = 0
                for x in l:
                    c = c + 1 if x else 0
                    mx = max(mx, c)
            return float(mx)
        if a == "max_linha":
            return float(max((sum(1 for x in l if x) for l in m), default=0))
        return float(len({(g[i][j] if m[i][j] else -1)
                          for i in range(h) for j in range(w)}))

    def __str__(self) -> str:
        p = self.pred
        if p == "cor":
            p = f"cor=={self.k}"
        elif p == "paridade":
            p = f"({self.eixo})%{self.mod}=={self.r}"
        elif p in ("vizinho_ig", "vizinho_dif"):
            p = f"{p}{self.d}"
        return f"{self.agr}({p})"


def gerar(n: int, semente: int) -> List[Regua]:
    rng = random.Random(semente)
    return [Regua(rng) for _ in range(n)]


# ══════════════════════════════════════════════ amostra
def coletar_pares(pasta: str, neg: int, semente: int
                  ) -> List[Tuple[str, List[Grid], List[Grid], Grid]]:
    """Por tarefa: (nome, prefixos corretos, prefixos aleatorios, alvo)."""
    rng = random.Random(semente)
    nomes = [n for n, _ in PRIMITIVAS]
    fora = []
    for nome, tr, te in carregar(pasta):
        p = resolver(tr, prof=3)
        if p is None or not all(p.aplicar(x) == y for x, y in te):
            continue
        if len(p.ops) < 2:
            continue
        ent, alvo = tr[0]
        pos, ng = [], []
        for k in range(1, len(p.ops)):
            r = aplicar(list(p.ops[:k]), ent)
            if r is None:
                break
            pos.append(r)
            for _ in range(neg):
                q = [rng.choice(nomes) for _ in range(k)]
                if list(q) == list(p.ops[:k]):
                    continue
                rr = aplicar(q, ent)
                if rr is not None:
                    ng.append(rr)
        if pos and ng:
            fora.append((nome, pos, ng, alvo))
    return fora


def sem_reguas(g: Grid, alvo: Grid, indices: Sequence[int],
               cache: Dict) -> float:
    """
    `indices` sao posicoes em `vivas`, NAO posicoes numa sublista.

    A primeira versao recebia a lista de reguas e indexava o cache por
    enumerate() sobre ela — ou seja, sempre pelas posicoes 0..k-1 de `vivas`,
    ignorando quais reguas tinham sido escolhidas. Os bracos "selecionadas" e
    "sorteadas" davam resultados identicos em todas as dobras, o que foi o
    sintoma que denunciou o erro. Sem esta correcao o experimento inteiro
    media a mesma coisa duas vezes.
    """
    acc = 0.0
    for idx in indices:
        x = cache[(id(g), idx)]
        y = cache[(id(alvo), idx)]
        e = max(abs(x), abs(y), 1.0)
        acc += 1.0 - min(abs(x - y) / e, 1.0)
    return acc / max(len(indices), 1)


def sem_mao(g: Grid, alvo: Grid) -> float:
    da, db = descrever(g), descrever(alvo)
    acc = 0.0
    for k in CHAVES:
        x, y = da[k], db[k]
        e = max(abs(x), abs(y), 1.0)
        acc += 1.0 - min(abs(x - y) / e, 1.0)
    return acc / len(CHAVES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", default=PASTA_TREINO)
    ap.add_argument("--geradas", type=int, default=700)
    ap.add_argument("--manter", type=int, default=9,
                    help="quantas reguas manter — casado com as 9 minhas")
    ap.add_argument("--neg", type=int, default=30)
    ap.add_argument("--semente", type=int, default=20260903)
    a = ap.parse_args()

    print("=" * 74)
    print("DESCRITOR 2 — o sistema inventa as proprias reguas")
    print("=" * 74)

    dados = coletar_pares(a.pasta, a.neg, a.semente)
    print(f"\n{len(dados)} tarefas com solucao de >= 2 passos")
    print(f"{sum(len(p) for _, p, _, _ in dados)} prefixos corretos, "
          f"{sum(len(n) for _, _, n, _ in dados)} aleatorios")
    if len(dados) < 4:
        print("amostra insuficiente para validacao cruzada. Parar.")
        return

    reguas = gerar(a.geradas, a.semente)

    # ── C0: descartar reguas constantes
    grades = []
    for _n, pos, ng, alvo in dados:
        grades.extend(pos)
        grades.extend(ng)
        grades.append(alvo)
    amostra = grades[::max(1, len(grades) // 120)]
    vivas = []
    for r in reguas:
        v = {r.valor(g) for g in amostra}
        if len(v) > 1:
            vivas.append(r)
    print(f"\nC0 · {len(vivas)}/{len(reguas)} reguas variam entre grades "
          f"(as constantes foram descartadas)")
    if len(vivas) < 20:
        print("     C0 · FALHA — quase tudo constante. Parar.")
        return
    print("     C0 · passa")

    cache: Dict = {}
    for g in grades:
        for idx, r in enumerate(vivas):
            cache[(id(g), idx)] = r.valor(g)

    def auc_reg(conj, indices):
        p, n = [], []
        for _nome, pos, ng, alvo in conj:
            for g in pos:
                p.append(sem_reguas(g, alvo, indices, cache))
            for g in ng:
                n.append(sem_reguas(g, alvo, indices, cache))
        return auc(p, n)

    def auc_uma(conj, r_idx):
        p, n = [], []
        for _nome, pos, ng, alvo in conj:
            ya = cache[(id(alvo), r_idx)]
            for g in pos:
                x = cache[(id(g), r_idx)]
                p.append(1 - min(abs(x - ya) / max(abs(x), abs(ya), 1.0), 1.0))
            for g in ng:
                x = cache[(id(g), r_idx)]
                n.append(1 - min(abs(x - ya) / max(abs(x), abs(ya), 1.0), 1.0))
        return auc(p, n)

    # ── validacao cruzada deixando uma tarefa de fora
    print(f"\nC1 · validacao cruzada, deixando uma tarefa de fora por vez")
    print(f"     seleciona {a.manter} reguas nas outras, mede na que ficou "
          f"de fora\n")
    print(f"{'tarefa':12s} {'selecionadas':>13} {'minhas 9':>10} "
          f"{'sorteadas':>11}")
    print("-" * 50)
    rng = random.Random(a.semente + 5)
    A_sel, A_mao, A_sor = [], [], []
    for i, alvo_t in enumerate(dados):
        treino = [d for j, d in enumerate(dados) if j != i]
        teste = [alvo_t]
        # seleciona pelas OUTRAS tarefas
        sel = sorted(range(len(vivas)),
                     key=lambda k: -auc_uma(treino, k))[:a.manter]
        sor = rng.sample(range(len(vivas)), a.manter)
        if i == 0:
            print("     reguas escolhidas na 1a dobra: "
                  + ", ".join(str(vivas[k]) for k in sel[:4]) + " ...")
            print()

        s = auc_reg(teste, sel)
        r = auc_reg(teste, sor)
        pm, nm = [], []
        for _n, pos, ng, alv in teste:
            pm += [sem_mao(g, alv) for g in pos]
            nm += [sem_mao(g, alv) for g in ng]
        m = auc(pm, nm)
        A_sel.append(s)
        A_mao.append(m)
        A_sor.append(r)
        print(f"{alvo_t[0]:12s} {s:>13.3f} {m:>10.3f} {r:>11.3f}")

    def md(v):
        v = [x for x in v if not math.isnan(x)]
        return sum(v) / len(v) if v else float("nan")

    print("-" * 50)
    print(f"{'MEDIA':12s} {md(A_sel):>13.3f} {md(A_mao):>10.3f} "
          f"{md(A_sor):>11.3f}")

    print("\n" + "=" * 74)
    s, m, r = md(A_sel), md(A_mao), md(A_sor)
    if s >= m and s > r + 0.05:
        print("C1 · as reguas que o sistema escolheu sozinho batem as minhas")
        print("     FORA da amostra de selecao. Procurar vazamento antes de")
        print("     comemorar: a selecao viu 8 tarefas, o teste viu 1.")
    elif abs(s - m) < 0.05:
        print("C2 · empata com a escolha manual. A geracao alcanca, nao supera.")
    else:
        print("C3 · sobreajuste. Com este tamanho de amostra a selecao decora")
        print("     em vez de aprender — e o numero diz quanto falta.")
    print(f"\namostra: {len(dados)} tarefas, "
          f"{sum(len(p) for _, p, _, _ in dados)} prefixos corretos.")
    print("Toda leitura acima esta limitada por isso, e nenhuma conclusao")
    print("sobre geracao de reguas em geral pode ser tirada daqui.")


if __name__ == "__main__":
    main()
