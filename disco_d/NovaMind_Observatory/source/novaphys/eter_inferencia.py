"""
novaphys.eter_inferencia — inferir o operador, em vez de sortear e peneirar
============================================================================

A DISTINCAO QUE ORIGINA ESTE ARQUIVO
--------------------------------------
`eter2.py` sorteava 7.000 operadores da meta-gramatica e peneirava pela
tarefa. A densidade de operador util deu 1 em 3.000: 99,97% do trabalho era
jogado fora. Gerar e catar os cacos.

O inverso e possivel e e o que este arquivo faz. A meta-gramatica e

    celula(i,j) = TESTE(i,j) ? LEITURA_v(i,j) : LEITURA_f(i,j)

e nos pares de treino o valor exigido de CADA celula e conhecido. Entao o
operador nao precisa ser adivinhado — ele pode ser RESOLVIDO:

  1. para cada LEITURA possivel, calcular em quais celulas ela produz o valor
     certo. Uma leitura correta em lugar nenhum e descartada na hora.
  2. para um par de leituras (v, f) sobreviventes, o TESTE fica determinado
     por restricoes:
         onde so v acerta   -> o teste PRECISA ser verdadeiro
         onde so f acerta   -> o teste PRECISA ser falso
         onde ambos acertam -> livre
         onde nenhum acerta -> este par e impossivel, descarta
  3. procurar um teste da gramatica que satisfaca as restricoes.

O passo 1 e o que muda tudo: ele reduz milhares de leituras a um punhado, e o
faz olhando o dado. Nao ha peneira depois — a poda acontece ANTES, e por
deducao.

O QUE ISTO ALCANCA E O QUE NAO ALCANCA
----------------------------------------
Alcanca exatamente as transformacoes que sao funcao LOCAL da grade: reflexoes,
rotacoes, translacoes, recoloracoes condicionais, automatos celulares, listras,
bordas, diagonais. Se a tarefa for uma dessas, este metodo acha o operador em
UMA passada, sem busca.

Nao alcanca nada que dependa de OBJETO como entidade — "pinte o maior", "conte
os objetos" — porque a gramatica nao tem o tipo Objeto. Isso e o prior de
objetidade e de numero, e e outro arquivo.

A comparacao honesta e contra `eter2`, que amostrava do MESMO espaco. Mesma
gramatica, mesmas leituras, mesmos testes. So muda se o operador e sorteado ou
deduzido.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  I0  PRE-CONDICAO. Numa tarefa cuja resposta e um espelho horizontal puro, o
      metodo tem que deduzir o operador correspondente. Se nao deduzir o caso
      trivial, ha bug e nada abaixo vale.

  I0b PRE-CONDICAO NOVA, depois do primeiro erro. A varredura de pares tem
      que ser COMPLETA. Se em algum ponto eu cortar em "top N", isto virou
      amostragem e a comparacao contra `eter2` perde o sentido — foi
      exatamente o que aconteceu na primeira versao.

  I1  a inferencia acha operador valido para >= 20x mais tarefas do que a
      amostragem do `eter2` achava
      -> deduzir supera sortear no mesmo espaco, e a diferenca e de mecanismo

  I2  ganho existe mas abaixo de 20x

  I3  sem ganho
      -> o gargalo nao era a forma de obter o operador

MINHA PREVISAO, escrita antes de rodar: I1 com folga na contagem de OPERADORES
deduzidos, e ganho modesto na contagem de TAREFAS resolvidas. A razao e que a
deducao deve achar operador local para muitas tarefas, mas a maioria das
tarefas do ARC precisa de mais de um passo ou precisa do tipo Objeto, que esta
fora desta gramatica.

Ou seja: espero que o MECANISMO se prove e que o NUMERO FINAL continue baixo.
Se isso acontecer, a leitura correta e "deduzir funciona, e o que falta agora
sao os priors de objetidade e numero" — nao "deduzir nao serve".
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

from novaphys.arc import Grid, carregar, dims, fundo, valida
from novaphys.eter import PASTA_AVAL

Cel = Tuple[int, int]
MATS = [(a, b, c, d)
        for a in (-1, 0, 1) for b in (-1, 0, 1)
        for c in (-1, 0, 1) for d in (-1, 0, 1)]
BORDAS = ("fundo", "toro", "corta")


# ══════════════════════════════════════════════ leituras
class Leitura:
    """Um valor por celula: da grade numa posicao transformada, ou constante."""

    __slots__ = ("const", "k", "m", "dy", "dx", "borda")

    def __init__(self, const=None, m=None, dy=0, dx=0, borda="fundo"):
        self.const = const is not None
        self.k = const
        self.m = m
        self.dy = dy
        self.dx = dx
        self.borda = borda

    def valores(self, g: Grid, bg: int) -> Optional[List[List[int]]]:
        h, w = dims(g)
        if self.const:
            return [[self.k] * w for _ in range(h)]
        a, b, c, d = self.m
        fora = []
        for i in range(h):
            linha = []
            for j in range(w):
                y = a * i + b * j + self.dy
                x = c * i + d * j + self.dx
                if self.borda == "toro":
                    linha.append(g[y % h][x % w])
                elif 0 <= y < h and 0 <= x < w:
                    linha.append(g[y][x])
                elif self.borda == "fundo":
                    linha.append(bg)
                else:
                    linha.append(-1)          # invalido: nunca casa
            fora.append(linha)
        return fora

    def __str__(self) -> str:
        if self.const:
            return f"={self.k}"
        a, b, c, d = self.m
        return f"g[{a}i{b:+d}j{self.dy:+d},{c}i{d:+d}j{self.dx:+d}]·{self.borda[:3]}"


def todas_leituras(cores: Sequence[int]) -> List[Leitura]:
    fora = [Leitura(const=k) for k in cores]
    for m in MATS:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for b in BORDAS:
                    fora.append(Leitura(m=m, dy=dy, dx=dx, borda=b))
    return fora


# ══════════════════════════════════════════════ testes
class Teste:
    __slots__ = ("tipo", "eixo", "mod", "r", "cor", "d")

    def __init__(self, tipo, eixo=None, mod=2, r=0, cor=0, d=(0, 1)):
        self.tipo = tipo
        self.eixo = eixo
        self.mod = mod
        self.r = r
        self.cor = cor
        self.d = d

    def mascara(self, g: Grid, bg: int) -> List[List[bool]]:
        h, w = dims(g)
        t = self.tipo
        m = [[False] * w for _ in range(h)]
        for i in range(h):
            for j in range(w):
                if t == "sempre":
                    m[i][j] = True
                elif t == "cor":
                    m[i][j] = g[i][j] == self.cor
                elif t == "naofundo":
                    m[i][j] = g[i][j] != bg
                elif t == "borda":
                    m[i][j] = i in (0, h - 1) or j in (0, w - 1)
                elif t == "paridade":
                    v = {"i": i, "j": j, "i+j": i + j, "i-j": i - j}[self.eixo]
                    m[i][j] = v % self.mod == self.r
                else:                                   # vizinho igual
                    dy, dx = self.d
                    ny, nx = i + dy, j + dx
                    m[i][j] = (0 <= ny < h and 0 <= nx < w
                               and g[i][j] == g[ny][nx])
        return m

    def __str__(self) -> str:
        t = self.tipo
        if t == "sempre":
            return "T"
        if t == "cor":
            return f"cor=={self.cor}"
        if t == "paridade":
            return f"({self.eixo})%{self.mod}=={self.r}"
        if t == "vizinho":
            return f"igual_a{self.d}"
        return t


def todos_testes(cores: Sequence[int]) -> List[Teste]:
    fora = [Teste("sempre"), Teste("naofundo"), Teste("borda")]
    for c in cores:
        fora.append(Teste("cor", cor=c))
    for e in ("i", "j", "i+j", "i-j"):
        for mod in (2, 3):
            for r in range(mod):
                fora.append(Teste("paridade", eixo=e, mod=mod, r=r))
    for d in ((0, 1), (1, 0), (1, 1), (1, -1)):
        fora.append(Teste("vizinho", d=d))
    return fora


# ══════════════════════════════════════════════ inferencia
def inferir(treino: Sequence[Tuple[Grid, Grid]]
            ) -> Tuple[Optional[Tuple[Teste, Leitura, Leitura]], int]:
    """
    Resolve para (teste, leitura_v, leitura_f) a partir dos pares. Devolve
    tambem quantas leituras sobreviveram a poda por dado — que e a medida de
    quanto a deducao economizou em relacao a sortear.
    """
    if any(dims(a) != dims(b) for a, b in treino):
        return None, 0
    bg = fundo(treino[0][0])
    cores = sorted({c for a, b in treino for l in (a + b) for c in l})

    leituras = todas_leituras(cores)

    # PASSO 1 — cobertura de cada leitura, como MASCARA DE BITS
    #
    # A primeira versao guardava conjuntos de celulas e mantinha toda leitura
    # que acertasse >= 1 celula. Numa grade cheia de fundo isso mantem quase
    # tudo: 1941 de 2197 sobreviveram, 88%. Ai eu cortava nos "top 60" e
    # varria 3.600 de 3,8 milhoes de pares — ou seja, AMOSTRAGEM disfarcada,
    # exatamente o que este arquivo existe para nao fazer.
    #
    # Com mascara de bits a restricao vira uma operacao inteira e o par pode
    # ser varrido por COMPLETO. Os bits de todos os pares de treino sao
    # concatenados num unico inteiro, entao cada teste custa um `&`.
    desloc = []
    base = 0
    for a, _b in treino:
        h, w = dims(a)
        desloc.append(base)
        base += h * w
    TODAS = (1 << base) - 1

    cobertura: List[Tuple[Leitura, int, int]] = []
    for L in leituras:
        bits = 0
        n = 0
        for k, (a, b) in enumerate(treino):
            v = L.valores(a, bg)
            h, w = dims(a)
            off = desloc[k]
            for i in range(h):
                vi, bi = v[i], b[i]
                for j in range(w):
                    if vi[j] == bi[j]:
                        bits |= 1 << (off + i * w + j)
                        n += 1
        if n:
            cobertura.append((L, bits, n))
    if not cobertura:
        return None, 0

    # uma leitura sozinha explicando tudo e o caso mais simples
    for L, bits, _n in cobertura:
        if bits == TODAS:
            return (Teste("sempre"), L, L), len(cobertura)

    # PASSO 2 — TODOS os pares, sem corte. Dado v, o que f precisa cobrir
    # esta determinado: exatamente o que v deixou de fora.
    cobertura.sort(key=lambda x: -x[2])
    testes = todos_testes(cores)
    mtestes = []
    for T in testes:
        bits = 0
        for k, (a, _b) in enumerate(treino):
            mk = T.mascara(a, bg)
            h, w = dims(a)
            off = desloc[k]
            for i in range(h):
                for j in range(w):
                    if mk[i][j]:
                        bits |= 1 << (off + i * w + j)
        mtestes.append(bits)

    pares_viaveis = 0
    for Lv, bv, _nv in cobertura:
        falta = TODAS & ~bv
        if falta == 0:
            continue
        for Lf, bf, _nf in cobertura:
            if falta & ~bf:            # f nao cobre o que v deixou faltando
                continue
            pares_viaveis += 1
            # o teste precisa ser VERDADEIRO onde so v acerta e FALSO onde
            # so f acerta. Onde ambos acertam, e livre.
            exige_v = bv & ~bf
            exige_f = bf & ~bv
            for T, bt in zip(testes, mtestes):
                if (exige_v & ~bt) == 0 and (exige_f & bt) == 0:
                    return (T, Lv, Lf), len(cobertura)
    return None, len(cobertura)


def aplicar(op: Tuple[Teste, Leitura, Leitura], g: Grid) -> Optional[Grid]:
    T, Lv, Lf = op
    bg = fundo(g)
    h, w = dims(g)
    mv = Lv.valores(g, bg)
    mf = Lf.valores(g, bg)
    mk = T.mascara(g, bg)
    fora = []
    for i in range(h):
        linha = []
        for j in range(w):
            x = mv[i][j] if mk[i][j] else mf[i][j]
            if x < 0:
                return None
            linha.append(x)
        fora.append(tuple(linha))
    r = tuple(fora)
    return r if valida(r) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", default=PASTA_AVAL)
    a = ap.parse_args()

    print("=" * 76)
    print("ETER POR INFERENCIA — deduzir o operador em vez de sortear")
    print("=" * 76)
    n_leit = len(todas_leituras(range(10)))
    n_test = len(todos_testes(range(10)))
    print(f"espaco: {n_leit} leituras x {n_leit} leituras x {n_test} testes")
    print(f"        = {n_leit*n_leit*n_test:,} operadores")
    print("`eter2` amostrava 7.000 deste espaco e achava util 1 em 3.000.")
    print("Aqui nenhum e sorteado: as leituras sao PODADAS pelo dado e o")
    print("teste sai por restricao.\n")

    # I0
    esp = (Teste("sempre"), Leitura(m=(1, 0, 0, -1), dy=0, dx=0), None)
    prova = ((1, 2, 3), (4, 5, 6), (7, 8, 9))
    alvo = tuple(tuple(l[::-1]) for l in prova)
    op, _n = inferir([(prova, alvo)])
    ok0 = op is not None and aplicar(op, prova) == alvo
    print(f"I0 · deduz espelho horizontal num caso trivial: "
          f"{'PASSA' if ok0 else 'FALHA'}"
          + (f"   [{op[0]} ? {op[1]} : {op[2]}]" if op else ""))
    if not ok0:
        print("     bug no arcabouco. Nao ler o resto.")
        return

    tarefas = carregar(a.pasta)
    t0 = time.time()
    achou_op = 0
    resolve: Set[str] = set()
    sobreviventes = []
    exemplos = []
    for nome, tr, te in tarefas:
        op, nsob = inferir(tr)
        if nsob:
            sobreviventes.append(nsob)
        if op is None:
            continue
        achou_op += 1
        if all(aplicar(op, x) == y for x, y in te):
            resolve.add(nome)
            if len(exemplos) < 12:
                exemplos.append(f"{nome}  [{op[0]} ? {op[1]} : {op[2]}]")
    dt = time.time() - t0

    med = sum(sobreviventes) / len(sobreviventes) if sobreviventes else 0
    print(f"\nleituras sobreviventes a poda por dado: {med:.0f} de {n_leit} "
          f"({med/n_leit:.1%})")
    print(f"operador deduzido para : {achou_op}/400 tarefas "
          f"({achou_op/4:.1f}%)")
    print(f"RESOLVE (acerta o teste): {len(resolve)}/400 "
          f"({len(resolve)/4:.2f}%)   ({dt:.0f}s)")

    print("\nexemplos deduzidos:")
    for e in exemplos:
        print("   " + e)

    print("\n" + "=" * 76)
    print("comparacao de MECANISMO contra `eter2` (mesmo espaco, amostrado):")
    print(f"   eter2, amostragem : util 1 em 3.000 sorteios")
    print(f"   aqui, inferencia  : operador valido em {achou_op}/400 tarefas")
    print("\nA leitura correta separa duas coisas: se o MECANISMO de deduzir")
    print("supera o de sortear (contagem de operadores), e se isso vira")
    print("TAREFA RESOLVIDA. Podem divergir, e provavelmente divergem — esta")
    print("gramatica nao tem o tipo Objeto, entao o prior de objetidade e de")
    print("numero continua fora do alcance dela.")


if __name__ == "__main__":
    main()
