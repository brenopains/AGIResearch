"""
novaphys.ordem1 — quantificador na linguagem das LEIS
=======================================================

A JOGADA, E ELA E A COMPOSICAO DE DUAS COISAS JA MEDIDAS
----------------------------------------------------------
Hoje mediu-se a mesma troca em dois lugares diferentes:

  1. `quantificado.py`, linguagem de OBJETIVOS. Proposicional se refuta
     sozinha em 24 de 25 mundos — a eliminacao chega ao conjunto vazio. UM
     atomo quantificado leva a 99,3% de transferencia para um labirinto nunca
     visto, por 1,09x de exploracao a mais.

  2. `deducao.py`, linguagem de LEIS. Deduzir de um simulador de particulas
     rendeu ZERO, e o motivo tem endereco: `celula(i,j) = TESTE ? LEITURA_v :
     LEITURA_f` e POR CELULA. Queda com empilhamento e consequencia de alcance
     longo. Nao e dificil de deduzir — e inexpressivel.

E a mesma parede, nos dois. O remedio ja foi medido num paciente; falta
aplica-lo no outro.

O QUE ENTRA
-------------
Leituras e testes que QUANTIFICAM sobre objetos e sobre raios:

    cor_obj(maior)      argmax sobre componentes conexas
    empilha(dir)        para cada linha, os nao-fundo empacotados no fim
                        — isto E a gravidade, e era inexpressivel
    raio(dir)           primeira celula ocupada naquela direcao
    dentro              cercado por forma fechada (inundacao a partir da borda)
    pertence(sel)       a celula esta no objeto selecionado
    contorno            a celula esta na fronteira do proprio objeto

E o mecanismo de acesso continua sendo o que ganhou por 3.000x: DEDUZIR, com a
mesma maquina de mascaras de bits do `eter_inferencia`. A interface ja
permitia isso e ninguem tinha usado — `Leitura.valores(g, bg)` recebe a GRADE
INTEIRA, entao uma leitura sempre pode calcular algo global e distribuir por
celula. A maquina nao muda; so o vocabulario fica quantificado.

ELEGANCIA COMO EFICIENCIA (a pergunta do usuario sobre simetria)
------------------------------------------------------------------
Em vez de N primitivas escritas uma a uma, sao CINCO quantificadores sobre
CINCO dominios. A descricao encolhe e o poder expressivo explode. `priors.py`
mostrou o custo de nao ter isso: OBJETIDADE rendeu 1 tarefa com 11 operadores,
porque so ~29 valores eram alcancaveis.

O CONTROLE
------------
Mesma maquina, mesmo criterio de aceitacao, mesmo verificador exato no teste.
Muda so o vocabulario:

    L1  leituras e testes de `eter_inferencia` (por celula)
    L2  os mesmos MAIS os quantificados

E a mesma forma de comparacao de `quantificado.py`, onde L2 contem a resposta
e L1 nao. Isso seria armadilha se a medida fosse ajuste no treino; nao e — o
verificador exige igualdade EXATA nas grades de TESTE, que a deducao nunca viu.
Ajuste extensional nao passa nesse verificador.

Reporta-se tambem o custo em pares varridos, porque L2 tem mais leituras e o
ganho tem que sobreviver a isso.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  O0  PRE-CONDICAO. L1 tem que reproduzir `eter_inferencia.py`: 3/400 com
      precisao 100%. Se nao reproduzir, este arquivo diverge e nada vale.

  O1  L2 >= 12/400   -> o quantificador na lei e o degrau que faltava
  O2  L2 entre 5 e 11 -> ajuda, na mesma ordem dos outros mecanismos
  O3  L2 <= 4        -> objeto e raio nao sao o que falta na expressividade

MINHA PREVISAO, e ela e a aposta mais alta da sessao, feita de proposito: O1.
A deducao exata tem precisao de 100% medida, entao cada lei nova que virar
expressavel vira tarefa resolvida sem falso positivo. E as leituras que entram
sao exatamente as que `priors.py` apontou como faltando. Aposto L2 entre 12 e
20 SO com um passo, e aposto que a maioria das novas vem de `empilha` e
`dentro` — gravidade e preenchimento, as duas familias que a gramatica por
celula nunca pode tocar.

Se der O3, o achado tambem e forte e fecha uma linha: significa que o ARC nao
e sobre objetos, e a intuicao de objetidade que atravessa a literatura inteira
esta errada para este benchmark.
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

from novaphys.arc import Grid, carregar, dims, fundo, valida
from novaphys.eter import PASTA_AVAL
from novaphys.eter_inferencia import Leitura, Teste, todas_leituras, todos_testes

DIRS = {"baixo": (1, 0), "cima": (-1, 0), "dir": (0, 1), "esq": (0, -1)}


# ══════════════════════════════════════════════ objetos: as componentes conexas
def objetos(g: Grid, bg: int, diag: bool
            ) -> List[Tuple[Set[Tuple[int, int]], int]]:
    h, w = dims(g)
    viz = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diag:
        viz += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    vis = [[False] * w for _ in range(h)]
    fora = []
    for i in range(h):
        for j in range(w):
            if vis[i][j] or g[i][j] == bg:
                continue
            cor = g[i][j]
            cel, fila = set(), deque([(i, j)])
            vis[i][j] = True
            while fila:
                y, x = fila.popleft()
                cel.add((y, x))
                for dy, dx in viz:
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < h and 0 <= nx < w and not vis[ny][nx]
                            and g[ny][nx] == cor):
                        vis[ny][nx] = True
                        fila.append((ny, nx))
            fora.append((cel, cor))
    return fora


def escolher(objs, sel: str):
    """argmax / argmin / o unico. Devolve None se a escolha for ambigua."""
    if not objs:
        return None
    if sel == "maior":
        m = max(len(c) for c, _ in objs)
        cand = [o for o in objs if len(o[0]) == m]
    elif sel == "menor":
        m = min(len(c) for c, _ in objs)
        cand = [o for o in objs if len(o[0]) == m]
    elif sel == "unico_tam":
        cont: Dict[int, int] = {}
        for c, _ in objs:
            cont[len(c)] = cont.get(len(c), 0) + 1
        cand = [o for o in objs if cont[len(o[0])] == 1]
    else:                                       # cor_rara
        cont = {}
        for _c, k in objs:
            cont[k] = cont.get(k, 0) + 1
        m = min(cont.values())
        cand = [o for o in objs if cont[o[1]] == m]
    return cand[0] if len(cand) == 1 else None


def _celulas_dentro(g: Grid, bg: int) -> Set[Tuple[int, int]]:
    """Fundo NAO alcancavel a partir da borda: o interior de forma fechada."""
    h, w = dims(g)
    vis = [[False] * w for _ in range(h)]
    fila = deque()
    for i in range(h):
        for j in (0, w - 1):
            if g[i][j] == bg and not vis[i][j]:
                vis[i][j] = True
                fila.append((i, j))
    for j in range(w):
        for i in (0, h - 1):
            if g[i][j] == bg and not vis[i][j]:
                vis[i][j] = True
                fila.append((i, j))
    while fila:
        y, x = fila.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if (0 <= ny < h and 0 <= nx < w and not vis[ny][nx]
                    and g[ny][nx] == bg):
                vis[ny][nx] = True
                fila.append((ny, nx))
    return {(i, j) for i in range(h) for j in range(w)
            if g[i][j] == bg and not vis[i][j]}


# ══════════════════════════════════════════════ leituras QUANTIFICADAS
class LeituraQ:
    """Mesma interface de `Leitura`; o valor pode depender da grade inteira."""

    __slots__ = ("tipo", "arg", "diag", "const")

    def __init__(self, tipo: str, arg=None, diag: bool = False):
        self.tipo, self.arg, self.diag = tipo, arg, diag
        self.const = False

    def valores(self, g: Grid, bg: int) -> List[List[int]]:
        h, w = dims(g)
        t = self.tipo
        if t == "cor_obj":
            o = escolher(objetos(g, bg, self.diag), self.arg)
            k = o[1] if o else -1
            return [[k] * w for _ in range(h)]

        if t == "empilha":
            dy, dx = DIRS[self.arg]
            fora = [[bg] * w for _ in range(h)]
            if dx == 0:                                   # colunas
                for j in range(w):
                    col = [g[i][j] for i in range(h) if g[i][j] != bg]
                    if dy > 0:
                        for k, v in enumerate(col):
                            fora[h - len(col) + k][j] = v
                    else:
                        for k, v in enumerate(col):
                            fora[k][j] = v
            else:                                         # linhas
                for i in range(h):
                    lin = [v for v in g[i] if v != bg]
                    if dx > 0:
                        for k, v in enumerate(lin):
                            fora[i][w - len(lin) + k] = v
                    else:
                        for k, v in enumerate(lin):
                            fora[i][k] = v
            return fora

        if t == "raio":
            dy, dx = DIRS[self.arg]
            fora = [[bg] * w for _ in range(h)]
            for i in range(h):
                for j in range(w):
                    y, x = i + dy, j + dx
                    while 0 <= y < h and 0 <= x < w:
                        if g[y][x] != bg:
                            fora[i][j] = g[y][x]
                            break
                        y, x = y + dy, x + dx
            return fora

        if t == "tam_obj":                     # tamanho como cor, mod 10
            o = escolher(objetos(g, bg, self.diag), self.arg)
            k = (len(o[0]) % 10) if o else -1
            return [[k] * w for _ in range(h)]

        return [[-1] * w for _ in range(h)]

    def __str__(self) -> str:
        d = "8" if self.diag else "4"
        return f"{self.tipo}({self.arg}){d}" if self.arg else self.tipo


class TesteQ:
    """Mesma interface de `Teste`; a mascara pode depender da grade inteira."""

    __slots__ = ("tipo", "arg", "diag")

    def __init__(self, tipo: str, arg=None, diag: bool = False):
        self.tipo, self.arg, self.diag = tipo, arg, diag

    def mascara(self, g: Grid, bg: int) -> List[List[bool]]:
        h, w = dims(g)
        t = self.tipo
        m = [[False] * w for _ in range(h)]
        if t == "dentro":
            for (i, j) in _celulas_dentro(g, bg):
                m[i][j] = True
            return m
        if t == "pertence":
            o = escolher(objetos(g, bg, self.diag), self.arg)
            if o:
                for (i, j) in o[0]:
                    m[i][j] = True
            return m
        if t == "mesma_cor":
            o = escolher(objetos(g, bg, self.diag), self.arg)
            if o:
                for i in range(h):
                    for j in range(w):
                        m[i][j] = g[i][j] == o[1]
            return m
        if t == "contorno":
            for i in range(h):
                for j in range(w):
                    if g[i][j] == bg:
                        continue
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = i + dy, j + dx
                        if not (0 <= ny < h and 0 <= nx < w) \
                                or g[ny][nx] != g[i][j]:
                            m[i][j] = True
                            break
            return m
        return m

    def __str__(self) -> str:
        return f"{self.tipo}({self.arg})" if self.arg else self.tipo


SELS = ("maior", "menor", "unico_tam", "cor_rara")


def leituras_quantificadas() -> List[LeituraQ]:
    fora = []
    for sel in SELS:
        for diag in (False, True):
            fora.append(LeituraQ("cor_obj", sel, diag))
    for sel in ("maior", "menor"):
        fora.append(LeituraQ("tam_obj", sel, False))
    for d in DIRS:
        fora.append(LeituraQ("empilha", d))
        fora.append(LeituraQ("raio", d))
    return fora


def testes_quantificados() -> List[TesteQ]:
    fora = [TesteQ("dentro"), TesteQ("contorno")]
    for sel in SELS:
        for diag in (False, True):
            fora.append(TesteQ("pertence", sel, diag))
        fora.append(TesteQ("mesma_cor", sel, False))
    return fora


# ══════════════════════════════════════════════ deducao com listas injetaveis
def inferir(treino, leituras, testes):
    """
    Identica a `eter_inferencia.inferir`, com as listas vindo de fora. Nenhuma
    heuristica nova: varredura COMPLETA de pares, nada de amostragem.
    """
    if any(dims(a) != dims(b) for a, b in treino):
        return None, 0
    bg = fundo(treino[0][0])
    desloc, base = [], 0
    for a, _b in treino:
        h, w = dims(a)
        desloc.append(base)
        base += h * w
    TODAS = (1 << base) - 1

    cob = []
    for L in leituras:
        bits, n = 0, 0
        try:
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
        except Exception:
            continue
        if n:
            cob.append((L, bits, n))
    if not cob:
        return None, 0
    for L, bits, _n in cob:
        if bits == TODAS:
            return (Teste("sempre"), L, L), 1

    cob.sort(key=lambda x: -x[2])
    mt = []
    for T in testes:
        bits = 0
        try:
            for k, (a, _b) in enumerate(treino):
                mk = T.mascara(a, bg)
                h, w = dims(a)
                off = desloc[k]
                for i in range(h):
                    for j in range(w):
                        if mk[i][j]:
                            bits |= 1 << (off + i * w + j)
        except Exception:
            bits = 0
        mt.append(bits)

    pares = 0
    for Lv, bv, _nv in cob:
        falta = TODAS & ~bv
        if falta == 0:
            continue
        for Lf, bf, _nf in cob:
            if falta & ~bf:
                continue
            pares += 1
            ev, ef = bv & ~bf, bf & ~bv
            for T, bt in zip(testes, mt):
                if (ev & ~bt) == 0 and (ef & bt) == 0:
                    return (T, Lv, Lf), pares
    return None, pares


def aplicar(op, g: Grid) -> Optional[Grid]:
    T, Lv, Lf = op
    bg = fundo(g)
    h, w = dims(g)
    try:
        m = T.mascara(g, bg)
        vv = Lv.valores(g, bg)
        vf = Lf.valores(g, bg)
    except Exception:
        return None
    fora = tuple(tuple(vv[i][j] if m[i][j] else vf[i][j] for j in range(w))
                 for i in range(h))
    return fora if valida(fora) else None


def avaliar(tarefas, quant: bool, rotulo: str):
    ok, exemplos, pares = set(), {}, 0
    t0 = time.time()
    for nome, tr, te in tarefas:
        cores = sorted({c for a, b in tr for l in (a + b) for c in l})
        L = list(todas_leituras(cores))
        T = list(todos_testes(cores))
        if quant:
            L += leituras_quantificadas()
            T += testes_quantificados()
        op, p = inferir(tr, L, T)
        pares += p
        if op is None:
            continue
        if all(aplicar(op, x) == y for x, y in te):
            ok.add(nome)
            exemplos[nome] = f"[{op[0]} ? {op[1]} : {op[2]}]"
    n = len(tarefas)
    print(f"  {rotulo:16s} {len(ok):3d}/{n}  ({len(ok)/n:5.2%})  "
          f"{pares//n:>8d} pares/tarefa  ({time.time()-t0:.0f}s)", flush=True)
    return ok, exemplos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", default=PASTA_AVAL)
    a = ap.parse_args()

    print("=" * 78)
    print("ORDEM1 — quantificador na linguagem das LEIS")
    print("=" * 78)
    tarefas = carregar(a.pasta)
    print(f"\n{len(leituras_quantificadas())} leituras quantificadas, "
          f"{len(testes_quantificados())} testes quantificados")
    print("verificador: igualdade EXATA nas grades de teste, nunca vistas\n")

    l1, _e1 = avaliar(tarefas, False, "L1 por-celula")
    l2, e2 = avaliar(tarefas, True, "L2 quantificada")

    print(f"\nL2 - L1 = {len(l2) - len(l1):+d}")
    print(f"so a L2 resolve : {sorted(l2 - l1)}")
    print(f"so a L1 resolve : {sorted(l1 - l2)}")
    print("\nleis novas descobertas:")
    for n in sorted(l2 - l1)[:14]:
        print(f"  {n}  {e2[n]}")

    print("\n" + "=" * 78)
    if len(l1) < 2:
        print(f"O0 · L1 deu {len(l1)}/400 e `eter_inferencia` deu 3. Divergiu.")
    if len(l2) >= 12:
        print("O1 · o quantificador na LEI e o degrau que faltava. Mesma")
        print("     maquina de deducao, mesmo verificador exato — so o")
        print("     vocabulario passou a quantificar sobre objetos e raios.")
    elif len(l2) >= 5:
        print("O2 · ajuda, na mesma ordem dos outros mecanismos da sessao.")
    else:
        print("O3 · objeto e raio nao sao o que falta. Se isso se sustentar,")
        print("     a intuicao de OBJETIDADE que atravessa a literatura esta")
        print("     errada para este benchmark.")


if __name__ == "__main__":
    main()
