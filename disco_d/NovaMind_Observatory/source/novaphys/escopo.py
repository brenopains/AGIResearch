"""novaphys.escopo — a operacao aprende ONDE nao se meter.

Ver PRE_REGISTRO_escopo.md. Nasce do rastreio de 0607ce86: o par escreve 506
celulas para consertar 6 e erra 120. Nao e fraqueza, e falta de ESCOPO.

`CondicaoAprendida` ja faz isso para UMA operacao (pintar fundo de uma cor).
Aqui e para uma operacao QUALQUER.
"""
from __future__ import annotations
import random
from math import log2

from novaphys import descricao as D
from novaphys.descricao import objetos, fundo, dims, BITS_COR

# ---- a linguagem de escopo: testes por celula, calculados da ENTRADA ----
def testes(g, bg):
    h, w = dims(g)
    nb = [[g[y][x] != bg for x in range(w)] for y in range(h)]
    lin = [sum(r) for r in nb]
    col = [sum(nb[y][x] for y in range(h)) for x in range(w)]
    T = {}
    T["fundo"] = lambda y, x: g[y][x] == bg
    T["nao_fundo"] = lambda y, x: g[y][x] != bg
    T["borda"] = lambda y, x: y in (0, h - 1) or x in (0, w - 1)
    T["interior"] = lambda y, x: 0 < y < h - 1 and 0 < x < w - 1
    T["linha_tem"] = lambda y, x: lin[y] > 0
    T["coluna_tem"] = lambda y, x: col[x] > 0
    T["linha_vazia"] = lambda y, x: lin[y] == 0
    T["coluna_vazia"] = lambda y, x: col[x] == 0
    for c in range(10):
        T[f"cor=={c}"] = (lambda c=c: (lambda y, x: g[y][x] == c))()
    T["par_i"] = lambda y, x: y % 2 == 0
    T["par_j"] = lambda y, x: x % 2 == 0
    T["diag"] = lambda y, x: (y + x) % 2 == 0
    viz = {}
    for y in range(h):
        for x in range(w):
            viz[(y, x)] = any(0 <= y+dy < h and 0 <= x+dx < w and g[y+dy][x+dx] != bg
                              for dy in (-1,0,1) for dx in (-1,0,1) if (dy,dx)!=(0,0))
    T["vizinho_obj"] = lambda y, x: viz[(y, x)]
    T["sem_vizinho"] = lambda y, x: not viz[(y, x)]
    return T


NOMES_T = None


class ComEscopo:
    """Envolve um codec e aprende, do treino, ONDE ele deve escrever.
    marcador = (marcador do codec, tupla de nomes de teste)."""

    def __init__(self, base, max_testes=2, embaralha=False, semente=17):
        self.base = base
        self.nome = f"esc:{base.nome}"
        self.max_testes = max_testes
        self.embaralha = embaralha        # PLACEBO: escopo sorteado
        self.rng = random.Random(semente)

    def propor(self, treino, alvo=None, saida_atual=None):
        """`saida_atual`: o que a pilha ja produz. SEM ele o aprendiz pontua
        'escritas corretas preservadas' — e a maioria cai em celula QUE JA ESTAVA
        CERTA, entao ele preserva o inutil e descarta o valioso (medido: escopo
        ('cor==0') conteve 506->300 e perdeu os 6 consertos). Com ele, pontua
        CONSERTA - QUEBRA, que e o que importa."""
        try:
            ms = list(self.base.propor(treino))[:3]
        except Exception:
            return []
        if not ms:
            return []
        amostras = []
        for a, b in treino:
            bg = fundo(a)
            amostras.append((a, b, bg, testes(a, bg)))
        nomes = sorted(amostras[0][3])
        fora = []
        for m in ms:
            # certas/erradas de cada teste, somadas sobre o treino
            pont = {}
            for nome in nomes:
                bom = ruim = 0
                for a, b, bg, T in amostras:
                    if len(a) != len(b) or len(a[0]) != len(b[0]):
                        continue
                    try:
                        pv = self.base.prever(m, a)
                    except Exception:
                        continue
                    f = T[nome]
                    sa = saida_atual.get(id(a)) if saida_atual else None
                    for (y, x), c in pv.items():
                        if not (0 <= y < len(b) and 0 <= x < len(b[0])):
                            continue
                        if not f(y, x):
                            continue
                        certo = (c == b[y][x])
                        if sa is None:
                            bom += certo; ruim += (not certo)
                        else:
                            ja_certo = (sa[y][x] == b[y][x])
                            if certo and not ja_certo:
                                bom += 1          # CONSERTA
                            elif (not certo) and ja_certo:
                                ruim += 1         # QUEBRA
                pont[nome] = (bom - ruim, bom, ruim)
            bons = sorted(nomes, key=lambda n: -pont[n][0])
            if self.embaralha:
                self.rng.shuffle(bons)
            for k in (1, self.max_testes):
                sel = tuple(bons[:k])
                if pont[sel[0]][0] <= 0 and not self.embaralha:
                    continue
                fora.append((m, sel))
        return fora[:6]

    def bits(self, mm):
        m, sel = mm
        try:
            return self.base.bits(m) + len(sel) * log2(26) + 1
        except Exception:
            return 99.0

    def prever(self, mm, a):
        m, sel = mm
        try:
            pv = self.base.prever(m, a)
        except Exception:
            return {}
        bg = fundo(a)
        T = testes(a, bg)
        fs = [T[n] for n in sel if n in T]
        if not fs:
            return {}
        return {(y, x): c for (y, x), c in pv.items() if all(f(y, x) for f in fs)}
