"""novaphys.tres_rapido — o MESMO retrocesso, motor compilado.

POR QUE
---------
`tres_em_linha.py` fechou em 4 rodadas com diagnostico cravado: a politica
(MDL -> memoria -> analogia) resolveu a ALOCACAO (18 -> 20), e a parede em ~22 e
do CONSTRUTOR — retrocesso em Python. A pergunta que sobra e a unica que ainda e
de arquitetura:

    a politica continua pagando quando o motor e forte?

Se a vantagem sumir com motor rapido, a politica so compensava lentidao.
Se persistir, a politica e real.

O QUE MUDA E O QUE NAO MUDA
-----------------------------
MUDA   : execucao (numba njit, bitmask, checagem de colinearidade por caminhada)
NAO MUDA: as orbitas, o filtro de orbita ruim, a poda de 2-por-linha/coluna, a
          poda de linha fechada (alvo_linha), o limite de passos, a ordem de
          escolha (preferencia/aleatoria), o verificador.

PORTAO DE CORRECAO, antes de qualquer medida de velocidade
------------------------------------------------------------
  V1  para n em 6..16 e todo grupo, o motor rapido acha solucao exatamente nos
      mesmos (n, g) que o motor Python acha
  V2  toda solucao do motor rapido passa no verificador INDEPENDENTE O(n^3)
  V3  o motor rapido e mais rapido (se nao for, nao ha experimento)
Se V1 ou V2 falharem, o porte esta errado e nada depois vale.
"""
from __future__ import annotations
import time
import numpy as np
from numba import njit

import sys
sys.path.insert(0, ('C:/Users/usuario/Documents/AGI 2.0').replace('C:/Users/usuario', __import__('os').path.expanduser('~').replace('\\', '/')))
from novaphys.tres_em_linha import GRUPOS, orbitas, verifica, _linha


def prepara(n, g):
    """Orbitas boas -> arrays planos. Mesmo filtro do original."""
    obs = orbitas(n, g)
    boas = []
    for o in obs:
        lin, ok = {}, True
        for i in range(len(o)):
            for j in range(i + 1, len(o)):
                L = _linha(o[i], o[j])
                lin.setdefault(L, set()).update((o[i], o[j]))
                if len(lin[L]) > 2:
                    ok = False
        if ok:
            boas.append(o)
    if not boas:
        return None
    ini = np.zeros(len(boas) + 1, np.int32)
    cel = []
    for i, o in enumerate(boas):
        ini[i + 1] = ini[i] + len(o)
        cel.extend([y * n + x for (x, y) in o])
    cel = np.array(cel, np.int32)
    ultima = np.full(n, -1, np.int32)
    for i, o in enumerate(boas):
        for (_x, y) in o:
            ultima[y] = max(ultima[y], i)
    if (ultima < 0).any():
        return None
    return ini, cel, ultima, len(boas)


@njit(cache=True)
def _mdc(a, b):
    while b:
        a, b = b, a % b
    return a


@njit(cache=True)
def _ok_add(p, ocup, pts, k, n):
    """p cria trio colinear com os k pontos ja postos?"""
    py, px = p // n, p % n
    for t in range(k):
        a = pts[t]
        ay, ax = a // n, a % n
        dy, dx = py - ay, px - ax
        gg = _mdc(abs(dy), abs(dx))
        sy, sx = dy // gg, dx // gg
        for dirn in (-1, 1):
            cy, cx = ay + sy * dirn, ax + sx * dirn
            while 0 <= cy < n and 0 <= cx < n:
                c = cy * n + cx
                if c != p and c != a and ocup[c]:
                    return False
                cy += sy * dirn
                cx += sx * dirn
    return True


@njit(cache=True)
def _busca(ini, cel, ultima, nob, n, limite, semente, pref, usa_pref):
    alvo = 2 * n
    ocup = np.zeros(n * n, np.uint8)
    pts = np.zeros(alvo + 8, np.int32)
    cl = np.zeros(n, np.int32)
    cc = np.zeros(n, np.int32)
    # pilha: nivel -> qual das duas escolhas estamos, e se pegamos
    qual = np.zeros(nob + 2, np.int8)
    peg = np.zeros(nob + 2, np.int8)
    ordem = np.zeros((nob + 2, 2), np.int8)
    estado = np.random.seed(semente)
    k = 0
    i = 0
    passos = 0
    qual[0] = -1
    while i >= 0:
        if qual[i] == -1:
            passos += 1
            if passos > limite:
                return np.empty(0, np.int32)
            # poda de linha fechada
            mau = False
            for y in range(n):
                if ultima[y] < i and cl[y] != 2:
                    mau = True
                    break
            if mau:
                i -= 1
                if i >= 0 and peg[i] == 1:
                    s, e = ini[i], ini[i + 1]
                    for t in range(e - 1, s - 1, -1):
                        c = cel[t]
                        ocup[c] = 0
                        k -= 1
                        cl[c // n] -= 1
                        cc[c % n] -= 1
                continue
            if i == nob:
                if k == alvo:
                    return pts[:k].copy()
                i -= 1
                if i >= 0 and peg[i] == 1:
                    s, e = ini[i], ini[i + 1]
                    for t in range(e - 1, s - 1, -1):
                        c = cel[t]
                        ocup[c] = 0
                        k -= 1
                        cl[c // n] -= 1
                        cc[c % n] -= 1
                continue
            sim = False
            if usa_pref:
                for t in range(ini[i], ini[i + 1]):
                    if pref[cel[t]]:
                        sim = True
                if np.random.random() < 0.1:
                    sim = not sim
            else:
                sim = np.random.random() < 0.5
            if sim:
                ordem[i, 0] = 1; ordem[i, 1] = 0
            else:
                ordem[i, 0] = 0; ordem[i, 1] = 1
            qual[i] = 0
            peg[i] = 0
        if qual[i] > 1:
            qual[i] = -1
            i -= 1
            if i >= 0 and peg[i] == 1:
                s, e = ini[i], ini[i + 1]
                for t in range(e - 1, s - 1, -1):
                    c = cel[t]
                    ocup[c] = 0
                    k -= 1
                    cl[c // n] -= 1
                    cc[c % n] -= 1
            continue
        pega = ordem[i, qual[i]]
        qual[i] += 1
        if pega == 1:
            s, e = ini[i], ini[i + 1]
            bom = True
            posto = 0
            for t in range(s, e):
                c = cel[t]
                y, x = c // n, c % n
                if cl[y] >= 2 or cc[x] >= 2 or not _ok_add(c, ocup, pts, k, n):
                    bom = False
                    break
                ocup[c] = 1
                pts[k] = c
                k += 1
                cl[y] += 1
                cc[x] += 1
                posto += 1
            if not bom:
                for t in range(posto):
                    c = pts[k - 1]
                    ocup[c] = 0
                    k -= 1
                    cl[c // n] -= 1
                    cc[c % n] -= 1
                continue
            peg[i] = 1
            i += 1
            qual[i] = -1
        else:
            peg[i] = 0
            i += 1
            qual[i] = -1
    return np.empty(0, np.int32)


def busca_rapida(n, g, prazo, semente=0, pref=None):
    pr = prepara(n, g)
    if pr is None:
        return None
    ini, cel, ultima, nob = pr
    pm = np.zeros(n * n, np.uint8)
    if pref:
        for (x, y) in pref:
            if 0 <= x < n and 0 <= y < n:
                pm[y * n + x] = 1
    lim = 20000 * n
    s = semente
    while time.time() < prazo:
        r = _busca(ini, cel, ultima, nob, n, lim, s, pm, pref is not None)
        s += 1
        if len(r):
            return [(int(c % n), int(c // n)) for c in r]
    return None
