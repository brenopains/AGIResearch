"""
novaphys.descricao — o nucleo de comprimento de descricao
===========================================================

A TESE
--------
Nao e sobre ARC. E sobre acumulacao.

Um agente que so aceita explicacao COMPLETA nao consegue acumular. Toda regra
parcial vale zero, entao toda regra precisa nascer inteira, entao regras nunca
se somam. E exatamente o estado deste projeto: catorze motores, cada um
resolvendo de 1 a 4 tarefas, uniao 33/400, e nenhuma tarefa resolvida por dois
motores juntos.

Medido hoje, no `atracao.py`: mecanismo certo, 21/21 pecas, dois dos tres pares
reproduzidos EXATAMENTE, uma peca errada. Resultado: ZERO. Cobertura mecanica
de 30,2% virou 0/400.

Com sinal binario nao ha gradiente. Sem gradiente nao ha refinamento. Sem
refinamento nao ha composicao. A "barreira de composicao" que este projeto vem
medindo ha semanas nao e do compositor — e consequencia do portao binario.

    Credito parcial e o que torna conhecimento composicional.
    Bits sao o unico credito parcial que nao e arbitrario, porque cobram
    os bits do proprio modelo — nao da para trapacear memorizando.

DE ONDE VEM (pesquisa, setembro de 2026)
------------------------------------------
Quatro resultados independentes convergem no mesmo ponto:

  CompressARC (Liao & Gu, premio ARC Prize 2025)
      MDL puro no tempo de inferencia, 76K parametros, zero pre-treino, zero
      dataset externo. 34,75% no treino e 20% no eval do ARC-AGI-1.
      arxiv.org/abs/2512.06104

  Ferre, ARC-MDL / MADIL
      MDL simbolico com modelos por objeto. Foi de 5 para 29/400 em um ano,
      30s por tarefa. arxiv.org/abs/2311.00545, arxiv.org/abs/2505.01081

  Li & Ellis, inducao vs transducao
      Os dois resolvem conjuntos DISJUNTOS, estaveis entre sementes. Todo topo
      de placar combina os dois. arxiv.org/abs/2411.02272

  ARC Prize 2025, relatorio tecnico
      O tema declarado do ano e "the refinement loop": transformar
      iterativamente uma versao do programa numa um pouco melhor, guiado por um
      sinal de feedback.

E a complementaridade com o CompressARC nao e esperanca: esta na lista de
falhas dos proprios autores. Eles declaram que a arquitetura NAO faz contagem,
conectividade/topologia, operacoes repetidas, transformacoes grandes (escala,
rotacao, duplicacao) nem mapa de cor. E item por item o que os nossos motores
fazem — `segmenta` (8 lentes de conectividade), `ladrilho` (48 hipoteses
exclusivas, precisao 100%), `censofino`, `compositor`.

O MECANISMO
-------------
Cada motor deixa de ser solucionador binario e vira CODEC: uma funcao PARCIAL
que reivindica algumas celulas da saida e se cala sobre as outras.

    custo(pilha) = SOMA bits_do_modelo(codec_i)          <- pago UMA vez
                 + SOMA_pares bits_literais(celulas erradas ou nao cobertas)

O modelo e pago uma unica vez e vale para TODOS os pares — e isso, e so isso,
que separa regra de memoria. Uma excecao literal custa
log2(H*W) + log2(10) bits por celula, entao um codec so entra na pilha se
economizar mais bits do que custa. Ockham deixa de ser ordenacao heuristica e
vira aritmetica.

A busca e o laco de refinamento: comeca com a pilha vazia (custo = codificar a
saida literal inteira), e a cada passo adiciona o codec que mais reduz o custo,
trabalhando sobre o RESIDUO do que ficou. Para quando nenhum codec se paga.

Isso e o que teria salvado o `atracao`: regra explica 20 das 21 pecas, residuo
= 1 peca, outro codec ataca o residuo, pilha fecha.

O que muda de verdade: uma tarefa pode ser resolvida por DOIS ou mais codecs
que, sozinhos, nao resolvem nenhuma. Isso e impossivel na arquitetura atual por
construcao.

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  D0  PRE-CONDICAO. A pilha tem que reproduzir pelo menos as tarefas que os
      codecs ja resolviam sozinhos. Se o nucleo perder tarefa conhecida, e erro
      de contabilidade de bits, nao resultado.

  D1  COMPOSICAO (a tese). Alguma tarefa resolvida por uma pilha de DOIS OU
      MAIS codecs, que nenhum codec sozinho resolve?
          >= 1  -> a tese esta certa: credito parcial produz composicao
             0  -> a tese esta errada, e o portao binario nao era o gargalo

  D2  >= 8 inedidas -> o credito parcial era o salto
  D3  3 a 7         -> ajuda no patamar das mudancas de representacao
  D4  <= 2          -> o nucleo esta certo e os codecs e que sao poucos

MINHA CALIBRAGEM: trinta e cinco previsoes, DEZ certas, as vinte e cinco
erradas todas otimistas. As tres de hoje deram zero.

PREVISAO: D1 >= 1 e D3. Aposto na composicao acontecer porque ela e mecanica —
`copia` mais um segundo codec ja e uma pilha de dois — e aposto BAIXO na
contagem porque so estou entregando oito codecs, e o censo diz que o perfil
maior (`novas_uma_cor` + `preserva`) tem 37 tarefas que precisam de um codec de
condicao que eu ainda nao sei escrever.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict, deque
from math import log2
from typing import Dict, List, Optional, Set, Tuple

from novaphys.arc import Grid, carregar, dims
from novaphys.compositor import fundo
from novaphys import forma
from novaphys.eter import PASTA_AVAL

ORTO = ((-1, 0), (1, 0), (0, -1), (0, 1))
BITS_COR = log2(10)
Cel = Tuple[int, int]


def _todas(g: Grid):
    h, w = dims(g)
    return {(i, j) for i in range(h) for j in range(w)}


# FATOR_EXC: multiplicador do preco da excecao. MEDIDO em 17/09/2026 e NAO E
# ALAVANCA — varredura de 0,5 a 8,0 (16x), teto casado, com 8 pares promovidos
# disponiveis:
#     fator  0,5 -> base 12, pares em pilha 0
#            1,0 -> base 12, pares em pilha 1
#            2,0 -> base 11, pares em pilha 1
#            4,0 -> base 11, pares em pilha 1
#            8,0 -> base 11, pares em pilha 1
# Encarecer NAO faz mecanismo entrar, e DERRUBA a base (quebra tarefa que
# legitimamente precisa de poucas excecoes). Barateando tambem nao ajuda.
# A hipotese "excecao ganha de mecanismo" esta REFUTADA: medindo os dois lados
# da conta, o par poupa 7-11x MAIS bits do que custa. Ver [[sala-de-operacoes-nulo]].
# Deixado em 1.0 = comportamento original, inerte. NAO re-testar.
FATOR_EXC = 1.0


def bits_excecao(n: int, h: int, w: int) -> float:
    """Custo de listar n celulas com cor arbitraria."""
    return FATOR_EXC * n * (log2(max(h * w, 2)) + BITS_COR)


# ---------------------------------------------------------------------------
# CODECS. Cada um: propor(treino) -> [modelo], e prever(modelo, a) -> {cel: cor}
# `prever` e PARCIAL: a ausencia de uma celula significa "nao reivindico".
# ---------------------------------------------------------------------------

def objetos(g: Grid, bg: int):
    h, w = dims(g)
    por_cor: Dict[int, Set[Cel]] = {}
    for i in range(h):
        for j in range(w):
            if g[i][j] != bg:
                por_cor.setdefault(g[i][j], set()).add((i, j))
    fora = []
    for k, s in por_cor.items():
        s = set(s)
        while s:
            p = s.pop()
            c = [p]
            f = deque([p])
            while f:
                y, x = f.popleft()
                for dy, dx in ORTO:
                    q = (y + dy, x + dx)
                    if q in s:
                        s.discard(q)
                        c.append(q)
                        f.append(q)
            fora.append((frozenset(c), k))
    return fora


def cercadas(g: Grid, bg: int) -> Set[Cel]:
    h, w = dims(g)
    vis = [[False] * w for _ in range(h)]
    f = deque()
    for i in range(h):
        for j in (0, w - 1):
            if g[i][j] == bg and not vis[i][j]:
                vis[i][j] = True
                f.append((i, j))
    for j in range(w):
        for i in (0, h - 1):
            if g[i][j] == bg and not vis[i][j]:
                vis[i][j] = True
                f.append((i, j))
    while f:
        y, x = f.popleft()
        for dy, dx in ORTO:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not vis[ny][nx] and g[ny][nx] == bg:
                vis[ny][nx] = True
                f.append((ny, nx))
    return {(i, j) for i in range(h) for j in range(w)
            if g[i][j] == bg and not vis[i][j]}


class Copia:
    """A saida repete a entrada. O codec mais barato e o mais util: ele come o
    fundo inteiro e deixa so o que MUDOU como residuo."""
    nome = "copia"

    def propor(self, treino, alvo=None):
        return [None]

    def bits(self, m):
        return 1.0

    def prever(self, m, a):
        h, w = dims(a)
        return {(i, j): a[i][j] for i in range(h) for j in range(w)}


class MapaCor:
    """Substituicao global de cor, deduzida do treino."""
    nome = "mapa_cor"

    def propor(self, treino, alvo=None):
        mp: Dict[int, int] = {}
        for k, (a, b) in enumerate(treino):
            h, w = dims(a)
            al = None if alvo is None else alvo[k]
            for i in range(h):
                for j in range(w):
                    if al is not None and (i, j) not in al:
                        continue      # ja explicado: nao e problema meu
                    c, d = a[i][j], b[i][j]
                    if c in mp and mp[c] != d:
                        return []
                    mp[c] = d
        mp = {c: d for c, d in mp.items() if c != d}
        return [tuple(sorted(mp.items()))] if mp else []

    def bits(self, m):
        return len(m) * 2 * BITS_COR

    def prever(self, m, a):
        mp = dict(m)
        h, w = dims(a)
        return {(i, j): mp[a[i][j]] for i in range(h) for j in range(w)
                if a[i][j] in mp}


class Preenche:
    """Regiao de fundo cercada recebe uma cor."""
    nome = "preenche"

    def propor(self, treino, alvo=None):
        cs = set()
        for a, b in treino:
            bg = fundo(a)
            for i, j in cercadas(a, bg):
                cs.add(b[i][j])
        return [c for c in sorted(cs)]

    def bits(self, m):
        return BITS_COR

    def prever(self, m, a):
        return {p: m for p in cercadas(a, fundo(a))}


class D4:
    """A saida e a entrada girada ou espelhada."""
    nome = "d4"

    def propor(self, treino, alvo=None):
        return list(range(8))

    def bits(self, m):
        return 3.0

    def prever(self, m, a):
        g = [list(l) for l in a]
        if m >= 4:
            g = [l[::-1] for l in g]
        for _ in range(m % 4):
            g = [list(r) for r in zip(*g[::-1])]
        if (len(g), len(g[0])) != dims(a):
            return {}
        return {(i, j): g[i][j] for i in range(len(g)) for j in range(len(g[0]))}


class Simetriza:
    """Completa a simetria: onde a entrada tem fundo, le o espelho."""
    nome = "simetriza"

    def propor(self, treino, alvo=None):
        return ["h", "v", "hv"]

    def bits(self, m):
        return 2.0

    def prever(self, m, a):
        bg = fundo(a)
        h, w = dims(a)
        fora = {}
        for i in range(h):
            for j in range(w):
                if a[i][j] != bg:
                    continue
                for esp in (m if m != "hv" else "hv"):
                    y, x = (i, w - 1 - j) if esp == "h" else (h - 1 - i, j)
                    if a[y][x] != bg:
                        fora[(i, j)] = a[y][x]
                        break
        return fora


class Atracao:
    """A peca escorrega ate encostar; a direcao e a que bate em objeto."""
    nome = "atracao"

    def propor(self, treino, alvo=None):
        ms = None
        for a, b in treino:
            bg = fundo(a)
            h, w = dims(a)
            cs = frozenset(a[i][j] for i in range(h) for j in range(w)
                           if a[i][j] != bg and b[i][j] != a[i][j])
            if not cs:
                return []
            ms = cs if ms is None else (ms & cs)
        if not ms:
            return []
        return [(ms, 4), (ms, 8)]

    def bits(self, m):
        return len(m[0]) * BITS_COR + 1

    def prever(self, m, a):
        moveis, nd = m
        bg = fundo(a)
        h, w = dims(a)
        objs = objetos(a, bg)
        dirs = ORTO if nd == 4 else ORTO + ((-1, -1), (-1, 1), (1, -1), (1, 1))
        fixos = {q for c, k in objs if k not in moveis for q in c}
        if not fixos:
            return {}
        fora: Dict[Cel, int] = {}
        for c, k in objs:
            if k not in moveis:
                continue
            cand = []
            for dy, dx in dirs:
                p = 0
                while True:
                    nv = [(y + dy * (p + 1), x + dx * (p + 1)) for y, x in c]
                    if any(not (0 <= y < h and 0 <= x < w) for y, x in nv):
                        p = -1
                        break
                    if any(q in fixos for q in nv):
                        break
                    p += 1
                if p > 0:
                    cand.append((p, (dy, dx)))
            if not cand:
                continue
            cand.sort()
            p, (dy, dx) = cand[0]
            for y, x in c:
                fora.setdefault((y, x), bg)
            for y, x in c:
                fora[(y + dy * p, x + dx * p)] = k
        return fora


class Constante:
    """Uma cor unica para tudo que sobrou."""
    nome = "constante"

    def propor(self, treino, alvo=None):
        cs = Counter()
        for a, b in treino:
            h, w = dims(a)
            for i in range(h):
                for j in range(w):
                    cs[b[i][j]] += 1
        return [c for c, _n in cs.most_common(3)]

    def bits(self, m):
        return BITS_COR

    def prever(self, m, a):
        h, w = dims(a)
        return {(i, j): m for i in range(h) for j in range(w)}


class LinhaObjeto:
    """Cada objeto pinta a linha e a coluna inteiras com a sua cor."""
    nome = "cruz"

    def propor(self, treino, alvo=None):
        return ["linha", "coluna", "cruz"]

    def bits(self, m):
        return 2.0

    def prever(self, m, a):
        bg = fundo(a)
        h, w = dims(a)
        fora = {}
        for c, k in objetos(a, bg):
            if len(c) > 4:
                continue
            for y, x in c:
                if m in ("linha", "cruz"):
                    for j in range(w):
                        fora[(y, j)] = k
                if m in ("coluna", "cruz"):
                    for i in range(h):
                        fora[(i, x)] = k
        return fora


class CorPorPropriedade:
    """
    Recolore cada objeto conforme uma PROPRIEDADE dele. E o perfil maior do
    censo fino: `novas_uma_cor` + `preserva` = 37 tarefas. Em pilha com `copia`
    e o teste mais direto da tese — duas camadas, nenhuma resolve sozinha.
    """
    nome = "cor_por_prop"
    PROPS = ("tam", "forma", "buracos", "altura", "largura", "posto")

    @staticmethod
    def _valor(prop, c, a, bg):
        ys = [y for y, _x in c]
        xs = [x for _y, x in c]
        if prop == "tam":
            return len(c)
        if prop == "altura":
            return max(ys) - min(ys) + 1
        if prop == "largura":
            return max(xs) - min(xs) + 1
        if prop == "forma":
            my, mx = min(ys), min(xs)
            return frozenset((y - my, x - mx) for y, x in c)
        if prop == "buracos":
            cx = {(y - min(ys), x - min(xs)) for y, x in c}
            hh = max(ys) - min(ys) + 1
            ww = max(xs) - min(xs) + 1
            vazio = {(i, j) for i in range(hh) for j in range(ww)} - cx
            vis, n = set(), 0
            for p in vazio:
                if p in vis:
                    continue
                pilha, comp, borda = [p], [], False
                vis.add(p)
                while pilha:
                    y, x = pilha.pop()
                    comp.append((y, x))
                    if y in (0, hh - 1) or x in (0, ww - 1):
                        borda = True
                    for dy, dx in ORTO:
                        q = (y + dy, x + dx)
                        if q in vazio and q not in vis:
                            vis.add(q)
                            pilha.append(q)
                if not borda:
                    n += 1
            return n
        return None

    def propor(self, treino, alvo=None):
        fora = []
        for prop in self.PROPS:
            mp: Dict = {}
            ok = True
            for _p, (a, b) in enumerate(treino):
                bg = fundo(a)
                objs = objetos(a, bg)
                al = None if alvo is None else alvo[_p]
                ordem = sorted(objs, key=lambda o: -len(o[0]))
                for r, (c, _k) in enumerate(ordem):
                    if al is not None and not (c & al):
                        continue      # objeto ja explicado
                    v = r if prop == "posto" else self._valor(prop, c, a, bg)
                    cs = {b[y][x] for y, x in c}
                    if len(cs) != 1:
                        ok = False
                        break
                    d = next(iter(cs))
                    if v in mp and mp[v] != d:
                        ok = False
                        break
                    mp[v] = d
                if not ok:
                    break
            if ok and mp and len(mp) < sum(len(objetos(a, fundo(a)))
                                           for a, _b in treino):
                fora.append((prop, tuple(sorted(mp.items(), key=repr))))
        return fora

    def bits(self, m):
        return len(m[1]) * (BITS_COR + 4) + 3

    def prever(self, m, a):
        prop, itens = m
        mp = dict(itens)
        bg = fundo(a)
        objs = objetos(a, bg)
        ordem = sorted(objs, key=lambda o: -len(o[0]))
        fora = {}
        for r, (c, _k) in enumerate(ordem):
            v = r if prop == "posto" else self._valor(prop, c, a, bg)
            if v in mp:
                for y, x in c:
                    fora[(y, x)] = mp[v]
        return fora


class Sovaco:
    """O caminho sai pelo canto concavo. Vem do relato falado do usuario."""
    nome = "sovaco"

    def propor(self, treino, alvo=None):
        return ["mutuo", "sovaco", "ocupada"]

    def bits(self, m):
        return 2.0

    def prever(self, m, a):
        bg = fundo(a)
        h, w = dims(a)
        sv = {}
        for i in range(h):
            for j in range(w):
                if a[i][j] != bg:
                    continue
                oc = [(dy, dx) for dy, dx in ORTO
                      if 0 <= i + dy < h and 0 <= j + dx < w
                      and a[i + dy][j + dx] != bg]
                if len(oc) < 2:
                    continue
                dy = -sum(d[0] for d in oc)
                dx = -sum(d[1] for d in oc)
                if dy and dx:
                    sv[(i, j)] = (dy // abs(dy), dx // abs(dx))
        fora = {}
        for (i, j), (dy, dx) in sv.items():
            y, x, cam = i, j, []
            while 0 <= y < h and 0 <= x < w and a[y][x] == bg:
                cam.append((y, x))
                y, x = y + dy, x + dx
            if not cam:
                continue
            ok = False
            if 0 <= y < h and 0 <= x < w and a[y][x] != bg:
                ok = m == "ocupada"
            if not ok and m in ("sovaco", "mutuo"):
                alvo = sv.get(cam[-1])
                if alvo is not None and cam[-1] != (i, j):
                    ok = m == "sovaco" or alvo == (-dy, -dx)
            if not ok:
                continue
            cor = None
            for ddy, ddx in ORTO:
                ny, nx = i + ddy, j + ddx
                if 0 <= ny < h and 0 <= nx < w and a[ny][nx] != bg:
                    cor = a[ny][nx]
                    break
            for p in cam:
                fora[p] = cor if cor is not None else 1
        return fora


class BuracoDono:
    """Regiao cercada recebe a cor do objeto que a cerca."""
    nome = "buraco_dono"

    def propor(self, treino, alvo=None):
        return [None]

    def bits(self, m):
        return 2.0

    def prever(self, m, a):
        bg = fundo(a)
        h, w = dims(a)
        fora = {}
        for i, j in cercadas(a, bg):
            viz = Counter()
            for dy, dx in ORTO:
                y, x = i + dy, j + dx
                if 0 <= y < h and 0 <= x < w and a[y][x] != bg:
                    viz[a[y][x]] += 1
            if viz:
                fora[(i, j)] = viz.most_common(1)[0][0]
        return fora


class Periodo:
    """
    O fundo e preenchido pelo padrao periodico do resto da grade. E a familia
    do "buraco no papel de parede", e compoe muito bem com `copia`.
    """
    nome = "periodo"

    def propor(self, treino, alvo=None):
        return [None]

    def bits(self, m):
        return 6.0

    def prever(self, m, a):
        bg = fundo(a)
        h, w = dims(a)
        cheio = [(i, j) for i in range(h) for j in range(w) if a[i][j] != bg]
        if not cheio:
            return {}
        melhor = None
        for py in range(1, h + 1):
            for px in range(1, w + 1):
                if py == h and px == w:
                    continue
                mp: Dict[Cel, int] = {}
                ok = True
                for i, j in cheio:
                    q = (i % py, j % px)
                    if q in mp and mp[q] != a[i][j]:
                        ok = False
                        break
                    mp[q] = a[i][j]
                if ok and len(mp) == py * px:
                    melhor = (py, px, mp)
                    break
            if melhor:
                break
        if not melhor:
            return {}
        py, px, mp = melhor
        return {(i, j): mp[(i % py, j % px)]
                for i in range(h) for j in range(w) if a[i][j] == bg}


class Gravidade:
    """Os objetos caem numa direcao ate travar."""
    nome = "gravidade"

    def propor(self, treino, alvo=None):
        return list(ORTO)

    def bits(self, m):
        return 2.0

    def prever(self, m, a):
        dy, dx = m
        bg = fundo(a)
        h, w = dims(a)
        objs = sorted(objetos(a, bg),
                      key=lambda o: -max(y * dy + x * dx for y, x in o[0]))
        ocup: Set[Cel] = set()
        fora: Dict[Cel, int] = {}
        for i in range(h):
            for j in range(w):
                fora[(i, j)] = bg
        for c, k in objs:
            p = 0
            while True:
                nv = [(y + dy * (p + 1), x + dx * (p + 1)) for y, x in c]
                if any(not (0 <= y < h and 0 <= x < w) for y, x in nv):
                    break
                if any(q in ocup for q in nv):
                    break
                p += 1
            for y, x in c:
                q = (y + dy * p, x + dx * p)
                ocup.add(q)
                fora[q] = k
        return fora


class Conecta:
    """Liga pares de celulas da mesma cor alinhadas em linha ou coluna."""
    nome = "conecta"

    def propor(self, treino, alvo=None):
        cs = {None}
        for _a, b in treino:
            h, w = dims(b)
            cs |= {b[i][j] for i in range(h) for j in range(w)}
        return sorted(cs, key=lambda x: (x is not None, x))

    def bits(self, m):
        return BITS_COR + 1

    def prever(self, m, a):
        bg = fundo(a)
        h, w = dims(a)
        fora: Dict[Cel, int] = {}
        for i in range(h):
            oc = [j for j in range(w) if a[i][j] != bg]
            for u, v in zip(oc, oc[1:]):
                if a[i][u] == a[i][v] and v > u + 1:
                    if all(a[i][j] == bg for j in range(u + 1, v)):
                        for j in range(u + 1, v):
                            fora[(i, j)] = a[i][u] if m is None else m
        for j in range(w):
            oc = [i for i in range(h) if a[i][j] != bg]
            for u, v in zip(oc, oc[1:]):
                if a[u][j] == a[v][j] and v > u + 1:
                    if all(a[i][j] == bg for i in range(u + 1, v)):
                        for i in range(u + 1, v):
                            fora[(i, j)] = a[u][j] if m is None else m
        return fora


class Contorno:
    """Desenha a moldura de cada objeto."""
    nome = "contorno"

    def propor(self, treino, alvo=None):
        cs = set()
        for _a, b in treino:
            h, w = dims(b)
            cs |= {b[i][j] for i in range(h) for j in range(w)}
        return sorted(cs)

    def bits(self, m):
        return BITS_COR

    def prever(self, m, a):
        bg = fundo(a)
        h, w = dims(a)
        fora = {}
        for i in range(h):
            for j in range(w):
                if a[i][j] != bg:
                    continue
                if any(0 <= i + dy < h and 0 <= j + dx < w
                       and a[i + dy][j + dx] != bg
                       for dy in (-1, 0, 1) for dx in (-1, 0, 1)):
                    fora[(i, j)] = m
        return fora


class ApagaPorPropriedade:
    """Apaga os objetos cuja propriedade cai num conjunto lido do treino."""
    nome = "apaga_prop"
    PROPS = ("tam", "cor", "forma")

    def propor(self, treino, alvo=None):
        fora = []
        for prop in self.PROPS:
            vs = set()
            for a, b in treino:
                bg = fundo(a)
                for c, k in objetos(a, bg):
                    if all(b[y][x] == bg for y, x in c):
                        if prop == "cor":
                            vs.add(k)
                        elif prop == "tam":
                            vs.add(len(c))
                        else:
                            ys = [y for y, _x in c]
                            xs = [x for _y, x in c]
                            vs.add(frozenset((y - min(ys), x - min(xs))
                                             for y, x in c))
            if vs:
                fora.append((prop, tuple(sorted(vs, key=repr))))
        return fora

    def bits(self, m):
        return len(m[1]) * 5 + 2

    def prever(self, m, a):
        prop, vs = m
        alvo = set(vs)
        bg = fundo(a)
        fora = {}
        for c, k in objetos(a, bg):
            if prop == "cor":
                v = k
            elif prop == "tam":
                v = len(c)
            else:
                ys = [y for y, _x in c]
                xs = [x for _y, x in c]
                v = frozenset((y - min(ys), x - min(xs)) for y, x in c)
            if v in alvo:
                for y, x in c:
                    fora[(y, x)] = bg
        return fora


class Moldura:
    """Pinta a borda da grade."""
    nome = "moldura"

    def propor(self, treino, alvo=None):
        cs = set()
        for _a, b in treino:
            h, w = dims(b)
            cs |= {b[0][j] for j in range(w)} | {b[h - 1][j] for j in range(w)}
        return sorted(cs)

    def bits(self, m):
        return BITS_COR

    def prever(self, m, a):
        h, w = dims(a)
        fora = {}
        for j in range(w):
            fora[(0, j)] = m
            fora[(h - 1, j)] = m
        for i in range(h):
            fora[(i, 0)] = m
            fora[(i, w - 1)] = m
        return fora


class Vizinhanca:
    """
    Regra celular: a cor sai da CONTAGEM de vizinhos ocupados. E o codec mais
    barato que expressa condicao local, e o censo dizia que faltava condicao.
    """
    nome = "vizinhanca"

    def propor(self, treino, alvo=None):
        for k8 in (False, True):
            mp: Dict[Tuple[int, int], int] = {}
            ok = True
            for _k, (a, b) in enumerate(treino):
                bg = fundo(a)
                h, w = dims(a)
                al = None if alvo is None else alvo[_k]
                for i in range(h):
                    for j in range(w):
                        if al is not None and (i, j) not in al:
                            continue
                        n = 0
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                if dy == 0 and dx == 0:
                                    continue
                                if not k8 and dy and dx:
                                    continue
                                y, x = i + dy, j + dx
                                if 0 <= y < h and 0 <= x < w and a[y][x] != bg:
                                    n += 1
                        ch = (a[i][j], n)
                        if ch in mp and mp[ch] != b[i][j]:
                            ok = False
                            break
                        mp[ch] = b[i][j]
                    if not ok:
                        break
                if not ok:
                    break
            if ok and mp:
                yield (k8, tuple(sorted(mp.items())))

    def bits(self, m):
        return len(m[1]) * (BITS_COR + 4) + 1

    def prever(self, m, a):
        k8, itens = m
        mp = dict(itens)
        bg = fundo(a)
        h, w = dims(a)
        fora = {}
        for i in range(h):
            for j in range(w):
                n = 0
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        if not k8 and dy and dx:
                            continue
                        y, x = i + dy, j + dx
                        if 0 <= y < h and 0 <= x < w and a[y][x] != bg:
                            n += 1
                ch = (a[i][j], n)
                if ch in mp:
                    fora[(i, j)] = mp[ch]
        return fora



# ---------------------------------------------------------------------------
# A BUSCA DE CONJUNCAO, com quatro ordens e orcamento casado.
#
# Todo candidato do `nec` e VERDADEIRO em todas as celulas-alvo, entao toda
# intersecao CONTEM o alvo. O erro e so falso positivo, e adicionar teste so
# encolhe. Achar a conjuncao exata e portanto cobrir os falsos positivos —
# a estrutura que separa uma ordem boa de uma ordem cega.
#
# MODO_CONJ escolhe a ordem. Nenhum braco avalia mais conjuncoes que o outro.
# ---------------------------------------------------------------------------

MODO_CONJ = "guloso"
ORC_CONJ = 1200
DIM_HDC = 4096
PROF_CONJ = 5


def _inter(cs, T):
    r = None
    for c in cs:
        r = set(T[c]) if r is None else (r & T[c])
    return r


def _erro(cs, Ts, alvos):
    """Falsos positivos totais. Zero = conjuncao exata."""
    n = 0
    for T, A in zip(Ts, alvos):
        r = _inter(cs, T)
        if r is None:
            return None
        n += len(r - A)
    return n


def _pool(nec, rnd, quanto):
    import itertools
    fora = []
    for k in (1, 2):
        for cs in itertools.combinations(range(len(nec)), k):
            fora.append(cs)
            if len(fora) >= quanto:
                return fora
    for _ in range(quanto - len(fora)):
        k = rnd.randint(3, min(PROF_CONJ, max(3, len(nec))))
        if k > len(nec):
            continue
        fora.append(tuple(sorted(rnd.sample(range(len(nec)), k))))
    return fora


def busca_conjuncao(nec, Ts, alvos):
    import random
    rnd = random.Random(20260909)
    modo = MODO_CONJ

    if modo == "guloso":
        # cobertura gulosa: a cada passo, o teste que mata mais falso positivo
        cs, orc = [], 0
        for _ in range(PROF_CONJ):
            e0 = _erro(tuple(cs), Ts, alvos) if cs else None
            if e0 == 0:
                break
            melhor, me = None, e0
            for i, c in enumerate(nec):
                if i in cs:
                    continue
                orc += 1
                if orc > ORC_CONJ:
                    break
                e = _erro(tuple(cs + [i]), Ts, alvos)
                if e is not None and (me is None or e < me):
                    melhor, me = i, e
            if melhor is None:
                break
            cs.append(melhor)
            if me == 0:
                return tuple(nec[i] for i in cs)
        return None

    cand = _pool(nec, rnd, ORC_CONJ * 2)
    if modo == "aleatorio":
        rnd.shuffle(cand)
    if modo == "hdc":
        # semeia, aprende prototipos BOM e RUIM, ordena o resto por
        # sim(v,BOM) - sim(v,RUIM). E o protocolo do `prion_hdc.py`, que e o
        # unico mecanismo de busca que ja pagou neste projeto.
        vt = [[1 if rnd.random() < 0.5 else -1 for _ in range(DIM_HDC)]
              for _ in nec]
        def vec(cs):
            v = [0] * DIM_HDC
            for i in cs:
                w = vt[i]
                for d in range(DIM_HDC):
                    v[d] += w[d]
            return v
        def sim(u, w):
            s = sum(x * y for x, y in zip(u, w))
            nu = sum(x * x for x in u) ** 0.5
            nw = sum(y * y for y in w) ** 0.5
            return s / (nu * nw) if nu and nw else 0.0
        semente = min(len(cand) // 4, ORC_CONJ // 3)
        aval = []
        for cs in cand[:semente]:
            e = _erro(tuple(nec[i] for i in cs), Ts, alvos)
            if e == 0:
                return tuple(nec[i] for i in cs)
            if e is not None:
                aval.append((e, cs))
        if aval:
            aval.sort()
            k = max(1, len(aval) // 5)
            bom = [0] * DIM_HDC
            ruim = [0] * DIM_HDC
            for _e, cs in aval[:k]:
                v = vec(cs)
                for d in range(DIM_HDC):
                    bom[d] += v[d]
            for _e, cs in aval[-k:]:
                v = vec(cs)
                for d in range(DIM_HDC):
                    ruim[d] += v[d]
            resto = cand[semente:]
            resto.sort(key=lambda cs: -(sim(vec(cs), bom) - sim(vec(cs), ruim)))
            cand = cand[:semente] + resto
        orc = semente
    else:
        orc = 0

    for cs in cand:
        if orc >= ORC_CONJ:
            break
        orc += 1
        if _erro(tuple(nec[i] for i in cs), Ts, alvos) == 0:
            return tuple(nec[i] for i in cs)
    return None


_CACHE_T: Dict = {}
# GANCHO DE PROPONENTE (2026-09-12). Quem decide a ORDEM em que os candidatos
# sao avaliados. None = comportamento historico (ordem dos CODECS). Um
# proponente precisa de .ordem(ctx, cand) -> lista reordenada e
# .registra(ctx, item, ganho). O teto de avaliacoes e o mesmo para todos,
# entao o que se mede e so a qualidade da ordem.
PROPONENTE = None
# GANCHO DE PRIOR (2026-09-12). A ideia do usuario: a AGI extrai BITS da LLM,
# nao texto. -log2 p_LLM(codec | estado) E um comprimento de descricao, entao a
# LLM vira um LIVRO DE CODIGOS que o criterio MDL consome na moeda dele.
# Diferenca crucial em relacao ao PROPONENTE: aquele so muda a ORDEM de
# avaliacao (e deu nulo limpo, 40/400 em quatro bracos); este muda o CRITERIO —
# um codec plausivel fica genuinamente mais barato e pode ganhar uma disputa
# apertada. Um prior errado nao corrompe: ele barateia uma descricao que paga
# caro no residuo, e perde. None = comportamento historico.
# Precisa de .total(pilha) -> float (bits) e .pergunta(contexto).
PRIOR = None


# ESTADO DA PILHA COMO VOCABULARIO.
# Ate aqui o `refoco` passava o RESIDUO — o que falta explicar. Mas a estrutura
# do que JA foi explicado nunca virava feature: o sistema descobria que
# `raio_de(sovaco_mutuo)` fecha 7 celulas e esquecia o fato no passo seguinte.
# Aqui as proprias descobertas entram no vocabulario, dentro da mesma tarefa.
# Os tres testes sao definiveis nos DOIS regimes — na deducao "explicada" e o
# complemento do residuo, na aplicacao e o que os codecs anteriores da pilha ja
# reivindicaram — o que e a condicao para nao virar vazamento.
_EST: Optional[List[Set]] = None


def _com_estado(T, k: int):
    if _EST is None or k >= len(_EST):
        return T
    ex = _EST[k]
    T = dict(T)
    T[("explicada",)] = set(ex)
    T[("nao_explicada",)] = {p for p in T[("fundo",)] | T[("ocupada",)]
                             if p not in ex}
    T[("viz_explicada",)] = {(i + dy, j + dx) for i, j in ex
                             for dy, dx in ORTO}
    return {c: v for c, v in T.items() if v}
DERIVA = True


_COMB = ("nao", "desl", "raio", "conta", "lin_tem", "col_tem")
# PROFUNDIDADE 2 (2026-09-11): negar um teste DERIVADO. Achado no `laco_grade.py`:
# a lei da lente precisa de "abaixo de uma particula E SEM massa na diagonal de
# cima" — os deslocamentos diagonais existem, mas `nao` so se aplicava a sementes.
# DESLIGADO por padrao: nada muda em nenhuma medida anterior.
# RESULTADO do teste de viabilidade (lente, semente fora do registro): a premissa
# era FALSA — a profundidade 1 ja exprime a lei por contagem (`viz8 = 1`), e com
# 16 cenas acerta 20/20 cenas novas. Ligado, isto achou uma lei MAIS BARATA que
# fecha o treino e erra 6/20 (com 8 e com 16 cenas). Fica desligado.
NAO_DERIVADO = False


def _bits_teste(c) -> float:
    """Teste derivado custa mais que teste base. E o MDL, e nao uma regra minha,
    que decide se a profundidade extra se paga. Semente 12, profundidade 1 = 18
    (como sempre), profundidade 2 = 24 (so existe com NAO_DERIVADO)."""
    if not c or c[0] not in _COMB:
        return 12.0
    if c[0] == "nao" and len(c) > 1 and c[1] in _COMB:
        return 24.0
    return 18.0


def _derivar(T, g, bg):
    """
    A GRAMATICA DE TESTES. Em vez de uma LISTA de ~50 testes escritos por mim,
    um conjunto pequeno de sementes fechado sob combinadores. O sistema passa a
    ter testes que ninguem escreveu.

        nao(t)            complemento
        desl(t, d)        a celula vizinha na direcao d satisfaz t
        raio(t, d)        existe ALGUMA celula na direcao d que satisfaz t
        conta(t, n)       exatamente n vizinhos ortogonais satisfazem t
        lin_tem(t)        a linha da celula contem alguma celula com t
        col_tem(t)        idem para a coluna

    Oito sementes x vinte e quatro combinadores = ~190 testes derivados, e o
    `raio` sozinho ja expressa coisas que a lista nao expressava: "existe algo
    ocupado acima de mim", que e a leitura direcional que o `alcance.py` mediu
    como faltante e que o objetivo quantificado provou valer 49,6% -> 99,3%.

    E O QUE ISTO NAO E: os combinadores ainda sao meus. O sistema gera testes
    que eu nao escrevi, a partir de uma gramatica que eu escrevi. Cada nivel
    desses move a contribuicao humana um degrau para cima — de codec, para
    linguagem de codec, para gramatica de teste — e nao ha nivel em que ela
    desapareca. Chamar isso de auto-expansao e honesto; chamar de auto-criacao
    do nada nao seria.
    """
    h, w = dims(g)
    cels = [(i, j) for i in range(h) for j in range(w)]
    U = set(cels)
    sem = [k for k in T if k[0] in ("fundo", "ocupada", "cercada")]
    sem += [k for k in T if k[0] == "cor"][:5]
    D: Dict = {}
    DIRS = (("N", -1, 0), ("S", 1, 0), ("O", 0, -1), ("L", 0, 1),
            ("NO", -1, -1), ("NL", -1, 1), ("SO", 1, -1), ("SL", 1, 1))
    for k in sem:
        S = T[k]
        if not S or len(S) == len(U):
            continue
        D[("nao",) + k] = U - S
        for nome, dy, dx in DIRS:
            D[("desl", nome) + k] = {(i, j) for i, j in cels
                                     if (i + dy, j + dx) in S}
            # varredura em O(celulas): raio[p] = (p+d in S) ou raio[p+d]
            R = set()
            fy = range(h - 1, -1, -1) if dy > 0 else range(h)
            fx = range(w - 1, -1, -1) if dx > 0 else range(w)
            for i in fy:
                for j in fx:
                    q = (i + dy, j + dx)
                    if q in S or q in R:
                        R.add((i, j))
            D[("raio", nome) + k] = R
        for n in range(0, 5):
            D[("conta", n) + k] = {
                (i, j) for i, j in cels
                if sum(1 for dy, dx in ORTO if (i + dy, j + dx) in S) == n}
        lin = {i for i, _j in S}
        col = {j for _i, j in S}
        D[("lin_tem",) + k] = {(i, j) for i, j in cels if i in lin}
        D[("col_tem",) + k] = {(i, j) for i, j in cels if j in col}
    if NAO_DERIVADO:
        for k2, S2 in list(D.items()):
            if k2[0] != "nao" and S2 and len(S2) < len(U):
                D[("nao",) + k2] = U - S2
    return D


class CondicaoAprendida:
    """
    Aprende a CONDICAO que seleciona as celulas, a partir do que a pilha ainda
    erra. E o codec que o censo do residuo pediu, e nao um palpite meu.

    Censo de 91 tarefas com residuo entre 1 e 30, depois de todos os mecanismos
    conhecidos:
        <=3 componentes conexos ... 51 (56%)
        residuo de uma cor so na entrada  48 (53%)
        vira uma cor so na saida .. 36 (40%)
        era FUNDO na entrada ...... 31 (34%)
    Perfil dominante: pintar de UMA cor um conjunto de celulas de FUNDO. A
    operacao ja existe na arquitetura. O que falta e QUAIS celulas — a condicao.

    O modelo e uma CONJUNCAO de ate tres testes por celula mais uma cor. A
    busca nao enumera conjuncoes: coleta os testes que sao verdadeiros em TODAS
    as celulas-alvo (necessarios) e procura um subconjunto pequeno cuja
    intersecao seja exatamente o alvo. Os bits pagam pelo tamanho da conjuncao,
    entao condicao comprida so entra se explicar muita celula — que e o que
    impede a memorizacao que matou o `segmenta` (127 ajustavam, 6 passavam).

    Isto ja foi tentado antes e deu ZERO (`celobj.py`, `condicao.py`). A
    diferenca de regime: la a condicao tinha que explicar a tarefa INTEIRA
    sozinha; aqui ela so precisa explicar o RESIDUO.
    """
    nome = "condicao"

    @staticmethod
    def _testes(g, bg):
        """cada teste -> conjunto de celulas onde ele vale."""
        ch = (g, bg)
        if ch in _CACHE_T:
            return _CACHE_T[ch]
        r = CondicaoAprendida._base(g, bg)
        if DERIVA:
            r.update(_derivar(r, g, bg))
        # 64 e nao 400. Em grades 30x30 cada entrada guarda ~300 conjuntos de
        # ate 900 celulas; com 400 entradas um worker chegou a ~2 GB de RAM, e
        # tres workers podem somar ~6 GB — provavel contribuinte dos travamentos
        # da maquina do usuario. O reuso que importa e DENTRO de uma tarefa
        # (3 a 8 grades x variantes de forma), e 64 cobre isso com folga.
        if len(_CACHE_T) > 64:
            _CACHE_T.clear()
        _CACHE_T[ch] = r
        return r

    @staticmethod
    def _base(g, bg):
        h, w = dims(g)
        T: Dict = {}
        cels = [(i, j) for i in range(h) for j in range(w)]
        T[("fundo",)] = {p for p in cels if g[p[0]][p[1]] == bg}
        T[("ocupada",)] = {p for p in cels if g[p[0]][p[1]] != bg}
        cores = {g[i][j] for i, j in cels}
        for k in cores:
            T[("cor", k)] = {p for p in cels if g[p[0]][p[1]] == k}
        for n in range(0, 5):
            T[("viz4", n)] = set()
            T[("viz8", n)] = set()
        for i, j in cels:
            n4 = sum(1 for dy, dx in ORTO
                     if 0 <= i + dy < h and 0 <= j + dx < w
                     and g[i + dy][j + dx] != bg)
            n8 = sum(1 for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                     if (dy or dx) and 0 <= i + dy < h and 0 <= j + dx < w
                     and g[i + dy][j + dx] != bg)
            T[("viz4", min(n4, 4))].add((i, j))
            T[("viz8", min(n8, 4))].add((i, j))
        T[("viz4+", 1)] = {p for n in (1, 2, 3, 4) for p in T[("viz4", n)]}
        T[("viz4+", 2)] = {p for n in (2, 3, 4) for p in T[("viz4", n)]}
        T[("cercada",)] = cercadas(g, bg)
        # SEGMENTACAO POR BLOCO.
        # O vocabulario so falava de celula e de vizinhanca. Nenhum teste dizia
        # "quantos vivos no MEU bloco" nem "que posicao eu ocupo nele" — e por
        # isso o `critters` (automato de blocos de Margolus) dava razao 1,000:
        # nao havia captura nem PARCIAL, porque particao em bloco nao e uma
        # afirmacao sobre celula. Nao faltava potencia; faltava o eixo.
        for bl in (2, 3):
            cnt: Dict[Tuple[int, int], int] = {}
            for i in range(h):
                for j in range(w):
                    if g[i][j] != bg:
                        cnt[(i // bl, j // bl)] = cnt.get((i // bl,
                                                           j // bl), 0) + 1
            for n in range(0, bl * bl + 1):
                r = {(i, j) for i in range(h) for j in range(w)
                     if cnt.get((i // bl, j // bl), 0) == n}
                if r:
                    T[("bloco_conta", bl, n)] = r
            for k in range(bl * bl):
                T[("bloco_pos", bl, k)] = {
                    (i, j) for i in range(h) for j in range(w)
                    if (i % bl) * bl + (j % bl) == k}
        # "O QUE NAO ESTOU VENDO": a anomalia.
        # Roda a regularidade mais forte da cena como MODELO e seleciona onde
        # ela FALHA. E o mesmo mecanismo do residuo do MDL, aplicado DENTRO de
        # uma cena em vez de ATRAVES da pilha — a expectativa menos a cena.
        # Em fisica e a mesma coisa: o residuo do modelo e onde mora o que o
        # modelo nao sabe.
        for e in ("h", "v", "d1", "d2"):
            q = set()
            for i in range(h):
                for j in range(w):
                    y, x = ((i, w - 1 - j) if e == "h" else
                            (h - 1 - i, j) if e == "v" else
                            (j, i) if e == "d1" else (w - 1 - j, h - 1 - i))
                    if 0 <= y < h and 0 <= x < w and g[y][x] != g[i][j]:
                        q.add((i, j))
            # so vale como anomalia se a cena for MAJORITARIAMENTE simetrica
            if q and len(q) < h * w * 0.35:
                T[("quebra_sim", e)] = q
        # quebra de periodo: o menor periodo que explica quase tudo
        melhor = None
        for py in range(1, h + 1):
            for px in range(1, w + 1):
                if py == h and px == w:
                    continue
                mp, mau = {}, 0
                for i in range(h):
                    for j in range(w):
                        q = (i % py, j % px)
                        if q not in mp:
                            mp[q] = g[i][j]
                        elif mp[q] != g[i][j]:
                            mau += 1
                if mau and mau < h * w * 0.25:
                    if melhor is None or mau < melhor[0]:
                        melhor = (mau, py, px, dict(mp))
        if melhor:
            _n, py, px, mp = melhor
            T[("quebra_per",)] = {(i, j) for i in range(h) for j in range(w)
                                  if mp.get((i % py, j % px)) != g[i][j]}
        # a peca fora do padrao: a forma que aparece uma vez so
        objs = objetos(g, bg)
        if len(objs) >= 3:
            ass = Counter()
            for c, _k in objs:
                ys = [y for y, _x in c]
                xs = [x for _y, x in c]
                ass[frozenset((y - min(ys), x - min(xs)) for y, x in c)] += 1
            un = set()
            for c, _k in objs:
                ys = [y for y, _x in c]
                xs = [x for _y, x in c]
                f = frozenset((y - min(ys), x - min(xs)) for y, x in c)
                if ass[f] == 1:
                    un |= set(c)
            if un and len(un) < sum(len(c) for c, _k in objs):
                T[("forma_unica",)] = un
            cnt = Counter(k for _c, k in objs)
            if len(cnt) >= 2:
                raro = min(cnt, key=lambda k: (cnt[k], k))
                T[("cor_rara",)] = {p for c, k in objs if k == raro for p in c}
        for L in LEITURAS:
            for t in TERMS:
                r = raios(L + t, g, bg)
                if r:
                    T[("raio_de", L + t)] = r
        objs = objetos(g, bg)
        linhas: Dict[int, Set[int]] = {}
        colunas: Dict[int, Set[int]] = {}
        caixas: Dict[int, Set] = {}
        for c, k in objs:
            linhas.setdefault(k, set()).update(y for y, _x in c)
            colunas.setdefault(k, set()).update(x for _y, x in c)
            ys = [y for y, _x in c]
            xs = [x for _y, x in c]
            caixas.setdefault(k, set()).update(
                (y, x) for y in range(min(ys), max(ys) + 1)
                for x in range(min(xs), max(xs) + 1))
        for k in linhas:
            T[("linha_de", k)] = {p for p in cels if p[0] in linhas[k]}
            T[("coluna_de", k)] = {p for p in cels if p[1] in colunas[k]}
            T[("caixa_de", k)] = set(caixas[k])
        T[("borda",)] = {p for p in cels
                         if p[0] in (0, h - 1) or p[1] in (0, w - 1)}
        T[("lin_par",)] = {p for p in cels if p[0] % 2 == 0}
        T[("col_par",)] = {p for p in cels if p[1] % 2 == 0}
        # entre dois ocupados na mesma linha / coluna
        ent = set()
        for i in range(h):
            oc = [j for j in range(w) if g[i][j] != bg]
            if len(oc) >= 2:
                ent |= {(i, j) for j in range(oc[0] + 1, oc[-1])}
        T[("entre_lin",)] = ent
        ent = set()
        for j in range(w):
            oc = [i for i in range(h) if g[i][j] != bg]
            if len(oc) >= 2:
                ent |= {(i, j) for i in range(oc[0] + 1, oc[-1])}
        T[("entre_col",)] = ent
        return T

    def propor(self, treino, alvo=None):
        if alvo is None:
            return []
        alvos, Ts = [], []
        cores = set()
        for k, (a, b) in enumerate(treino):
            al = alvo[k]
            if not al or len(al) > 200:
                return []
            cores |= {b[y][x] for y, x in al}
            alvos.append(set(al))
            Ts.append(self._testes(a, fundo(a)))
        if len(cores) != 1:
            return []
        cor = next(iter(cores))
        # testes NECESSARIOS: valem em todas as celulas-alvo de todos os pares
        nec = [c for c in Ts[0]
               if all(c in T and A <= T[c] for T, A in zip(Ts, alvos))]
        if not nec:
            return []
        cs = busca_conjuncao(nec, Ts, alvos)
        return [(cs, cor)] if cs else []

    def bits(self, m):
        return sum(_bits_teste(c) for c in m[0]) + BITS_COR

    def prever(self, m, a):
        cs, cor = m
        T = _com_estado(self._testes(a, fundo(a)), 0)
        r = None
        for c in cs:
            if c not in T:
                return {}
            r = set(T[c]) if r is None else (r & T[c])
        return {p: cor for p in (r or ())}


def _carrega_biblioteca():
    """
    A biblioteca de abstracoes que o `abstrai.py` extraiu do proprio corpus por
    anti-unificacao. Ela muda DUAS coisas, e as duas sao o mecanismo do Stitch:

    1. BITS. Um programa que casa uma abstracao paga so os BURACOS — o
       esqueleto ja foi pago uma vez na biblioteca. Entao ele fica mais barato
       e ganha comparacoes de MDL que antes perdia.
    2. SEMENTE. A busca comeca com o esqueleto ja aplicado e o pintor fixo, em
       vez de comecar do zero. Espaco menor, e semeado num ponto que ja provou
       servir noutra tarefa.

    O (2) e a compressao prospectiva implementada: o util no passado e tentado
    primeiro no futuro, e o criterio de aceitacao continua sendo bits.
    """
    import json
    import os
    for c in ("biblioteca_abstracoes.json",
              os.path.join(os.path.dirname(__file__), "..",
                           "biblioteca_abstracoes.json")):
        try:
            d = json.loads(io.open(c, encoding="utf-8").read())
            return [(frozenset(tuple(t) for t in a["comum"]),
                     a["buracos"], tuple(a["pintor"])) for a in d]
        except Exception:
            continue
    return []


import io

BIBLIOTECA = _carrega_biblioteca()


def usar_biblioteca(arq):
    """Troca a biblioteca em uso. `None` desliga (braco de base)."""
    global BIBLIOTECA
    import json
    if arq is None:
        BIBLIOTECA = []
        return
    d = json.loads(io.open(arq, encoding="utf-8").read())
    BIBLIOTECA = [(frozenset(tuple(x) for x in a["comum"]),
                   a["buracos"], tuple(a["pintor"])) for a in d]


DIAG = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def leitura(nome, g: Grid, bg: int) -> Dict[Cel, Tuple[int, int]]:
    """
    LEITURA: uma direcao CALCULADA PELA CENA, nao escolhida pela busca.

    A diferenca entre parametro livre e leitura e a lei mais estavel deste
    projeto. Parametro livre multiplica o espaco de BUSCA — 15 tentativas,
    zero. Leitura multiplica o espaco EXPRESSIVO — 6 tentativas, +22.

    `prop(1,0)` tinha a direcao cozida como constante, entao "propague para onde
    a forma abre" era inexprimivel — nao por faltar a direcao na lista, mas por
    ela ter que ser lida no momento da APLICACAO e nao no da DEDUCAO.

    Duas destas leituras estavam presas dentro de motores sob medida:

      sovaco   e a regra que o usuario verbalizou resolvendo `55059096`:
               "o caminho sai pelo sovaco da cruz, e os dois sovacos precisam
               estar na mesma direcao". Como motor: precisao 100%, UMA tarefa.
               Como leitura: vale em toda tarefa.
      ponta    e a leitura do `atracao.py`, onde a direcao saia por peca.

    Devolve as celulas-FONTE e a direcao de cada uma.
    """
    h, w = dims(g)
    fora: Dict[Cel, Tuple[int, int]] = {}
    if nome.startswith("op_"):
        return {p: (-d[0], -d[1])
                for p, d in leitura(nome[3:], g, bg).items()}
    for i in range(h):
        for j in range(w):
            oc = [(dy, dx) for dy, dx in ORTO
                  if 0 <= i + dy < h and 0 <= j + dx < w
                  and g[i + dy][j + dx] != bg]
            vz = [(dy, dx) for dy, dx in ORTO
                  if 0 <= i + dy < h and 0 <= j + dx < w
                  and g[i + dy][j + dx] == bg]
            if nome == "sovaco":
                if g[i][j] != bg or len(oc) < 2:
                    continue
                dy = -sum(d[0] for d in oc)
                dx = -sum(d[1] for d in oc)
                if dy and dx:
                    fora[(i, j)] = (dy // abs(dy), dx // abs(dx))
            elif nome == "ponta":
                if g[i][j] == bg or len(oc) != 1:
                    continue
                fora[(i, j)] = (-oc[0][0], -oc[0][1])
            elif nome == "abertura":
                if g[i][j] == bg or len(vz) != 1:
                    continue
                fora[(i, j)] = vz[0]
            elif nome == "mais_vazio":
                if g[i][j] == bg:
                    continue
                melhor, mk = None, -1
                for dy, dx in ORTO:
                    y, x, k = i + dy, j + dx, 0
                    while 0 <= y < h and 0 <= x < w and g[y][x] == bg:
                        k += 1
                        y, x = y + dy, x + dx
                    if k > mk:
                        melhor, mk = (dy, dx), k
                if melhor and mk > 0:
                    fora[(i, j)] = melhor
    return fora


LEITURAS = ("sovaco", "ponta", "abertura", "mais_vazio",
            "op_sovaco", "op_ponta")

TERMS = ("", "_mutuo", "_ocupada")


def raios(nome: str, g: Grid, bg: int) -> Set[Cel]:
    """
    As celulas que os raios de uma leitura cobrem — uma REGIAO, nao uma pintura.

    Foi aqui que eu errei o desenho na primeira versao: enfiei regiao E cor no
    mesmo pintor, cozinhando a cor da origem. Na `55059096` as fontes batiam
    12/12 com o `sovaco.py` e o resultado dava zero, porque o motor pinta com
    CONSTANTE 2 lida do diff, e nao com a cor da origem.

    Raio e SELETOR; cor e PINTOR. Sao os dois eixos que a propria medida ja
    tinha separado — os esqueletos que se repetem entre tarefas sao os pintores,
    os seletores sao especificos. Misturar os dois destroi as duas metades.

    A terminacao vem junto porque ela e parte da REGIAO, nao da cor:
        (vazio)    todo raio conta
        _mutuo     so o raio cujo destino e uma fonte que aponta de VOLTA
                   — e o "ambos os sovacos na mesma direcao" do usuario
        _ocupada   so o raio que termina batendo num objeto
    """
    base = nome
    term = ""
    for t in TERMS[1:]:
        if nome.endswith(t):
            base, term = nome[:-len(t)], t
            break
    fontes = leitura(base, g, bg)
    h, w = dims(g)
    fora: Set[Cel] = set()
    for (i, j), (dy, dx) in fontes.items():
        y, x, cam = i, j, []
        while 0 <= y < h and 0 <= x < w and g[y][x] == bg:
            cam.append((y, x))
            y, x = y + dy, x + dx
        if not cam:
            continue
        if term == "_ocupada":
            if not (0 <= y < h and 0 <= x < w and g[y][x] != bg):
                continue
        elif term == "_mutuo":
            alvo = fontes.get(cam[-1])
            if cam[-1] == (i, j) or alvo != (-dy, -dx):
                continue
        fora |= set(cam)
    return fora


class Programa:
    """
    O CODEC INVENTADO. Ate aqui todo codec era uma classe que eu escrevi a mao;
    aqui o codec vira um PROGRAMA numa linguagem pequena, e o sistema procura o
    programa. Inventar deixa de ser autoria e vira busca.

        PINTAR( seletor )  COM ( pintor )

        seletor  conjuncao de ate quatro testes de celula do vocabulario da
                 `CondicaoAprendida` (~50 familias: fundo, cor k, viz4/viz8 = n,
                 cercada, linha_de k, caixa_de k, entre_lin, borda, paridade...)
        pintor   const(c) | copia | dono | vizinho | linha | coluna

    A LINGUAGEM SUBSUME BOA PARTE DO QUE EU ESCREVI A MAO:
        Copia       = PINTAR(tudo)             COM copia
        Preenche    = PINTAR(cercada)          COM const(c)
        Constante   = PINTAR(tudo)             COM const(c)
        BuracoDono  = PINTAR(cercada)          COM vizinho
        Contorno    = PINTAR(fundo & viz8>=1)  COM const(c)
        Moldura     = PINTAR(borda)            COM const(c)
        condicao+constante = UM programa, nao uma pilha de dois

    E gera programas que ninguem escreveu, que e o ponto.

    A BUSCA E COBERTURA, NAO ENUMERACAO
    -------------------------------------
    Para cada pintor P, sobre a base rebaseada:
        OK(P)  celulas onde P acerta a saida
        MAU(P) celulas onde P esta definido e ERRA
    Um programa e SOLIDO quando o seletor nao pega nenhuma celula de MAU. Entao
    a busca e: encolher o seletor ate matar todo MAU, preservando o maximo do
    RESIDUO. Guloso, profundidade 4.

    Nao precisa cobrir tudo — credito parcial. Um programa que explica 30
    celulas do residuo e solido entra na pilha e o proximo ataca o que sobrou.
    E o MDL decide: `len(testes)*12 + bits do pintor` contra os bits que ele
    economiza. Programa comprido so entra se pagar, o que e o que impede a
    memorizacao que matou o `segmenta` (127 ajustavam o treino, 6 passavam).
    """
    nome = "programa"

    @staticmethod
    def _pinta(pin, g, bg):
        """cor prevista por celula, ou ausencia = nao me pronuncio."""
        h, w = dims(g)
        cels = [(i, j) for i in range(h) for j in range(w)]
        t = pin[0]
        if t == "const":
            return {p: pin[1] for p in cels}
        if t == "copia":
            return {p: g[p[0]][p[1]] for p in cels}
        if t == "dono":
            fora = {}
            for c, k in sorted(objetos(g, bg), key=lambda o: -len(o[0])):
                ys = [y for y, _x in c]
                xs = [x for _y, x in c]
                for y in range(min(ys), max(ys) + 1):
                    for x in range(min(xs), max(xs) + 1):
                        fora.setdefault((y, x), k)
            return fora
        if t == "vizinho":
            fora = {}
            for i, j in cels:
                vz = Counter()
                for dy, dx in ORTO:
                    y, x = i + dy, j + dx
                    if 0 <= y < h and 0 <= x < w and g[y][x] != bg:
                        vz[g[y][x]] += 1
                if vz:
                    fora[(i, j)] = vz.most_common(1)[0][0]
            return fora
        if t == "prop":
            # PINTOR QUE PROPAGA. A cor escorre na direcao d: cada celula
            # herda de quem esta a montante, e a heranca ATRAVESSA a grade.
            # Varredura em O(celulas) na ordem do fluxo.
            #   cor[p] = g[p-d]  se ocupada,  senao  cor[p-d]
            # Os seis pintores anteriores eram todos LOCAIS E INSTANTANEOS: liam
            # a celula ou a vizinhanca e emitiam cor. Nenhum tinha consequencia
            # a longa distancia — que e exatamente o que `deducao.py` mediu como
            # inexprimivel, com resultado +0.
            dy, dx = pin[1], pin[2]
            fora = {}
            fy = range(h) if dy >= 0 else range(h - 1, -1, -1)
            fx = range(w) if dx >= 0 else range(w - 1, -1, -1)
            for i in fy:
                for j in fx:
                    y, x = i - dy, j - dx
                    if not (0 <= y < h and 0 <= x < w):
                        continue
                    if g[y][x] != bg:
                        fora[(i, j)] = g[y][x]
                    elif (y, x) in fora:
                        fora[(i, j)] = fora[(y, x)]
            return fora
        if t == "raio_l":
            # RAIO COM DIRECAO LIDA. Cada fonte anda na direcao que a CENA
            # deu a ela, pintando o fundo com a cor da origem. Com
            # leitura="sovaco" isto e o `sovaco.py` inteiro, virado recurso de
            # linguagem em vez de motor de uma tarefa so.
            fontes = leitura(pin[1], g, bg)
            fora = {}
            for (i, j), (dy, dx) in fontes.items():
                cor = g[i][j] if g[i][j] != bg else None
                if cor is None:
                    for ddy, ddx in ORTO:
                        y, x = i + ddy, j + ddx
                        if 0 <= y < h and 0 <= x < w and g[y][x] != bg:
                            cor = g[y][x]
                            break
                if cor is None:
                    continue
                y, x = i, j
                while 0 <= y < h and 0 <= x < w and g[y][x] == bg:
                    fora[(y, x)] = cor
                    y, x = y + dy, x + dx
            return fora
        if t == "compl":
            # COMPLEMENTO: ocupada vira fundo, fundo vira a cor dominante.
            # E a operacao do Margolus, e ela nao existia em pintor nenhum —
            # todos os nossos ou copiam, ou constantes, ou leem vizinho.
            dom = Counter(g[i][j] for i, j in cels if g[i][j] != bg)
            if not dom:
                return {}
            c0 = dom.most_common(1)[0][0]
            return {(i, j): (bg if g[i][j] != bg else c0) for i, j in cels}
        if t == "prox":
            # Voronoi por cor: cada celula recebe a cor da celula ocupada mais
            # proxima. Propagacao isotropica, BFS multi-fonte, O(celulas).
            fila = deque()
            fora = {}
            for i, j in cels:
                if g[i][j] != bg:
                    fora[(i, j)] = g[i][j]
                    fila.append((i, j))
            while fila:
                y, x = fila.popleft()
                for dy, dx in ORTO:
                    q = (y + dy, x + dx)
                    if (0 <= q[0] < h and 0 <= q[1] < w and q not in fora):
                        fora[q] = fora[(y, x)]
                        fila.append(q)
            return fora
        if t == "esp":
            e = pin[1]
            fora = {}
            for i, j in cels:
                y, x = ((i, w - 1 - j) if e == "h" else
                        (h - 1 - i, j) if e == "v" else
                        (j, i) if e == "d1" else (w - 1 - j, h - 1 - i))
                if 0 <= y < h and 0 <= x < w and g[y][x] != bg:
                    fora[(i, j)] = g[y][x]
            return fora
        if t == "rep":
            py, px = pin[1], pin[2]
            mp = {}
            for i, j in cels:
                if g[i][j] != bg:
                    q = (i % py, j % px)
                    if q in mp and mp[q] != g[i][j]:
                        return {}
                    mp[q] = g[i][j]
            return {(i, j): mp[(i % py, j % px)] for i, j in cels
                    if (i % py, j % px) in mp}
        if t in ("linha", "coluna"):
            fora = {}
            for i, j in cels:
                cs = ({g[i][x] for x in range(w) if g[i][x] != bg} if t == "linha"
                      else {g[y][j] for y in range(h) if g[y][j] != bg})
                if len(cs) == 1:
                    fora[(i, j)] = next(iter(cs))
            return fora
        return {}

    def propor(self, treino, alvo=None):
        if alvo is None:
            alvo = [{(i, j) for i in range(dims(a)[0])
                     for j in range(dims(a)[1])} for a, _b in treino]
        if not any(alvo):
            return []
        cores = set()
        for k, (_a, b) in enumerate(treino):
            cores |= {b[y][x] for y, x in alvo[k]}
        if len(cores) > 4:
            cores = set(list(sorted(cores))[:4])
        pintores = [("copia",), ("dono",), ("vizinho",), ("linha",),
                    ("coluna",), ("prox",), ("compl",)]
        pintores += [("prop", dy, dx) for dy, dx in
                     ORTO + ((-1, -1), (-1, 1), (1, -1), (1, 1))]
        pintores += [("esp", e) for e in ("h", "v", "d1", "d2")]
        pintores += [("raio_l", L) for L in LEITURAS]
        h0, w0 = dims(treino[0][0])
        for py in (1, 2, 3):
            for px in (1, 2, 3):
                if (py, px) != (1, 1) and py <= h0 and px <= w0:
                    pintores.append(("rep", py, px))
        pintores += [("const", c) for c in sorted(cores)]

        Ts = [_com_estado(CondicaoAprendida._testes(a, fundo(a)), k)
              for k, (a, _b) in enumerate(treino)]
        chaves = [c for c in Ts[0] if all(c in T for T in Ts)]
        # SEMEADURA REVERTIDA. Ela custou DUAS tarefas (`903d1b4a`,
        # `c7d4e6ad`) mesmo com a biblioteca vazia: mudar a ordem em que os
        # pintores sao oferecidos muda o caminho do guloso, e as duas pilhas
        # vencedoras viraram `programa + copia` com residuo 24 e 62.
        # E comprou zero — a biblioteca deu +0 em TODAS as condicoes, inclusive
        # na vazada, que tinha vantagem maxima. Mudanca que custa e nao paga sai.
        # O preco em BITS fica: esse e o criterio do Stitch e nao mexe na ordem.
        fora = []
        for pin, ini in [(p, ()) for p in pintores]:
            OK, MAU, ALV = [], [], []
            for k, (a, b) in enumerate(treino):
                pr = self._pinta(pin, a, fundo(a))
                ok = {p for p, c in pr.items() if b[p[0]][p[1]] == c}
                OK.append(ok)
                MAU.append({p for p in pr if p not in ok})
                ALV.append(alvo[k] & ok)
            if not any(ALV):
                continue
            # SO testes NECESSARIOS: verdadeiros em toda celula que este pintor
            # ganharia. Intersecao deles preserva o ganho por construcao, entao
            # o guloso so precisa encolher MAU. Sem isso o seletor comeca no
            # grid inteiro e nao ha profundidade que mate MAU — foi o que fez a
            # primeira versao propor ZERO programa em toda tarefa.
            nec = [c for c in chaves
                   if all(A <= T[c] for T, A in zip(Ts, ALV))]
            if not nec:
                continue
            sel: List = [c for c in ini if c in nec]
            S = [set(o | m) for o, m in zip(OK, MAU)]
            for c in sel:
                S = [x & T[c] for x, T in zip(S, Ts)]
            for _d in range(5):
                if not any(s & m for s, m in zip(S, MAU)):
                    break
                melhor, mk, guarda = None, None, None
                for c in nec:
                    if c in sel:
                        continue
                    novos = [s & T[c] for s, T in zip(S, Ts)]
                    mata = (sum(len(s & m) for s, m in zip(S, MAU))
                            - sum(len(s & m) for s, m in zip(novos, MAU)))
                    if mata <= 0:
                        continue
                    if mk is None or mata > mk:
                        melhor, mk, guarda = c, mata, novos
                if melhor is None:
                    break
                sel.append(melhor)
                S = guarda
            if any(s & m for s, m in zip(S, MAU)):
                continue                       # nao ficou solido
            ganho = sum(len(s & a2) for s, a2 in zip(S, ALV))
            if ganho < 1:
                continue
            fora.append((tuple(sel), pin, ganho))
        fora.sort(key=lambda x: -x[2])
        return [(sel, pin) for sel, pin, _g in fora[:6]]

    @staticmethod
    def _abstracao(m):
        """A abstracao da biblioteca que cobre este programa, se houver."""
        sel, pin = m
        for comum, _bur, ab in BIBLIOTECA:
            if not comum <= frozenset(sel):
                continue
            if ab[0] == "fixo" and tuple(pin) == ab[1:]:
                return comum
            if ab[0] == "familia" and pin[0] == ab[1]:
                return comum
        return None

    def propor_k(self, treino, k: int):
        """
        Gera programas cuja soundness e avaliada DEPOIS DE k APLICACOES.

        O `propor` normal so gera programa que ganha celula em UMA aplicacao —
        e um componente de processo multi-passo ganha ZERO por construcao. No
        simulador de particulas isso e a diferenca entre 6/6 e 0/6: a pilha de
        K=1 mais `VEZES:2` reproduz 8/8 dos pares de K=2, mas o programa que
        ela usa NUNCA e proposto sobre dados de K=2.

        E a mesma doenca em tres niveis. Consertei a SELECAO com a vaga
        reservada (`ENTAO`, +3). A GERACAO tinha a mesma e eu nao tinha visto:
        gerar por ganho imediato e cego a qualquer coisa cujo valor so aparece
        depois de composta consigo mesma.

        Aqui o guloso encolhe o seletor para matar MAU calculado sobre o
        resultado de k aplicacoes, recomputando a cada passo — o seletor muda o
        estado intermediario, entao nao da para desacoplar.
        """
        h, w = dims(treino[0][0])
        cels = [(i, j) for i in range(h) for j in range(w)]

        def kfold(sel, pin, g, bg):
            expl = set()
            for _ in range(k):
                pr = self.prever((tuple(sel), pin), g)
                expl |= set(pr)
                g = tuple(tuple(pr.get((i, j), g[i][j]) for j in range(w))
                          for i in range(h))
            return g, expl

        cores = set()
        for _a, b in treino:
            cores |= {b[i][j] for i, j in cels}
        pintores = [("copia",), ("vizinho",), ("dono",), ("prox",)]
        pintores += [("prop", dy, dx) for dy, dx in ORTO]
        pintores += [("const", c) for c in sorted(cores)[:3]]
        Ts = [CondicaoAprendida._testes(a, fundo(a)) for a, _b in treino]
        chaves = [c for c in Ts[0] if all(c in T for T in Ts)]
        fora = []
        for pin in pintores:
            # SO SOLIDEZ, NAO COBERTURA.
            # A primeira versao exigia que o programa cobrisse TODAS as
            # celulas e devolveu zero — violando o principio que funda esta
            # arquitetura. A solucao de K=1 e `programa + constante`: o
            # programa cobre parte e outro codec cobre o resto. Exigir
            # completude aqui e reintroduzir o portao binario pela porta dos
            # fundos.
            def erra(sel):
                n = 0
                for a, b in treino:
                    g, e = kfold(sel, pin, a, fundo(a))
                    n += sum(1 for q in e if g[q[0]][q[1]] != b[q[0]][q[1]])
                return n

            def ganha(sel):
                n = 0
                for a, b in treino:
                    g, e = kfold(sel, pin, a, fundo(a))
                    n += sum(1 for q in e if g[q[0]][q[1]] == b[q[0]][q[1]]
                             and a[q[0]][q[1]] != b[q[0]][q[1]])
                return n

            sel: List = []
            for _d in range(4):
                if erra(sel) == 0:
                    break
                melhor, mk = None, erra(sel)
                for c in chaves:
                    if c in sel:
                        continue
                    m2 = erra(sel + [c])
                    if m2 < mk:
                        melhor, mk = c, m2
                if melhor is None:
                    break
                sel.append(melhor)
            if erra(sel) == 0 and ganha(sel) > 0:
                fora.append((tuple(sel), pin))
        return fora


    def bits(self, m):
        sel, pin = m
        ab = self._abstracao(m)
        if ab is not None:
            # so os buracos: o esqueleto ja foi pago na biblioteca
            livres = [c for c in sel if c not in ab]
            return sum(_bits_teste(c) for c in livres) + 6.0
        # pintor que propaga custa mais: ele fala sobre a grade inteira a
        # partir de poucas fontes, entao tem que economizar mais para entrar.
        cp = (BITS_COR if pin[0] == "const"
              else 7.0 if pin[0] == "raio_l"
              else 5.0 if pin[0] in ("prop", "prox", "esp", "rep") else 3.0)
        return sum(_bits_teste(c) for c in sel) + cp + 2

    def prever(self, m, a):
        sel, pin = m
        bg = fundo(a)
        h, w = dims(a)
        T = _com_estado(CondicaoAprendida._testes(a, bg), 0)
        S = {(i, j) for i in range(h) for j in range(w)}
        for c in sel:
            if c not in T:
                return {}
            S &= T[c]
        pr = self._pinta(pin, a, bg)
        return {p: pr[p] for p in S if p in pr}


class VizinhancaCor:
    """
    (minha cor, quantos vizinhos tem a cor K) -> nova cor.

    O `Vizinhanca` conta vizinhos NAO-FUNDO. Nenhum dos mundos que davam zero
    conta nao-fundo: todos contam vizinhos de uma COR ESPECIFICA.

        brian      0 -> 1  se exatamente 2 vizinhos tem cor 1
        fio        1 -> 3  se 1 ou 2 vizinhos tem cor 3
        greenberg  0 -> 1  se algum vizinho tem cor 1
        ciclico    s -> s+1 se algum vizinho tem cor s+1

    Em Brian's Brain uma celula refrataria e vizinha OCUPADA e nao conta para
    acender. Por isso os quatro mundos davam 0/2: nao faltava refratariedade
    nem excitabilidade — faltava CONTAGEM POR COR.

    E o `ciclico` pede a versao RELATIVA: vizinhos com a minha cor mais um. Isso
    e o mesmo primitivo com o alvo lido da propria celula em vez de fixo, que e
    a distincao entre constante e leitura que ja rendeu antes.
    """
    nome = "viz_cor"

    @staticmethod
    def _conta(g, i, j, alvo, k8, h, w, minha, teto=9):
        """
        `teto` AGRUPA a contagem. Com teto=1 a chave vira "zero ou algum",
        com teto=2 vira "0, 1, ou 2+".

        Sem isso a tabela e sobre a contagem EXATA: 3 cores x 9 contagens = 27
        chaves, e o `greenberg` fechava o treino com 17 delas e falhava no
        teste — memorizacao, o mesmo modo que matou o `segmenta.py` (127
        ajustavam, 6 passavam). E a regra verdadeira nem precisa de contagem
        exata: "se ALGUM vizinho tem cor 1".

        Agrupar e o anti-memorizacao certo aqui: colapsa 9 valores em 2 ou 3,
        a tabela cabe no treino, e o que sobra generaliza.
        """
        return min(VizinhancaCor._conta_cru(g, i, j, alvo, k8, h, w, minha),
                   teto)

    @staticmethod
    def _conta_cru(g, i, j, alvo, k8, h, w, minha):
        if alvo[0] == "abs":
            c = alvo[1]
        else:
            c = (minha + alvo[1]) % alvo[2]
        n = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if not (dy or dx) or (not k8 and dy and dx):
                    continue
                y, x = i + dy, j + dx
                if 0 <= y < h and 0 <= x < w and g[y][x] == c:
                    n += 1
        return n

    def propor(self, treino, alvo=None):
        cores = set()
        for a, b in treino:
            h, w = dims(a)
            cores |= {a[i][j] for i in range(h) for j in range(w)}
            cores |= {b[i][j] for i in range(h) for j in range(w)}
        nc = max(cores) + 1 if cores else 1
        cands = [("abs", c) for c in sorted(cores)]
        cands += [("rel", d, n) for d in (1, -1) for n in (2, 3, 4, 5)
                  if n <= nc + 1]
        fora = []
        for k8 in (True, False):
          for teto in (1, 2, 3, 9):
            for al in cands:
                mp = {}
                ok = True
                for p_, (a, b) in enumerate(treino):
                    h, w = dims(a)
                    ali = None if alvo is None else alvo[p_]
                    for i in range(h):
                        for j in range(w):
                            if ali is not None and (i, j) not in ali:
                                continue
                            ch = (a[i][j],
                                  self._conta(a, i, j, al, k8, h, w,
                                              a[i][j], teto))
                            if ch in mp and mp[ch] != b[i][j]:
                                ok = False
                                break
                            mp[ch] = b[i][j]
                        if not ok:
                            break
                    if not ok:
                        break
                if ok and mp:
                    fora.append((k8, al, tuple(sorted(mp.items())), teto))
        # tabela curta primeiro: menos chaves = menos memorizacao
        fora.sort(key=lambda m: len(m[2]))
        return fora[:12]

    def bits(self, m):
        return len(m[2]) * (BITS_COR + 4) + 8

    def prever(self, m, a):
        k8, al, itens, teto = m
        mp = dict(itens)
        h, w = dims(a)
        fora = {}
        for i in range(h):
            for j in range(w):
                ch = (a[i][j],
                      self._conta(a, i, j, al, k8, h, w, a[i][j], teto))
                if ch in mp:
                    fora[(i, j)] = mp[ch]
        return fora


class Aritmetica:
    """
    nova cor = minha cor + a  +  b x (vizinhos ortogonais que satisfazem um
    teste), com corte em [0, 9].

    A linguagem trata cor como ROTULO. O modelo de areia trata como NUMERO:
    `saida = cor - limiar + nº de vizinhos acima do limiar`. Sem soma sobre cor
    nao ha pilha de areia, nem contagem acumulada, nem nada quantitativo.
    """
    nome = "aritmetica"

    def propor(self, treino, alvo=None):
        lims = set()
        for a, b in treino:
            h, w = dims(a)
            lims |= {a[i][j] for i in range(h) for j in range(w)}
        fora = []
        for lim in sorted(x for x in lims if x >= 2):
            for aa in (0, -lim):
                for bb in (0, 1):
                    fora.append((lim, aa, bb))
        return fora[:24]

    def bits(self, m):
        return 3 * BITS_COR + 4

    def prever(self, m, a):
        lim, aa, bb = m
        h, w = dims(a)
        fora = {}
        for i in range(h):
            for j in range(w):
                n = sum(1 for dy, dx in ORTO
                        if 0 <= i + dy < h and 0 <= j + dx < w
                        and a[i + dy][j + dx] >= lim)
                v = a[i][j] + (aa if a[i][j] >= lim else 0) + bb * n
                fora[(i, j)] = max(0, min(9, v))
        return fora


class RelacaoBinaria:
    """
    PARA CADA x EM particao: aja sobre x em funcao de eps(x), o UNICO objeto
    que satisfaz a relacao rho(x, .).

    O maior buraco isolado do sistema, e quatro lentes independentes apontaram
    para ele: os ~200 testes sao TODOS unarios sobre celula. Nao existe uma
    unica relacao binaria entre objetos. "O objeto da mesma forma que este",
    "o mais proximo", "o que me contem" — nada disso e dizivel.

    E o custo esta em BITS, nao em busca: o espaco de modelos e 2 particoes x
    8 relacoes x 4 acoes = 64. Minusculo. Sintetizadores fogem de variavel
    ligada porque ela explode a busca; aqui a busca ja foi medida como valendo
    zero 16 vezes, e quem decide e o MDL. E a assimetria que este projeto pode
    explorar e outros nao.

    O quantificador ja tem precedente medido: no `xadrez.py`, representacao
    proposicional travou em 95,06% por colisao de features e um quantificador
    levou a 100% E TRANSFERIU de KRK para KQK.
    """
    nome = "relacao"
    PART = ("cor", "any")
    RHO = ("mesma_forma", "mesma_cor", "adjacente", "alinhado_h",
           "alinhado_v", "contem", "mais_proximo", "maior")
    ACAO = ("cor_de_eps", "apaga_x", "estampa_eps", "cor_de_x_em_eps")

    @staticmethod
    def _objs(g, bg, modo):
        h, w = dims(g)
        if modo == "cor":
            return objetos(g, bg)
        cel = {(i, j) for i in range(h) for j in range(w) if g[i][j] != bg}
        fora = []
        s = set(cel)
        while s:
            p0 = s.pop()
            c = [p0]
            f = deque([p0])
            while f:
                y, x = f.popleft()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        q = (y + dy, x + dx)
                        if q in s:
                            s.discard(q)
                            c.append(q)
                            f.append(q)
            fora.append((frozenset(c), g[c[0][0]][c[0][1]]))
        return fora

    @staticmethod
    def _norm(c):
        my = min(y for y, _x in c)
        mx = min(x for _y, x in c)
        return frozenset((y - my, x - mx) for y, x in c)

    @classmethod
    def _eps(cls, x, cx, objs, rho):
        """O UNICO z que satisfaz rho(x, z). None se zero ou mais de um."""
        cand = []
        for z, cz in objs:
            if z == x:
                continue
            if rho == "mesma_forma":
                ok = cls._norm(z) == cls._norm(x)
            elif rho == "mesma_cor":
                ok = cz == cx
            elif rho == "adjacente":
                ok = any(abs(a - c) <= 1 and abs(b - d) <= 1
                         for a, b in x for c, d in z)
            elif rho == "alinhado_h":
                ok = bool({y for y, _ in x} & {y for y, _ in z})
            elif rho == "alinhado_v":
                ok = bool({v for _, v in x} & {v for _, v in z})
            elif rho == "contem":
                ys = [y for y, _ in z]
                xs = [v for _, v in z]
                ok = all(min(ys) <= y <= max(ys) and min(xs) <= v <= max(xs)
                         for y, v in x)
            elif rho == "mais_proximo":
                ok = True
            elif rho == "maior":
                ok = len(z) > len(x)
            else:
                ok = False
            if ok:
                cand.append((z, cz))
        if rho == "mais_proximo" and cand:
            cx0 = sum(y for y, _ in x) / len(x)
            cy0 = sum(v for _, v in x) / len(x)
            cand.sort(key=lambda t: min((y - cx0) ** 2 + (v - cy0) ** 2
                                        for y, v in t[0]))
            return cand[0]
        if rho == "maior" and cand:
            cand.sort(key=lambda t: -len(t[0]))
            return cand[0] if len(cand) == 1 or len(cand[0][0]) > len(
                cand[1][0]) else None
        return cand[0] if len(cand) == 1 else None

    def propor(self, treino, alvo=None):
        fora = []
        for part in self.PART:
            for rho in self.RHO:
                for ac in self.ACAO:
                    m = (part, rho, ac)
                    solido = True
                    ganha = 0
                    for k, (a, b) in enumerate(treino):
                        pr = self.prever(m, a)
                        if not pr:
                            solido = False
                            break
                        for q, c in pr.items():
                            if q[0] >= dims(b)[0] or q[1] >= dims(b)[1]:
                                solido = False
                                break
                            if b[q[0]][q[1]] != c:
                                solido = False
                                break
                            if a[q[0]][q[1]] != c:
                                ganha += 1
                        if not solido:
                            break
                    if solido and ganha > 0:
                        fora.append(m)
        return fora

    def bits(self, m):
        return log2(len(self.PART)) + log2(len(self.RHO)) +             log2(len(self.ACAO)) + 2

    def prever(self, m, a):
        part, rho, ac = m
        bg = fundo(a)
        h, w = dims(a)
        objs = self._objs(a, bg, part)
        if len(objs) < 2 or len(objs) > 60:
            return {}
        fora = {}
        for x, cx in objs:
            e = self._eps(x, cx, objs, rho)
            if e is None:
                continue
            z, cz = e
            if ac == "cor_de_eps":
                for q in x:
                    fora[q] = cz
            elif ac == "apaga_x":
                for q in x:
                    fora[q] = bg
            elif ac == "cor_de_x_em_eps":
                for q in z:
                    fora[q] = cx
            elif ac == "estampa_eps":
                ys = [y for y, _v in x]
                xs = [v for _y, v in x]
                zy = [y for y, _v in z]
                zx = [v for _y, v in z]
                dy, dx = min(ys) - min(zy), min(xs) - min(zx)
                for y, v in z:
                    q = (y + dy, v + dx)
                    if 0 <= q[0] < h and 0 <= q[1] < w:
                        fora[q] = cz
        return fora


CODECS = [Copia(), MapaCor(), Preenche(), D4(), Simetriza(), Atracao(),
          Constante(), LinhaObjeto(), CorPorPropriedade(), Sovaco(),
          BuracoDono(), Periodo(), Gravidade(), Conecta(), Contorno(),
          ApagaPorPropriedade(), Moldura(), Vizinhanca(),
          CondicaoAprendida(), Programa(),
          VizinhancaCor(), Aritmetica(), RelacaoBinaria()]


# ---------------------------------------------------------------------------
# O LACO DE REFINAMENTO
# ---------------------------------------------------------------------------

ENTAO = "ENTAO"          # materializa UMA vez e segue lendo o novo
ATE = "ATE"              # repete o estagio ate o PONTO FIXO
VEZES = tuple(f"VEZES:{k}" for k in range(2, 7))   # repete EXATAMENTE k vezes
MARCAS = (ENTAO, ATE) + VEZES
TETO_ITER = 30


def _marca(x) -> bool:
    return isinstance(x, str)


def _quantas(m: str) -> int:
    return int(m.split(":")[1])


def _passo(pilha, treino):
    """
    Executa a pilha com ESTAGIOS e devolve, por par, (grade final, explicadas).

    Dois marcadores, e a diferenca entre eles e a diferenca entre um passo e um
    PROCESSO:

        ENTAO  materializa uma vez e segue. Composicao sequencial.
        ATE    repete o estagio ate a grade parar de mudar. PONTO FIXO.

    `ATE` e o que `fluido.py` precisava e nao tinha, e por isso deu F0 (0/400):
    agua enche o que ela mesma ja encheu, e o numero de repeticoes depende da
    entrada, entao nao da para aproximar com um numero fixo de estagios. E o
    mesmo item que os autores do CompressARC listam como falha declarada da
    arquitetura deles — "repeated operations, iteration".

    Uma celula so conta como EXPLICADA se algum codec a reivindicou em alguma
    iteracao. Celula nunca reivindicada e residuo mesmo que coincida por acaso.
    """
    fora = []
    for a, b in treino:
        h, w = dims(a)

        def materializa(g, pred):
            return tuple(tuple(pred.get((i, j), g[i][j]) for j in range(w))
                         for i in range(h))

        def rodada(g, cods):
            global _EST
            pred: Dict[Cel, int] = {}
            for cd, m in cods:
                _EST = [set(pred)]
                try:
                    for p, c in cd.prever(m, g).items():
                        pred.setdefault(p, c)
                except Exception:
                    pass
            _EST = None
            return pred

        estagios, cur = [], []
        for it in pilha:
            if _marca(it):
                estagios.append((cur, it))
                cur = []
            else:
                cur.append(it)
        estagios.append((cur, None))

        g = a
        expl: Set[Cel] = set()
        for cods, marca in estagios:
            if not cods:
                continue
            if marca is ATE:
                for _ in range(TETO_ITER):
                    pred = rodada(g, cods)
                    expl |= set(pred)
                    ng = materializa(g, pred)
                    if ng == g:
                        break
                    g = ng
            elif marca is not None and marca.startswith("VEZES:"):
                # ITERACAO CONTADA. `ATE` roda ate estabilizar; o mundo roda
                # EXATAMENTE k vezes, e as duas coisas nao coincidem. No
                # simulador de particulas isso e a diferenca entre 6/6 e 0/6:
                # com K=1 o nucleo fecha todos os episodios fisicos, com K>=2
                # fecha zero, porque "aplicar a regra local duas vezes" nao era
                # exprimivel em nenhuma profundidade.
                for _ in range(_quantas(marca)):
                    pred = rodada(g, cods)
                    expl |= set(pred)
                    g = materializa(g, pred)
            else:
                pred = rodada(g, cods)
                expl |= set(pred)
                g = materializa(g, pred)
        fora.append((g, expl))
    return fora


def custo(pilha, treino) -> Tuple[float, int]:
    """bits totais e numero de celulas ainda nao explicadas ou erradas."""
    bits = sum((2.0 if it in (ENTAO, ATE) else 2.0 + log2(_quantas(it)))
               if _marca(it) else it[0].bits(it[1]) for it in pilha)
    # REGUA CORRIGIDA. Duas coisas estavam erradas e todo numero desta sessao
    # foi medido com elas:
    #
    # 1. o preco da celula errada usava `dims(treino[0][0])` — as dimensoes do
    #    PRIMEIRO par — para os erros de TODOS os pares. Auditado nas 400:
    #    308 tarefas (77%) e 867 de 1363 pares (64%) mis-escalados, com fator
    #    de 0,108 a 3,907. Agora cada par paga pelo seu proprio tamanho.
    #
    # 2. a pilha nao pagava a IDENTIDADE do codec em cada slot, so os
    #    parametros dele. Profundidade era literalmente de graca: uma pilha de
    #    4 ganhava ~18 bits, o equivalente a quase 2 celulas erradas gratis.
    #    Agora cada slot custa log2(|CODECS|).
    bits += len(pilha) * log2(max(len(CODECS), 2))
    erradas = 0
    exc = 0.0
    for (g, expl), (a, b) in zip(_passo(pilha, treino), treino):
        hb, wb = dims(b)
        if dims(g) != (hb, wb):
            erradas += hb * wb
            exc += bits_excecao(hb * wb, hb, wb)
            continue
        n = 0
        for i in range(hb):
            for j in range(wb):
                if (i, j) not in expl or g[i][j] != b[i][j]:
                    n += 1
        erradas += n
        exc += bits_excecao(n, hb, wb)
    return bits + exc, erradas


def residuo(pilha, treino):
    """Por par, as celulas que a pilha AINDA ERRA."""
    fora = []
    for (g, expl), (a, b) in zip(_passo(pilha, treino), treino):
        h, w = dims(a)
        fora.append({(i, j) for i in range(h) for j in range(w)
                     if (i, j) not in expl or g[i][j] != b[i][j]})
    return fora


def _estagio(pilha, treino):
    """O treino como o PROXIMO codec o ve: grade corrente contra a saida."""
    ult = max((i for i, x in enumerate(pilha) if _marca(x)), default=-1)
    if ult < 0:
        return treino
    corte = ult
    ps = _passo(pilha[:corte + 1], treino)
    return [(g, b) for (g, _e), (_a, b) in zip(ps, treino)]


TETO_CEL = 1_500_000    # OPERACOES DE CELULA por tarefa; 0 = sem teto
# O teto em AVALIACOES nao normaliza: numa grade 30x30 com 3 pares cada
# avaliacao varre 2.700 celulas, e `propor` varre ~200 testes sobre elas.
# Com teto de 4.000 avaliacoes a `05a7bcf2` ainda levava 518s contra 1,2s da
# `55059096`. O orcamento tem que ser em CELULAS: assim cada tarefa paga
# proporcional ao que consome, e nenhuma sequestra a linha.


def _montar_base(treino, fundo_max: int = 5, codecs=None,
                 refoco: bool = True, feixe: int = 4, estagios: int = 2,
                 criterio: str = "exato", teto: int = None):
    """
    TETO POR TAREFA, EM AVALIACOES.

    `if melhor[2] == 0: break` faz a busca sair cedo SO QUANDO RESOLVE. Logo o
    custo e proporcional ao FRACASSO — e com 393 tarefas inalcançaveis, quase
    todo o orcamento vai para tarefas com probabilidade marginal de pagar perto
    de zero. Medido:

        00576224   2 pares,     8 celulas    13,4s    14.407 avaliacoes
        009d5c81   5 pares,   980 celulas     3,5s     1.580
        05a7bcf2   3 pares, 2.700 celulas   >380s     NAO TERMINOU

    Uma tarefa 30x30 consome sozinha dezenas de vezes a media e trava a linha
    inteira. As baratas subsidiam as caras, e nenhuma paga pelo compute que usa.

    O teto e em AVALIACOES e nao em segundos: deterministico e reproduzivel,
    entao dois bracos de um experimento pareado veem exatamente o mesmo corte.
    """
    """
    `criterio` decide o que e "melhor".

      exato       ordena por (residuo, bits). Certo para o ARC, onde a saida e
                  construida por um humano e nao tem ruido.
      compressao  ordena so por BITS. Certo para dado real, onde residuo zero e
                  inalcancavel por construcao.

    A medida que motivou isto (`ruido.py`, regras publicadas de Wolfram):

        ruido     razao     comprime   residuo 0
          0%      0,118      4/4        4/4
          3%      0,406      4/4        0/4
         10%      0,644      4/4        0/4
         20%      0,889      4/4        0/4

    Sob `residuo == 0` o sistema morre a 3% de ruido. Sob compressao sobrevive a
    20%. A unica coisa entre esta arquitetura e dado real era qual numero serve
    de limiar — quinta instancia de "exigir perfeicao e cegueira", e a primeira
    em que a medida veio ANTES do conserto.
    """
    cods = CODECS if codecs is None else codecs
    copia_cd = next((cd for cd in cods if cd.nome == "copia"), None)

    def propostas(base, alvo):
        fora = []
        for cd in cods:
            try:
                for m in cd.propor(base, alvo):
                    fora.append((cd, m))
            except Exception:
                pass
        return fora

    por_aval = max(1, sum(dims(b)[0] * dims(b)[1] for _a, b in treino))
    # BUG CORRIGIDO: o comentario dizia "0 = sem teto", mas `max(0 // x, 40)`
    # transformava 0 no teto MAIS RESTRITIVO possivel — 40 avaliacoes. Um
    # diagnostico rodado com TETO_CEL=0 "sem teto" rodou em 0s com pilha de 1
    # codec e pareceu mostrar que a lente era inexprimivel. Era o teto.
    if teto is not None:
        lim = teto
    elif TETO_CEL:
        lim = max(TETO_CEL // por_aval, 40)   # piso: algumas avaliacoes
    else:
        lim = 0                               # sem teto, de verdade
    # TETO ADAPTATIVO: nao corta quem ainda esta melhorando.
    # O teto fixo custou a `aa4ec2a5`, que fechava antes e passou a parar em
    # residuo 161 — exatamente o modo de falha previsto: "se as caras forem as
    # que fechariam com mais tempo, o teto corta o que pagava". A extensao so
    # e concedida enquanto o residuo CAI; quem estagnou e cortado na hora.
    prorroga = 2
    e_ant = None
    nav = [0]
    b0, e0 = custo([], treino)
    melhor = ([], b0, e0)
    vivos = [([], b0)]
    for _passo_i in range(fundo_max):
        novos = []
        for pilha, _b in vivos:
            # UM `_passo` POR PAI, nao tres.
            # `residuo` e a publicacao do estado chamavam `_passo(pilha,
            # treino)` separadamente sobre o MESMO prefixo, a cada pai do
            # feixe, a cada passo. Medido: ~7.500 re-execucoes completas da
            # pilha por tarefa, ~880 avaliacoes/s, ~3 milhoes para as 400 — a
            # linha e limitada por AVALIACAO, nao por proposta. Isto elimina
            # metade do desperdicio de prefixo sem mudar nenhum resultado.
            ps = _passo(pilha, treino) if pilha else None
            base = _estagio(pilha, treino)
            alvo = None
            if refoco:
                if ps is None:
                    alvo = [_todas(a) for a, _b in treino]
                else:
                    alvo = []
                    for (g, expl), (a, b) in zip(ps, treino):
                        h, w = dims(b)
                        alvo.append({(i, j) for i in range(h)
                                     for j in range(w)
                                     if (i, j) not in expl
                                     or g[i][j] != b[i][j]})
            global _EST
            # REIVINDICADA, nao "reivindicada e certa".
            # O complemento do residuo seria "acertou", e isso NAO existe no
            # teste — condicionar sobre ele seria vazamento: o programa saberia
            # quais palpites deram certo, informacao que so a saida tem.
            # `_passo` devolve exatamente o conjunto reivindicado, que e o mesmo
            # que a aplicacao ve. As duas semanticas passam a coincidir.
            _EST = [e for _g, e in ps] if ps is not None else None
            cand = propostas(base, alvo)
            # ENTAO so entra se ja houver o que materializar e sobrar estagio
            nm = sum(1 for x in pilha if _marca(x))
            if pilha and not _marca(pilha[-1]) and nm < estagios:
                cand = cand + [ENTAO, ATE]
            # MOVIMENTO ATOMICO `codec + VEZES:k`.
            # Para achar "a regra R aplicada k vezes", a busca precisa primeiro
            # hipotetizar R — e R aplicada UMA vez produz um estado intermediario
            # ERRADO, entao ganha zero bits e nunca e proposta. Anexar `VEZES`
            # depois nao adianta: a essa altura os codecs escolhidos ja sao os
            # que pagavam em um passo. O par tem que entrar junto ou nao entra.
            # E o mesmo modo de falha do `ENTAO` invisivel, um nivel abaixo:
            # componente de processo multi-passo tem valor imediato nulo.
            # `propor_k` DESLIGADO. Medido: `propor` custa 0,0s e `propor_k(2)`
            # custa 17,4s — mil vezes mais, e roda para k=2..6 a cada passo do
            # feixe, para cada modelo de forma. Isso estrangulou a suite por
            # horas e eu nao tinha medido antes de ligar.
            # E o beneficio ja estava medido em ZERO: com ele ligado, o segundo
            # dominio deu 6/24 com K>=2 em zero, identico a sem. Custa mil vezes
            # e compra nada, entao sai. A funcao fica para estudo.
            atomicos = []
            ctx = None
            if PROPONENTE is not None:
                ctx = PROPONENTE.contexto(treino, base, alvo, pilha)
                cand = PROPONENTE.ordem(ctx, cand)
            if PRIOR is not None:
                # UMA consulta por passo da busca, nao por candidato: a LLM ve o
                # estado (dimensoes, cores, pilha, quanto falta) e devolve uma
                # distribuicao sobre os codecs. Um token, ~1,3s sem cache.
                try:
                    PRIOR.pergunta(PRIOR.contexto(treino, pilha, alvo))
                except Exception:
                    pass
            for it in cand:
                if not _marca(it) and any(
                        not _marca(x) and x[0] is it[0] and x[1] == it[1]
                        for x in pilha):
                    continue
                posicoes = ([len(pilha)] if _marca(it)
                            else range(len(pilha) + 1))
                for pos in posicoes:
                    if lim and nav[0] >= lim:
                        break
                    c2 = pilha[:pos] + [it] + pilha[pos:]
                    nav[0] += 1
                    try:
                        b, e = custo(c2, treino)
                        if PRIOR is not None:
                            b += PRIOR.total(c2)
                    except Exception:
                        continue
                    novos.append((c2, b, e))
                    if PROPONENTE is not None and e < melhor[2]:
                        PROPONENTE.registra(ctx, it, melhor[2] - e)
                    # COMPLETAR COM COPIA (cegueira do ganho imediato, nivel 4).
                    # Achado no `laco_grade.py`: numa permutacao de cores,
                    # `[mapa_cor, copia]` custa 69,8 bits e generaliza 100%;
                    # `[viz_cor]` (decora cor+vizinhanca) custa 100,4 e generaliza
                    # 78%. A REGUA preferia o certo; a BUSCA escolhia o decorador —
                    # `[mapa_cor]` sozinho deixa o fundo sem dono (6076 bits) e
                    # morre antes de receber a copia, e `viz_cor` fecha sozinho e a
                    # busca para. Codec PARCIAL so mostra o valor completado.
                    # Todo candidato que nao fecha e avaliado tambem com `copia` no
                    # FIM (primeiro-que-escreve-ganha: ela so preenche o que ninguem
                    # reivindicou), e a versao completa entra se for melhor. As
                    # avaliacoes extras CONTAM no teto — nada de graca.
                    if (copia_cd is not None and e > 0 and not _marca(c2[-1])
                            and not any(not _marca(x) and x[0] is copia_cd for x in c2)
                            and not (lim and nav[0] >= lim)):
                        c3 = c2 + [(copia_cd, None)]
                        nav[0] += 1
                        try:
                            b3, e3 = custo(c3, treino)
                        except Exception:
                            continue
                        if (e3, b3) < (e, b):
                            novos.append((c3, b3, e3))
            for it, v in atomicos:
                if lim and nav[0] >= lim:
                    break
                c2 = pilha + [it, v]
                try:
                    b, e = custo(c2, treino)
                    if PRIOR is not None:
                        b += PRIOR.total(c2)
                except Exception:
                    continue
                novos.append((c2, b, e))
        estourou = lim and nav[0] >= lim
        # queda SIGNIFICATIVA, nao qualquer queda: a versao frouxa deixou a
        # `05a7bcf2` prorrogar indefinidamente caindo 694->663 por passo, e
        # voltou aos 537s que o teto tinha resolvido.
        if estourou and prorroga > 0 and e_ant is not None                 and melhor[2] < e_ant * 0.85:
            prorroga -= 1
            lim = int(lim * 1.8)
            estourou = False
        e_ant = melhor[2]
        if not novos or estourou:
            if novos:
                novos.sort(key=lambda t: (t[2], t[1]))
                if (novos[0][2], novos[0][1]) < (melhor[2], melhor[1]):
                    melhor = novos[0]
            break
        if criterio == "compressao":
            novos.sort(key=lambda t: t[1])
            for c2, b, e in novos:
                if b < melhor[1]:
                    melhor = (c2, b, e)
        else:
            novos.sort(key=lambda t: (t[2], t[1]))
            for c2, b, e in novos:
                if (e, b) < (melhor[2], melhor[1]):
                    melhor = (c2, b, e)
            if melhor[2] == 0:
                break
        novos.sort(key=lambda t: t[1])
        vistos, vivos = set(), []
        # VAGA RESERVADA PARA O ESTAGIO.
        # `ENTAO` custa 2 bits e nao explica NENHUMA celula no instante em que
        # entra: ele e, por construcao, um movimento que sempre piora agora e
        # so paga depois. Um feixe ordenado por bits o descarta em todo passo,
        # em toda tarefa — eu tinha construido um operador que a minha propria
        # busca nao podia escolher, e o primeiro resultado (`ENTAO` em zero
        # pilhas) media a busca, nao o benchmark.
        # A vaga reservada e o minimo para a medida existir. Continua sendo o
        # MDL que decide no fim: a pilha com estagio so vence se fechar o
        # residuo, e paga os bits dela como qualquer outra.
        # UMA VAGA POR MARCADOR. Tanto `ENTAO` quanto `ATE` custam bits e nao
        # explicam nenhuma celula no instante em que entram — ganho estritamente
        # diferido. Sem vaga reservada, um feixe ordenado por bits os descarta
        # em todo passo de toda tarefa, e a medida vira uma medida da BUSCA.
        for mk in (ENTAO, ATE, "VEZES"):
            t = next((t for t in novos
                      if any(_marca(x) and x.startswith(mk) for x in t[0])),
                     None)
            if t is None:
                continue
            c2, b, _e = t
            ch = tuple(x if _marca(x) else (x[0].nome, repr(x[1]))
                       for x in c2)
            if ch in vistos:
                continue
            vistos.add(ch)
            vivos.append((c2, b))
        for c2, b, _e in novos:
            ch = tuple(x if _marca(x) else (x[0].nome, repr(x[1]))
                       for x in c2)
            if ch in vistos:
                continue
            vistos.add(ch)
            vivos.append((c2, b))
            if len(vivos) >= feixe + 3:
                break
    return melhor[0], melhor[2], melhor[1]


def montar(treino, fundo_max: int = 5, codecs=None, formas=None,
           refoco: bool = True, feixe: int = 4, criterio: str = "exato",
           teto: int = None):
    """
    Escolhe um MODELO DE FORMA e depois a pilha de codecs sobre a base
    rebaseada. A forma paga os bits dela como qualquer outro modelo, entao
    `igual` (1 bit) so perde para uma forma cara quando ela economiza de
    verdade — e e isso que protege as tarefas de dimensoes iguais que ja
    passavam (portao F0).
    """
    cands = forma.propor(treino) if formas is None else formas
    melhor = None
    for fm in cands:
        reb, ok = [], True
        for a, b in treino:
            g = forma.base(fm, a)
            if g is None or dims(g) != dims(b):
                ok = False
                break
            reb.append((g, b))
        if not ok:
            continue
        try:
            # `teto` explicito (avaliacoes) sobrepoe TETO_CEL // celulas — quem
            # junta experiencias precisa pensar MAIS com mais dado, nao menos.
            pilha, err, bits = _montar_base(reb, fundo_max, codecs,
                                            refoco, feixe, 2, criterio, teto)
        except Exception:
            continue
        tot = bits + forma.bits(fm)
        # `mc` so pode ser lido DEPOIS de saber que `melhor` existe. Calcular
        # antes estourava com None, o `except` externo engolia, e todo modelo
        # de forma era descartado — 8 mundos viraram 0 e so o portao Z0 pegou.
        ch = (tot,) if criterio == "compressao" else (err, tot)
        if melhor is None:
            melhor = (fm, pilha, err, tot)
        else:
            mc = ((melhor[3],) if criterio == "compressao"
                  else (melhor[2], melhor[3]))
            if ch < mc:
                melhor = (fm, pilha, err, tot)
    if melhor is None:
        return None, [], 10**9   # sentinela de falha, nao residuo de 1
    return melhor[0], melhor[1], melhor[2]


def literal(treino) -> float:
    """
    LINHA DE BASE HONESTA: entropia empirica da saida, mais o custo do proprio
    histograma. Nao `celulas x log2(10)`.

    A versao anterior assumia distribuicao UNIFORME sobre 10 cores. Numa grade
    90% fundo isso e caro demais, e qualquer modelo que diga "quase tudo e
    fundo" parece comprimir sem capturar nada. Foi o que fez o braco NULO dar
    razao 0,247 no `queda` e 0,490 na `maioria` — o embaralhamento preserva o
    histograma, e o histograma sozinho ja comprime contra uma base uniforme.

    Com a entropia empirica como base, o modelo so "comprime" se vencer o
    codigo que ja conhece a distribuicao de cor. E o piso certo do MDL.
    """
    tot = 0.0
    for _a, b in treino:
        h, w = dims(b)
        n = h * w
        c = Counter(b[i][j] for i in range(h) for j in range(w))
        ent = -sum((v / n) * log2(v / n) for v in c.values() if v)
        tot += n * ent + len(c) * log2(max(n, 2))
    return tot


def razao(pilha, treino) -> float:
    """bits(modelo) + bits(erros) sobre a linha de base. < 1 = comprime."""
    if not pilha:
        return 1.0
    lit = literal(treino)
    b, _e = custo(pilha, treino)
    return min(1.0, b / lit) if lit > 0 else 1.0


def aplicar(regra, a: Grid) -> Optional[Grid]:
    fm, pilha = regra
    g = forma.base(fm, a)
    if g is None:
        return None
    h, w = dims(g)
    (saida, expl), = _passo(pilha, [(g, g)])
    if len(expl) < h * w:
        return None
    return saida


def _uma(args):
    nome, tr, te = args
    try:
        fm, pilha, err = montar(tr)
        if err != 0 or not pilha:
            return nome, 0, False, (), None
        ok = all(aplicar((fm, pilha), x) == y for x, y in te)
        return (nome, len(pilha), ok,
                tuple(x if _marca(x) else x[0].nome for x in pilha),
                fm[0])
    except Exception:
        return nome, 0, False, (), None


JA = {"00576224", "00dbd492", "070dd51e", "0a2355a6", "0c786b71",
      "0f63c0b9", "12eac192", "140c817e", "15663ba9", "17b80ad2",
      "195ba7dc", "1acc24af", "1c0d0a4b", "21f83797", "3194b014",
      "32e9702f", "332efdb3", "358ba94e", "37d3e8b2", "45bbe264",
      "47996f11", "55059096", "59341089", "5d2a5c43", "5ffb2104",
      "60c09cac", "66e6c45b", "72a961c9", "7953d61e", "7d1f7ee8",
      "833dafe3", "84f2aca1", "8ee62060", "903d1b4a", "981571dc",
      "9ddd00f0", "aa18de87", "aa4ec2a5", "ae58858e", "af22c60d",
      "b1fc8b8e", "bc4146bd", "be03b35f", "bf32578f", "c48954c1",
      "c7d4e6ad", "ca8f78db", "ccd554ac", "cd3c21df", "d282b262",
      "e0fb7511", "ed98d772", "f45f5ca7", "fc754716",
      # remedicao com a regua corrigida (2026-09-10): uniao 54 -> 59
      "45737921", "48131b3c", "ce039d91", "d37a1ef5", "f0df5ff0",
      # completar com copia na busca (2026-09-11): nucleo 40 -> 44, uniao 59 -> 61
      "4f537728", "bf699163"}


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=8)
    a = ap.parse_args()

    print("=" * 76)
    print("DESCRICAO — pilha de codecs guiada por bits, sobre o residuo")
    print("=" * 76)
    tarefas = carregar(PASTA_AVAL)
    t0 = time.time()
    with Pool(a.nucleos) as pool:
        res = pool.map(_uma, [(n, tr, te) for n, tr, te in tarefas],
                       chunksize=2)
    passou = {n: f for n, _k, o, f, _fm in res if o}
    formas = {n: fm for n, _k, o, _f, fm in res if o}
    comp = {n: f for n, k, o, f, _fm in res if o and k >= 2}
    novas = sorted(set(passou) - JA)

    print("")
    print(f"  ({time.time()-t0:.0f}s)")
    print(f"  passou no TESTE: {len(passou):3d}/400")
    print(f"  formas usadas  : "
          f"{dict(Counter(formas.values()).most_common())}")
    print("")
    print(f"  D1 . resolvidas por pilha de 2+ codecs: {len(comp)}")
    for n, f in sorted(comp.items()):
        print(f"        {n}  {' + '.join(f)}")
    print("")
    print(f"  tarefas: {sorted(passou)}")
    print(f"  INEDITAS: {len(novas)}  {novas}")
    dif = sorted(n for n, fm in formas.items() if fm != "igual")
    print(f"  por forma != igual: {len(dif)}  {dif}")
    tam = Counter(len(f) for f in passou.values())
    print(f"  tamanho das pilhas: {dict(sorted(tam.items()))}")
    uso = Counter(c for f in passou.values() for c in f)
    print(f"  uso de codec: {dict(uso.most_common())}")

    print("")
    print("=" * 76)
    if not comp:
        print("D1 FALHOU . credito parcial NAO produziu composicao. A tese de")
        print("     que o portao binario era o gargalo esta errada.")
    else:
        print("D1 . credito parcial produz composicao. A tese se sustenta.")
    if len(novas) >= 8:
        print("D2 . o credito parcial era o salto.")
    elif len(novas) >= 3:
        print("D3 . ajuda no patamar das mudancas de representacao.")
    else:
        print("D4 . o nucleo esta certo e os codecs e que sao poucos.")


if __name__ == "__main__":
    main()
