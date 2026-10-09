"""
novaphys.suite — quais primitivos sobrevivem em TODOS os mundos?
=================================================================

A PERGUNTA
------------
Nao e "quanto o sistema pontua em cada dominio". E:

    quais familias de teste e quais pintores aparecem nas solucoes
    vencedoras de TODOS os mundos?

Isso e a definicao empirica de conhecimento nuclear. Um primitivo que vale em
quatro mundos e um CONCEITO; um que vale num so e conveniencia de dominio. E a
unica forma de descobrir o que e geral em vez de assumir.

Ate aqui `desl(N, ocupada)` atravessou do ARC para o simulador de particulas.
Um cruzamento nao e uma tabela.

OS QUATRO MUNDOS, E POR QUE ESTES
-----------------------------------
Todos com regra PUBLICADA e anterior, nunca inventada por mim para casar com o
vocabulario — que e o erro que invalidaria a suite inteira.

  vida     automatos celulares 2D: Life B3/S23, HighLife B36/S23, Seeds B2/S,
           Day&Night B3678/S34678. Conway 1970.
  fita     automatos celulares 1D, regras 30, 90, 110, 184 de Wolfram, sobre
           uma grade de UMA linha. Testa quais primitivos sobrevivem sem
           estrutura vertical — `viz4` e `desl` N/S ficam degenerados de
           proposito.
  pilha    modelo de areia abeliano (Bak-Tang-Wiesenfeld 1987): celula com 4 ou
           mais graos tomba e distribui 1 para cada vizinho ortogonal.
  queda    o simulador de particulas do `deducao.py`, ja medido em 6/24.

CONTROLE PAREADO EM TODOS
---------------------------
Cada mundo tem um gemeo com regra SORTEADA de mesmo formato e mesmo custo:
nascimento/sobrevivencia aleatorios em `vida`, tabela de 8 bits sorteada em
`fita`, limiar e distribuicao sorteados em `pilha`, `fisico=False` em `queda`.

Sem isso, "resolveu no mundo B" pode ser so "meu vocabulario ajusta qualquer
coisa" — que e o modo de falha que matou o `segmenta.py` (127 ajustavam o
treino, 6 passavam no teste).

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  S0  PRE-CONDICAO. `queda` tem que dar 6/24 no braco real, que e a medida do
      `dominio2.py`. Se cair, e erro de adaptacao e nao resultado.

  S1  real > sorteado em CADA mundo -> o vocabulario captura estrutura, e nao
      ajusta ruido. Um mundo em que o sorteado empata e um mundo onde estamos
      sobreajustando.

  S2  A TABELA. Quantos primitivos aparecem em 4 mundos, em 3, em 2, em 1.
      >= 1 em quatro mundos -> ha nucleo comum medido
      nenhum em mais de dois -> nao ha nucleo; o que parecia geral era
                                coincidencia de dois espacos discretos

SEMENTE PRE-REGISTRADA: 20260909

MINHA CALIBRAGEM: cinquenta e duas previsoes, DEZOITO certas, quase todas as
erradas otimistas. E hoje eu ja errei tres vezes por violar coisa que o proprio
projeto tinha medido.

PREVISAO: S1 em `vida`, `pilha` e `queda`; empate ou zero duplo em `fita`,
porque uma grade de uma linha degenera metade do vocabulario. E na tabela,
aposto que `viz4`/`viz8` e `desl` aparecem em tres ou quatro — sao afirmacoes
sobre vizinhanca, e vizinhanca e o que todo espaco discreto tem. Aposto que
`sovaco`, `raio_de` e `quebra_sim` NAO atravessam: sao afirmacoes sobre forma,
e forma e conveniencia de dominio.
"""

from __future__ import annotations

import argparse
import random
import time
from collections import Counter

from novaphys.arc import dims
from novaphys import descricao as D
from novaphys import deducao as DD

SEMENTE = 20260909
LADO = 12
TAREFAS = None


# ─────────────────────────────────────────────────────────── vida (CA 2D)
VIDA = {"life": ({3}, {2, 3}), "highlife": ({3, 6}, {2, 3}),
        "seeds": ({2}, set()), "daynight": ({3, 6, 7, 8}, {3, 4, 6, 7, 8})}


def _ca2d(g, nasce, sobrevive, cor):
    h, w = dims(g)
    fora = []
    for i in range(h):
        lin = []
        for j in range(w):
            n = sum(1 for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                    if (dy or dx) and 0 <= i + dy < h and 0 <= j + dx < w
                    and g[i + dy][j + dx] != 0)
            viva = g[i][j] != 0
            lin.append(cor if ((viva and n in sobrevive)
                               or (not viva and n in nasce)) else 0)
        fora.append(tuple(lin))
    return tuple(fora)


def _cena(rng, dens=0.3, cor=5, lado=LADO):
    return tuple(tuple(cor if rng.random() < dens else 0 for _ in range(lado))
                 for _ in range(lado))


# ─────────────────────────────────────────────────────────── fita (CA 1D)
def _ca1d(g, regra, cor):
    lin = g[0]
    w = len(lin)
    fora = []
    for j in range(w):
        e = 1 if (j > 0 and lin[j - 1]) else 0
        c = 1 if lin[j] else 0
        d = 1 if (j + 1 < w and lin[j + 1]) else 0
        idx = (e << 2) | (c << 1) | d
        fora.append(cor if (regra >> idx) & 1 else 0)
    return (tuple(fora),)


# ─────────────────────────────────────────────────────────── pilha (areia)
def _areia(g, limiar, cor):
    h, w = dims(g)
    m = [list(l) for l in g]
    novo = [[0] * w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            if m[i][j] >= limiar:
                novo[i][j] -= limiar
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    y, x = i + dy, j + dx
                    if 0 <= y < h and 0 <= x < w:
                        novo[y][x] += 1
    return tuple(tuple(max(0, min(9, m[i][j] + novo[i][j])) for j in range(w))
                 for i in range(h))


def _cena_areia(rng, lado=LADO):
    return tuple(tuple(rng.randint(0, 5) for _ in range(lado))
                 for _ in range(lado))


def _monta():
    """Cada tarefa: 6 pares de treino do MESMO fenomeno, 2 de teste."""
    rng = random.Random(SEMENTE)
    T = []
    # vida
    for nome, (nas, sob) in VIDA.items():
        for real in (True, False):
            n2, s2 = ((nas, sob) if real else
                      (set(rng.sample(range(1, 9), len(nas))),
                       set(rng.sample(range(0, 9), max(1, len(sob))))))
            pr = []
            for _ in range(8):
                a = _cena(rng)
                b = _ca2d(a, n2, s2, 5)
                if a != b:
                    pr.append((a, b))
            if len(pr) >= 5:
                T.append((f"vida/{nome}", "vida", real, pr))
    # fita
    for r in (30, 90, 110, 184):
        for real in (True, False):
            rr = r if real else rng.randint(1, 254)
            pr = []
            for _ in range(8):
                a = (tuple(5 if rng.random() < 0.4 else 0
                           for _ in range(LADO * 2)),)
                b = _ca1d(a, rr, 5)
                if a != b:
                    pr.append((a, b))
            if len(pr) >= 5:
                T.append((f"fita/{rr}", "fita", real, pr))
    # pilha
    for lim in (4, 5, 6, 8):
        for real in (True, False):
            lv = lim if real else rng.randint(2, 9)
            pr = []
            for _ in range(8):
                a = _cena_areia(rng)
                b = _areia(a, lv, 0)
                if a != b:
                    pr.append((a, b))
            if len(pr) >= 5:
                T.append((f"pilha/{lv}", "pilha", real, pr))
    # queda
    for s_ in range(4):
        for real in (True, False):
            for k, obs in enumerate(DD.pares(real, 8, 2, SEMENTE + s_)):
                if len(obs) >= 5:
                    T.append((f"queda/k{k+1}s{s_}", "queda", real, obs))
    return T


def _ini():
    global TAREFAS
    TAREFAS = _monta()
    D.usar_biblioteca(None)


def _uma(ti):
    nome, mundo, real, pr = TAREFAS[ti]
    tr, te = pr[:-2], pr[-2:]
    try:
        fm, pilha, err = D.montar(tr, refoco=True, feixe=4)
        if err != 0 or not pilha:
            return mundo, real, 0, ()
        if not all(D.aplicar((fm, pilha), x) == y for x, y in te):
            return mundo, real, 0, ()
        simb = set()
        for x in pilha:
            if D._marca(x):
                simb.add(f"marca:{x.split(':')[0]}")
                continue
            simb.add(f"codec:{x[0].nome}")
            if x[0].nome == "programa":
                sel, pin = x[1]
                simb.add(f"pintor:{pin[0]}")
                for c in sel:
                    simb.add(f"teste:{c[0]}")
        return mundo, real, 1, tuple(sorted(simb))
    except Exception:
        return mundo, real, 0, ()


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=3)
    a = ap.parse_args()
    _ini()

    print("=" * 76)
    print("SUITE — quais primitivos sobrevivem em todos os mundos?")
    print("=" * 76)
    print(f"  {len(TAREFAS)} tarefas em 4 mundos, todas com controle pareado")
    t0 = time.time()
    with Pool(a.nucleos, initializer=_ini) as pool:
        res = pool.map(_uma, range(len(TAREFAS)), chunksize=1)

    print(f"\n  ({time.time()-t0:.0f}s)\n")
    print("  mundo    regra real   regra sorteada")
    print("  -------  ----------   --------------")
    mundos = ["vida", "fita", "pilha", "queda"]
    for m in mundos:
        nr = sum(o for mm, r, o, _s in res if mm == m and r)
        tr_ = sum(1 for mm, r, _o, _s in res if mm == m and r)
        na = sum(o for mm, r, o, _s in res if mm == m and not r)
        ta = sum(1 for mm, r, _o, _s in res if mm == m and not r)
        print(f"  {m:7s}  {nr:3d}/{tr_:<6d}   {na:3d}/{ta:<6d}")

    onde = {}
    for m, r, o, s in res:
        if r and o:
            for x in s:
                onde.setdefault(x, set()).add(m)
    print("\n  S2 . TABELA DO NUCLEO")
    print("    n  primitivo                mundos")
    print("   --  ----------------------   ------")
    for x, ms in sorted(onde.items(), key=lambda t: (-len(t[1]), t[0])):
        print(f"   {len(ms):2d}  {x:22s}   {','.join(sorted(ms))}")

    nuc = [x for x, ms in onde.items() if len(ms) >= 3]
    print("")
    print("=" * 76)
    print(f"PRIMITIVOS EM 3+ MUNDOS: {len(nuc)}  {sorted(nuc)}")
    if not onde:
        print("S0 FALHOU: nenhum mundo resolvido; a medida nao existe.")
    elif nuc:
        print("S2 . HA NUCLEO COMUM MEDIDO. Estes primitivos nao sao conceitos")
        print("     de dominio — valem onde houver espaco discreto.")
    else:
        print("S2 . NAO ha nucleo. O que parecia geral era coincidencia de")
        print("     dois espacos discretos parecidos.")


if __name__ == "__main__":
    main()
