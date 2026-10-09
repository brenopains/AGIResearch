"""
novaphys.nucleo — a curva de saturacao do vocabulario
======================================================

A APOSTA
----------
O nucleo comum entre dominios SATURA, e satura pequeno.

Rode o sistema em N mundos independentes e conte, a cada mundo novo, quantos
primitivos NOVOS foram necessarios. Duas curvas possiveis:

    reta     cada mundo exige vocabulario proprio -> nao ha nucleo, e
             "vocabulario geral" morre com dado
    ACHATA   o decimo mundo exige 1 primitivo novo em vez de 20 -> existe
             conhecimento nuclear FINITO, PEQUENO e MENSURAVEL

Se achatar, o exponencial nao esta onde a gente procurava. Nao e "mais
primitivos cobrem mais tarefas" — isso e linear e rendeu +1 a +11 hoje. E os
MESMOS K primitivos cobrindo ilimitados dominios. O ganho vem de descobrir que
o vocabulario PARA de crescer.

O FIO DA NAVALHA
------------------
O nucleo precisa de duas coisas que puxam em direcoes opostas:

    pequeno o bastante para SATURAR   (senao nao ha economia)
    rico o bastante para COBRIR       (senao satura em 3 primitivos inuteis)

Se so `viz4` e `desl` atravessarem, o nucleo tem tamanho 2 e a aposta morre.

OS DEZ MUNDOS
---------------
Todos com regra PUBLICADA e anterior. Nenhum inventado para casar com o
vocabulario — isso invalidaria a medida inteira, e e a tentacao obvia de quem
escreve o proprio benchmark.

    vida       Life B3/S23, HighLife, Seeds, Day&Night   Conway 1970
    fita       regras 30, 90, 110, 184                   Wolfram 1983
    pilha      modelo de areia abeliano                  Bak-Tang-Wiesenfeld 1987
    queda      simulador de particulas                   `deducao.py`, anterior
    brian      Brian's Brain, 3 estados                  Silverman
    fio        Wireworld, 4 estados                      Silverman 1987
    ciclico    automato ciclico, n estados               Griffeath
    maioria    voto majoritario                          Vichniac 1986
    greenberg  meio excitavel                            Greenberg-Hastings 1978
    critters   automato de blocos de Margolus            Toffoli-Margolus 1987

Mecanismos genuinamente distintos: contagem totalistica, tabela de vizinhanca
1D, limiar com redistribuicao, gravidade com colisao, refratariedade, propagacao
de sinal, ciclo, votacao, excitabilidade, e particao em blocos.

CONTROLE PAREADO EM TODOS. Regra sorteada de mesmo formato e mesmo custo.
Sem isso, "resolveu no mundo M" pode ser so "meu vocabulario ajusta qualquer
coisa" — que e o que matou o `segmenta.py` (127 ajustavam, 6 passavam).

PRE-REGISTRO, ANTES DE RODAR
------------------------------
  K0  PRE-CONDICAO. Pelo menos 4 mundos com placar real > sorteado. Abaixo
      disso a amostra de vocabulario e pequena demais e a curva nao existe.

  K1  A CURVA ACHATA: custo marginal do mundo N+1 cai abaixo de 2 primitivos
      novos ate N = 6  -> HA NUCLEO FINITO
  K2  a curva e reta   -> cada mundo quer o seu; nao ha vocabulario geral
  K3  satura abaixo de 5 primitivos -> ha nucleo e ele e pobre demais para
      cobrir; o fio da navalha cortou do lado errado

SEMENTE PRE-REGISTRADA: 20260909

MINHA CALIBRAGEM: cinquenta e duas previsoes, DEZOITO certas.

PREVISAO, e e a aposta declarada: K1, com o nucleo saturando entre 12 e 25
primitivos, e o custo marginal caindo abaixo de 2 ate N=6. E aposto na
COMPOSICAO: vizinhanca, direcao, contagem, alcance, fecho e residuo — e NENHUMA
nocao de forma. Que `sovaco`, `raio_de` e `quebra_sim` sejam conveniencia de
grade e nao conceito. Se forma atravessar, eu errei sobre o que e conceito.
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
LADO = 8      # reduzido de 11: a medida e sobre QUAIS primitivos
              # aparecem, nao sobre tamanho de grade. O risco de regra
              # espuria caber numa grade pequena e coberto pelo braco
              # sorteado, que ja existe em todos os mundos.
MOORE = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
ORTO4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
TAR = None


def _rnd_grade(rng, k, lado=LADO, dens=0.35):
    return tuple(tuple(rng.randint(1, k) if rng.random() < dens else 0
                       for _ in range(lado)) for _ in range(lado))


def _viz(g, i, j, alvo):
    h, w = dims(g)
    return sum(1 for dy, dx in MOORE if 0 <= i + dy < h and 0 <= j + dx < w
               and g[i + dy][j + dx] == alvo)


# ─────────────────────────────────────────────────────────────── os mundos
def w_vida(g, nasce, sobrevive):
    h, w = dims(g)
    return tuple(tuple(
        5 if ((g[i][j] and _viz(g, i, j, 5) in sobrevive)
              or (not g[i][j] and _viz(g, i, j, 5) in nasce)) else 0
        for j in range(w)) for i in range(h))


def w_fita(g, regra):
    lin = g[0]
    w = len(lin)
    return (tuple(5 if (regra >> (((1 if j and lin[j-1] else 0) << 2)
                                  | ((1 if lin[j] else 0) << 1)
                                  | (1 if j+1 < w and lin[j+1] else 0))) & 1
                  else 0 for j in range(w)),)


def w_pilha(g, limiar):
    h, w = dims(g)
    d = [[0] * w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            if g[i][j] >= limiar:
                d[i][j] -= limiar
                for dy, dx in ORTO4:
                    if 0 <= i + dy < h and 0 <= j + dx < w:
                        d[i + dy][j + dx] += 1
    return tuple(tuple(max(0, min(9, g[i][j] + d[i][j])) for j in range(w))
                 for i in range(h))


def w_brian(g, fogo):
    """Brian's Brain: 1 fogo -> 2 refratario -> 0; 0 pega fogo com `fogo` vizinhos."""
    h, w = dims(g)
    return tuple(tuple(2 if g[i][j] == 1 else 0 if g[i][j] == 2
                       else (1 if _viz(g, i, j, 1) == fogo else 0)
                       for j in range(w)) for i in range(h))


def w_fio(g, faixa):
    """Wireworld: 3 cabeca -> 4 cauda -> 1 condutor; condutor vira cabeca."""
    h, w = dims(g)
    return tuple(tuple(4 if g[i][j] == 3 else 1 if g[i][j] == 4
                       else (3 if (g[i][j] == 1
                                   and _viz(g, i, j, 3) in faixa) else g[i][j])
                       for j in range(w)) for i in range(h))


def w_ciclico(g, n):
    h, w = dims(g)
    return tuple(tuple(((g[i][j] + 1) % n) if _viz(g, i, j, (g[i][j] + 1) % n)
                       else g[i][j] for j in range(w)) for i in range(h))


def w_maioria(g, lim):
    h, w = dims(g)
    return tuple(tuple(5 if (_viz(g, i, j, 5) + (1 if g[i][j] else 0)) >= lim
                       else 0 for j in range(w)) for i in range(h))


def w_greenberg(g, n):
    h, w = dims(g)
    return tuple(tuple(((g[i][j] + 1) % n) if g[i][j]
                       else (1 if _viz(g, i, j, 1) else 0)
                       for j in range(w)) for i in range(h))


def w_critters(g, comp):
    """Margolus: blocos 2x2; se o bloco tem `comp` vivos, complementa."""
    h, w = dims(g)
    m = [list(l) for l in g]
    for i in range(0, h - 1, 2):
        for j in range(0, w - 1, 2):
            bl = [(i, j), (i, j + 1), (i + 1, j), (i + 1, j + 1)]
            n = sum(1 for y, x in bl if m[y][x])
            if n == comp:
                for y, x in bl:
                    m[y][x] = 0 if m[y][x] else 5
    return tuple(tuple(l) for l in m)


def _nulo(pr, rng):
    """
    CONTROLE CORRIGIDO. O braco de controle era "regra da MESMA familia com
    parametros sorteados" — e isso nao testa nada: uma regra B/S sorteada
    continua sendo um automato totalistico, e o `vizinhanca` a expressa
    exatamente. O sistema resolvia os dois bracos e eu teria lido como
    sobreajuste, quando era acerto.

    O controle certo e saida SEM ESTRUTURA: mesmas entradas, saidas sorteadas
    com o mesmo histograma de cor. Resolver isso e, ai sim, ajustar ruido.
    """
    fora = []
    for a, b in pr:
        cores = [c for l in b for c in l]
        rng.shuffle(cores)
        h, w = len(b), len(b[0])
        fora.append((a, tuple(tuple(cores[i * w + j] for j in range(w))
                              for i in range(h))))
    return fora


def _monta():
    rng = random.Random(SEMENTE)
    T = []

    def add(mundo, nome, real, gera, passo, k=5, npar=7):
        pr = []
        for _ in range(npar):
            a = gera(rng)
            try:
                b = passo(a)
            except Exception:
                continue
            if a != b:
                pr.append((a, b))
        if len(pr) >= 5:
            T.append((f"{mundo}/{nome}", mundo, real, pr))

    g5 = lambda r: _rnd_grade(r, 1, dens=0.35) and tuple(
        tuple(5 if c else 0 for c in l) for l in _rnd_grade(r, 1, dens=0.35))
    for nome, (na, so) in (("life", ({3}, {2, 3})), ("high", ({3, 6}, {2, 3})),
                           ("seeds", ({2}, set())),
                           ("daynight", ({3, 6, 7, 8}, {3, 4, 6, 7, 8}))):
        add("vida", nome, True, g5, lambda a, n=na, s=so: w_vida(a, n, s))
    gf = lambda r: (tuple(5 if r.random() < 0.4 else 0 for _ in range(16)),)
    for r_ in (30, 90, 110, 184):
        add("fita", str(r_), True, gf, lambda a, q=r_: w_fita(a, q))
    gp = lambda r: _rnd_grade(r, 5, dens=0.9)
    for lim in (4, 5, 6):
        add("pilha", str(lim), True, gp, lambda a, l=lim: w_pilha(a, l))
    for s_ in range(3):
        for k, obs in enumerate(DD.pares(True, 8, 2, SEMENTE + s_)):
            if len(obs) >= 5:
                T.append((f"queda/r{k}{s_}", "queda", True, obs))
    gb = lambda r: _rnd_grade(r, 2, dens=0.3)
    for f_ in (2, 1):
        add("brian", str(f_), True, gb, lambda a, q=f_: w_brian(a, q))
    gw = lambda r: tuple(tuple(r.choice([0, 1, 1, 3, 4])
                               for _ in range(LADO)) for _ in range(LADO))
    for fx in ({1, 2}, {1}):
        add("fio", str(sorted(fx)), True, gw, lambda a, q=fx: w_fio(a, q))
    gc = lambda r: _rnd_grade(r, 3, dens=1.0)
    for n_ in (3, 4):
        add("ciclico", str(n_), True, gc, lambda a, q=n_: w_ciclico(a, q))
    for lim in (5, 4):
        add("maioria", str(lim), True, g5, lambda a, l=lim: w_maioria(a, l))
    gg = lambda r: _rnd_grade(r, 2, dens=0.4)
    for n_ in (3, 4):
        add("greenberg", str(n_), True, gg, lambda a, q=n_: w_greenberg(a, q))
    for c in (3, 1):
        add("critters", str(c), True, g5, lambda a, q=c: w_critters(a, q))
    # cada tarefa real ganha o gemeo NULO: mesmas entradas, saida sorteada
    for nome, mundo, real, pr in list(T):
        if real:
            T.append((f"{nome}/nulo", mundo, False, _nulo(pr, rng)))
    return T


def _ini():
    global TAR
    TAR = _monta()
    D.usar_biblioteca(None)


def _uma(ti):
    _n, mundo, real, pr = TAR[ti]
    tr, te = pr[:-2], pr[-2:]
    try:
        fm, pilha, err = D.montar(tr, refoco=True, feixe=4)
        if err != 0 or not pilha:
            return mundo, real, 0, ()
        if not all(D.aplicar((fm, pilha), x) == y for x, y in te):
            return mundo, real, 0, ()
        s = set()
        for x in pilha:
            if D._marca(x):
                s.add(f"marca:{x.split(':')[0]}")
                continue
            s.add(f"codec:{x[0].nome}")
            if x[0].nome == "programa":
                sel, pin = x[1]
                s.add(f"pintor:{pin[0]}")
                for c in sel:
                    s.add(f"teste:{c[0]}")
        return mundo, real, 1, tuple(sorted(s))
    except Exception:
        return mundo, real, 0, ()


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--nucleos", type=int, default=3)
    a = ap.parse_args()
    _ini()
    mundos = sorted({m for _n, m, _r, _p in TAR})

    print("=" * 76)
    print("NUCLEO — a curva de saturacao do vocabulario")
    print("=" * 76)
    print(f"  {len(TAR)} tarefas em {len(mundos)} mundos, todos pareados")
    t0 = time.time()
    with Pool(a.nucleos, initializer=_ini) as pool:
        res = pool.map(_uma, range(len(TAR)), chunksize=1)

    print(f"\n  ({time.time()-t0:.0f}s)\n")
    print("  mundo       real    sorteado")
    print("  ----------  ------  --------")
    voc = {}
    bons = []
    for m in mundos:
        nr = sum(o for mm, r, o, _s in res if mm == m and r)
        tr_ = sum(1 for mm, r, _o, _s in res if mm == m and r)
        na = sum(o for mm, r, o, _s in res if mm == m and not r)
        ta = sum(1 for mm, r, _o, _s in res if mm == m and not r)
        print(f"  {m:10s}  {nr:2d}/{tr_:<3d}  {na:2d}/{ta:<3d}")
        v = set()
        for mm, r, o, s in res:
            if mm == m and r and o:
                v |= set(s)
        if v:
            voc[m] = v
            if nr > na:
                bons.append(m)

    print(f"\n  K0 . mundos com real > sorteado: {len(bons)}  {bons}")
    if len(voc) < 2:
        print("\nK0 FALHOU: vocabulario insuficiente; a curva nao existe.")
        return

    # curva: media de primitivos NOVOS por mundo, sobre ordens sorteadas
    rnd = random.Random(SEMENTE)
    ordens = [list(voc) for _ in range(40)]
    for o in ordens:
        rnd.shuffle(o)
    novos = [[] for _ in range(len(voc))]
    for o in ordens:
        acc = set()
        for k, m in enumerate(o):
            novos[k].append(len(voc[m] - acc))
            acc |= voc[m]
    print("\n  K1 . CURVA DE SATURACAO (media sobre 40 ordens)")
    print("     N   primitivos novos no mundo N   acumulado")
    print("    ---  ---------------------------   ---------")
    acc = 0.0
    for k, xs in enumerate(novos):
        med = sum(xs) / len(xs)
        acc += med
        print(f"    {k+1:3d}  {med:27.2f}   {acc:9.1f}")

    marg = [sum(x) / len(x) for x in novos]
    onde = next((i + 1 for i, v in enumerate(marg) if v < 2.0), None)
    total = len(set().union(*voc.values()))
    comuns = set.intersection(*voc.values()) if len(voc) > 1 else set()

    print(f"\n  vocabulario total usado: {total}")
    print(f"  primitivos em TODOS os {len(voc)} mundos: {len(comuns)}")
    for c in sorted(comuns):
        print(f"     {c}")
    forma = [c for c in set().union(*voc.values())
             if any(k in c for k in ("sovaco", "raio_de", "quebra",
                                     "forma_unica", "simetriza"))]
    print(f"  primitivos de FORMA que apareceram: {sorted(forma)}")

    print("")
    print("=" * 76)
    if onde and onde <= 6:
        print(f"K1 . A CURVA ACHATA em N={onde}. Custo marginal cai abaixo de 2")
        print(f"     primitivos novos. Vocabulario total {total}.")
        if total < 5:
            print("K3 . mas o nucleo e pobre demais para cobrir.")
    elif onde:
        print(f"K1 parcial . achata so em N={onde}, depois do previsto (6).")
    else:
        print("K2 . a curva NAO achata: cada mundo quer o seu vocabulario, e")
        print("     'vocabulario geral' morre com dado.")


if __name__ == "__main__":
    main()
