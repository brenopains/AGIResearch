"""
novaphys.gramatica — o agente estende a propria gramatica
===========================================================

O GAP, VISTO AO VIVO
----------------------
Para o mundo magnetico funcionar (`generalidade`, `conceito_util`), EU acrescentei
a mao uma direcao nova a gramatica das leis: n^, a velocidade girada de 90 graus.
Sem ela, a lei a = A E n^ nao e escrevivel de forma compacta — com a energia como
conceito, ela ainda precisaria de h, que nao esta no caderno. E o padrao medido
seis vezes no projeto, acontecendo de novo: o que eu entrego vira o teto.

O MECANISMO
-------------
Em cada crise, alem dos conceitos (`conceito_util`, sala de espera), o agente
propoe DIRECOES DERIVADAS das que ja tem, sem observar nada — so rearranjando o
que conhece:
    girar de 90 graus:     rot(r^), rot(t^), rot(v^)
    somar e normalizar:    (a + b)/|a + b| e (a - b)/|a - b| para cada par
Descartadas as que coincidem com uma direcao existente (|cos| medio >= 0,999). Cada
candidata e julgada pela MESMA utilidade dos conceitos: com ela na gramatica, a lei
redescoberta sobre os dados guardados fica mais curta em bits (erro + estrutura; e
cada direcao a mais encarece TODO termo em log2 do numero de direcoes)? Entra a
que mais poupa, se poupar.

O que ainda e meu, dito antes: as OPERACOES de derivacao (girar, somar) e a
profundidade 1 (so derivadas das tres de base). O teto desce um nivel — de
"direcoes" para "operacoes sobre direcoes" — e nao some.

PRE-REGISTRO, ANTES DE RODAR — sementes 20261010 (magnetico) e 20261011
(einstein), 10 correntes; n^ DESLIGADA (a gramatica de 3 direcoes de sempre)
---------------------------------------------------------------------------------
Bracos: util2 (sala de espera de conceitos, gramatica fixa) x cresce (o mesmo +
direcoes derivadas).
  L1  cresce cria uma direcao equivalente a n^ (|cos| >= 0,999 com v girada) ate a
      tarefa 19 em >= 8/10 correntes magneticas
  L2  cresce - util2 >= +20 na fase magnetica
  L3  einstein: cresce >= util2 - 5 (crescer nao estraga o caso que nao precisa)
PREVISAO: L1 passa (rot(v^) e a unica derivada que torna a lei um termo), L2
passa (sem n^, util2 fica em epiciclo — aposto ~30%), L3 passa (a utilidade barra
direcao inutil).

RESULTADO (3847 s):
                       util2 (gramatica fixa)   CRESCE
  magnetico, fase 1          0%                  95%    n^ criada 10/10: rot(v^)
  einstein,  fase 1         84%                  84%    nenhuma direcao em 3/4 mostradas
                                                        (uma vez '(t^+v^)', sem custo)
L1 PASSOU (10/10). L2 PASSOU (+95). L3 PASSOU (+0).
Sem n^, nem o conceito certo (energia) torna a lei escrevivel: util2 da 0%, nao os
~30% que previ. O agente derivou sozinho a direcao que EU tinha acrescentado a mao,
em todas as correntes, e a utilidade barrou o crescimento onde ele nao pagava.
O teto desceu um nivel: de "direcoes" para "operacoes sobre direcoes" (girar,
somar) — que continuam minhas.
"""

from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import itertools
import time

import numpy as np

from novaphys import orbita as O
from novaphys import caderno as C
from novaphys import autonomo as A
from novaphys import conceito_util as U

SEM_MAG, SEM_EIN = 20261010, 20261011


def _base(S):
    x, y, vx, vy = S[..., 0], S[..., 1], S[..., 2], S[..., 3]
    r, v = np.hypot(x, y), np.hypot(vx, vy)
    return {"r^": np.stack([x / r, y / r], -1), "t^": np.stack([-y / r, x / r], -1),
            "v^": np.stack([vx / v, vy / v], -1)}


class Dir:
    """Direcao derivada das tres de base: girar de 90 graus, ou somar/subtrair e
    normalizar. Objeto (e nao lambda) para ser comparavel e reprodutivel."""

    def __init__(self, op, a, b=None):
        self.op, self.a, self.b = op, a, b
        self.nome = f"rot({a})" if op == "rot" else f"({a}{op}{b})"

    def __call__(self, S):
        d = _base(S)
        if self.op == "rot":
            u = d[self.a]
            return np.stack([-u[..., 1], u[..., 0]], -1)
        w = d[self.a] + d[self.b] if self.op == "+" else d[self.a] - d[self.b]
        return w / np.maximum(np.linalg.norm(w, axis=-1, keepdims=True), 1e-12)


def _coincide(u, w):
    return abs(np.mean(np.sum(u * w, -1))) >= 0.999


def derivadas(existentes, Sref):
    """Candidatas de profundidade 1 que ainda nao existem (nem com sinal trocado)."""
    base = _base(Sref)
    ja = [base[k] for k in base] + [f(Sref) for f in existentes.values()]
    cands = [Dir("rot", a) for a in base]
    for a, b in itertools.combinations(base, 2):
        cands += [Dir("+", a, b), Dir("-", a, b)]
    out = []
    for c in cands:
        u = c(Sref)
        if any(_coincide(u, w) for w in ja):
            continue
        out.append(c)
        ja.append(u)
    return out


def _com_dirs(dirs):
    O.DIRS_EXTRA.clear()
    O.DIRS_EXTRA.update(dirs)


def cresce(mem, dirs, guardadas, S, D, base):
    """Propoe direcoes derivadas; adota a que mais encurta a lei, se encurtar."""
    Sa = np.concatenate([g[1] for g in guardadas] + [S])
    Da = np.concatenate([g[2] for g in guardadas] + [D])
    Sref = O.estados(np.random.default_rng(base + 17), 300)
    _com_dirs(dirs)
    L0 = U.comprimento(dict(mem.conceitos), Sa, Da, base)
    melhor = None
    for c in derivadas(dirs, Sref):
        _com_dirs({**dirs, c.nome: c})
        L = U.comprimento(dict(mem.conceitos), Sa, Da, base)
        if melhor is None or L < melhor[0]:
            melhor = (L, c)
    _com_dirs(dirs)
    if melhor is not None and melhor[0] < L0:
        dirs[melhor[1].nome] = melhor[1]
        _com_dirs(dirs)
        return True
    return False


def corrente(args):
    braco, s, semente, mundos = args
    C.MUNDOS = mundos
    O.DIR_EXTRA = False
    dirs = {}
    _com_dirs(dirs)
    mem = C.Memoria(0.7)
    pool, usadas, guardadas, primeiro = [], [], [], {}
    lei, pode_observar, fase1 = [], True, None
    limiar_chi = 1 + 3 * np.sqrt(2 / (2 * C.N_LAB))
    for k in range(A.N_PASSOS):
        mundo = C.mundo_de(k)
        base = semente + 1000 * s + k
        rng = np.random.default_rng(base)
        S = O.estados(rng, C.N_LAB)
        D = O.mede(mundo, S, rng)
        if A.surpresa(lei, S, D, mem) > limiar_chi:
            if pode_observar and len(usadas) < A.ORCAMENTO:
                usadas.append(k)
                novos = U.candidatos(pool, mundo, base)
                pool += novos
                pode_observar = bool(novos)
            if pool:
                Sa = np.concatenate([g[1] for g in guardadas] + [S])
                Da = np.concatenate([g[2] for g in guardadas] + [D])
                esc = U.reavalia(pool, Sa, Da, base)
                if esc is not mem.conceitos.get("q0"):
                    mem.conceitos = {} if esc is None else {"q0": esc}
            if braco == "cresce":
                cresce(mem, dirs, guardadas, S, D, base)
            lei = A.reanalisa_consenso(mem, guardadas, S, D, base, limiar_chi)
        ac, chaves = C.descobre(S, D, mem, base + 3)
        lei = ac
        primeiro[k] = C.acerta(ac, mem, mundo)
        guardadas.append((k, S, D, base + 3))
        mem.envelhece()
        for ch in chaves:
            mem.usa(ch)
        if pool and len(guardadas) in U.DOBRAS:
            Sa = np.concatenate([g[1] for g in guardadas])
            Da = np.concatenate([g[2] for g in guardadas])
            esc = U.reavalia(pool, Sa, Da, base + 1)
            if esc is not mem.conceitos.get("q0"):
                mem.conceitos = {} if esc is None else {"q0": esc}
                lei = A.reanalisa_consenso(mem, guardadas[:-1], S, D, base, limiar_chi)
        if k == 19:
            fase1 = (list(dirs), U._ids(mem, semente))
    Sr = O.estados(np.random.default_rng(semente + 998), 300)
    n_hat = np.stack([-Sr[:, 3], Sr[:, 2]], 1) / np.hypot(Sr[:, 2], Sr[:, 3])[:, None]
    tem_n = any(_coincide(f(Sr), n_hat) for f in dirs.values())
    n19 = fase1 is not None and any(_coincide(dirs[nm](Sr), n_hat)
                                    for nm in fase1[0] if nm in dirs)
    return braco, s, primeiro, mundos[0], list(dirs), n19, tem_n, fase1[1] if fase1 else []


def main() -> None:
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--correntes", type=int, default=10)
    ap.add_argument("--nucleos", type=int, default=3)
    a = ap.parse_args()
    n = a.correntes
    trab = ([(b, s, SEM_MAG, ("magnetico", "oblato")) for s in range(n) for b in ("util2", "cresce")]
            + [(b, s, SEM_EIN, ("einstein", "oblato")) for s in range(n) for b in ("util2", "cresce")])
    print("=" * 76)
    print("GRAMATICA — o agente deriva direcoes novas quando a lei nao cabe")
    print("=" * 76)
    t0 = time.time()
    with Pool(a.nucleos) as pool:
        res = pool.map(corrente, trab, chunksize=1)
    print(f"\n  ({time.time()-t0:.0f}s)")
    tab = {}
    for univ in ("magnetico", "einstein"):
        print(f"\n  --- universo {univ} -> oblato ---")
        for b in ("util2", "cresce"):
            rr = [r for r in res if r[3] == univ and r[0] == b]
            f1 = sum(r[2][k] for r in rr for k in range(20)) / (20 * len(rr))
            f3 = sum(r[2][k] for r in rr for k in range(25, 40)) / (15 * len(rr))
            tab[(univ, b)] = f1
            n19 = sum(1 for r in rr if r[5])
            print(f"    {b:7s} fase 1 {100*f1:3.0f}%   fase 3 {100*f3:3.0f}%   "
                  f"criou n^ ate a tarefa 19: {n19}/{n}   direcoes criadas: "
                  + "; ".join(str(r[4]) for r in rr[:4]))
    print("")
    print("=" * 76)
    l1 = sum(1 for r in res if r[3] == "magnetico" and r[0] == "cresce" and r[5])
    print(f"L1 {'.' if l1 >= 8 else 'FALHOU .'} cresce cria n^ ate a tarefa 19: {l1}/{n}")
    l2 = tab[("magnetico", "cresce")] - tab[("magnetico", "util2")]
    print(f"L2 {'.' if l2 >= 0.20 else 'FALHOU .'} cresce - util2 (magnetico) = {100*l2:+.0f} (limiar +20)")
    l3 = tab[("einstein", "cresce")] - tab[("einstein", "util2")]
    print(f"L3 {'.' if l3 >= -0.05 else 'FALHOU .'} cresce - util2 (einstein) = {100*l3:+.0f} (limiar >= -5)")


if __name__ == "__main__":
    main()
