"""
encarnar.encarnar — a operacao "tornar-se X"
=============================================
De posicoes cruas: velocidade e aceleracao por diferencas. Para cada ser X a mente
pergunta: DE ONDE o movimento de X fica mais simples de descrever?

Pontos de vista (todos gerados pela mesma operacao, para qualquer ser):
  mundo        eixos fixos da arena
  eu           o PROPRIO corpo de X: frente = a velocidade de X, lado = girada
  ser Y        o corpo de OUTRO ser Y: centro em Y, frente de Y, lado de Y
  junto de Y   centrado em Y, sem girar
  placebo      dois quadros sem dono: centro e giro aleatorios (mesmo numero de atomos)

Em cada ponto de vista, os atomos sao so geometria do quadro (nao semantica):
posicao e velocidade relativas de cada outro ser (e do centro da arena) nesses
eixos, distancia, 1/d, 1/d^2, direcao unitaria, e a rapidez do dono do quadro.
A lei e a menor descricao (MDL): regressao esparsa gulosa ORTOGONAL (OMP) sobre
atomos e produtos de dois atomos; para quando o proximo termo nao paga em bits.
"""
from __future__ import annotations

import math
import zlib

import numpy as np

from mundo import DT, rot, unit


def cinematica(X):
    """Diferencas centrais: posicao, velocidade, aceleracao (passos internos)."""
    V = (X[2:] - X[:-2]) / (2 * DT)
    A = (X[2:] - 2 * X[1:-1] + X[:-2]) / DT ** 2
    return X[1:-1], V, A


def _eixos(v):
    t = unit(v)
    return t, rot(t)


def _no_quadro(vec, t, nh):
    return np.stack([(vec * t).sum(-1), (vec * nh).sum(-1)], -1)


def _atomos_ref(dp, dv, t, nh, nome):
    """Geometria de um ponto de referencia visto num quadro."""
    q = _no_quadro(dp, t, nh)
    qd = _no_quadro(dv, t, nh)
    d = np.maximum(np.linalg.norm(dp, axis=-1), 1e-6)
    u = q / d[:, None]
    cols = [q[:, 0], q[:, 1], qd[:, 0], qd[:, 1], d, 1 / d, 1 / d ** 2, u[:, 0], u[:, 1]]
    nomes = [f"{nome}.x", f"{nome}.y", f"{nome}.vx", f"{nome}.vy", f"|{nome}|", f"1/|{nome}|",
             f"1/|{nome}|2", f"{nome}^x", f"{nome}^y"]
    return cols, nomes


def ponto_de_vista(X, i, pv, rng=None, com_quadro=False):
    """Atomos (matriz) e alvo (aceleracao de i no quadro) para o ponto de vista pv."""
    P, V, A = cinematica(X)
    n = P.shape[1]
    T = P.shape[0]
    if pv == "mundo":
        t = np.tile([1.0, 0.0], (T, 1))
        nh = rot(t)
        orig_p, orig_v, a_ref, dono_v = np.zeros((T, 2)), np.zeros((T, 2)), np.zeros((T, 2)), V[:, i]
    elif pv == "eu":
        t, nh = _eixos(V[:, i])
        orig_p, orig_v, a_ref, dono_v = P[:, i], V[:, i], np.zeros((T, 2)), V[:, i]
    elif pv.startswith("ser "):
        j = int(pv.split()[1])
        t, nh = _eixos(V[:, j])
        orig_p, orig_v, a_ref, dono_v = P[:, j], V[:, j], A[:, j], V[:, j]
    elif pv.startswith("junto "):
        j = int(pv.split()[1])
        t = np.tile([1.0, 0.0], (T, 1))
        nh = rot(t)
        orig_p, orig_v, a_ref, dono_v = P[:, j], V[:, j], A[:, j], V[:, j]
    elif pv.startswith("placebo"):
        # quadro SEM dono: centro e giro em passeio aleatorio suave (nao pode
        # coincidir por acaso com um giro constante de nenhum ser)
        r = np.random.default_rng(zlib.crc32(f"{pv}|{i}|{T}".encode()))
        k = np.exp(-np.arange(-40, 41) ** 2 / (2 * 12.0 ** 2))
        k /= k.sum()
        w = np.convolve(r.normal(0, 1.0, T + 80), k, "same")[40:40 + T] * 3.0
        ang = np.cumsum(w) * DT + r.uniform(0, 2 * np.pi)
        t = np.stack([np.cos(ang), np.sin(ang)], 1)
        nh = rot(t)
        vc = np.stack([np.convolve(r.normal(0, 1.0, T + 80), k, "same")[40:40 + T] for _ in range(2)], 1) * 2.0
        orig_v = vc
        orig_p = r.uniform(-3, 3, 2) + np.cumsum(vc, 0) * DT
        a_ref = np.zeros((T, 2))
        dono_v = orig_v + 0.0
    else:
        raise ValueError(pv)
    cols, nomes = [], []
    # o proprio X visto do quadro (se o quadro nao e o dele)
    if pv not in ("eu",):
        c, nm = _atomos_ref(P[:, i] - orig_p, V[:, i] - orig_v, t, nh, "eu")
        cols += c
        nomes += nm
    # cada outro ser e o centro da arena, vistos do quadro (relativos a X)
    for j in list(range(n)) + ["centro"]:
        if j == i:
            continue
        pj = np.zeros((T, 2)) if j == "centro" else P[:, j]
        vj = np.zeros((T, 2)) if j == "centro" else V[:, j]
        c, nm = _atomos_ref(pj - P[:, i], vj - V[:, i], t, nh, f"s{j}" if j != "centro" else "c")
        cols += c
        nomes += nm
    sp = np.linalg.norm(dono_v, axis=-1)
    cols.append(sp)
    nomes.append("rapidez")
    F = np.stack(cols, 1)
    y = _no_quadro(A[:, i] - a_ref, t, nh)
    if com_quadro:
        return F, nomes, y, (t, nh, a_ref, A[:, i])
    return F, nomes, y


def _base(F, nomes, grau=2):
    """Atomos (grau 1) ou atomos e produtos de dois atomos (grau 2)."""
    k = F.shape[1]
    cols, nm = [F[:, a] for a in range(k)], list(nomes)
    for a in range(k if grau == 2 else 0):
        for b in range(a, k):
            cols.append(F[:, a] * F[:, b])
            nm.append(f"{nomes[a]}*{nomes[b]}")
    B = np.stack(cols, 1)
    ok = np.isfinite(B).all(0) & (B.std(0) > 1e-9)
    B, nm = B[:, ok], [x for x, o in zip(nm, ok) if o]
    return B, nm


def lei_mdl(B, nomes, y, sig2, max_termos=8):
    """Selecao por MINIMOS QUADRADOS ORTOGONAIS (cada passo escolhe o termo que
    mais reduz o erro DEPOIS de reajustar todos — nao o mais correlacionado, que
    se perde: ver memoria guloso-nao-ortogonal), parada por bits:
    n/2 log2(RSS/n + sig2) + por termo (1/2 log2 n + log2 m)."""
    n, m = B.shape
    mu, sd = B.mean(0), B.std(0)
    Zp = (B - mu) / sd
    norm2 = (Zp ** 2).sum(0)
    yc = y - y.mean()
    res = yc.copy()
    rss = float(res @ res)

    def bits(r, k):
        return n / 2 * math.log2(r / n + sig2) + k * (0.5 * math.log2(n) + math.log2(m))

    melhor, sel = bits(rss, 0), []
    while len(sel) < max_termos:
        sc = (Zp.T @ res) ** 2 / np.maximum(norm2, 1e-12)
        sc[norm2 < 1e-8 * n] = -1
        if sel:
            sc[sel] = -1
        j = int(np.argmax(sc))
        if sc[j] <= 0:
            break
        b = bits(rss - float(sc[j]), len(sel) + 1)
        if b >= melhor:
            break
        q = Zp[:, j] / math.sqrt(norm2[j])
        res = res - q * float(q @ res)
        rss = float(res @ res)
        Zp = Zp - np.outer(q, q @ Zp)
        norm2 = (Zp ** 2).sum(0)
        melhor = b
        sel.append(j)
    termos = [nomes[s] for s in sel]
    coef = None
    if sel:
        Z = (B[:, sel] - mu[sel]) / sd[sel]
        coef, *_ = np.linalg.lstsq(Z, yc, rcond=None)
    return melhor, termos, sel, coef, (mu, sd, float(y.mean()))


def descreve(X, i, pontos, sig2):
    """Para o ser i: bits de cada ponto de vista (as duas componentes)."""
    out = {}
    for pv in pontos:
        F, nm, y = ponto_de_vista(X, i, pv)
        bases = [_base(F, nm, 1), _base(F, nm, 2)]
        tot, termos = 0.0, []
        for comp in (0, 1):
            # MDL sobre classes: o mais simples primeiro (so atomos), depois
            # produtos; fica a de menos bits (+1 bit para dizer qual classe)
            cand = [lei_mdl(B, nmb, y[:, comp], sig2) for B, nmb in bases]
            b, t, *_ = min(cand, key=lambda z: z[0])
            tot += b + 1.0
            termos.append(t)
        out[pv] = (tot + math.log2(len(pontos)), termos)
    return out


def pontos_de_vista(n, i):
    pv = ["mundo", "eu"] + [f"ser {j}" for j in range(n) if j != i] + \
         [f"junto {j}" for j in range(n) if j != i] + ["placebo a", "placebo b"]
    return pv
