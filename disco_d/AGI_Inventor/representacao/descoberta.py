"""
descoberta — a grandeza que se conserva, e o uso dela para prever
====================================================================

Lado do APRENDIZ. Sobre o estado aprendido z = (u, v, u', v') de cada entidade,
a classe barata da proposta: q(z) = z'Qz + b'z — os 14 monomios de grau <= 2,
resolvidos JUNTOS (sem enumerar subconjuntos) pelo menor autovetor generalizado
de (variancia DENTRO das trajetorias, variancia TOTAL). Nada nomeia h, centro ou
lei; o centro, se precisar, sai dos termos lineares.

USO: um campo de aceleracao aprendido por vizinhos (k = 8) sobre as posicoes do
ajuste — sem lei nenhuma — rola o estado para a frente; a variante com a grandeza
projeta a velocidade a cada passo para manter q no valor inicial.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.spatial import cKDTree

NOMES = ["u", "v", "du", "dv", "u2", "uv", "udu", "udv", "v2", "vdu", "vdv",
         "du2", "dudv", "dv2"]


def monomios(Z):
    u, v, du, dv = Z[:, 0], Z[:, 1], Z[:, 2], Z[:, 3]
    return np.stack([u, v, du, dv, u * u, u * v, u * du, u * dv, v * v, v * du,
                     v * dv, du * du, du * dv, dv * dv], 1)


def _grad_vel(Z):
    """d monomios / d(du, dv): (n, 14, 2)."""
    u, v, du, dv = Z[:, 0], Z[:, 1], Z[:, 2], Z[:, 3]
    z0 = np.zeros_like(u)
    o = np.ones_like(u)
    g_du = np.stack([z0, z0, o, z0, z0, z0, u, z0, z0, v, z0, 2 * du, dv, z0], 1)
    g_dv = np.stack([z0, z0, z0, o, z0, z0, z0, u, z0, z0, v, z0, du, 2 * dv], 1)
    return np.stack([g_du, g_dv], 2)


def acha(trajs, ridge=1e-6):
    """trajs: lista de Z (n_i, 4). Devolve o modelo de q e o lambda do ajuste."""
    F = [monomios(Z) for Z in trajs if len(Z) >= 10]
    G = np.concatenate(F)
    mu, sd = G.mean(0), G.std(0)
    sd[sd < 1e-9] = 1.0
    Fs = [(f - mu) / sd for f in F]
    Gs = np.concatenate(Fs)
    T = Gs.T @ Gs / len(Gs)
    W = sum((f - f.mean(0)).T @ (f - f.mean(0)) for f in Fs) / len(Gs)
    T += ridge * np.trace(T) / len(T) * np.eye(len(T))
    vals, vecs = eigh(W, T)
    w = vecs[:, 0]
    w = w / np.linalg.norm(w)
    return {"mu": mu, "sd": sd, "w": w, "lam": float(vals[0])}


def q(mod, Z):
    return ((monomios(Z) - mod["mu"]) / mod["sd"]) @ mod["w"]


def lam(mod, trajs):
    qs = [q(mod, Z) for Z in trajs if len(Z) >= 10]
    if not qs:
        return 1.0
    todos = np.concatenate(qs)
    dentro = sum(((x - x.mean()) ** 2).sum() for x in qs) / len(todos)
    total = todos.var()
    return float(dentro / total) if total > 0 else 1.0


def sigma_dentro(mod, trajs):
    qs = [q(mod, Z) for Z in trajs if len(Z) >= 10]
    return float(np.sqrt(sum(((x - x.mean()) ** 2).sum() for x in qs)
                         / sum(len(x) for x in qs)))


# ================================================================ o USO
class Campo:
    """Aceleracao aprendida por vizinhos: nenhuma lei, so o que foi visto."""

    def __init__(self, P, A, k=8):
        self.arv = cKDTree(P)
        self.A = A
        self.k = k

    def __call__(self, x):
        d, i = self.arv.query(x, self.k)
        w = 1 / (d + 1e-3)
        return (self.A[i] * w[:, None]).sum(0) / w.sum()


def rola(z0, tempos, campo, mod=None):
    """Euler semi-implicito; com `mod`, projeta a velocidade para manter q(z0)."""
    z = z0.astype(float).copy()
    alvo = q(mod, z[None])[0] if mod is not None else None
    out = [z[:2].copy()]
    for dt in np.diff(tempos):
        z[2:] += campo(z[:2]) * dt
        if mod is not None:
            for _ in range(2):
                qz = q(mod, z[None])[0]
                g = (_grad_vel(z[None])[0] * (mod["w"] / mod["sd"])[:, None]).sum(0)
                n2 = g @ g
                if n2 > 1e-12:
                    z[2:] += (alvo - qz) * g / n2
        z[:2] += z[2:] * dt
        out.append(z[:2].copy())
    return np.array(out)
