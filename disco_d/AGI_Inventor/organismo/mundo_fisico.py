"""
organismo.mundo_fisico — o multiverso onde o organismo vive (dominio 1)
=========================================================================
Um fluxo de PROBLEMAS vindos de mundos de forca central e afins. O organismo
nunca recebe o nome do mundo: so dados. Cada problema traz:

  L (lei):      estados (x, y, vx, vy) com aceleracao medida (ruido conhecido),
                as vezes trajetorias curtas. Resposta: uma lei a(estado).
                Verificador: erro da lei em 400 estados novos do mesmo mundo.
  Q (pergunta): poucos estados rotulados por uma pergunta sobre o FUTURO do
                corpo (escapa? passa perto do centro? ...), mais trajetorias.
                Resposta: rotulos de 200 estados novos. Verificador: acuracia.

Variacao de superficie: a escala (mu e as constantes de cada mundo) muda a cada
problema. O que se conserva entre encontros e a FORMA da lei e o SUPORTE das
grandezas, nao os coeficientes.

Fontes: 'percebido' (o proprio organismo mediu), 'relato:A' (fonte que relata
mundos que tambem sao percebidos), 'relato:F' (fonte que relata mundos que nunca
aparecem percebidos — a ficcao deste brinquedo). Relatos nao tem verificador.

Nada aqui e lido pelo organismo alem dos dados e da fonte.
"""
from __future__ import annotations

import numpy as np

SIG_A = 0.004          # ruido do instrumento de aceleracao (conhecido pelo organismo)
SIG_X = 0.002          # ruido das trajetorias
T_TRAJ = 6.0
N_PTS = 25


# ─── as leis verdadeiras ─────────────────────────────────────────────────────

def params(fam, rng):
    p = {"mu": float(rng.uniform(0.6, 1.6))}
    if fam == "einstein":
        p["A"] = float(rng.uniform(0.02, 0.04))
    elif fam == "oblato":
        p["K"] = float(rng.uniform(0.02, 0.04))
    elif fam == "spin_orbita":
        p["J"] = float(rng.uniform(0.03, 0.06))
    elif fam == "magnetico":
        p["A"] = float(rng.uniform(0.05, 0.09))
    elif fam == "mola":
        p["k"] = float(rng.uniform(0.5, 1.5))
    elif fam == "arrasto":
        p["g"] = float(rng.uniform(0.03, 0.08))
    elif fam == "ciclotron":
        p["B"] = float(rng.uniform(0.6, 1.4))
    elif fam == "vortice":
        p["W"] = float(rng.uniform(0.05, 0.10))
    return p


def acel(fam, p, S):
    x, y, vx, vy = S[..., 0], S[..., 1], S[..., 2], S[..., 3]
    r = np.hypot(x, y)
    v = np.maximum(np.hypot(vx, vy), 1e-9)
    rh = np.stack([x / r, y / r], -1)
    nh = np.stack([-vy / v, vx / v], -1)
    mu = p["mu"]
    grav = -(mu / r ** 2)[..., None] * rh
    if fam == "kepler":
        return grav
    if fam == "einstein":
        h = x * vy - y * vx
        return grav - (p["A"] * h ** 2 / r ** 4)[..., None] * rh
    if fam == "oblato":
        return grav - (p["K"] / r ** 4)[..., None] * rh
    if fam == "spin_orbita":
        # acoplamento spin-orbita: termo radial proporcional a h (impar: depende
        # do SENTIDO da orbita; r, v, vr nao carregam sentido)
        h = x * vy - y * vx
        return grav - (p["J"] * h / r ** 4)[..., None] * rh
    if fam == "magnetico":
        E = 0.5 * v ** 2 - mu / r
        return grav + (p["A"] * E)[..., None] * nh
    if fam == "mola":
        return -p["k"] * np.stack([x, y], -1)
    if fam == "arrasto":
        return grav - p["g"] * np.stack([vx, vy], -1)
    if fam == "ciclotron":
        return p["B"] * np.stack([-vy, vx], -1)
    if fam == "vortice":
        return grav + (p["W"] * v / r)[..., None] * nh
    raise ValueError(fam)


def _v_circ(fam, p, r):
    if fam == "mola":
        return np.sqrt(p["k"]) * r
    if fam == "ciclotron":
        return np.ones_like(r)
    return np.sqrt(p["mu"] / r)


def estados(fam, p, rng, n, faixa=(0.6, 1.3)):
    r = rng.uniform(0.7, 3.0, n)
    th = rng.uniform(0, 2 * np.pi, n)
    v = _v_circ(fam, p, r) * rng.uniform(faixa[0], faixa[1], n)
    ang = rng.uniform(-0.6, 0.6, n)
    sentido = np.where(rng.random(n) < 0.3, -1.0, 1.0)
    vt, vr = v * np.cos(ang) * sentido, v * np.sin(ang)
    c, s = np.cos(th), np.sin(th)
    return np.stack([r * c, r * s, vr * c - vt * s, vr * s + vt * c], 1)


def _deriv(fam, p, S):
    return np.concatenate([S[:, 2:], acel(fam, p, S)], 1)


def integra(fam, p, S0, ts, dt=0.01):
    """RK4 vetorizado; devolve (n, len(ts), 4)."""
    S, t, out = S0.copy(), 0.0, [S0.copy()]
    for tn in ts[1:]:
        while t < tn - 1e-12:
            h = min(dt, tn - t)
            k1 = _deriv(fam, p, S)
            k2 = _deriv(fam, p, S + h / 2 * k1)
            k3 = _deriv(fam, p, S + h / 2 * k2)
            k4 = _deriv(fam, p, S + h * k3)
            S = S + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        out.append(S.copy())
    return np.stack(out, 1)


def trajetorias(fam, p, rng, n):
    m = 3 * n
    S0 = estados(fam, p, rng, m)
    T = integra(fam, p, S0, np.linspace(0, T_TRAJ, N_PTS), dt=0.02)
    rr = np.hypot(T[..., 0], T[..., 1])
    ok = (rr.min(1) > 0.35) & (rr.max(1) < 8) & np.isfinite(T).all((1, 2))
    T = T[ok][:n]
    return T + rng.normal(0, SIG_X, T.shape)


# ─── perguntas sobre o futuro ────────────────────────────────────────────────
# Cada molde devolve uma GRANDEZA verdadeira por estado; o rotulo e grandeza >
# LIMIAR. Perguntas de sinal/evento tem limiar natural (sentido: h > 0; escapa:
# r passa de 12); perguntas de tamanho usam a mediana da populacao do problema.
# O limiar nao e dado ao organismo: os exemplos o definem.

LIMIAR_NATURAL = {"sentido": 0.0, "raio_cresce": 0.0, "escapa": 12.0,
                  "periapse_direita": 0.0}
FAIXA_Q = (0.7, 1.6)

def _futuro(fam, p, S0, T, n=161):
    ts = np.linspace(0, T, n)
    return ts, integra(fam, p, S0, ts, dt=0.02)


QUANTIDADES = ("rmin", "rmax", "vmin", "vmax", "varre", "r_final", "x_final", "t_peri")
HORIZONTES = (5, 15, 40)


def moldes_gramatica():
    """Moldes LACRAVEIS: 'q:<grandeza>@<horizonte>'. Nenhum coincide com os
    moldes de desenvolvimento (que usam horizonte 3 ou 30)."""
    return [f"q:{q}@{h}" for q in QUANTIDADES for h in HORIZONTES]


def _grandeza_gramatica(molde, fam, p, S0):
    q, h = molde[2:].split("@")
    H = float(h)
    n = int(max(41, H * 8)) + 1
    ts, F = _futuro(fam, p, S0, H, n=n)
    r = np.hypot(F[..., 0], F[..., 1])
    v = np.hypot(F[..., 2], F[..., 3])
    r = np.where(np.isfinite(r), np.minimum(r, 1e3), 1e3)
    v = np.where(np.isfinite(v), np.minimum(v, 1e3), 1e3)
    if q == "rmin":
        return r.min(1)
    if q == "rmax":
        return r.max(1)
    if q == "vmin":
        return v.min(1)
    if q == "vmax":
        return v.max(1)
    if q == "varre":
        ang = np.unwrap(np.arctan2(F[..., 1], F[..., 0]), axis=1)
        return np.abs(ang[:, -1] - ang[:, 0])
    if q == "r_final":
        return r[:, -1]
    if q == "x_final":
        return np.where(np.isfinite(F[:, -1, 0]), F[:, -1, 0], 0.0)
    if q == "t_peri":
        dr = np.diff(r, axis=1)
        out = np.full(len(S0), H)
        for k in range(len(S0)):
            loc = np.where((dr[k, :-1] < 0) & (dr[k, 1:] >= 0))[0]
            if len(loc):
                out[k] = ts[loc[0] + 1]
        return out
    raise ValueError(molde)


def grandeza_q(molde, fam, p, S0):
    if molde.startswith("q:"):
        return _grandeza_gramatica(molde, fam, p, S0)
    x, y, vx, vy = S0.T
    if molde == "sentido":
        return x * vy - y * vx
    if molde == "raio_cresce":
        return x * vx + y * vy
    if molde in ("escapa", "rmin_pequeno", "vmax_alta", "periapse_direita",
                 "cruza_x_logo", "periodo_longo", "rmax_grande"):
        ts, F = _futuro(fam, p, S0, 30.0 if molde != "cruza_x_logo" else 3.0)
        r = np.hypot(F[..., 0], F[..., 1])
        r = np.where(np.isfinite(r), r, 1e9)
        if molde == "escapa":
            return r.max(1)
        if molde == "rmax_grande":
            return np.minimum(r.max(1), 50.0)
        if molde == "rmin_pequeno":
            return -r.min(1)
        if molde == "vmax_alta":
            v = np.hypot(F[..., 2], F[..., 3])
            return np.where(np.isfinite(v), v, 1e9).max(1)
        if molde == "periapse_direita":
            # primeiro minimo local de r depois de t=0 (ou o minimo global)
            i = np.argmin(r[:, 1:], 1) + 1
            dr = np.diff(r, axis=1)
            for k in range(len(S0)):
                loc = np.where((dr[k, :-1] < 0) & (dr[k, 1:] >= 0))[0]
                if len(loc):
                    i[k] = loc[0] + 1
            return F[np.arange(len(S0)), i, 0]
        if molde == "cruza_x_logo":
            ang = np.unwrap(np.arctan2(F[..., 1], F[..., 0]), axis=1)
            return -np.abs(ang[:, -1] - ang[:, 0])
        if molde == "periodo_longo":
            ang = np.unwrap(np.arctan2(F[..., 1], F[..., 0]), axis=1)
            vol = np.abs(ang[:, -1] - ang[:, 0]) / (2 * np.pi)
            return -vol
    raise ValueError(molde)


MOLDES_DEV = ("sentido", "escapa", "raio_cresce", "periapse_direita")
MOLDES_RESERVA = ("rmin_pequeno", "vmax_alta", "cruza_x_logo", "periodo_longo",
                  "rmax_grande")


# ─── o problema ──────────────────────────────────────────────────────────────

class Problema:
    """O que o organismo ve: dados, fonte, tipo, orcamento. O verificador fica
    fechado aqui dentro; so devolve sucesso/fracasso DEPOIS da resposta."""

    def __init__(self, idx, fam, fonte, tipo, rng, n_lab=24, n_traj=6,
                 com_traj=True, molde=None, n_ex=16):
        self.idx, self.fam, self.fonte, self.tipo, self.molde = idx, fam, fonte, tipo, molde
        self.p = params(fam, rng)
        self._rng_ver = np.random.default_rng(rng.integers(2 ** 31))
        self.T = trajetorias(fam, self.p, rng, n_traj) if com_traj else None
        if tipo == "L":
            self.S = estados(fam, self.p, rng, n_lab)
            self.A = acel(fam, self.p, self.S) + rng.normal(0, SIG_A, (n_lab, 2))
        else:
            S = estados(fam, self.p, rng, n_ex + 200, faixa=FAIXA_Q)
            g = grandeza_q(molde, fam, self.p, S)
            lim = LIMIAR_NATURAL.get(molde, np.median(g))
            y = (g > lim).astype(int)
            self.S, self.y = S[:n_ex], y[:n_ex]
            self._Sq, self._yq = S[n_ex:], y[n_ex:]
        self.verificavel = fonte == "percebido"

    # o que o organismo pode pedir
    def consultas(self):
        return self._Sq if self.tipo == "Q" else None

    # o verificador
    def verifica(self, resposta):
        if self.tipo == "L":
            lei = resposta
            if lei is None:
                return False, float("inf")
            S = estados(self.fam, self.p, self._rng_ver, 400)
            verd = acel(self.fam, self.p, S)
            try:
                pred = lei(S)
            except Exception:
                return False, float("inf")
            rms = float(np.sqrt(((pred - verd) ** 2).mean()))
            return bool(np.isfinite(rms) and rms < 1.0 * SIG_A), rms / SIG_A
        rot = resposta
        if rot is None:
            return False, 0.0
        acc = float((np.asarray(rot) == self._yq).mean())
        return acc >= 0.9, acc
