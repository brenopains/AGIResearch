"""
mundo_orbitas — o mundo das orbitas entregue como PIXELS (lado do avaliador)
==============================================================================

Gera episodios de video de 3 corpos em torno de um centro invisivel, com camera
afim desconhecida e armadilhas de percepcao (ver PRE_REGISTRO_etapa1.md). O
aprendiz so recebe `quadros`, `tempos` e as fronteiras de episodio; `verdade` e
do avaliador.

A aceleracao e a de `vendor/novaphys__orbita.py` (newton, GM = 1) e o RK4 com
passo 0,005 e o de `vendor/novaphys__inventa.py`.
"""

from __future__ import annotations

import numpy as np

LADO = 128
FPS = 30.0
DUR = 10.0


def acel(P):
    r = np.linalg.norm(P, axis=1, keepdims=True)
    return -P / r ** 3


def _deriv(S):
    return np.concatenate([S[:, 2:], acel(S[:, :2])], 1)


def integra(S0, ts, dt=0.005):
    S, t, out = S0.copy(), 0.0, [S0.copy()]
    for tn in ts[1:]:
        while t < tn - 1e-12:
            h = min(dt, tn - t)
            k1 = _deriv(S)
            k2 = _deriv(S + h / 2 * k1)
            k3 = _deriv(S + h / 2 * k2)
            k4 = _deriv(S + h * k3)
            S = S + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        out.append(S.copy())
    return np.stack(out, 1)                     # (corpos, T, 4)


def camera(rng):
    s = rng.uniform(16, 22)
    th = rng.uniform(0, 2 * np.pi)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    an = rng.uniform(0.85, 1.15)
    sh = rng.uniform(-0.15, 0.15)
    A = s * R @ np.array([[an, sh], [0, 1 / an]])
    if rng.random() < 0.5:
        A = A @ np.array([[1, 0], [0, -1]])
    t = np.array([LADO / 2, LADO / 2]) + rng.uniform(-10, 10, 2)
    return A, t


PALETA = {"A": [(230, 60, 60), (60, 200, 80), (70, 110, 240), (230, 200, 50),
                (200, 80, 220), (60, 210, 210)],
          "B": [(250, 140, 30), (140, 240, 120), (150, 90, 250), (240, 120, 180),
                (120, 180, 255), (255, 255, 140)]}
RAIO = {"A": 2.5, "B": 3.5}


def _condicoes(rng, n):
    out = []
    while len(out) < n:
        r = rng.uniform(0.8, 2.5)
        th = rng.uniform(0, 2 * np.pi)
        v = rng.uniform(0.8, 1.2) / np.sqrt(r)
        ang = rng.uniform(-0.4, 0.4)
        sentido = -1.0 if rng.random() < 0.3 else 1.0
        vt, vr = v * np.cos(ang) * sentido, v * np.sin(ang)
        c, s = np.cos(th), np.sin(th)
        out.append([r * c, r * s, vr * c - vt * s, vr * s + vt * c])
    return np.array(out)


def episodio(rng, cam, aparencia="A", armadilhas=True):
    A, t0 = cam
    passos = int(DUR * FPS)
    dts = (1 / FPS) * (1 + rng.uniform(-0.1, 0.1, passos))
    ts = np.concatenate([[0.0], np.cumsum(dts)[:-1]])
    R = RAIO[aparencia]
    cruza = armadilhas and rng.random() < 0.5    # T1: sorteado ANTES das condicoes
    for _ in range(200):                        # corpos que ficam dentro da imagem
        S0 = _condicoes(rng, 3)
        if cruza:                               # o gemeo: raio parecido, sentido oposto
            r0 = np.linalg.norm(S0[0, :2])
            ang = np.arctan2(S0[0, 1], S0[0, 0]) + rng.uniform(0.6, 1.6)
            r1 = r0 * rng.uniform(0.95, 1.05)
            v1 = np.linalg.norm(S0[0, 2:]) * rng.uniform(0.97, 1.03)
            sent = -np.sign(S0[0, 0] * S0[0, 3] - S0[0, 1] * S0[0, 2])
            c, s_ = np.cos(ang), np.sin(ang)
            S0[1] = [r1 * c, r1 * s_, -sent * v1 * s_, sent * v1 * c]
        S = integra(S0, ts)
        rr = np.linalg.norm(S[..., :2], axis=2)
        pix = S[..., :2] @ A.T + t0
        if rr.min() > 0.35 and pix.min() > R + 3 and pix.max() < LADO - R - 3:
            break
    cores = [PALETA[aparencia][i] for i in rng.choice(6, 3, replace=False)]
    arm = {"mesma_cor": False, "bicolor": None, "oclusor": None}
    if cruza:
        cores[1] = cores[0]
        arm["mesma_cor"] = True
    segunda = None
    if armadilhas and rng.random() < 0.5:
        arm["bicolor"] = 2
        outras = [c for c in PALETA[aparencia] if c not in cores]
        segunda = outras[rng.integers(len(outras))]
    if armadilhas and rng.random() < 0.5:
        x0, y0 = rng.integers(20, LADO - 44, 2)
        arm["oclusor"] = (int(x0), int(y0), 24)
    yy, xx = np.mgrid[0:LADO, 0:LADO].astype(float)
    quadros = np.zeros((passos, LADO, LADO, 3), np.uint8)
    for k in range(passos):
        img = np.zeros((LADO, LADO, 3))
        for b in range(3):
            cx, cy = pix[b, k]
            d = np.hypot(xx - cx, yy - cy)
            cob = np.clip(R + 0.5 - d, 0, 1)[..., None]
            cor = np.broadcast_to(np.array(cores[b], float), img.shape).copy()
            if arm["bicolor"] == b:
                lado = (xx < cx) if aparencia == "A" else (yy < cy)
                cor[lado] = segunda
            img = img * (1 - cob) + cor * cob
        if arm["oclusor"]:
            x0, y0, w = arm["oclusor"]
            img[y0:y0 + w, x0:x0 + w] = (128, 128, 128)
        quadros[k] = np.clip(img, 0, 255).astype(np.uint8)
    h = S[..., 0] * S[..., 3] - S[..., 1] * S[..., 2]
    verdade = {"pix": pix, "estado": S, "h": h[:, 0], "armadilhas": arm, "raio": R}
    return quadros, ts, verdade


def mundo(semente, n_ajuste=12, n_selecao=6, n_aval=12, armadilhas=True):
    """Os tres conjuntos de um mundo. A avaliacao troca a aparencia."""
    rng = np.random.default_rng(semente)
    cam = camera(rng)
    ajuste = [episodio(rng, cam, "A", armadilhas) for _ in range(n_ajuste)]
    selecao = [episodio(rng, cam, "A", armadilhas) for _ in range(n_selecao)]
    aval = [episodio(rng, cam, "B", armadilhas) for _ in range(n_aval)]
    return {"camera": cam, "ajuste": ajuste, "selecao": selecao, "aval": aval}
