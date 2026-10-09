"""
mente_web.arena — um mundo FISICO que ela pode assistir (lado do MUNDO, nao da mente)
======================================================================================
Pedido de origem (ARQUITETURA.md, secao 23, e a revisao externa do GPT de 09/10): a
internet e so-leitura e quase nunca mostra uma lei limpa; para ela descobrir um
conceito que ninguem deu, reutiliza-lo num problema diferente e revisa-lo quando o
mundo o contradiz, precisa de um lugar onde existam leis — e onde elas MUDEM sem
aviso. A arena e esse lugar. Ela chega pela porta "arena" e ASSISTE, como a um video
qualquer: quadros RGB 128x128, 5 por segundo, em tempo real. Nada mais.

O que ela NAO recebe: o nome do mundo, a lei, a camera, quantos corpos ha, quem e
quem, onde fica o centro, o que e conservado. Tudo isso esta em `verdade`, que so o
avaliador le (experimento_laboratorio.py) — a mente nunca abre arena.json.

Os mundos (unidades do mundo; a camera afim os leva para pixels):
  giro        a = +C_GIRO   * s^2 * rot90(v)   cada corpo gira com velocidade angular
                                               proporcional ao QUADRADO da propria
                                               rapidez: o conceito util e s^2
  contragiro  a = -C_CONTRA * s^2 * rot90(v)   outra lei, o MESMO conceito (s^2):
                                               serve para medir reutilizacao
  mola_h      a = -C_H * h^2 * x               mola central cuja rigidez depende do
                                               momento angular h = x vy - y vx (o
                                               centro e invisivel): o conceito util e h
  mola        a = -K * x                       mola comum: ha MUITAS grandezas
                                               conservadas (h, energia, Fradkin), e
                                               NENHUMA e necessaria para a lei —
                                               controle de "conservado nao e util"

Armadilhas de percepcao (de PRE_REGISTRO_etapa1.md): oclusor cinza parado por cima
(o corpo some atras dele e volta), dois corpos da MESMA cor, corpo bicolor.

Na vida dela os universos se sucedem a cada 12 visitas (giro, mola_h, contragiro,
mola, de novo giro...), sem aviso. A camera e uma so por vida.

MAOS (transplante 11): numa visita ela pode, em vez de so assistir, MONTAR a cena —
dizer onde cada coisa comeca e com que velocidade, nas coordenadas que ELA percebe
(centro do quadro = 0, meia-largura = 1; velocidade em meias-larguras por segundo):
    arena://k?exp=x,y,vx,vy;x,y,vx,vy;...
A arena converte pela camera (que ela nao conhece) e roda a lei do universo daquela
visita. O mundo impoe limites (no maximo 4 corpos, posicao dentro do quadro, rapidez
<= MAX_VEL_EXP) e nao promete que as coisas fiquem a vista: se sairem do quadro, saem.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np

LADO = 128
FPS = 5.0
MAX_CORPOS_EXP = 4
MAX_VEL_EXP = 0.6        # meias-larguras por segundo (~38 px/s): mais que isto o quadro nem acompanha
DUR = 24.0
C_GIRO, C_CONTRA, C_H, K_MOLA = 1.0, 1.4, 0.09, 0.25
UNIVERSOS = ("giro", "mola_h", "contragiro", "mola")
VISITAS_POR_UNIVERSO = 12
PALETA = [(230, 60, 60), (60, 200, 80), (70, 110, 240), (230, 200, 50), (200, 80, 220),
          (60, 210, 210), (240, 140, 40), (220, 220, 220)]
CINZA_OCLUSOR = (110, 110, 110)


# ── a fisica ──────────────────────────────────────────────────────────────────
def acel(mundo, S):
    x, v = S[:, :2], S[:, 2:]
    rot = np.stack([-v[:, 1], v[:, 0]], 1)
    s2 = (v ** 2).sum(1, keepdims=True)
    if mundo == "giro":
        return C_GIRO * s2 * rot
    if mundo == "contragiro":
        return -C_CONTRA * s2 * rot
    if mundo == "mola_h":
        h = (x[:, 0] * v[:, 1] - x[:, 1] * v[:, 0])[:, None]
        return -C_H * h ** 2 * x
    if mundo == "mola":
        return -K_MOLA * x
    raise ValueError(mundo)


def integra(mundo, S0, tempos, dt=0.02):
    def d(S):
        return np.concatenate([S[:, 2:], acel(mundo, S)], 1)
    S, t, out = S0.astype(float).copy(), 0.0, []
    for tn in tempos:
        while t < tn - 1e-12:
            h = min(dt, tn - t)
            k1 = d(S)
            k2 = d(S + h / 2 * k1)
            k3 = d(S + h / 2 * k2)
            k4 = d(S + h * k3)
            S = S + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        out.append(S.copy())
    return np.stack(out, 1)                          # (corpos, T, 4)


def camera(rng):
    """Camera afim fixa por vida, desconhecida dela."""
    s = rng.uniform(13.0, 15.0)
    th = rng.uniform(0, 2 * np.pi)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    an = rng.uniform(0.9, 1.1)
    sh = rng.uniform(-0.08, 0.08)
    A = s * R @ np.array([[an, sh], [0, 1 / an]])
    if rng.random() < 0.5:
        A = A @ np.array([[1, 0], [0, -1]])
    t = np.array([LADO / 2, LADO / 2]) + rng.uniform(-6, 6, 2)
    return A, t


def _estado_inicial(mundo, rng):
    if mundo in ("giro", "contragiro"):
        s = rng.uniform(0.4, 1.0 if mundo == "giro" else 0.85)
        x = rng.uniform(-2.6, 2.6, 2)
        ang = rng.uniform(0, 2 * np.pi)
        return np.array([x[0], x[1], s * np.cos(ang), s * np.sin(ang)])
    r = rng.uniform(1.2, 3.0)
    th = rng.uniform(0, 2 * np.pi)
    x = r * np.array([np.cos(th), np.sin(th)])
    tang = np.array([-np.sin(th), np.cos(th)]) * (1 if rng.random() < 0.7 else -1)
    rad = np.array([np.cos(th), np.sin(th)])
    if mundo == "mola_h":
        h = rng.uniform(0.9, 2.6)
        vt = h / r
    else:
        vt = rng.uniform(0.3, 1.3)
    v = vt * tang + rng.uniform(-0.3, 0.3) * rad
    return np.array([x[0], x[1], v[0], v[1]])


# ── um episodio ───────────────────────────────────────────────────────────────
@dataclass
class Episodio:
    quadros: np.ndarray                  # (T, LADO, LADO, 3) uint8 — o que ELA ve
    tempos: np.ndarray                   # (T,) segundos — o relogio do video
    verdade: dict = field(default_factory=dict)   # so o avaliador


def _desenha(img, c, raio, cor, cor2=None, eixo=None):
    if not np.all(np.isfinite(c)):
        return
    H, W = img.shape[:2]
    x0, x1 = max(0, int(c[0] - raio - 2)), min(W, int(c[0] + raio + 3))
    y0, y1 = max(0, int(c[1] - raio - 2)), min(H, int(c[1] + raio + 3))
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx + 0.5 - c[0], yy + 0.5 - c[1])
    cob = np.clip(raio + 0.5 - d, 0, 1)[..., None]
    if cor2 is not None:
        lado = ((xx + 0.5 - c[0]) * eixo[0] + (yy + 0.5 - c[1]) * eixo[1]) >= 0
        col = np.where(lado[..., None], np.array(cor2, float), np.array(cor, float))
    else:
        col = np.array(cor, float)
    reg = img[y0:y1, x0:x1].astype(float)
    img[y0:y1, x0:x1] = np.clip(reg * (1 - cob) + col * cob, 0, 255).astype(np.uint8)


def episodio(mundo, rng, cam, dur=DUR, n_corpos=None, armadilhas=True, tempo_jitter=0.01, estados0=None):
    """estados0: estados iniciais do MUNDO escolhidos por um experimento (ver de_percebido);
    sem eles, o mundo sorteia (e so aceita quem fica a vista o video todo)."""
    A, tc = cam
    T = int(round(dur * FPS))
    tempos = np.arange(T) / FPS + rng.uniform(-tempo_jitter, tempo_jitter, T)
    tempos[0] = 0.0
    tempos = np.maximum.accumulate(tempos)
    corpos = []
    if estados0 is not None:
        n = len(estados0)
        for S0 in estados0:
            tr = integra(mundo, np.asarray(S0, float)[None], tempos)[0]
            corpos.append({"mundo": tr, "px": tr[:, :2] @ A.T + tc})
    else:
        n = n_corpos or int(rng.choice([2, 3, 3, 3, 4]))
    for _ in range(n if estados0 is None else 0):
        for _tent in range(200):
            S0 = _estado_inicial(mundo, rng)
            tr = integra(mundo, S0[None], tempos)[0]
            px = tr[:, :2] @ A.T + tc
            if px.min() >= 6 and px.max() <= LADO - 6 and np.all(np.isfinite(px)):
                break
        corpos.append({"mundo": tr, "px": px})
    oc = None
    if armadilhas and rng.random() < 0.5:
        ox, oy = rng.uniform(30, LADO - 54, 2)
        oc = (float(ox), float(oy), 24.0, 24.0)
    cores = [PALETA[i] for i in rng.choice(len(PALETA), n, replace=False)]
    if armadilhas and n >= 2 and rng.random() < 0.3:
        cores[1] = cores[0]                              # dois corpos da MESMA cor
    bicolor = int(rng.integers(n)) if (armadilhas and rng.random() < 0.3) else -1
    raios = rng.uniform(2.5, 3.5, n)
    eixo = rng.normal(size=2)
    eixo /= np.linalg.norm(eixo)
    quadros = np.zeros((T, LADO, LADO, 3), np.uint8)
    for k in range(T):
        for i, c in enumerate(corpos):
            cor2 = PALETA[(PALETA.index(cores[i]) + 3) % len(PALETA)] if i == bicolor else None
            _desenha(quadros[k], c["px"][k], raios[i], cores[i], cor2, eixo)
        if oc is not None:
            x, y, w, h = (int(round(v)) for v in oc)
            quadros[k, y:y + h, x:x + w] = CINZA_OCLUSOR
    for i, c in enumerate(corpos):
        p = c["px"]
        vis = (p[:, 0] >= raios[i]) & (p[:, 0] <= LADO - raios[i]) & (p[:, 1] >= raios[i]) & (p[:, 1] <= LADO - raios[i])
        if oc is not None:
            x, y, w, h = oc
            dentro = (p[:, 0] > x - raios[i] * 0.5) & (p[:, 0] < x + w + raios[i] * 0.5) & \
                     (p[:, 1] > y - raios[i] * 0.5) & (p[:, 1] < y + h + raios[i] * 0.5)
            vis &= ~dentro
        c.update(visivel=vis, cor=cores[i], raio=float(raios[i]), bicolor=(i == bicolor))
    return Episodio(quadros, tempos, {"mundo": mundo, "camera": (A, tc), "corpos": corpos, "oclusor": oc,
                                      "mesma_cor": bool(n >= 2 and cores[0] == cores[1])})


# ── grandezas verdadeiras em PIXELS (so o avaliador usa) ──────────────────────
def verdadeiras_px(cam, Zpx):
    """Zpx: estados (n,4) em pixels/segundo. -> dict com s2 e h verdadeiros (unidades do mundo)."""
    A, tc = cam
    Ai = np.linalg.inv(A)
    x = (Zpx[:, :2] - tc) @ Ai.T
    v = Zpx[:, 2:] @ Ai.T
    return {"s2": (v ** 2).sum(1), "h": x[:, 0] * v[:, 1] - x[:, 1] * v[:, 0]}


def universo_da_visita(k):
    return UNIVERSOS[(k // VISITAS_POR_UNIVERSO) % len(UNIVERSOS)]


# ── maos: o experimento dela, das coordenadas que ELA percebe para o mundo ────
def limita_experimento(estados):
    """o que o mundo aceita: ate MAX_CORPOS_EXP corpos, posicao dentro do quadro, rapidez
    ate MAX_VEL_EXP (coordenadas percebidas)."""
    out = []
    for u in list(estados)[:MAX_CORPOS_EXP]:
        u = np.asarray(u, float)
        if u.shape != (4,) or not np.all(np.isfinite(u)):
            continue
        p = np.clip(u[:2], -0.92, 0.92)
        v = u[2:]
        s = float(np.linalg.norm(v))
        if s > MAX_VEL_EXP:
            v = v * (MAX_VEL_EXP / s)
        out.append(np.concatenate([p, v]))
    return out


def de_percebido(cam, u):
    """estado percebido (x, y, vx, vy; centro 0, meia-largura 1, por segundo) -> estado do mundo."""
    A, tc = cam
    Ai = np.linalg.inv(A)
    u = np.asarray(u, float)
    x = Ai @ (u[:2] * (LADO / 2) + LADO / 2 - tc)
    v = Ai @ (u[2:] * (LADO / 2))
    return np.array([x[0], x[1], v[0], v[1]])


def url_experimento(k, estados):
    return f"arena://{int(k)}?exp=" + ";".join(",".join(f"{float(c):.4f}" for c in u)
                                               for u in limita_experimento(estados))


def le_url(url):
    """'arena://k' ou 'arena://k?exp=...' -> (k, estados percebidos ou None)."""
    corpo = str(url).split("//", 1)[1]
    k, _, q = corpo.partition("?")
    est = None
    if q.startswith("exp="):
        est = []
        for parte in q[4:].split(";"):
            try:
                est.append([float(c) for c in parte.split(",")])
            except ValueError:
                continue
        est = limita_experimento(est) or None
    return int(k), est


# ── a arena na vida dela ──────────────────────────────────────────────────────
class ArenaViva:
    """O MUNDO da arena na pasta da vida (arena.json): quantas visitas, a semente da
    camera. A mente nunca le este arquivo; ela so recebe quadros."""

    def __init__(self, pasta, semente=20261009):
        self.arq = os.path.join(pasta, "arena.json")
        self.st = {"visitas": 0, "semente": int(semente)}
        if os.path.exists(self.arq):
            try:
                self.st.update(json.load(open(self.arq, encoding="utf-8")))
            except Exception:
                pass
        self.cam = camera(np.random.default_rng(self.st["semente"]))

    def _grava(self):
        tmp = self.arq + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.st, f)
        os.replace(tmp, self.arq)

    def nova_visita(self):
        """-> url opaca da proxima experiencia (nao diz o universo)."""
        k = int(self.st["visitas"])
        self.st["visitas"] = k + 1
        self._grava()
        return f"arena://{k}"

    def gera(self, url, dur=DUR):
        k, est = le_url(url)
        rng = np.random.default_rng([self.st["semente"], k])
        if est is not None:
            return episodio(universo_da_visita(k), rng, self.cam, dur=dur,
                            estados0=[de_percebido(self.cam, u) for u in est])
        return episodio(universo_da_visita(k), rng, self.cam, dur=dur)

    def assistir(self, url, segundos, tempo_real=True):
        """Como corpo.assistir: gera (t, quadro RGB, audio=None). 1 s de video = 1 s de vida."""
        ep = self.gera(url, dur=min(DUR, float(segundos)))
        t0 = time.time()
        for t, q in zip(ep.tempos, ep.quadros):
            if tempo_real:
                espera = t0 + float(t) - time.time()
                if espera > 0:
                    time.sleep(espera)
            yield float(t), q, None
