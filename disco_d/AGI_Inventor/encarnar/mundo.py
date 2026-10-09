"""
encarnar.mundo — uma arena com varios seres (Heider e Simmel, computavel)
==========================================================================
A mente so recebe POSICOES ao longo do tempo (com ruido). Nada de tipo, nome,
alvo, forca. As regras abaixo ficam com o avaliador.

Seres:
  mola       fisica: puxado para o centro da arena, com atrito    a = -k p - c v
  ima        fisica: empurrado de lado pela propria velocidade    a = B rot(v)
  persegue Y agente: acerta a velocidade e VIRA para onde Y esta  (eixos do proprio corpo)
  foge Y     agente: acerta a velocidade e vira para LONGE de Y
  satelite Y fisica de outro corpo: cai em Y como um planeta      (centrado em Y)
  formacao L agente: guarda uma posicao fixa NO CORPO do lider L  (eixos do lider)
Reserva (so no teste, nunca vistos no desenvolvimento):
  intercepta Y  vira para onde Y VAI estar (mira a frente)
  distancia Y   persegue de longe e foge de perto (guarda distancia)
"""
from __future__ import annotations

import numpy as np

DT = 0.05
TIPOS_DEV = ("mola", "ima", "persegue", "foge", "satelite", "formacao")
TIPOS_RESERVA = ("intercepta", "distancia")
SIG_P = 2e-5           # ruido de medida da posicao (conhecido, como o SIG_A do organismo)


def rot(v):
    return np.stack([-v[..., 1], v[..., 0]], -1)


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


class Mundo:
    def __init__(self, rng, n=5, tipos=None):
        self.n = n
        self.rng_obs = np.random.default_rng(rng.integers(2 ** 31))
        if tipos is None:
            tipos = list(rng.choice(TIPOS_DEV, n))
        self.ent = []
        for i, t in enumerate(tipos):
            e = {"tipo": str(t)}
            outros = [j for j in range(n) if j != i]
            if t in ("persegue", "foge", "satelite", "formacao", "intercepta", "distancia"):
                e["alvo"] = int(rng.choice(outros))
            e["k"], e["c"] = float(rng.uniform(0.3, 0.8)), float(rng.uniform(0.1, 0.4))
            e["B"] = float(rng.uniform(0.6, 1.4)) * float(rng.choice([-1, 1]))
            e["v0"] = float(rng.uniform(0.6, 1.2))
            e["g1"], e["g2"] = float(rng.uniform(0.5, 1.2)), float(rng.uniform(1.0, 2.5))
            e["mu"] = float(rng.uniform(0.6, 1.4))
            e["o"] = rng.uniform(-1.0, 1.0, 2) + np.array([-1.0, 0.0])
            e["R"] = float(rng.uniform(1.0, 2.0))
            self.ent.append(e)
        # formacao precisa de lider que nao dependa dela (sem ciclo)
        for i, e in enumerate(self.ent):
            if e["tipo"] == "formacao" and self.ent[e["alvo"]]["tipo"] == "formacao":
                e["tipo"] = "persegue"
        self.P = rng.uniform(-3, 3, (n, 2))
        self.V = unit(rng.normal(0, 1, (n, 2))) * rng.uniform(0.5, 1.0, (n, 1))
        for i, e in enumerate(self.ent):
            if e["tipo"] == "satelite":            # comeca perto do corpo central, em orbita
                j = e["alvo"]
                d = rng.uniform(0.8, 1.5)
                a = rng.uniform(0, 2 * np.pi)
                rr = np.array([np.cos(a), np.sin(a)]) * d
                self.P[i] = self.P[j] + rr
                self.V[i] = self.V[j] + rot(rr / d) * np.sqrt(e["mu"] / d) * rng.uniform(0.9, 1.1)

    def acel(self, P, V):
        A = np.zeros_like(P)
        ordem = [i for i, e in enumerate(self.ent) if e["tipo"] != "formacao"] + \
                [i for i, e in enumerate(self.ent) if e["tipo"] == "formacao"]
        for i in ordem:
            e = self.ent[i]
            t = e["tipo"]
            v = V[i]
            sp = max(np.linalg.norm(v), 1e-9)
            th = v / sp
            nh = rot(th)
            if t == "mola":
                A[i] = -e["k"] * P[i] - e["c"] * v
            elif t == "ima":
                A[i] = e["B"] * rot(v)
            elif t in ("persegue", "foge", "intercepta", "distancia"):
                j = e["alvo"]
                d = P[j] - P[i]
                if t == "intercepta":
                    d = d + 1.0 * V[j]               # mira onde Y vai estar
                dist = max(np.linalg.norm(d), 1e-9)
                seno = (d @ nh) / dist
                sinal = -1.0 if t == "foge" else 1.0
                if t == "distancia":
                    sinal = np.tanh(dist - e["R"])   # longe: aproxima; perto: afasta
                a_frente = e["g1"] * (e["v0"] - sp)
                a_lado = sinal * e["g2"] * sp * seno
                A[i] = a_frente * th + a_lado * nh
            elif t == "satelite":
                j = e["alvo"]
                d = P[i] - P[j]
                r = max(np.linalg.norm(d), 1e-9)
                A[i] = A[j] - e["mu"] * d / r ** 3
            elif t == "formacao":
                L = e["alvo"]
                vl = V[L]
                tl = vl / max(np.linalg.norm(vl), 1e-9)
                Rl = np.stack([tl, rot(tl)], 1)        # colunas: eixos do corpo do lider
                q = Rl.T @ (P[i] - P[L])
                qd = Rl.T @ (V[i] - V[L])
                A[i] = A[L] + Rl @ (e["k"] * (e["o"] - q) - e["c"] * 2 * qd)
        return A

    def simula(self, T=24.0):
        """Devolve posicoes observadas (passos, n, 2) com ruido de medida."""
        n_p = int(T / DT)
        P, V = self.P.copy(), self.V.copy()
        out = [P.copy()]
        h = DT / 4
        for _ in range(n_p):
            for _s in range(4):                      # RK2 com passo fino
                A1 = self.acel(P, V)
                Pm, Vm = P + h / 2 * V, V + h / 2 * A1
                A2 = self.acel(Pm, Vm)
                P, V = P + h * Vm, V + h * A2
            out.append(P.copy())
        X = np.stack(out)
        return X, X + self.rng_obs.normal(0, SIG_P, X.shape)
