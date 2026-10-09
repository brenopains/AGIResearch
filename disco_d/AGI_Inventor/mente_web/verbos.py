"""
mente_web.verbos — VERBOS VISUAIS: a transformacao como coisa separada da cena
================================================================================
Pergunta do usuario (08/10): ao ver a Lua "girando", ela extraiu a ROTACAO, o tamanho,
a cor — como algo que da para levar a outras cenas e compor? Resposta: nao. O
preditor so antecipava o proximo quadro daquele video. Este orgao faz a separacao.

  E (o "como mudou"):  de dois mapas do cortex (antes, depois) tira UM codigo curto
                       tau (8 numeros), com media sobre a imagem inteira: tau nao
                       sabe O QUE esta na cena, so COMO ela mudou.
  F (aplicar o verbo): mapa antes + tau -> mapa depois. Com as coordenadas de cada
                       celula na entrada (girar e aproximar dependem de onde se esta).
  G (compor verbos):   tau(1->2) e tau(2->3) -> tau(1->3), SEM aplicar nada: um
                       operador que recebe transformacoes (segunda ordem).

Tudo aprende so com a experiencia DELA: quadros seguidos de video e os
micro-movimentos do proprio olho (olhar 8 px ao lado: deslocamento com copia do
comando motor, como um bebe aprende o que o proprio olho faz).

Verbos = agrupamentos de tau sem rotulo (vocabulario estavel). A SONDA (so para
medir, nunca treina) aplica a imagens dela transformacoes conhecidas — girar,
aproximar, deslizar, mudar a cor — e mede: (1) se os tau de cada familia se
separam; (2) TRANSPORTE: o tau tirado da imagem A, aplicado a imagem B, preve B
transformada melhor que um verbo errado (placebo); (3) COMPOSICAO: girar e depois
aproximar (aplicando os dois, ou compondo com G) preve o composto nunca visto.
Metodo e controles de novaphys/nuvem, segunda_ordem e primitiva_do_dado (AGI 2.0).

Ja existe no mundo: autoencoders de transformacao (Hinton 2011), redes relacionais
de pares de imagens (Memisevic 2013). Novo aqui: aprender so da propria vida na
internet, com verbos sem rotulo, testados por transporte x placebo e composicao.
"""
from __future__ import annotations

import os

import numpy as np

from cortex import CANAIS, GRADE, _col2im, _dlr, _im2col, _lr

DT = 8               # numeros do codigo de transformacao (tau)
H1, H2 = 32, 48      # canais internos de E e F
MEM = 1500           # pares guardados (replay)


class Verbos:
    FORMAS = {"E1": (9 * 3 * CANAIS, H1), "e1": (H1,), "E2": (9 * H1, H1), "e2": (H1,), "E3": (H1, DT), "e3": (DT,),
              "F1": (9 * (CANAIS + DT + 2), H2), "f1": (H2,), "F2": (9 * H2, CANAIS), "f2": (CANAIS,),
              "G1": (2 * DT, 32), "g1": (32,), "G2": (32, DT), "g2": (DT,)}

    def __init__(self, semente=0):
        rng = np.random.default_rng(semente)
        self.p = {k: (rng.normal(0, np.sqrt(2.0 / f[0]), f) if len(f) == 2 else np.zeros(f)).astype(np.float32)
                  for k, f in self.FORMAS.items()}
        self.p["F2"] *= 0.1                           # comeca perto de "nada muda"
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.passo_adam = 0
        self.mu = np.zeros(CANAIS, np.float32)        # normalizacao corrente dos mapas do cortex
        self.sd = np.ones(CANAIS, np.float32)
        self.n_norm = 0
        self.pares = np.zeros((MEM, 2, GRADE, GRADE, CANAIS), np.float16)
        self.n_pares = 0
        self.trincas = np.zeros((MEM // 3, 3, GRADE, GRADE, CANAIS), np.float16)
        self.n_trincas = 0
        self.passos = 0
        self.curva = []
        self._err = []
        self._rng = rng
        y, x = np.mgrid[0:GRADE, 0:GRADE].astype(np.float32)
        self.coord = np.stack([(x - 3.5) / 3.5, (y - 3.5) / 3.5], -1)    # [8,8,2]

    # ── normalizacao (o cortex ainda aprende: a escala dos mapas anda) ──────
    def _norm(self, m, atualiza=False):
        m = np.asarray(m, np.float32)
        if atualiza:
            flat = m.reshape(-1, CANAIS)
            a = 1.0 / min(self.n_norm + 1, 200)
            self.mu = (1 - a) * self.mu + a * flat.mean(0)
            self.sd = (1 - a) * self.sd + a * (flat.std(0) + 1e-3)
            self.n_norm += 1
        return (m - self.mu) / self.sd

    # ── E: o codigo da mudanca ───────────────────────────────────────────────
    def _E(self, a, b):
        P = self.p
        x = np.concatenate([a, b, b - a], -1)
        c1 = _im2col(x, 3, 1, 1)
        z1 = c1 @ P["E1"] + P["e1"]
        h1 = _lr(z1)
        c2 = _im2col(h1, 3, 1, 1)
        z2 = c2 @ P["E2"] + P["e2"]
        h2 = _lr(z2)
        pool = h2.mean((1, 2))
        tau = pool @ P["E3"] + P["e3"]
        return tau, (x, c1, z1, h1, c2, z2, h2, pool)

    def _E_volta(self, dtau, cache, g):
        P = self.p
        x, c1, z1, h1, c2, z2, h2, pool = cache
        B = len(x)
        g["E3"] = g.get("E3", 0) + pool.T @ dtau
        g["e3"] = g.get("e3", 0) + dtau.sum(0)
        dh2 = np.broadcast_to((dtau @ P["E3"].T)[:, None, None, :] / (GRADE * GRADE), h2.shape)
        dz2 = dh2 * _dlr(z2)
        g["E2"] = g.get("E2", 0) + c2.reshape(-1, c2.shape[-1]).T @ dz2.reshape(-1, H1)
        g["e2"] = g.get("e2", 0) + dz2.sum((0, 1, 2))
        dh1 = _col2im(dz2 @ P["E2"].T, (B, GRADE, GRADE, H1), 3, 1, 1)
        dz1 = dh1 * _dlr(z1)
        g["E1"] = g.get("E1", 0) + c1.reshape(-1, c1.shape[-1]).T @ dz1.reshape(-1, H1)
        g["e1"] = g.get("e1", 0) + dz1.sum((0, 1, 2))

    # ── F: aplicar um verbo a um mapa ────────────────────────────────────────
    def _F(self, a, tau):
        P = self.p
        B = len(a)
        t = np.broadcast_to(tau[:, None, None, :], (B, GRADE, GRADE, DT))
        cc = np.broadcast_to(self.coord[None], (B, GRADE, GRADE, 2))
        x = np.concatenate([a, t, cc], -1)
        c1 = _im2col(x, 3, 1, 1)
        z1 = c1 @ P["F1"] + P["f1"]
        h1 = _lr(z1)
        c2 = _im2col(h1, 3, 1, 1)
        d = c2 @ P["F2"] + P["f2"]
        return a + d, (x, c1, z1, h1, c2)

    def _F_volta(self, dout, cache, g):
        """devolve (d mapa de entrada, d tau)."""
        P = self.p
        x, c1, z1, h1, c2 = cache
        B = len(x)
        g["F2"] = g.get("F2", 0) + c2.reshape(-1, c2.shape[-1]).T @ dout.reshape(-1, CANAIS)
        g["f2"] = g.get("f2", 0) + dout.sum((0, 1, 2))
        dh1 = _col2im(dout @ P["F2"].T, (B, GRADE, GRADE, H2), 3, 1, 1)
        dz1 = dh1 * _dlr(z1)
        g["F1"] = g.get("F1", 0) + c1.reshape(-1, c1.shape[-1]).T @ dz1.reshape(-1, H2)
        g["f1"] = g.get("f1", 0) + dz1.sum((0, 1, 2))
        dx = _col2im(dz1 @ P["F1"].T, x.shape, 3, 1, 1)
        da = dout + dx[..., :CANAIS]
        dtau = dx[..., CANAIS:CANAIS + DT].sum((1, 2))
        return da, dtau

    # ── G: compor dois verbos sem aplicar ────────────────────────────────────
    def _G(self, t1, t2):
        P = self.p
        x = np.concatenate([t1, t2], -1)
        z = x @ P["G1"] + P["g1"]
        h = _lr(z)
        return h @ P["G2"] + P["g2"], (x, z, h)

    def _adam(self, g, lr=1e-3, b1=0.9, b2=0.999):
        self.passo_adam += 1
        t = self.passo_adam
        for k, gk in g.items():
            gk = np.asarray(gk, np.float32)
            self.m[k] = b1 * self.m[k] + (1 - b1) * gk
            self.v[k] = b2 * self.v[k] + (1 - b2) * gk * gk
            self.p[k] -= (lr * (self.m[k] / (1 - b1 ** t)) / (np.sqrt(self.v[k] / (1 - b2 ** t)) + 1e-8)).astype(np.float32)

    # ── a vida: guardar pares e aprender ─────────────────────────────────────
    def guarda_par(self, a, b):
        self._norm(np.stack([a, b]), atualiza=True)
        self.pares[self.n_pares % MEM] = np.stack([a, b]).astype(np.float16)
        self.n_pares += 1

    def guarda_trinca(self, a, b, c):
        self.trincas[self.n_trincas % len(self.trincas)] = np.stack([a, b, c]).astype(np.float16)
        self.n_trincas += 1

    def perda_par(self, A, Bm):
        tau, ce = self._E(A, Bm)
        pred, cf = self._F(A, tau)
        return float(((pred - Bm) ** 2).mean()), float(((A - Bm) ** 2).mean())

    def aprende(self, lote=16):
        """um passo: pares (E+F reconstroem o 'depois' de 'antes' + tau) e, se houver,
        trincas (G compoe os dois tau no tau do salto inteiro)."""
        k = min(self.n_pares, MEM)
        if k < 8:
            return None
        idx = self._rng.integers(0, k, size=min(lote, k))
        par = self.pares[idx].astype(np.float32)
        A, Bm = self._norm(par[:, 0]), self._norm(par[:, 1])
        tau, ce = self._E(A, Bm)
        pred, cf = self._F(A, tau)
        perda = float(((pred - Bm) ** 2).mean())
        base = float(((A - Bm) ** 2).mean())
        g = {}
        dout = 2.0 * (pred - Bm) / pred.size
        _da, dtau = self._F_volta(dout, cf, g)
        self._E_volta(dtau, ce, g)
        kt = min(self.n_trincas, len(self.trincas))
        if kt >= 4:                                   # segunda ordem: compor sem aplicar
            tr = self.trincas[self._rng.integers(0, kt, size=min(8, kt))].astype(np.float32)
            a, b, c = self._norm(tr[:, 0]), self._norm(tr[:, 1]), self._norm(tr[:, 2])
            t12, _ = self._E(a, b)
            t23, _ = self._E(b, c)
            t13, _ = self._E(a, c)
            gp, (x, z, h) = self._G(t12, t23)
            dg = 2.0 * (gp - t13) / gp.size
            g["G2"] = h.T @ dg
            g["g2"] = dg.sum(0)
            dz = (dg @ self.p["G2"].T) * _dlr(z)
            g["G1"] = x.T @ dz
            g["g1"] = dz.sum(0)
        self._adam(g)
        self.passos += 1
        self._err.append(perda / (base + 1e-9))
        if len(self._err) >= 25:
            self.curva.append((self.passos, round(float(np.mean(self._err)), 4)))
            self._err = []
            if len(self.curva) > 1000:
                del self.curva[:250]
        return perda / (base + 1e-9)

    # ── usar ─────────────────────────────────────────────────────────────────
    def tau(self, a, b):
        t, _ = self._E(self._norm(np.asarray(a)[None]), self._norm(np.asarray(b)[None]))
        return t[0]

    def aplica(self, m, tau, vezes=1):
        """aplica o verbo tau ao mapa m (do cortex) 'vezes' vezes; devolve mapa do cortex."""
        x = self._norm(np.asarray(m)[None])
        for _ in range(vezes):
            x, _c = self._F(x, np.asarray(tau, np.float32)[None])
        return (x * self.sd + self.mu)[0]

    def compoe(self, t1, t2):
        g, _ = self._G(np.asarray(t1, np.float32)[None], np.asarray(t2, np.float32)[None])
        return g[0]

    # ── guardar ──────────────────────────────────────────────────────────────
    def salva(self, arq):
        d = {"p_" + k: v for k, v in self.p.items()}
        d.update({"m_" + k: v for k, v in self.m.items()})
        d.update({"v_" + k: v for k, v in self.v.items()})
        d.update(mu=self.mu, sd=self.sd, pares=self.pares[:min(self.n_pares, MEM)],
                 trincas=self.trincas[:min(self.n_trincas, len(self.trincas))],
                 contas=np.array([self.passo_adam, self.n_norm, self.n_pares, self.n_trincas, self.passos], np.int64),
                 curva=np.array(self.curva, np.float64).reshape(-1, 2))
        tmp = arq + ".tmp.npz"
        np.savez_compressed(tmp, **d)
        if os.path.exists(arq):
            os.replace(arq, arq + ".anterior")
        os.replace(tmp, arq)

    def carrega(self, arq):
        for a in (arq, arq + ".anterior"):
            if not os.path.exists(a):
                continue
            try:
                d = np.load(a)
                for k in self.p:
                    self.p[k], self.m[k], self.v[k] = d["p_" + k], d["m_" + k], d["v_" + k]
                self.mu, self.sd = d["mu"], d["sd"]
                pr, tr = d["pares"], d["trincas"]
                self.pares[:len(pr)], self.trincas[:len(tr)] = pr, tr
                self.passo_adam, self.n_norm, self.n_pares, self.n_trincas, self.passos = [int(v) for v in d["contas"]]
                self.curva = [(int(a_), float(b_)) for a_, b_ in d["curva"]]
                return True
            except Exception:
                continue
        return False


# ── a SONDA: so mede, nunca treina ───────────────────────────────────────────
FAMILIAS = ("girar", "aproximar", "deslizar", "cor")


def transforma(img, fam, sinal=1):
    """Transformacoes CONHECIDAS aplicadas a uma olhada dela (64x64 RGB uint8)."""
    from PIL import Image, ImageEnhance
    im = Image.fromarray(img)
    if fam == "girar":
        out = im.rotate(12 * sinal, resample=Image.BILINEAR, fillcolor=tuple(int(c) for c in img.reshape(-1, 3).mean(0)))
    elif fam == "aproximar":
        f = 1.2 if sinal > 0 else 0.85
        w = int(round(64 / f))
        o = (64 - w) // 2
        if f > 1:
            out = im.crop((o, o, o + w, o + w)).resize((64, 64), Image.BILINEAR)
        else:
            base = Image.new("RGB", (64, 64), tuple(int(c) for c in img.reshape(-1, 3).mean(0)))
            base.paste(im.resize((int(64 * f), int(64 * f)), Image.BILINEAR), ((64 - int(64 * f)) // 2,) * 2)
            out = base
    elif fam == "deslizar":
        out = Image.fromarray(np.roll(img, 8 * sinal, axis=1))
    else:
        out = ImageEnhance.Color(ImageEnhance.Brightness(im).enhance(1.25 if sinal > 0 else 0.8)).enhance(1.6)
    return np.asarray(out, np.uint8)


def sonda(verbos, cortex, olhadas, rng):
    """Mede, com olhadas dela que o orgao nunca viu transformadas desse jeito:
    separacao das familias, transporte (verbo de A em B) x placebo (verbo de outra
    familia) x nada muda, e composicao (girar + aproximar) x controles."""
    n = len(olhadas)
    if n < 12:
        return None
    mapas = {}
    for fam in FAMILIAS:
        tr = np.stack([transforma(o, fam) for o in olhadas])
        mapas[fam] = cortex.ve(tr)[0]
    base = cortex.ve(olhadas)[0]
    comp = cortex.ve(np.stack([transforma(transforma(o, "girar"), "aproximar") for o in olhadas]))[0]
    taus = {fam: np.stack([verbos.tau(base[i], mapas[fam][i]) for i in range(n)]) for fam in FAMILIAS}
    # (1) separacao: vizinho mais proximo (deixando o proprio de fora) acerta a familia?
    T = np.concatenate([taus[f] for f in FAMILIAS])
    rot = np.repeat(np.arange(len(FAMILIAS)), n)
    D = ((T[:, None] - T[None]) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    separa = float((rot[D.argmin(1)] == rot).mean())
    # (2) transporte: verbo tirado de A, aplicado a B (B != A)
    erro = lambda m, alvo: float(((m - alvo) ** 2).mean())   # noqa: E731
    vence, razao = [], []
    for fam in FAMILIAS:
        outra = [f for f in FAMILIAS if f != fam]
        for i in range(n):
            j = (i + 1 + int(rng.integers(n - 1))) % n
            certo = erro(verbos.aplica(base[i], taus[fam][j]), mapas[fam][i])
            plac = erro(verbos.aplica(base[i], taus[outra[int(rng.integers(3))]][j]), mapas[fam][i])
            nada = erro(base[i], mapas[fam][i])
            vence.append(certo < plac and certo < nada)
            razao.append(certo / (min(plac, nada) + 1e-9))
    por_fam = {fam: float(np.mean(vence[k * n:(k + 1) * n])) for k, fam in enumerate(FAMILIAS)}
    # (3) composicao: girar e depois aproximar (verbos de OUTRA olhada) -> composto nunca visto
    c_seq, c_g, c_um, c_errado = [], [], [], []
    for i in range(n):
        j = (i + 1 + int(rng.integers(n - 1))) % n
        tg, ta = taus["girar"][j], taus["aproximar"][j]
        alvo = comp[i]
        c_seq.append(erro(verbos.aplica(verbos.aplica(base[i], tg), ta), alvo))
        c_g.append(erro(verbos.aplica(base[i], verbos.compoe(tg, ta)), alvo))
        c_um.append(min(erro(verbos.aplica(base[i], tg), alvo), erro(verbos.aplica(base[i], ta), alvo)))
        c_errado.append(erro(verbos.aplica(verbos.aplica(base[i], taus["deslizar"][j]), taus["cor"][j]), alvo))
    return {"n": n, "separa": round(separa, 3), "acaso_separa": round(1 / len(FAMILIAS), 3),
            "transporte_vence": round(float(np.mean(vence)), 3), "transporte_por_familia": por_fam,
            "transporte_razao_mediana": round(float(np.median(razao)), 3),
            "compor_sequencia": round(float(np.mean(c_seq)), 4), "compor_G": round(float(np.mean(c_g)), 4),
            "compor_so_um": round(float(np.mean(c_um)), 4), "compor_errado": round(float(np.mean(c_errado)), 4),
            "compor_vence": round(float(np.mean([a < min(b, c) for a, b, c in zip(c_seq, c_um, c_errado)])), 3)}
