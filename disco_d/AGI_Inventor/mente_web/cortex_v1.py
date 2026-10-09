"""
mente_web.cortex — o CORTEX VISUAL dela, que aprende sozinho e do zero
=======================================================================
Antes a visao era 32x32 em cinza e um dicionario de retalhos 8x8: menos que a
retina de um peixe. Aqui:

  - FOVEA colorida 64x64 em resolucao total da tela (numa tela comum vista a
    ~60 cm isso e ~2 graus de angulo visual: o tamanho da fovea humana);
  - uma rede que COMPRIME cada olhada em 64 numeros e outra que tenta
    reconstruir a olhada a partir deles. Nenhum peso pre-treinado: comeca vendo
    borrao e aprende so com o que ELA olha;
  - surpresa = erro de reconstrucao (o que o cortex ainda nao sabe ver);
  - curiosidade = PROGRESSO de compressao em olhadas que ele ainda nao treinou
    (Schmidhuber), como ja era a curiosidade do texto;
  - TEMPO: um preditor que, de dois quadros, preve o codigo do proximo. Assistir
    um video e antecipar o que vem; o erro dessa previsao e a surpresa do video;
  - SALIENCIA aprendida: o mapa de onde o cortex erra mais na tela (meia
    resolucao) e para onde a atencao quer ir (cf. Itti e Baldi 2009: surpresa
    bayesiana atrai o olhar humano). Nao vem do HTML: vem do que ela ja sabe ver.

Numpy puro (sem biblioteca de rede neural): ~0,18 M parametros, poucos MB de RAM.
Camadas: retalhos 8x8 -> 48 canais -> blocos 2x2 -> 64 canais -> codigo de 64;
o decodificador e o espelho. (Convolucao com passo = tamanho do filtro.)
"""
from __future__ import annotations

import io
import os

import numpy as np

LADO = 64            # fovea: 64x64 pixels em resolucao total
DIM = 64             # o codigo de cada olhada
MEMORIA = 2048       # olhadas guardadas para reaprender (replay, como o hipocampo no sono)


def _lr(x):
    return np.where(x > 0, x, 0.1 * x)


def _dlr(x):
    return np.where(x > 0, 1.0, 0.1).astype(np.float32)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _patch(x, k):
    B, H, W, C = x.shape
    return x.reshape(B, H // k, k, W // k, k, C).transpose(0, 1, 3, 2, 4, 5).reshape(B, H // k, W // k, k * k * C)


def _unpatch(p, k, C):
    B, h, w, _ = p.shape
    return p.reshape(B, h, w, k, k, C).transpose(0, 1, 3, 2, 4, 5).reshape(B, h * k, w * k, C)


class Cortex:
    FORMAS = {"W1": (192, 48), "b1": (48,), "W2": (192, 64), "b2": (64,), "W3": (1024, DIM), "b3": (DIM,),
              "V3": (DIM, 1024), "c3": (1024,), "V2": (64, 192), "c2": (192,), "V1": (48, 192), "c1": (192,),
              "T1": (2 * DIM, 128), "t1": (128,), "T2": (128, DIM), "t2": (DIM,)}

    def __init__(self, semente=0):
        rng = np.random.default_rng(semente)
        self.p = {}
        for k, f in self.FORMAS.items():
            self.p[k] = (rng.normal(0, np.sqrt(2.0 / f[0]), f) if len(f) == 2 else np.zeros(f)).astype(np.float32)
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.passo_adam = 0
        self.mem = np.zeros((MEMORIA, LADO, LADO, 3), np.uint8)
        self.n_mem = 0
        self.olhadas = 0               # quantas olhadas ja aprendeu
        self.passos_tempo = 0
        self.curva = []                # (olhadas, erro medio recente): a curva de aprender a ver
        self._erros = []
        self._rng = rng

    # ── a rede ───────────────────────────────────────────────────────────────
    def _fwd(self, imgs):
        P = self.p
        x = np.asarray(imgs, np.float32) / 255.0
        B = len(x)
        p1 = _patch(x, 8)
        a1 = p1 @ P["W1"] + P["b1"]
        p2 = _patch(_lr(a1), 2)
        a2 = p2 @ P["W2"] + P["b2"]
        h2 = _lr(a2).reshape(B, 1024)
        z = h2 @ P["W3"] + P["b3"]
        a3 = z @ P["V3"] + P["c3"]
        g2 = _lr(a3).reshape(B, 4, 4, 64)
        a4 = g2 @ P["V2"] + P["c2"]
        g1 = _unpatch(_lr(a4), 2, 48)
        a5 = g1 @ P["V1"] + P["c1"]
        o = _sig(a5)
        r = _unpatch(o, 8, 3)
        return x, r, (p1, a1, p2, a2, h2, z, a3, g2, a4, g1, o)

    def _decod(self, z):
        P = self.p
        B = len(z)
        g2 = _lr(z @ P["V3"] + P["c3"]).reshape(B, 4, 4, 64)
        g1 = _unpatch(_lr(g2 @ P["V2"] + P["c2"]), 2, 48)
        return _unpatch(_sig(g1 @ P["V1"] + P["c1"]), 8, 3)

    def _grad(self, x, r, c):
        P = self.p
        p1, a1, p2, a2, h2, z, a3, g2, a4, g1, o = c
        B = len(x)
        g = {}
        dr = 2.0 * (r - x) / r.size
        da5 = _patch(dr, 8) * o * (1 - o)
        g["V1"] = g1.reshape(-1, 48).T @ da5.reshape(-1, 192)
        g["c1"] = da5.reshape(-1, 192).sum(0)
        da4 = _patch(da5 @ P["V1"].T, 2) * _dlr(a4)
        g["V2"] = g2.reshape(-1, 64).T @ da4.reshape(-1, 192)
        g["c2"] = da4.reshape(-1, 192).sum(0)
        da3 = (da4 @ P["V2"].T).reshape(B, 1024) * _dlr(a3)
        g["V3"] = z.T @ da3
        g["c3"] = da3.sum(0)
        dz = da3 @ P["V3"].T
        g["W3"] = h2.T @ dz
        g["b3"] = dz.sum(0)
        da2 = (dz @ P["W3"].T).reshape(B, 4, 4, 64) * _dlr(a2)
        g["W2"] = p2.reshape(-1, 192).T @ da2.reshape(-1, 64)
        g["b2"] = da2.reshape(-1, 64).sum(0)
        da1 = _unpatch(da2 @ P["W2"].T, 2, 48) * _dlr(a1)
        g["W1"] = p1.reshape(-1, 192).T @ da1.reshape(-1, 48)
        g["b1"] = da1.reshape(-1, 48).sum(0)
        return g

    def _adam(self, g, lr=1e-3, b1=0.9, b2=0.999):
        self.passo_adam += 1
        t = self.passo_adam
        for k, gk in g.items():
            gk = gk.astype(np.float32)
            self.m[k] = b1 * self.m[k] + (1 - b1) * gk
            self.v[k] = b2 * self.v[k] + (1 - b2) * gk * gk
            mh = self.m[k] / (1 - b1 ** t)
            vh = self.v[k] / (1 - b2 ** t)
            self.p[k] -= (lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)

    # ── ver ──────────────────────────────────────────────────────────────────
    def ve(self, imgs, reconstroi=False):
        """olhadas uint8 [B,64,64,3] -> (codigos [B,DIM], erro [B], reconstrucao uint8 ou None)"""
        x, r, c = self._fwd(imgs)
        e = ((r - x) ** 2).mean((1, 2, 3))
        rec = (np.clip(r, 0, 1) * 255).astype(np.uint8) if reconstroi else None
        return c[5], e, rec

    def aprende(self, imgs):
        """um passo de aprendizado com as olhadas novas + lembrancas (replay)."""
        imgs = np.asarray(imgs, np.uint8)
        for im in imgs:
            self.mem[self.n_mem % MEMORIA] = im
            self.n_mem += 1
        k = min(self.n_mem, MEMORIA)
        lemb = self.mem[self._rng.integers(0, k, size=min(16, k))]
        lote = np.concatenate([imgs, lemb])
        x, r, c = self._fwd(lote)
        perda = float(((r - x) ** 2).mean())
        self._adam(self._grad(x, r, c))
        self.olhadas += len(imgs)
        self._erros.append(perda)
        if len(self._erros) >= 50:
            self.curva.append((self.olhadas, round(float(np.mean(self._erros)), 5)))
            self._erros = []
            if len(self.curva) > 2000:
                del self.curva[:500]
        return perda

    def progresso(self, treino, teste):
        """Progresso de compressao: erro nas olhadas de TESTE antes e depois de aprender
        com as de TREINO (as de teste ainda nao foram treinadas)."""
        _z, e0, _r = self.ve(teste)
        self.aprende(treino)
        _z, e1, _r = self.ve(teste)
        return float(e0.mean() - e1.mean()), float(e0.mean())

    # ── tempo: antecipar o proximo quadro ────────────────────────────────────
    def preve(self, z_ant, z_atual, imagina=False):
        P = self.p
        h = _lr(np.concatenate([z_ant, z_atual])[None] @ P["T1"] + P["t1"])
        zp = h @ P["T2"] + P["t2"]
        img = (np.clip(self._decod(zp)[0], 0, 1) * 255).astype(np.uint8) if imagina else None
        return zp[0], img

    def aprende_tempo(self, Z):
        """Z: codigos de quadros seguidos [T,DIM]. Treina o preditor e devolve o erro
        medio da previsao ANTES de aprender (a surpresa do trecho)."""
        Z = np.asarray(Z, np.float32)
        if len(Z) < 3:
            return None
        P = self.p
        ent = np.concatenate([Z[:-2], Z[1:-1]], 1)
        alvo = Z[2:]
        a = ent @ P["T1"] + P["t1"]
        h = _lr(a)
        pr = h @ P["T2"] + P["t2"]
        perda = float(((pr - alvo) ** 2).mean())
        d = 2.0 * (pr - alvo) / pr.size
        da = (d @ P["T2"].T) * _dlr(a)
        self._adam({"T2": h.T @ d, "t2": d.sum(0), "T1": ent.T @ da, "t1": da.sum(0)})
        self.passos_tempo += 1
        return perda

    # ── saliencia: onde ela ainda nao sabe ver ──────────────────────────────
    def saliencia(self, tela, passo=48):
        """tela RGB uint8 [H,W,3] em resolucao total. Olha a tela a MEIA resolucao em
        janelas 64x64 e devolve [(erro, cx, cy)] em coordenadas da tela."""
        h, w = tela.shape[:2]
        meia = tela[: h - h % 2, : w - w % 2].reshape(h // 2, 2, w // 2, 2, 3).mean((1, 3)).astype(np.uint8)
        H, W = meia.shape[:2]
        jan, centros = [], []
        for y in range(0, max(1, H - LADO + 1), passo):
            for x in range(0, max(1, W - LADO + 1), passo):
                jan.append(meia[y:y + LADO, x:x + LADO])
                centros.append(((x + LADO / 2) * 2, (y + LADO / 2) * 2))
        if not jan:
            return []
        _z, e, _r = self.ve(np.stack(jan))
        return [(float(e[i]), centros[i][0], centros[i][1]) for i in range(len(jan))]

    # ── guardar ──────────────────────────────────────────────────────────────
    def salva(self, arq):
        tmp = arq + ".tmp.npz"
        d = {"p_" + k: v for k, v in self.p.items()}
        d.update({"m_" + k: v for k, v in self.m.items()})
        d.update({"v_" + k: v for k, v in self.v.items()})
        d["mem"] = self.mem[:min(self.n_mem, MEMORIA)]
        d["contas"] = np.array([self.passo_adam, self.n_mem, self.olhadas, self.passos_tempo], np.int64)
        d["curva"] = np.array(self.curva, np.float64).reshape(-1, 2)
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
                m = d["mem"]
                self.mem[:len(m)] = m
                self.passo_adam, self.n_mem, self.olhadas, self.passos_tempo = [int(v) for v in d["contas"]]
                self.curva = [(int(a_), float(b_)) for a_, b_ in d["curva"]]
                return True
            except Exception:
                continue
        return False


def fovea(tela, cx, cy, lado=LADO):
    """Recorta a fovea (resolucao total) centrada em (cx, cy); borda preenchida de cinza."""
    h, w = tela.shape[:2]
    x0, y0 = int(round(cx - lado / 2)), int(round(cy - lado / 2))
    out = np.full((lado, lado, 3), 128, np.uint8)
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(w, x0 + lado), min(h, y0 + lado)
    if xb > xa and yb > ya:
        out[ya - y0:yb - y0, xa - x0:xb - x0] = tela[ya:yb, xa:xb]
    return out


def reduz(img, lado=LADO):
    """Uma imagem inteira (ou quadro de video) vista de uma vez: reduzida a 64x64."""
    from PIL import Image
    return np.asarray(Image.fromarray(img).resize((lado, lado), Image.BILINEAR), np.uint8)


def jpg_de(img, q=80):
    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(img).save(b, "JPEG", quality=q)
    return b.getvalue()
