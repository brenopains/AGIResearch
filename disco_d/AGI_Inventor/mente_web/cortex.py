"""
mente_web.cortex — o CORTEX VISUAL dela, que aprende sozinho e do zero (versao 2)
==================================================================================
  - FOVEA colorida 64x64 em resolucao total da tela (numa tela comum vista a
    ~60 cm isso e ~2 graus de angulo visual: o tamanho da fovea humana);
  - um MAPA RETINOTOPICO 8x8x32 (2.048 numbers): cada celula descreve um pedaco
    de ~22 px da olhada (mais ou menos uma letra de texto comum), na posicao em
    que ele esta — como a primeira area visual do cortex humano (V1). O cortex
    reconstroi a olhada a partir desse mapa. Nenhum peso pre-treinado: comeca
    vendo borrao e aprende so com o que ELA olha;
  - surpresa = erro de reconstrucao; curiosidade = PROGRESSO de compressao numa
    olhada ainda nao treinada (Schmidhuber);
  - TEMPO: um preditor LOCAL (cada celula olha a vizinhanca 3x3 nos dois quadros
    anteriores) preve o mapa do proximo quadro — como movimento: o que esta aqui
    vai para ali. O erro dessa previsao e a surpresa do video;
  - SALIENCIA aprendida: onde o cortex mais erra na tela (meia resolucao).

Versao 1 (cortex_v1.py) espremia a olhada inteira em 64 numeros: o erro parou de
cair em ~4.000 olhadas (teto de capacidade) e ladrilhos 8x8 sem sobreposicao
deixavam a reconstrucao quadriculada. Aqui as convolucoes se sobrepoem (filtro 4,
passo 2) e o gargalo e 32x maior.

Antes do cortex, a RETINA: tira o brilho medio e normaliza o contraste de cada olhada
(sem isso a v2 aprendia so a desenhar fundo branco). Numpy puro, ~70 mil parametros.
"""
from __future__ import annotations

import io
import os

import numpy as np

VERSAO = 2
LADO = 64            # fovea: 64x64 pixels em resolucao total
GRADE = 8            # o mapa: 8x8 celulas
CANAIS = 32          # numeros por celula
K1, K2 = 16, 32      # canais das camadas intermediarias
PISO_CONTRASTE = 0.05
MEMORIA = 2048       # olhadas guardadas para reaprender (replay, como o hipocampo no sono)


def _lr(x):
    return np.where(x > 0, x, 0.1 * x)


def _dlr(x):
    return np.where(x > 0, 1.0, 0.1).astype(np.float32)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _im2col(x, k, s, p):
    B, H, W, C = x.shape
    xp = np.pad(x, ((0, 0), (p, p), (p, p), (0, 0))) if p else x
    Ho, Wo = (H + 2 * p - k) // s + 1, (W + 2 * p - k) // s + 1
    cols = np.empty((B, Ho, Wo, k, k, C), x.dtype)
    for i in range(k):
        for j in range(k):
            cols[:, :, :, i, j, :] = xp[:, i:i + s * Ho:s, j:j + s * Wo:s, :]
    return cols.reshape(B, Ho, Wo, k * k * C)


def _col2im(cols, forma, k, s, p):
    """adjunto do _im2col (soma o que se sobrepoe)."""
    B, H, W, C = forma
    Ho, Wo = cols.shape[1:3]
    cols = cols.reshape(B, Ho, Wo, k, k, C)
    xp = np.zeros((B, H + 2 * p, W + 2 * p, C), cols.dtype)
    for i in range(k):
        for j in range(k):
            xp[:, i:i + s * Ho:s, j:j + s * Wo:s, :] += cols[:, :, :, i, j, :]
    return xp[:, p:p + H, p:p + W, :] if p else xp


class Cortex:
    # conv(filtro 4, passo 2, borda 1): 64 -> 32 -> 16 -> 8 ; transposta: 8 -> 16 -> 32 -> 64
    FORMAS = {"E1": (48, K1), "e1": (K1,), "E2": (16 * K1, K2), "e2": (K2,), "E3": (16 * K2, CANAIS), "e3": (CANAIS,),
              "D3": (CANAIS, 16 * K2), "d3": (K2,), "D2": (K2, 16 * K1), "d2": (K1,), "D1": (K1, 48), "d1": (3,),
              # tempo: [mapa t-1, mapa t] -> vizinhanca 3x3 -> 64 -> 3x3 -> mapa t+1
              "T1": (9 * 2 * CANAIS, 64), "t1": (64,), "T2": (9 * 64, CANAIS), "t2": (CANAIS,)}

    def __init__(self, semente=0):
        rng = np.random.default_rng(semente)
        self.p = {}
        for k, f in self.FORMAS.items():
            if len(f) == 2:
                fan = f[0] if k[0] in "ET" else f[0] * 4          # transposta: cada saida recebe ~4 entradas
                self.p[k] = rng.normal(0, np.sqrt(2.0 / fan), f).astype(np.float32)
            else:
                self.p[k] = np.zeros(f, np.float32)
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
    def _enc(self, x):
        P = self.p
        B = len(x)
        c1 = _im2col(x, 4, 2, 1)
        a1 = c1 @ P["E1"] + P["e1"]
        h1 = _lr(a1)
        c2 = _im2col(h1, 4, 2, 1)
        a2 = c2 @ P["E2"] + P["e2"]
        h2 = _lr(a2)
        c3 = _im2col(h2, 4, 2, 1)
        z = c3 @ P["E3"] + P["e3"]                                  # o mapa [B,8,8,32]
        return z, (c1, a1, c2, a2, c3, B)

    def _dec(self, z):
        P = self.p
        B = len(z)
        a3 = _col2im(z @ P["D3"], (B, 16, 16, K2), 4, 2, 1) + P["d3"]
        g3 = _lr(a3)
        a2 = _col2im(g3 @ P["D2"], (B, 32, 32, K1), 4, 2, 1) + P["d2"]
        g2 = _lr(a2)
        o = _col2im(g2 @ P["D1"], (B, 64, 64, 3), 4, 2, 1) + P["d1"]       # saida linear (contraste normalizado)
        return o, (a3, g3, a2, g2, o)

    @staticmethod
    def _retina(imgs):
        """Como a retina: tira o brilho medio e normaliza o contraste da olhada (o que
        importa e a DIFERENCA; fundo branco nao ocupa a atencao do cortex)."""
        x = np.asarray(imgs, np.float32) / 255.0
        mu = x.mean((1, 2, 3), keepdims=True)
        sd = x.std((1, 2, 3), keepdims=True) + PISO_CONTRASTE
        return (x - mu) / sd, mu, sd

    def _fwd(self, imgs):
        x, mu, sd = self._retina(imgs)
        z, ce = self._enc(x)
        r, cd = self._dec(z)
        return x, r, z, ce, cd, mu, sd

    def _grad(self, x, r, z, ce, cd):
        P = self.p
        c1, a1, c2, a2, c3, B = ce
        a3, g3, d2a, g2, o = cd
        g = {}
        da1 = 2.0 * (r - x) / r.size                                 # [B,64,64,3]
        g["d1"] = da1.sum((0, 1, 2))
        dc = _im2col(da1, 4, 2, 1)                                   # [B,32,32,48]
        g["D1"] = g2.reshape(-1, K1).T @ dc.reshape(-1, 48)
        dg2 = dc @ P["D1"].T
        da2 = dg2 * _dlr(d2a)
        g["d2"] = da2.sum((0, 1, 2))
        dc = _im2col(da2, 4, 2, 1)                                   # [B,16,16,16*K1]
        g["D2"] = g3.reshape(-1, K2).T @ dc.reshape(-1, 16 * K1)
        dg3 = dc @ P["D2"].T
        da3 = dg3 * _dlr(a3)
        g["d3"] = da3.sum((0, 1, 2))
        dc = _im2col(da3, 4, 2, 1)                                   # [B,8,8,16*K2]
        g["D3"] = z.reshape(-1, CANAIS).T @ dc.reshape(-1, 16 * K2)
        dz = dc @ P["D3"].T                                          # [B,8,8,32]
        g["E3"] = c3.reshape(-1, 16 * K2).T @ dz.reshape(-1, CANAIS)
        g["e3"] = dz.sum((0, 1, 2))
        dh2 = _col2im(dz @ P["E3"].T, (B, 16, 16, K2), 4, 2, 1)
        da = dh2 * _dlr(a2)
        g["E2"] = c2.reshape(-1, 16 * K1).T @ da.reshape(-1, K2)
        g["e2"] = da.sum((0, 1, 2))
        dh1 = _col2im(da @ P["E2"].T, (B, 32, 32, K1), 4, 2, 1)
        da = dh1 * _dlr(a1)
        g["E1"] = c1.reshape(-1, 48).T @ da.reshape(-1, K1)
        g["e1"] = da.sum((0, 1, 2))
        return g

    def _adam(self, g, lr=2e-3, b1=0.9, b2=0.999):
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
        """olhadas uint8 [B,64,64,3] -> (mapas [B,8,8,32], erro [B], reconstrucao uint8 ou None)"""
        x, r, z, _ce, _cd, mu, sd = self._fwd(imgs)
        e = ((r - x) ** 2).mean((1, 2, 3))
        rec = (np.clip(r * sd + mu, 0, 1) * 255).astype(np.uint8) if reconstroi else None
        return z, e, rec

    def desenha(self, z, mu=0.5, sd=0.25):
        """mapa(s) [B,8,8,32] -> imagem(ns) uint8: o que o cortex 've' quando pensa nesse mapa."""
        o, _c = self._dec(np.asarray(z, np.float32))
        return (np.clip(o * sd + mu, 0, 1) * 255).astype(np.uint8)

    def aprende(self, imgs):
        """um passo de aprendizado com as olhadas novas + lembrancas (replay)."""
        imgs = np.asarray(imgs, np.uint8)
        for im in imgs:
            self.mem[self.n_mem % MEMORIA] = im
            self.n_mem += 1
        k = min(self.n_mem, MEMORIA)
        lemb = self.mem[self._rng.integers(0, k, size=min(8, k))]
        lote = np.concatenate([imgs, lemb])
        x, r, z, ce, cd, _mu, _sd = self._fwd(lote)
        perda = float(((r - x) ** 2).mean())
        self._adam(self._grad(x, r, z, ce, cd))
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

    # ── as celulas do mapa que tem algo (contraste) ─────────────────────────
    @staticmethod
    def celulas(img, mapa, limiar=0.04):
        """Celulas do mapa cujo pedaco da olhada tem contraste (fundo liso nao conta: a
        retina responde a diferenca, nao a cor parada). -> [n,32], [(i,j)]"""
        x = np.asarray(img, np.float32) / 255.0
        cel = x.reshape(GRADE, LADO // GRADE, GRADE, LADO // GRADE, 3).std((1, 3)).mean(-1)
        idx = np.argwhere(cel > limiar)
        return mapa[idx[:, 0], idx[:, 1]], [tuple(map(int, ij)) for ij in idx]

    # ── tempo: antecipar o proximo quadro (local, como movimento) ────────────
    def _tempo(self, ent):
        P = self.p
        c1 = _im2col(ent, 3, 1, 1)
        a = c1 @ P["T1"] + P["t1"]
        h = _lr(a)
        c2 = _im2col(h, 3, 1, 1)
        return c2 @ P["T2"] + P["t2"], (c1, a, h, c2)

    def preve(self, z_ant, z_atual, imagina=False, mu=0.5, sd=0.25):
        ent = np.concatenate([z_ant, z_atual], -1)[None].astype(np.float32)
        zp, _c = self._tempo(ent)
        img = self.desenha(zp, mu, sd)[0] if imagina else None
        return zp[0], img

    def aprende_tempo(self, Z):
        """Z: mapas de quadros seguidos [T,8,8,32]. Treina o preditor e devolve o erro
        medio da previsao ANTES de aprender (a surpresa do trecho)."""
        Z = np.asarray(Z, np.float32)
        if len(Z) < 3:
            return None
        P = self.p
        ent = np.concatenate([Z[:-2], Z[1:-1]], -1)
        alvo = Z[2:]
        pr, (c1, a, h, c2) = self._tempo(ent)
        perda = float(((pr - alvo) ** 2).mean())
        d = 2.0 * (pr - alvo) / pr.size
        g = {"T2": c2.reshape(-1, 9 * 64).T @ d.reshape(-1, CANAIS), "t2": d.sum((0, 1, 2))}
        dh = _col2im(d @ P["T2"].T, h.shape, 3, 1, 1)
        da = dh * _dlr(a)
        g["T1"] = c1.reshape(-1, 9 * 2 * CANAIS).T @ da.reshape(-1, 64)
        g["t1"] = da.sum((0, 1, 2))
        self._adam(g)
        self.passos_tempo += 1
        return perda

    def surpresa(self, Z):
        """erro medio da previsao do mapa do proximo quadro num trecho [T,8,8,32], sem aprender."""
        Z = np.asarray(Z, np.float32)
        if len(Z) < 3:
            return None
        pr, _c = self._tempo(np.concatenate([Z[:-2], Z[1:-1]], -1))
        return float(((pr - Z[2:]) ** 2).mean())

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
        e = np.concatenate([self.ve(np.stack(jan[i:i + 16]))[1] for i in range(0, len(jan), 16)])
        return [(float(e[i]), centros[i][0], centros[i][1]) for i in range(len(jan))]

    # ── guardar ──────────────────────────────────────────────────────────────
    def salva(self, arq):
        tmp = arq + ".tmp.npz"
        d = {"p_" + k: v for k, v in self.p.items()}
        d.update({"m_" + k: v for k, v in self.m.items()})
        d.update({"v_" + k: v for k, v in self.v.items()})
        d["mem"] = self.mem[:min(self.n_mem, MEMORIA)]
        d["contas"] = np.array([self.passo_adam, self.n_mem, self.olhadas, self.passos_tempo, VERSAO], np.int64)
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
                if len(d["contas"]) < 5 or int(d["contas"][4]) != VERSAO:
                    continue                                    # cortex de outra versao: nao serve
                for k in self.p:
                    self.p[k], self.m[k], self.v[k] = d["p_" + k], d["m_" + k], d["v_" + k]
                m = d["mem"]
                self.mem[:len(m)] = m
                self.passo_adam, self.n_mem, self.olhadas, self.passos_tempo = [int(v) for v in d["contas"][:4]]
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
