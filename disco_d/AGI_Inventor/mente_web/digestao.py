"""
mente_web.digestao — digerir sem semantica pronta
===================================================
Nasce pronto (estrutura, nao significado):
  texto   bytes. Um modelo preditivo de bytes (contextos de 1 a 5 bytes, tabelas
          com hash) mede a SURPRESA de cada byte. Onde a surpresa tem pico, ha
          fronteira (Harris 1955: a imprevisibilidade do proximo simbolo marca a
          fronteira de unidade). Os pedacos entre fronteiras que voltam em >= 2
          paginas viram ESTRUTURA sem nome. Ninguem diz o que e palavra.
  visao   quadro 32x32 em cinza -> 16 retalhos 8x8 normalizados -> prototipos
          aprendidos online (nasce um quando o retalho fica longe de todos, em
          comparacao com a historia de distancias).
  audio   janela de 200 ms -> 24 bandas log de 80 Hz a 8 kHz (coclea fixa) ->
          prototipos aprendidos online.
  juntos  coincidencia no MESMO instante (video com audio no mesmo relogio):
          prototipo visual x prototipo sonoro; e estrutura de texto da pagina x
          prototipo visual/sonoro da midia que estava nela.

O dado bruto e transitorio: nada de bytes, quadros ou som fica guardado — so
contagens, prototipos, estruturas e resumos de episodio.
"""
from __future__ import annotations

import functools
import hashlib
import math

import numpy as np
from numba import njit

K = 5
M = 1 << 16
BETA = 2.0
D_CTX = 512


@functools.lru_cache(maxsize=12000)
def _idx(chave: bytes, d=D_CTX):
    sem = int.from_bytes(hashlib.blake2b(chave, digest_size=8).digest(), "little")
    v = np.where(np.random.default_rng(sem).random(d) < 0.5, -1.0, 1.0).astype(np.float32)
    v.flags.writeable = False
    return v


@njit(cache=True)
def _processa(b, cont, tot, beta, atualiza, sur):
    n = b.shape[0]
    for i in range(n):
        p = 1.0 / 256.0
        h = 0
        rows = np.empty(K, np.int64)
        nk = 0
        for k in range(1, K + 1):
            if i - k < 0:
                break
            h = (h * 1000003 + b[i - k] + 1) & 0xFFFFFFFF
            row = ((h ^ (k * 2654435761)) & 0xFFFFFFFF) % M
            rows[k - 1] = row
            nk = k
            N = tot[k - 1, row]
            if N > 0:
                p = (cont[k - 1, row, b[i]] + beta * p) / (N + beta)
        sur[i] = -math.log2(p)
        if atualiza:
            for k in range(nk):
                row = rows[k]
                if cont[k, row, b[i]] >= 65000:
                    s = 0
                    for j in range(256):
                        cont[k, row, j] = cont[k, row, j] // 2
                        s += cont[k, row, j]
                    tot[k, row] = s
                cont[k, row, b[i]] += 1
                tot[k, row] += 1


class Texto:
    def __init__(self):
        self.cont = np.zeros((K, M, 256), np.uint16)
        self.tot = np.zeros((K, M), np.uint32)
        self.bytes_lidos = 0

    def surpresa(self, dados: bytes, atualiza=True):
        b = np.frombuffer(dados, np.uint8)
        sur = np.empty(len(b), np.float64)
        _processa(b, self.cont, self.tot, BETA, atualiza, sur)
        if atualiza:
            self.bytes_lidos += len(b)
        return sur


def segmenta(dados: bytes, sur: np.ndarray, maxlen=24):
    """Pedacos entre picos de surpresa acima da mediana do proprio trecho."""
    n = len(dados)
    if n < 3:
        return []
    med = float(np.median(sur))
    pico = np.zeros(n, bool)
    pico[1:-1] = (sur[1:-1] > sur[:-2]) & (sur[1:-1] >= sur[2:]) & (sur[1:-1] > med)
    cortes = [0] + [i for i in np.nonzero(pico)[0]] + [n]
    out = []
    for a, b in zip(cortes[:-1], cortes[1:]):
        while b - a > maxlen:
            out.append((a, a + maxlen))
            a += maxlen
        if b > a:
            out.append((a, b))
    return out


class Prototipos:
    """Prototipos aprendidos online; nasce um quando a distancia ao mais proximo
    passa da media + 2 desvios das distancias que o proprio sistema ja viu."""

    def __init__(self, dim, teto_memoria=1024, nome="v"):
        self.dim, self.teto, self.nome = dim, teto_memoria, nome
        self.P = np.zeros((0, dim), np.float32)
        self.n = np.zeros(0, np.int64)
        self.ult = np.zeros(0, np.int64)
        self.dist_m, self.dist_v, self.vistos = 0.0, 1.0, 0
        self.nascidos = 0

    def codifica(self, X, t, aprende=True):
        X = np.asarray(X, np.float32)
        out = []
        for x in X:
            if len(self.P) == 0:
                self._nasce(x, t)
                out.append(0)
                continue
            d = np.sqrt(((self.P - x) ** 2).sum(1))
            j = int(np.argmin(d))
            dj = float(d[j])
            lim = self.dist_m + 2.0 * math.sqrt(max(self.dist_v, 1e-9))
            if aprende:
                self.vistos += 1
                a = 1.0 / min(self.vistos, 500)
                dm = dj - self.dist_m
                self.dist_m += a * dm
                self.dist_v = (1 - a) * self.dist_v + a * dm * dm
            # um padrao so aprende com o que e PARECIDO com ele (dentro do raio tipico);
            # o diferente vira padrao novo. Antes, o 1o padrao aprendia a media de TUDO,
            # virava um borrao, e todo retalho normalizado ficava a mesma distancia dele:
            # nada parecia novo e a visao ficava com 1 padrao so (gemeas, 08/10).
            aquecendo = self.vistos <= 20
            if aprende and ((aquecendo and dj > self.dist_m) or (not aquecendo and dj > lim)):
                j = self._nasce(x, t)
            elif aprende and (aquecendo or dj <= lim):
                self.n[j] += 1
                self.P[j] += (x - self.P[j]) / min(self.n[j], 200)
                self.ult[j] = t
            out.append(j)
        return out

    def _nasce(self, x, t):
        if len(self.P) >= self.teto:
            # memoria finita: o menos usado e mais antigo cede o lugar
            j = int(np.argmin(self.n * 1.0 + self.ult * 1e-6))
            self.P[j], self.n[j], self.ult[j] = x, 1, t
        else:
            self.P = np.vstack([self.P, x[None]])
            self.n = np.append(self.n, 1)
            self.ult = np.append(self.ult, t)
            j = len(self.P) - 1
        self.nascidos += 1
        return j


def retalhos(quadro):
    """16 retalhos 8x8 normalizados de um quadro 32x32."""
    R = quadro.reshape(4, 8, 4, 8).transpose(0, 2, 1, 3).reshape(16, 64)
    R = R - R.mean(1, keepdims=True)
    s = R.std(1, keepdims=True)
    return np.where(s > 1e-3, R / np.maximum(s, 1e-3), 0.0)


_BANDAS = np.geomspace(80, 8000, 25)


def coclea(audio, sr=16000):
    """24 bandas log de energia de um trecho de audio (estrutura fixa)."""
    if audio is None or len(audio) < 256:
        return None
    n = 512
    quadros = [audio[i:i + n] for i in range(0, len(audio) - n + 1, n // 2)] or [np.pad(audio, (0, n - len(audio)))]
    E = np.zeros(24)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    jan = np.hanning(n)
    for q in quadros:
        S = np.abs(np.fft.rfft(q * jan)) ** 2
        for k in range(24):
            m = (freqs >= _BANDAS[k]) & (freqs < _BANDAS[k + 1])
            E[k] += S[m].sum()
    E = np.log10(E / len(quadros) + 1e-10)
    if E.max() < -8:
        return None            # silencio
    return ((E - E.mean()) / (E.std() + 1e-6)).astype(np.float32)


def descritor_global(quadro):
    """v1: a cena inteira (retina + V1 fixos, sem semantica; cf. GIST, Oliva e
    Torralba 2001): luminancia numa grade 4x4, energia de gradiente em 4
    orientacoes numa grade 4x4 e histograma de luminancia em 8 faixas."""
    q = np.asarray(quadro, np.float32)
    lum = q.reshape(4, 8, 4, 8).mean((1, 3)).ravel()
    gy, gx = np.gradient(q)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.arctan2(gy, gx) % np.pi) / np.pi * 4.0
    ori = np.minimum(ang.astype(int), 3)
    E = np.zeros((4, 4, 4), np.float32)
    for o in range(4):
        E[o] = (mag * (ori == o)).reshape(4, 8, 4, 8).sum((1, 3))
    E = np.log1p(E.ravel() * 4.0)
    h = np.histogram(q, bins=8, range=(0, 1))[0].astype(np.float32) / q.size
    d = np.concatenate([lum * 2.0, E, h * 4.0]).astype(np.float32)
    return (d - d.mean()) / (d.std() + 1e-6)


class Vocabulario:
    """Vocabulario visual ESTAVEL (k-medias online, K fixo). Os Prototipos acima crescem
    ate um teto e depois RECICLAM a palavra menos usada: no espaco do cortex v2 o teto de
    4.096 estourou em 30 min e cada palavra reciclada levava junto o que estava ligado a
    ela (08/10). Aqui: enquanto ha vaga, so nasce palavra para o que esta longe de todas
    (mais que a media + 2 desvios das distancias ja vistas: as sementes se espalham
    no tempo e pegam experiencias variadas);
    cheio, cada palavra so se REFINA com o que cai nela (passo 1/n, minimo 1/500, para
    acompanhar o cortex que ainda aprende). Nenhuma morre."""

    def __init__(self, dim, K=1024, nome="r"):
        self.dim, self.K, self.nome = dim, K, nome
        self.P = np.zeros((0, dim), np.float32)
        self.n = np.zeros(0, np.int64)
        self.ult = np.zeros(0, np.int64)
        self.dist_m, self.dist_v, self.vistos = 0.0, 1.0, 0
        self.nascidos = 0

    def codifica(self, X, t, aprende=True):
        X = np.asarray(X, np.float32)
        out = []
        for x in X:
            if len(self.P) == 0:
                if aprende:
                    self._nasce(x, t)
                    out.append(0)
                continue
            d = np.sqrt(((self.P - x) ** 2).sum(1))
            j = int(np.argmin(d))
            dj = float(d[j])
            if aprende:
                self.vistos += 1
                a = 1.0 / min(self.vistos, 2000)
                dm = dj - self.dist_m
                self.dist_m += a * dm
                self.dist_v = (1 - a) * self.dist_v + a * dm * dm
                if len(self.P) < self.K and dj > self.dist_m + 2.0 * math.sqrt(max(self.dist_v, 1e-12)):
                    out.append(self._nasce(x, t))
                    continue
                self.n[j] += 1
                self.P[j] += (x - self.P[j]) / min(self.n[j], 500)
                self.ult[j] = t
            out.append(j)
        return out

    def _nasce(self, x, t):
        self.P = np.vstack([self.P, x[None]])
        self.n = np.append(self.n, 1)
        self.ult = np.append(self.ult, t)
        self.nascidos += 1
        return len(self.P) - 1
