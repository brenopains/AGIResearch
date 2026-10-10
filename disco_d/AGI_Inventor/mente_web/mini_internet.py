"""
mente_web.mini_internet — uma INTERNET DE TESTE, sintetica e declarada (so para testes)
=======================================================================================
O conteiner de desenvolvimento nao alcanca a internet. Para testar de ponta a ponta o
caminho que a mente usa no navegador de verdade (telas, blocos de texto, imagens, videos
com som), este corpo imita o CorpoNavegador: paginas em portugues sobre cinco assuntos
(celula, fogueira, mar, floresta, cidade), cada uma com texto, figuras desenhadas por
procedimento e as vezes um video com som proprio do assunto (fogo que tremula e estala,
ondas que andam, folhas que balancam, carros que passam, celulas que se dividem).

NADA disto e o mundo real e NADA disto vai para a mente viva: e um banco de provas. O que
um teste aqui mostra e que o CODIGO faz o que diz (a cadeia palavra -> o que ela viu ->
o que ela imagina), nao que ela entende o mundo; isso so a vida no PC do dono mostra.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from corpo import Pagina

LARG, ALT = 1024, 700
ASSUNTOS = ("celula", "fogueira", "mar", "floresta", "cidade")
DOMINIOS = ("saber.org", "ciencia.net", "blog.com", "wiki.br", "natureza.info", "mundo.tv")
FRASES = {
    "celula": ["A celula tem membrana, citoplasma e nucleo.", "Cada celula vive dentro do organismo e se divide em duas.",
               "No microscopio a celula parece uma bolha verde com um nucleo escuro.", "O nucleo da celula guarda a informacao."],
    "fogueira": ["A fogueira queima a lenha e solta fumaca.", "O fogo da fogueira e laranja e amarelo e estala.",
                 "As chamas da fogueira sobem e tremulam no escuro.", "Perto da fogueira a luz e quente."],
    "mar": ["O mar tem ondas que batem na praia.", "A agua do mar e azul e salgada.",
            "As ondas do mar andam ate a areia e voltam.", "O barulho do mar e grave e constante."],
    "floresta": ["A floresta tem arvores altas e folhas verdes.", "Na floresta os passaros cantam nos galhos.",
                 "O vento balanca as folhas da floresta.", "Os troncos da floresta sao marrons e retos."],
    "cidade": ["A cidade tem predios, ruas e carros.", "As janelas da cidade acendem a noite.",
               "Os carros da cidade passam pela rua sem parar.", "Na cidade o barulho e um zumbido continuo."],
}
LIGA = ["Isto e muito comum.", "Muita gente estuda isso.", "Veja a figura ao lado.", "Leia mais abaixo.",
        "Este texto e sobre isso.", "Ha muito mais para saber."]


# ── figuras e videos por assunto (procedimentos simples, deterministicos por semente) ──
def _fundo(H, W, cor):
    return np.tile(np.array(cor, np.float32), (H, W, 1))


def _disco(img, cx, cy, r, cor, borda=None):
    H, W = img.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.hypot(xx - cx, yy - cy)
    img[d <= r] = cor
    if borda is not None:
        img[(d > r) & (d <= r + 2)] = borda


def figura(assunto, var, H=160, W=220, t=0.0):
    """um quadro do assunto no instante t (s); var: 8 numeros em [-1, 1] sorteados UMA vez por
    figura ou video (a variacao entre figuras do mesmo assunto)."""
    v = np.asarray(var, float)
    yy, xx = np.mgrid[0:H, 0:W]
    if assunto == "celula":
        img = _fundo(H, W, (225, 240, 225))
        for i in range(4):
            cx = W * (0.2 + 0.2 * i) + 10 * v[i] + 6 * math.sin(t + i)
            cy = H * (0.35 + 0.3 * ((i * 37) % 3) / 2) + 10 * v[i + 4] + 4 * math.cos(0.7 * t + i)
            r = H * 0.14 + 3 * (i % 2) + (4 * math.sin(0.5 * t) if i == 1 else 0)
            _disco(img, cx, cy, r, (120, 200, 120), borda=(40, 110, 40))
            _disco(img, cx + 3, cy - 2, r * 0.3, (60, 40, 110))
        return img.clip(0, 255).astype(np.uint8)
    if assunto == "fogueira":
        img = _fundo(H, W, (20, 15, 25))
        img[(yy > H - 30) & (yy < H - 18) & (np.abs(xx - W / 2 - 8 * v[0]) < 60)] = (110, 60, 25)
        for k in range(5):
            cx = W / 2 - 40 + 20 * k + 8 * v[0]
            h = H * 0.4 + 25 * (0.5 + 0.5 * math.sin(7.0 * t + 1.7 * k + 3 * v[k + 1]))
            base = H - 30
            dentro = (yy < base) & (yy > base - h) & (np.abs(xx - cx) < 14 * (yy - (base - h)) / h)
            frac = np.clip((base - yy) / h, 0, 1)
            cor = np.stack([255 - 10 * frac, 200 - 140 * frac, 40 * (1 - frac)], -1)
            img[dentro] = cor[dentro]
        return (img * (1.0 + 0.15 * math.sin(11.0 * t + v[6]))).clip(0, 255).astype(np.uint8)
    if assunto == "mar":
        img = _fundo(H, W, (150, 200, 240))
        mar = yy > H * (0.4 + 0.1 * v[0])
        img[mar] = (30, 80, 170)
        onda = np.sin((xx - 40 * t) / (12.0 + 3 * v[1]) + yy / 9.0) > 0.85
        img[mar & onda] = (230, 240, 255)
        return img.clip(0, 255).astype(np.uint8)
    if assunto == "floresta":
        img = _fundo(H, W, (190, 220, 200))
        for k in range(6):
            cx = W * (0.09 + 0.16 * k) + 6 * v[k]
            img[(np.abs(xx - cx) < 5) & (yy > H * 0.38)] = (100, 65, 35)
            _disco(img, cx + 6 * math.sin(1.5 * t + k), H * 0.31, H * 0.14, (30, 120, 40))
        return img.clip(0, 255).astype(np.uint8)
    if assunto == "cidade":
        img = _fundo(H, W, (60, 60, 80))
        for k in range(5):
            x0, alt = W * (0.05 + 0.19 * k) + 4 * v[k], H * (0.44 + 0.09 * (k % 3))
            pred = (xx > x0) & (xx < x0 + W * 0.15) & (yy > H - 40 - alt) & (yy < H - 40)
            img[pred] = (120, 120, 130)
            jan = pred & ((xx - int(x0)) % 10 < 4) & ((yy % 12) < 5)
            img[jan] = (240, 220, 90) if math.sin(2.0 * t + k + v[5]) > -0.3 else (70, 70, 80)
        img[yy > H - 40] = (40, 40, 45)
        for c in range(3):
            cx = (W * (0.15 + 0.32 * c) + 40 * t + 10 * v[6]) % (W + 30) - 15
            img[(np.abs(xx - cx) < 12) & (yy > H - 30) & (yy < H - 20)] = (200, 40, 40)
        return img.clip(0, 255).astype(np.uint8)
    raise ValueError(assunto)


def som(assunto, rng, t, n=3200, fs=16000):
    """0,2 s de som do assunto a partir do instante t."""
    tt = t + np.arange(n) / fs
    if assunto == "fogueira":
        s = 0.05 * rng.normal(size=n)
        for _ in range(int(rng.integers(3, 9))):
            i = int(rng.integers(n - 50))
            s[i:i + 50] += 0.6 * np.exp(-np.arange(50) / 8.0) * rng.choice([-1, 1])
        return s.astype(np.float32)
    if assunto == "mar":
        ruido = np.convolve(rng.normal(size=n + 64), np.ones(64) / 64, "valid")[:n]
        return (2.5 * ruido * (0.6 + 0.4 * np.sin(2 * np.pi * 0.2 * tt))).astype(np.float32)
    if assunto == "floresta":
        s = 0.01 * rng.normal(size=n)
        if rng.random() < 0.6:
            f = 2500 + 1500 * np.linspace(0, 1, n)
            s += 0.3 * np.sin(2 * np.pi * np.cumsum(f) / fs) * np.exp(-np.arange(n) / 900.0)
        return s.astype(np.float32)
    if assunto == "cidade":
        return (0.2 * np.sin(2 * np.pi * 120 * tt) + 0.1 * np.sin(2 * np.pi * 240 * tt)
                + 0.03 * rng.normal(size=n)).astype(np.float32)
    return (0.003 * rng.normal(size=n)).astype(np.float32)


# ── o corpo ───────────────────────────────────────────────────────────────────
class CorpoSintetico:
    """Imita o CorpoNavegador no que a mente usa: paginas com TELAS, olhar a tela, assistir
    um video da pagina (quadros RGB + som), voltar, pesquisar."""
    txt = "navegador"

    def __init__(self, semente=0, video_prob=0.5):
        self.rng = np.random.default_rng(semente)
        self.n_pedidos = 0
        self.hist = []
        self._pg = None
        self.video_prob = video_prob
        self.ultimo_clique = None
        self.assistiu_info = {}
        self.video_caixa = None
        self.cont = 0

    # paginas
    def _pagina(self, assunto=None, com_video=None):
        self.n_pedidos += 1
        self.cont += 1
        assunto = assunto or ASSUNTOS[int(self.rng.integers(len(ASSUNTOS)))]
        dom = DOMINIOS[int(self.rng.integers(len(DOMINIOS)))]
        url = f"sint://{dom}/{assunto}/{self.cont}"
        frs = list(FRASES[assunto])
        blocos_txt = []
        for _ in range(5):
            self.rng.shuffle(frs)
            blocos_txt.append(" ".join(frs[:3] + [LIGA[int(self.rng.integers(len(LIGA)))]]))
        com_video = (self.rng.random() < self.video_prob) if com_video is None else com_video
        blocos, imagens, videos, y = [], [], [], 80
        for i, t in enumerate(blocos_txt):
            blocos.append((40, y, 600, 90, t))
            if i < 2:
                imagens.append((700, y, 220, 160, f"{assunto} figura {i + 1}"))
            y += 260
        if com_video:
            videos.append((680, 360, 240, 180, "video"))
        pg = Pagina(url, dom, texto=" ".join(blocos_txt).encode("utf-8"),
                    links=[(f"mais sobre {a}".encode(), f"sint://{dom}/{a}/link") for a in ASSUNTOS],
                    midias=[("imagem", f"{url}#img{i}", f"{assunto} figura {i + 1}".encode()) for i in range(2)]
                    + ([("video", "el:0", f"video da {assunto}".encode())] if com_video else []))
        telas = []
        for i in range(int(math.ceil((y + 100) / ALT))):
            y0, y1 = i * ALT, (i + 1) * ALT
            bls = [b for b in blocos if y0 <= b[1] < y1]
            ims = [m for m in imagens if y0 <= m[1] + m[3] / 2 < y1]
            vds = [v for v in videos if y0 <= v[1] + v[3] / 2 < y1]
            if bls or ims or vds:
                telas.append({"y": y0, "texto": " ".join(b[4] for b in bls).encode("utf-8"),
                              "blocos": bls, "imagens": ims, "videos": vds})
        pg.telas = telas
        pg.assunto = assunto
        pg.semente = int(self.rng.integers(1 << 30))
        self._pg = pg
        return pg

    def aleatorio(self, porta):
        return self._pagina(com_video=True if porta == "video" else None)

    def abrir(self, u, empilha=True, origem=""):
        partes = str(u).split("/")
        a = partes[3] if len(partes) > 3 and partes[3] in ASSUNTOS else None
        if empilha and self._pg is not None:
            self.hist.append(self._pg)
        return self._pagina(a)

    def pesquisar(self, barra, texto):
        t = str(texto).lower()
        a = next((x for x in ASSUNTOS if x in t), None)
        return self._pagina(a)

    def usar_formulario(self, f, texto):
        return self.pesquisar("form", texto)

    def pode_voltar(self):
        return bool(self.hist)

    def voltar(self):
        pg = self.hist.pop() if self.hist else self._pagina()
        self._pg = pg
        return pg

    def ver(self, u):
        a = str(u).split("/")[3] if len(str(u).split("/")) > 3 else "celula"
        img = figura(a if a in ASSUNTOS else "celula", self.rng.uniform(-1, 1, 8))
        return np.asarray(Image.fromarray(img).convert("L").resize((32, 32)), np.float32) / 255.0

    # olhar
    def olha_tela(self, tela):
        pg = self._pg
        tela_img = Image.new("RGB", (LARG, ALT), (250, 250, 248))
        d = ImageDraw.Draw(tela_img)
        rng = np.random.default_rng([pg.semente, tela["y"]])
        for (x, y, w, h, t) in tela["blocos"]:
            yy = y - tela["y"]
            linhas = [t[i:i + 92] for i in range(0, len(t), 92)]
            for j, ln in enumerate(linhas[:5]):
                d.text((x, yy + 16 * j), ln, fill=(20, 20, 20))
        for (x, y, w, h, alt) in tela["imagens"]:
            tela_img.paste(Image.fromarray(figura(pg.assunto, rng.uniform(-1, 1, 8), h, w)), (int(x), int(y - tela["y"])))
        for (x, y, w, h, _k) in tela["videos"]:
            tela_img.paste(Image.fromarray(figura(pg.assunto, rng.uniform(-1, 1, 8), h, w)), (int(x), int(y - tela["y"])))
        return np.asarray(tela_img, np.uint8), tela["y"]

    # assistir
    def assistir(self, chave, segundos, passo=0.2):
        pg = self._pg
        if pg is None:
            return
        self.assistiu_info = {"clicou": True, "mao": False, "ouviu": 0, "trechos": 0, "sem_som_do_site": False}
        rng = np.random.default_rng([pg.semente, 99])
        var = rng.uniform(-1, 1, 8)
        dur = min(float(segundos), 20.0)
        t = 0.0
        while t < dur:
            q = figura(pg.assunto, var, 120, 160, t)
            a = som(pg.assunto, rng, t)
            self.assistiu_info["trechos"] += 1
            self.assistiu_info["ouviu"] += 1
            yield t, q, a
            t += passo

    def _mostra(self, *a, **k):
        pass
