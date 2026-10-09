"""
novaphys.render_gif — o GIF que explica sem legenda
=====================================================

DE ONDE VEM
-------------
Do usuario: "quero ver algo que de pra eu mostrar pra alguem de fora, um
humano, e ele entender so de bater o olho no gif".

A primeira tentativa (`render.py`) desenhava os canais do substrato como
curvas. Isso serve para depurar e nao comunica nada para quem chega de fora:
canal 0 contra canal 1 nao quer dizer coisa alguma sem o contexto todo.

Aqui o desenho e literal. Dois paineis lado a lado com o MESMO objeto:

    ESQUERDA   o que ela IMAGINOU que ia acontecer
    DIREITA    o que ACONTECEU de fato

O objeto esta inteiro quando o alvo e -1 e estilhacado quando e +1. Se os dois
paineis fazem a mesma coisa, ela acertou. Quando divergem, o painel da esquerda
ganha um anel vermelho. E so isso — nao precisa saber o que e um hipervetor.

HONESTIDADE DO DESENHO
------------------------
O estado subjacente e BINARIO (inteiro / estilhacado). O GIF interpola a
abertura dos cacos entre quadros so para o olho acompanhar; o rotulo em cada
painel mostra o estado binario verdadeiro, sem suavizacao. A barra do tempo no
rodape marca em vermelho cada instante de divergencia — ela e exata.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CEU = (13, 17, 23)
CARTAO = (22, 27, 34)
LINHA = (48, 54, 61)
TINTA = (230, 237, 243)
FRACO = (139, 148, 158)
OBJ = (255, 166, 87)
OBJ_ESC = (191, 110, 45)
ERRO = (248, 81, 73)
OK = (63, 185, 80)


def _fonte(tam: int):
    for nome in ("consola.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(nome, tam)
        except OSError:
            continue
    return ImageFont.load_default()


def _cacos(cx: float, cy: float, raio: float, abertura: float,
           n: int = 7) -> list:
    """O objeto como n cunhas. abertura=0 inteiro, 1 totalmente estilhacado."""
    peds = []
    for i in range(n):
        a0 = 2 * math.pi * i / n
        a1 = 2 * math.pi * (i + 1) / n
        am = (a0 + a1) / 2
        d = abertura * raio * 0.85
        gira = abertura * 0.5 * (1 if i % 2 else -1)
        dx, dy = math.cos(am) * d, math.sin(am) * d
        pts = []
        for a in (0.0, a0 - am, a1 - am):
            ang = am + a + gira
            r = 0.0 if a == 0.0 else raio
            pts.append((cx + dx + math.cos(ang) * r,
                        cy + dy + math.sin(ang) * r))
        peds.append(pts)
    return peds


def _painel(d: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int,
            titulo: str, abertura: float, estado: int, erro: bool,
            f_tit, f_pq) -> None:
    borda = ERRO if erro else LINHA
    d.rounded_rectangle([x, y, x + w, y + h], 14, fill=CARTAO,
                        outline=borda, width=3 if erro else 1)
    d.text((x + 18, y + 14), titulo, font=f_tit, fill=TINTA)
    cx, cy = x + w / 2, y + h / 2 + 18
    raio = min(w, h) * 0.26
    if abertura < 0.15:
        # INTEIRO: um poligono solido, SEM linhas internas. A primeira versao
        # desenhava as cunhas sempre, e o objeto "inteiro" parecia uma pizza
        # ja fatiada — o contraste com o estilhacado sumia.
        pts = [(cx + math.cos(2 * math.pi * i / 7) * raio,
                cy + math.sin(2 * math.pi * i / 7) * raio) for i in range(7)]
        d.polygon(pts, fill=OBJ, outline=(255, 210, 170))
    else:
        for pts in _cacos(cx, cy, raio, abertura):
            d.polygon(pts, fill=OBJ_ESC, outline=(0, 0, 0))
    rot = "ESTILHACADO" if estado > 0 else "INTEIRO"
    cor = ERRO if erro else FRACO
    cx2 = x + w / 2 - d.textlength(rot, font=f_pq) / 2
    d.text((cx2, y + h - 34), rot, font=f_pq, fill=cor)


def para_gif(caminho: str, y_real: Sequence[int], y_imaginado: Sequence[int],
             chamada: str, rodape: str, ms: int = 170, suave: float = 0.85,
             tam: Tuple[int, int] = (900, 520)) -> str:
    """
    `ms` e `suave` foram calibrados depois de MEDIR o ritmo do sinal: o estado
    tem run mediano de 3 instantes, entao a 55 ms cada estado durava ~165 ms —
    rapido demais para ler, e com suavizacao 0,45 a figura nunca alcancava o
    rotulo (um quadro dizia INTEIRO com o objeto desenhado estilhacado). A 170
    ms e com 0,85 a figura acompanha o rotulo.
    """
    W, H = tam
    yr = [1 if v > 0 else -1 for v in y_real]
    yi = [1 if v > 0 else -1 for v in y_imaginado]
    n = min(len(yr), len(yi))
    f_g, f_t, f_p = _fonte(26), _fonte(17), _fonte(14)

    pw, ph = (W - 3 * 26) // 2, 300
    px1, px2, py = 26, 26 * 2 + pw, 96
    ab_i = ab_r = 0.0
    quadros = []
    acertos = 0
    for t in range(n):
        alvo_i = 1.0 if yi[t] > 0 else 0.0
        alvo_r = 1.0 if yr[t] > 0 else 0.0
        ab_i += (alvo_i - ab_i) * suave
        ab_r += (alvo_r - ab_r) * suave
        erro = yi[t] != yr[t]
        acertos += (not erro)

        im = Image.new("RGB", (W, H), CEU)
        d = ImageDraw.Draw(im)
        d.text((26, 26), chamada, font=f_g, fill=TINTA)
        d.text((26, 62), rodape, font=f_p, fill=FRACO)

        _painel(d, px1, py, pw, ph, "O QUE ELA IMAGINOU", ab_i, yi[t],
                erro, f_t, f_p)
        _painel(d, px2, py, pw, ph, "O QUE ACONTECEU", ab_r, yr[t],
                False, f_t, f_p)

        by, bh = py + ph + 34, 18
        d.rounded_rectangle([26, by, W - 26, by + bh], 5, fill=CARTAO)
        larg = (W - 52) / n
        for i in range(t + 1):
            c = ERRO if yi[i] != yr[i] else OK
            d.rectangle([26 + i * larg, by, 26 + (i + 1) * larg, by + bh],
                        fill=c)
        pct = 100.0 * acertos / (t + 1)
        txt = f"acertou {pct:.0f}% dos instantes ate aqui"
        d.text((26, by + bh + 12), txt, font=f_p, fill=FRACO)
        quadros.append(im)

    quadros[0].save(caminho, save_all=True, append_images=quadros[1:],
                    duration=ms, loop=0, optimize=True)
    return caminho
