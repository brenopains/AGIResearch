"""
mente_web.mundos — os MODELOS DE MUNDO dela, decodificados para gente ver (so LEITURA)
======================================================================================
Pedido do usuario (10/10): "clicar no modelo de mundo aprendido e ver como ela ve esse mundo,
sem simular dados falsos". Este modulo nao aprende e nao decide nada: le o que ela guardou
e passa pelos decodificadores DELA. Cada imagem, posicao e espectro publicado aqui vem de:
  - o CORTEX dela (Cortex.desenha decodifica um mapa interno em pixels) e as 2.048 olhadas
    reais que ela guarda (cortex.mem);
  - as LIGACOES que ela confirmou contra controle (texto <-> palavra visual, texto <-> som,
    texto <-> verbo visual), com o teste de acaso dela;
  - o LABORATORIO: cenas (fundo + recorte real de cada coisa + a camada da FRENTE que ela
    deduziu pela oclusao), leis e conceitos dela, a mesma rolagem que ela usa para prever.

Dois tipos de modelo de mundo:
  MUNDO FISICO   uma teoria do laboratorio = os videos que uma lei dela explica; camadas
                 (fundo < coisas < frente), as coisas com a aparencia real, o movimento pela
                 lei dela a partir dos comecos REAIS (comparavel ao que aconteceu) e de
                 comecos novos (imaginacao), e as correntes da lei (como as coisas andam ali).
  CONCEITO       um pedaco de texto que ela ligou a coisas vistas/ouvidas: o ARQUETIPO (a
                 media, no espaco interno dela, das olhadas reais em que essas coisas
                 apareceram, decodificada pelo cortex dela), as lembrancas (como ela ve cada
                 uma), os recortes reais das palavras visuais, o espectro dos sons, e os verbos
                 visuais (transformacoes) que ela viu acontecer junto, aplicados ao arquetipo.

Escolhas de EXIBICAO (declaradas no painel, nao sao dela): brilho/contraste ao decodificar
(media das olhadas usadas), a distancia entre as camadas no holodeck, cores das linhas.
"""
from __future__ import annotations

import hashlib
import io
import math
import os
import time

import numpy as np

import cortex as CX

MAX_CONCEITOS = 16
MAX_FISICOS = 6
LEMBRANCAS = 6


def _png(img_rgba):
    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(img_rgba).save(b, "PNG", optimize=True)
    return b.getvalue()


def _grava(arq, dados):
    tmp = arq + ".tmp"
    with open(tmp, "wb") as f:
        f.write(dados)
    os.replace(tmp, arq)


def chave_de(s):
    return hashlib.md5(s if isinstance(s, bytes) else str(s).encode("utf-8")).hexdigest()[:10]


def txt(b):
    return b.decode("utf-8", "replace") if isinstance(b, (bytes, bytearray)) else str(b)


# ── memoria visual: onde cada palavra visual aparece nas olhadas REAIS guardadas ──────
def indice_memoria(cx, olho, lote=128):
    """-> {palavra visual: [(indice da olhada, i, j), ...]} sobre as olhadas guardadas
    (cx.mem), sem aprender nada (codifica(aprende=False)). Da mais recente para a mais
    antiga."""
    k = min(cx.n_mem, CX.MEMORIA)
    if k == 0 or not len(olho.P):
        return {}
    ordem = [(cx.n_mem - 1 - j) % CX.MEMORIA for j in range(k)]
    idx = {}
    for a in range(0, k, lote):
        g = ordem[a:a + lote]
        z = cx.ve(cx.mem[g])[0]
        for gi, zi in zip(g, z):
            V, ij = CX.Cortex.celulas(cx.mem[gi], zi)
            if not len(V):
                continue
            for w, (i, j) in zip(olho.codifica(V, 0, aprende=False), ij):
                idx.setdefault(int(w), []).append((int(gi), int(i), int(j)))
    return idx


def recorte(img, i, j, lado=24):
    """o pedaco REAL da olhada onde a celula (i, j) do mapa esta (~ o campo de visao dela)."""
    c = CX.LADO // CX.GRADE
    y0 = int(np.clip(i * c + c // 2 - lado // 2, 0, CX.LADO - lado))
    x0 = int(np.clip(j * c + c // 2 - lado // 2, 0, CX.LADO - lado))
    return img[y0:y0 + lado, x0:x0 + lado]


def arquetipo(cx, imgs):
    """a media, no espaco interno dela, de varias olhadas reais, decodificada pelo cortex dela.
    brilho/contraste: a media das olhadas usadas (escolha de exibicao declarada)."""
    imgs = np.asarray(imgs, np.uint8)
    z = cx.ve(imgs)[0]
    x = imgs.astype(np.float32) / 255.0
    mu = float(x.mean())
    sd = float(x.std((1, 2, 3)).mean()) + CX.PISO_CONTRASTE
    return cx.desenha(z.mean(0, keepdims=True), mu, sd)[0], z.mean(0)


def palavra_decodificada(cx, olho, j, fundo):
    """a palavra visual j sozinha, no centro de um mapa de fundo NEUTRO (o mapa de uma olhada
    cinza lisa), decodificada pelo cortex dela."""
    M = fundo.copy()
    c0 = CX.GRADE // 2 - 1
    M[c0:c0 + 2, c0:c0 + 2] = olho.P[j]
    return cx.desenha(M[None].astype(np.float32))[0]


def _mapa_neutro(cx):
    cinza = np.full((1, CX.LADO, CX.LADO, 3), 128, np.uint8)
    return cx.ve(cinza)[0][0]


# ── CONCEITOS: pedacos de texto ligados ao que ela viu, ouviu e viu acontecer ─────────
def conceitos(m, idx, max_c=MAX_CONCEITOS):
    """os pedacos com mais ligacoes confirmadas (validas primeiro)."""
    por = {}
    for (s, pr) in m.gr.conf:
        d = por.setdefault(s, {"r": [], "a": [], "verbo": []})
        if pr[0] in ("r", "a"):
            d[pr[0]].append(int(pr[1]))
    gv = getattr(m, "gv", None)
    if gv is not None:
        for (s, pr) in gv.conf:
            if s in por or len(por) < 4 * max_c:
                por.setdefault(s, {"r": [], "a": [], "verbo": []})["verbo"].append(int(pr[1]))
    validas = m._validas() if hasattr(m, "_validas") else set()

    def peso(item):
        s, d = item
        nv = sum(("r", j) in validas for j in d["r"])
        return (nv, len(d["r"]) + len(d["a"]) + len(d["verbo"]), len(txt(s)))
    escolhidos = sorted(por.items(), key=peso, reverse=True)[:max_c]
    out = []
    for s, d in escolhidos:
        imgs = []
        for j in d["r"]:
            for (g, i, jj) in idx.get(j, [])[:LEMBRANCAS]:
                if g not in [x[0] for x in imgs]:
                    imgs.append((g, i, jj, j))
        E = m.E.get(s, {}) if hasattr(m, "E") else {}
        out.append({"s": s, "pedaco": txt(s), "chave": chave_de(s), "nivel": E.get("nivel"),
                    "dominios": sorted(E.get("doms", []))[:6] if isinstance(E.get("doms"), (set, list)) else [],
                    "visuais": [{"id": j, "valida": ("r", j) in validas, "vezes": int(m.olho.n[j]) if j < len(m.olho.n) else 0,
                                 "na_memoria": len(idx.get(j, []))} for j in d["r"]],
                    "sons": [{"id": k, "bandas": [round(float(x), 3) for x in m.aud.P[k]]} for k in d["a"] if k < len(m.aud.P)],
                    "verbos": d["verbo"], "lembrancas": imgs[:LEMBRANCAS],
                    "fontes": sorted({str(e).split("#")[0][:90] for pr in [("r", j) for j in d["r"]] + [("a", k) for k in d["a"]]
                                      for e in (m.gr.c.get((s, pr), [0, 0, 0, set()])[3] or [])})[:5]})
    return out


def folha_conceito(m, c, fundo_neutro):
    """uma folha PNG (grade de 7 x 4 quadros de 64 px):
      linhas 0-1, colunas 0-1  o ARQUETIPO (grande)
      linha 0, colunas 2-6     como ela VE cada lembranca (reconstrucao do cortex dela)
      linha 1, colunas 2-6     a olhada REAL guardada (o que chegou ao olho dela)
      linha 2, colunas 0-1     recortes reais (24 px) onde as palavras visuais ligadas aparecem
      linha 2, colunas 2-6     o verbo visual ligado, aplicado ao arquetipo 0..4 vezes
      linha 3, colunas 0-3     as palavras visuais ligadas, sozinhas, decodificadas
    -> (png, rects)"""
    from PIL import Image
    cx = m.cortex
    L = CX.LADO
    tela = np.zeros((4 * L, 7 * L, 4), np.uint8)
    rects = {"arquetipo": None, "como_ve": [], "real": [], "recortes": [], "verbo": [], "palavras": []}

    def poe(img, x0, y0):
        h, w = img.shape[:2]
        tela[y0:y0 + h, x0:x0 + w, :3] = img
        tela[y0:y0 + h, x0:x0 + w, 3] = 255
        return [x0, y0, w, h]
    lemb = [cx.mem[g] for (g, _i, _j, _w) in c["lembrancas"]]
    z_arq = None
    if lemb:
        arq, z_arq = arquetipo(cx, lemb)
        rects["arquetipo"] = poe(np.asarray(Image.fromarray(arq).resize((2 * L, 2 * L), Image.NEAREST)), 0, 0)
        rec = cx.ve(np.stack(lemb), reconstroi=True)[2]
        for k in range(min(5, len(lemb))):
            rects["como_ve"].append(poe(rec[k], (2 + k) * L, 0))
            rects["real"].append(poe(lemb[k], (2 + k) * L, L))
        for k, (g, i, j, _w) in enumerate(c["lembrancas"][:6]):
            rects["recortes"].append(poe(recorte(cx.mem[g], i, j), 4 + (k % 3) * 42, 2 * L + 6 + (k // 3) * 30))
    vb = getattr(m, "verbos", None)
    if z_arq is not None and vb is not None and c["verbos"] and c["verbos"][0] < len(m.vocab_verbos.P):
        x = np.asarray(lemb, np.float32) / 255.0
        mu, sd = float(x.mean()), float(x.std((1, 2, 3)).mean()) + CX.PISO_CONTRASTE
        for k in range(5):
            im = cx.desenha(vb.aplica(z_arq, m.vocab_verbos.P[c["verbos"][0]], k)[None], mu, sd)[0]
            rects["verbo"].append(poe(im, (2 + k) * L, 2 * L))
    for k, v in enumerate(c["visuais"][:4]):
        rects["palavras"].append(poe(palavra_decodificada(cx, m.olho, v["id"], fundo_neutro), k * L, 3 * L))
    return _png(tela), rects


# ── MUNDOS FISICOS: as teorias do laboratorio, em camadas ─────────────────────────────
def regra_em_palavras(lei, NOMES):
    """o que a lei dela usa, em portugues de gente (pelo tipo de termo, nao inventa nada)."""
    if lei is None or not lei.get("termos"):
        return ["nada acelera: as coisas seguem reto (inercia)"]
    peso = {}
    for j, w in zip(lei["termos"], lei["W"]):
        peso[NOMES[j]] = float(np.abs(w).sum())
    tot = sum(peso.values()) + 1e-12
    frases = []
    pos = sum(v for k, v in peso.items() if k in ("x", "y", "x²", "x·y", "y²"))
    vel = sum(v for k, v in peso.items() if k in ("vx", "vy", "vx²", "vx·vy", "vy²"))
    mix = sum(v for k, v in peso.items() if k in ("x·vx", "x·vy", "y·vx", "y·vy"))
    conc = sum(v for k, v in peso.items() if k.startswith("q"))
    if conc / tot > 0.05:
        frases.append(f"depende da grandeza que ela INVENTOU ({lei.get('conceito')}) — {round(100 * conc / tot)}% da lei")
    if vel / tot > 0.05:
        frases.append(f"depende de para onde e quao rapido a coisa anda — {round(100 * vel / tot)}%")
    if pos / tot > 0.05:
        frases.append(f"depende de ONDE a coisa esta (um puxao ou empurrao pelo lugar) — {round(100 * pos / tot)}%")
    if mix / tot > 0.05:
        frases.append(f"depende do lugar e do movimento juntos — {round(100 * mix / tot)}%")
    return frases or ["termos pequenos demais para resumir"]


GIRO_MIN = 0.8        # 'gira' so com o mesmo sentido em >= 80% dos momentos vistos
CENTRO_R2 = 0.5       # 'puxa para um centro' so se a mola linear explica >= 50% da aceleracao
VEL_R = 0.5           # 'quanto mais rapido, mais gira' so com correlacao >= 0,5


def frases_da_lei(lab, T, Zv):
    """REGRAS EM FRASES, cada uma medida na lei DELA sobre os estados que ela VIU (nada da
    verdade da arena): -> [{chave, frase, numero}]. Tela: y para baixo (horario = na tela)."""
    import laboratorio as LB
    if T["lei"] is None or not len(Zv):
        return []
    Z = Zv[np.linalg.norm(Zv[:, 2:], axis=1) > 1e-3]
    if len(Z) < 10:
        return []
    a = LB.acel_lei(T["lei"], Z, lab.conceitos)
    v = Z[:, 2:]
    s2 = (v ** 2).sum(1)
    cruz = v[:, 0] * a[:, 1] - v[:, 1] * a[:, 0]
    w = cruz / s2                                   # rad/s: quanto a direcao gira
    out = []
    pos = float((cruz > 0).mean())
    sentido = max(pos, 1 - pos)
    if sentido >= GIRO_MIN and float(np.median(np.abs(w))) > 0.05:
        voltas = float(np.median(np.abs(w))) / (2 * math.pi)
        out.append({"chave": "gira", "sentido": "horario" if pos >= 0.5 else "anti-horario",
                    "frase": f"as coisas GIRAM, sempre no sentido {'horário' if pos >= 0.5 else 'anti-horário'} na tela "
                             f"({round(100 * sentido)}% dos momentos; {voltas:.2f} volta por segundo na rapidez típica)",
                    "numero": round(sentido, 3), "voltas_s": round(voltas, 3)})
        r = float(np.corrcoef(np.abs(w), s2)[0, 1]) if np.std(s2) > 0 and np.std(w) > 0 else 0.0
        if r >= VEL_R:
            out.append({"chave": "gira_com_rapidez", "frase": f"quanto mais rápido a coisa anda, mais ela gira (r = {r:.2f})",
                        "numero": round(r, 3)})
    # puxao para um centro: a ~ -k (x - c), por minimos quadrados sobre os estados vistos
    X = np.column_stack([Z[:, :2], np.ones(len(Z))])
    coef, *_ = np.linalg.lstsq(X, a, rcond=None)
    pred = X @ coef
    r2 = 1 - float(((a - pred) ** 2).sum()) / max(float(((a - a.mean(0)) ** 2).sum()), 1e-12)
    M = coef[:2].T                                  # a = M x + b
    k = -float(np.trace(M)) / 2
    if r2 >= CENTRO_R2 and k > 0:
        try:
            c = np.linalg.solve(M, -coef[2])
        except np.linalg.LinAlgError:
            c = None
        if c is not None and np.all(np.abs(c) < 3):
            fora = float((((a * (Z[:, :2] - c)).sum(1)) < 0).mean())
            out.append({"chave": "centro", "centro": [round(float(c[0]), 3), round(float(c[1]), 3)],
                        "frase": f"as coisas são PUXADAS para um centro em ({c[0]:.2f}; {c[1]:.2f}) — ela o deduziu, ele não aparece "
                                 f"no vídeo (a mola explica {round(100 * r2)}% da aceleração; puxa para dentro em {round(100 * fora)}% dos momentos)",
                        "numero": round(r2, 3)})
    conc = T["lei"].get("conceito")
    if conc is not None and conc in lab.conceitos:
        out.append({"chave": "conceito", "frase": f"cada coisa carrega uma grandeza que NÃO muda enquanto ela se move ({conc}, "
                                                  f"que ela inventou): {LB.formula_conceito(lab.conceitos[conc])}",
                    "numero": round(float(lab.conceitos[conc]["lam"]), 4)})
    if not out:
        out.append({"chave": "outra", "frase": "nenhuma das frases simples (girar, puxar para um centro) descreve esta lei; ver a fórmula",
                    "numero": None})
    return out


def nomes_do_mundo(m, tid, k=4):
    """os NOMES que ela mesma deu a este mundo: pedacos do texto das paginas dos videos ligados a
    teoria (gt, com controle). Confirmados = 'rotulo dela'; so contados = 'so co-ocorrencia'."""
    gt = getattr(m, "gt", None)
    if gt is None:
        return []
    out = []
    for (s, pr), e in gt.c.items():
        if pr == ("t", int(tid)) and e[0] >= 2:
            out.append({"pedaco": txt(s), "vezes": int(e[0]), "confirmado": (s, pr) in gt.conf,
                        "videos": len(e[3])})
    out.sort(key=lambda x: (-x["confirmado"], -x["vezes"]))
    return out[:k]


def fisicos(lab, max_t=MAX_FISICOS, m_ref=None):
    """as teorias com cena guardada (aparencia real), as que explicam mais videos primeiro."""
    import laboratorio as LB
    cenas = getattr(lab, "cenas", {})
    out = []
    for T in sorted(lab.teorias.values(), key=lambda T: -len(T["membros"])):
        com = [e for e in T["membros"] if e in cenas]
        if not com:
            continue
        e = com[-1]
        c = cenas[e]
        ep = lab.eps[e]
        H, W = c["forma"][:2]
        L = c["L"]
        # as coisas reais deste video: o comeco de cada trilha e o caminho real
        reais, imag, retos, epocas = [], [], [], []
        eps_T = T.get("epocas", [])
        for tr in ep["trilhas"][:6]:
            Zc, okc = LB.estados_causais(tr)
            ii = np.nonzero(okc)[0]
            if not len(ii):
                continue
            i0 = int(ii[0])
            t = tr["t"].astype(float)[i0:] - float(tr["t"][i0])
            sel = np.linspace(0, len(t) - 1, min(len(t), 40)).astype(int)
            reais.append({"t": t[sel].round(2).tolist(), "p": tr["p"][i0:][sel].astype(float).round(4).tolist()})
            imag.append(lab._caminho(T["lei"], Zc[i0], t[sel]).round(4).tolist())
            retos.append(lab._caminho(None, Zc[i0], t[sel]).round(4).tolist())
            epocas.append([lab.caminho_da_epoca(x, Zc[i0], t[sel]).round(4).tolist() for x in eps_T])
        # quanto cada EPOCA da lei erra estes caminhos reais (mediana, px em 128): o refinamento
        erros_epoca = []
        for k, x in enumerate(eps_T):
            ds = [np.linalg.norm(np.asarray(epocas[i][k]) - np.asarray(reais[i]["p"]), axis=1).mean() * LB.PX
                  for i in range(len(reais))]
            erros_epoca.append({"episodio": x["episodio"], "videos": x["videos"], "lei": LB.formula_lei(x["lei"]),
                                "erro_px": round(float(np.median(ds)), 2) if ds else None})
        rng = np.random.default_rng([len(lab.ordem), T["id"]])
        Z0 = lab.estados_imaginaveis(T, 3, rng)
        novos = None
        if Z0 is not None:
            ts = np.arange(31) / 5.0
            P = lab._posicoes(T["lei"], Z0, tuple(ts[1:]))
            novos = {"estados": Z0.round(4).tolist(), "t": ts.tolist(),
                     "caminhos": np.concatenate([Z0[:, None, :2], P], 1).round(4).tolist()}
        # correntes da lei: de uma grade de lugares, com rapidez e direcoes tipicas do que ela viu;
        # 'visto' = quantos estados reais dela cairam perto do comeco (sem dados = extrapolacao)
        vistos = [lab._der(x)["Z"] for x in T["membros"][-30:]]
        vistos = [z for z in vistos if len(z)]
        correntes, visto, frases = [], [], []
        if vistos:
            Zv = np.concatenate(vistos)
            frases = frases_da_lei(lab, T, Zv)
            rap = float(np.median(np.linalg.norm(Zv[:, 2:], axis=1)))
            g = np.linspace(-0.6, 0.6, 4)
            starts = np.array([[x, y, rap * math.cos(a), rap * math.sin(a)] for x in g for y in g
                               for a in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2)])[::3]
            Pc = lab._posicoes(T["lei"], starts, tuple(np.arange(1, 9) * 0.5))
            correntes = np.concatenate([starts[:, None, :2], Pc], 1).round(3).tolist()
            visto = [int((np.linalg.norm(Zv[:, :2] - s0[:2], axis=1) < 0.2).sum()) for s0 in starts]
        fontes = sorted({lab.eps[x]["fonte"] for x in T["membros"]})
        erro = [lab.eps[x].get("erro_1a") for x in T["membros"] if lab.eps[x].get("erro_1a") is not None]
        iner = [lab.eps[x].get("inercia") for x in T["membros"] if lab.eps[x].get("inercia") is not None]
        conc = T["lei"].get("conceito") if T["lei"] else None
        out.append({"id": T["id"], "episodios": len(T["membros"]), "fontes": fontes,
                    "nasceu_no_video": T.get("nasceu"), "lei": LB.formula_lei(T["lei"]),
                    "regra": regra_em_palavras(T["lei"], LB.NOMES_BASE + LB.NOMES_CONC), "frases": frases,
                    "conceito": None if conc is None or conc not in lab.conceitos else
                    {"id": conc, "formula": LB.formula_conceito(lab.conceitos[conc])},
                    "erro_1a_px": round(float(np.median(erro)), 2) if erro else None,
                    "inercia_px": round(float(np.median(iner)), 2) if iner else None,
                    "cena_do_video": int(e), "url": ep.get("url", "")[:90], "forma": [int(H), int(W)], "L": int(L),
                    "tem_frente": c.get("frente") is not None, "reais": reais, "imaginados_do_real": imag,
                    "retos": retos, "epocas": epocas, "erros_epoca": erros_epoca,
                    "novos": novos, "correntes": correntes, "visto": visto, "historia": T.get("historia", [])[-12:],
                    "nomes": nomes_do_mundo(m_ref, T["id"]) if m_ref is not None else [],
                    "_cena": c})
    return out[:max_t]


def atlas_fisico(c):
    """PNG com as camadas: [fundo | frente (so o que ela deduziu estar na frente, o resto
    transparente)] e, embaixo, os recortes reais das coisas (com a mascara como transparencia)."""
    H, W = c["forma"][:2]
    sp = c["sprites"]
    lado = max(r.shape[0] for r, _m in sp) if sp else 11
    A = np.zeros((H + lado + 2, max(2 * W, lado * len(sp) + 2), 4), np.uint8)
    A[:H, :W, :3] = c["fundo"]
    A[:H, :W, 3] = 255
    rects = {"fundo": [0, 0, W, H], "frente": None, "coisas": []}
    if c.get("frente") is not None:
        A[:H, W:2 * W, :3] = c["fundo"]
        A[:H, W:2 * W, 3] = np.where(c["frente"], 255, 0).astype(np.uint8)
        rects["frente"] = [W, 0, W, H]
    for k, (rec, mask) in enumerate(sp):
        h, w = rec.shape[:2]
        x0, y0 = k * lado, H + 2
        A[y0:y0 + h, x0:x0 + w, :3] = rec
        A[y0:y0 + h, x0:x0 + w, 3] = np.where(mask, 255, 0).astype(np.uint8)
        rects["coisas"].append([x0, y0, w, h])
    return _png(A), rects


# ── publicar ──────────────────────────────────────────────────────────────────
def publica(m, pasta, cache=None):
    """escreve <pasta>/mundos.json e <pasta>/mundos/*.png. cache: dict guardado pela mente
    entre chamadas (indice da memoria e quando foi feito). Devolve o resumo publicado."""
    import json
    cache = {} if cache is None else cache
    dest = os.path.join(pasta, "mundos")
    os.makedirs(dest, exist_ok=True)
    agora = time.time()
    out = {"quando": agora, "fisicos": [], "conceitos": [], "limites": [
        "Profundidade: so a ORDEM de camadas que ela deduziu de quem esconde quem (2,5D); a distancia entre as camadas na tela e escolha de exibicao.",
        "Uma palavra visual dela e um padrao de ~22 px (do tamanho de uma letra), nao um objeto inteiro.",
        "O arquetipo e decodificado HOJE pelo cortex de hoje; o brilho e o contraste vem da media das olhadas usadas.",
        "Som: ela guarda so a cor do espectro (24 bandas) de trechos de 0,2 s; ritmo, altura exata e volume nao existem na memoria dela.",
        "Fogo, luz, sombra e mudanca de cor de UM objeto ainda nao sao representados; os verbos visuais sao transformacoes da cena inteira."]}
    lab = getattr(m, "lab", None)
    if lab is not None:
        for f in fisicos(lab, m_ref=m):
            c = f.pop("_cena")
            png, rects = atlas_fisico(c)
            nome = f"T{f['id']}.png"
            _grava(os.path.join(dest, nome), png)
            f["atlas"] = nome
            f["rects"] = rects
            out["fisicos"].append(f)
    cx = getattr(m, "cortex", None)
    if cx is not None and len(getattr(m, "olho").P) and getattr(m, "gr", None) is not None and m.gr.conf:
        if cache.get("idx") is None or agora - cache.get("idx_t", 0) > 600 or cache.get("idx_n") != cx.n_mem and \
                agora - cache.get("idx_t", 0) > 120:
            cache["idx"] = indice_memoria(cx, m.olho)
            cache["idx_t"], cache["idx_n"] = agora, cx.n_mem
        fundo = _mapa_neutro(cx)
        for c in conceitos(m, cache["idx"]):
            png, rects = folha_conceito(m, c, fundo)
            nome = f"c_{c['chave']}.png"
            _grava(os.path.join(dest, nome), png)
            c.pop("s")
            c["lembrancas"] = len(c["lembrancas"])
            c["folha"] = nome
            c["rects"] = rects
            out["conceitos"].append(c)
    arq = os.path.join(pasta, "mundos.json")
    _grava(arq, json.dumps(out, ensure_ascii=False).encode("utf-8"))
    return out
