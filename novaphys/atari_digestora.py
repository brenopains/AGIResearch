"""
novaphys.atari_digestora — o mundo PERCEBIDO: a maquina acha sozinha onde o humano olha?
==========================================================================================

DE ONDE VEM
-------------
No xadrez ([[xadrez-extracao]]) as relacoes vinham prontas das REGRAS. Num mundo
percebido nao ha regra: ha pixels. O usuario pediu "percepcao temporal enquanto
consome", "gente jogando jogos", extracao "sem um deus escrevendo a mao". O
Atari-HEAD (Zenodo, CC-BY) tem, por quadro: a imagem, a acao do jogador, a
recompensa e o OLHAR gravado — onde a atencao humana estava.

O QUE A MAQUINA RECEBE — dito por inteiro
-------------------------------------------
  pixels, a acao de cada quadro, a recompensa de cada quadro
  COESAO    pixels vizinhos de mesma cor sao um objeto (primitiva inata, Spelke)
  CONTINUIDADE  o objeto no quadro seguinte e o mais proximo de mesma cor
Nao recebe: qual objeto e o jogador, o que e perigoso, o que vale ponto, e NUNCA
o olhar (so o relatorio le o olhar).

O QUE ELA EXTRAI SOZINHA (numa partida de treino)
---------------------------------------------------
  TIPOS      objetos agrupados por cor e ordem de grandeza da area
  EU         o tipo cujo movimento mais depende da ACAO (informacao mutua acao ->
             deslocamento): contingencia, como o bebe descobre a propria mao
  O QUE INFORMA A ESCOLHA   o tipo cuja posicao RELATIVA ao EU mais preve a
             proxima acao — relevancia pela escolha, o sinal que no xadrez previu
             o lance humano (+0,091)
  SURPRESA   o objeto que mais foge da previsao por inercia (modelo de mundo
             minimo: continua na mesma velocidade)

PRE-REGISTRO — escrito antes de rodar
---------------------------------------
Avaliacao numa partida DIFERENTE (outra sessao) de cada um dos 6 jogos (breakout,
freeway, montezuma_revenge, ms_pacman, space_invaders, seaquest). ACERTO = o
olhar medio do quadro cai dentro da caixa do objeto escolhido, dilatada em 8 px
(precisao tipica de rastreador de olhar).
  M1 (principal) o objeto do tipo QUE INFORMA A ESCOLHA mais proximo do EU
  M2  o EU     |  M3  o objeto mais SURPREENDENTE
  LINHAS DE BASE: B1 objeto sorteado (esperanca exata), B2 o maior, B3 o que mais
  se move
  A0  PRE-CONDICAO (o olhar esta na escala da tela): a distancia do olhar ao
      objeto mais proximo e menor que a do olhar de OUTRO quadro em >= 5 jogos
  A1  M1 acerta >= 10 pontos percentuais acima da MELHOR linha de base em >= 4
      dos 6 jogos, com IC95 de bootstrap por blocos acima de zero -> a maquina
      extrai sozinha o que o humano considera relevante
  A2  2 ou 3 jogos | A3 <= 1
PREVISAO: A1 a 0,30, A2 a 0,40, A3 a 0,30. O olhar humano segue muito o que se
MOVE; a linha de base B3 e forte e pode ganhar de tudo.

CONSERTOS DE INSTRUMENTO NA FUMACA (600 quadros, antes da rodada que vale):
acerto era pela CAIXA e a parede em U do breakout cobre a tela (o "maior" acertava
100%) -> agora pela distancia aos PIXELS; o tipo por "cor + ordem de grandeza da
area" misturava raquete com caco de tijolo da mesma cor -> agora cor + FORMA; o
tar.bz2 era relido do comeco a cada quadro -> uma passada so. Na 2a fumaca: a
raquete EM MOVIMENTO sai borrada (bordas noutro tom, corpo encolhe) e "mesma cor
exata" a quebrava em pedacos que nunca se moviam -> objeto = regiao conexa nao
fundo, cor dominante como traco; e a IM ganhou correcao de vies (menos a media
com a acao embaralhada) — a bola vencia a contingencia com 26 pares.

RESULTADO (24/09): A3, 0 de 6. Escala do olhar conferida 6/6; EU certo em ~4/6
(raquete, galinha, Pac-Man, submarino — leitura pela cor). "Informa a escolha"
perde do "mais se move" em 5/6 (Pac-Man +5,1 pp, abaixo dos 10). Causa principal:
o TIPO por aparencia quebra quando o objeto sai borrado, e o alvo some.

TRILHAS — 2a pergunta, pre-registrada ANTES de rodar
------------------------------------------------------
Duas mudancas, cada uma respondendo a um defeito medido acima:
  IDENTIDADE POR CONTINUIDADE: o objeto herda a trilha do mais proximo do quadro
  anterior (qualquer cor); o tipo e o da TRILHA inteira (aparencia mais frequente
  na vida dela), nao o do quadro.
  A PERGUNTA JUSTA: a relevancia ACRESCENTA a saliencia? MC = o objeto do tipo que
  informa a escolha mais proximo do EU; se nao houver, o que mais se move.
DADOS NOVOS: sessoes 3 (extrai) e 4 (avalia) de cada jogo, nunca tocadas.
  T1  MC >= +5 pp sobre a MELHOR linha de base, IC95 acima de zero, em >= 4 dos
      6 jogos (limiar menor que o de A1 porque a pergunta mudou de SUBSTITUIR a
      saliencia para ACRESCENTAR a ela — dito antes de rodar; o +10 pp tambem e
      relatado)
  T2  2 ou 3 | T3 <= 1
PREVISAO: T1 a 0,30, T2 a 0,35, T3 a 0,35.
FUMACA (sessoes 1-2, ja usadas): o "informa a escolha" virou os DIGITOS DO PLACAR —
parados, a posicao relativa deles so codifica ONDE O EU ESTA. Conserto antes da
rodada que vale: informacao mutua CONDICIONADA a posicao do EU (a informacao alem
do que eu ja sei sobre mim).
"""

from __future__ import annotations

import io
import math
import os
import random
import sys
import tarfile
import time
import zipfile
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

PASTA = r"D:\agi_dados\atari_head"
JOGOS = ("breakout", "freeway", "montezuma_revenge", "ms_pacman",
         "space_invaders", "seaquest")
MAX_Q = 6000
DIL = 8


# ============================================================ dado cru
def ensaios(jogo: str) -> List[str]:
    z = zipfile.ZipFile(os.path.join(PASTA, jogo + ".zip"))
    return sorted(i.filename for i in z.infolist() if i.filename.endswith(".tar.bz2"))


def carrega(jogo: str, ensaio: str, max_q: Optional[int] = None):
    """Quadros (em ordem), acao, recompensa e olhar medio por quadro."""
    import cv2
    max_q = max_q or MAX_Q
    z = zipfile.ZipFile(os.path.join(PASTA, jogo + ".zip"))
    linhas = z.read(ensaio.replace(".tar.bz2", ".txt")).decode().splitlines()[1:]
    meta = {}
    for l in linhas:
        c = l.split(",")
        if len(c) < 7 or c[5] == "null":
            continue
        try:
            n = int(c[0].rsplit("_", 1)[1])
        except ValueError:
            continue
        olhar = None
        if c[6] != "null":
            v = [float(x) for x in c[6:] if x]
            pts = [(v[i], v[i + 1]) for i in range(0, len(v) - 1, 2)
                   if 0 <= v[i] < 160 and 0 <= v[i + 1] < 210]
            if pts:
                olhar = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
        meta[n] = (int(c[5]), float(c[4]) if c[4] != "null" else 0.0, olhar)
    brutos = {}
    with tarfile.open(fileobj=io.BytesIO(z.read(ensaio)), mode="r|bz2") as t:
        for m in t:                               # UMA passada, em sequencia
            if m.name.endswith(".png"):
                n = int(m.name.rsplit("_", 1)[1][:-4])
                if n in meta:
                    brutos[n] = t.extractfile(m).read()
    return [(brutos[n],) + meta[n] for n in sorted(brutos)[:max_q]]   # PNG: decodifica na hora


# ============================================================ COESAO
def percebe(im) -> List[dict]:
    """
    Coesao: cada regiao CONEXA que nao e fundo e um objeto, qualquer que seja a
    mistura de cores (a raquete em movimento sai borrada em dois tons e continua
    sendo uma coisa so). A cor DOMINANTE e um traco do objeto, nao a fronteira.
    """
    from scipy import ndimage
    if isinstance(im, (bytes, bytearray)):
        import cv2
        im = cv2.imdecode(np.frombuffer(im, np.uint8), cv2.IMREAD_COLOR)
    cod = (im[:, :, 0].astype(np.int32) << 16) | (im[:, :, 1].astype(np.int32) << 8) | im[:, :, 2]
    vals, cont = np.unique(cod, return_counts=True)
    fundo = vals[cont.argmax()]
    lab, k = ndimage.label(cod != fundo, structure=np.ones((3, 3), int))
    obs = []
    for sl, idx in zip(ndimage.find_objects(lab), range(1, k + 1)):
        ys, xs = sl
        m = lab[sl] == idx
        py, px = np.nonzero(m)
        area = int(len(py))
        if area > 0.25 * cod.size:
            continue
        cs, ns = np.unique(cod[sl][m], return_counts=True)
        obs.append({"cor": int(cs[ns.argmax()]), "area": area,
                    "px": (py + ys.start, px + xs.start),
                    "caixa": (ys.start, ys.stop - 1, xs.start, xs.stop - 1),
                    "c": ((ys.start + ys.stop - 1) / 2, (xs.start + xs.stop - 1) / 2)})
    return obs


def tipo(o) -> Tuple[int, int, int]:
    """Cor dominante + altura + ordem de grandeza da largura (o borrado muda a largura)."""
    t, b, l, r = o["caixa"]
    return (o["cor"], b - t + 1, int(round(math.log2(r - l + 1))))


# ======================================================= CONTINUIDADE
def liga(ant: List[dict], atu: List[dict]) -> None:
    """Cada objeto atual herda velocidade do mais proximo de mesma cor no anterior."""
    por_cor = defaultdict(list)
    for o in ant:
        por_cor[o["cor"]].append(o)
    for o in atu:
        melhor, dm = None, 20.0
        for p in por_cor.get(o["cor"], ()):
            if not 0.5 <= o["area"] / p["area"] <= 2:
                continue
            d = math.hypot(o["c"][0] - p["c"][0], o["c"][1] - p["c"][1])
            if d < dm:
                melhor, dm = p, d
        if melhor is not None:
            o["v"] = (o["c"][0] - melhor["c"][0], o["c"][1] - melhor["c"][1])
            vp = melhor.get("v") or (0.0, 0.0)
            prev = (melhor["c"][0] + vp[0], melhor["c"][1] + vp[1])
            o["surpresa"] = math.hypot(o["c"][0] - prev[0], o["c"][1] - prev[1])
        else:
            o["v"] = None
            o["surpresa"] = 0.0


def _sinal(x, lim=0.5):
    return 0 if abs(x) < lim else (1 if x > 0 else -1)


def _im_bruta(pares) -> float:
    n = len(pares)
    if n == 0:
        return 0.0
    cab, ca, cb = Counter(pares), Counter(a for a, _ in pares), Counter(b for _, b in pares)
    return sum(c / n * math.log2(c * n / (ca[a] * cb[b])) for (a, b), c in cab.items())


def _im(pares) -> float:
    """
    Informacao mutua (bits) MENOS a media com a 1a coluna embaralhada: amostra
    pequena infla a IM (a bola 'ganhava' com 26 pares por puro vies).
    """
    if not pares:
        return 0.0
    rnd = random.Random(len(pares))
    a = [x for x, _ in pares]
    b = [y for _, y in pares]
    nulo = 0.0
    for _ in range(3):
        rnd.shuffle(a)
        nulo += _im_bruta(list(zip(a, b))) / 3
    return _im_bruta(pares) - nulo


# ============================================================ EXTRAI
def extrai(seq) -> dict:
    """Numa partida de treino, SEM olhar: quem e o EU, o que informa a escolha."""
    pres = Counter()
    mov = defaultdict(list)                 # tipo -> [(acao, (sx, sy))]
    quadros = []
    ant = None
    for im, acao, recomp, _olhar in seq:
        obs = percebe(im)
        if ant is not None:
            liga(ant, obs)
        quadros.append((obs, acao))
        for tp in {tipo(o) for o in obs}:
            pres[tp] += 1
        ant = obs
    n = len(quadros)
    # EU: contingencia acao(t) -> deslocamento do tipo em t+1
    for t in range(n - 1):
        acao = quadros[t][1]
        por_tipo = defaultdict(list)
        for o in quadros[t + 1][0]:
            if o.get("v") is not None:
                por_tipo[tipo(o)].append(o["v"])
        for tp, vs in por_tipo.items():
            if len(vs) <= 3:
                dy = sum(v[0] for v in vs) / len(vs)
                dx = sum(v[1] for v in vs) / len(vs)
                mov[tp].append((acao, (_sinal(dx), _sinal(dy))))
    cont = {tp: _im(ps) for tp, ps in mov.items() if pres[tp] >= 0.3 * n and len(ps) >= 200}
    eu = max(cont, key=cont.get) if cont else None
    # O QUE INFORMA A ESCOLHA: posicao relativa ao EU no quadro t -> acao em t+1
    info = {}
    if eu is not None:
        rel = defaultdict(list)
        for t in range(n - 1):
            obs, _a = quadros[t]
            eus = [o for o in obs if tipo(o) == eu]
            if not eus:
                continue
            e = eus[0]["c"]
            por_tipo = defaultdict(list)
            for o in obs:
                tp = tipo(o)
                if tp != eu:
                    por_tipo[tp].append(o)
            for tp, os_ in por_tipo.items():
                o = min(os_, key=lambda p: math.hypot(p["c"][0] - e[0], p["c"][1] - e[1]))
                chave = (_sinal(o["c"][1] - e[1], 2), _sinal(o["c"][0] - e[0], 2))
                rel[tp].append((chave, quadros[t + 1][1]))
        info = {tp: _im(ps) for tp, ps in rel.items() if pres[tp] >= 0.2 * n and len(ps) >= 200}
    alvo = max(info, key=info.get) if info else None
    return {"eu": eu, "alvo": alvo, "cont": cont, "info": info}


# ============================================================ AVALIA
def _dist(o, g) -> float:
    """Distancia (Chebyshev) do ponto de olhar ao PIXEL mais proximo do objeto."""
    py, px = o["px"]
    return float(np.min(np.maximum(np.abs(py - g[1]), np.abs(px - g[0]))))


def _acerta(o, g) -> bool:
    return _dist(o, g) <= DIL


def avalia(seq, mod, rnd) -> dict:
    """Numa partida NOVA: para cada quadro com olhar, quem cada principio escolhe."""
    brac = defaultdict(list)
    dist, dist_outro = [], []
    olhares = [g for *_x, g in seq if g is not None]
    ant = None
    for im, _a, _r, g in seq:
        obs = percebe(im)
        if ant is not None:
            liga(ant, obs)
        ant = obs
        if g is None or not obs:
            continue

        def d_min(pt):
            return min(_dist(o, pt) for o in obs)
        dist.append(d_min(g))
        dist_outro.append(d_min(rnd.choice(olhares)))
        brac["B1 sorteado"].append(sum(_acerta(o, g) for o in obs) / len(obs))
        brac["B2 maior"].append(float(_acerta(max(obs, key=lambda o: o["area"]), g)))
        movs = [o for o in obs if o.get("v")]
        rapido = max(movs, key=lambda o: math.hypot(*o["v"])) if movs else None
        brac["B3 mais se move"].append(float(_acerta(rapido, g)) if rapido else 0.0)
        surp = max(obs, key=lambda o: o.get("surpresa", 0.0))
        brac["M3 surpresa"].append(float(_acerta(surp, g)))
        eus = [o for o in obs if tipo(o) == mod["eu"]]
        brac["M2 EU"].append(float(_acerta(eus[0], g)) if eus else 0.0)
        alvos = [o for o in obs if tipo(o) == mod["alvo"]]
        if alvos:
            e = eus[0]["c"] if eus else (105, 80)
            o = min(alvos, key=lambda p: math.hypot(p["c"][0] - e[0], p["c"][1] - e[1]))
            brac["M1 informa a escolha"].append(float(_acerta(o, g)))
        else:
            brac["M1 informa a escolha"].append(0.0)
    a0 = sum(d < d2 for d, d2 in zip(dist, dist_outro)) / max(1, len(dist))
    return {"bracos": brac, "a0": a0, "d": (np.median(dist), np.median(dist_outro))}


def _ic(dif, rnd, bloco=50, reps=1000):
    blocos = [dif[i:i + bloco] for i in range(0, len(dif), bloco)]
    ms = sorted(sum(sum(b) for b in amostra) / sum(len(b) for b in amostra)
                for amostra in ([rnd.choice(blocos) for _ in blocos] for _ in range(reps)))
    return ms[int(0.025 * reps)], ms[int(0.975 * reps)]


def um_jogo(jogo: str) -> dict:
    t0 = time.time()
    es = ensaios(jogo)
    treino, teste = carrega(jogo, es[0]), carrega(jogo, es[1])
    mod = extrai(treino)
    rnd = random.Random(hash(jogo) % 1000)
    av = avalia(teste, mod, rnd)
    br = av["bracos"]
    taxas = {k: sum(v) / len(v) for k, v in br.items()}
    base = max(("B1 sorteado", "B2 maior", "B3 mais se move"), key=lambda k: taxas[k])
    dif = [a - b for a, b in zip(br["M1 informa a escolha"], br[base])]
    ic = _ic(dif, random.Random(1))
    return {"jogo": jogo, "eu": mod["eu"], "alvo": mod["alvo"], "taxas": taxas,
            "base": base, "ic": ic, "a0": av["a0"], "d": av["d"],
            "n": len(dif), "s": time.time() - t0,
            "cont": sorted(mod["cont"].items(), key=lambda x: -x[1])[:3],
            "info": sorted(mod["info"].items(), key=lambda x: -x[1])[:3]}


def _cor(c):
    return "#%06x" % c


def main():
    from multiprocessing import Pool
    with Pool(3) as p:                               # RAM: ~2 GB livres de verdade
        rs = p.map(um_jogo, JOGOS, 1)
    print("=" * 78)
    print("ATARI-HEAD — a maquina acha sozinha onde o humano olha?")
    print("=" * 78)
    a0_ok, a1 = 0, 0
    for r in rs:
        tx = r["taxas"]
        m1 = tx["M1 informa a escolha"]
        ganho = m1 - tx[r["base"]]
        ok = ganho >= 0.10 and r["ic"][0] > 0
        a1 += ok
        a0_ok += r["a0"] > 0.5
        print("\n  %s  (%d quadros com olhar, %.0f s)" % (r["jogo"].upper(), r["n"], r["s"]))
        print("    A0 olhar mais perto de objeto que olhar de outro quadro: %.0f%% "
              "(mediana %.1f px x %.1f px)" % (100 * r["a0"], r["d"][0], r["d"][1]))
        def _t(tp):
            return "cor %s %dx%d" % (_cor(tp[0]), tp[1], tp[2]) if tp else "-"
        print("    EU = %s | informa a escolha = %s" % (_t(r["eu"]), _t(r["alvo"])))
        for k in sorted(tx):
            print("      %-22s %5.1f%%" % (k, 100 * tx[k]))
        print("    M1 - %s = %+.1f pp (IC95 %+.1f a %+.1f) %s"
              % (r["base"], 100 * ganho, 100 * r["ic"][0], 100 * r["ic"][1],
                 "PASSA" if ok else "-----"))
    ver = "A1" if a1 >= 4 else "A2" if a1 >= 2 else "A3"
    print("\n  A0: %d de 6 jogos | VEREDITO: %s (%d de 6)" % (a0_ok, ver, a1))


# ============================================ 2a pergunta: TRILHAS e SALIENCIA
def trilhas(seq) -> List[Tuple[List[dict], int]]:
    """
    Percebe e LIGA pela continuidade: cada objeto herda a trilha do objeto mais
    proximo do quadro anterior (qualquer cor, area ate 3x), guloso por distancia.
    O TIPO do objeto e o da TRILHA inteira — a aparencia mais frequente ao longo
    da vida dela (cor em 3 bits por canal, ordem de grandeza da area). O borrado
    de um quadro nao muda mais o que a coisa E.
    """
    quadros = []
    ant: List[dict] = []
    prox_id = 0
    for im, acao, _r, _g in seq:
        obs = percebe(im)
        pares = []
        for i, p in enumerate(ant):
            for j, o in enumerate(obs):
                d = math.hypot(o["c"][0] - p["c"][0], o["c"][1] - p["c"][1])
                if d < 20 and 1 / 3 <= o["area"] / p["area"] <= 3:
                    pares.append((d, i, j))
        pares.sort()
        usou_i, usou_j = set(), set()
        for d, i, j in pares:
            if i in usou_i or j in usou_j:
                continue
            usou_i.add(i)
            usou_j.add(j)
            p, o = ant[i], obs[j]
            o["id"] = p["id"]
            o["v"] = (o["c"][0] - p["c"][0], o["c"][1] - p["c"][1])
            vp = p.get("v") or (0.0, 0.0)
            o["surpresa"] = math.hypot(o["c"][0] - p["c"][0] - vp[0],
                                       o["c"][1] - p["c"][1] - vp[1])
        for j, o in enumerate(obs):
            if j not in usou_j:
                o["id"] = prox_id
                prox_id += 1
                o["v"] = None
                o["surpresa"] = 0.0
        quadros.append((obs, acao))
        ant = obs
    aparencias = defaultdict(Counter)
    for obs, _a in quadros:
        for o in obs:
            c = o["cor"]
            q = ((c >> 21) & 7, (c >> 13) & 7, (c >> 5) & 7)
            aparencias[o["id"]][(q, int(round(math.log2(max(o["area"], 1)))))] += 1
    tipo_de = {i: c.most_common(1)[0][0] for i, c in aparencias.items()}
    for obs, _a in quadros:
        for o in obs:
            o["tipo"] = tipo_de[o["id"]]
    return quadros


def _im_cond(triplas) -> float:
    """
    I(X; A | Z) com Z = onde o EU esta: a informacao ALEM do que eu ja sei sobre
    mim mesmo. Um marco parado (o placar) so codifica a posicao do EU e vale ~0.
    Corrigida pelo vies: menos a media com A embaralhado DENTRO de cada Z.
    """
    por_z = defaultdict(list)
    for x, a, z in triplas:
        por_z[z].append((x, a))
    n = len(triplas)
    rnd = random.Random(n)
    total = 0.0
    for z, ps in por_z.items():
        if len(ps) < 20:
            continue
        real = _im_bruta(ps)
        xs = [x for x, _ in ps]
        as_ = [a for _, a in ps]
        nulo = 0.0
        for _ in range(3):
            rnd.shuffle(as_)
            nulo += _im_bruta(list(zip(xs, as_))) / 3
        total += len(ps) / n * (real - nulo)
    return total


def extrai2(quadros) -> dict:
    """EU e O QUE INFORMA A ESCOLHA, agora por tipo-de-trilha."""
    n = len(quadros)
    pres = Counter()
    for obs, _a in quadros:
        for tp in {o["tipo"] for o in obs}:
            pres[tp] += 1
    mov = defaultdict(list)
    for t in range(n - 1):
        acao = quadros[t][1]
        por = defaultdict(list)
        for o in quadros[t + 1][0]:
            if o.get("v") is not None:
                por[o["tipo"]].append(o["v"])
        for tp, vs in por.items():
            if len(vs) <= 3:
                mov[tp].append((acao, (_sinal(sum(v[1] for v in vs) / len(vs)),
                                       _sinal(sum(v[0] for v in vs) / len(vs)))))
    cont = {tp: _im(ps) for tp, ps in mov.items() if pres[tp] >= 0.3 * n and len(ps) >= 200}
    eu = max(cont, key=cont.get) if cont else None
    info = {}
    if eu is not None:
        rel = defaultdict(list)
        for t in range(n - 1):
            obs = quadros[t][0]
            eus = [o for o in obs if o["tipo"] == eu]
            if not eus:
                continue
            e = eus[0]["c"]
            por = defaultdict(list)
            for o in obs:
                if o["tipo"] != eu:
                    por[o["tipo"]].append(o)
            for tp, os_ in por.items():
                o = min(os_, key=lambda p: math.hypot(p["c"][0] - e[0], p["c"][1] - e[1]))
                rel[tp].append(((_sinal(o["c"][1] - e[1], 2), _sinal(o["c"][0] - e[0], 2)),
                                quadros[t + 1][1], (int(e[0]) // 16, int(e[1]) // 16)))
        info = {tp: _im_cond(ps) for tp, ps in rel.items()
                if pres[tp] >= 0.2 * n and len(ps) >= 200}
    return {"eu": eu, "alvo": max(info, key=info.get) if info else None,
            "cont": cont, "info": info}


def avalia2(seq, quadros, mod) -> dict:
    brac = defaultdict(list)
    for (obs, _a), (_im_, _ac, _r, g) in zip(quadros, seq):
        if g is None or not obs:
            continue
        brac["B1 sorteado"].append(sum(_acerta(o, g) for o in obs) / len(obs))
        brac["B2 maior"].append(float(_acerta(max(obs, key=lambda o: o["area"]), g)))
        movs = [o for o in obs if o.get("v")]
        rapido = max(movs, key=lambda o: math.hypot(*o["v"])) if movs else None
        brac["B3 mais se move"].append(float(_acerta(rapido, g)) if rapido else 0.0)
        eus = [o for o in obs if o["tipo"] == mod["eu"]]
        brac["M2 EU"].append(float(_acerta(eus[0], g)) if eus else 0.0)
        alvos = [o for o in obs if o["tipo"] == mod["alvo"]]
        if alvos:
            e = eus[0]["c"] if eus else (105, 80)
            o = min(alvos, key=lambda p: math.hypot(p["c"][0] - e[0], p["c"][1] - e[1]))
            brac["M1 informa a escolha"].append(float(_acerta(o, g)))
            brac["MC relevancia + saliencia"].append(float(_acerta(o, g)))
        else:
            brac["M1 informa a escolha"].append(0.0)
            brac["MC relevancia + saliencia"].append(
                float(_acerta(rapido, g)) if rapido else 0.0)
    return brac


def um_jogo2(jogo: str) -> dict:
    t0 = time.time()
    es = ensaios(jogo)
    tr_seq, te_seq = carrega(jogo, es[2]), carrega(jogo, es[3])   # sessoes NOVAS
    mod = extrai2(trilhas(tr_seq))
    br = avalia2(te_seq, trilhas(te_seq), mod)
    tx = {k: sum(v) / len(v) for k, v in br.items()}
    base = max(("B1 sorteado", "B2 maior", "B3 mais se move"), key=lambda k: tx[k])
    dif = [a - b for a, b in zip(br["MC relevancia + saliencia"], br[base])]
    return {"jogo": jogo, "eu": mod["eu"], "alvo": mod["alvo"], "taxas": tx, "base": base,
            "ic": _ic(dif, random.Random(1)), "n": len(dif), "s": time.time() - t0}


def main2():
    from multiprocessing import Pool
    with Pool(3) as p:
        rs = p.map(um_jogo2, JOGOS, 1)
    print("=" * 78)
    print("ATARI-HEAD 2 — identidade por TRILHA; a relevancia ACRESCENTA a saliencia?")
    print("=" * 78)
    passa = passa10 = 0
    for r in rs:
        tx = r["taxas"]
        ganho = tx["MC relevancia + saliencia"] - tx[r["base"]]
        ok = ganho >= 0.05 and r["ic"][0] > 0
        passa += ok
        passa10 += ganho >= 0.10 and r["ic"][0] > 0
        print("\n  %s  (%d quadros com olhar, %.0f s) | EU %s | informa %s"
              % (r["jogo"].upper(), r["n"], r["s"], r["eu"], r["alvo"]))
        for k in sorted(tx):
            print("      %-26s %5.1f%%" % (k, 100 * tx[k]))
        print("    MC - %s = %+.1f pp (IC95 %+.1f a %+.1f) %s"
              % (r["base"], 100 * ganho, 100 * r["ic"][0], 100 * r["ic"][1],
                 "PASSA" if ok else "-----"))
    ver = "T1" if passa >= 4 else "T2" if passa >= 2 else "T3"
    print("\n  VEREDITO: %s (%d de 6 com +5 pp; %d de 6 com +10 pp)" % (ver, passa, passa10))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "trilhas":
        main2()
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "fumaca":
        import novaphys.atari_digestora as A
        A.MAX_Q = 600
        r = A.um_jogo("breakout")
        print(r)
    else:
        main()
