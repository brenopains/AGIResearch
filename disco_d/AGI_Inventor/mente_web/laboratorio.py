"""
mente_web.laboratorio — o CICLO DE DESCOBERTA ligado a mente viva (transplante 10)
====================================================================================
De onde vem. A revisao externa de 09/10 (GPT, lendo o ARQUITETURA.md inteiro) e a
auditoria que a confirmou: os mecanismos que SOBREVIVERAM aos controles no AGI 2.0
nunca foram ligados a mente viva —
  representacao/percepcao  pixels -> objetos pela CONTINUIDADE (Etapa 1: h em 3/3)
  novaphys/inventa         inventar a grandeza conservada (h em 10/10, placebo 0)
  novaphys/caderno         acumular conclusoes ENTRINCHEIRA o primeiro erro;
                           reanalisar os dados guardados com olhos frescos: 37% -> 88%
  novaphys/autonomo        crise = a lei que ja tenho erra acima do ruido; resposta =
                           reanalise por consenso (83% contra 68% do meu calendario)
  novaphys/conceito_util   o conceito e julgado pelo que PAGA na lei, nao pelo que
                           comprime; sala de espera refeita quando a evidencia dobra
  novaphys/sob_demanda     guardar tudo e ordenar na hora vence filtrar na entrada
Este orgao e esses seis mecanismos num laco so, rodando sobre o que ELA assiste.

O laco, por episodio (um video, da internet ou da arena):
  1. ENTIDADES. Fundo = o que fica parado no video (mediana no tempo); objeto = regiao
     conexa do que difere do fundo (coesao, qualquer mistura de cores). IDENTIDADE vem
     da continuidade (licao do Atari-HEAD: aparencia engana), por atribuicao otima com
     previsao de velocidade. PERMANENCIA: trilha que some e trilha que aparece ate
     1,5 s depois, perto de onde a primeira iria estar, sao a MESMA coisa (religar) —
     "o que aconteceu com aquilo que eu estava acompanhando?".
  2. HIPOCAMPO. Guarda o episodio CRU (tempos e posicoes das trilhas), nao a conclusao.
     Estados, leis e conceitos sao sempre RE-DERIVADOS do cru: a memoria e aquilo sobre
     o qual ela ainda pode aprender.
  3. TEORIAS. Cada teoria e uma lei de movimento (aceleracao como combinacao de termos
     do estado, escolhida por bits) + o ruido proprio dela, medido FORA da amostra.
     Antes de aprender com o episodio, ela PREVE onde cada coisa estara 2 s depois (a
     primeira tentativa). Se a melhor teoria explica dentro do proprio ruido: o
     episodio entra nela. Se nao: CRISE.
  4. CRISE -> REANALISE POR CONSENSO. Nova teoria a partir do que surpreendeu; ela
     incorpora os episodios GUARDADOS (de qualquer teoria) que explica abaixo do ruido
     e melhor do que a teoria onde estavam — o passado e REINTERPRETADO. Teoria que fica
     sem episodio morre.
  5. CONCEITO. Na crise e sempre que a evidencia de uma teoria DOBRA, ela procura
     grandezas que ficam constantes em cada trajetoria e variam entre trajetorias
     (menor autovetor generalizado sobre os 14 monomios de grau <= 2 do estado, regua
     em bits, validada em trajetorias fora do ajuste). As aceitas vao para a SALA DE
     ESPERA. A escolha (nenhum conceito, ou um) e feita pela previsao fora da amostra
     da lei redescoberta com cada opcao. Conceitos sao do VOCABULARIO, nao da teoria:
     um conceito inventado num mundo pode ser reutilizado noutro.

O que e meu (declarado): o alfabeto (14 monomios + o conceito e o quadrado dele vezes
cada variavel), o horizonte de 2 s, os limiares de percepcao herdados da Etapa 1, o
alarme de 1% (convencao estatistica), e os tetos de memoria (orcamento do PC).
O que NAO faz ainda: linguagem, outros agentes. E o laboratorio NAO entra na economia
dela neste transplante: aprende e e medido, mas o que ela escolhe fazer continua
decidido pela moeda unica de antes.

TRANSPLANTE 11 (secao 5, MAOS): na arena ela pode MONTAR a cena em vez de so assistir.
As rivais sao as explicacoes que competem hoje (a lei da teoria atual, a mesma teoria
com cada outra opcao de conceito, e re-ajustes em reamostras das trajetorias dela); o
experimento e o lugar e a velocidade de cada coisa em que as rivais MAIS DISCORDAM do
que vai acontecer. Depois de ver, ganha a rival que previu melhor, trajetoria a
trajetoria; se for outra opcao de conceito, a sala de espera e refeita na hora.
Teste com controle (escolha ao acaso do MESMO sorteio de candidatos):
PRE_REGISTRO_experimento_ativo.md.
"""
from __future__ import annotations

import collections
import math
import time

import numpy as np
from scipy import ndimage
from scipy.linalg import LinAlgError, eigh
from scipy.optimize import linear_sum_assignment

L_MAX = 128               # lado maior do quadro que o laboratorio olha (px)
LIMIAR_FG = 40            # diferenca minima do fundo (soma RGB), herdada de percepcao.py
FRACAO_FUNDO_MAX = 0.35   # mais que isto mudando o tempo todo = camera mexendo: sem objetos
MAX_FRAG = 40             # fragmentos por quadro (os maiores)
MAX_PASSO = 8.0           # px (em 128) entre quadros seguidos
LACUNA_S = 3.0            # permanencia: quanto tempo uma coisa pode sumir e ser a mesma
MIN_PONTOS = 10
JANELA = 7
HORIZONTE = 2.0           # s: a primeira tentativa preve onde cada coisa estara
DT_ROLA = 0.05
PX = 64.0                 # 1 unidade normalizada = 64 px num quadro de 128
ALARME = 0.01             # crise: erro acima do quantil 99% do ruido da propria teoria
PISO_RAZAO = 0.02         # erro relativo abaixo disto e arredondamento, nao crise
PISO_ERRO_PX = 0.5        # idem em pixels
MAX_TERMOS = 10
MAX_EPISODIOS = 3000
MAX_CONSENSO = 800        # episodios guardados que a reanalise varre (os mais recentes)
MAX_TEORIAS = 40
K_CONCEITOS = 5
MIN_TRAJ_INVENTA = 16       # 8 para achar, 8 para validar fora (com 3 arcos qualquer forma quadratica 'se conserva')
MIN_PONTOS_INVENTA = 20
EXP_CANDIDATOS = 300      # maos: cenas sorteadas entre as quais ela escolhe
EXP_HORIZONTES = (1.0, 2.0, 3.0)
EXP_A_VISTA = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)   # a teoria atual preve que fica no quadro por 6 s
EXP_FATOR = (0.7, 1.3)    # intervencao: a rapidez de uma situacao que ela VIU, vezes isto
EXP_GIRO = math.pi / 6    # ... e a direcao girada ate 30 graus
EXP_INVERTE = 0.25        # ... ou invertida
EXP_CORPOS = 3
EXP_SEPARACAO = 0.2       # meias-larguras entre as coisas, no comeco e no caminho previsto (nao se fundirem)
EXP_CAMINHO = tuple(np.arange(0.5, 6.01, 0.5))
EXP_BOOT = 4              # reamostras das trajetorias no comite de rivais

NOMES_MON = ["x", "y", "vx", "vy", "x²", "x·y", "x·vx", "x·vy", "y²", "y·vx", "y·vy", "vx²", "vx·vy", "vy²"]
NOMES_BASE = ["1"] + NOMES_MON
NOMES_CONC = ["q", "q²", "q·x", "q·y", "q·vx", "q·vy", "q²·x", "q²·y", "q²·vx", "q²·vy"]


# ═════════════════════════════════════════════════════════════════════════════
# 1. ENTIDADES: coesao + continuidade + permanencia
# ═════════════════════════════════════════════════════════════════════════════
def _rgb(q):
    q = np.asarray(q)
    if q.dtype != np.uint8:
        q = (np.clip(q, 0, 1) * 255).astype(np.uint8)
    if q.ndim == 2:
        q = np.repeat(q[:, :, None], 3, 2)
    return q


def reduz(q, lado=L_MAX):
    q = _rgb(q)
    H, W = q.shape[:2]
    m = max(H, W)
    if m <= lado:
        return q
    from PIL import Image
    f = lado / m
    return np.asarray(Image.fromarray(q).resize((max(1, round(W * f)), max(1, round(H * f))), Image.BILINEAR),
                      np.uint8)


def fragmentos(q, fundo, limiar):
    d = np.abs(q.astype(np.int16) - fundo.astype(np.int16)).sum(2)
    fg = d > limiar
    if not fg.any():
        return [], float(fg.mean())
    lab, n = ndimage.label(fg, structure=np.ones((3, 3), int))
    idx = lab.ravel()
    cnt = np.bincount(idx, minlength=n + 1)
    w = d.ravel().astype(np.float64)
    sw = np.bincount(idx, weights=w, minlength=n + 1) + 1e-9
    yy, xx = np.indices(lab.shape)
    sx = np.bincount(idx, weights=w * xx.ravel(), minlength=n + 1)
    sy = np.bincount(idx, weights=w * yy.ravel(), minlength=n + 1)
    cores = np.stack([np.bincount(idx, weights=q[..., c].ravel().astype(np.float64), minlength=n + 1)
                      for c in range(3)], 1) / np.maximum(cnt, 1)[:, None]
    teto = 0.15 * lab.size
    ok = [i for i in range(1, n + 1) if 3 <= cnt[i] <= teto]
    ok.sort(key=lambda i: -cnt[i])
    out = [{"c": np.array([sx[i] / sw[i], sy[i] / sw[i]]), "area": float(cnt[i]), "cor": cores[i]}
           for i in ok[:MAX_FRAG]]
    return out, float(fg.mean())


def _trilha_nova(k, f):
    return {"k": [k], "p": [f["c"]], "a": [f["area"]], "cor": [f["cor"]]}


def rastreia(frs, tempos, esc):
    """Continuidade: cada objeto continua a trilha cuja posicao PREVISTA (velocidade da
    propria trilha) esta mais perto; atribuicao otima entre todos de uma vez."""
    trilhas, ativas = [], []
    gate = MAX_PASSO * esc
    for k, fs in enumerate(frs):
        dono = {}
        if ativas and fs:
            prev = []
            for i in ativas:
                T = trilhas[i]
                p = T["p"][-1]
                if len(T["k"]) >= 2:
                    dt0 = tempos[T["k"][-1]] - tempos[T["k"][-2]]
                    v = (T["p"][-1] - T["p"][-2]) / max(dt0, 1e-3)
                    p = p + v * (tempos[k] - tempos[T["k"][-1]])
                prev.append(p)
            P = np.array(prev)
            C = np.array([f["c"] for f in fs])
            D = np.linalg.norm(P[:, None] - C[None], axis=2)
            A0 = np.array([trilhas[i]["a"][-1] for i in ativas])
            A1 = np.array([f["area"] for f in fs])
            razao = A1[None] / A0[:, None]
            cor = np.linalg.norm(np.array([trilhas[i]["cor"][-1] for i in ativas])[:, None]
                                 - np.array([f["cor"] for f in fs])[None], axis=2)
            custo = D + 0.02 * cor
            custo[(D > gate) | (razao < 1 / 3) | (razao > 3)] = 1e6
            r, c = linear_sum_assignment(custo)
            for a, b in zip(r, c):
                if custo[a, b] < 1e5:
                    dono[b] = ativas[a]
        novas = []
        for j, f in enumerate(fs):
            if j in dono:
                T = trilhas[dono[j]]
                T["k"].append(k)
                T["p"].append(f["c"])
                T["a"].append(f["area"])
                T["cor"].append(f["cor"])
                novas.append(dono[j])
            else:
                trilhas.append(_trilha_nova(k, f))
                novas.append(len(trilhas) - 1)
        ativas = novas
    out = []
    for T in trilhas:
        out.append({"k": np.array(T["k"]), "p": np.array(T["p"], float), "a": np.array(T["a"], float),
                    "cor": np.array(T["cor"], float).mean(0), "quebras": []})
    return out


def cortar(trilhas, tempos, esc, eventos):
    """Fusao (area salta) sai; onde o passo foge da propria historia, a trilha e cortada."""
    out = []
    for T in trilhas:
        k, P, ar = T["k"], T["p"], T["a"]
        bom = ar < 1.6 * np.median(ar)
        if (~bom).any():
            eventos["cortar_fusao"] = eventos.get("cortar_fusao", 0) + int((~bom).sum())
        idx = np.nonzero(bom)[0]
        cortes = [0]
        for x, (a, b, c) in enumerate(zip(idx[:-2], idx[1:-1], idx[2:])):
            if k[c] - k[b] > 1 or k[b] - k[a] > 1:
                continue
            prev = P[b] + (P[b] - P[a]) * (tempos[k[c]] - tempos[k[b]]) / max(tempos[k[b]] - tempos[k[a]], 1e-3)
            if np.linalg.norm(P[c] - prev) > 3.0 * esc:
                cortes.append(x + 2)
                eventos["cortar_salto"] = eventos.get("cortar_salto", 0) + 1
        cortes.append(len(idx))
        for a, b in zip(cortes[:-1], cortes[1:]):
            sel = idx[a:b]
            if not len(sel):
                continue
            for s in np.split(sel, np.nonzero(np.diff(k[sel]) > 1)[0] + 1):
                if len(s):
                    out.append({"k": k[s], "p": P[s], "a": ar[s], "cor": T["cor"], "quebras": []})
    return out


def _extrapola(T, tempos, t_alvo, do_fim=True, n=10):
    k, P = (T["k"][-n:], T["p"][-n:]) if do_fim else (T["k"][:n], T["p"][:n])
    ref = tempos[k[-1] if do_fim else k[0]]
    tt = tempos[k] - ref
    grau = min(2, len(k) - 1)
    M = np.stack([tt ** g for g in range(grau + 1)], 1)
    coef, *_ = np.linalg.lstsq(M, P, rcond=None)
    dt = t_alvo - ref
    return sum(coef[g] * dt ** g for g in range(grau + 1))


def religar(segs, tempos, esc, eventos):
    """PERMANENCIA: fins e comecos de trilhas competem por atribuicao otima; custo = o
    quanto cada uma, estendida no tempo, erra a outra (nos dois sentidos)."""
    segs = sorted(segs, key=lambda T: T["k"][0])
    for _volta in range(3):
        ids = [i for i, T in enumerate(segs) if len(T["k"]) >= 3]
        if len(ids) < 2:
            break
        C = np.full((len(ids), len(ids)), 1e6)
        for a, i in enumerate(ids):
            A = segs[i]
            for b, j in enumerate(ids):
                B = segs[j]
                if i == j or B["k"][0] <= A["k"][-1]:
                    continue
                t0, t1 = tempos[A["k"][-1]], tempos[B["k"][0]]
                if not 0 < t1 - t0 <= LACUNA_S:
                    continue
                e1 = np.linalg.norm(_extrapola(A, tempos, t1, True) - B["p"][0])
                e2 = np.linalg.norm(_extrapola(B, tempos, t0, False) - A["p"][-1])
                # a incerteza cresce com o caminho percorrido escondido: o erro de extrapolar
                # e medido contra (8 px + metade desse caminho)
                k_a = A["k"][-min(len(A["k"]), 4):]
                vel = np.linalg.norm(A["p"][-1] - A["p"][-len(k_a)]) / max(tempos[k_a[-1]] - tempos[k_a[0]], 1e-3)
                C[a, b] = ((e1 + e2) / 2) / (8.0 * esc + 0.5 * vel * (t1 - t0))
        if C.min() > 1.0:
            break
        ra, rb = linear_sum_assignment(C)
        juntar = {ids[a]: ids[b] for a, b in zip(ra, rb) if C[a, b] <= 1.0}
        if not juntar:
            break
        eventos["religar"] = eventos.get("religar", 0) + len(juntar)
        usados, novos = set(), []
        for i, T in enumerate(segs):
            if i in usados or i in juntar.values():
                continue
            cad, x = dict(T), i
            while x in juntar:
                B = segs[juntar[x]]
                n0 = len(cad["k"])
                cad = {"k": np.concatenate([cad["k"], B["k"]]), "p": np.concatenate([cad["p"], B["p"]]),
                       "a": np.concatenate([cad["a"], B["a"]]), "cor": cad["cor"],
                       "quebras": list(cad["quebras"]) + [n0] + [n0 + q for q in B["quebras"]]}
                usados.add(juntar[x])
                x = juntar[x]
            novos.append(cad)
        segs = sorted(novos, key=lambda T: T["k"][0])
    return segs


def percebe(quadros, tempos, religa=True):
    """quadros: sequencia de imagens (qualquer tamanho; RGB uint8 ou cinza [0,1]).
    -> dict(trilhas=[{t, p (normalizado), quebras, k}], L, motivo, eventos)"""
    tempos = np.asarray(tempos, float)
    qs = [reduz(q) for q in quadros]
    if len(qs) < MIN_PONTOS:
        return {"trilhas": [], "motivo": "curto", "eventos": {}, "L": 0}
    forma = qs[0].shape
    if max(forma[:2]) < 48:
        return {"trilhas": [], "motivo": "olho_pequeno", "eventos": {}, "L": 0}
    from PIL import Image
    qs = [q if q.shape == forma else np.asarray(Image.fromarray(q).resize((forma[1], forma[0])), np.uint8) for q in qs]
    Q = np.stack(qs)
    amostra = Q[np.linspace(0, len(Q) - 1, min(60, len(Q))).astype(int)]
    fundo = np.median(amostra, 0).astype(np.uint8)
    dif = np.abs(amostra.astype(np.int16) - fundo.astype(np.int16)).sum(3)
    mad = float(np.median(np.abs(dif - np.median(dif)))) * 1.4826
    limiar = max(LIMIAR_FG, 6 * mad)
    L = max(forma[:2])
    esc = L / 128.0
    frs, fr_fg = [], []
    for q in Q:
        f, x = fragmentos(q, fundo, limiar)
        frs.append(f)
        fr_fg.append(x)
    if np.median(fr_fg) > FRACAO_FUNDO_MAX:
        return {"trilhas": [], "motivo": "fundo_instavel", "eventos": {}, "L": L}
    eventos = {}
    tr = rastreia(frs, tempos, esc)
    tr = cortar(tr, tempos, esc, eventos)
    if religa:
        tr = religar(tr, tempos, esc, eventos)
    centro = np.array([forma[1] / 2, forma[0] / 2])
    out = []
    for T in tr:
        if len(T["k"]) < MIN_PONTOS:
            continue
        t = tempos[T["k"]]
        vel = np.linalg.norm(np.diff(T["p"], axis=0), axis=1) / np.maximum(np.diff(t), 1e-3)
        if np.median(vel) < 1.0 * esc:
            continue                                   # parado: e fundo que nao foi pego
        out.append({"t": t.astype(np.float32), "p": ((T["p"] - centro) / (L / 2)).astype(np.float32),
                    "quebras": [int(x) for x in T["quebras"]], "k": T["k"].astype(np.int32),
                    "cor": [int(c) for c in T["cor"]]})
    return {"trilhas": out, "motivo": "ok" if out else "sem_objetos", "eventos": eventos, "L": L,
            "forma": [int(forma[0]), int(forma[1])]}


# ═════════════════════════════════════════════════════════════════════════════
# 2. ESTADO: posicao, velocidade e aceleracao pelo tempo real
# ═════════════════════════════════════════════════════════════════════════════
def _sem_lacuna(t, a, b):
    dt = np.diff(t[a:b])
    if len(dt) == 0:
        return False
    return dt.max() <= 2.5 * max(np.median(np.diff(t)), 1e-3)


def estados(tr):
    """-> Z (n,4), A (n,2), ok (n,) pelo ajuste quadratico local SIMETRICO (para leis e conceitos)."""
    t, P, qb = tr["t"].astype(float), tr["p"].astype(float), tr["quebras"]
    n = len(t)
    Z, A, ok = np.zeros((n, 4)), np.zeros((n, 2)), np.zeros(n, bool)
    meia = JANELA // 2
    for i in range(n):
        a, b = max(0, i - meia), min(n, i + meia + 1)
        for q in qb:
            if a < q <= i:
                a = q
            if i < q < b:
                b = q
        if b - a < 5 or not _sem_lacuna(t, a, b):
            continue
        tt = t[a:b] - t[i]
        M = np.stack([np.ones_like(tt), tt, tt ** 2], 1)
        coef, *_ = np.linalg.lstsq(M, P[a:b], rcond=None)
        Z[i] = [coef[0, 0], coef[0, 1], coef[1, 0], coef[1, 1]]
        A[i] = 2 * coef[2]
        ok[i] = True
    return Z, A, ok


def estados_causais(tr):
    """Estado so com o PASSADO (para prever): nao atravessa juncoes de permanencia."""
    t, P, qb = tr["t"].astype(float), tr["p"].astype(float), tr["quebras"]
    n = len(t)
    Z, ok = np.zeros((n, 4)), np.zeros(n, bool)
    for i in range(n):
        a = max(0, i - JANELA + 1)
        for q in qb:
            if a < q <= i:
                a = q
        if i + 1 - a < 4 or not _sem_lacuna(t, a, i + 1):
            continue
        tt = t[a:i + 1] - t[i]
        M = np.stack([np.ones_like(tt), tt, tt ** 2], 1)
        coef, *_ = np.linalg.lstsq(M, P[a:i + 1], rcond=None)
        Z[i] = [coef[0, 0], coef[0, 1], coef[1, 0], coef[1, 1]]
        ok[i] = True
    return Z, ok


def pontos_de_previsao(tr, Zc, okc, max_por=40):
    """Partidas (estado causal em t) e chegadas (posicao observada em t + ~2 s): a coisa
    pode ter sumido atras de algo no caminho — a permanencia e parte do teste."""
    t = tr["t"].astype(float)
    alvo_t = t + HORIZONTE
    j = np.searchsorted(t, alvo_t)
    sel = []
    for i in np.nonzero(okc)[0]:
        cands = [x for x in (j[i] - 1, j[i]) if 0 <= x < len(t)]
        if not cands:
            continue
        x = min(cands, key=lambda x: abs(t[x] - alvo_t[i]))
        if abs(t[x] - alvo_t[i]) <= 0.11 and x > i:
            sel.append((i, x))
    if len(sel) > max_por:
        sel = [sel[int(round(u))] for u in np.linspace(0, len(sel) - 1, max_por)]
    if not sel:
        return np.zeros((0, 4)), np.zeros((0, 2)), np.zeros(0)
    ii = np.array([a for a, _ in sel])
    xx = np.array([b for _, b in sel])
    return Zc[ii], tr["p"][xx].astype(float), (t[xx] - t[ii])


# ═════════════════════════════════════════════════════════════════════════════
# 3. ALFABETO, CONCEITOS e LEIS
# ═════════════════════════════════════════════════════════════════════════════
def monomios(Z):
    x, y, vx, vy = Z[:, 0], Z[:, 1], Z[:, 2], Z[:, 3]
    return np.stack([x, y, vx, vy, x * x, x * y, x * vx, x * vy, y * y, y * vx, y * vy,
                     vx * vx, vx * vy, vy * vy], 1)


def valor_conceito(c, Z):
    return ((((monomios(Z) - c["mu"]) / c["sd"]) @ c["w"]) - c["m0"]) / c["esc"]


def phi(Z, qv=None):
    base = np.concatenate([np.ones((len(Z), 1)), monomios(Z)], 1)
    if qv is None:
        return base
    x, y, vx, vy = Z[:, 0], Z[:, 1], Z[:, 2], Z[:, 3]
    q, q2 = qv, qv * qv
    return np.concatenate([base, np.stack([q, q2, q * x, q * y, q * vx, q * vy, q2 * x, q2 * y, q2 * vx, q2 * vy], 1)], 1)


def _limpa(Z, A):
    m = np.linalg.norm(A, axis=1)
    med = np.median(m)
    mad = np.median(np.abs(m - med)) * 1.4826 + 1e-9
    ok = m <= med + 8 * mad
    return Z[ok], A[ok]


def ajusta(Z, A, qv=None, termos=None, max_termos=MAX_TERMOS):
    """Lei = aceleracao como combinacao de termos. Sem `termos`: selecao gulosa conjunta
    (os dois eixos), parando quando os bits (BIC) param de cair."""
    X = phi(Z, qv)
    mu = X.mean(0)
    sd = X.std(0)
    mu[0], sd[0] = 0.0, 1.0
    sd[sd < 1e-9] = 1.0
    Xs = (X - mu) / sd
    N = len(Xs)
    G = Xs.T @ Xs
    b = Xs.T @ A
    yy = (A ** 2).sum(0)

    def rss(S):
        if not S:
            return yy.copy(), np.zeros((0, 2))
        Gs = G[np.ix_(S, S)] + 1e-8 * N * np.eye(len(S))
        W = np.linalg.solve(Gs, b[S])
        return np.maximum(yy - (b[S] * W).sum(0), 1e-12 * N), W

    if termos is None:
        S, (r0, _w) = [], rss([])
        bic = N * np.log(r0 / N).sum()
        while len(S) < max_termos:
            melhor = None
            for j in range(Xs.shape[1]):
                if j in S:
                    continue
                r, _w = rss(S + [j])
                bj = N * np.log(r / N).sum() + 2 * (len(S) + 1) * math.log(N)
                if melhor is None or bj < melhor[0]:
                    melhor = (bj, j)
            if melhor is None or melhor[0] >= bic:
                break
            bic = melhor[0]
            S.append(melhor[1])
        termos = S
    _r, W = rss(list(termos))
    return {"termos": list(termos), "W": W, "mu": mu, "sd": sd}


def lei_nula():
    return {"termos": [], "W": np.zeros((0, 2)), "mu": np.zeros(15), "sd": np.ones(15), "conceito": None}


def acel_lei(lei, Z, conceitos):
    """aceleracao prevista; `lei` pode ter correcoes empilhadas (braco CONCLUSOES)."""
    if lei is None:
        return np.zeros((len(Z), 2))
    c = lei.get("conceito")
    qv = valor_conceito(conceitos[c], Z) if c is not None else None
    if qv is not None and lei.get("q_faixa"):
        qv = np.clip(qv, *lei["q_faixa"])          # fora do que ja viu, o conceito nao inventa valores
    X = (phi(Z, qv) - lei["mu"]) / lei["sd"]
    a = X[:, lei["termos"]] @ lei["W"] if lei["termos"] else np.zeros((len(Z), 2))
    for cor in lei.get("correcoes", []):
        a = a + acel_lei(cor, Z, conceitos)
    return np.clip(np.nan_to_num(a), -50, 50)


def rola(lei, conceitos, Z0, dts):
    if not len(Z0):
        return np.zeros((0, 2))
    z = Z0.astype(float).copy()
    n = max(1, int(math.ceil(float(np.max(dts)) / DT_ROLA)))
    h = (dts / n)[:, None]
    for _ in range(n):
        if lei is not None:
            z[:, 2:] += acel_lei(lei, z, conceitos) * h
        z[:, :2] += z[:, 2:] * h
        z = np.clip(z, -20, 20)
    return z[:, :2]


def formula_lei(lei, conceitos=None):
    if lei is None:
        return "inércia (nada acelera)"
    nomes = NOMES_BASE + NOMES_CONC
    partes = []
    for eixo, nome in ((0, "ax"), (1, "ay")):
        ts = []
        for j, w in zip(lei["termos"], lei["W"][:, eixo]):
            coef = w / lei["sd"][j]
            if abs(w) > 0.02 * (np.abs(lei["W"][:, eixo]).max() + 1e-12):
                ts.append(f"{coef:+.2f}·{nomes[j]}")
        partes.append(f"{nome} = " + (" ".join(ts) if ts else "0"))
    s = " ; ".join(partes)
    if lei.get("correcoes"):
        s += f" (+{len(lei['correcoes'])} correções empilhadas)"
    return s


def formula_conceito(c):
    w = c["w"] / c["sd"]
    w = np.where(c["w"] != 0, w, 0.0)
    w = w / (np.abs(w).max() + 1e-12)
    return " ".join(f"{v:+.2f}·{n}" for v, n in sorted(zip(w, NOMES_MON), key=lambda z: -abs(z[0])) if abs(v) >= 0.1)


# ── inventar a grandeza (inventa.py / descoberta.py, sobre o estado DELA) ────
def _lam(Q):
    dentro = sum(((q - q.mean()) ** 2).sum() for q in Q)
    todos = np.concatenate(Q)
    tot = ((todos - todos.mean()) ** 2).sum()
    return float(dentro / tot) if tot > 0 else 1.0


def _lam_med(Q):
    todos = np.concatenate(Q)
    tot = todos.var()
    return float(np.median([q.var() for q in Q]) / tot) if tot > 0 else 1.0


def _poupanca(lam, N, ntraj):
    return 0.5 * N * math.log2(1 / max(lam, 1e-300)) - 0.5 * ntraj * math.log2(max(N / ntraj, 1.0))


def _estrutura(N):
    return math.log2(K_CONCEITOS) + 0.5 * (len(NOMES_MON) - 1) * math.log2(max(N, 2))


def inventa(trajs, rng, placebo=False, k=K_CONCEITOS, kmax=4):
    """trajs: lista de Z (n_i,4). -> ate k candidatos DISTINTOS [{mu, sd, w, m0, esc, lam, lam_fora, termos}].

    Como novaphys/inventa.py (10/10): combinacoes ESPARSAS de ate 4 monomios, cada uma
    pelo menor autovetor generalizado (variancia dentro das trajetorias / total). A 5
    quadros/s a velocidade e ruidosa e a forma conjunta de 14 termos espalhava peso
    em termos que nao importam (checagem dev, 09/10). Regua: o que se conserva em
    trajetorias FORA do ajuste, pela MEDIANA entre trajetorias (uma trajetoria ruim
    nao decide), em bits contados por TRAJETORIA (pontos da mesma trajetoria nao sao
    independentes), menos o custo de escolher os termos.
    placebo: os pontos sao SORTEADOS entre as trajetorias antes (nada pode ser constante
    por trajetoria) e o melhor candidato e devolvido sem a regua — mesmo alfabeto,
    mesmo procedimento, sem a estrutura."""
    import itertools
    trajs = [Z for Z in trajs if len(Z) >= MIN_PONTOS_INVENTA]
    if len(trajs) < MIN_TRAJ_INVENTA:
        return []
    if placebo:
        tam = [len(Z) for Z in trajs]
        todos = np.concatenate(trajs)[rng.permutation(sum(tam))]
        trajs = np.split(todos, np.cumsum(tam)[:-1])
    ordem = rng.permutation(len(trajs))
    tr = [trajs[i] for i in ordem[0::2]]
    ho = [trajs[i] for i in ordem[1::2]]
    m = len(NOMES_MON)
    G = np.concatenate([monomios(Z) for Z in tr])
    mu, sd = G.mean(0), G.std(0)
    sd[sd < 1e-9] = 1.0
    Fs = [(monomios(Z) - mu) / sd for Z in tr]
    Fh = [(monomios(Z) - mu) / sd for Z in ho]
    Gs = np.concatenate(Fs)
    Gc = Gs - Gs.mean(0)
    Tm = Gc.T @ Gc / len(Gs)
    Wm = sum((f - f.mean(0)).T @ (f - f.mean(0)) for f in Fs) / len(Gs)

    def lam_med(F, w, S):
        """fracao da variancia que fica DENTRO das trajetorias: media das trajetorias, sem as
        20% piores (uma trajetoria mal percebida nao decide; mas conservar-se so na MAIORIA
        — como o centro do circulo de quem gira numa velocidade parecida — nao basta)."""
        qs = [f[:, S] @ w for f in F]
        tot = np.concatenate(qs).var()
        if tot <= 0:
            return 1.0
        r = np.sort([q.var() for q in qs])
        return float(r[:max(1, int(math.ceil(0.8 * len(r))))].mean() / tot)

    cands = []
    for kk in range(1, kmax + 1):
        custo = math.log2(math.comb(m, kk)) + math.log2(kmax)
        for S in itertools.combinations(range(m), kk):
            S = list(S)
            Ts = Tm[np.ix_(S, S)]
            ev = np.linalg.eigvalsh(Ts)
            if ev.min() <= 1e-9 * max(ev.max(), 1e-300):
                continue
            try:
                _vals, vecs = eigh(Wm[np.ix_(S, S)], Ts)
            except LinAlgError:
                continue
            w = vecs[:, 0]
            lt, lh = lam_med(Fs, w, S), lam_med(Fh, w, S)
            bt = 0.5 * len(tr) * math.log2(1 / max(lt, 1e-300)) - custo
            bh = 0.5 * len(ho) * math.log2(1 / max(lh, 1e-300)) - custo
            cands.append((bh, bt, S, w, lt, lh))
    cands.sort(key=lambda c: -c[0])
    Zh = np.concatenate(ho)
    out, vistos = [], []
    for bh, bt, S, w, lt, lh in cands:
        if len(out) >= (1 if placebo else k):
            break
        if not placebo and (bh <= 0 or bt <= 0):
            break
        wf = np.zeros(m)
        wf[S] = w
        qh = ((monomios(Zh) - mu) / sd) @ wf
        if any(abs(np.corrcoef(qh, v)[0, 1]) > 0.95 for v in vistos):
            continue
        vistos.append(qh)
        qt = Gs @ wf
        out.append({"mu": mu, "sd": sd, "w": wf, "m0": float(qt.mean()), "esc": float(qt.std() + 1e-12),
                    "lam": float(lt), "lam_fora": float(lh), "bits_fora": float(bh),
                    "termos": [NOMES_MON[i] for i in S], "placebo": bool(placebo)})
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 4. O LABORATORIO: hipocampo, teorias, crise, consenso, sala de espera
# ═════════════════════════════════════════════════════════════════════════════
class Laboratorio:
    """bracos (so para o experimento; a mente viva usa o COMPLETO):
      olhos_frescos  na crise e na sala de espera a lei e RE-DERIVADA do cru; sem isso
                     a lei de nascimento fica e o resto vira correcao empilhada
      reinterpreta   episodios guardados podem mudar de teoria (consenso)
      inventa        True | False | "placebo"
      transfere      conceitos do vocabulario servem a qualquer teoria"""

    def __init__(self, semente=0, olhos_frescos=True, reinterpreta=True, inventa=True, transfere=True):
        self.rng = np.random.default_rng(semente)
        self.cfg = {"olhos_frescos": olhos_frescos, "reinterpreta": reinterpreta, "inventa": inventa,
                    "transfere": transfere}
        self.eps = {}                  # id -> episodio CRU (+ a teoria que o explica hoje)
        self.ordem = []                # ids em ordem de chegada
        self.teorias = {}              # id -> teoria
        self.conceitos = {}            # id -> conceito (vocabulario)
        self.n_ep = self.n_teo = self.n_conc = 0
        self.contas = {"episodios": 0, "sem_objetos": 0, "crises": 0, "explicados": 0, "reinterpretados": 0,
                       "conceitos_inventados": 0, "conceitos_reusados": 0, "trocas_de_conceito": 0,
                       "teorias_mortas": 0, "religacoes": 0, "conceitos_revisados": 0,
                       "hipoteses_abandonadas": 0}
        self.recentes = []             # eventos para o painel
        self._cache = {}
        self.ev = None                 # registro externo (ontogenia), opcional

    # ── persistencia: o cache e re-derivavel ──────────────────────────────────
    def __getstate__(self):
        d = dict(self.__dict__)
        d["_cache"] = {}
        d["ev"] = None
        return d

    def _log(self, nome, **kw):
        kw["quando"] = time.time()
        self.recentes = (self.recentes + [dict(ev=nome, **kw)])[-60:]
        if self.ev is not None:
            try:
                self.ev("laboratorio_" + nome, **kw)
            except Exception:
                pass

    # ── derivados de um episodio (sempre do cru) ──────────────────────────────
    def _der(self, eid):
        d = self._cache.get(eid)
        if d is not None:
            return d
        ep = self.eps[eid]
        Zs, As, trajs, P0, PA, PD, dono, por = [], [], [], [], [], [], [], []
        for it, tr in enumerate(ep["trilhas"]):
            Z, A, ok = estados(tr)
            por.append((Z[ok], A[ok]))
            if ok.sum():
                Zs.append(Z[ok])
                As.append(A[ok])
                trajs.append(Z[ok])
            Zc, okc = estados_causais(tr)
            z0, alvo, dt = pontos_de_previsao(tr, Zc, okc)
            P0.append(z0)
            PA.append(alvo)
            PD.append(dt)
            dono.append(np.full(len(z0), it))
        d = {"Z": np.concatenate(Zs) if Zs else np.zeros((0, 4)), "A": np.concatenate(As) if As else np.zeros((0, 2)),
             "trajs": trajs, "z0": np.concatenate(P0) if P0 else np.zeros((0, 4)),
             "alvo": np.concatenate(PA) if PA else np.zeros((0, 2)), "dt": np.concatenate(PD) if PD else np.zeros(0),
             "trilha": np.concatenate(dono) if dono else np.zeros(0, int), "por_trilha": por}
        self._cache[eid] = d
        return d

    def _erros_pts(self, lei, eids):
        """erro (px em 128) de cada ponto de previsao, para varios episodios de uma vez."""
        ds = [self._der(e) for e in eids]
        z0 = np.concatenate([d["z0"] for d in ds]) if ds else np.zeros((0, 4))
        if not len(z0):
            return [np.zeros(0) for _ in eids]
        alvo = np.concatenate([d["alvo"] for d in ds])
        dt = np.concatenate([d["dt"] for d in ds])
        e = np.linalg.norm(rola(lei, self.conceitos, z0, dt) - alvo, axis=1) * PX
        return np.split(e, np.cumsum([len(d["z0"]) for d in ds])[:-1])

    def _erros_traj(self, lei, eids):
        """por episodio: o erro mediano de CADA trajetoria (os pontos de uma trajetoria nao
        sao independentes; a trajetoria e a unidade)."""
        out = []
        for e, pts in zip(eids, self._erros_pts(lei, eids)):
            d = self._der(e)
            out.append(np.array([np.median(pts[d["trilha"] == it]) for it in np.unique(d["trilha"])]) if len(pts)
                       else np.zeros(0))
        return out

    def _erro_ep(self, lei, eids):
        """erro do episodio em px (128) = MEDIANA dos erros das suas trajetorias (descritivo)."""
        return [float(np.median(x)) if len(x) else None for x in self._erros_traj(lei, eids)]

    def _inercia_traj(self, e):
        d = self._der(e)
        if "e0" not in d:
            d["e0"] = np.maximum(self._erros_traj(None, [e])[0], 1e-3)
        return d["e0"]

    def _razao_ep(self, lei, eids):
        """erro RELATIVO do episodio: mediana, entre as trajetorias, de (erro da lei / erro de
        'nada acelera'). Sem escala: num video em que tudo anda depressa, errar mais pixels
        nao e crise; mudar de mundo muda todas as trajetorias; um objeto mal rastreado, uma."""
        out = []
        for e, x in zip(eids, self._erros_traj(lei, eids)):
            out.append(float(np.median(x / self._inercia_traj(e))) if len(x) else None)
        return out

    # ── leis de uma teoria ────────────────────────────────────────────────────
    def _dados(self, eids, conceito, trilhas=None):
        Z, A = [], []
        for e in eids:
            d = self._der(e)
            if not len(d["Z"]):
                continue
            Z.append(d["Z"])
            A.append(d["A"])
        if not Z:
            return None, None, None
        Z, A = _limpa(np.concatenate(Z), np.concatenate(A))
        qv = valor_conceito(self.conceitos[conceito], Z) if conceito is not None else None
        return Z, A, qv

    def _lei(self, eids, conceito, termos=None):
        Z, A, qv = self._dados(eids, conceito)
        if Z is None or len(Z) < 8:
            return lei_nula()
        lei = ajusta(Z, A, qv, termos)
        lei["conceito"] = conceito
        if qv is not None:
            lei["q_faixa"] = (float(qv.min()), float(qv.max()))
        return lei

    def _dobras(self, eids):
        """duas metades para medir FORA da amostra: por episodio; com um so, por trilha."""
        if len(eids) >= 2:
            ordem = list(eids)
            return [(ordem[0::2], ordem[1::2]), (ordem[1::2], ordem[0::2])]
        return None

    def _fora_da_amostra(self, eids, conceito):
        """erros por TRAJETORIA (mediana dos pontos) da lei ajustada numa metade e medida
        na outra. Com um episodio so, as metades sao as trilhas dele."""
        dob = self._dobras(eids)
        if dob is not None:
            errs = []
            for a, b in dob:
                lei = self._lei(a, conceito)
                for e, pts in zip(b, self._erros_pts(lei, b)):
                    d = self._der(e)
                    e0 = self._inercia_traj(e)
                    for j, it in enumerate(np.unique(d["trilha"])):
                        x = pts[d["trilha"] == it]
                        if len(x):
                            errs.append((float(np.median(x)), x, float(e0[j])))
            return errs
        e = eids[0]
        d = self._der(e)
        its = np.unique(d["trilha"])
        if len(its) < 2 or not len(d["Z"]):
            return []
        # por trilha: a lei de umas trilhas preve as outras
        errs = []
        e0 = self._inercia_traj(e)
        metades = [its[0::2], its[1::2]]
        for a, b in (metades, metades[::-1]):
            Z = np.concatenate([d["por_trilha"][it][0] for it in a])
            A = np.concatenate([d["por_trilha"][it][1] for it in a])
            if len(Z) < 8:
                continue
            Z, A = _limpa(Z, A)
            qv = valor_conceito(self.conceitos[conceito], Z) if conceito is not None else None
            lei = ajusta(Z, A, qv)
            lei["conceito"] = conceito
            if qv is not None:
                lei["q_faixa"] = (float(qv.min()), float(qv.max()))
            m = np.isin(d["trilha"], b)
            if not m.any():
                continue
            pts = np.linalg.norm(rola(lei, self.conceitos, d["z0"][m], d["dt"][m]) - d["alvo"][m], axis=1) * PX
            for it in b:
                x = pts[d["trilha"][m] == it]
                if len(x):
                    errs.append((float(np.median(x)), x, float(e0[int(np.nonzero(its == it)[0][0])])))
        return errs

    def _ruido(self, T):
        """o ruido PROPRIO da teoria: erros RELATIVOS de trajetorias fora da amostra (ou, sem
        como medir, dentro)."""
        errs = self._fora_da_amostra(T["membros"], T["lei"].get("conceito") if T["lei"] else None)
        if errs:
            r = np.array([m / max(e0, 1e-3) for m, _x, e0 in errs])
            px = np.array([m for m, _x, _e0 in errs])
        else:
            ets = self._erros_traj(T["lei"], T["membros"])
            r = np.concatenate([x / self._inercia_traj(e) for e, x in zip(T["membros"], ets)] or [np.zeros(0)])
            px = np.concatenate(ets or [np.zeros(0)])
        T["ruido"] = r.astype(np.float32)[-2000:]
        T["ruido_px"] = px.astype(np.float32)[-2000:]
        T["q99"] = {}

    def _limiar(self, T, m, chave="ruido"):
        """o erro (relativo, ou em px com chave="ruido_px") de um episodio com m trajetorias
        que o proprio ruido da teoria (trajetorias FORA da amostra) passaria em 1% das vezes."""
        r = T.get(chave)
        piso = PISO_RAZAO if chave == "ruido" else PISO_ERRO_PX
        if r is None or len(r) < 2 or m <= 0:
            return piso
        mb = int(min(m, 50))
        c = T.setdefault("q99", {})
        if (chave, mb) not in c:
            amostras = self.rng.choice(r, (400, mb), replace=True)
            c[(chave, mb)] = float(np.quantile(np.median(amostras, 1), 1 - ALARME))
        return max(c[(chave, mb)], piso)

    def _fresca(self, e, conceito):
        """o que os dados do episodio, SOZINHOS, preveem: a lei de umas trilhas dele prevendo as
        outras (fora da amostra). -> {trilha: erro mediano px}. Cache por (episodio, conceito)."""
        d = self._der(e)
        chave = ("fresca", conceito)
        if chave in d:
            return d[chave]
        out = {}
        its = np.unique(d["trilha"])
        if len(its) >= 2 and len(d["Z"]):
            metades = [its[0::2], its[1::2]]
            for a, b in (metades, metades[::-1]):
                Z = np.concatenate([d["por_trilha"][it][0] for it in a])
                A = np.concatenate([d["por_trilha"][it][1] for it in a])
                if len(Z) < 8:
                    continue
                Z, A = _limpa(Z, A)
                qv = valor_conceito(self.conceitos[conceito], Z) if conceito is not None else None
                lei = ajusta(Z, A, qv)
                lei["conceito"] = conceito
                if qv is not None:
                    lei["q_faixa"] = (float(qv.min()), float(qv.max()))
                m = np.isin(d["trilha"], b)
                if not m.any():
                    continue
                pts = np.linalg.norm(rola(lei, self.conceitos, d["z0"][m], d["dt"][m]) - d["alvo"][m], axis=1) * PX
                for it in b:
                    x = pts[d["trilha"][m] == it]
                    if len(x):
                        out[int(it)] = float(np.median(x))
        d[chave] = out
        return out

    def _bits_contra(self, T, e, et=None):
        """bits com que o episodio SOZINHO preve melhor que a teoria T: soma por trajetoria de
        log2(erro da teoria / erro do episodio sozinho), menos o custo de abrir uma teoria nova."""
        fr = self._fresca(e, T["lei"].get("conceito") if T["lei"] else None)
        if not fr:
            return -np.inf
        its = np.unique(self._der(e)["trilha"])
        if et is None:
            et = self._erros_traj(T["lei"], [e])[0]
        b = sum(math.log2(max(et[j], 0.05) / max(fr[int(it)], 0.05)) for j, it in enumerate(its) if int(it) in fr)
        return b - math.log2(len(self.teorias) + 1)

    def _explica(self, T, e, r=None, px=None, et=None):
        """A teoria EXPLICA o episodio se faz melhor que 'nada acelera' e o proprio episodio,
        sozinho, NAO a supera fora da amostra. Crise = os meus dados novos, sem a minha teoria,
        preveem melhor do que com ela (sem limiar meu: so a regua de bits)."""
        if px is None:
            px = self._erro_ep(T["lei"], [e])[0]
        if px is None:
            return False
        ine = self.eps[e].get("inercia")
        if ine is None:
            ine = self._erro_ep(None, [e])[0]
        if not px < ine or self._bits_contra(T, e, et) > 0:
            return False
        # e o erro nao e anomalo para o ruido da propria teoria nas DUAS escalas ao mesmo tempo
        # (pixels E relativo a inercia): um video rapido erra mais pixels, um lento oscila mais
        # em termos relativos; outro mundo erra mais nas duas
        if r is None:
            r = self._razao_ep(T["lei"], [e])[0]
        m = self._m(e)
        return not (r is not None and r > self._limiar(T, m) and px > self._limiar(T, m, "ruido_px"))

    def _m(self, e):
        return len(np.unique(self._der(e)["trilha"]))

    def _opcoes(self, T):
        if not self.cfg["transfere"]:
            return [None] + [c for c, x in self.conceitos.items() if T["id"] in x["teorias"]]
        return [None] + list(self.conceitos)

    @staticmethod
    def _bits_pareados(base, outro, n_ops):
        """bits de uma opcao contra "nenhum conceito": soma, por TRAJETORIA medida fora da
        amostra, de log2(erro sem / erro com) — comparacao pareada (a mesma trajetoria nos
        dois), robusta a uma trajetoria que explode — menos o custo de ter escolhido."""
        eb = np.array([x[0] for x in base])
        ec = np.array([x[0] for x in outro])
        if not len(eb) or len(eb) != len(ec):
            return -np.inf
        return float(np.sum(np.log2(np.maximum(eb, 0.05) / np.maximum(ec, 0.05)))) - math.log2(max(n_ops, 2))

    def _escolhe_conceito(self, T, membros=None):
        """SALA DE ESPERA: nenhum conceito, ou um do vocabulario — o que mais reduz o erro da
        previsao FORA da amostra, trajetoria a trajetoria. Um conceito so entra se os bits
        pagarem o custo de ter escolhido entre as opcoes. Conceito que a lei nem usa nao e
        opcao (seria "nenhum" com outro nome)."""
        membros = T["membros"] if membros is None else membros
        ops = self._opcoes(T)
        base = self._fora_da_amostra(membros, None)
        tabela = {None: (float(np.median([x[0] for x in base])) if base else None, 0.0)}
        escolha, melhor = None, 0.0
        for c in ops:
            if c is None:
                continue
            lei_c = self._lei(membros, c)
            if not any(j >= len(NOMES_BASE) for j in lei_c["termos"]):
                continue
            errs = self._fora_da_amostra(membros, c)
            bits = self._bits_pareados(base, errs, len(ops))
            tabela[c] = (float(np.median([x[0] for x in errs])) if errs else None, round(bits, 1))
            if bits > melhor:
                escolha, melhor = c, bits
        return escolha, tabela

    def _refaz(self, T, conceito, selecao=True):
        if selecao or T["lei"] is None:
            T["lei"] = self._lei(T["membros"], conceito)
        else:
            T["lei"] = self._lei(T["membros"], T["lei"]["conceito"], T["lei"]["termos"])
        self._ruido(T)

    # ── conceitos ─────────────────────────────────────────────────────────────
    def _inventar(self, T):
        modo = self.cfg["inventa"]
        if not modo:
            return []
        trajs = [z for e in T["membros"] for z in self._der(e)["trajs"]]
        cands = inventa(trajs, self.rng, placebo=(modo == "placebo"))
        if not cands:
            return []
        Zref = np.concatenate(trajs)
        if len(Zref) > 4000:
            Zref = Zref[self.rng.choice(len(Zref), 4000, replace=False)]
        novos = []
        for c in cands:
            qc = valor_conceito(c, Zref)
            dup = None
            for cid, x in self.conceitos.items():
                if not self.cfg["transfere"] and T["id"] not in x["teorias"]:
                    continue
                r = np.corrcoef(qc, valor_conceito(x, Zref))[0, 1]
                if np.isfinite(r) and abs(r) > 0.95:
                    dup = cid
                    break
            if dup is not None:
                x = self.conceitos[dup]
                x["teorias"].add(T["id"])
                self.contas["conceitos_reusados"] += 1
                # a MESMA ideia, medida com mais trajetorias: se ela se conserva melhor aqui do
                # que a forma antiga, o conceito e REVISADO (mesmo nome, forma nova) e as leis
                # que o usam sao re-derivadas. Sem olhos frescos, a forma antiga fica.
                l_novo = _lam_med([valor_conceito(c, z) for z in trajs if len(z) >= MIN_PONTOS_INVENTA])
                l_velho = _lam_med([valor_conceito(x, z) for z in trajs if len(z) >= MIN_PONTOS_INVENTA])
                if self.cfg["olhos_frescos"] and not c["placebo"] and l_novo < 0.8 * l_velho:
                    antes = formula_conceito(x)
                    for k_ in ("mu", "sd", "w", "m0", "esc", "lam", "lam_fora"):
                        x[k_] = c[k_]
                    x["revisoes"] = x.get("revisoes", 0) + 1
                    for d_ in self._cache.values():
                        d_.pop(("fresca", dup), None)
                    self.contas["conceitos_revisados"] = self.contas.get("conceitos_revisados", 0) + 1
                    for U in self.teorias.values():
                        if U["lei"] and U["lei"].get("conceito") == dup:
                            self._refaz(U, dup, selecao=True)
                    self._log("conceito_revisado", conceito=dup, teoria=T["id"], antes=antes,
                              agora=formula_conceito(x), lam_antes=round(l_velho, 5), lam_agora=round(l_novo, 5))
                continue
            cid = f"c{self.n_conc}"
            self.n_conc += 1
            c.update(id=cid, nasceu=self.contas["episodios"], de_teoria=T["id"], teorias={T["id"]})
            self.conceitos[cid] = c
            self.contas["conceitos_inventados"] += 1
            novos.append(cid)
            self._log("conceito_inventado", conceito=cid, teoria=T["id"], formula=formula_conceito(c),
                      lam=round(c["lam"], 5), lam_fora=round(c["lam_fora"], 5), placebo=c["placebo"])
        return novos

    def _sala_de_espera(self, T, motivo):
        """inventa (se pode) e refaz a escolha do conceito; se a escolha mudou, re-deriva a lei."""
        self._inventar(T)
        antes = T["lei"].get("conceito") if T["lei"] else None
        escolha, tabela = self._escolhe_conceito(T)
        T["n_escolha"] = len(T["membros"])
        if escolha != antes:
            self.contas["trocas_de_conceito"] += 1
            if self.cfg["olhos_frescos"]:
                self._refaz(T, escolha, selecao=True)
            else:
                self._corrige(T, escolha)
            self._log("conceito_escolhido", teoria=T["id"], de=antes, para=escolha, motivo=motivo,
                      formula=formula_conceito(self.conceitos[escolha]) if escolha else None,
                      lei=formula_lei(T["lei"]),
                      opcoes={str(k): v for k, v in tabela.items()})
            if escolha is not None:
                self.conceitos[escolha]["teorias"].add(T["id"])
            return True
        return False

    def _corrige(self, T, conceito):
        """braco CONCLUSOES: a lei de nascimento FICA (coeficientes e termos); o conceito so
        entra como correcao ajustada no que sobra (o epiciclo mais um remendo)."""
        if conceito is None or T["lei"] is None:
            return
        Z, A, qv = self._dados(T["membros"], conceito)
        if Z is None:
            return
        res = A - acel_lei(T["lei"], Z, self.conceitos)
        cor = ajusta(Z, res, qv)
        cor["conceito"] = conceito
        cor["q_faixa"] = (float(qv.min()), float(qv.max()))
        T["lei"].setdefault("correcoes", []).append(cor)
        self._ruido(T)

    # ── crise e consenso ──────────────────────────────────────────────────────
    def _nova_teoria(self, eid):
        tid = self.n_teo
        self.n_teo += 1
        T = {"id": tid, "membros": [eid], "lei": None, "ruido": None, "q99": {}, "nasceu": self.contas["episodios"],
             "n_escolha": 1, "usada": self.contas["episodios"]}
        self.teorias[tid] = T
        c0 = None
        if len(self.conceitos):
            c0, _tab = self._escolhe_conceito(T, [eid])
        T["lei"] = self._lei([eid], c0)
        self._ruido(T)
        return T

    def _move(self, eid, T, erro):
        ep = self.eps[eid]
        velha = ep.get("teoria")
        if velha == T["id"]:
            ep["erro"] = erro
            return
        if velha is not None and velha in self.teorias:
            V = self.teorias[velha]
            V["membros"] = [e for e in V["membros"] if e != eid]
        ep["teoria"], ep["erro"] = T["id"], erro
        if eid not in T["membros"]:
            T["membros"].append(eid)

    def _consenso(self, T, eid_origem):
        """olhos frescos, comecando pelo que surpreendeu e incorporando os episodios
        GUARDADOS que a lei nova explica abaixo do proprio ruido e melhor do que hoje."""
        if not self.cfg["reinterpreta"]:
            return 0
        guardados = [e for e in self.ordem[-MAX_CONSENSO:] if e != eid_origem and len(self._der(e)["z0"])]
        movidos = {}
        for _it in range(4):
            errs = self._razao_ep(T["lei"], guardados)
            pxs = self._erro_ep(T["lei"], guardados)
            novos = []
            for e, er, px in zip(guardados, errs, pxs):
                if er is None:
                    continue
                atual = self.eps[e].get("erro")
                if self._explica(T, e, er, px) and \
                        (atual is None or self.eps[e].get("teoria") == T["id"] or er < atual):
                    novos.append((e, er))
            alvo = {eid_origem} | {e for e, _ in novos}
            if alvo == set(T["membros"]):
                break
            antes = {e: (self.eps[e].get("teoria"), self.eps[e].get("erro")) for e, _ in novos}
            for e, er in novos:
                movidos[e] = antes[e]
            # quem estava nela e a lei nova ja nao explica volta a ficar sem teoria
            for e in list(T["membros"]):
                if e not in alvo:
                    T["membros"].remove(e)
                    self.eps[e]["teoria"], self.eps[e]["erro"] = None, None
            for e, er in novos:
                self._move(e, T, er)
            self._refaz(T, T["lei"].get("conceito"), selecao=True)
        n_re = 0
        for e, (velha, er0) in movidos.items():
            if self.eps[e].get("teoria") == T["id"] and velha != T["id"]:
                n_re += 1
                self.contas["reinterpretados"] += 1
                if n_re <= 5:
                    self._log("reinterpretou", episodio=e, de_teoria=velha, para_teoria=T["id"],
                              erro_antes=None if er0 is None else round(er0, 2),
                              erro_agora=round(self.eps[e]["erro"], 2))
        self._enterra()
        return n_re

    def _abandona_hipoteses(self, atual):
        """Hipotese (teoria de UM episodio) que nenhum episodio novo confirmou em 3 episodios
        e abandonada: o episodio dela volta a memoria sem teoria (a reanalise pode pega-lo).
        Sem isso, uma hipotese frouxa nascida de uma crise falsa virava sumidouro: com ruido
        largo, 'explicava' mal o mundo seguinte e escondia a crise (checagem dev 102)."""
        for tid in [t for t, T in self.teorias.items() if len(T["membros"]) == 1 and t != atual
                    and self.contas["episodios"] - T["nasceu"] >= 3]:
            e = self.teorias[tid]["membros"][0]
            self.eps[e]["teoria"], self.eps[e]["erro"] = None, None
            del self.teorias[tid]
            self.contas["hipoteses_abandonadas"] = self.contas.get("hipoteses_abandonadas", 0) + 1

    def _enterra(self):
        for tid in [t for t, T in self.teorias.items() if not T["membros"]]:
            del self.teorias[tid]
            self.contas["teorias_mortas"] += 1
        if len(self.teorias) > MAX_TEORIAS:
            for tid in sorted(self.teorias, key=lambda t: (len(self.teorias[t]["membros"]), self.teorias[t]["usada"]))[
                    : len(self.teorias) - MAX_TEORIAS]:
                for e in self.teorias[tid]["membros"]:
                    self.eps[e]["teoria"], self.eps[e]["erro"] = None, None
                del self.teorias[tid]
                self.contas["teorias_mortas"] += 1

    def _esquece(self):
        while len(self.ordem) > MAX_EPISODIOS:
            e = self.ordem.pop(0)
            t = self.eps[e].get("teoria")
            if t in self.teorias:
                T = self.teorias[t]
                T["membros"] = [x for x in T["membros"] if x != e]
            del self.eps[e]
            self._cache.pop(e, None)
        self._enterra()

    # ── o episodio ────────────────────────────────────────────────────────────
    def observa(self, quadros, tempos, fonte="video", url="", t_vida=0, percepcao=None):
        """Um video inteiro que ela assistiu. -> resumo (o que percebeu, o que previu, o que fez)."""
        per = percepcao if percepcao is not None else percebe(quadros, tempos)
        self.contas["religacoes"] += int(per["eventos"].get("religar", 0))
        if not per["trilhas"]:
            self.contas["sem_objetos"] += 1
            return {"motivo": per["motivo"], "trilhas": 0}
        eid = self.n_ep
        self.n_ep += 1
        self.contas["episodios"] += 1
        self.eps[eid] = {"id": eid, "fonte": fonte, "url": str(url)[:160], "quando": time.time(), "t_vida": int(t_vida),
                         "L": per["L"], "forma": per.get("forma"),
                         "trilhas": [{k: tr[k] for k in ("t", "p", "quebras")} for tr in per["trilhas"]],
                         "teoria": None, "erro": None}
        self.ordem.append(eid)
        d = self._der(eid)
        n = len(d["z0"])
        res = {"episodio": eid, "trilhas": len(per["trilhas"]), "pontos": n, "religou": int(per["eventos"].get("religar", 0)),
               "motivo": "ok"}
        if n == 0:
            res["motivo"] = "sem_previsao"
            self._esquece()
            return res
        # 1a TENTATIVA: antes de aprender qualquer coisa deste episodio
        inercia = self._erro_ep(None, [eid])[0]
        cands = sorted(self.teorias.values(), key=lambda T: -T["usada"])[:30]
        errs = [(self._razao_ep(T["lei"], [eid])[0], self._erro_ep(T["lei"], [eid])[0], T) for T in cands]
        errs = [z for z in errs if z[0] is not None]
        melhor = min(errs, key=lambda z: z[0]) if errs else None      # a PREVISAO: a melhor teoria, qualquer
        res.update(inercia=round(inercia, 3), primeira=round(melhor[1] if melhor else inercia, 3),
                   razao=round(melhor[0], 4) if melhor else 1.0, teoria_prevista=melhor[2]["id"] if melhor else None)
        self.eps[eid].update(erro_1a=res["primeira"], inercia=inercia, razao_1a=res["razao"])
        m_tr = self._m(eid)
        expl = [z for z in errs if self._explica(z[2], eid, z[0], z[1])]
        expl = min(expl, key=lambda z: z[0]) if expl else None
        if expl is not None:
            T = expl[2]
            self._move(eid, T, expl[0])
            T["usada"] = self.contas["episodios"]
            self.contas["explicados"] += 1
            res["crise"] = False
            if self.cfg["olhos_frescos"] or not T["lei"].get("correcoes"):
                self._refaz(T, T["lei"]["conceito"], selecao=False)
            if len(T["membros"]) >= 2 * T["n_escolha"]:
                if self.cfg["olhos_frescos"]:
                    self._refaz(T, T["lei"]["conceito"], selecao=True)
                if self._sala_de_espera(T, "evidencia_dobrou"):
                    self._consenso(T, eid)
        else:
            self.contas["crises"] += 1
            res["crise"] = True
            T = self._nova_teoria(eid)
            self.eps[eid]["teoria"] = T["id"]
            self.eps[eid]["erro"] = self._razao_ep(T["lei"], [eid])[0]
            n_re = self._consenso(T, eid)
            if self._sala_de_espera(T, "crise"):
                n_re += self._consenso(T, eid)
            res["reinterpretados"] = n_re
            self._log("crise", episodio=eid, erro_px=res["primeira"], erro_relativo=res["razao"],
                      limiar=round(self._limiar(melhor[2], m_tr), 3) if melhor else None,
                      teoria_que_falhou=melhor[2]["id"] if melhor else None, nova_teoria=T["id"],
                      membros=len(T["membros"]), reinterpretados=n_re, lei=formula_lei(T["lei"]),
                      fonte=fonte)
        res["teoria"] = self.eps[eid].get("teoria")
        self._abandona_hipoteses(res["teoria"])
        T = self.teorias.get(res["teoria"])
        res["conceito"] = T["lei"].get("conceito") if T and T["lei"] else None
        self._esquece()
        return res

    # ── sono: o que ficou pendente ────────────────────────────────────────────
    def sono(self, orcamento=3):
        """No sono, as teorias cuja evidencia dobrou desde a ultima escolha passam pela sala de espera."""
        feitas = 0
        for T in sorted(self.teorias.values(), key=lambda T: -len(T["membros"])):
            if feitas >= orcamento:
                break
            if len(T["membros"]) >= 2 * T["n_escolha"]:
                if self._sala_de_espera(T, "sono"):
                    self._consenso(T, T["membros"][-1])
                feitas += 1
        return feitas

    # ═════════════════════════════════════════════════════════════════════════
    # 5. MAOS: o experimento que ELA escolhe para separar rivais (transplante 11)
    # ═════════════════════════════════════════════════════════════════════════
    def _rng_exp(self, fluxo):
        """sorteios das maos fora do self.rng (que decide o resto do laboratorio)."""
        return np.random.default_rng([self.n_ep, self.n_teo, self.n_conc, int(fluxo)])

    def teoria_atual(self, fonte="arena"):
        """a teoria que explica o episodio mais recente dessa fonte."""
        for e in reversed(self.ordem):
            ep = self.eps[e]
            if ep["fonte"] == fonte and ep.get("teoria") in self.teorias:
                return self.teorias[ep["teoria"]]
        return None

    def rivais(self, fonte="arena", n_boot=EXP_BOOT):
        """As explicacoes que competem HOJE pelo que ela ve nessa fonte: a lei da teoria atual;
        a mesma teoria com cada outra opcao da sala de espera (outro conceito, ou nenhum); e a
        mesma teoria re-ajustada em reamostras das suas trajetorias (o quanto os dados dela
        ainda deixam a lei em aberto). Um comite feito so do que ela tem — nenhum oraculo."""
        T = self.teoria_atual(fonte)
        if T is None or T["lei"] is None:
            return []
        c_at = T["lei"].get("conceito")
        rv = [{"nome": f"T{T['id']}" + (f"·{c_at}" if c_at else ""), "tipo": "atual", "lei": T["lei"], "conceito": c_at}]
        for c in self._opcoes(T):
            if c == c_at:
                continue
            lei = self._lei(T["membros"], c)
            if c is not None and not any(j >= len(NOMES_BASE) for j in lei["termos"]):
                continue
            rv.append({"nome": f"T{T['id']}" + (f"·{c}" if c else "·sem conceito"), "tipo": "conceito", "lei": lei,
                       "conceito": c})
        por = [pt for e in T["membros"] for pt in self._der(e)["por_trilha"] if len(pt[0]) >= 5]
        rng = self._rng_exp(1)
        if len(por) >= 2:
            for b in range(n_boot):
                sel = rng.integers(len(por), size=len(por))
                Z = np.concatenate([por[i][0] for i in sel])
                A = np.concatenate([por[i][1] for i in sel])
                if len(Z) < 8:
                    continue
                Z, A = _limpa(Z, A)
                qv = valor_conceito(self.conceitos[c_at], Z) if c_at is not None else None
                lei = ajusta(Z, A, qv)
                lei["conceito"] = c_at
                if qv is not None:
                    lei["q_faixa"] = (float(qv.min()), float(qv.max()))
                rv.append({"nome": f"reamostra {b + 1}", "tipo": "reamostra", "lei": lei, "conceito": c_at})
        for r in rv:
            r["teoria"] = T["id"]
        return rv

    def _posicoes(self, lei, Z0, horizontes):
        """posicoes previstas por uma lei (a mesma rolagem da previsao) em varios horizontes: (n, h, 2)."""
        z = np.asarray(Z0, float).copy()
        out, t = [], 0.0
        for th in horizontes:
            n = max(1, int(math.ceil((th - t) / DT_ROLA - 1e-9)))
            h = (th - t) / n
            for _ in range(n):
                if lei is not None:
                    z[:, 2:] += acel_lei(lei, z, self.conceitos) * h
                z[:, :2] += z[:, 2:] * h
                z = np.clip(z, -20, 20)
            t = th
            out.append(z[:, :2].copy())
        return np.stack(out, 1)

    def propoe_experimento(self, rivais=None, fonte="arena", n_corpos=EXP_CORPOS, n_cand=EXP_CANDIDATOS,
                           aleatorio=False):
        """MAOS: onde por cada coisa e com que velocidade para que as rivais DISCORDEM o mais
        possivel do que vai acontecer nos 3 s seguintes — o experimento que mais separa as
        explicacoes dela.
        Candidatos = INTERVENCOES em situacoes que ela VIU (o estado de uma coisa num episodio da
        teoria atual, com a rapidez vezes 0,7-1,3 e a direcao girada ate 30 graus ou invertida):
        mudar uma coisa de cada vez num caso real. So valem os que a teoria atual preve que ficam
        no quadro por 6 s, cujo desfecho fica a vista sob TODAS as rivais, que os olhos DELA
        acompanham (nunca mais que meio passo maximo de rastreio por quadro) e que nao se
        encostam no caminho previsto.
        Checagens dev 101-103: (1) candidatos uniformes numa faixa de rapidez que crescia com o
        que ela mesma pedia -> corrida ate rapidezes que ela nao rastreava (episodios sem
        objeto, repetidos para sempre); (2) sem as regras de "a vista sob todas" e "nao se
        encostam", a maior discordancia caia na borda do quadro e em corpos que se fundiam.
        aleatorio=True e o CONTROLE: os mesmos candidatos e regras, escolha ao acaso.
        -> {estados (coordenadas percebidas), divergencia prevista, o que cada rival preve}"""
        rv = self.rivais(fonte) if rivais is None else rivais
        if len(rv) < 2:
            return None
        T = self.teorias.get(rv[0]["teoria"])
        vistos = [self._der(e)["z0"] for e in (T["membros"][-60:] if T else [])]
        vistos = [z for z in vistos if len(z)]
        if not vistos:
            return None
        Zv = np.concatenate(vistos)
        dts = [np.median(np.diff(tr["t"])) for e in T["membros"][-10:] for tr in self.eps[e]["trilhas"] if len(tr["t"]) > 1]
        olho = 0.5 * MAX_PASSO / max(float(np.median(dts)) if dts else 0.2, 1e-3) / PX   # meias-larguras/s
        self.n_exp = getattr(self, "n_exp", 0) + 1
        rng = np.random.default_rng([self.n_ep, self.n_teo, self.n_conc, self.n_exp, 2])
        base = Zv[rng.integers(len(Zv), size=n_cand)]
        f = rng.uniform(*EXP_FATOR, n_cand)
        th = rng.uniform(-EXP_GIRO, EXP_GIRO, n_cand) + np.where(rng.random(n_cand) < EXP_INVERTE, np.pi, 0.0)
        vx, vy = base[:, 2] * f, base[:, 3] * f
        p = np.clip(base[:, :2] + rng.normal(0, 0.05, (n_cand, 2)), -0.85, 0.85)
        Z0 = np.column_stack([p, np.cos(th) * vx - np.sin(th) * vy, np.sin(th) * vx + np.cos(th) * vy])
        Pv = self._posicoes(rv[0]["lei"], Z0, EXP_A_VISTA)
        passo = np.linalg.norm(np.diff(np.concatenate([Z0[:, None, :2], Pv], 1), axis=1), axis=2) \
            / np.diff((0.0,) + EXP_A_VISTA)[None]
        P = [self._posicoes(r["lei"], Z0, EXP_HORIZONTES) for r in rv]
        # o desfecho tem de ficar A VISTA sob TODAS as rivais (experimento cujo resultado sai do
        # quadro numa delas nao decide nada), e a vista da atual por 6 s; e os olhos acompanham
        ok = (np.abs(Pv) <= 0.9).all(axis=(1, 2)) & (np.linalg.norm(Z0[:, 2:], axis=1) <= olho) & (passo <= olho).all(1)
        for Pr in P:
            ok &= (np.abs(Pr) <= 0.9).all(axis=(1, 2))
        dif = np.stack([np.minimum(np.linalg.norm(Pr - P[0], axis=2), 1.0) for Pr in P[1:]], 0)
        div = dif.mean(0).mean(1) * PX                                 # px (em 128), media das rivais e dos horizontes
        cand = np.nonzero(ok)[0]
        if not len(cand):
            return None
        ordem = rng.permutation(cand) if aleatorio else cand[np.argsort(-div[cand], kind="stable")]
        # as coisas nao podem se encostar: separadas no comeco e no caminho que a atual preve
        Pc = np.concatenate([Z0[:, None, :2], self._posicoes(rv[0]["lei"], Z0, EXP_CAMINHO)], 1)
        esc = []
        for i in ordem:
            if all(np.linalg.norm(Pc[i] - Pc[j], axis=1).min() >= EXP_SEPARACAO for j in esc):
                esc.append(int(i))
            if len(esc) >= n_corpos:
                break
        ruido = float(np.median(T["ruido_px"])) if T.get("ruido_px") is not None and len(T["ruido_px"]) else None
        return {"estados": Z0[esc].round(4).tolist(), "divergencia_px": div[esc].round(2).tolist(),
                "divergencia_tipica_px": round(float(np.median(div[cand])), 2), "ruido_teoria_px": ruido,
                "teoria": T["id"], "rivais": [r["nome"] for r in rv], "aleatorio": bool(aleatorio),
                "candidatos_validos": int(len(cand)), "rapidez_max_olho": round(olho, 3),
                "horizontes": list(EXP_HORIZONTES),
                "previsto": {r["nome"]: P[k][esc].round(4).tolist() for k, r in enumerate(rv)}}

    def resolve_experimento(self, eid, rivais, proposta=None):
        """Depois de ver o que aconteceu: qual rival previu melhor, trajetoria a trajetoria (bits
        pareados contra a atual). Se vence uma rival de OUTRA opcao de conceito por mais do que o
        custo de ter escolhido entre elas, a sala de espera da teoria que guardou o episodio e
        refeita agora: o experimento e evidencia como qualquer outra, so que pedida."""
        if eid is None or eid not in self.eps or len(rivais) < 2:
            return None
        ets = [self._erros_traj(r["lei"], [eid])[0] for r in rivais]
        if not len(ets[0]):
            return None
        base = np.maximum(ets[0], 0.05)
        bits = [float(np.sum(np.log2(base / np.maximum(e, 0.05)))) for e in ets]
        custo = math.log2(len(rivais))
        k = int(np.argmax(bits))
        venc = rivais[k]
        self.contas["experimentos"] = self.contas.get("experimentos", 0) + 1
        decisivo = (max(bits) - min(bits)) > custo
        if decisivo:
            self.contas["experimentos_decisivos"] = self.contas.get("experimentos_decisivos", 0) + 1
        revisou = False
        if venc["tipo"] == "conceito" and bits[k] > custo:
            Th = self.teorias.get(self.eps[eid].get("teoria"))
            if Th is not None and Th["id"] == venc["teoria"] and self._sala_de_espera(Th, "experimento"):
                self._consenso(Th, eid)
                revisou = True
        out = {"episodio": eid, "teoria": rivais[0]["teoria"], "vencedora": venc["nome"], "tipo": venc["tipo"],
               "bits_vencedora": round(bits[k], 1), "discriminacao_bits": round(max(bits) - min(bits), 1),
               "decisivo": bool(decisivo), "revisou": revisou,
               "erros_px": {r["nome"]: round(float(np.median(e)), 3) for r, e in zip(rivais, ets) if len(e)},
               "pedido": bool(proposta is not None and not proposta.get("aleatorio"))}
        self._log("experimento", **{k_: v for k_, v in out.items() if k_ != "erros_px"},
                  erros_px=dict(list(out["erros_px"].items())[:6]))
        self.ultimo_experimento = dict(out, proposta=proposta, quando=time.time())
        return out

    def preve_sem_aprender(self, percepcao, tid):
        """TESTE: o erro da 1a tentativa (2 s) da teoria tid num episodio que ela NAO guarda e do
        qual nao aprende nada; e o da inercia. -> (erro px, inercia px) ou None."""
        if not percepcao["trilhas"]:
            return None
        tmp = ("teste", self.n_ep)
        self.eps[tmp] = {"trilhas": [{k: tr[k] for k in ("t", "p", "quebras")} for tr in percepcao["trilhas"]]}
        try:
            if not len(self._der(tmp)["z0"]):
                return None
            T = self.teorias.get(tid)
            ine = self._erro_ep(None, [tmp])[0]
            return (self._erro_ep(T["lei"], [tmp])[0] if T else ine), ine
        finally:
            self.eps.pop(tmp, None)
            self._cache.pop(tmp, None)

    def progresso(self, eid, n=10):
        """O que este episodio diz do APRENDIZADO dela naquela regiao do mundo (progresso de
        aprendizagem, Oudeyer, Kaplan e Hafner 2007): a 1a tentativa — sempre fora da amostra —
        comparada a mediana das n anteriores da mesma fonte e do mesmo tipo (assistida ou montada
        por ela). Fracao do erro relativo que deixou de existir, em [-1, 1]; None sem historia."""
        ep = self.eps.get(eid)
        if ep is None or ep.get("razao_1a") is None:
            return None
        montado = "?exp=" in ep["url"]
        prev = []
        for e in reversed(self.ordem):
            x = self.eps[e]
            if e != eid and x["fonte"] == ep["fonte"] and ("?exp=" in x["url"]) == montado \
                    and x.get("razao_1a") is not None:
                prev.append(x["razao_1a"])
                if len(prev) >= n:
                    break
        if len(prev) < 3:
            return None
        m = float(np.median(prev))
        return float(np.clip((m - ep["razao_1a"]) / max(m, 1e-3), -1, 1))

    # ── vitrine e holograma: SO LEITURA, para o painel (nada aqui decide nada) ──
    def _caminho(self, lei, z0, ts):
        """a mesma rolagem da previsao, guardando a posicao nos instantes pedidos."""
        z = np.asarray(z0, float).copy()[None]
        out, t = [], 0.0
        for tt in ts:
            while t < tt - 1e-9:
                h = min(DT_ROLA, tt - t)
                z[:, 2:] += acel_lei(lei, z, self.conceitos) * h
                z[:, :2] += z[:, 2:] * h
                z = np.clip(z, -20, 20)
                t += h
            out.append(z[0, :2].copy())
        return np.array(out)

    def vitrine(self, eid, n=8):
        """o episodio recem-analisado como ela o viu: as trilhas das coisas que seguiu e, para
        alguns instantes, o caminho que a teoria dela PREVIU para os 2 s seguintes, o da inercia
        e onde a coisa de fato estava. Coordenadas normalizadas (centro do quadro, meia-largura 1)."""
        ep = self.eps.get(eid)
        if ep is None:
            return None
        d = self._der(eid)
        T = self.teorias.get(ep.get("teoria"))
        trilhas = []
        for tr in ep["trilhas"]:
            P = tr["p"].astype(float)
            sel = np.linspace(0, len(P) - 1, min(len(P), 120)).astype(int)
            trilhas.append(P[sel].round(4).tolist())
        prev = []
        if len(d["z0"]):
            for i in np.linspace(0, len(d["z0"]) - 1, min(n, len(d["z0"]))).astype(int):
                ts = np.linspace(0, float(d["dt"][i]), 9)
                prev.append({"de": d["z0"][i, :2].round(4).tolist(),
                             "teoria": self._caminho(T["lei"], d["z0"][i], ts).round(4).tolist() if T else None,
                             "inercia": self._caminho(None, d["z0"][i], ts).round(4).tolist(),
                             "real": d["alvo"][i].round(4).tolist(), "trilha": int(d["trilha"][i])})
        return {"episodio": eid, "fonte": ep["fonte"], "url": ep["url"], "forma": ep.get("forma"),
                "teoria": ep.get("teoria"), "conceito": T["lei"].get("conceito") if T else None,
                "lei": formula_lei(T["lei"]) if T else None, "erro_1a_px": ep.get("erro_1a"),
                "inercia_px": ep.get("inercia"), "trilhas": trilhas, "previsoes": prev}

    def holograma(self, max_teorias=6, max_cenas=2, max_traj=4, max_conceitos=8):
        """O que ela IMAGINA com o que aprendeu — e so com isso. Para cada teoria: cenas REAIS da
        memoria dela e a mesma cena IMAGINADA pela lei dela a partir do primeiro instante de cada
        coisa (quanto tempo a imaginacao fica perto do real e honesto). Para cada conceito: o valor
        dele ao longo de trajetorias reais (linha reta = se conserva) e onde elas passaram."""
        out = {"teorias": [], "conceitos": []}
        for T in sorted(self.teorias.values(), key=lambda T: -len(T["membros"]))[:max_teorias]:
            cenas = []
            for e in T["membros"][-max_cenas:]:
                ep = self.eps[e]
                tracks = []
                for tr in ep["trilhas"][:max_traj]:
                    Zc, okc = estados_causais(tr)
                    idx = np.nonzero(okc)[0]
                    if not len(idx):
                        continue
                    i0 = int(idx[0])
                    t = tr["t"].astype(float)[i0:] - float(tr["t"][i0])
                    P = tr["p"].astype(float)[i0:]
                    sel = np.linspace(0, len(t) - 1, min(len(t), 48)).astype(int)
                    tracks.append({"t": t[sel].round(2).tolist(), "real": P[sel].round(4).tolist(),
                                   "imaginado": self._caminho(T["lei"], Zc[i0], t[sel]).round(4).tolist()})
                if tracks:
                    cenas.append({"episodio": e, "fonte": ep["fonte"], "url": ep["url"], "trilhas": tracks})
            out["teorias"].append({"id": T["id"], "episodios": len(T["membros"]),
                                   "conceito": T["lei"].get("conceito") if T["lei"] else None,
                                   "lei": formula_lei(T["lei"]), "cenas": cenas})
        for cid, c in list(self.conceitos.items())[-max_conceitos:]:
            usam = [T for T in self.teorias.values() if T["lei"] and T["lei"].get("conceito") == cid]
            fonte = usam or [T for T in self.teorias.values() if T["id"] in c["teorias"]]
            trajs = []
            for T in fonte[:2]:
                for e in T["membros"][-3:]:
                    for Z in self._der(e)["trajs"]:
                        if len(trajs) >= 6 or len(Z) < MIN_PONTOS:
                            continue
                        q = valor_conceito(c, Z)
                        sel = np.linspace(0, len(Z) - 1, min(len(Z), 48)).astype(int)
                        trajs.append({"pos": Z[sel, :2].round(4).tolist(), "q": q[sel].round(4).tolist()})
            out["conceitos"].append({"id": cid, "formula": formula_conceito(c), "lam": round(c["lam"], 5),
                                     "teorias": sorted(c["teorias"]), "em_uso": bool(usam),
                                     "revisoes": c.get("revisoes", 0), "nasceu": c.get("nasceu"),
                                     "placebo": c.get("placebo", False), "trajetorias": trajs})
        return out

    # ── o que mostrar (painel) ────────────────────────────────────────────────
    def resumo(self):
        ts = sorted(self.teorias.values(), key=lambda T: -len(T["membros"]))[:8]
        return {"contas": dict(self.contas), "episodios_guardados": len(self.eps),
                "teorias": [{"id": T["id"], "episodios": len(T["membros"]),
                             "conceito": T["lei"].get("conceito") if T["lei"] else None,
                             "lei": formula_lei(T["lei"]),
                             "ruido_relativo": round(float(np.median(T["ruido"])), 3) if T.get("ruido") is not None and len(T["ruido"]) else None,
                             "fontes": dict(collections.Counter(self.eps[e]["fonte"] for e in T["membros"]))}
                            for T in ts],
                "conceitos": [{"id": c, "formula": formula_conceito(x), "lam": round(x["lam"], 5),
                               "teorias": sorted(x["teorias"]), "placebo": x["placebo"]} for c, x in self.conceitos.items()],
                "recentes": self.recentes[-12:]}
