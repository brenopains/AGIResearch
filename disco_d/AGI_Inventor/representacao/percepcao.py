"""
percepcao — fragmentos, trilhas e as REVISOES pedidas pela evidencia
=====================================================================

Lado do APRENDIZ: so recebe quadros, tempos e fronteiras de episodio.

Primitivas declaradas (PRE_REGISTRO_etapa1.md): fragmento = componente conexo de
mesma cromaticidade quantizada; continuidade = vizinho mais proximo de mesma
cromaticidade, com checagem de ida e volta; agregacao; diferenca entre instantes.

Cada REVISAO nasce de uma evidencia medida no proprio fluxo e fica registrada em
`eventos` (o que foi proposto, por que, e se ficou):
  compor   duas trilhas que andam JUNTAS (deslocamento relativo constante)
  cortar   salto de area (fusao) ou passo incompativel com a propria historia
  religar  fins e comecos de segmentos, por atribuicao otima; custo = erro de
           extrapolacao e, quando a descoberta ja existe, a diferenca de q
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

LIMIAR_FG = 40
MAX_PASSO = 6.0


# ============================================================ fragmentos
def fragmentos(img):
    s = img.astype(np.int32).sum(2)
    fg = s > LIMIAR_FG
    if not fg.any():
        return []
    croma = np.round(img[fg].astype(float) / s[fg][:, None] * 6).astype(int)
    chave = croma[:, 0] * 100 + croma[:, 1] * 10 + croma[:, 2]
    mapa = np.full(s.shape, -1, int)
    mapa[fg] = chave
    out = []
    for k in np.unique(chave):
        lab, n = ndimage.label(mapa == k)
        for i, sl in enumerate(ndimage.find_objects(lab), 1):
            m = lab[sl] == i
            area = int(m.sum())
            if area < 3:
                continue
            ys, xs = np.nonzero(m)
            w = s[sl][m].astype(float)
            borda_m = m & ~ndimage.binary_erosion(m)
            by, bx = np.nonzero(borda_m)
            if len(by) > 1024:
                sel = np.linspace(0, len(by) - 1, 1024).astype(int)
                by, bx = by[sel], bx[sel]
            out.append({"c": np.array([(xs + sl[1].start) @ w / w.sum(),
                                       (ys + sl[0].start) @ w / w.sum()]),
                        "area": area, "chave": int(k),
                        "borda": np.stack([bx + sl[1].start, by + sl[0].start], 1)
                        .astype(np.int16),
                        "caixa": np.array([xs.min() + sl[1].start, ys.min() + sl[0].start,
                                           xs.max() + sl[1].start, ys.max() + sl[0].start],
                                          float)})
    return out


def percebe_episodio(quadros):
    return [fragmentos(q) for q in quadros]


# ================================================ continuidade (congelado)
def rastreia(frs):
    """Vizinho mais proximo de mesma cromaticidade, com checagem ida-e-volta."""
    trilhas, abertas = [], {}
    for k, fs in enumerate(frs):
        novas = {}
        usados = set()
        dono = {}                               # fragmento -> trilha (a entrada NAO e tocada)
        if abertas:
            ids = list(abertas)
            for j, f in enumerate(fs):
                cand = [(np.linalg.norm(f["c"] - trilhas[i]["pos"][-1]), i) for i in ids
                        if trilhas[i]["chave"] == f["chave"]]
                if not cand:
                    continue
                d, i = min(cand)
                if d > MAX_PASSO or i in usados:
                    continue
                volta = min((np.linalg.norm(g["c"] - trilhas[i]["pos"][-1]), jj)
                            for jj, g in enumerate(fs) if g["chave"] == f["chave"])
                if volta[1] != j:
                    continue
                usados.add(i)
                T = trilhas[i]
                T["q"].append(k)
                T["pos"].append(f["c"])
                T["area"].append(f["area"])
                T["caixa"].append(f["caixa"])
                T["borda"].append(f["borda"])
                novas[i] = True
                dono[j] = i
        for j, f in enumerate(fs):
            if j not in dono:
                trilhas.append({"chave": f["chave"], "q": [k], "pos": [f["c"]],
                                "area": [f["area"]], "caixa": [f["caixa"]],
                                "borda": [f["borda"]]})
                novas[len(trilhas) - 1] = True
                dono[j] = len(trilhas) - 1
        abertas = novas
    for T in trilhas:
        T["q"] = np.array(T["q"])
        T["pos"] = np.array(T["pos"])
        T["area"] = np.array(T["area"], float)
        T["caixa"] = np.array(T["caixa"], float)
    return trilhas


# ============================================== estado: diferenca no tempo
def estado(T, ts, janela=7):
    """Posicao, velocidade e aceleracao por ajuste quadratico local (tempos reais)."""
    q, P = T["q"], T["pos"]
    n = len(q)
    Z, A, ok = np.zeros((n, 4)), np.zeros((n, 2)), np.zeros(n, bool)
    meia = janela // 2
    for i in range(n):
        a, b = max(0, i - meia), min(n, i + meia + 1)
        if b - a < 5:
            continue
        tt = ts[q[a:b]] - ts[q[i]]
        if np.any(np.diff(q[a:b]) > 2):          # atravessa lacuna: nao estima
            continue
        M = np.stack([np.ones_like(tt), tt, tt ** 2], 1)
        coef, *_ = np.linalg.lstsq(M, P[a:b], rcond=None)
        Z[i] = [coef[0, 0], coef[0, 1], coef[1, 0], coef[1, 1]]
        A[i] = 2 * coef[2]
        ok[i] = True
    return Z, A, ok


def estado_causal(T, ts, janela=7):
    """Estado so com o PASSADO disponivel (para prever o proximo quadro)."""
    q, P = T["q"], T["pos"]
    n = len(q)
    quebras = np.array(T.get("quebras", []), int)       # juncoes que a revisao achou:
    Z, ok = np.zeros((n, 4)), np.zeros(n, bool)          # o estado NAO as atravessa
    for i in range(n):
        a = max(0, i - janela + 1)
        if len(quebras):
            dentro = quebras[(quebras > a) & (quebras <= i)]
            if len(dentro):
                a = int(dentro.max())
        if i + 1 - a < 4 or np.any(np.diff(q[a:i + 1]) > 2):
            continue
        tt = ts[q[a:i + 1]] - ts[q[i]]
        M = np.stack([np.ones_like(tt), tt, tt ** 2], 1)
        coef, *_ = np.linalg.lstsq(M, P[a:i + 1], rcond=None)
        Z[i] = [coef[0, 0], coef[0, 1], coef[1, 0], coef[1, 1]]
        ok[i] = True
    return Z, ok


def em_movimento(T, ts):
    if len(T["q"]) < 3:
        return False
    v = np.linalg.norm(np.diff(T["pos"], axis=0), axis=1) / np.diff(ts[T["q"]])
    return np.median(v) >= 1.0


# ================================================================ REVISOES
def compor(trilhas, eventos, tol=0.35, min_q=10):
    """Trilhas coexistentes que andam juntas viram UMA entidade (evidencia: rigidez)."""
    vivas = list(range(len(trilhas)))
    feito = True
    while feito:
        feito = False
        for ia in range(len(vivas)):
            for ib in range(ia + 1, len(vivas)):
                A, B = trilhas[vivas[ia]], trilhas[vivas[ib]]
                comum, ja, jb = np.intersect1d(A["q"], B["q"], return_indices=True)
                if len(comum) < min_q:
                    continue
                d = A["pos"][ja] - B["pos"][jb]
                dist = np.linalg.norm(d, axis=1).mean()
                raio = np.sqrt(max(A["area"].mean(), B["area"].mean()) / np.pi)
                if d.std(0).max() > tol or dist > 2.5 * raio:
                    continue
                if np.linalg.norm(np.diff(A["pos"][ja], axis=0), axis=1).mean() < 0.05:
                    continue                    # parados juntos nao sao evidencia
                wa, wb = A["area"][ja], B["area"][jb]
                pos = (A["pos"][ja] * wa[:, None] + B["pos"][jb] * wb[:, None]) \
                    / (wa + wb)[:, None]
                ca, cb = A["caixa"][ja], B["caixa"][jb]
                caixa = np.concatenate([np.minimum(ca[:, :2], cb[:, :2]),
                                        np.maximum(ca[:, 2:], cb[:, 2:])], 1)
                borda = [np.concatenate([A["borda"][x], B["borda"][y]]) for x, y in zip(ja, jb)]
                novo = {"chave": min(A["chave"], B["chave"]), "q": comum, "pos": pos,
                        "area": wa + wb, "caixa": caixa, "borda": borda}
                eventos.append(("compor", len(comum), round(float(d.std(0).max()), 3)))
                trilhas[vivas[ia]] = novo
                vivas.pop(ib)
                feito = True
                break
            if feito:
                break
    return [trilhas[i] for i in vivas]


def cortar(trilhas, ts, eventos, salto_area=1.6, salto_px=3.0):
    """Tira quadros de fusao e corta onde o passo foge da propria historia."""
    out = []
    for T in trilhas:
        q, P, ar = T["q"], T["pos"], T["area"]
        med = np.median(ar)
        bom = ar < salto_area * med
        if (~bom).any():
            eventos.append(("cortar_fusao", int((~bom).sum())))
        cortes = [0]
        idx = np.nonzero(bom)[0]
        for a, b, c in zip(idx[:-2], idx[1:-1], idx[2:]):
            if q[c] - q[b] > 1 or q[b] - q[a] > 1:
                continue
            prev = P[b] + (P[b] - P[a]) * (ts[q[c]] - ts[q[b]]) / (ts[q[b]] - ts[q[a]])
            if np.linalg.norm(P[c] - prev) > salto_px:
                cortes.append(np.searchsorted(idx, c))
                eventos.append(("cortar_salto", round(float(np.linalg.norm(P[c] - prev)), 2)))
        cortes.append(len(idx))
        segs = []
        for a, b in zip(cortes[:-1], cortes[1:]):
            sel = idx[a:b]
            if len(sel) == 0:
                continue
            # quebra tambem em lacunas deixadas pela fusao
            quebras = np.nonzero(np.diff(q[sel]) > 1)[0] + 1
            for s in np.split(sel, quebras):
                if len(s):
                    segs.append({"chave": T["chave"], "q": q[s], "pos": P[s], "area": ar[s],
                                 "caixa": T["caixa"][s], "borda": [T["borda"][x] for x in s]})
        out += segs
    return out


def _extrapola(T, ts, t_alvo, do_fim=True, n=10):
    q, P = (T["q"][-n:], T["pos"][-n:]) if do_fim else (T["q"][:n], T["pos"][:n])
    if len(q) < 3:
        return P[-1 if do_fim else 0]
    tt = ts[q] - ts[q[-1 if do_fim else 0]]
    M = np.stack([np.ones_like(tt), tt, tt ** 2], 1)
    coef, *_ = np.linalg.lstsq(M, P, rcond=None)
    dt = t_alvo - ts[q[-1 if do_fim else 0]]
    return coef[0] + coef[1] * dt + coef[2] * dt ** 2


def religar(segs, ts, eventos, qfun=None, sigma_q=None, lacuna=1.5, max_custo=8.0):
    """
    Hipoteses alternativas (quem continua quem) competem por um custo: erro de
    extrapolacao nos dois sentidos e, se a descoberta ja existe, a diferenca de q
    — a regularidade descoberta pedindo revisao da identidade.
    """
    segs = sorted(segs, key=lambda T: T["q"][0])
    for _volta in range(3):
        fins = [i for i, T in enumerate(segs) if len(T["q"]) >= 3]
        comecos = fins
        C = np.full((len(fins), len(comecos)), 1e6)
        for a, i in enumerate(fins):
            A = segs[i]
            for b, j in enumerate(comecos):
                B = segs[j]
                if i == j:
                    continue
                t0, t1 = ts[A["q"][-1]], ts[B["q"][0]]
                if not (0 < t1 - t0 <= lacuna) or B["q"][0] <= A["q"][-1]:
                    continue
                e1 = np.linalg.norm(_extrapola(A, ts, t1, True) - B["pos"][0])
                e2 = np.linalg.norm(_extrapola(B, ts, t0, False) - A["pos"][-1])
                custo = (e1 + e2) / 2
                if qfun is not None and sigma_q:
                    qa, qb = qfun(A, ts), qfun(B, ts)
                    if qa is not None and qb is not None:
                        custo += 2 * abs(qa - qb) / sigma_q
                C[a, b] = custo
        if not np.isfinite(C).any() or C.min() > max_custo:
            break
        ra, rb = linear_sum_assignment(C)
        pares = [(fins[a], comecos[b], C[a, b]) for a, b in zip(ra, rb) if C[a, b] <= max_custo]
        if not pares:
            break
        juntar = {}
        for i, j, c in pares:                   # a atribuicao ja da 1 par por fim e por
            juntar[i] = j                       # comeco; o tempo crescente impede ciclo
            eventos.append(("religar", round(float(c), 2), qfun is not None))
        usados, novos = set(), []
        for i, T in enumerate(segs):
            if i in usados or i in juntar.values():
                continue
            cad = T
            k = i
            while k in juntar:
                j = juntar[k]
                B = segs[j]
                quebras = list(cad.get("quebras", [])) + [len(cad["q"])] +                     [len(cad["q"]) + x for x in B.get("quebras", [])]
                cad = {"chave": cad["chave"], "quebras": quebras,
                       "q": np.concatenate([cad["q"], B["q"]]),
                       "pos": np.concatenate([cad["pos"], B["pos"]]),
                       "area": np.concatenate([cad["area"], B["area"]]),
                       "caixa": np.concatenate([cad["caixa"], B["caixa"]]),
                       "borda": cad["borda"] + B["borda"]}
                usados.add(j)
                k = j
            novos.append(cad)
        segs = sorted(novos, key=lambda T: T["q"][0])
    return segs
