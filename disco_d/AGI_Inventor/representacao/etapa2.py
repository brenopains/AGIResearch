"""
etapa2 — Breakout observacional: relacoes que explicam mudanca de movimento
=============================================================================

Criterios em PRE_REGISTRO_etapa2.md (escrito antes de rodar).

  python etapa2.py fumaca     sessoes 0/1/2 (ja tocadas), orcamento pequeno
  python etapa2.py            trios A, B, C (sessoes nunca tocadas)
"""

from __future__ import annotations

import io
import json
import os
import sys
import tarfile
import time
import zipfile

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import cv2
import numpy as np
from scipy import ndimage

import descoberta as D
import etapa1 as E
import percepcao as P
import relacoes as RL

ZIP = r"D:\agi_dados\atari_head\breakout.zip"
TRIOS = {"A": (4, 5, 6), "B": (7, 8, 9), "C": (10, 11, 12)}


# ============================================================ leitor corrigido
def le_sessao(indice, n):
    """TODOS os quadros em ordem (inclusive acao nula); tempo = indice (tempo de jogo)."""
    z = zipfile.ZipFile(ZIP)
    es = sorted(i.filename for i in z.infolist() if i.filename.endswith(".tar.bz2"))
    acao = {}
    for l in z.read(es[indice].replace(".tar.bz2", ".txt")).decode().splitlines()[1:]:
        c = l.split(",")
        try:
            acao[int(c[0].rsplit("_", 1)[1])] = int(c[5]) if c[5] != "null" else -1
        except (ValueError, IndexError):
            continue
    brutos = {}
    with tarfile.open(fileobj=io.BytesIO(z.read(es[indice])), mode="r|bz2") as t:
        for m in t:
            if m.name.endswith(".png"):
                brutos[int(m.name.rsplit("_", 1)[1][:-4])] = t.extractfile(m).read()
    ks = sorted(brutos)[:n]
    frs, regs = [], []
    for k in ks:
        im = cv2.cvtColor(cv2.imdecode(np.frombuffer(brutos[k], np.uint8), cv2.IMREAD_COLOR),
                          cv2.COLOR_BGR2RGB)
        frs.append(P.fragmentos(im))
        regs.append(_regioes(im))
    acoes = np.array([acao.get(k, -1) for k in ks], float)   # acao nula = -1, MANTIDA
    return frs, regs, np.arange(len(ks), dtype=float), acoes


def _regioes(im):
    """Alvo comum do avaliador: regioes conexas nao-fundo, qualquer cor."""
    s = im.astype(np.int32).sum(2)
    lab, n = ndimage.label(s > P.LIMIAR_FG, structure=np.ones((3, 3), int))
    if n == 0:
        return np.zeros((0, 6))
    idx = np.arange(1, n + 1)
    cy = ndimage.sum(np.indices(s.shape)[0] * s, lab, idx) / ndimage.sum(s, lab, idx)
    cx = ndimage.sum(np.indices(s.shape)[1] * s, lab, idx) / ndimage.sum(s, lab, idx)
    area = ndimage.sum(np.ones_like(s), lab, idx)
    ok = area < 0.25 * s.size
    caixas = np.array([[sl[1].start, sl[0].start, sl[1].stop, sl[0].stop]
                       for sl in ndimage.find_objects(lab)], float)
    return np.concatenate([np.stack([cx, cy], 1), caixas], 1)[ok]


# ============================================================ percepcao
def percepcoes(conj):
    """CONGELADO e REVISAVEL (volta escolhida na SELECAO, como na etapa 1)."""
    cong = {k: [P.rastreia(frs) for frs, ts in v] for k, v in conj.items()}
    ev = []
    segs = {k: [E.segmentos_revisavel(frs, ts, ev) for frs, ts in v] for k, v in conj.items()}
    v1 = {k: [P.religar(s, ts, ev) for s, (frs, ts) in zip(segs[k], conj[k])] for k in conj}
    cands = [("volta0", cong), ("volta1", v1)]
    r1 = E._descobre(v1, conj)
    if r1 is not None:
        sig = D.sigma_dentro(r1["mod"], [Z for _T, Z, _A, _q in r1["est"]["ajuste"]])
        v2 = {k: [P.religar(s, ts, ev, E._qfun(r1["mod"]), sig) for s, (frs, ts)
                  in zip(segs[k], conj[k])] for k in conj}
        cands.append(("volta2", v2))
    notas = []
    for nome, ents in cands:
        r = E._descobre(ents, conj)
        if r is not None:
            notas.append((r["lam_sel"], nome, ents))
    if notas:
        _l, escolha, rev = min(notas, key=lambda x: x[0])
    else:
        escolha, rev = "volta1 (nenhuma grandeza achada)", v1
    return cong, rev, escolha, E._conta(ev), [(n, round(l, 4)) for l, n, _e in notas]


# ============================================================ avaliacao
def _treina(ents_aj, ts_aj, ac_aj):
    Xi, Xr, R = [], [], []
    for ents, ts, ac in zip(ents_aj, ts_aj, ac_aj):
        a, b, c, _ = RL.amostras(ents, ts, ac)
        if len(a):
            Xi.append(a)
            Xr.append(b)
            R.append(c)
    return np.concatenate(Xi), np.concatenate(Xr), np.concatenate(R)


def _previsoes(ents, ts, ac, mod):
    """Por quadro t+1: posicoes previstas de cada entidade com estado causal em t."""
    Xi, Xr, R, info = RL.amostras(ents, ts, ac)
    prev = {}
    if len(Xr):
        rp = mod.predict(Xr)
        for (k, i, _p, p0, desl), r in zip(info, rp):     # posicao OBSERVADA em t
            prev.setdefault(k + 1, []).append(p0 + desl + r)
    return prev


def erro_regioes(regs, prev):
    """Alvos comuns: regioes em t+1 que se moveram 0,5-8 px; erro a previsao mais proxima."""
    erros = []
    for k in range(1, len(regs)):
        A, B = regs[k - 1], regs[k]
        if len(A) == 0 or len(B) == 0:
            continue
        for reg in B:
            p = reg[:2]
            j = int(np.linalg.norm(A[:, :2] - p, axis=1).argmin())
            desl = p - A[j, :2]
            cantos = reg[2:] - A[j, 2:]                   # translacao INTEIRA: caixa anda junto
            if not 0.5 <= np.linalg.norm(desl) <= 8:
                continue
            if np.abs(cantos[:2] - cantos[2:]).max() > 1.5 or                     np.abs(cantos[:2] - desl).max() > 1.5:
                continue
            ps = prev.get(k, [])
            e = min((np.linalg.norm(x - p) for x in ps), default=8.0)
            erros.append(min(e, 8.0))
    return np.array(erros)


def um_trio(nome, idx, n=(2400, 1200, 3000)):
    t0 = time.time()
    conj, regs, acoes = {}, {}, {}
    for k, i, nn in zip(("ajuste", "selecao", "aval"), idx, n):
        frs, rg, ts, ac = le_sessao(i, nn)
        conj[k] = [(frs, ts)]
        regs[k] = rg
        acoes[k] = ac
    t_le = time.time() - t0
    cong, rev, escolha, eventos, notas = percepcoes(conj)
    rng = np.random.default_rng(11)
    out = {"trio": nome, "sessoes": idx, "escolha": escolha, "eventos": eventos,
           "lam_sel_voltas": notas, "t_leitura_s": round(t_le, 1)}
    ts_aj = [ts for _f, ts in conj["ajuste"]]
    ts_av = [ts for _f, ts in conj["aval"]]
    # E2-rel e E2-ctl, na percepcao REVISAVEL
    ac_aj, ac_av = [acoes["ajuste"]], acoes["aval"]
    Xi, Xr, R = _treina(rev["ajuste"], ts_aj, ac_aj)
    mi, mr = RL.arvore(Xi, R), RL.arvore(Xr, R)
    Yi, Yr, S, info = RL.amostras(rev["aval"][0], ts_av[0], ac_av)
    ordem = np.argsort([x[0] for x in info], kind="stable")
    Yi, Yr, S = Yi[ordem], Yr[ordem], S[ordem]
    info = [info[j] for j in ordem]
    muda = np.linalg.norm(S, axis=1) > 1.5
    ei, er = RL.erro(mi, Yi, S), RL.erro(mr, Yr, S)
    dif = (ei - er)[muda]
    out["E2-rel"] = {"n_mudanca": int(muda.sum()), "n_total": len(S),
                     "erro_ind": float(ei[muda].mean()), "erro_rel": float(er[muda].mean()),
                     "razao": float(er[muda].mean() / ei[muda].mean()),
                     "ic": RL.ic_blocos(dif, rng),
                     "erro_ind_tudo": float(ei.mean()), "erro_rel_tudo": float(er.mean())}
    imp = dict(zip(RL.NOMES_REL, np.round(mr.feature_importances_, 3).tolist()))
    out["importancia"] = imp
    # parceira mais frequente nos intervalos de mudanca (leitura)
    parc = {}
    for (k, i, p, _p0, _d), m in zip(info, muda):
        if m and p >= 0:
            parc[p] = parc.get(p, 0) + 1
    ents_av = rev["aval"][0]
    top = sorted(parc.items(), key=lambda x: -x[1])[:3]
    out["parceiras"] = [{"n": c, "chave": ents_av[p]["chave"],
                         "pos_media": np.round(ents_av[p]["pos"].mean(0), 1).tolist(),
                         "area": float(np.median(ents_av[p]["area"]))} for p, c in top]
    # controle: relacoes embaralhadas entre amostras
    Xc = Xr.copy()
    Xc[:, 5:] = Xc[rng.permutation(len(Xc)), 5:]         # col 4 = acao, fica
    Yc = Yr.copy()
    Yc[:, 5:] = Yc[rng.permutation(len(Yc)), 5:]
    ec = RL.erro(RL.arvore(Xc, R), Yc, S)
    out["E2-ctl"] = {"erro_rel_embaralhado": float(ec[muda].mean()),
                     "ganho": float(1 - ec[muda].mean() / ei[muda].mean())}
    # E2-rev: alvos comuns (regioes), cada percepcao com sua arvore relacional
    erros = {}
    for nome_s, ents in (("congelado", cong), ("revisavel", rev)):
        a, b, c = _treina(ents["ajuste"], ts_aj, ac_aj)
        m = RL.arvore(b, c)
        prev = _previsoes(ents["aval"][0], ts_av[0], ac_av, m)
        erros[nome_s] = erro_regioes(regs["aval"], prev)
    d = erros["congelado"] - erros["revisavel"]
    out["E2-rev"] = {"n_alvos": len(d), "erro_cong": float(erros["congelado"].mean()),
                     "erro_rev": float(erros["revisavel"].mean()),
                     "razao": float(erros["revisavel"].mean() / erros["congelado"].mean()),
                     "ic": RL.ic_blocos(d, rng)}
    out["t_total_s"] = round(time.time() - t0, 1)
    return out


def veredito(rs):
    n = len(rs)
    precisa = 2 if n == 3 else n
    rel = sum(1 for r in rs if r["E2-rel"]["razao"] <= 0.80 and r["E2-rel"]["ic"][0] > 0)
    rev = sum(1 for r in rs if r["E2-rev"]["razao"] <= 0.90 and r["E2-rev"]["ic"][0] > 0)
    ctl = sum(1 for r in rs if r["E2-ctl"]["ganho"] <= 0.05)
    return {"E2-rel": "PASSA" if rel >= precisa else "falha", "E2-rel_n": rel,
            "E2-rev": "PASSA" if rev >= precisa else "falha", "E2-rev_n": rev,
            "E2-ctl": "PASSA" if ctl >= precisa else "falha", "E2-ctl_n": ctl}


def main(trios, n, rotulo):
    os.makedirs("resultados", exist_ok=True)
    rs = []
    for nome, idx in trios.items():
        rs.append(um_trio(nome, idx, n))
        r = rs[-1]
        print("TRIO %s sessoes %s | escolha %s | %.0f s" % (nome, idx, r["escolha"], r["t_total_s"]))
        a = r["E2-rel"]
        print("  E2-rel mudanca (n=%d de %d): indep %.2f px, relacional %.2f px, razao %.2f, "
              "IC %s" % (a["n_mudanca"], a["n_total"], a["erro_ind"], a["erro_rel"], a["razao"],
                         np.round(a["ic"], 3).tolist()))
        print("         tudo: indep %.2f, relacional %.2f" % (a["erro_ind_tudo"], a["erro_rel_tudo"]))
        print("  E2-ctl relacoes embaralhadas: ganho %.1f%%" % (100 * r["E2-ctl"]["ganho"]))
        b = r["E2-rev"]
        print("  E2-rev alvos %d: congelado %.2f px, revisavel %.2f px, razao %.2f, IC %s"
              % (b["n_alvos"], b["erro_cong"], b["erro_rev"], b["razao"],
                 np.round(b["ic"], 3).tolist()))
        print("  importancia: %s" % r["importancia"])
        print("  parceiras nos intervalos de mudanca: %s" % r["parceiras"])
        print("  eventos: %s | lam_sel: %s" % (r["eventos"], r["lam_sel_voltas"]), flush=True)
        json.dump(rs, open("resultados/etapa2_%s.json" % rotulo, "w"), indent=1, default=float)
    ver = veredito(rs)
    print("\nVEREDITO: %s" % ver)
    json.dump({"trios": rs, "veredito": ver}, open("resultados/etapa2_%s.json" % rotulo, "w"),
              indent=1, default=float)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fumaca":
        main({"F": (0, 1, 2)}, (1200, 600, 1500), "fumaca")
    else:
        main(TRIOS, (2400, 1200, 3000), "piloto")
