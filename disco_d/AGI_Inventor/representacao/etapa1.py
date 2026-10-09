"""
etapa1 — percepcao revisavel x congelada, das orbitas em pixels ate h
=======================================================================

Criterios e previsao em PRE_REGISTRO_etapa1.md (escrito antes de rodar).

  python etapa1.py checagem     1 mundo, conjuntos menores (<= 15 min)
  python etapa1.py              3 mundos, o piloto que vale
"""

from __future__ import annotations

import json
import os
import sys
import time

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

import descoberta as D
import mundo_orbitas as M
import percepcao as P

SAIDA = "resultados"


# ======================================================= lado do APRENDIZ
def entidades_congelado(frs):
    return P.rastreia(frs)


def segmentos_revisavel(frs, ts, eventos):
    tr = P.rastreia(frs)
    tr = P.compor(tr, eventos)
    return P.cortar(tr, ts, eventos)


def estados(ents, ts):
    """Z e aceleracao validos de cada entidade que se move."""
    out = []
    for T in ents:
        if len(T["q"]) < 10 or not P.em_movimento(T, ts):
            continue
        Z, A, ok = P.estado(T, ts)
        if ok.sum() >= 10:
            out.append((T, Z[ok], A[ok], T["q"][ok]))
    return out


def _qfun(mod):
    def f(T, ts):
        Z, _A, ok = P.estado(T, ts)
        return float(D.q(mod, Z[ok]).mean()) if ok.sum() >= 5 else None
    return f


def sistema(nome, conj, rev=True):
    """
    conj: {'ajuste': [(frs, ts)], 'selecao': [...], 'aval': [...]}.
    Devolve o estado final escolhido NA SELECAO, e os eventos de revisao.
    """
    eventos = []
    if not rev:
        ents = {k: [entidades_congelado(frs) for frs, ts in v] for k, v in conj.items()}
        return _descobre(ents, conj), eventos, "congelado"
    segs = {k: [segmentos_revisavel(frs, ts, eventos) for frs, ts in v]
            for k, v in conj.items()}
    candidatos = []
    base = {k: [entidades_congelado(frs) for frs, ts in v] for k, v in conj.items()}
    candidatos.append(("volta0 (sem revisao)", _descobre(base, conj)))
    e1 = {k: [P.religar(s, ts, eventos) for s, (frs, ts) in zip(segs[k], conj[k])]
          for k in conj}
    r1 = _descobre(e1, conj)
    candidatos.append(("volta1 (compor/cortar/religar)", r1))
    if r1 is not None:
        sig = D.sigma_dentro(r1["mod"], [Z for _T, Z, _A, _q in r1["est"]["ajuste"]])
        e2 = {k: [P.religar(s, ts, eventos, _qfun(r1["mod"]), sig)
                  for s, (frs, ts) in zip(segs[k], conj[k])] for k in conj}
        candidatos.append(("volta2 (religar usando q)", _descobre(e2, conj)))
    validos = [(n, c) for n, c in candidatos if c is not None]
    if not validos:
        return None, eventos, "nenhum"
    melhor = min(validos, key=lambda nc: nc[1]["lam_sel"])     # decide na SELECAO
    return melhor[1], eventos, melhor[0]


def _descobre(ents, conj):
    est = {k: [x for e, (frs, ts) in zip(ents[k], conj[k]) for x in
               [(T, Z, A, q, ts) for T, Z, A, q in estados(e, ts)]]
           for k in conj}
    est = {k: [(T, Z, A, q) for T, Z, A, q, _ts in v] for k, v in est.items()}
    trajs = [Z for _T, Z, _A, _q in est["ajuste"]]
    if len(trajs) < 3:
        return None
    try:
        mod = D.acha(trajs)
    except Exception:
        return None
    lam_sel = D.lam(mod, [Z for _T, Z, _A, _q in est["selecao"]])
    return {"mod": mod, "est": est, "ents": ents, "lam_sel": lam_sel}


# ======================================================= lado do AVALIADOR
def _corpo(pos, k, verdade):
    d = np.linalg.norm(verdade["pix"][:, k] - pos, axis=1)
    b = int(d.argmin())
    return b if d[b] <= verdade["raio"] + 2.5 else None


def pureza(ents_por_ep, verdades, ts_por_ep):
    certos = total = 0
    for ents, verd, ts in zip(ents_por_ep, verdades, ts_por_ep):
        for T in ents:
            if len(T["q"]) < 10 or not P.em_movimento(T, ts):
                continue
            cs = [_corpo(p, k, verd) for k, p in zip(T["q"], T["pos"])]
            val = [c for c in cs if c is not None]
            if not val:
                total += len(cs)
                continue
            maj = max(set(val), key=val.count)
            certos += sum(c == maj for c in cs)
            total += len(cs)
    return certos / max(total, 1)


def avalia(res, conj, verdades):
    """corr com h, lambda, rolagem com e sem q, pureza — nos episodios de AVALIACAO."""
    mod, est = res["mod"], res["est"]
    trajs_av = [Z for _T, Z, _A, _q in est["aval"]]
    lam_av = D.lam(mod, trajs_av)
    ents_aval = res["ents"]["aval"]
    qs, hs = [], []
    idx = 0
    por_ep = []
    for (frs, ts), verd, ents in zip(conj["aval"], verdades["aval"], ents_aval):
        for T, Z, A, q in estados(ents, ts):
            por_ep.append((T, Z, q, ts, verd))
    for T, Z, q, ts, verd in por_ep:
        if len(q) < 60:
            continue
        cs = [_corpo(p, k, verd) for k, p in zip(T["q"], T["pos"])]
        val = [c for c in cs if c is not None]
        if not val:
            continue
        maj = max(set(val), key=val.count)
        qs.append(D.q(mod, Z).mean())
        hs.append(verd["h"][maj])
    corr = float(abs(np.corrcoef(qs, hs)[0, 1])) if len(qs) >= 4 else 0.0
    # USO: campo aprendido no AJUSTE; rolagem de 2 s na AVALIACAO
    Pp = np.concatenate([Z[:, :2] for _T, Z, _A, _q in est["ajuste"]])
    Aa = np.concatenate([A for _T, _Z, A, _q in est["ajuste"]])
    campo = D.Campo(Pp, Aa)
    e_sem, e_com = [], []
    for T, Z, q, ts, verd in por_ep:
        if len(q) < 90:
            continue
        i0 = 5
        k0 = q[i0]
        if k0 + 60 >= len(ts):
            continue
        cs = [_corpo(p, k, verd) for k, p in zip(T["q"], T["pos"])]
        val = [c for c in cs if c is not None]
        if not val:
            continue
        maj = max(set(val), key=val.count)
        tempos = ts[k0:k0 + 61]
        verdade = verd["pix"][maj, k0:k0 + 61]
        e_sem.append(np.linalg.norm(D.rola(Z[i0], tempos, campo) - verdade, axis=1).mean())
        e_com.append(np.linalg.norm(D.rola(Z[i0], tempos, campo, mod) - verdade,
                                    axis=1).mean())
    rol = (float(np.mean(e_sem)), float(np.mean(e_com))) if e_sem else (None, None)
    pur = pureza(ents_aval, verdades["aval"], [ts for _f, ts in conj["aval"]])
    return {"corr": corr, "lam_aval": lam_av, "n_ent": len(qs), "rolagem": rol,
            "pureza": pur, "lam_sel": res["lam_sel"]}


# ============================================ substituicoes diagnosticas
def _ents_verdade(verd, ts, qual):
    """Entidades montadas com a VERDADE (so para localizar o gargalo)."""
    out = []
    for b in range(verd["pix"].shape[0]):
        pos = verd["pix"][b]
        out.append({"chave": b, "q": np.arange(len(ts)), "pos": pos,
                    "area": np.full(len(ts), 20.0)})
    return out


def diagnostico(conj, verdades, cam):
    """identidade verdadeira; posicao verdadeira; posicao + velocidade verdadeiras."""
    out = {}
    # identidade verdadeira sobre as posicoes PERCEBIDAS
    ents = {}
    for k in conj:
        ents[k] = []
        for (frs, ts), verd in zip(conj[k], verdades[k]):
            grupos = {}
            for kq, fs in enumerate(frs):
                for f in fs:
                    b = _corpo(f["c"], kq, verd)
                    if b is None:
                        continue
                    g = grupos.setdefault(b, {"q": [], "pos": [], "area": []})
                    if g["q"] and g["q"][-1] == kq:      # dois fragmentos do mesmo corpo
                        g["pos"][-1] = (g["pos"][-1] + f["c"]) / 2
                        continue
                    g["q"].append(kq)
                    g["pos"].append(f["c"])
                    g["area"].append(f["area"])
            ents[k].append([{"chave": b, "q": np.array(g["q"]), "pos": np.array(g["pos"]),
                             "area": np.array(g["area"], float)} for b, g in grupos.items()])
    r = _descobre(ents, conj)
    out["identidade_verdadeira"] = avalia(r, conj, verdades) if r else None
    ents = {k: [_ents_verdade(verd, ts, "pos") for (frs, ts), verd in zip(conj[k], verdades[k])]
            for k in conj}
    r = _descobre(ents, conj)
    out["posicao_verdadeira"] = avalia(r, conj, verdades) if r else None
    # posicao + velocidade verdadeiras: Z direto, sem diferencas finitas
    A, _t = cam
    trajs = {}
    for k in conj:
        trajs[k] = []
        for (frs, ts), verd in zip(conj[k], verdades[k]):
            for b in range(3):
                vel = verd["estado"][b, :, 2:] @ A.T
                trajs[k].append(np.concatenate([verd["pix"][b], vel], 1))
    mod = D.acha(trajs["ajuste"])
    qs = [D.q(mod, Z).mean() for Z in trajs["aval"]]
    hs = [verd["h"][b] for verd in verdades["aval"] for b in range(3)]
    out["posicao_e_velocidade_verdadeiras"] = {
        "corr": float(abs(np.corrcoef(qs, hs)[0, 1])),
        "lam_aval": D.lam(mod, trajs["aval"])}
    return out


# ================================================================ o piloto
def prepara(semente, n=(12, 6, 12), embaralha=False):
    W = M.mundo(semente, *n)
    rng = np.random.default_rng(1000 + semente)
    conj, verd = {}, {}
    for k in ("ajuste", "selecao", "aval"):
        conj[k], verd[k] = [], []
        for quadros, ts, v in W[k]:
            frs = P.percebe_episodio(quadros)
            if embaralha:                       # CONTROLE: o tempo destruido
                frs = [frs[i] for i in rng.permutation(len(frs))]
            conj[k].append((frs, ts))
            verd[k].append(v)
    return conj, verd, W["camera"]


def um_mundo(semente, n=(12, 6, 12)):
    t0 = time.time()
    conj, verd, cam = prepara(semente, n)
    t_perc = time.time() - t0
    r_cong, _ev0, _ = sistema("congelado", conj, rev=False)
    r_rev, eventos, escolha = sistema("revisavel", conj, rev=True)
    out = {"semente": semente, "t_percepcao_s": round(t_perc, 1),
           "congelado": avalia(r_cong, conj, verd) if r_cong else None,
           "revisavel": avalia(r_rev, conj, verd) if r_rev else None,
           "escolha_revisavel": escolha,
           "eventos": _conta(eventos),
           "coef_q": dict(zip(D.NOMES, np.round(r_rev["mod"]["w"], 3).tolist())) if r_rev else None}
    out["diagnostico"] = diagnostico(conj, verd, cam)
    del conj
    cconj, cverd, _ = prepara(semente, n, embaralha=True)
    rc, _e, _ = sistema("revisavel", cconj, rev=True)
    out["controle_lam_sel"] = rc["lam_sel"] if rc else None
    out["t_total_s"] = round(time.time() - t0, 1)
    return out


def _conta(eventos):
    c = {}
    for e in eventos:
        c[e[0]] = c.get(e[0], 0) + 1
    c["religar_com_q"] = sum(1 for e in eventos if e[0] == "religar" and e[2])
    return c


def veredito(rs):
    def conta(f):
        return sum(1 for r in rs if f(r))
    e1h = conta(lambda r: r["revisavel"] and r["revisavel"]["corr"] >= 0.95
                and r["revisavel"]["lam_aval"] <= 0.10)
    e1u = conta(lambda r: r["revisavel"] and r["revisavel"]["rolagem"][0]
                and r["revisavel"]["rolagem"][1] <= 0.80 * r["revisavel"]["rolagem"][0])
    def pegou(r):                                # mudanca registrada antes do piloto
        c, d = r["congelado"], r["diagnostico"].get("identidade_verdadeira")
        if not c:
            return False
        return c["pureza"] < 0.98 or (d is not None and c["lam_aval"] >= 2 * d["lam_aval"])
    armadilha = conta(pegou)

    def rev_ganha(r):
        a, b = r["revisavel"], r["congelado"]
        if not a or not b:
            return False
        melhor_prev = (a["rolagem"][1] is not None and b["rolagem"][1] is not None
                       and a["rolagem"][1] < b["rolagem"][1])
        return a["pureza"] > b["pureza"] and (a["lam_aval"] < b["lam_aval"] or melhor_prev)
    e1r = conta(rev_ganha)
    e1c = conta(lambda r: r["controle_lam_sel"] is None or r["controle_lam_sel"] > 0.5)
    n = len(rs)
    precisa = 2 if n == 3 else max(1, n)
    return {"E1-h": "PASSA" if e1h >= precisa else "falha", "E1-h_n": e1h,
            "E1-uso": "PASSA" if e1u >= precisa else "falha", "E1-uso_n": e1u,
            "E1-rev": ("INCONCLUSIVO (armadilha nao pegou)" if armadilha < precisa
                       else "PASSA" if e1r >= precisa else "falha"), "E1-rev_n": e1r,
            "armadilha_pegou_n": armadilha,
            "E1-ctl": "PASSA" if e1c == n else "falha", "E1-ctl_n": e1c}


def imprime(rs, ver):
    print("=" * 78)
    print("ETAPA 1 — das orbitas em pixels ate h: revisavel x congelado")
    print("=" * 78)
    for r in rs:
        print("\nMUNDO %d  (percepcao %.0f s, total %.0f s) | escolha do revisavel: %s"
              % (r["semente"], r["t_percepcao_s"], r["t_total_s"], r["escolha_revisavel"]))
        for s in ("congelado", "revisavel"):
            a = r[s]
            if not a:
                print("  %-10s nenhuma grandeza" % s)
                continue
            print("  %-10s corr %.3f | lam_aval %.3f | lam_sel %.3f | pureza %.3f | "
                  "rolagem sem q %.2f px, com q %.2f px | %d entidades"
                  % (s, a["corr"], a["lam_aval"], a["lam_sel"], a["pureza"],
                     a["rolagem"][0] or -1, a["rolagem"][1] or -1, a["n_ent"]))
        print("  eventos: %s" % r["eventos"])
        print("  controle (tempo embaralhado) lam_sel: %s" % r["controle_lam_sel"])
        for k, v in r["diagnostico"].items():
            if v:
                print("  diag %-34s corr %.3f | lam_aval %.3f" % (k, v["corr"], v["lam_aval"]))
        print("  q (revisavel): %s" % r["coef_q"])
    print("\nVEREDITO: %s" % ver)


def main(sementes=(1, 2, 3), n=(12, 6, 12), rotulo="piloto"):
    os.makedirs(SAIDA, exist_ok=True)
    rs = []
    for s in sementes:
        rs.append(um_mundo(s, n))
        json.dump(rs, open(os.path.join(SAIDA, "etapa1_%s.json" % rotulo), "w"), indent=1,
                  default=float)
        print("  mundo %d pronto (%.0f s)" % (s, rs[-1]["t_total_s"]), flush=True)
    ver = veredito(rs)
    imprime(rs, ver)
    json.dump({"mundos": rs, "veredito": ver},
              open(os.path.join(SAIDA, "etapa1_%s.json" % rotulo), "w"), indent=1, default=float)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "checagem":
        main((101,), (4, 2, 4), "checagem")
    else:
        main()
