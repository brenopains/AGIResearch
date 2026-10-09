"""
separa_voltas — POS-HOC (nao estava no pre-registro da etapa 1), rotulado assim
================================================================================

O piloto passou E1-rev, mas o revisavel escolhia entre 3 voltas NA SELECAO; nao
mediu, na avaliacao, se o ganho vem da revisao por CONTINUIDADE (volta 1) ou da
revisao pela DESCOBERTA (volta 2: religar usando q). A hipotese da proposta e a
segunda. Aqui as voltas sao avaliadas lado a lado nos mesmos mundos 1-3, mais um
PLACEBO da volta 2: religar usando uma combinacao QUALQUER dos mesmos 14 monomios
(nao conservada), na mesma escala. Se o placebo ajudar tanto quanto q, o ganho e
"mais uma passada de religacao", nao a descoberta.

Leitura combinada antes de rodar: a descoberta corrige a percepcao se, em >= 2 de
3 mundos, a volta 2 tiver pureza >= volta 1 E lam_aval menor que a volta 1 E
menor que o placebo. (Pos-hoc: e evidencia, nao veredito.)
"""

from __future__ import annotations

import json
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

import descoberta as D
import etapa1 as E
import percepcao as P


def voltas(semente):
    conj, verd, _cam = E.prepara(semente)
    ev = []
    segs = {k: [E.segmentos_revisavel(frs, ts, ev) for frs, ts in v] for k, v in conj.items()}
    base = {k: [E.entidades_congelado(frs) for frs, ts in v] for k, v in conj.items()}
    r = {"volta0 sem revisao": E._descobre(base, conj)}
    e1 = {k: [P.religar(s, ts, ev) for s, (frs, ts) in zip(segs[k], conj[k])] for k in conj}
    r1 = E._descobre(e1, conj)
    r["volta1 continuidade"] = r1
    sig = D.sigma_dentro(r1["mod"], [Z for _T, Z, _A, _q in r1["est"]["ajuste"]])
    e2 = {k: [P.religar(s, ts, ev, E._qfun(r1["mod"]), sig) for s, (frs, ts)
              in zip(segs[k], conj[k])] for k in conj}
    r["volta2 usando q"] = E._descobre(e2, conj)
    rng = np.random.default_rng(7000 + semente)
    w = rng.normal(size=len(r1["mod"]["w"]))
    mp = dict(r1["mod"], w=w / np.linalg.norm(w))
    sig_p = D.sigma_dentro(mp, [Z for _T, Z, _A, _q in r1["est"]["ajuste"]])
    ep = {k: [P.religar(s, ts, ev, E._qfun(mp), sig_p) for s, (frs, ts)
              in zip(segs[k], conj[k])] for k in conj}
    r["volta2 PLACEBO"] = E._descobre(ep, conj)
    return {n: (E.avalia(x, conj, verd) if x else None) for n, x in r.items()}


def main():
    out = {}
    for s in (1, 2, 3):
        out[s] = voltas(s)
        print("MUNDO %d" % s, flush=True)
        for n, a in out[s].items():
            if a is None:
                print("  %-22s nenhuma grandeza" % n)
                continue
            print("  %-22s corr %.3f | lam_aval %.3f | lam_sel %.3f | pureza %.3f | "
                  "rolagem %.2f -> %.2f" % (n, a["corr"], a["lam_aval"], a["lam_sel"],
                                             a["pureza"], a["rolagem"][0], a["rolagem"][1]))
    ok = 0
    for s, r in out.items():
        a1, a2, ap = r["volta1 continuidade"], r["volta2 usando q"], r["volta2 PLACEBO"]
        if a1 and a2 and ap and a2["pureza"] >= a1["pureza"] and \
                a2["lam_aval"] < a1["lam_aval"] and a2["lam_aval"] < ap["lam_aval"]:
            ok += 1
    print("\nLEITURA (pos-hoc): a descoberta corrige a percepcao em %d de 3 mundos -> %s"
          % (ok, "SIM" if ok >= 2 else "NAO SE SUSTENTA"))
    json.dump({str(k): v for k, v in out.items()},
              open("resultados/separa_voltas.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
