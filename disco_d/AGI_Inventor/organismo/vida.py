"""
organismo.vida — uma vida inteira de 120 problemas, sem reiniciar
===================================================================
Tres fases. O organismo nunca e avisado de fase, mundo ou troca.

  Fase 1 (0-39)   kepler, mola, arrasto, oblato, einstein, spin_orbita (precisa
                  de h), relatos confiaveis (fonte A) e FICCAO (vortice, fonte F:
                  precisa da direcao girada). Perguntas: sentido, raio_cresce.
  Fase 2 (40-79)  aparece o MAGNETICO (precisa da direcao girada + E). Voltam
                  os mundos da fase 1. Perguntas: escapa (E), sentido (h),
                  periapse_direita (LRL: conceito construido SOBRE h).
  Fase 3 (80-119) aparece o CICLOTRON (precisa da direcao girada; contexto
                  distante de onde ela nasceu). Perguntas LACRADAS: tres moldes
                  sorteados de MOLDES_RESERVA antes de qualquer rodada final.

Orcamento EXTERNO por problema (unidades de computo do material). A sobra pode
virar sonho (so no braco com sonho).
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time

import numpy as np

import mundo_fisico as M
import organismo as O

ORC_L = 300.0
ORC_Q = 120.0


def agenda(semente, lacrados):
    rng = np.random.default_rng(semente)
    P, F_, A_ = "percebido", "relato:F", "relato:A"

    def L(f, n, fonte=P):
        return [(f, fonte, "L", None)] * n

    def Q(f, m, n):
        return [(f, P, "Q", m)] * n
    f1 = (L("kepler", 4) + L("mola", 4) + L("arrasto", 4) + L("oblato", 4)
          + L("einstein", 4) + L("spin_orbita", 6) + L("kepler", 2, A_) + L("mola", 2, A_)
          + L("vortice", 5, F_) + Q("kepler", "sentido", 2) + Q("mola", "sentido", 1)
          + Q("kepler", "raio_cresce", 2))
    f2 = (L("magnetico", 6) + L("spin_orbita", 4) + L("einstein", 2) + L("oblato", 2)
          + L("kepler", 4) + L("arrasto", 2) + L("mola", 2) + L("vortice", 3, F_)
          + L("kepler", 1, A_) + L("mola", 1, A_)
          + Q("kepler", "escapa", 3) + Q("magnetico", "escapa", 2)
          + Q("spin_orbita", "sentido", 1) + Q("magnetico", "sentido", 1)
          + Q("kepler", "sentido", 1) + Q("kepler", "periapse_direita", 4)
          + Q("kepler", "raio_cresce", 1))
    f3 = (L("ciclotron", 6) + L("magnetico", 4) + L("spin_orbita", 3) + L("kepler", 4)
          + L("mola", 2) + L("einstein", 2) + L("oblato", 1) + L("arrasto", 1)
          + L("vortice", 2, F_) + Q("magnetico", "escapa", 1) + Q("kepler", "periapse_direita", 1)
          + Q("spin_orbita", "sentido", 1))
    for m in lacrados:
        f3 += Q("kepler", m, 2) + Q("magnetico", m, 1) + Q("spin_orbita", m, 1)
    fases = []
    for f in (f1, f2, f3):
        f = list(f)
        rng.shuffle(f)
        fases.append(f)
    itens = []
    for k, f in enumerate(fases):
        for it in f:
            itens.append((k + 1,) + it)
    return itens, int(rng.integers(2 ** 31))


def vive(braco, semente, lacrados, pasta, n_max=None, ablacao=True):
    os.makedirs(pasta, exist_ok=True)
    arq = os.path.join(pasta, f"{braco}_s{semente}.json")
    if os.path.exists(arq):
        return json.load(open(arq, encoding="utf-8"))
    itens, sem_mundo = agenda(semente, lacrados)
    if n_max:
        itens = itens[:n_max]
    rng_mundo = np.random.default_rng(sem_mundo)
    flags = O.BRACOS[braco]
    log = open(os.path.join(pasta, f"{braco}_s{semente}.jsonl"), "w", encoding="utf-8")
    org = O.Organismo(flags, semente * 7 + 1, log=log)
    vistos = set()
    linhas = []
    t0 = time.time()
    for idx, (fase, fam, fonte, tipo, molde) in enumerate(itens):
        com_traj = True if tipo == "Q" else bool(rng_mundo.random() < 0.5)
        prob = M.Problema(idx, fam, fonte, tipo, rng_mundo, com_traj=com_traj, molde=molde,
                          n_traj=8)
        orc = ORC_L if tipo == "L" else ORC_Q
        recorrente = (fam, tipo) in vistos
        abl = None
        if ablacao and tipo == "Q" and molde in lacrados:
            # contrafactual: a MESMA mente, sem os conceitos (ablacao), no mesmo problema
            copia = _sem_log(org)
            for e in list(copia.E.values()):
                if e.tipo == "conceito" or (e.tipo in ("lei", "coalizao") and e.pais):
                    copia.E.pop(e.id, None)
            ok_abl, _ = copia.resolve(prob, orc)
            abl = bool(ok_abl)
        ok, ws = org.resolve(prob, orc)
        if fonte == "percebido":
            vistos.add((fam, tipo))
        linhas.append(dict(idx=idx, fase=fase, fam=fam, fonte=fonte, tipo=tipo, molde=molde,
                           ok=bool(ok), gasto=ws.gasto, recorrente=recorrente,
                           lacrado=molde in lacrados if molde else False, ok_ablacao=abl,
                           acoes=len(ws.acoes), usados=(ws.melhor or {}).get("usados", []),
                           desc=(ws.melhor or {}).get("desc") or
                           (repr(ws.melhor["resposta"]) if ws.melhor and tipo == "L" else None)))
    log.close()
    prof = [e.payload["c"].profundidade for e in org.E.values() if e.tipo == "conceito"]
    escopos = []
    for e in org.E.values():
        if not e.regs or e.tipo not in ("conceito", "direcao", "op", "lei", "aberto"):
            continue
        pi = org.escopo(e)
        pagou = sorted({f for f, o in e.fams if o > 0 and f})
        escopos.append(dict(id=e.id, tipo=e.tipo, n=len(e.regs),
                            beta_medio=float(pi @ O.GRADE_BETA), p_global=float(pi[0]),
                            familias_pagou=pagou,
                            desc=(O._descreve(e) if e.tipo != "op" else e.id)))
    res = dict(braco=braco, semente=semente, lacrados=list(lacrados), linhas=linhas,
               segundos=time.time() - t0, n_estruturas=len(org.E),
               tipos={t: sum(1 for e in org.E.values() if e.tipo == t)
                      for t in {e.tipo for e in org.E.values()}},
               prof_max=max(prof, default=0), T_final=org.T_vida, tau=org.tau,
               fontes={k: v[0] / sum(v) for k, v in org.fonte_cred.items()},
               escopos=escopos)
    json.dump(res, open(arq, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return res


def _sem_log(org):
    log = org.log
    org.log = None
    try:
        c = copy.deepcopy(org)
    finally:
        org.log = log
    return c


if __name__ == "__main__":
    braco = sys.argv[1]
    sem = int(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else None
    r = vive(braco, sem, ("periodo_longo", "rmax_grande", "vmax_alta"), "resultados/fumaca", n_max=n)
    ok = [l["ok"] for l in r["linhas"]]
    print(braco, sem, f"{sum(ok)}/{len(ok)}", f"{r['segundos']:.0f}s", r["tipos"])
