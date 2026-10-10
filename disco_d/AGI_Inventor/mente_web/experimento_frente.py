"""
mente_web.experimento_frente — ela deduz QUEM ESTA NA FRENTE (o primeiro degrau do 3D)?
========================================================================================
Pre-registro: PRE_REGISTRO_frente.md (escrito ANTES da rodada que vale).

A mente so recebe quadros planos. A camada da FRENTE (laboratorio.frente) e deduzida de
onde as coisas SOMEM e reaparecem: o caminho escondido passa por tras de algo. O avaliador
sabe onde o oclusor da arena esta (verdade) e confere:
  parte 1  em 24 episodios por vida (os 4 mundos, com as armadilhas de sempre): quando ha
           oclusor, ela acha a camada da frente? acha a forma certa (IoU >= 0,8)? quando nao
           ha, ela inventa uma (falso alarme)?
  parte 2  a imaginacao que RESPEITA a frente (as coisas somem atras dela) bate com o filme
           real melhor do que a imaginacao "achatada" (coisas desenhadas por cima de tudo)?
           Filmes do giro, comecos dela cujo caminho (pela lei dela) cruza a frente deduzida;
           o filme real roda com o MESMO oclusor; erro = o que se move, aos 0-3 s.

Uso:
  python experimento_frente.py --vidas 10 --semente 20261310 --proc 4 --tag registrada
  python experimento_frente.py --vidas 3 --semente 101 --proc 3 --tag dev
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arena as AR  # noqa: E402
import laboratorio as LB  # noqa: E402
from experimento_sonho import _movel  # noqa: E402

N_EP, N_TREINO, N_FILMES = 24, 8, 10


def verdade(oc):
    m = np.zeros((AR.LADO, AR.LADO), bool)
    if oc is not None:
        x, y, w, h = (int(round(v)) for v in oc)
        m[y:y + h, x:x + w] = True
    return m


def uma_vida(sem):
    t0 = time.time()
    cam = AR.camera(np.random.default_rng(sem))
    p1 = []
    for k in range(N_EP):
        m = AR.UNIVERSOS[k % 4]
        ep = AR.episodio(m, np.random.default_rng([sem, k]), cam)
        per = LB.percebe(ep.quadros, ep.tempos)
        c = LB.cena(ep.quadros, per)
        f = c.get("frente") if c else None
        oc = ep.verdade["oclusor"]
        iou = None
        if f is not None and oc is not None:
            v = verdade(oc)
            iou = float((f & v).sum() / max((f | v).sum(), 1))
        p1.append({"k": k, "oclusor": oc is not None, "deduziu": f is not None, "iou": iou})
    # parte 2: imaginacao com e sem a frente, contra o filme real com o mesmo oclusor
    lab = LB.Laboratorio(sem)
    cenas = []
    for k in range(N_TREINO):
        ep = AR.episodio("giro", np.random.default_rng([sem, 300 + k]), cam)
        per = LB.percebe(ep.quadros, ep.tempos)
        r = lab.observa(None, None, fonte="arena", url=f"arena://{300 + k}", percepcao=per)
        if r.get("episodio") is not None:
            c = LB.cena(ep.quadros, per)
            lab.guarda_cena(r["episodio"], c)
            if c is not None and c.get("frente") is not None and ep.verdade["oclusor"] is not None:
                cenas.append((c, ep.verdade["oclusor"]))
    T = lab.teoria_atual("arena")
    filmes = []
    if T is not None and cenas:
        rf = np.random.default_rng([sem, 55])
        tent = 0
        while len(filmes) < N_FILMES and tent < 60:
            tent += 1
            c, oc = cenas[len(filmes) % len(cenas)]
            Z0 = lab.estados_imaginaveis(T, 12, rf)
            if Z0 is None:
                break
            ts = np.arange(16) / AR.FPS
            P = lab._posicoes(T["lei"], Z0, tuple(ts[1:]))
            L, H, W = c["L"], c["forma"][0], c["forma"][1]
            cruza = []
            for i in range(len(Z0)):
                px = np.concatenate([Z0[i:i + 1, :2], P[i]], 0) * (L / 2) + np.array([W / 2, H / 2])
                xi, yi = np.round(px[:, 0]).astype(int), np.round(px[:, 1]).astype(int)
                ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
                cruza.append(bool(ok.any() and c["frente"][yi[ok], xi[ok]].any()))
            sel = [i for i in range(len(Z0)) if cruza[i]][:3]
            if not sel:
                continue
            Z0 = Z0[sel]
            ep = AR.episodio("giro", np.random.default_rng([sem, 7000 + tent]), cam, dur=3.2, armadilhas=False,
                             oclusor=oc, estados0=[AR.de_percebido(cam, u) for u in Z0])
            tt = np.asarray(ep.tempos, float)
            real = _movel(ep.quadros)
            erros = {}
            for nome, prof in (("com_frente", True), ("achatada", False)):
                im = _movel(lab.imagina_video(c, Z0, T["lei"], tt, profundidade=prof)[0])
                erros[nome] = float(np.mean([((a - b) ** 2).mean() for a, b in zip(im[1:], real[1:])]))
            filmes.append(erros)
    return {"semente": sem, "parte1": p1, "filmes": filmes, "cenas_com_frente": len(cenas),
            "segundos": round(time.time() - t0, 1)}


def placar(vidas):
    P = {"n_vidas": len(vidas)}
    com = [e for v in vidas for e in v["parte1"] if e["oclusor"]]
    sem_ = [e for v in vidas for e in v["parte1"] if not e["oclusor"]]
    det = [e for e in com if e["deduziu"]]
    P["F1_deteccao"] = {"episodios_com_oclusor": len(com), "deduziu": len(det),
                        "fracao": round(len(det) / max(len(com), 1), 3)}
    P["F2_forma_certa"] = {"deteccoes": len(det), "iou_maior_0_8": sum(1 for e in det if e["iou"] >= 0.8),
                           "fracao": round(sum(1 for e in det if e["iou"] >= 0.8) / max(len(det), 1), 3),
                           "iou_mediano": round(float(np.median([e["iou"] for e in det])), 3) if det else None}
    P["F3_falso_alarme"] = {"episodios_sem_oclusor": len(sem_), "inventou": sum(e["deduziu"] for e in sem_),
                            "fracao": round(sum(e["deduziu"] for e in sem_) / max(len(sem_), 1), 3)}
    av = []
    for v in vidas:
        if len(v["filmes"]) >= 3:
            a = float(np.median([f["com_frente"] for f in v["filmes"]]))
            b = float(np.median([f["achatada"] for f in v["filmes"]]))
            av.append((v["semente"], a, b))
    P["F4_imaginacao_respeita_a_frente"] = {"vidas_avaliaveis": len(av), "vidas_com_frente_melhor": sum(a < b for _s, a, b in av),
                                            "por_vida": [[s, round(a, 7), round(b, 7)] for s, a, b in av]}
    V = {"F1": P["F1_deteccao"]["fracao"] >= 0.6, "F2": P["F2_forma_certa"]["fracao"] >= 0.9,
         "F3": P["F3_falso_alarme"]["fracao"] <= 0.05,
         "F4": len(av) >= 6 and sum(a < b for _s, a, b in av) >= 0.8 * len(av)}
    P["VEREDITOS"] = {k: "PASSA" if x else "FALHA" for k, x in V.items()}
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vidas", type=int, default=10)
    ap.add_argument("--semente", type=int, default=20261310)
    ap.add_argument("--proc", type=int, default=1)
    ap.add_argument("--tag", default="dev")
    a = ap.parse_args()
    sems = [a.semente + i for i in range(a.vidas)]
    t0 = time.time()
    if a.proc > 1:
        from multiprocessing import Pool
        with Pool(a.proc) as p:
            vidas = p.map(uma_vida, sems)
    else:
        vidas = [uma_vida(s) for s in sems]
    P = placar(vidas)
    pasta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")
    os.makedirs(pasta, exist_ok=True)
    arq = os.path.join(pasta, f"frente_{a.tag}.json")
    with open(arq, "w", encoding="utf-8") as f:
        json.dump({"placar": P, "vidas": vidas, "segundos": round(time.time() - t0, 1)}, f, ensure_ascii=False,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(json.dumps(P, ensure_ascii=False, indent=1, default=str))
    print(f"({time.time() - t0:.0f} s) -> {arq}")


if __name__ == "__main__":
    main()
