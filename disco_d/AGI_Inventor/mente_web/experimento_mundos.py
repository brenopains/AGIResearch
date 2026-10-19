"""
mente_web.experimento_mundos — o que a aba MUNDOS mostra e VERDADE sobre o mundo?
==================================================================================
Pre-registro: PRE_REGISTRO_mundos.md (escrito ANTES da rodada que vale).

teste_mundos.py ja confere que a tela mostra exatamente o que ela tem (fidelidade). Aqui a
pergunta e outra: o que ela tem, do jeito que a tela mostra, bate com o MUNDO? O avaliador
conhece a arena (camera e leis) e confere, numa vida de 48 videos (giro, mola_h, contragiro,
mola; 12 de cada, sem aviso; oclusor em ~50%):

  W1 um mundo dela e um mundo   cada teoria com >= 6 videos vem de UM universo da arena
  W2 as setas apontam certo     onde ela VIU (>= 5 estados dela perto), a aceleracao da lei
                                dela aponta como a verdadeira (cosseno)
  W3 vale onde ela nunca viu    onde ela NAO viu (0 estados perto), a lei dela continua apontando
                                certo? (generalizacao; a hipotese original dos projetistas, 'la ela
                                erra mais', caiu no dev: la ela acerta igual)
  W4 as frases estao certas     'gira' com o sentido verdadeiro na tela; 'puxa para um centro'
                                no centro verdadeiro; nenhuma frase falsa
  W5 imaginar daqui             comecos novos dela: a lei dela erra menos que a inercia aos 2 s
  W6 o refinamento e real       em videos NOVOS, a lei da ultima epoca preve melhor que a da
                                primeira

Uso:
  python experimento_mundos.py --vidas 10 --semente 20261320 --proc 4 --tag registrada
  python experimento_mundos.py --vidas 3 --semente 101 --proc 3 --tag dev
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import time

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arena as AR  # noqa: E402
import laboratorio as LB  # noqa: E402
import mundos as MU  # noqa: E402

PLANO = [m for m in AR.UNIVERSOS for _ in range(12)]
MIN_VIDEOS = 6


def acel_verdadeira(cam, mundo, Z):
    """aceleracao VERDADEIRA nos estados percebidos Z (coordenadas dela)."""
    A, tc = cam
    Ai = np.linalg.inv(A)
    x = (Z[:, :2] * 64 + 64 - tc) @ Ai.T
    v = (Z[:, 2:] * 64) @ Ai.T
    a = AR.acel(mundo, np.concatenate([x, v], 1))
    return (a @ A.T) / 64.0


def cosseno(a, b):
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    return (a * b).sum(1) / np.maximum(na * nb, 1e-12)


def erro_com_lei(lab, per, lei, conceito):
    """1a tentativa (2 s) de uma lei qualquer (com o conceito como era) num video que ela nao
    guarda. -> erro mediano px ou None."""
    if not per["trilhas"]:
        return None
    tmp = ("teste", lab.n_ep)
    lab.eps[tmp] = {"trilhas": [{k: tr[k] for k in ("t", "p", "quebras")} for tr in per["trilhas"]]}
    c = lei.get("conceito") if lei else None
    velho = lab.conceitos.get(c) if c is not None else None
    try:
        if c is not None and conceito is not None:
            lab.conceitos[c] = conceito
        if c is not None and c not in lab.conceitos:
            return None
        if not len(lab._der(tmp)["z0"]):
            return None
        return lab._erro_ep(lei, [tmp])[0]
    finally:
        lab.eps.pop(tmp, None)
        lab._cache.pop(tmp, None)
        if c is not None and velho is not None:
            lab.conceitos[c] = velho


def uma_vida(sem):
    t0 = time.time()
    cam = AR.camera(np.random.default_rng(sem))
    lab = LB.Laboratorio(sem)
    uni_de = {}
    for k, m in enumerate(PLANO):
        ep = AR.episodio(m, np.random.default_rng([sem, k]), cam)
        per = LB.percebe(ep.quadros, ep.tempos)
        r = lab.observa(None, None, fonte="arena", url=f"arena://{k}", percepcao=per)
        if r.get("episodio") is not None:
            uni_de[r["episodio"]] = m
            lab.guarda_cena(r["episodio"], LB.cena(ep.quadros, per))
    centro = (cam[1] - 64) / 64
    mundos = []
    for T in sorted(lab.teorias.values(), key=lambda T: -len(T["membros"])):
        if len(T["membros"]) < MIN_VIDEOS:
            continue
        cont = collections.Counter(uni_de.get(e) for e in T["membros"])
        u, n_u = cont.most_common(1)[0]
        w = {"teoria": T["id"], "videos": len(T["membros"]), "universo": u, "pureza": round(n_u / len(T["membros"]), 3),
             "conceito": T["lei"].get("conceito") if T["lei"] else None}
        # W2/W3: grade de sondas (lugares x direcoes, rapidez tipica dela), classificada pelo 'visto'
        Zv = np.concatenate([lab._der(e)["Z"] for e in T["membros"] if len(lab._der(e)["Z"])])
        rap = float(np.median(np.linalg.norm(Zv[:, 2:], axis=1)))
        g = np.linspace(-0.85, 0.85, 8)
        sond = np.array([[x, y, rap * math.cos(a), rap * math.sin(a)] for x in g for y in g
                         for a in np.arange(8) * math.pi / 4])
        visto = np.array([(np.linalg.norm(Zv[:, :2] - s[:2], axis=1) < 0.2).sum() for s in sond])
        a_ela = LB.acel_lei(T["lei"], sond, lab.conceitos)
        a_ver = acel_verdadeira(cam, u, sond)
        forte = np.linalg.norm(a_ver, axis=1) > 0.1 * np.median(np.linalg.norm(a_ver, axis=1))
        cs = cosseno(a_ela, a_ver)
        sv, nv = forte & (visto >= 5), forte & (visto == 0)
        w["cos_visto"] = round(float(np.median(cs[sv])), 3) if sv.any() else None
        w["cos_nao_visto"] = round(float(np.median(cs[nv])), 3) if nv.any() else None
        w["n_visto"], w["n_nao_visto"] = int(sv.sum()), int(nv.sum())
        # W4: as frases contra a verdade
        fr = MU.frases_da_lei(lab, T, Zv)
        chaves = {f["chave"]: f for f in fr}
        Zt = Zv[np.linalg.norm(Zv[:, 2:], axis=1) > 1e-3]
        at = acel_verdadeira(cam, u, Zt)
        cr = Zt[:, 2] * at[:, 1] - Zt[:, 3] * at[:, 0]
        sentido_v = "horario" if (cr > 0).mean() >= 0.5 else "anti-horario"
        if u in ("giro", "contragiro"):
            ok = "gira" in chaves and chaves["gira"]["sentido"] == sentido_v and "centro" not in chaves
        else:
            ok = "centro" in chaves and float(np.linalg.norm(np.array(chaves["centro"]["centro"]) - centro)) <= 0.15
        w["frases"] = [f["frase"] for f in fr]
        w["frase_certa"] = bool(ok)
        w["sentido_verdadeiro"] = sentido_v
        w["sentido_dela"] = chaves.get("gira", {}).get("sentido")
        w["centro_dela"] = chaves.get("centro", {}).get("centro")
        # W5: comecos novos dela, filme real da arena, aos 2 s
        rng = np.random.default_rng([sem, 61, T["id"]])
        Z0 = lab.estados_imaginaveis(T, 6, rng)
        e5 = []
        if Z0 is not None:
            ep = AR.episodio(u, np.random.default_rng([sem, 8100 + T["id"]]), cam, dur=2.4, armadilhas=False,
                             estados0=[AR.de_percebido(cam, s) for s in Z0])
            k2 = int(np.argmin(np.abs(np.asarray(ep.tempos) - 2.0)))
            real = np.stack([(c["px"][k2] - 64) / 64 for c in ep.verdade["corpos"]])
            ela = lab._posicoes(T["lei"], Z0, (float(ep.tempos[k2]),))[:, 0]
            ret = lab._posicoes(None, Z0, (float(ep.tempos[k2]),))[:, 0]
            e5 = [float(np.median(np.linalg.norm(ela - real, axis=1)) * 64), float(np.median(np.linalg.norm(ret - real, axis=1)) * 64)]
        w["imaginar_daqui_px"] = [round(x, 3) for x in e5] if e5 else None
        # W6: primeira x ultima epoca em videos NOVOS
        eps = T.get("epocas", [])
        if len(eps) >= 4:
            ef, el = [], []
            for r_ in range(12):
                ep = AR.episodio(u, np.random.default_rng([sem, 9100 + 100 * T["id"] + r_]), cam)
                per = LB.percebe(ep.quadros, ep.tempos)
                a = erro_com_lei(lab, per, eps[0]["lei"], eps[0]["conceito"])
                b = erro_com_lei(lab, per, eps[-1]["lei"], eps[-1]["conceito"])
                if a is not None and b is not None:
                    ef.append(a)
                    el.append(b)
            w["epocas"] = len(eps)
            w["refinamento_px"] = [round(float(np.median(ef)), 3), round(float(np.median(el)), 3)] if ef else None
        mundos.append(w)
    cobertura = {}
    for u in AR.UNIVERSOS:
        tot = sum(1 for e, m in uni_de.items() if m == u)
        melhor = 0
        for T in lab.teorias.values():
            melhor = max(melhor, sum(1 for e in T["membros"] if uni_de.get(e) == u))
        cobertura[u] = round(melhor / max(tot, 1), 3)
    return {"semente": sem, "mundos": mundos, "cobertura": cobertura, "segundos": round(time.time() - t0, 1)}


def placar(vidas):
    P = {"n_vidas": len(vidas)}
    ws = [w for v in vidas for w in v["mundos"]]
    P["W1_pureza"] = {"mundos": len(ws), "puros_0_8": sum(w["pureza"] >= 0.8 for w in ws),
                      "fracao": round(sum(w["pureza"] >= 0.8 for w in ws) / max(len(ws), 1), 3),
                      "vidas_cada_universo_com_mundo": sum(all(x >= 0.5 for x in v["cobertura"].values()) for v in vidas)}

    def med(xs):
        xs = [x for x in xs if x is not None]
        return float(np.median(xs)) if xs else None
    cv = [med([w["cos_visto"] for w in v["mundos"]]) for v in vidas]
    cn = [med([w["cos_nao_visto"] for w in v["mundos"]]) for v in vidas]
    P["W2_setas_onde_viu"] = {"cos_por_vida": [None if x is None else round(x, 3) for x in cv],
                              "vidas_cos_0_8": sum(1 for x in cv if x is not None and x >= 0.8)}
    par = [(a, b) for a, b in zip(cv, cn) if a is not None and b is not None]
    P["W3_vale_onde_nao_viu"] = {"cos_nao_visto_por_vida": [None if x is None else round(x, 3) for x in cn],
                                 "vidas_avaliaveis": sum(1 for x in cn if x is not None),
                                 "vidas_cos_0_8": sum(1 for x in cn if x is not None and x >= 0.8),
                                 "descritivo_vidas_nao_visto_pior": sum(b < a for a, b in par)}
    P["W4_frases"] = {"mundos": len(ws), "certas": sum(w["frase_certa"] for w in ws),
                      "fracao": round(sum(w["frase_certa"] for w in ws) / max(len(ws), 1), 3),
                      "por_universo": {u: [sum(w["frase_certa"] for w in ws if w["universo"] == u),
                                           sum(1 for w in ws if w["universo"] == u)] for u in AR.UNIVERSOS}}
    e5 = [med([w["imaginar_daqui_px"][0] - w["imaginar_daqui_px"][1] for w in v["mundos"] if w["imaginar_daqui_px"]])
          for v in vidas]
    P["W5_imaginar_daqui"] = {"vidas_lei_melhor_que_reto": sum(1 for x in e5 if x is not None and x < 0),
                              "vidas_avaliaveis": sum(1 for x in e5 if x is not None)}
    e6 = [med([w["refinamento_px"][1] - w["refinamento_px"][0] for w in v["mundos"] if w.get("refinamento_px")])
          for v in vidas]
    P["W6_refinamento"] = {"vidas_ultima_melhor": sum(1 for x in e6 if x is not None and x < 0),
                           "vidas_avaliaveis": sum(1 for x in e6 if x is not None),
                           "diferenca_px_por_vida": [None if x is None else round(x, 3) for x in e6]}
    V = {"W1": P["W1_pureza"]["fracao"] >= 0.8 and P["W1_pureza"]["vidas_cada_universo_com_mundo"] >= 8,
         "W2": P["W2_setas_onde_viu"]["vidas_cos_0_8"] >= 8,
         "W3": P["W3_vale_onde_nao_viu"]["vidas_avaliaveis"] >= 6
         and P["W3_vale_onde_nao_viu"]["vidas_cos_0_8"] >= 0.8 * P["W3_vale_onde_nao_viu"]["vidas_avaliaveis"],
         "W4": P["W4_frases"]["fracao"] >= 0.8,
         "W5": P["W5_imaginar_daqui"]["vidas_lei_melhor_que_reto"] >= 8,
         "W6": P["W6_refinamento"]["vidas_avaliaveis"] >= 6
         and P["W6_refinamento"]["vidas_ultima_melhor"] >= 0.7 * P["W6_refinamento"]["vidas_avaliaveis"]}
    P["VEREDITOS"] = {k: "PASSA" if x else "FALHA" for k, x in V.items()}
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vidas", type=int, default=10)
    ap.add_argument("--semente", type=int, default=20261320)
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
    arq = os.path.join(pasta, f"mundos_{a.tag}.json")
    with open(arq, "w", encoding="utf-8") as f:
        json.dump({"placar": P, "vidas": vidas, "segundos": round(time.time() - t0, 1)}, f, ensure_ascii=False,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(json.dumps(P, ensure_ascii=False, indent=1, default=str))
    print(f"({time.time() - t0:.0f} s) -> {arq}")


if __name__ == "__main__":
    main()
