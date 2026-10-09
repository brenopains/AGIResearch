"""
mente_web.experimento_laboratorio — o laboratorio contra os seus controles, na arena
=====================================================================================
Pre-registro: PRE_REGISTRO_laboratorio.md (escrito ANTES da rodada que vale).

Uma VIDA = 72 episodios (videos de 24 s, 5 quadros/s, 128x128) de uma arena com
camera desconhecida, cujas leis trocam sem aviso:
    0-15 giro | 16-31 mola_h | 32-47 contragiro | 48-63 mola | 64-71 giro de novo
A percepcao (pixels -> trilhas) roda UMA vez por episodio e e a mesma para todos os
bracos; os bracos so diferem no que o laboratorio faz com ela.

Bracos:
  COMPLETO           o orgao como vai para a mente viva
  CONCLUSOES         a lei de nascimento fica; conceito so entra como correcao
                     empilhada; nada guardado muda de teoria (caderno: "leis velhas")
  SEM_CONCEITO       sem invencao
  PLACEBO            a invencao roda sobre pontos SORTEADOS entre trajetorias (mesmo
                     alfabeto, mesmo procedimento, sem estrutura) e passa pela mesma
                     sala de espera
  SEM_TRANSFERENCIA  cada teoria so pode usar conceitos que ela mesma inventou

Uso:
  python experimento_laboratorio.py --vidas 10 --semente 20261010 --proc 4 --tag registrada
  python experimento_laboratorio.py --vidas 1 --semente 102 --tag dev     (checagem)
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

PLANO = ["giro"] * 16 + ["mola_h"] * 16 + ["contragiro"] * 16 + ["mola"] * 16 + ["giro"] * 8
FASES = {"giro": (0, 16), "mola_h": (16, 32), "contragiro": (32, 48), "mola": (48, 64), "retorno": (64, 72)}
BRACOS = {
    "COMPLETO": {},
    "CONCLUSOES": {"olhos_frescos": False, "reinterpreta": False},
    "SEM_CONCEITO": {"inventa": False},
    "PLACEBO": {"inventa": "placebo"},
    "SEM_TRANSFERENCIA": {"transfere": False},
}


# ── percepcao e verdade (uma vez por vida) ────────────────────────────────────
def _cobre(tr, corpo, k_frame, tol=3.0):
    """a trilha esta sobre o corpo no quadro k?"""
    j = np.nonzero(tr["k"] == k_frame)[0]
    if not len(j):
        return False
    p = tr["p"][j[0]] * 64 + 64
    return float(np.linalg.norm(p - corpo["px"][k_frame])) <= tol


def permanencia(ep, per):
    """eventos de oclusao (visivel -> escondido >= 2 quadros -> visivel): a MESMA trilha
    cobre o corpo antes e depois?"""
    ok = tot = 0
    for c in ep.verdade["corpos"]:
        vis = c["visivel"]
        k = 1
        while k < len(vis):
            if vis[k - 1] and not vis[k]:
                a = k - 1
                b = k
                while b < len(vis) and not vis[b]:
                    b += 1
                if b < len(vis) and b - a - 1 >= 2:
                    tot += 1
                    if any(_cobre(tr, c, a) and _cobre(tr, c, b) for tr in per["trilhas"]):
                        ok += 1
                k = b
            k += 1
    return ok, tot


def referencia(sem, cam):
    """estados percebidos de episodios NOVOS de cada mundo + as grandezas verdadeiras."""
    Zs = []
    for i, m in enumerate(AR.UNIVERSOS):
        for r in range(3):
            ep = AR.episodio(m, np.random.default_rng([sem, 9000 + 10 * i + r]), cam)
            for tr in LB.percebe(ep.quadros, ep.tempos)["trilhas"]:
                Z, _A, ok = LB.estados(tr)
                Zs.append(Z[ok])
    Z = np.concatenate(Zs)
    Zpx = np.concatenate([Z[:, :2] * 64 + 64, Z[:, 2:] * 64], 1)
    return Z, AR.verdadeiras_px(cam, Zpx)


def controle_permutado(sem, cam):
    """O inventor DE VERDADE (com a regua) sobre trajetorias do giro: (a) como foram
    percebidas; (b) com os pontos SORTEADOS entre as trajetorias (nada pode ser constante
    por trajetoria). -> (aceitas reais, aceitas permutadas, trajetorias)."""
    rng = np.random.default_rng([sem, 7777])
    trajs = []
    for r in range(12):
        ep = AR.episodio("giro", np.random.default_rng([sem, 8000 + r]), cam)
        for tr in LB.percebe(ep.quadros, ep.tempos)["trilhas"]:
            Z, _A, ok = LB.estados(tr)
            trajs.append(Z[ok])
    reais = len(LB.inventa(trajs, np.random.default_rng([sem, 1])))
    tam = [len(z) for z in trajs]
    todos = np.concatenate(trajs)[rng.permutation(sum(tam))]
    perm = len(LB.inventa(list(np.split(todos, np.cumsum(tam)[:-1])), np.random.default_rng([sem, 1])))
    return reais, perm, len(trajs)


def percebe_vida(sem):
    cam = AR.camera(np.random.default_rng(sem))
    pers, perm, perm0, armad = [], [0, 0], [0, 0], []
    for k, m in enumerate(PLANO):
        ep = AR.episodio(m, np.random.default_rng([sem, k]), cam)
        per = LB.percebe(ep.quadros, ep.tempos)
        pers.append(per)
        if ep.verdade["oclusor"] is not None:
            a, b = permanencia(ep, per)
            perm[0] += a
            perm[1] += b
            a, b = permanencia(ep, LB.percebe(ep.quadros, ep.tempos, religa=False))
            perm0[0] += a
            perm0[1] += b
        armad.append({"oclusor": ep.verdade["oclusor"] is not None, "mesma_cor": ep.verdade["mesma_cor"]})
    return cam, pers, {"permanencia": perm, "permanencia_congelada": perm0}


# ── uma vida de um braco ──────────────────────────────────────────────────────
def identidade(lab, cid, Zref, vref):
    if cid is None or cid not in lab.conceitos:
        return None
    q = LB.valor_conceito(lab.conceitos[cid], Zref)
    return {k: round(float(abs(np.corrcoef(q, vref[k])[0, 1])), 4) for k in ("s2", "h")}


def recuperadas(lab, ks):
    """falhas antigas: os 4 primeiros episodios de uma fase sao explicados HOJE (por
    alguma teoria, abaixo do ruido dela e melhor que a inercia)?"""
    n = 0
    for e in ks:
        if e not in lab.eps:
            continue
        m = lab._m(e)
        for T in lab.teorias.values():
            er = lab._razao_ep(T["lei"], [e])[0]
            if er is not None and er <= lab._limiar(T, m) and er < 1.0:
                n += 1
                break
    return n


def roda_braco(args):
    sem, braco, pers, Zref, vref = args
    lab = LB.Laboratorio(sem, **BRACOS[braco])
    hist, fim_fase = [], {}
    eid_de_k = {}
    t0 = time.time()
    for k, (m, per) in enumerate(zip(PLANO, pers)):
        r = lab.observa(None, None, fonte="arena", percepcao=per)
        if "episodio" in r:
            eid_de_k[k] = r["episodio"]
        hist.append({"k": k, "mundo": m, "crise": bool(r.get("crise")), "inercia": r.get("inercia"),
                     "primeira": r.get("primeira"), "razao": r.get("razao"), "teoria": r.get("teoria"), "conceito": r.get("conceito"),
                     "teoria_prevista": r.get("teoria_prevista")})
        for nome, (a, b) in FASES.items():
            if k == b - 1:
                fim_fase[nome] = {"conceito": r.get("conceito"), "teoria": r.get("teoria"),
                                  "identidade": identidade(lab, r.get("conceito"), Zref, vref),
                                  "nasceu_na_teoria": lab.conceitos[r["conceito"]]["de_teoria"]
                                  if r.get("conceito") in lab.conceitos else None,
                                  "nasceu_no_episodio": lab.conceitos[r["conceito"]]["nasceu"]
                                  if r.get("conceito") in lab.conceitos else None}
    rec = {nome: recuperadas(lab, [eid_de_k[k] for k in range(a, a + 4) if k in eid_de_k])
           for nome, (a, b) in FASES.items() if nome != "retorno"}
    return {"semente": sem, "braco": braco, "hist": hist, "fim_fase": fim_fase, "contas": lab.contas,
            "recuperadas": rec, "segundos": round(time.time() - t0, 1),
            "conceitos": {c: {"formula": LB.formula_conceito(x), "nasceu": x["nasceu"], "de_teoria": x["de_teoria"],
                              "identidade": identidade(lab, c, Zref, vref), "revisoes": x.get("revisoes", 0)}
                          for c, x in lab.conceitos.items()}}


def uma_vida(sem, bracos):
    t0 = time.time()
    cam, pers, perc = percebe_vida(sem)
    Zref, vref = referencia(sem, cam)
    ctl = controle_permutado(sem, cam)
    out = [roda_braco((sem, b, pers, Zref, vref)) for b in bracos]
    return {"semente": sem, "percepcao": perc,
            "controle_permutado": {"aceitas_reais": ctl[0], "aceitas_permutadas": ctl[1], "trajetorias": ctl[2]},
            "bracos": out, "segundos": round(time.time() - t0, 1)}


# ── placar ────────────────────────────────────────────────────────────────────
def erro_fases(hist, fases=("giro", "mola_h", "contragiro")):
    xs = [h["primeira"] for h in hist for f in fases
          if FASES[f][0] + 4 <= h["k"] < FASES[f][1] and h["primeira"] is not None]
    return float(np.median(xs)) if xs else None


def placar(vidas):
    P = {}
    br = {b["braco"]: [] for v in vidas for b in v["bracos"]}
    for v in vidas:
        for b in v["bracos"]:
            br[b["braco"]].append((v, b))
    n = len(vidas)

    def ident(b, fase, chave):
        x = b["fim_fase"].get(fase, {}).get("identidade")
        return x is not None and x[chave] >= 0.99

    if "COMPLETO" in br:
        P["L1_invencao_s2_no_giro"] = sum(ident(b, "giro", "s2") for _v, b in br["COMPLETO"])
        P["L2_revisao_h_na_mola_h"] = sum(ident(b, "mola_h", "h") for _v, b in br["COMPLETO"])
        P["L3_reuso_mesmo_conceito_no_contragiro"] = sum(
            ident(b, "contragiro", "s2") and b["fim_fase"]["contragiro"]["conceito"] == b["fim_fase"]["giro"]["conceito"]
            for _v, b in br["COMPLETO"])
        P["L4_mola_sem_conceito"] = sum(b["fim_fase"].get("mola", {}).get("conceito") is None for _v, b in br["COMPLETO"])

        def crise_na_troca(b, k0):
            return any(h["crise"] for h in b["hist"] if k0 <= h["k"] < k0 + 5)
        P["L7a_crise_nas_3_trocas"] = sum(all(crise_na_troca(b, k0) for k0 in (16, 32, 48)) for _v, b in br["COMPLETO"])
        P["L7b_reconhece_o_giro_de_volta"] = sum(
            (not b["hist"][64]["crise"]) and b["hist"][64]["teoria_prevista"] == b["fim_fase"]["giro"]["teoria"]
            for _v, b in br["COMPLETO"])
        P["crises_por_vida_COMPLETO"] = [sum(h["crise"] for h in b["hist"]) for _v, b in br["COMPLETO"]]
        P["recuperadas_COMPLETO"] = [b["recuperadas"] for _v, b in br["COMPLETO"]]
    E = {nome: [erro_fases(b["hist"]) for _v, b in lst] for nome, lst in br.items()}
    P["erro_1a_tentativa_px_por_braco"] = {k: [round(x, 3) if x is not None else None for x in v] for k, v in E.items()}
    P["erro_1a_mediana_entre_vidas"] = {k: round(float(np.median([x for x in v if x is not None])), 3) for k, v in E.items()}
    inerc = [erro_fases([dict(h, primeira=h["inercia"]) for h in b["hist"]]) for _v, b in br.get("COMPLETO", [])]
    P["inercia_mediana_entre_vidas"] = round(float(np.median(inerc)), 3) if inerc else None
    med = P["erro_1a_mediana_entre_vidas"]

    def vence(a, b_):
        return sum(x < y for x, y in zip(E[a], E[b_]) if x is not None and y is not None)
    if "SEM_CONCEITO" in E and "COMPLETO" in E:
        P["L5_conceito_paga"] = {"razao": round(med["COMPLETO"] / med["SEM_CONCEITO"], 3),
                                 "vidas_completo_melhor": vence("COMPLETO", "SEM_CONCEITO")}
    if "PLACEBO" in E and "SEM_CONCEITO" in E:
        P["L6_placebo"] = {"razao_placebo_sem": round(med["PLACEBO"] / med["SEM_CONCEITO"], 3),
                           "vidas_completo_melhor_que_placebo": vence("COMPLETO", "PLACEBO")}
    if "CONCLUSOES" in E:
        P["L8_reanalise_vence_conclusoes"] = {"vidas": vence("COMPLETO", "CONCLUSOES"),
                                              "razao": round(med["COMPLETO"] / med["CONCLUSOES"], 3)}
        P["recuperadas_CONCLUSOES"] = [b["recuperadas"] for _v, b in br["CONCLUSOES"]]
    if "SEM_TRANSFERENCIA" in br:
        def ini(b):
            xs = [h["primeira"] for h in b["hist"] if 32 <= h["k"] < 40 and h["primeira"] is not None]
            return float(np.median(xs)) if xs else None
        a = [ini(b) for _v, b in br["COMPLETO"]]
        c = [ini(b) for _v, b in br["SEM_TRANSFERENCIA"]]
        P["L9_transferencia"] = {"completo": [round(x, 3) for x in a], "sem_transferencia": [round(x, 3) for x in c],
                                 "vidas_completo_melhor": sum(x < y for x, y in zip(a, c))}
    pm = [sum(v["percepcao"]["permanencia"][i] for v in vidas) for i in (0, 1)]
    p0 = [sum(v["percepcao"]["permanencia_congelada"][i] for v in vidas) for i in (0, 1)]
    P["P1_permanencia"] = {"religar": pm, "fracao": round(pm[0] / max(pm[1], 1), 3),
                           "congelada": p0, "fracao_congelada": round(p0[0] / max(p0[1], 1), 3)}
    P["C1_permutado_nenhuma_aceita"] = sum(v["controle_permutado"]["aceitas_permutadas"] == 0 for v in vidas)
    P["C1_reais_alguma_aceita"] = sum(v["controle_permutado"]["aceitas_reais"] > 0 for v in vidas)
    P["n_vidas"] = n
    # vereditos pelos criterios do PRE_REGISTRO_laboratorio.md
    V = {}
    if "COMPLETO" in br:
        V["L1"] = P["L1_invencao_s2_no_giro"] >= 8
        V["L2"] = P["L2_revisao_h_na_mola_h"] >= 8
        V["L3"] = P["L3_reuso_mesmo_conceito_no_contragiro"] >= 8
        V["L4"] = P["L4_mola_sem_conceito"] >= 8
        V["L7a"] = P["L7a_crise_nas_3_trocas"] >= 7
        V["L7b"] = P["L7b_reconhece_o_giro_de_volta"] >= 7
    if "L5_conceito_paga" in P:
        V["L5"] = P["L5_conceito_paga"]["razao"] <= 0.85 and P["L5_conceito_paga"]["vidas_completo_melhor"] >= 8
    if "L6_placebo" in P:
        V["L6"] = P["L6_placebo"]["razao_placebo_sem"] >= 0.90 and P["L6_placebo"]["vidas_completo_melhor_que_placebo"] >= 8
    if "L8_reanalise_vence_conclusoes" in P:
        V["L8"] = P["L8_reanalise_vence_conclusoes"]["vidas"] >= 7
    if "L9_transferencia" in P:
        V["L9"] = P["L9_transferencia"]["vidas_completo_melhor"] >= 7
    V["P1"] = P["P1_permanencia"]["fracao"] >= 0.70 and P["P1_permanencia"]["fracao_congelada"] <= 0.35
    V["C1"] = P["C1_permutado_nenhuma_aceita"] >= 9 and P["C1_reais_alguma_aceita"] >= 9
    P["VEREDITOS"] = {k: ("PASSA" if v else "FALHA") for k, v in V.items()}
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vidas", type=int, default=10)
    ap.add_argument("--semente", type=int, default=20261010)
    ap.add_argument("--proc", type=int, default=1)
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--bracos", default=",".join(BRACOS))
    a = ap.parse_args()
    bracos = a.bracos.split(",")
    sems = [a.semente + i for i in range(a.vidas)]
    t0 = time.time()
    if a.proc > 1:
        from multiprocessing import Pool
        with Pool(a.proc) as p:
            vidas = p.starmap(uma_vida, [(s, bracos) for s in sems])
    else:
        vidas = [uma_vida(s, bracos) for s in sems]
    P = placar(vidas)
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados"), exist_ok=True)
    arq = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados", f"laboratorio_{a.tag}.json")
    with open(arq, "w", encoding="utf-8") as f:
        json.dump({"placar": P, "vidas": vidas, "segundos": round(time.time() - t0, 1), "plano": PLANO},
                  f, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(json.dumps(P, ensure_ascii=False, indent=1, default=str))
    print(f"({time.time() - t0:.0f} s) -> {arq}")


if __name__ == "__main__":
    main()
