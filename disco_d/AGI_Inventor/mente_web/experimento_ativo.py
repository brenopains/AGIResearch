"""
mente_web.experimento_ativo — as MAOS dela contra os seus controles, na arena
==============================================================================
Pre-registro: PRE_REGISTRO_experimento_ativo.md (escrito ANTES da rodada que vale).

Pergunta: escolher ela mesma a cena (onde cada coisa comeca e com que velocidade) para
SEPARAR as explicacoes rivais faz ela aprender o mundo melhor do que so assistir — e
melhor do que mexer no mundo ao acaso?

Uma VIDA = 30 episodios de treino (videos de 24 s, 5 quadros/s, 128x128) de uma arena
com camera desconhecida, leis trocando sem aviso:
    0-9 giro | 10-19 mola_h | 20-29 contragiro
e, ao fim de cada mundo, 6 episodios de TESTE novos daquele mundo (sorteados pelo
mundo, iguais para todos os bracos), previstos pela teoria que explicou o ultimo
episodio de treino, sem aprender com eles.

Bracos (o UNICO que muda e quem escolhe o comeco de cada episodio de treino):
  PASSIVO    o mundo sorteia, como sempre (ela so assiste)
  ALEATORIO  ela monta a cena, mas ao ACASO entre os mesmos candidatos e regras do
             ATIVO (controle: mexer no mundo, cobrir outras rapidezes, 3 corpos sempre)
  ATIVO      ela monta a cena onde as rivais dela mais DISCORDAM
Nos tres, antes de cada episodio ela forma as rivais e, depois, julga qual previu
melhor (resolve_experimento) — o mesmo trabalho de laboratorio; so a escolha difere.
O primeiro episodio de cada vida e sorteado pelo mundo nos tres (sem teoria, sem rivais).

Uso:
  python experimento_ativo.py --vidas 10 --semente 20261011 --proc 4 --tag registrada
  python experimento_ativo.py --vidas 3 --semente 101 --proc 3 --tag dev
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
from experimento_laboratorio import identidade, referencia  # noqa: E402

MUNDOS = [("giro", 10), ("mola_h", 10), ("contragiro", 10)]
PLANO = [m for m, n in MUNDOS for _ in range(n)]
N_TESTE = 6
BRACOS = ("PASSIVO", "ALEATORIO", "ATIVO")
CERTO = {"giro": "s2", "mola_h": "h", "contragiro": "s2"}


def _gm(xs):
    xs = [x for x in xs if x is not None]
    return float(np.exp(np.mean(np.log(np.maximum(xs, 1e-3))))) if xs else None


def roda_braco(sem, braco, cam, passivos, testes, Zref, vref):
    lab = LB.Laboratorio(sem)
    hist, teste, eids = [], {}, {}
    t0 = time.time()
    for k, m in enumerate(PLANO):
        rv = lab.rivais("arena")
        prop = None
        if braco != "PASSIVO" and rv:
            prop = lab.propoe_experimento(rv, aleatorio=(braco == "ALEATORIO"))
        if prop is not None:
            est = AR.limita_experimento(prop["estados"])
            ep = AR.episodio(m, np.random.default_rng([sem, k, 1]), cam,
                             estados0=[AR.de_percebido(cam, u) for u in est])
            per = LB.percebe(ep.quadros, ep.tempos)
        else:
            per = passivos[k]
        r = lab.observa(None, None, fonte="arena", url=f"arena://{k}", percepcao=per)
        res = lab.resolve_experimento(r.get("episodio"), rv, prop)
        if r.get("episodio") is not None:
            eids.setdefault(m, []).append(r["episodio"])
        c = r.get("conceito")
        idt = identidade(lab, c, Zref, vref)
        hist.append({"k": k, "mundo": m, "montou": prop is not None, "trilhas": r.get("trilhas"),
                     "crise": bool(r.get("crise")), "primeira": r.get("primeira"), "inercia": r.get("inercia"),
                     "teoria": r.get("teoria"), "conceito": c, "identidade": idt,
                     "certo": bool(idt is not None and idt[CERTO[m]] >= 0.99),
                     "divergencia_escolhida_px": float(np.mean(prop["divergencia_px"])) if prop else None,
                     "divergencia_tipica_px": prop["divergencia_tipica_px"] if prop else None,
                     "validos": prop["candidatos_validos"] if prop else None,
                     "n_rivais": len(rv), "discriminacao_bits": res["discriminacao_bits"] if res else None,
                     "vencedora_tipo": res["tipo"] if res else None, "revisou": bool(res and res["revisou"])})
        fim = k == len(PLANO) - 1 or PLANO[k + 1] != m
        if fim:
            tid = r.get("teoria")
            if tid not in lab.teorias:
                T = lab.teoria_atual("arena")
                tid = T["id"] if T else None
            prev = [lab.preve_sem_aprender(p, tid) for p in testes[m]]
            erros = [x[0] for x in prev if x is not None]
            iner = [x[1] for x in prev if x is not None]
            conc = lab.teorias[tid]["lei"].get("conceito") if tid in lab.teorias else None
            idt = identidade(lab, conc, Zref, vref)
            # secundario (descritivo): a teoria que guarda MAIS episodios de treino deste mundo
            donos = [lab.eps[e].get("teoria") for e in eids.get(m, []) if e in lab.eps]
            donos = [t for t in donos if t in lab.teorias]
            tmain = max(set(donos), key=donos.count) if donos else None
            prev2 = [lab.preve_sem_aprender(p, tmain) for p in testes[m]] if tmain is not None else []
            erros2 = [x[0] for x in prev2 if x is not None]
            teste[m] = {"erro_px": round(float(np.median(erros)), 3) if erros else None,
                        "erro_px_principal": round(float(np.median(erros2)), 3) if erros2 else None,
                        "teoria_principal": tmain,
                        "inercia_px": round(float(np.median(iner)), 3) if iner else None,
                        "erros": [round(x, 3) for x in erros], "teoria": tid, "conceito": conc, "identidade": idt,
                        "certo": bool(idt is not None and idt[CERTO[m]] >= 0.99)}
    primeiro = {}
    for m, _n in MUNDOS:
        hs = [h for h in hist if h["mundo"] == m]
        ks = [i for i, h in enumerate(hs) if h["certo"]]
        primeiro[m] = ks[0] if ks else None
    return {"braco": braco, "hist": hist, "teste": teste, "primeiro_certo": primeiro, "contas": lab.contas,
            "teste_gm_px": _gm([teste[m]["erro_px"] for m, _n in MUNDOS]), "segundos": round(time.time() - t0, 1)}


def uma_vida(sem, bracos=BRACOS):
    t0 = time.time()
    cam = AR.camera(np.random.default_rng(sem))
    passivos = []
    for k, m in enumerate(PLANO):
        ep = AR.episodio(m, np.random.default_rng([sem, k]), cam)
        passivos.append(LB.percebe(ep.quadros, ep.tempos))
    testes = {}
    for i, (m, _n) in enumerate(MUNDOS):
        testes[m] = []
        for r in range(N_TESTE):
            ep = AR.episodio(m, np.random.default_rng([sem, 6000 + 10 * i + r]), cam)
            testes[m].append(LB.percebe(ep.quadros, ep.tempos))
    Zref, vref = referencia(sem, cam)
    out = [roda_braco(sem, b, cam, passivos, testes, Zref, vref) for b in bracos]
    return {"semente": sem, "bracos": out, "segundos": round(time.time() - t0, 1)}


# ── placar ────────────────────────────────────────────────────────────────────
def placar(vidas):
    P = {"n_vidas": len(vidas)}
    br = {b: [next(x for x in v["bracos"] if x["braco"] == b) for v in vidas] for b in BRACOS
          if all(any(x["braco"] == b for x in v["bracos"]) for v in vidas)}
    G = {b: [x["teste_gm_px"] for x in lst] for b, lst in br.items()}
    P["teste_px_por_vida"] = {b: [round(x, 3) if x is not None else None for x in v] for b, v in G.items()}
    P["teste_px_mediana"] = {b: round(float(np.median([x for x in v if x is not None])), 3) for b, v in G.items()}
    P["teste_px_por_mundo"] = {b: {m: round(float(np.median([x["teste"][m]["erro_px"] for x in lst
                                                            if x["teste"][m]["erro_px"] is not None])), 3)
                                   for m, _n in MUNDOS} for b, lst in br.items()}
    G2 = {b: [_gm([x["teste"][m].get("erro_px_principal") for m, _n in MUNDOS]) for x in lst] for b, lst in br.items()}
    P["secundario_teoria_principal_px_mediana"] = {b: round(float(np.median([x for x in v if x is not None])), 3)
                                                   for b, v in G2.items()}
    if "ATIVO" in G2 and "ALEATORIO" in G2:
        P["secundario_ativo_vence_aleatorio_vidas"] = sum(x < y for x, y in zip(G2["ATIVO"], G2["ALEATORIO"])
                                                          if x is not None and y is not None)
    P["inercia_px_por_mundo"] = {m: round(float(np.median([x["teste"][m]["inercia_px"] for x in br["PASSIVO"]])), 3)
                                 for m, _n in MUNDOS} if "PASSIVO" in br else None

    def vence(a, b):
        return sum(x < y for x, y in zip(G[a], G[b]) if x is not None and y is not None)
    if "ATIVO" in G and "ALEATORIO" in G:
        P["E1_ativo_vence_aleatorio"] = {"vidas": vence("ATIVO", "ALEATORIO"),
                                         "razao": round(P["teste_px_mediana"]["ATIVO"] / P["teste_px_mediana"]["ALEATORIO"], 3)}
    if "ATIVO" in G and "PASSIVO" in G:
        P["E2_ativo_vence_passivo"] = {"vidas": vence("ATIVO", "PASSIVO"),
                                       "razao": round(P["teste_px_mediana"]["ATIVO"] / P["teste_px_mediana"]["PASSIVO"], 3)}
    if "ALEATORIO" in G and "PASSIVO" in G:
        P["aleatorio_vs_passivo"] = {"vidas_aleatorio_melhor": vence("ALEATORIO", "PASSIVO"),
                                     "razao": round(P["teste_px_mediana"]["ALEATORIO"] / P["teste_px_mediana"]["PASSIVO"], 3)}
    P["E3_conceito_certo_no_fim"] = {b: {m: sum(x["teste"][m]["certo"] for x in lst) for m, _n in MUNDOS}
                                     for b, lst in br.items()}
    P["E3_primeiro_episodio_certo"] = {b: {m: [x["primeiro_certo"][m] for x in lst] for m, _n in MUNDOS}
                                       for b, lst in br.items()}

    def disc(x):
        xs = [h["discriminacao_bits"] for h in x["hist"] if h["montou"] and h["discriminacao_bits"] is not None]
        return float(np.median(xs)) if xs else None
    D = {b: [disc(x) for x in lst] for b, lst in br.items() if b != "PASSIVO"}
    P["discriminacao_bits_por_vida"] = {b: [round(x, 2) if x is not None else None for x in v] for b, v in D.items()}
    if "ATIVO" in D and "ALEATORIO" in D:
        P["E4_maos_separam_rivais"] = {"vidas": sum(a > b for a, b in zip(D["ATIVO"], D["ALEATORIO"])
                                                    if a is not None and b is not None)}
    P["trilhas_por_episodio_montado"] = {b: round(float(np.mean([h["trilhas"] for x in lst for h in x["hist"]
                                                                 if h["k"] > 0 and h["trilhas"] is not None])), 2)
                                         for b, lst in br.items()}
    P["revisoes_por_experimento"] = {b: sum(h["revisou"] for x in lst for h in x["hist"]) for b, lst in br.items()}
    P["conceitos_inventados"] = {b: [x["contas"]["conceitos_inventados"] for x in lst] for b, lst in br.items()}
    V = {}
    if "E1_ativo_vence_aleatorio" in P:
        V["E1"] = P["E1_ativo_vence_aleatorio"]["vidas"] >= 7
    if "E2_ativo_vence_passivo" in P:
        V["E2"] = P["E2_ativo_vence_passivo"]["vidas"] >= 7
    if "ATIVO" in br and "ALEATORIO" in br:
        ca = sum(P["E3_conceito_certo_no_fim"]["ATIVO"].values())
        cr = sum(P["E3_conceito_certo_no_fim"]["ALEATORIO"].values())
        V["E3"] = ca >= cr + 3          # em 30 (vida x mundo)
    if "E4_maos_separam_rivais" in P:
        V["E4"] = P["E4_maos_separam_rivais"]["vidas"] >= 8
    P["VEREDITOS"] = {k: ("PASSA" if v else "FALHA") for k, v in V.items()}
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vidas", type=int, default=10)
    ap.add_argument("--semente", type=int, default=20261011)
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
    arq = os.path.join(pasta, f"ativo_{a.tag}.json")
    with open(arq, "w", encoding="utf-8") as f:
        json.dump({"placar": P, "vidas": vidas, "segundos": round(time.time() - t0, 1), "plano": PLANO},
                  f, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(json.dumps(P, ensure_ascii=False, indent=1, default=str))
    print(f"({time.time() - t0:.0f} s) -> {arq}")


if __name__ == "__main__":
    main()
