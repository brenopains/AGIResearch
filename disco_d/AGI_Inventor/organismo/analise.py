"""
organismo.analise — os portoes, lidos dos JSON de cada vida (e do log de ontogenia)
=====================================================================================
Pareado por semente: o fluxo de problemas e o mundo sao identicos entre bracos
na mesma semente. Um portao passa se a diferenca media pareada atinge o limiar
E o intervalo bootstrap de 90% da diferenca fica inteiro do lado certo do zero.
Nada aqui decide o veredito por frase: imprime a tabela; a leitura e humana.
"""
import glob
import json
import os
import sys

import numpy as np

INV_L = {"spin_orbita", "magnetico", "ciclotron"}
INV_Q = {"sentido", "escapa", "periapse_direita"}


def carrega(pasta):
    vidas = {}
    for arq in glob.glob(os.path.join(pasta, "*.json")):
        r = json.load(open(arq, encoding="utf-8"))
        vidas[(r["braco"], r["semente"])] = r
    return vidas


def _media(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float("nan")


def metricas(r):
    L = [l for l in r["linhas"] if l["fonte"] == "percebido"]
    ok = lambda ls: _media([float(l["ok"]) for l in ls])
    t3 = [l for l in L if l["idx"] >= 80]
    acertos_t3 = sum(l["ok"] for l in t3)
    inv = [l for l in L if (l["tipo"] == "L" and l["fam"] in INV_L)
           or (l["tipo"] == "Q" and l["molde"] in INV_Q)]
    lac = [l for l in L if l.get("lacrado")]
    return dict(
        total=ok(L), rec=ok([l for l in L if l["recorrente"]]),
        h2=ok([l for l in L if l["idx"] >= 60]), t3=ok(t3),
        custo_t3=sum(l["gasto"] for l in t3) / max(acertos_t3, 1),
        custo_total=sum(l["gasto"] for l in L) / max(sum(l["ok"] for l in L), 1),
        inv=ok(inv), mag=ok([l for l in L if l["fam"] == "magnetico" and l["tipo"] == "L"]),
        cic=ok([l for l in L if l["fam"] == "ciclotron" and l["tipo"] == "L"]),
        lac=ok(lac), lac_abl=_media([float(l["ok_ablacao"]) for l in lac
                                      if l["ok_ablacao"] is not None]),
        n_lac=len(lac))


def profundidade2(pasta, braco, sem):
    """Algum problema RESOLVIDO usou conceito de profundidade >= 2?"""
    arq = os.path.join(pasta, f"{braco}_s{sem}.jsonl")
    if not os.path.exists(arq):
        return None
    prof, usados_ok = {}, []
    for linha in open(arq, encoding="utf-8"):
        d = json.loads(linha)
        if d["ev"] == "nascimento" and d.get("tipo") == "conceito":
            desc = d.get("desc") or {}
            base = {"x", "y", "vx", "vy", "r", "1/r"}
            # profundidade 2 NAO trivial: algum monomio mistura atomo de base com
            # conceito (h e h^2 sao conservados so porque h e; Runge-Lenz nao)
            mistura = any(any(a in base for a in m) and any(a not in base for a in m)
                          for m in desc.get("suporte", []))
            prof[d["id"]] = desc.get("prof", 1) if mistura else 1
    r = json.load(open(os.path.join(pasta, f"{braco}_s{sem}.json"), encoding="utf-8"))
    for l in r["linhas"]:
        if l["ok"] and l["fonte"] == "percebido":
            usados_ok += l.get("usados", [])
    return any(prof.get(u, 0) >= 2 for u in usados_ok)


def boot(d, rng, n=4000):
    d = np.asarray(d, float)
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return float(np.quantile(m, 0.05)), float(np.quantile(m, 0.95))


def pareado(vidas, a, b, chave, sementes, sinal=+1):
    d = []
    for s in sementes:
        if (a, s) in vidas and (b, s) in vidas:
            ma, mb = metricas(vidas[(a, s)])[chave], metricas(vidas[(b, s)])[chave]
            if np.isfinite(ma) and np.isfinite(mb):
                d.append(sinal * (ma - mb))
    return d


def main(pasta):
    vidas = carrega(pasta)
    sementes = sorted({s for (_b, s) in vidas})
    bracos = sorted({b for (b, _s) in vidas})
    rng = np.random.default_rng(0)
    print(f"{len(vidas)} vidas, sementes {sementes}")
    print("\nMEDIAS POR BRACO (verificaveis)")
    cols = ["total", "rec", "h2", "t3", "inv", "mag", "cic", "lac", "lac_abl", "custo_t3", "custo_total"]
    print(f"{'braco':14s}" + "".join(f"{c:>11s}" for c in cols))
    for b in bracos:
        ms = [metricas(vidas[(b, s)]) for s in sementes if (b, s) in vidas]
        print(f"{b:14s}" + "".join(f"{_media([m[c] for m in ms]):11.3f}" for c in cols))

    def linha(nome, a, b, chave, limiar, sinal=+1, escala=100):
        d = pareado(vidas, a, b, chave, sementes, sinal)
        if not d:
            print(f"{nome:5s} sem dados")
            return
        lo, hi = boot(d, rng)
        m = float(np.mean(d))
        passa = m * escala >= limiar and lo > 0
        print(f"{nome:5s} {a} - {b} [{chave}]: media {m*escala:+.1f} (IC90 {lo*escala:+.1f} a "
              f"{hi*escala:+.1f}), n={len(d)}, a favor {sum(x > 0 for x in d)}/{len(d)}  "
              f"limiar {limiar:+} -> {'PASSA' if passa else 'falha'}")

    print("\nPORTOES (pre-registrados em PRE_REGISTRO.md)")
    linha("G1", "ORGANISMO", "PIPELINE", "total", 10)
    linha("G2a", "ORGANISMO", "SEM_MESO", "rec", 5)
    linha("G2b", "ORGANISMO", "PLACEBO_MESO", "rec", 5)
    linha("G3", "ORGANISMO", "TETOS", "h2", 5)
    d4 = pareado(vidas, "ORGANISMO", "SEM_COAL", "t3", sementes)
    c_o = _media([metricas(vidas[("ORGANISMO", s)])["custo_t3"] for s in sementes if ("ORGANISMO", s) in vidas])
    c_c = _media([metricas(vidas[("SEM_COAL", s)])["custo_t3"] for s in sementes if ("SEM_COAL", s) in vidas])
    if d4:
        print(f"G4    custo por acerto (t3) ORGANISMO {c_o:.1f} / SEM_COAL {c_c:.1f} = {c_o / c_c:.2f} "
              f"(limiar <= 0,80); acerto t3 {np.mean(d4)*100:+.1f} (limiar >= -2) -> "
              f"{'PASSA' if c_o / c_c <= 0.80 and np.mean(d4) * 100 >= -2 else 'falha'}")
    linha("G5", "ORGANISMO", "SEM_REGIME", "inv", 5)
    linha("G6", "ORGANISMO", "SEM_PROC2", "mag", 5)
    d7 = []
    for s in sementes:
        if ("ORGANISMO", s) in vidas:
            m = metricas(vidas[("ORGANISMO", s)])
            if m["n_lac"]:
                d7.append(m["lac"] - m["lac_abl"])
    if d7:
        lo, hi = boot(d7, rng)
        print(f"G7    lacradas: ORGANISMO com conceitos - mesma mente sem conceitos: "
              f"{np.mean(d7)*100:+.1f} (IC90 {lo*100:+.1f} a {hi*100:+.1f}), limiar +15 -> "
              f"{'PASSA' if np.mean(d7) * 100 >= 15 and lo > 0 else 'falha'}")
    p2 = [profundidade2(pasta, "ORGANISMO", s) for s in sementes]
    p2 = [p for p in p2 if p is not None]
    print(f"G8    vidas com conceito de profundidade >= 2 usado em acerto: {sum(p2)}/{len(p2)} "
          f"(limiar >= 5/10) -> {'PASSA' if sum(p2) >= 5 else 'falha'}")
    d9 = pareado(vidas, "ORGANISMO", "SEM_META", "total", sementes)
    if d9:
        c_o = _media([metricas(vidas[("ORGANISMO", s)])["custo_total"] for s in sementes if ("ORGANISMO", s) in vidas])
        c_m = _media([metricas(vidas[("SEM_META", s)])["custo_total"] for s in sementes if ("SEM_META", s) in vidas])
        lo, _hi = boot(d9, rng)
        ok9 = (np.mean(d9) * 100 >= 5 and lo > 0) or (np.mean(d9) * 100 >= -2 and c_o / c_m <= 0.85)
        print(f"G9    ORGANISMO - SEM_META: acerto {np.mean(d9)*100:+.1f}, custo/acerto {c_o:.1f}/{c_m:.1f} "
              f"= {c_o / c_m:.2f} -> {'PASSA' if ok9 else 'falha'}")
    linha("G10", "ORGANISMO", "SEM_SONHO", "rec", 3)
    linha("G11", "ORGANISMO", "SEM_ECO", "inv", 5)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "resultados/final")
