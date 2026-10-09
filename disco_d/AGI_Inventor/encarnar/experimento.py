"""
encarnar.experimento — mundos lacrados T1 (tipos de desenvolvimento) e T2 (reserva)
python experimento.py   -> resultados/teste.json e o veredito A1..A5
"""
import collections
import json
import math
import os
import re
import time

import numpy as np

from encarnar import _base, lei_mdl, ponto_de_vista, pontos_de_vista
from mundo import DT, SIG_P, TIPOS_DEV, Mundo

SIG2 = (SIG_P * np.sqrt(6) / DT ** 2) ** 2
COM_ALVO = ("persegue", "foge", "satelite", "formacao", "intercepta", "distancia")


def ajusta_preve(F, nm, y, quad, tr, te):
    """Lei por MDL nos passos de treino; previsao da aceleracao NO MUNDO nos de teste."""
    t, nh, a_ref, a_lab = quad
    bases = [_base(F, nm, 1), _base(F, nm, 2)]
    tot, termos, yhat = 0.0, [], np.zeros((len(te), 2))
    for comp in (0, 1):
        melhores = []
        for B, nmb in bases:
            b, tm, sel, coef, (mu, sd, ym) = lei_mdl(B[tr], nmb, y[tr, comp], SIG2)
            if sel:
                Zte = (B[te][:, sel] - mu[sel]) / sd[sel]
                pr = Zte @ coef + ym
            else:
                pr = np.full(len(te), ym)
            melhores.append((b, tm, pr))
        b, tm, pr = min(melhores, key=lambda z: z[0])
        tot += b + 1.0
        termos.append(tm)
        yhat[:, comp] = pr
    a_hat = a_ref[te] + yhat[:, :1] * t[te] + yhat[:, 1:] * nh[te]
    r = a_lab[te] - a_hat
    # erro em unidades do ruido CONHECIDO do instrumento (1 = perfeito); R2 engana
    # quando o ser ja parou e so sobra ruido no futuro
    return tot, termos, float(np.sqrt((r ** 2).mean()) / math.sqrt(SIG2))


def citados(termos):
    c = collections.Counter(int(x) for tm in termos for nome in tm for x in re.findall(r"s(\d+)", nome))
    return c


def um_mundo(sem, tipos=None):
    rng = np.random.default_rng(sem)
    m = Mundo(rng, 5, tipos)
    X, Xo = m.simula(24.0)
    T = Xo.shape[0] - 2
    tr, te = np.arange(int(0.75 * T)), np.arange(int(0.75 * T), T)
    out = []
    for i, e in enumerate(m.ent):
        pvs = pontos_de_vista(5, i)
        res = {}
        for pv in pvs:
            F, nm, y, quad = ponto_de_vista(Xo, i, pv, com_quadro=True)
            if not np.isfinite(F).all() or not np.isfinite(y).all():
                continue
            b, tm, r2 = ajusta_preve(F, nm, y, quad, tr, te)
            res[pv] = (b + math.log2(len(pvs)), tm, r2)
        esc = min(res, key=lambda k: res[k][0])
        c = citados(res[esc][1])
        if esc.startswith(("ser ", "junto ")):
            # no quadro de Y, a relacao com Y esta no proprio quadro (atomos "eu.*")
            j = int(esc.split()[1])
            c[j] += sum(1 for tm in res[esc][1] for nome in tm if re.search(r"(^|[^s\d])eu", nome))
        alvo_achado = c.most_common(1)[0][0] if c else None
        verdadeiro = {"mola": ["mundo", "eu"], "ima": ["mundo", "eu"], "persegue": ["eu"],
                      "foge": ["eu"], "intercepta": ["eu"], "distancia": ["eu"],
                      "satelite": [f"junto {e.get('alvo')}", f"ser {e.get('alvo')}", "eu", "mundo"],
                      "formacao": [f"ser {e.get('alvo')}"]}[e["tipo"]]
        b_esc = res[esc][0]
        b_verd = min((res[v][0] for v in verdadeiro if v in res), default=float("inf"))
        status = "certo" if esc in verdadeiro else ("equivalente" if b_verd - b_esc < 20 else "errado")
        out.append(dict(mundo=sem, ser=i, tipo=e["tipo"], alvo=e.get("alvo"), escolhido=esc,
                        status=status, bits=round(b_esc, 1), bits_mundo=round(res["mundo"][0], 1),
                        erro_escolhido=round(res[esc][2], 3), erro_mundo=round(res["mundo"][2], 3),
                        placebo=esc.startswith("placebo"), cita=dict(c), alvo_achado=alvo_achado,
                        lei=res[esc][1]))
    return out


if __name__ == "__main__":
    os.makedirs("resultados", exist_ok=True)
    t0 = time.time()
    linhas = []
    for s in range(1000, 1040):
        linhas += [dict(r, teste="T1") for r in um_mundo(s)]
        print("T1", s, round(time.time() - t0), flush=True)
    for s in range(2000, 2020):
        r = np.random.default_rng(s + 7)
        tipos = ["intercepta", "distancia"] + list(r.choice(TIPOS_DEV, 3))
        r.shuffle(tipos)
        linhas += [dict(x, teste="T2") for x in um_mundo(s, tipos)]
        print("T2", s, round(time.time() - t0), flush=True)
    json.dump(linhas, open("resultados/teste.json", "w"), indent=1, default=str)

    def frac(xs):
        return (sum(xs) / len(xs)) if xs else float("nan")
    T1 = [x for x in linhas if x["teste"] == "T1"]
    T2 = [x for x in linhas if x["teste"] == "T2"]
    alvo1 = [x["alvo_achado"] == x["alvo"] for x in T1 if x["tipo"] in COM_ALVO]
    A1 = frac(alvo1)
    A2 = frac([x["placebo"] for x in linhas])
    ag = [x for x in T1 if x["tipo"] in ("persegue", "foge")]
    razao = [x["erro_mundo"] / x["erro_escolhido"] for x in ag]
    A3g, A3r = float(np.median(razao)), float(np.median([x["erro_escolhido"] for x in ag]))
    fis = [not x["cita"] for x in T1 if x["tipo"] in ("mola", "ima")]
    lig = [bool(x["cita"]) for x in T1 if x["tipo"] in ("persegue", "foge", "satelite", "formacao")]
    res2 = [x for x in T2 if x["tipo"] in ("intercepta", "distancia")]
    A5a = frac([x["alvo_achado"] == x["alvo"] for x in res2])
    A5g = float(np.median([x["erro_mundo"] / x["erro_escolhido"] for x in res2]))
    print("\n=== VEREDITO ===")
    print(f"A1 alvo certo (T1, seres com alvo): {A1:.3f} (n={len(alvo1)}; exige >= 0,80)  {'PASSA' if A1 >= 0.8 else 'FALHA'}")
    print(f"A2 placebo escolhido: {A2:.3f} (exige <= 0,05)  {'PASSA' if A2 <= 0.05 else 'FALHA'}")
    print(f"A3 agentes: erro mundo/escolhido mediano {A3g:.2f}x, erro escolhido mediano {A3r:.2f} ruidos (n={len(ag)}; exige >= 1,5x e <= 2,0)  {'PASSA' if A3g >= 1.5 and A3r <= 2.0 else 'FALHA'}")
    print(f"A4 fisica sem citar outro: {frac(fis):.3f} (n={len(fis)}); ligados citando: {frac(lig):.3f} (n={len(lig)}) (exige >= 0,85 cada)  {'PASSA' if frac(fis) >= 0.85 and frac(lig) >= 0.85 else 'FALHA'}")
    print(f"A5 reserva: alvo {A5a:.3f}, erro mundo/escolhido mediano {A5g:.2f}x (n={len(res2)}; exige 0,60 e 1,5x)  {'PASSA' if A5a >= 0.6 and A5g >= 1.5 else 'FALHA'}")
    por = collections.defaultdict(list)
    for x in linhas:
        por[x["tipo"]].append(x)
    for tp, xs in por.items():
        st = collections.Counter(x["status"] for x in xs)
        print(f"  {tp:11s} n={len(xs):3d} alvo={frac([x['alvo_achado']==x['alvo'] for x in xs]) if tp in COM_ALVO else float('nan'):.2f} "
              f"erro esc {np.median([x['erro_escolhido'] for x in xs]):.2f} mundo {np.median([x['erro_mundo'] for x in xs]):.2f} pv {dict(st)}")
    print("tempo", round(time.time() - t0), "s")
