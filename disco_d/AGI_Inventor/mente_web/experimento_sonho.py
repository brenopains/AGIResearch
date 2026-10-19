"""
mente_web.experimento_sonho — o VIDEO MENTAL do laboratorio ensina a visao?
============================================================================
Pre-registro: PRE_REGISTRO_video_mental.md (escrito ANTES da rodada que vale).

Pergunta: os filmes que ela IMAGINA com as leis que descobriu (laboratorio.imagina_video:
o fundo e a aparencia real das coisas de uma cena vista, comecos que ela nunca viu,
movimento pela lei dela) ensinam o preditor de tempo do cortex dela a antecipar o
proximo quadro de videos REAIS novos — melhor do que imaginar tudo andando reto, e
melhor do que rever o que ja viu?

Uma VIDA (camera desconhecida): dois mundos, giro e depois mola_h. Em cada mundo:
  1. TREINO REAL (o tronco, comum a todos os bracos): 6 videos da arena; o laboratorio
     os observa (teorias, conceitos) e guarda a cena; o cortex os assiste como na mente
     viva (aprende a ver cada quadro; preditor de tempo a cada 5 quadros, janela de 12).
  2. SONHO (cada braco parte de uma COPIA do tronco; a parte que ve fica CONGELADA, so o
     preditor de tempo treina; 60 passos, janelas de 12 quadros):
        REVER          janelas dos 6 videos reais de novo
        SONHO_LEI      10 filmes imaginados pela teoria dela (a do ultimo video do mundo)
        SONHO_INERCIA  os MESMOS 10 comecos e cenas, tudo andando reto (placebo)
        SONHO_VERDADE  os mesmos comecos e cenas, movidos pela lei VERDADEIRA (teto; ela
                       nunca teria isto — so mede quanto do teto a lei dela alcanca)
  3. TESTE: 4 videos reais NOVOS do mundo; surpresa = erro medio da previsao do mapa do
     proximo quadro (o mesmo numero que a mente viva chama de surpresa do video).
  O tronco segue para o proximo mundo sem os sonhos (os bracos nunca se acumulam).
  4. FILME IMAGINADO x FILME REAL: 10 comecos novos escolhidos por ela; ela imagina o filme
     ANTES; depois a arena roda o filme de verdade daquele comeco. Compara-se o que SE MOVE
     (cada filme menos o proprio fundo, borrado com sigma 3 px) aos 1, 2 e 3 s, contra o
     filme imaginado "tudo reto" e (na mola_h) o filme imaginado com a lei do OUTRO mundo
     dela (a teoria do giro).

Uso:
  python experimento_sonho.py --vidas 10 --semente 20261210 --proc 4 --tag registrada
  python experimento_sonho.py --vidas 3 --semente 101 --proc 3 --tag dev
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arena as AR  # noqa: E402
import cortex as CX  # noqa: E402
import laboratorio as LB  # noqa: E402

MUNDOS = ("giro", "mola_h")
N_REAL, N_TESTE, N_SONHOS, PASSOS_SONHO, JANELA_T = 6, 4, 10, 60, 12
DUR_SONHO, FPS = 6.0, 5.0
BRACOS = ("REVER", "SONHO_LEI", "SONHO_INERCIA", "SONHO_VERDADE")


def q64(q):
    return CX.reduz(np.asarray(q, np.uint8))


def assiste(cx, quadros):
    """como _assistir na mente viva: aprende a ver cada quadro; preditor de tempo a cada 5."""
    Z = []
    for q in quadros:
        x = q64(q)[None]
        z, _e, _r = cx.ve(x)
        cx.aprende(x)
        Z.append(z[0])
        if len(Z) >= 6 and len(Z) % 5 == 0:
            cx.aprende_tempo(np.stack(Z[-12:]))


def mapas(cx, quadros):
    return cx.ve(np.stack([q64(q) for q in quadros]))[0]


def surpresa(cx, Z):
    """erro medio da previsao do mapa do proximo quadro (sem aprender)."""
    ent = np.concatenate([Z[:-2], Z[1:-1]], -1).astype(np.float32)
    pr, _c = cx._tempo(ent)
    return float(((pr - Z[2:]) ** 2).mean())


def caminho_verdadeiro(cam, mundo, Z0, ts):
    """os mesmos comecos (coordenadas percebidas) movidos pela lei VERDADEIRA, de volta a
    coordenadas percebidas (so o avaliador sabe fazer isto)."""
    A, tc = cam
    S0 = np.stack([AR.de_percebido(cam, u) for u in Z0])
    tr = AR.integra(mundo, S0, ts)                                   # (n, T, 4)
    px = tr[..., :2] @ A.T + tc
    return (px - AR.LADO / 2) / (AR.LADO / 2)


def _borra(x, s=3.0):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(x, (s, s, 0))


def _movel(qs):
    """o que se move num filme: cada quadro menos o fundo do proprio filme (mediana), borrado."""
    Q = np.asarray(qs, np.float32) / 255.0
    f = np.median(Q, 0)
    return [_borra(q - f) for q in Q]


def compara_filmes(imaginado, real, ks=(5, 10, 15)):
    a, b = _movel(imaginado), _movel(real)
    return [float(((a[k] - b[k]) ** 2).mean()) for k in ks]


def uma_vida(sem):
    t0 = time.time()
    cam = AR.camera(np.random.default_rng(sem))
    lab = LB.Laboratorio(sem)
    tronco = CX.Cortex(sem)
    rng = np.random.default_rng([sem, 31])
    ts = np.arange(int(DUR_SONHO * FPS) + 1) / FPS
    out = {"semente": sem, "mundos": {}}
    teoria_do_mundo = {}
    for i, m in enumerate(MUNDOS):
        reais, eids = [], []
        for k in range(N_REAL):
            ep = AR.episodio(m, np.random.default_rng([sem, 100 * i + k]), cam)
            per = LB.percebe(ep.quadros, ep.tempos)
            r = lab.observa(None, None, fonte="arena", url=f"arena://{100 * i + k}", percepcao=per)
            if r.get("episodio") is not None:
                eids.append(r["episodio"])
                lab.guarda_cena(r["episodio"], LB.cena(ep.quadros, per))
            assiste(tronco, ep.quadros)
            reais.append(ep.quadros)
        Zreal = [mapas(tronco, q) for q in reais]
        testes = [AR.episodio(m, np.random.default_rng([sem, 5000 + 100 * i + r_]), cam) for r_ in range(N_TESTE)]
        Zteste = [mapas(tronco, ep.quadros) for ep in testes]
        tid = lab.eps[eids[-1]].get("teoria") if eids else None
        T = lab.teorias.get(tid) or lab.teoria_atual("arena")
        cenas = [e for e in eids if e in getattr(lab, "cenas", {})]
        sonhos = {"SONHO_LEI": [], "SONHO_INERCIA": [], "SONHO_VERDADE": []}
        erro_lei_px = []
        if T is not None and cenas:
            for d in range(N_SONHOS):
                c = lab.cenas[cenas[int(rng.integers(len(cenas)))]]
                Z0 = lab.estados_imaginaveis(T, 3, rng)
                if Z0 is None:
                    continue
                q_lei, P_lei = lab.imagina_video(c, Z0, T["lei"], ts)
                q_in, _P = lab.imagina_video(c, Z0, None, ts)
                Pv = caminho_verdadeiro(cam, m, Z0, ts)
                q_v = np.stack([LB.desenha_cena(c, Pv[:, k]) for k in range(len(ts))])
                erro_lei_px.append(float(np.median(np.linalg.norm(P_lei[:, 10] - Pv[:, 10], axis=1)) * LB.PX))
                sonhos["SONHO_LEI"].append(mapas(tronco, q_lei))
                sonhos["SONHO_INERCIA"].append(mapas(tronco, q_in))
                sonhos["SONHO_VERDADE"].append(mapas(tronco, q_v))
        res = {"sem_sonho": float(np.mean([surpresa(tronco, Z) for Z in Zteste])),
               "teoria": None if T is None else T["id"], "conceito": None if T is None else T["lei"].get("conceito"),
               "lei": None if T is None else LB.formula_lei(T["lei"]), "sonhos": len(sonhos["SONHO_LEI"]),
               "erro_da_lei_em_2s_px": round(float(np.median(erro_lei_px)), 3) if erro_lei_px else None}
        for b in BRACOS:
            cx = copy.deepcopy(tronco)
            fonte = Zreal if b == "REVER" else sonhos[b]
            if not fonte:
                res[b] = None
                continue
            rb = np.random.default_rng([sem, 77, i])                 # as MESMAS escolhas de janela em todos
            for _p in range(PASSOS_SONHO):
                Z = fonte[int(rb.integers(len(fonte)))]
                a = int(rb.integers(0, max(1, len(Z) - JANELA_T)))
                cx.aprende_tempo(Z[a:a + JANELA_T])
            res[b] = float(np.mean([surpresa(cx, Z) for Z in Zteste]))
        # 4. filme imaginado ANTES x filme real
        filmes = {"lei": [], "inercia": [], "outro_mundo": []}
        outro = teoria_do_mundo.get("giro") if m == "mola_h" else None
        if T is not None and cenas:
            rf = np.random.default_rng([sem, 88, i])
            for d in range(N_SONHOS):
                Z0 = lab.estados_imaginaveis(T, 3, rf)
                if Z0 is None:
                    continue
                c = lab.cenas[cenas[int(rf.integers(len(cenas)))]]
                ep = AR.episodio(m, np.random.default_rng([sem, 7000 + 100 * i + d]), cam, dur=DUR_SONHO,
                                 armadilhas=False, estados0=[AR.de_percebido(cam, u) for u in Z0])
                tt = np.asarray(ep.tempos, float)
                filmes["lei"].append(compara_filmes(lab.imagina_video(c, Z0, T["lei"], tt)[0], ep.quadros))
                filmes["inercia"].append(compara_filmes(lab.imagina_video(c, Z0, None, tt)[0], ep.quadros))
                if outro is not None:
                    filmes["outro_mundo"].append(compara_filmes(lab.imagina_video(c, Z0, outro["lei"], tt)[0], ep.quadros))
        res["filme_erro_1_2_3s"] = {k: (np.median(np.array(v), 0).round(7).tolist() if v else None) for k, v in filmes.items()}
        if T is not None:
            teoria_do_mundo[m] = copy.deepcopy(T)
        out["mundos"][m] = res
    out["contas"] = lab.contas
    out["segundos"] = round(time.time() - t0, 1)
    return out


def _gm(xs):
    xs = [x for x in xs if x is not None]
    return float(np.exp(np.mean(np.log(np.maximum(xs, 1e-12))))) if len(xs) == len(MUNDOS) else None


def placar(vidas):
    P = {"n_vidas": len(vidas)}
    G = {b: [_gm([v["mundos"][m][b] for m in MUNDOS]) for v in vidas] for b in BRACOS + ("sem_sonho",)}
    P["surpresa_por_vida"] = {b: [None if x is None else round(x, 6) for x in xs] for b, xs in G.items()}
    P["surpresa_mediana"] = {b: round(float(np.median([x for x in xs if x is not None])), 6) for b, xs in G.items()
                             if any(x is not None for x in xs)}
    P["por_mundo_mediana"] = {m: {b: round(float(np.median([v["mundos"][m][b] for v in vidas
                                                             if v["mundos"][m][b] is not None])), 6)
                                  for b in BRACOS + ("sem_sonho",)} for m in MUNDOS}
    P["erro_da_lei_em_2s_px"] = {m: [v["mundos"][m]["erro_da_lei_em_2s_px"] for v in vidas] for m in MUNDOS}

    def vence(a, b):
        return sum(x < y for x, y in zip(G[a], G[b]) if x is not None and y is not None)
    P["S1_lei_vence_inercia"] = {"vidas": vence("SONHO_LEI", "SONHO_INERCIA"),
                                 "razao": round(P["surpresa_mediana"]["SONHO_LEI"] / P["surpresa_mediana"]["SONHO_INERCIA"], 4)}
    P["S2_lei_vence_rever"] = {"vidas": vence("SONHO_LEI", "REVER"),
                               "razao": round(P["surpresa_mediana"]["SONHO_LEI"] / P["surpresa_mediana"]["REVER"], 4)}
    P["V0_verdade_vence_inercia"] = vence("SONHO_VERDADE", "SONHO_INERCIA")
    P["inercia_vence_rever"] = vence("SONHO_INERCIA", "REVER")

    # filme imaginado x real: por vida, media geometrica entre os mundos do erro aos 2 s
    def filme(v, chave, k=1, mundos=MUNDOS):
        xs = [v["mundos"][m]["filme_erro_1_2_3s"][chave] for m in mundos]
        if any(x is None for x in xs):
            return None
        return float(np.exp(np.mean(np.log([max(x[k], 1e-12) for x in xs]))))
    F = {c: [filme(v, c) for v in vidas] for c in ("lei", "inercia")}
    F["lei_mola_h"] = [filme(v, "lei", mundos=("mola_h",)) for v in vidas]
    F["outro_mundo_mola_h"] = [filme(v, "outro_mundo", mundos=("mola_h",)) for v in vidas]
    P["filme_2s_por_vida"] = {c: [None if x is None else round(x, 7) for x in xs] for c, xs in F.items()}
    P["filme_por_horizonte_mediana"] = {m: {c: (np.median([v["mundos"][m]["filme_erro_1_2_3s"][c] for v in vidas
                                                           if v["mundos"][m]["filme_erro_1_2_3s"][c] is not None], 0)
                                                .round(7).tolist() if any(v["mundos"][m]["filme_erro_1_2_3s"][c] is not None
                                                                          for v in vidas) else None)
                                            for c in ("lei", "inercia", "outro_mundo")} for m in MUNDOS}

    def vf(a, b):
        return sum(x < y for x, y in zip(F[a], F[b]) if x is not None and y is not None)
    P["M1_filme_lei_vence_reto"] = vf("lei", "inercia")
    P["M2_filme_lei_vence_lei_do_outro_mundo"] = vf("lei_mola_h", "outro_mundo_mola_h")
    V0 = P["V0_verdade_vence_inercia"] >= 7
    P["VEREDITOS"] = {"V0": "PASSA" if V0 else "FALHA",
                      "S1": ("PASSA" if P["S1_lei_vence_inercia"]["vidas"] >= 7 else "FALHA") if V0 else "SEM_VALOR (V0 falhou)",
                      "S2": ("PASSA" if P["S2_lei_vence_rever"]["vidas"] >= 7 else "FALHA") if V0 else "SEM_VALOR (V0 falhou)",
                      "M1": "PASSA" if P["M1_filme_lei_vence_reto"] >= 8 else "FALHA",
                      "M2": "PASSA" if P["M2_filme_lei_vence_lei_do_outro_mundo"] >= 8 else "FALHA"}
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vidas", type=int, default=10)
    ap.add_argument("--semente", type=int, default=20261210)
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
    arq = os.path.join(pasta, f"sonho_{a.tag}.json")
    with open(arq, "w", encoding="utf-8") as f:
        json.dump({"placar": P, "vidas": vidas, "segundos": round(time.time() - t0, 1)}, f, ensure_ascii=False,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(json.dumps(P, ensure_ascii=False, indent=1, default=str))
    print(f"({time.time() - t0:.0f} s) -> {arq}")


if __name__ == "__main__":
    main()
