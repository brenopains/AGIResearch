"""
executar — os quatro bracos num jogo, com o mesmo orcamento e as mesmas sementes
==================================================================================

  python executar.py fumaca breakout 20000     orcamento pequeno, so para medir
  python executar.py dev                        Breakout e Pong, 100k passos
  python executar.py lacrados                   SO depois do codigo congelado
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

import agente as G
import ambiente as A

BRACOS = ("U", "C", "I", "P")


def um_braco(jogo, braco, semente, orcamento, episodios=5, teto=10_000):
    res = G.coleta(jogo, braco, semente, orcamento, log=lambda s: print(s, flush=True))
    t = time.time()
    ant = G.antecipacao(jogo, res, semente)
    pts = G.avalia_pontos(jogo, res, semente, episodios, teto)
    pts_ap = G.avalia_pontos(jogo, res, semente, 1, teto, aparencia=(2, 0, 1))
    lab = res["lab"]
    return {"jogo": jogo, "braco": braco, "semente": semente, "orcamento": orcamento,
            "antecipacao_mse": ant, "pontos": pts, "media": float(np.mean(pts)),
            "hns": float(A.hns(jogo, np.mean(pts))), "pontos_aparencia_trocada": pts_ap,
            "eventos": lab.eventos, "n_variaveis": len(lab.vars), "n_experiencias": len(lab.progs),
            "variaveis": lab.descreve(), "n_sondas": res["n_sondas"],
            "retornos_coleta_ultimos": res["retornos_coleta"][-10:],
            "t_coleta_s": round(res["t_coleta_s"], 1), "t_aval_s": round(time.time() - t, 1)}


def rodar(jogos, orcamento, rotulo, semente=1):
    os.makedirs("resultados", exist_ok=True)
    arq = "resultados/%s.json" % rotulo
    out = json.load(open(arq)) if os.path.exists(arq) else []
    feitos = {(r["jogo"], r["braco"]) for r in out}
    for jogo in jogos:
        for b in BRACOS:
            if (jogo, b) in feitos:
                continue
            r = um_braco(jogo, b, semente, orcamento)
            out.append(r)
            json.dump(out, open(arq, "w"), indent=1, default=float)
            print("  %-12s %s | antecipacao %.5f | pontos %s | HNS %.3f | variaveis %d | "
                  "experiencias %d | coleta %.0f s, avaliacao %.0f s"
                  % (jogo, b, r["antecipacao_mse"], np.round(r["pontos"], 1).tolist(), r["hns"],
                     r["n_variaveis"], r["n_experiencias"], r["t_coleta_s"], r["t_aval_s"]),
                  flush=True)
    return out


def hash_codigo():
    h = hashlib.sha256()
    for f in ("ambiente.py", "instrumentos.py", "agente.py", "executar.py"):
        h.update(open(f, "rb").read())
    return h.hexdigest().upper()


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if modo == "fumaca":
        jogo = sys.argv[2] if len(sys.argv) > 2 else "breakout"
        orc = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
        rodar([jogo], orc, "fumaca_%s_%d" % (jogo, orc))
    elif modo == "dev":
        print("hash do codigo: %s" % hash_codigo(), flush=True)
        rodar(["breakout", "pong"], 100_000, "dev")
    elif modo == "lacrados":
        lac = json.load(open("LACRE_jogos.json"))["lacrados"]
        print("hash do codigo (CONGELADO): %s" % hash_codigo(), flush=True)
        rodar(lac, 100_000, "lacrados")
