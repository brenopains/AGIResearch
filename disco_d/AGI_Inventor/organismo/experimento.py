"""
organismo.experimento — todos os bracos x sementes finais (retomavel)
=======================================================================
Dois processos de CPU, sem CUDA, BLAS com 1 thread (limites da maquina do
usuario: ~2 GB livres reais). Cada vida grava o proprio JSON; o que ja existe e
pulado. Uso:  python experimento.py [pasta]
"""
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

SEMENTES_FINAIS = list(range(1001, 1011))


def tarefa(args):
    braco, sem, lacrados, pasta = args
    import vida
    t0 = time.time()
    vida.vive(braco, sem, tuple(lacrados), pasta)
    return braco, sem, round(time.time() - t0, 1)


if __name__ == "__main__":
    from multiprocessing import Pool
    import organismo as O
    pasta = sys.argv[1] if len(sys.argv) > 1 else "resultados/final"
    lacre = json.load(open("LACRE.json"))
    trabalhos = [(b, s, lacre["lacrados"], pasta) for s in SEMENTES_FINAIS for b in O.BRACOS]
    t0 = time.time()
    with Pool(2) as pool:
        for i, (b, s, seg) in enumerate(pool.imap_unordered(tarefa, trabalhos)):
            print(f"[{i + 1}/{len(trabalhos)}] {b} s{s} {seg}s  (total {time.time() - t0:.0f}s)",
                  flush=True)
