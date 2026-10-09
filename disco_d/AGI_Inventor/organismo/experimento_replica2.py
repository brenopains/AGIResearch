"""Vida 2: 6 bracos x sementes NOVAS 2001-2010 (retomavel). Uso: python experimento2.py"""
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

BRACOS2 = ("ESCOPO", "LARGURA_3", "PIPELINE")
SEMENTES2 = list(range(3001, 3021))


def tarefa(args):
    braco, sem, lacrados, pasta = args
    import vida
    t0 = time.time()
    vida.vive(braco, sem, tuple(lacrados), pasta)
    return braco, sem, round(time.time() - t0, 1)


if __name__ == "__main__":
    from multiprocessing import Pool
    pasta = sys.argv[1] if len(sys.argv) > 1 else "resultados/replica2"
    lacre = json.load(open("LACRE.json"))
    trabalhos = [(b, s, lacre["lacrados"], pasta) for s in SEMENTES2 for b in BRACOS2]
    t0 = time.time()
    with Pool(2) as pool:
        for i, (b, s, seg) in enumerate(pool.imap_unordered(tarefa, trabalhos)):
            print(f"[{i + 1}/{len(trabalhos)}] {b} s{s} {seg}s  (total {time.time() - t0:.0f}s)", flush=True)
