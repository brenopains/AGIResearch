import time, shutil, sys, traceback
import organismo as O, vida
shutil.rmtree("resultados/dev", ignore_errors=True)
DEVL = ("periodo_longo", "rmax_grande", "vmax_alta")   # ja expostos; so para exercitar a ablacao
for b in O.BRACOS:
    t0 = time.time()
    try:
        r = vida.vive(b, 1, DEVL, "resultados/dev")
        print(f"{b:13s} roda ok  {time.time()-t0:5.1f}s  estruturas {r['n_estruturas']}  tipos {r['tipos']}", flush=True)
    except Exception:
        print(b, "ERRO"); traceback.print_exc(); sys.exit(1)
