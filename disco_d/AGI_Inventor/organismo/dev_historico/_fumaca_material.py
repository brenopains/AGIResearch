import time, numpy as np
import mundo_fisico as M, material_fisica as F
rng = np.random.default_rng(1)
# 1) lei de base em kepler, einstein, magnetico, mola, arrasto, ciclotron
for fam in ["kepler","einstein","oblato","magnetico","mola","arrasto","ciclotron","vortice"]:
    P = M.Problema(0, fam, "percebido", "L", rng)
    t0=time.time()
    leis, bits, c = F.ajusta_lei([(P.S,P.A)], {}, [])
    ok, rms = P.verifica(leis[0])
    print(f"{fam:10s} base: {leis[0]!r:60.60s} bits {bits:7.1f} custo {c:6.2f} ok {ok} rms/sig {rms:6.2f} {time.time()-t0:.2f}s")
