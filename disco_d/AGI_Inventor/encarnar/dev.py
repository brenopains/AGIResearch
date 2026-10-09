import time, sys
import numpy as np
from mundo import Mundo, SIG_P, DT
from encarnar import descreve, pontos_de_vista
sig2 = (SIG_P * np.sqrt(6) / DT ** 2) ** 2
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
tipos = ["mola", "ima", "persegue", "foge", "formacao"]
m = Mundo(rng, 5, tipos)
X, Xo = m.simula(24.0)
print("ok sim", X.shape, "max |p|", np.abs(X).max().round(1))
for i, e in enumerate(m.ent):
    t0 = time.time()
    d = descreve(Xo, i, pontos_de_vista(5, i), sig2)
    orden = sorted(d.items(), key=lambda kv: kv[1][0])
    print(f"ser {i} [{e['tipo']} alvo={e.get('alvo')}] -> {orden[0][0]} ({orden[0][1][0]:.0f} bits) | mundo {d['mundo'][0]:.0f} | eu {d['eu'][0]:.0f} | placebo {min(d['placebo a'][0], d['placebo b'][0]):.0f}  ({time.time()-t0:.1f}s)")
    print("      lei:", orden[0][1][1])
