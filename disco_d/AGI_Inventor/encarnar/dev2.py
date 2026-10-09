import sys, time
import numpy as np
from mundo import Mundo, SIG_P, DT
from encarnar import descreve, pontos_de_vista
sig2 = (SIG_P * np.sqrt(6) / DT ** 2) ** 2
rng = np.random.default_rng(int(sys.argv[1]))
tipos = sys.argv[2].split(",")
m = Mundo(rng, len(tipos), tipos)
X, Xo = m.simula(24.0)
print("max |p|", np.abs(X).max().round(1))
for i, e in enumerate(m.ent):
    d = descreve(Xo, i, pontos_de_vista(len(tipos), i), sig2)
    o = sorted(d.items(), key=lambda kv: kv[1][0])
    print(f"ser {i} [{e['tipo']} alvo={e.get('alvo')}] -> {o[0][0]} ({o[0][1][0]:.0f}) 2o {o[1][0]} ({o[1][1][0]:.0f}) | mundo {d['mundo'][0]:.0f} | plac {min(d['placebo a'][0], d['placebo b'][0]):.0f}")
    print("      lei:", o[0][1][1])
