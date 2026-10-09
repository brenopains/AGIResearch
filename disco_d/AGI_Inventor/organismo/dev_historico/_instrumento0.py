import numpy as np
import mundo_fisico as M, material_fisica as F
rng = np.random.default_rng(5)
h = F.Conceito([("x","vy"),("y","vx")], np.array([1.,-1.]), 1.0, {}, "h")
def E_or(mu): return F.Conceito([("vx","vx"),("vy","vy"),("1/r",)], np.array([.5,.5,-mu]), 1.0, {}, "E")
NROT=("rot","v^")
fams = ["kepler","einstein","oblato","magnetico","mola","arrasto","ciclotron","vortice"]
print("N=24 | base | base+rot(v) | oraculo(h,E,rot) | base 3 encontros conjuntos")
for fam in fams:
    r = {"base":0,"rot":0,"or":0,"pool":0}
    for rep in range(20):
        Ps = [M.Problema(0, fam, "percebido", "L", rng) for _ in range(3)]
        P = Ps[0]
        l,_,_ = F.ajusta_lei([(P.S,P.A)], {}, []); r["base"] += P.verifica(l[0])[0]
        l,_,_ = F.ajusta_lei([(P.S,P.A)], {}, [NROT]); r["rot"] += P.verifica(l[0])[0]
        l,_,_ = F.ajusta_lei([(P.S,P.A)], {"h":h,"E":E_or(P.p["mu"])}, [NROT]); r["or"] += P.verifica(l[0])[0]
        l,_,_ = F.ajusta_lei([(Q.S,Q.A) for Q in Ps], {}, []); r["pool"] += P.verifica(l[0])[0]
    print(f"{fam:10s} {r['base']:2d}/20 | {r['rot']:2d}/20 | {r['or']:2d}/20 | {r['pool']:2d}/20")
