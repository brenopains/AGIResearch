import numpy as np
import mundo_fisico as M, material_fisica as F
rng = np.random.default_rng(3)
h = F.Conceito([("x","vy"),("y","vx")], np.array([1.,-1.]), 1.0, {}, "h")
NROT=("rot","v^")
casos = {"einstein": ({"h":h}, [], [(("r",),(-2.,),"r^"), (("r","h"),(-4.,2.),"r^")]),
         "vortice": ({}, [NROT], [(("r",),(-2.,),"r^"), (("r","v"),(-1.,1.),"rot(v^)")]),
         "oblato": ({}, [], [(("r",),(-2.,),"r^"), (("r",),(-4.,),"r^")])}
for nlab in (24, 48, 96):
  for fam,(cc,ext,forma) in casos.items():
    rf, rl, okf, okl = [],[],0,0
    for rep in range(20):
        P = M.Problema(0, fam, "percebido", "L", rng, n_lab=nlab)
        lf,_,_ = F.ajusta_lei([(P.S,P.A)], cc, ext, forma=forma)
        ll,_,_ = F.ajusta_lei([(P.S,P.A)], cc, ext)
        a,b = P.verifica(lf[0]); c,d = P.verifica(ll[0])
        rf.append(b); rl.append(d); okf+=a; okl+=c
    print(f"N={nlab:3d} {fam:9s} forma verdadeira ok {okf:2d}/20 rms med {np.median(rf):.2f} | ajustada ok {okl:2d}/20 rms med {np.median(rl):.2f}  ex {ll[0]!r:60.60s}")
