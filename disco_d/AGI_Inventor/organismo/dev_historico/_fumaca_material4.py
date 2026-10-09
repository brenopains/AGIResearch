import numpy as np
import mundo_fisico as M, material_fisica as F
rng = np.random.default_rng(3)
h = F.Conceito([("x","vy"),("y","vx")], np.array([1.,-1.]), 1.0, {}, "h")
NROT=("rot","v^")
casos = {"einstein": ({"h":h}, [], [(("r",),(-2.,),"r^"), (("r","h"),(-4.,2.),"r^")]),
         "oblato": ({}, [], [(("r",),(-2.,),"r^"), (("r",),(-4.,),"r^")]),
         "vortice": ({}, [NROT], [(("r",),(-2.,),"r^"), (("r","v"),(-1.,1.),"rot(v^)")])}
for nlab in (24, 96):
  for fam,(cc,ext,forma) in casos.items():
    busca, dado = 0, 0
    for rep in range(20):
        P = M.Problema(0, fam, "percebido", "L", rng, n_lab=nlab)
        lf,bf,_ = F.ajusta_lei([(P.S,P.A)], cc, ext, forma=forma)
        ll,bl,_ = F.ajusta_lei([(P.S,P.A)], cc, ext)
        if not P.verifica(ll[0])[0]:
            if bf < bl: busca += 1
            else: dado += 1
    print(f"N={nlab} {fam:9s} falhas: verdadeira MAIS CURTA (defeito de busca) {busca:2d} | achada mais curta (falta dado) {dado:2d}")
