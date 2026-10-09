import numpy as np, sys
sys.path.insert(0, ".")
import mundo_fisico as M, material_fisica as F
rng = np.random.default_rng(77)
h = F.Conceito([("x","vy"),("y","vx")], np.array([1.,-1.]), 1.0, {}, "h")
def E_or(mu): return F.Conceito([("vx","vx"),("vy","vy"),("1/r",)], np.array([.5,.5,-mu]), 1.0, {}, "E")
for molde in ["q:rmin@15", "q:varre@40", "q:vmax@5"]:
  for fam in ["kepler","magnetico","spin_orbita"]:
    rb=ro=0; ab=ao=[]
    accb, acco = [], []
    for rep in range(8):
        P = M.Problema(0, fam, "percebido", "Q", rng, molde=molde)
        yb,_,_,_ = F.prever_q(P.S,P.y,P.consultas(),{})
        yo,_,_,_ = F.prever_q(P.S,P.y,P.consultas(),{"h":h,"E":E_or(P.p["mu"])})
        a,b = P.verifica(yb); c,d = P.verifica(yo)
        rb+=a; ro+=c; accb.append(b); acco.append(d)
    print(f"{molde:11s} {fam:11s} base {rb}/8 (acc {np.mean(accb):.2f})  oraculo h,E {ro}/8 (acc {np.mean(acco):.2f})")
