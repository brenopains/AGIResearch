import numpy as np
import mundo_fisico as M, material_fisica as F
rng = np.random.default_rng(6)
h = F.Conceito([("x","vy"),("y","vx")], np.array([1.,-1.]), 1.0, {}, "h")
def E_or(mu): return F.Conceito([("vx","vx"),("vy","vy"),("1/r",)], np.array([.5,.5,-mu]), 1.0, {}, "E")
for fam in ["spin_orbita"]:
    r={"b":0,"o":0}
    for rep in range(20):
        P = M.Problema(0, fam, "percebido", "L", rng)
        l,_,_ = F.ajusta_lei([(P.S,P.A)], {}, []); r["b"]+=P.verifica(l[0])[0]
        l,_,_ = F.ajusta_lei([(P.S,P.A)], {"h":h}, []); r["o"]+=P.verifica(l[0])[0]
    print(fam, "base", r["b"], "/20  oraculo h", r["o"], "/20   ex", l[0])
# perguntas
import time
for molde in M.MOLDES_DEV + M.MOLDES_RESERVA:
  for fam in ["kepler","mola","spin_orbita","magnetico"]:
    rb=ro=0; t0=time.time()
    for rep in range(10):
        P = M.Problema(0, fam, "percebido", "Q", rng, molde=molde)
        Sq = P.consultas()
        mu=P.p["mu"]
        # LRL oraculo (kepler): A_x = vy*h - mu x/r
        lx = F.Conceito([("vy","h"),("x","1/r")], np.array([1.,-mu]),1.0,{"h":h},"Ax")
        ly = F.Conceito([("vx","h"),("y","1/r")], np.array([-1.,-mu]),1.0,{"h":h},"Ay")
        yb,_,_,_ = F.prever_q(P.S,P.y,Sq,{})
        yo,_,_,_ = F.prever_q(P.S,P.y,Sq,{"h":h,"E":E_or(mu),"Ax":lx,"Ay":ly})
        rb+=P.verifica(yb)[0]; ro+=P.verifica(yo)[0]
    print(f"{molde:16s} {fam:11s} base {rb}/10  oraculo {ro}/10   {time.time()-t0:.1f}s")
