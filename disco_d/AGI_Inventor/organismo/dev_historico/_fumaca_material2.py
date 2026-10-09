import time, numpy as np
import mundo_fisico as M, material_fisica as F
rng = np.random.default_rng(2)
def h_or():
    return F.Conceito([("x","vy"),("y","vx")], np.array([1.,-1.]), 1.0, {}, "h")
def E_or(mu):
    return F.Conceito([("vx","vx"),("vy","vy"),("1/r",)], np.array([.5,.5,-mu]), 1.0, {}, "E")
NROT = ("rot","v^")
for fam, conc, ext in [("einstein",{"h":h_or()},[]),("magnetico",None,[NROT]),("ciclotron",{},[NROT]),("vortice",{},[NROT])]:
    oks=[]
    for rep in range(10):
        P = M.Problema(0, fam, "percebido", "L", rng)
        cc = conc if conc is not None else {"E":E_or(P.p["mu"])}
        leis,bits,c = F.ajusta_lei([(P.S,P.A)], cc, ext)
        oks.append(P.verifica(leis[0])[0])
    print(f"{fam:10s} oraculo: {sum(oks)}/10   ex {leis[0]!r:70.70s}")
# invencao
for fam in ["kepler","einstein","magnetico","mola","ciclotron"]:
    p = M.params(fam, rng)
    Ttr = M.trajetorias(fam,p,rng,6); Tho = M.trajetorias(fam,p,rng,6)
    t0=time.time()
    out,c = F.inventa(Ttr,Tho,{})
    print(f"{fam:10s} invent: {len(out)} candidatas custo {c:.1f} {time.time()-t0:.2f}s")
    S = M.estados(fam,p,rng,300)
    x,y,vx,vy=S.T; r=np.hypot(x,y); v2=vx**2+vy**2
    hh = x*vy-y*vx; EE = .5*v2-p["mu"]/r
    for sup,w,g,esc,pais in out[:6]:
        C = F.Conceito(sup,w,esc,pais,"q")
        q=C(S)
        print(f"     {str(sup):55.55s} g {g:7.1f} corr h {abs(np.corrcoef(q,hh)[0,1]):.4f} corr E {abs(np.corrcoef(q,EE)[0,1]):.4f} corr v2 {abs(np.corrcoef(q,v2)[0,1]):.3f}")
