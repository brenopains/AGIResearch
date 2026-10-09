"""
H103 — A cultura ESCALA com o tamanho da sociedade? (efeito Tasmania de Henrich, testado)
==========================================================================================
Afirmacao forte e testavel: "quanto maior/mais conectada a sociedade, mais profunda a cultura
cumulativa que ela sustenta" (Henrich; populacoes pequenas PERDEM complexidade — efeito Tasmania).
Rodamos o ratchet (cultura+ToM, ensino na fronteira como o H99) variando o tamanho da populacao P
e medimos a profundidade tecnologica de equilibrio. Se cresce com P, o teto e PARAMETRICO (escala),
nao fundamental. Vetorizado GPU.
"""
import os, time, numpy as np, torch
DEV="cuda" if torch.cuda.is_available() else "cpu"
T=200; STEPS=4000; P_INNOV=0.02; TURNOVER=0.02
torch.manual_seed(0)

def run(P, seed=1):
    g=torch.Generator(device=DEV).manual_seed(seed)
    k=torch.zeros(P,T,dtype=torch.bool,device=DEV); k[:,0]=True; k[:,1]=True
    ar=torch.arange(P,device=DEV)
    def frontier(k):
        r=torch.zeros_like(k); r[:,2:]=k[:,1:-1]&k[:,0:-2]&(~k[:,2:]); return torch.argmax(r.float(),1), r.any(1)
    for s in range(STEPS):
        f,has=frontier(k)
        inv=has&(torch.rand(P,device=DEV,generator=g)<P_INNOV); k[ar[inv],f[inv]]=True
        partner=torch.randperm(P,device=DEV,generator=g)
        f,has=frontier(k); ok=has & k[partner,f]; k[ar[ok],f[ok]]=True   # ToM: ensina a fronteira
        die=torch.rand(P,device=DEV,generator=g)<TURNOVER
        k[die]=False; k[die,0]=True; k[die,1]=True
    d=(k.float()*torch.arange(T,device=DEV).float()).max(1).values
    return float(d.max()), float(d.mean())

def main():
    t0=time.time()
    print("="*64); print(f"H103 Cultura escala com a sociedade? (efeito Tasmania) em {DEV} | cap T={T}")
    print(f"\n  {'populacao':>10} {'prof.MAX':>9} {'prof.media':>11}")
    res=[]
    for P in (128,512,2048,8192,32768):
        mx,me=run(P); res.append((P,mx,me)); print(f"  {P:>10} {mx:>9.0f} {me:>11.1f}")
    print("="*64)
    scales = res[-1][1] > res[0][1]*1.5
    print(f"H103 — {time.time()-t0:.0f}s | prof.MAX: P={res[0][0]} -> {res[0][1]:.0f} | P={res[-1][0]} -> {res[-1][1]:.0f}")
    print(f"  VEREDITO: {'ESCALA — sociedade maior sustenta cultura mais profunda (efeito Tasmania confirmado; teto e PARAMETRICO, nao fundamental)' if scales else 'nao escalou claramente neste range'}.")
    print("  Honesto: cultura cumulativa medida; profundidade = comprimento da cadeia tecnologica sustentada. Nao e AGI.")

if __name__=="__main__": main()
