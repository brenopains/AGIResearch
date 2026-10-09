"""
H60 — Vetor-memoria do complexo central: integracao de "home vector" + robustez populacional
==============================================================================================
Segue a linha biologica. O CX da mosca nao so rastreia direcao (ring-attractor, H57-59) — ele
INTEGRA UM VETOR (path integration / "home vector"): acumula deslocamento numa populacao
sinusoidal (celulas tipo PFN/CPU4) ancorada no bump de heading. Ref: Stone et al. 2017
(Current Biology, modelo de path integration do CX), Hulse 2020 (conectoma).

Duas variaveis ACOPLADAS no MESMO substrato: heading (onde estou virado) + vetor acumulado
(onde fica "casa"). Testamos:
 (1) o circuito integra o home-vector com baixo erro (segunda faculdade do mesmo substrato);
 (2) HIPOTESE BIOLOGICA: o CODIGO POPULACIONAL (16 wedges) degrada graciosamente sob ruido
     por-neuronio, enquanto um integrador MINIMO de 2 numeros (dx,dy) e fragil ao mesmo ruido.
CPU, segundos. Honesto: e o modelo de path integration documentado do CX (nao inedito); nossa
parte e usa-lo como faculdade no nosso sistema + medir a robustez. NAO move ARC, NAO e AGI.
"""
import os, numpy as np
import importlib.util
spec=importlib.util.spec_from_file_location("fc",os.path.join(os.path.dirname(__file__),"exp_h57_flycoord.py"))
FC=importlib.util.module_from_spec(spec); spec.loader.exec_module(FC)
N=FC.N; TH=FC.TH                       # 16 wedges, headings preferidos
rng=np.random.RandomState(3)

def random_walk(T):
    """sequencia de (heading, speed). heading deriva; speed ~ U(0.5,1.5)."""
    h=rng.uniform(0,2*np.pi); steps=[]
    for _ in range(T):
        h=(h+rng.uniform(-0.6,0.6))%(2*np.pi); sp=rng.uniform(0.5,1.5); steps.append((h,sp))
    return steps

def true_vector(steps):
    z=sum(sp*np.exp(1j*h) for h,sp in steps); return z   # deslocamento liquido (complexo)

# ---------- integrador CX: codigo POPULACIONAL sinusoidal (16 wedges) ----------
def cx_integrate(steps, neuron_noise=0.0):
    mem=np.zeros(N)                                   # celulas de memoria por wedge (PFN/CPU4)
    for h,sp in steps:
        drive=sp*np.maximum(0.0,np.cos(h-TH))         # componente do movimento na dir. de cada wedge
        mem=mem+drive
        if neuron_noise>0: mem=mem+rng.randn(N)*neuron_noise
    z=np.sum(mem*np.exp(1j*TH))                        # decodifica vetor da populacao
    return z

# ---------- integrador MINIMO: 2 numeros (dx,dy) ----------
def min_integrate(steps, neuron_noise=0.0):
    dx=dy=0.0
    for h,sp in steps:
        dx+=sp*np.cos(h); dy+=sp*np.sin(h)
        if neuron_noise>0: dx+=rng.randn()*neuron_noise; dy+=rng.randn()*neuron_noise
    return dx+1j*dy

def ang_err(zhat,ztrue):
    return abs(np.angle(zhat/ (ztrue+1e-12)))          # erro angular do home-vector (rad)

def main():
    print("="*72)
    print("H60 — vetor-memoria (home vector) do complexo central: 2 variaveis no mesmo substrato")
    T=20; trials=300
    # (1) integracao limpa
    cx_e=[]; mn_e=[]
    for _ in range(trials):
        st=random_walk(T); zt=true_vector(st)
        cx_e.append(ang_err(cx_integrate(st),zt)); mn_e.append(ang_err(min_integrate(st),zt))
    print(f"\n(1) Integracao do home-vector (T={T} passos, sem ruido) — erro angular medio:")
    print(f"    CX populacional (16 wedges): {np.mean(cx_e):.3f} rad")
    print(f"    integrador minimo (dx,dy)  : {np.mean(mn_e):.3f} rad")
    print(f"    (ambos integram bem; CX usa codigo sinusoidal, o minimo usa Cartesiano ideal)")

    print(f"\n(2) HIPOTESE: robustez a ruido POR-NEURONIO (mesmo sigma nos dois) — erro angular:")
    print(f"    {'sigma':>6} | {'CX populacional':>16} | {'minimo (dx,dy)':>15}")
    for sig in (0.0,0.5,1.0,2.0,4.0):
        ce=[]; me=[]
        for _ in range(trials):
            st=random_walk(T); zt=true_vector(st)
            ce.append(ang_err(cx_integrate(st,sig),zt)); me.append(ang_err(min_integrate(st,sig),zt))
        print(f"    {sig:>6} | {np.mean(ce):>16.3f} | {np.mean(me):>15.3f}")
    print(f"\n  Leitura honesta: se o CX (16 unidades) mantem erro menor que o minimo (2 numeros)")
    print(f"  sob o MESMO ruido por unidade, o CODIGO POPULACIONAL e a vantagem biologica real")
    print(f"  (a redundancia distribuida promedia o ruido ~1/sqrt(N)). Segunda faculdade no MESMO")
    print(f"  substrato do ring (heading + vetor). NAO move o ARC, NAO e AGI — peca de coordenacao.")

if __name__=="__main__": main()
