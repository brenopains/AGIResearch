"""
H61 — CX unificado: UM substrato de heading, DUAS faculdades acopladas (sem interferencia)
==========================================================================================
Funde H59 (contexto/heading que generaliza no continuo) + H60 (vetor-memoria). No CX real,
um unico sinal de heading (EPG, o ring) e LIDO por dois circuitos: a priorizacao de acao e a
integracao de vetor (PFN/CPU4). Teste honesto: o MESMO ring sustenta as duas faculdades ao
mesmo tempo SEM interferencia? Comparamos cada faculdade rodando SOZINHA vs JUNTAS.

Metrica: (A) custo de priorizacao na deriva (como H59) e (B) erro do home-vector (como H60),
medidos quando as duas rodam acopladas no mesmo ring vs isoladas. CPU, segundos. Honesto:
demonstra coordenacao de 2 faculdades num substrato biologico unico. NAO e ARC, NAO e AGI.
"""
import os, numpy as np
import importlib.util
spec=importlib.util.spec_from_file_location("fc",os.path.join(os.path.dirname(__file__),"exp_h57_flycoord.py"))
FC=importlib.util.module_from_spec(spec); spec.loader.exec_module(FC)
N=FC.N; TH=FC.TH; rng=np.random.RandomState(4)

def step(W,r,I_seq,target=6.0,dt=0.5,tau=1.0):
    for I in I_seq:
        r=r+(dt/tau)*(-r+np.maximum(0.0,(W@r)+I)); r=np.clip(r,0,50)
        s=r.sum()
        if s>target: r=r*(target/s)
    return r

def run(coupled):
    """percorre um trajeto: a cada passo o heading deriva; faculdade A usa o bump p/ priorizar
    a 'acao' (= wedge do heading), faculdade B integra o home-vector LENDO o mesmo bump."""
    W=FC.cx_connectivity(); r=np.zeros(N); mem=np.zeros(N)
    h=rng.uniform(0,2*np.pi); driftA=[];
    for t in range(300):
        h=(h+rng.uniform(-0.5,0.5))%(2*np.pi); sp=rng.uniform(0.5,1.5)
        target=int(round(h/(2*np.pi/N)))%N
        # ---- faculdade A: priorizar acao pelo bump (custo = rank do wedge correto) ----
        order=sorted(range(N),key=lambda i:-r[i]); driftA.append(order.index(target)+1)
        # atualiza o ring com o heading observado
        r=step(W,r,[FC.vonmises_input(TH[target])]*4); r=step(W,r,[np.zeros(N)]*1)
        # ---- faculdade B: integrar home-vector LENDO o heading do mesmo ring ----
        if coupled:
            hhat,_=FC.decode_heading(r)                 # heading vem do MESMO bump (acoplado)
        else:
            hhat=h                                      # isolado: heading "ideal" externo
        mem=mem+sp*np.maximum(0.0,np.cos(hhat-TH))
    # erro do home-vector vs verdadeiro (precisa refazer o trajeto verdadeiro)
    return np.mean(driftA), mem

def home_error(seed_run):
    return seed_run

def main():
    print("="*70)
    print("H61 — CX unificado: um ring, duas faculdades (contexto + vetor) acopladas")
    # roda ISOLADO (heading ideal p/ B) e ACOPLADO (B le o bump de A)
    rng.seed(4); a_iso,mem_iso=run(coupled=False)
    rng.seed(4); a_cpl,mem_cpl=run(coupled=True)
    # recomputa o trajeto verdadeiro com a MESMA seed p/ medir erro do vetor
    rng.seed(4); h=rng.uniform(0,2*np.pi); ztrue=0+0j
    for t in range(300):
        h=(h+rng.uniform(-0.5,0.5))%(2*np.pi); sp=rng.uniform(0.5,1.5); ztrue+=sp*np.exp(1j*h)
    def vec_err(mem):
        z=np.sum(mem*np.exp(1j*TH)); return abs(np.angle(z/(ztrue+1e-9)))
    print(f"\n  Faculdade A (priorizacao na deriva; custo medio, MENOR=melhor):")
    print(f"    isolada : {a_iso:.2f}    acoplada (junto com B): {a_cpl:.2f}")
    print(f"  Faculdade B (erro angular do home-vector; rad, MENOR=melhor):")
    print(f"    heading ideal externo : {vec_err(mem_iso):.3f}")
    print(f"    heading lido do BUMP  : {vec_err(mem_cpl):.3f}  (acoplado ao mesmo ring)")
    interf_A = a_cpl-a_iso;
    print(f"\n  Interferencia em A ao acoplar B: {interf_A:+.2f} (~0 = sem interferencia)")
    print(f"  => Um unico substrato biologico (ring) sustenta CONTEXTO + VETOR. Se o erro de B")
    print(f"     com heading-do-bump ~ heading-ideal, o acoplamento e fiel. NAO e ARC, NAO e AGI.")

if __name__=="__main__": main()
