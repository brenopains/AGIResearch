"""
H59 — "o que nao vimos": onde a estrutura de ANEL realmente importa (generalizacao continua)
=============================================================================================
H58 mostrou (honesto) que em regimes DISCRETOS um leaky counter ganha do ring-attractor —
o util ali e ESQUECER, nao a estrutura de anel. Pergunta que faltou: e quando o contexto
DERIVA SUAVEMENTE num continuo e e preciso prever uma primitiva ADJACENTE ainda nao vista?

A superpotencia biologica do ring-attractor e exatamente essa: o bump se ESPALHA para os
vizinhos (interpola no manifold continuo). Um leaky counter trata primitivas como
independentes -> nao generaliza para o vizinho-nao-visto. Hipotese: nas transicoes de
deriva (a primitiva que resolve acabou de mudar para um vizinho), o RING tem custo menor
que o LEAKY, porque ja pre-ativou o vizinho.

Tarefa: contexto c faz random walk no anel; primitiva que resolve = wedge mais proximo de c.
Metrica: custo de busca GERAL e nas transicoes de DERIVA (onde sol != sol anterior).
CPU, segundos. Honesto: testa QUANDO a estrutura biologica de anel ajuda — nao e ARC/AGI.
"""
import os, numpy as np
import importlib.util
spec=importlib.util.spec_from_file_location("fc",os.path.join(os.path.dirname(__file__),"exp_h57_flycoord.py"))
FC=importlib.util.module_from_spec(spec); spec.loader.exec_module(FC)
N=FC.N; TH=FC.TH; NPRIM=N
rng=np.random.RandomState(2)
STREAM=400

def make_drift_stream(p_move=0.5):
    """c deriva suavemente (+/-1 wedge) no anel; sol = round(c)."""
    tasks=[]; c=rng.randint(NPRIM)
    for t in range(STREAM):
        if rng.rand()<p_move: c=(c+rng.choice([-1,1]))%NPRIM
        tasks.append(c)
    return tasks

def cost(order,sol): return order.index(sol)+1

def run_global(tasks):
    from collections import Counter; u=Counter(); co=[]
    for s in tasks:
        co.append(cost(sorted(range(NPRIM),key=lambda i:-u[i]),s)); u[s]+=1
    return np.array(co)
def run_leaky(tasks,decay=0.85):
    w=np.zeros(NPRIM); co=[]
    for s in tasks:
        co.append(cost(sorted(range(NPRIM),key=lambda i:-w[i]),s)); w*=decay; w[s]+=1.0
    return np.array(co)
def _step(W,r,I_seq,dt=0.5,tau=1.0,target=6.0):
    for I in I_seq:
        r=r+(dt/tau)*(-r+np.maximum(0.0,(W@r)+I)); r=np.clip(r,0,50)
        s=r.sum()                      # Delta7: inibicao global = normalizacao divisiva (bump unico)
        if s>target: r=r*(target/s)
    return r
def run_ring(tasks,reinforce=4,decay_steps=1):
    W=FC.cx_connectivity(); r=np.zeros(NPRIM); co=[]
    for s in tasks:
        co.append(cost(sorted(range(NPRIM),key=lambda i:-r[i]),s))
        r=_step(W,r,[FC.vonmises_input(TH[s])]*reinforce)
        r=_step(W,r,[np.zeros(NPRIM)]*decay_steps)
    return np.array(co)

def split_drift(tasks,co):
    """separa custo em passos de DERIVA (sol mudou) vs ESTAVEL (sol igual ao anterior)."""
    drift=[]; stable=[]
    for t in range(1,len(tasks)):
        (drift if tasks[t]!=tasks[t-1] else stable).append(co[t])
    return np.mean(drift) if drift else 0, np.mean(stable) if stable else 0, len(drift)

def main():
    print("="*72)
    print("H59 — onde a estrutura de ANEL importa: contexto que DERIVA no continuo")
    print(f"  fluxo={STREAM} | {NPRIM} primitivas | acaso~{(NPRIM+1)/2:.1f}")
    print(f"\n  {'p_move':>6} | {'metodo':>10} | {'custo GERAL':>11} | {'DERIVA':>7} {'ESTAVEL':>8}")
    for pm in (0.3,0.6,0.9):
        tasks=make_drift_stream(pm)
        for name,fn in [('GLOBAL',run_global),('LEAKY',run_leaky),('RING(bio)',run_ring)]:
            co=fn(tasks); d,s,nd=split_drift(tasks,co)
            print(f"  {pm:>6} | {name:>10} | {co.mean():>11.2f} | {d:>7.2f} {s:>8.2f}")
        print()
    print("  Leitura: o que importa e a coluna DERIVA (prever o vizinho recem-mudado).")
    print("  Se RING < LEAKY na DERIVA, a estrutura de anel (generalizacao p/ vizinho) ajuda")
    print("  de verdade — a superpotencia biologica que o leaky counter nao tem. Honesto: NAO e ARC/AGI.")

if __name__=="__main__": main()
