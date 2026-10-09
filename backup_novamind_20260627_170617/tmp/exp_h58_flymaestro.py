"""
H58 — FlyMaestro: o ring-attractor do conectoma COORDENA o maestro (contexto persistente)
==========================================================================================
Conecta de fato o circuito biologico do H57 aos nossos modulos. Cenario honesto onde a
COORDENACAO importa: um FLUXO de tarefas com REGIMES — o conjunto de primitivas uteis muda
ao longo do tempo (troca a cada K tarefas), e o agente NAO recebe o rotulo do regime.

Hipotese: a memoria persistente-mas-atualizavel do ring-attractor (bump de "contexto")
rastreia o regime atual e prioriza as primitivas certas -> MENOS busca. Comparado a um
maestro com contagem GLOBAL (Counter, estilo nova_maestro), que confunde regimes (lembra
tudo, esquece devagar) e re-descobre a cada troca.

Metrica: custo medio de busca (nº de primitivas tentadas ate resolver) e velocidade de
ADAPTACAO logo apos cada troca de regime. CPU, segundos. Honesto: testa se um coordenador
biologico primitivo melhora a coordenacao temporal; NAO e ARC, NAO e AGI.
"""
import os, numpy as np
from collections import Counter
import importlib.util

# reusa o ring-attractor do H57
spec=importlib.util.spec_from_file_location("fc",os.path.join(os.path.dirname(__file__),"exp_h57_flycoord.py"))
FC=importlib.util.module_from_spec(spec); spec.loader.exec_module(FC)
N=FC.N; TH=FC.TH

rng=np.random.RandomState(1)
NPRIM=N                          # uma primitiva por wedge do anel
REGIME_W=3                       # primitivas ativas por regime (contiguas no anel)
STREAM=240                       # tamanho do fluxo

def make_stream(switch):
    tasks=[]; centers=[]
    c=rng.randint(NPRIM)
    for t in range(STREAM):
        if t>0 and t%switch==0:
            c=(c+rng.randint(3,NPRIM-2))%NPRIM        # salta para outro regime
        active=[(c+d)%NPRIM for d in range(-(REGIME_W//2),REGIME_W//2+1)]
        sol=active[rng.randint(len(active))]          # primitiva que resolve esta tarefa
        tasks.append(sol); centers.append(c)
    return tasks,centers

def solve_with_order(order, sol):
    """tenta primitivas na ordem dada; custo = posicao (1-based) da que resolve."""
    return order.index(sol)+1

# ---------- maestro A: contagem GLOBAL (baseline estilo nova_maestro) ----------
def run_global(tasks):
    useful=Counter(); costs=[]
    for sol in tasks:
        order=sorted(range(NPRIM), key=lambda i:-useful[i])
        costs.append(solve_with_order(order,sol)); useful[sol]+=1
    return np.array(costs)

# ---------- maestro B: contexto via RING-ATTRACTOR biologico ----------
def run_ring(tasks, decay_steps=1, reinforce_steps=4):
    W=FC.cx_connectivity(); r=np.zeros(NPRIM); costs=[]
    for sol in tasks:
        # ordena primitivas pela ativacao do bump (contexto atual)
        order=sorted(range(NPRIM), key=lambda i:-r[i])
        costs.append(solve_with_order(order,sol))
        # atualiza o contexto: injeta a primitiva que resolveu (passos suficientes p/ formar bump)
        r=_step(W,r,[FC.vonmises_input(TH[sol])]*reinforce_steps)
        # passo(s) sem entrada entre tarefas (persistencia: o bump segura o contexto)
        r=_step(W,r,[np.zeros(NPRIM)]*decay_steps)
    return np.array(costs)

# ---------- controle: contador global COM esquecimento (isola decay vs estrutura de anel) ----------
def run_leaky(tasks, decay=0.85):
    w=np.zeros(NPRIM); costs=[]
    for sol in tasks:
        order=sorted(range(NPRIM), key=lambda i:-w[i])
        costs.append(solve_with_order(order,sol)); w*=decay; w[sol]+=1.0
    return np.array(costs)

def _step(W,r,I_seq,dt=0.5,tau=1.0):
    for I in I_seq:
        r=r+(dt/tau)*(-r+np.maximum(0.0,(W@r)+I)); r=np.clip(r,0,50)
    return r

def adapt_profile(costs):
    """custo medio em funcao da posicao DESDE a ultima troca de regime (0..SWITCH-1)."""
    prof=np.zeros(SWITCH); cnt=np.zeros(SWITCH)
    for t,c in enumerate(costs):
        p=t%SWITCH; prof[p]+=c; cnt[p]+=1
    return prof/np.maximum(cnt,1)

def main():
    print("="*72)
    print("H58 FlyMaestro — ring-attractor do conectoma como CONTEXTO do maestro")
    print(f"  fluxo={STREAM} | {NPRIM} primitivas | regime={REGIME_W} ativas | acaso~{(NPRIM+1)/2:.1f}")
    print(f"  custo medio de busca (MENOR=melhor), varrendo velocidade de troca de regime:\n")
    print(f"  {'troca a cada':>12} | {'GLOBAL':>8} {'LEAKY':>8} {'RING(bio)':>10} | melhor")
    for switch in (3,4,6,8,12):
        tasks,_=make_stream(switch)
        cg=run_global(tasks).mean(); cl=run_leaky(tasks).mean(); cr=run_ring(tasks).mean()
        best=min([('GLOBAL',cg),('LEAKY',cl),('RING',cr)],key=lambda x:x[1])[0]
        print(f"  {switch:>12} | {cg:>8.2f} {cl:>8.2f} {cr:>10.2f} | {best}")
    print("\n  Leitura honesta: GLOBAL=Counter (nunca esquece); LEAKY=Counter com esquecimento")
    print("  (isola o efeito de 'esquecer'); RING=circuito biologico (esquece + generaliza p/ vizinhos).")
    print("  O ring so deveria vencer se a estrutura de ANEL (vizinhanca dos regimes contiguos)")
    print("  ajudar alem do simples esquecimento. NAO e ARC, NAO e AGI.")

if __name__=="__main__": main()
