"""Regression checks for competition between cheap and expensive exploration arms."""
import collections
import os
from pathlib import Path
import sys

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from mente import Mente,Reg
import digestao as G

def subject():
    m=Mente.__new__(Mente)
    m.t=0;m.regs={};m.lac={};m.custo={'porta:arena':0.5,'porta:wiki_en':10}
    m.funciona={'porta:arena':[1000,0],'porta:wiki_en':[1000,0]}
    xp=G._idx(b'regression_context').astype(np.float32)
    xp/=np.linalg.norm(xp)
    m.ctx_pagina=lambda pg:xp
    m.disponiveis=lambda pg:['porta:arena','porta:wiki_en']
    m.candidatas=lambda n:[]
    x=np.concatenate([xp,G._idx(b'sem_lacuna')/np.sqrt(G.D_CTX)])/np.sqrt(2)
    return m,x

def choices(m):
    count=collections.Counter()
    for seed in range(1000):
        m.rng=np.random.default_rng(seed)
        count[m.escolhe(None)[0]]+=1
    return count

m,x=subject()
for action in m.disponiveis(None):
    reg=m.regs[action]=Reg()
    for i in range(500):reg.add(x,i%2==0)
equal=choices(m)
assert 350<=equal['porta:arena']<=650,dict(equal)

m,x=subject()
for action in m.disponiveis(None):
    reg=m.regs[action]=Reg()
    for i in range(500):reg.add(x,(i%10<4) if action=='porta:arena' else (i%10<7))
better=choices(m)
assert better['porta:wiki_en']>=950,dict(better)

m,x=subject()
m.funciona['porta:wiki_en']=[0,1000]
failed=choices(m)
assert failed['porta:arena']>=990,dict(failed)
print('OK: retorno igual nao fica preso na opcao barata:',dict(equal))
print('OK: porta lenta com melhor retorno vence:',dict(better))
print('OK: ferramenta indisponivel continua penalizada:',dict(failed))
