"""Testa o novo python_brain.hdv"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from core.hdc_memory import HDCMemory

h = HDCMemory(device='cuda', hdv_paths=['D:/novamind_data/hdv/python_brain.hdv'])
m = h.matrices['python_brain']
print(f'python_brain: {m.shape[0]} itens')

queries = [
    'is even number',
    'sort array',
    'gravity force',
    'encode text',
    'bubble sort',
    'fibonacci',
    'palindrome string',
]

for q in queries:
    results = h.query(q, top_k=3, min_sim=0.05)
    print(f'\nBusca: "{q}"')
    for sim, meta, src in results:
        txt = meta.get('text','')[:80]
        print(f'  sim={sim:.3f}  {txt}')
