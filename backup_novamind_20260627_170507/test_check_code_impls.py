"""Verifica se as implementacoes de codigo estao no HDV"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from core.hdc_memory import HDCMemory

h = HDCMemory(device='cuda', hdv_paths=['D:/novamind_data/hdv/python_brain.hdv'])
idx = h.indices['python_brain']

code_items = [m for m in idx if m.get('source') == 'code']
print(f'Code implementations: {len(code_items)}')
for m in code_items:
    print(f'  {m["id"]}: {m["text"][:80]}')

print()
query = 'def is_even'
print(f'Busca por: "{query}"')
r = h.query(query, top_k=5, min_sim=0.05)
for sim, meta, src in r:
    print(f'  sim={sim:.3f} {meta.get("text","")[:80]}')

print()
query = 'def soma'
print(f'Busca por: "{query}"')
r = h.query(query, top_k=5, min_sim=0.05)
for sim, meta, src in r:
    print(f'  sim={sim:.3f} {meta.get("text","")[:80]}')
