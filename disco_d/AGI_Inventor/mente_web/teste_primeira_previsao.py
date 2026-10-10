"""A teoria escolhida para prever nao pode depender do desfecho do video novo."""
import numpy as np
import arena
from laboratorio import Laboratorio

lab = Laboratorio(51)
rng = np.random.default_rng(51)
cam = arena.camera(rng)
for i, mundo in enumerate(['giro']*3 + ['mola']*3):
    ep = arena.episodio(mundo, np.random.default_rng(100+i), cam)
    lab.observa(ep.quadros, ep.tempos, fonte='arena', url=f'treino://{i}')
prior = lab.teoria_atual('arena')['id']
ep = arena.episodio('giro', np.random.default_rng(909), cam)
r = lab.observa(ep.quadros, ep.tempos, fonte='arena', url='teste://futuro')
assert r['teoria_prevista'] == prior, (prior, r)
assert r['melhor_retrospectiva'] != prior, 'Controle nao separou a teoria anterior da melhor depois do video'
assert r['primeira'] > 1, 'A mudanca de mundo deveria surpreender a teoria anterior'
# Uma historica medida retrospectivamente nao pode pagar como progresso causal.
for e in lab.ordem[:-1]:
    lab.eps[e].pop('criterio_primeira', None)
assert lab.progresso(r['episodio']) is None, 'Misturou metricas historicas incompatíveis'
print('OK: primeira teoria', prior, '; melhor depois do video', r['melhor_retrospectiva'],
      '; erro causal', r['primeira'], 'px. Aprender do video nao altera a previsao registrada.')
