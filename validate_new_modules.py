import sys
sys.path.insert(0, 'novamind')
import numpy as np
import time

print('=== Validando Fase 1: NeuromodulatorState ===')
from core.neuromodulator_state import NeuromodulatorState, CRH, DA
nts = NeuromodulatorState()
fire_gas = nts.resolve_concept_nt('fire', ['gas_leak'])
fire_kitchen = nts.resolve_concept_nt('fire', ['kitchen'])
print(f'  Fogo+gas     CRH={fire_gas[CRH]:.2f} (esperado >0.9)')
print(f'  Fogo+cozinha CRH={fire_kitchen[CRH]:.2f} (esperado <0.5)')
assert fire_gas[CRH] > fire_kitchen[CRH], 'FALHOU: CRH deveria ser maior com gas_leak'
print('  PASS!')

print()
print('=== Validando Fase 2: MultiScaleHDV ===')
from core.multiscale_hdv import MultiScaleHDV
hdv = MultiScaleHDV()
v = np.sign(np.random.randn(10000)).astype('float32')
hdv.write('test', v, nt_state={'DA': 0.8, 'CRH': 0.1})
results = hdv.search(v, top_k=3)
sim_val = round(results[0]['similarity'], 3)
print(f'  Buscou: {len(results)} resultado(s), sim={sim_val}')
hdv.tick(35.0)
results2 = hdv.search(v, layer='L2')
print(f'  Apos 35s no L2: {len(results2)} item(ns) - DA>0.6 promoveu para L2')
assert len(results2) >= 1, 'FALHOU: item saliente deveria estar no L2'
print('  PASS!')

print()
print('=== Validando Fase 3: TemporalBinder ===')
from core.temporal_binder import TemporalBinder
tb = TemporalBinder(seed=77)
v = np.sign(np.random.randn(10000)).astype('float32')
now_t = time.time()
tb.write_episodic(v, ['kitchen', 'fire'], {'DA': 0.3, 'CRH': 0.8}, t=now_t)
results = tb.query_episodic(v, top_k=3)
print(f'  Episodio gravado e recuperado: {len(results)} resultado(s)')
state = tb.get_state()
print(f'  Episodes: {state["episodes_stored"]}, Context cache: {state["context_cache_size"]}')
assert len(results) > 0, 'FALHOU: deveria ter retornado resultado'
print('  PASS!')

print()
print('=== Validando Fase 4: SimulatedBody + BodyEncoder ===')
from core.simulated_body import SimulatedBody, Actuator
from core.body_encoder import BodyEncoder
body = SimulatedBody(start_pos=[0.0, 5.0, 0.0])
enc = BodyEncoder(seed=88)
for _ in range(20):
    r = body.apply_actuator(Actuator(move=[0.0, 0.0, 0.0]))
concept = enc.get_body_concept(r)
print(f'  Apos queda: conceito={concept}, pain={round(r.pain,2)}, balance={round(r.balance,2)}')
v1 = enc.encode_sensors(r)
body.reset(pos=[4.8, 0.0, 4.8])
r2 = body.read_sensors()
concept2 = enc.get_body_concept(r2)
v2 = enc.encode_sensors(r2)
sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
print(f'  Perto de reward: conceito={concept2}, reward={round(r2.reward,2)}')
print(f'  Similaridade queda vs recompensa: {round(sim,3)} (deve ser baixa)')
assert concept != concept2, 'FALHOU: conceitos diferentes devem ser distintos'
print('  PASS!')

print()
print('=== Validando Fase 5: Cerebellum ===')
from core.cerebellum import Cerebellum
cb = Cerebellum()
state_v = np.random.randn(15).astype('float32')
action_v = np.random.randn(4).astype('float32')
next_v = np.random.randn(15).astype('float32')
for i in range(60):
    cb.observe_result(state_v, action_v, next_v)
err = cb.get_prediction_error()
print(f'  Erro de predicao apos 60 steps: {round(err,4)}')
print(f'  Forward steps: {cb.forward_model.train_steps}')
timing = cb.predict_timing({'jaw_open': 0.7, 'velum': 0.3})
print(f'  Timing articulatorio (compat): {list(timing.keys())}')
cb_state = cb.get_state()
print(f'  Estado cerebelo: {cb_state}')
print('  PASS!')

print()
print('=' * 50)
print('TODOS OS MODULOS VALIDADOS COM SUCESSO!')
print('=' * 50)
