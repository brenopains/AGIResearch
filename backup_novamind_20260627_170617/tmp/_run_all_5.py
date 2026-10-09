"""Roda todas as 5 frentes em sequência, salvando resultado consolidado."""
import time
import sys
import json
sys.path.insert(0, ('C:\\Users\\usuario\\Desktop\\AGI 2\\novamind').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~')))
from agi_test_all_5 import (load_arc_tasks,
                            frente_1_biblioteca,
                            frente_2_recrutamento,
                            frente_3_nascer_operador,
                            frente_4_memoria_cross,
                            frente_5_mcts_guiado)

tasks = load_arc_tasks(limit=25)
print(f"Tasks carregadas: {len(tasks)}")
print()

results = {}
t0 = time.time()

# F1 — só roda uma vez e reusa
print("[1/5] FRENTE 1: BIBLIOTECA")
results['F1'] = frente_1_biblioteca(tasks, max_time_per_task=8.0)
print(f"  -> {results['F1']['solved']}/{results['F1']['total']} = {results['F1']['pct']:.1%} em {results['F1']['time']:.1f}s\n")

# F2
print("[2/5] FRENTE 2: AUTO-RECRUTAMENTO")
results['F2'] = frente_2_recrutamento(tasks)
print(f"  -> {results['F2'].get('solved', 0)}/{results['F2'].get('total', 25)} = {results['F2'].get('pct', 0):.1%} em {results['F2'].get('time', 0):.1f}s\n")

# F3
print("[3/5] FRENTE 3: NASCER OPERADOR")
results['F3'] = frente_3_nascer_operador(tasks)
print(f"  -> {results['F3'].get('solved', 0)}/{results['F3'].get('total', 25)} = {results['F3'].get('pct', 0):.1%} em {results['F3'].get('time', 0):.1f}s\n")

# F4
print("[4/5] FRENTE 4: MEMÓRIA CROSS-DOMAIN")
results['F4'] = frente_4_memoria_cross(tasks)
print(f"  -> {results['F4'].get('solved', 0)}/{results['F4'].get('total', 25)} = {results['F4'].get('pct', 0):.1%} em {results['F4'].get('time', 0):.1f}s\n")

# F5
print("[5/5] FRENTE 5: MCTS GUIADO")
results['F5'] = frente_5_mcts_guiado(tasks, max_time_per_task=12.0)
print(f"  -> {results['F5'].get('solved', 0)}/{results['F5'].get('total', 25)} = {results['F5'].get('pct', 0):.1%} em {results['F5'].get('time', 0):.1f}s\n")

total_time = time.time() - t0

# Salva
out_path = ('C:\\Users\\usuario\\Desktop\\AGI 2\\novamind\\tmp\\agi_test_results.json').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~'))
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump({
        'n_tasks': len(tasks),
        'total_time_s': total_time,
        'fronts': results,
    }, f, ensure_ascii=False, indent=2, default=str)
print(f"\nResultados salvos em: {out_path}")
print(f"Tempo total: {total_time:.1f}s")
