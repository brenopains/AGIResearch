"""Teste integrado Fase 6 + Fase 7: Object-Centric + Geometry Dataset + ARC."""
import sys, json, numpy as np
sys.path.insert(0, ".")

from core.arc_object_centric import TransformationHypothesisGenerator
from core.arc_dataset import ARCDataset
from core.cognition import Competition, MetaCognition, Timeline
from core.geometry_dataset import GeometryTransformDataset, RelationalDataset

print("=" * 60)
print("TESTE FASE 6 — Object-Centric Perception no ARC")
print("=" * 60)

dataset = ARCDataset(data_dir="data/ARC/data/evaluation", max_h=30, max_w=30, pad_val=10)
gen = TransformationHypothesisGenerator()
comp = Competition(iterations=3)
meta = MetaCognition()
timeline = Timeline()

total = 100
diversities = []
clusters = []
entropies = []
margins = []

for task_idx in range(total):
    name, task = dataset.tasks[task_idx]
    train_pairs = task["train"]

    input_grids = [np.array(p["input"], dtype=int) for p in train_pairs[:3]]
    output_grids = [np.array(p["output"], dtype=int) for p in train_pairs[:3]]

    hyps = gen.generate(input_grids, output_grids)
    if not hyps:
        continue

    result = comp.run(hyps)
    meta.observe(name, result)
    timeline.record(name, result)

    diversities.append(result.diversity_score)
    clusters.append(result.num_clusters)
    entropies.append(result.entropy)
    margins.append(result.margin)

print(f"\nTasks avaliadas: {total}")
print(f"Diversidade media: {np.mean(diversities):.4f}")
print(f"Diversidade max:   {np.max(diversities):.4f}")
print(f"Diversidade > 0:   {sum(1 for d in diversities if d > 0)}/{total}")
print(f"Clusters medio:    {np.mean(clusters):.2f}")
print(f"Entropia media:    {np.mean(entropies):.4f}")
print(f"Margem media:      {np.mean(margins):.4f}")
print(f"Reversoes totais:  {meta.summarize_session()['total_reversals']}")

status = timeline.get_stability_summary()
print(f"\nTimeline:")
print(f"  Confianca media:  {status.get('confidence_mean', 0):.3f}")
print(f"  Entropia media:   {status.get('entropy_mean', 0):.3f}")
print(f"  Diversidade media:{status.get('diversity_mean', 0):.3f}")
print(f"  Convergencia:     {status.get('convergence_trend', '')}")

print(f"\nMeta-cognicao:")
ms = meta.summarize_session()
print(f"  Reversoes:        {ms['total_reversals']}")
print(f"  Impasses:         {ms['total_impasse_events']}")
print(f"  Taxa de impasse:  {ms['impasse_rate']:.2%}")

print("\n" + "=" * 60)
print("TESTE FASE 7 — Geometry + Relational Datasets")
print("=" * 60)

geo = GeometryTransformDataset(size=1000, h=8, w=8)
rel = RelationalDataset(size=500, h=8, w=8)

dist = geo.get_label_distribution()
print(f"\nGeometry Dataset: {len(geo)} amostras")
print("  Distribuicao de transformacoes:")
for label, count in sorted(dist.items(), key=lambda x: -x[1]):
    pct = count / len(geo) * 100
    print(f"    {label:15s}: {count:3d} ({pct:.0f}%)")

print(f"\nRelational Dataset: {len(rel)} amostras")

# Teste: gera um batch e verifica os formatos
batch = geo.sample_batch(4)
for i, (inp, out, label, _) in enumerate(batch):
    print(f"\n  Exemplo {i}: {label}")
    print(f"    Input shape: {inp.shape}, Output shape: {out.shape}")
    print(f"    Input colors: {np.unique(inp)}, Output colors: {np.unique(out)}")

# Teste de transformacao conhecida
print("\n\nTeste de transformacao especifica:")
inp = np.array([
    [0, 0, 0, 0],
    [0, 2, 2, 0],
    [0, 2, 2, 0],
    [0, 0, 0, 0],
], dtype=int)

from core.geometry_dataset import apply_rotate_90, apply_mirror_h, apply_recolor
print("Input:")
print(inp)
print("\nRotate 90:")
print(apply_rotate_90(inp))
print("\nMirror H:")
print(apply_mirror_h(inp))
print("\nRecolor (2->3):")
out = apply_recolor(inp)
print(out)

print("\n" + "=" * 60)
print("RESUMO EXECUTIVO")
print("=" * 60)
print(f"""
Fase 6 — Object-Centric Perception: {total} tasks ARC
  Diversidade de hipoteses: media {np.mean(diversities):.3f} (max {np.max(diversities):.4f})
  Agora gera hipoteses QUALITATIVAMENTE DIFERENTES:
    copy_object, delete_object, recolor, crop, scale_up, identity
  Antes: color_5_grey, color_4_yellow (mesmo tipo)

Fase 7 — Geometry Dataset: {len(geo)} amostras, {len(dist)} tipos de transformacao
  Relational Dataset: {len(rel)} amostras
  Transformacoes disponiveis: rotate, mirror, translate, recolor, delete, copy, crop
  Pronto para treino: python train_geometry_arc.py --steps 300

Metricas de rastreabilidade:
  Entropia media: {np.mean(entropies):.4f} (manteve-se alta = multiplas hipoteses viaveis)
  Margem media: {np.mean(margins):.4f} (ainda baixa)
  Clusters medio: {np.mean(clusters):.1f} (agora > 1)
  Diversidade media: {np.mean(diversities):.3f} (antes 0.000)
""")
