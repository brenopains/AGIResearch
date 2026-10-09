"""Teste da camada Object-Centric Perception em tasks ARC reais."""
import sys, json, numpy as np
sys.path.insert(0, ".")

from core.arc_object_centric import ARCGridParser, RelationExtractor, TransformationHypothesisGenerator, GridVisualizer
from core.arc_dataset import ARCDataset
from core.cognition import Competition
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = ARCDataset(data_dir="data/ARC/data/evaluation", max_h=30, max_w=30, pad_val=10, dev=device)

parser = ARCGridParser()
rel_ext = RelationExtractor()
gen = TransformationHypothesisGenerator()
competition = Competition(iterations=3)

print(f"Total tasks: {len(dataset.tasks)}")
print("=" * 60)

n_test = min(10, len(dataset.tasks))
all_diversities = []

for task_idx in range(n_test):
    name, task = dataset.tasks[task_idx]
    train_pairs = task["train"]
    test_pairs = task["test"]

    # Converte grids de treino para numpy
    input_grids = []
    output_grids = []
    for pair in train_pairs[:3]:
        inp = np.array(pair["input"], dtype=int)
        out = np.array(pair["output"], dtype=int)
        input_grids.append(inp)
        output_grids.append(out)

    # Extrai objetos do primeiro par
    if input_grids:
        objects_in = parser.extract_objects(input_grids[0])
        objects_out = parser.extract_objects(output_grids[0])
        relations_in = rel_ext.extract(objects_in)
        relations_out = rel_ext.extract(objects_out)

        print(f"\n[{name}]")
        print(f"  Input objects:  {len(objects_in)}")
        for obj in objects_in:
            print(f"    {obj.id}: color={obj.color} shape={obj.shape_type} "
                  f"area={len(obj.pixels)} center=({obj.center[0]:.0f},{obj.center[1]:.0f})")
        print(f"  Output objects: {len(objects_out)}")
        for obj in objects_out:
            print(f"    {obj.id}: color={obj.color} shape={obj.shape_type} "
                  f"area={len(obj.pixels)} center=({obj.center[0]:.0f},{obj.center[1]:.0f})")
        print(f"  Input relations: {len(relations_in)}")
        for r in relations_in[:5]:
            print(f"    {r['from']} {r['type']} {r['to']}")
        print(f"  Output relations: {len(relations_out)}")
        for r in relations_out[:5]:
            print(f"    {r['from']} {r['type']} {r['to']}")

        # Gera hipoteses de transformacao
        hypotheses = gen.generate(input_grids, output_grids)
        print(f"  Hypotheses generated: {len(hypotheses)}")
        for h in hypotheses[:8]:
            print(f"    {h.name}: conf={h.confidence:.3f}")

        # Roda competicao
        if hypotheses:
            result = competition.run(hypotheses)
            print(f"  Competition:")
            print(f"    Winner: {result.winner} (conf={result.winner_confidence:.3f})")
            print(f"    Diversity: {result.diversity_score:.4f}")
            print(f"    Clusters: {result.num_clusters}")
            print(f"    Entropy: {result.entropy:.4f}")
            print(f"    Margin: {result.margin:.4f}")
            all_diversities.append(result.diversity_score)
        else:
            print(f"  No hypotheses generated")
            all_diversities.append(0.0)

print("\n" + "=" * 60)
print(f"Media de diversidade: {np.mean(all_diversities):.4f}")
print(f"Max diversidade: {np.max(all_diversities):.4f}")
print(f"Tasks com diversidade > 0: {sum(1 for d in all_diversities if d > 0)}/{n_test}")
