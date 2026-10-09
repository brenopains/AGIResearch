"""Teste rapido da instrumentacao nova."""
import sys
sys.path.insert(0, ".")

from core.cognition import *
from core.evidence_builders.taxonomy_builder import TaxonomyBuilder
from core.evidence_builders.episodic_builder import EpisodicBuilder

# Test contracts
cb = ContributionBreakdown(source=EvidenceSource.PERCEPTUAL, score=0.5, n_items=3)
cs = ConvergenceStep(iteration=0, scores={"a": 0.5, "b": 0.3}, winner="a", margin=0.2, entropy=0.8)
h = Hypothesis(name="test", category="is_a", contribution_breakdown=[cb])

# Test competition
comp = Competition(iterations=3)
result = comp.run([h])
print(f"Winner: {result.winner}")
print(f"Diversity: {result.diversity_score}")
print(f"Clusters: {result.num_clusters}")
print(f"Reversals: {result.winner_reversals}")
print(f"Velocity: {result.convergence_velocity}")
print(f"Steps: {len(result.convergence_steps)}")
for step in result.convergence_steps:
    print(f"  {step.to_dict()}")

print(f"Breakdown: {result.hypotheses[0].contribution_breakdown}")

# Test timeline
tl = Timeline()
tl.record("test", result, attention_timeline=[{"concept": "whale", "activation": 0.8}])
print(f"Timeline: {len(tl.entries)} entries")
print(f"Diversity: {tl.get_diversity_trace()}")

# Test meta
meta = MetaCognition()
obs = meta.observe("test", result)
print(f"Impasse: {obs['impasse']}")
print(f"Style: {obs['reasoning_style']}")

# Test pipeline
pipe = CognitionPipeline(competition=comp)
result2 = pipe.run("test focus")
print(f"Pipeline: {result2.winner}")

# Test ARC trace integration
from core.grid_cognition import GridWorldModel
from core.arc_dataset import ARCDataset
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = ARCDataset(data_dir="data/ARC/data/evaluation", max_h=30, max_w=30, pad_val=10, dev=device)
model = GridWorldModel(d=128, heads=4, layers=3, max_hw=900, colors=11).to(device)
ckpt = torch.load("arc_model_trained.pt", map_location=device)
if "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
elif isinstance(ckpt, dict):
    model.load_state_dict(ckpt, strict=False)
model.eval()

# Uma task
name, task = dataset.tasks[0]
train_pairs = task["train"]
test_pairs = task["test"]
K = min(3, len(train_pairs))

xs_list = [dataset.pad_grid(train_pairs[k % len(train_pairs)]["input"]) for k in range(K)]
ys_list = [dataset.pad_grid(train_pairs[k % len(train_pairs)]["output"]) for k in range(K)]
xs = torch.stack(xs_list).unsqueeze(0).to(device)
ys = torch.stack(ys_list).unsqueeze(0).to(device)

q_pair = test_pairs[0]
xq = dataset.pad_grid(q_pair["input"]).unsqueeze(0).unsqueeze(0).to(device)
yq = dataset.pad_grid(q_pair["output"]).unsqueeze(0).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(xq, (xs, ys))
    pred = logits.argmax(dim=2).reshape(1, 1, 900)

# Monta hipoteses a partir dos logits
probs = torch.softmax(logits, dim=2)
hypotheses = []
for cls_idx in range(11):
    class_prob = float(probs[0, 0, 0, cls_idx].cpu())
    if class_prob < 0.05:
        continue
    h = Hypothesis(
        name=f"color_{cls_idx}",
        category="recoloring",
        description=f"grid uses color {cls_idx} (prob={class_prob:.2f})",
        evidence_for=[EvidenceItem(
            source=EvidenceSource.PERCEPTUAL,
            relation=EvidenceRelation.HAS_PROPERTY,
            confidence=class_prob,
            raw_score=class_prob,
            trace=f"Transformer logit class {cls_idx}",
        )],
        confidence=class_prob,
        evidence_total=class_prob,
    )
    hypotheses.append(h)

# Roda competicao instrumentada
result3 = comp.run(hypotheses)
print(f"\nARC Task {name}:")
print(f"  Winner: {result3.winner} (conf={result3.winner_confidence:.3f})")
print(f"  Margin: {result3.margin:.4f}")
print(f"  Entropy: {result3.entropy:.4f}")
print(f"  Diversity: {result3.diversity_score:.4f}")
print(f"  Clusters: {result3.num_clusters}")
print(f"  Reversals: {result3.winner_reversals}")

for h in result3.hypotheses[:3]:
    print(f"  Hypothesis: {h.name} (conf={h.confidence:.3f})")
    for cb in h.contribution_breakdown:
        print(f"    {cb.source.value}: score={cb.score:.3f} weight={cb.weight:.3f} items={cb.n_items}")
    print(f"    evidence_total={h.evidence_total:.3f}")

print("\nALL TESTS PASSED")
