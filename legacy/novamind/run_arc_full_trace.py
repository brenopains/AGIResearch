"""
ARC-AGI Full Benchmark com Rastreabilidade Completa
====================================================
Para cada task ARC, registra:
  1. Percepcao: codificacao HDV dos grids
  2. Hipotese: regra de transformacao inferida
  3. Evidencias: taxonomicas, episodicas, perceptuais, causais
  4. Competicao: ranking, margin, entropy, stability
  5. Decisao: output gerado vs expected
  6. Acerto/Erro com analise do gargalo

Uso:
    python run_arc_full_trace.py --tasks 10     # 10 tasks
    python run_arc_full_trace.py --tasks 400    # full ARC
    python run_arc_full_trace.py --tasks 10 --trace-only  # so trace, sem modelo
"""
import os, sys, json, time, argparse, math
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.grid_cognition import GridWorldModel
from core.arc_dataset import ARCDataset
from core.cognition import (
    CognitionPipeline, Competition, ConfidenceCalibrator,
    ConsistencyChecker, Translator, Timeline, MetaCognition,
    EvidenceItem, EvidenceSource, EvidenceRelation, Hypothesis, CompetitionResult,
)


COLOR_NAMES = [
    "black", "blue", "red", "green", "yellow",
    "grey", "fuchsia", "orange", "azure", "maroon",
]
ARC_EVAL_DIR = "data/ARC/data/evaluation"
ARC_TRAIN_DIR = "data/ARC/data/training"
MAX_GRID = 30
N_COLORS = 10


@dataclass
class TaskTrace:
    """Rastreabilidade completa de uma task ARC."""
    task_id: str = ""
    n_support: int = 0
    input_grids: List[List[List[int]]] = field(default_factory=list)
    output_grids: List[List[List[int]]] = field(default_factory=list)

    # Percepcao
    perception_time_ms: float = 0.0
    input_hdv_similarity: float = 0.0
    pattern_detected: str = ""

    # Hipotese
    hypothesis: str = ""
    hypothesis_confidence: float = 0.0
    hypothesis_entropy: float = 0.0
    hypothesis_margin: float = 0.0
    hypothesis_stability: List[float] = field(default_factory=list)
    hypotheses_ranked: List[dict] = field(default_factory=list)
    counterfactuals: List[dict] = field(default_factory=list)

    # Evidencias
    evidence_counts: Dict[str, int] = field(default_factory=dict)
    evidence_by_source: Dict[str, float] = field(default_factory=dict)

    # ─── Novos campos (Fase 2-5) ────────────────────────
    contribution_breakdown: List[dict] = field(default_factory=list)
    convergence_steps: List[dict] = field(default_factory=list)
    diversity_score: float = 0.0
    num_clusters: int = 1
    winner_reversals: int = 0
    convergence_velocity: float = 0.0
    meta_impasse: dict = field(default_factory=dict)
    meta_reasoning_style: dict = field(default_factory=dict)
    meta_knowledge_gaps: List[str] = field(default_factory=list)

    # Decisao
    generated_output: List[List[int]] = field(default_factory=list)
    expected_output: List[List[int]] = field(default_factory=list)
    pixel_accuracy: float = 0.0
    exact_match: bool = False
    error_pixels: int = 0
    total_pixels: int = 0

    # Tempo
    total_time_ms: float = 0.0

    # Gargalo identificado
    bottleneck: str = ""


def grid_to_str(grid: List[List[int]]) -> str:
    """Grid 2D -> string colorida."""
    if not grid or not grid[0]:
        return ""
    lines = []
    for row in grid:
        lines.append("".join(str(c) for c in row))
    return "\n".join(lines)


def compare_grids(pred: np.ndarray, target: np.ndarray) -> Tuple[float, int, int]:
    """Compara grids: pixel accuracy, numero de erros, total."""
    if pred.shape != target.shape:
        return 0.0, int(pred.size), int(pred.size)
    total = int(pred.size)
    wrong = int((pred != target).sum())
    acc = 1.0 - wrong / max(1, total)
    return acc, wrong, total


def extract_pattern_name(input_grids: List[np.ndarray], output_grids: List[np.ndarray]) -> str:
    """Tenta nomear o padrao ARC baseado nas transformacoes."""
    if len(input_grids) < 2:
        return "unknown"

    # Detecta tipo basico de transformacao
    h_in, w_in = input_grids[0].shape
    h_out, w_out = output_grids[0].shape

    if h_out > h_in or w_out > w_in:
        return "expansion"
    if h_out < h_in or w_out < w_in:
        return "compression"
    if h_out == h_in and w_out == w_in:
        # Verifica se e color change
        colors_in = set(input_grids[0].flatten().tolist())
        colors_out = set(output_grids[0].flatten().tolist())
        if colors_in != colors_out:
            return "recoloring"
        return "transformation"

    return "unknown"


def classify_bottleneck(trace: TaskTrace) -> str:
    """Classifica o gargalo principal baseado no trace (agora com instrumentacao)."""
    if trace.perception_time_ms > 5000:
        return "PERCEPTION_SLOW"
    if trace.hypothesis_confidence < 0.3 and trace.hypothesis_entropy > 0.5:
        return "HIGH_UNCERTAINTY"
    if trace.hypothesis_margin < 0.05:
        return "TIGHT_MARGIN"
    if trace.exact_match:
        return "NONE"
    if trace.pixel_accuracy > 0.8:
        return "NEAR_MISS"
    if trace.pixel_accuracy > 0.5:
        return "PARTIAL_PATTERN"
    if trace.hypothesis == "unknown" or not trace.hypothesis:
        return "NO_HYPOTHESIS"
    if trace.evidence_counts.get("taxonomy", 0) == 0 and trace.evidence_counts.get("causal", 0) == 0:
        return "LACKS_EVIDENCE"
    if trace.num_clusters <= 1 and trace.diversity_score < 0.05:
        return "NO_DIVERSITY"
    if trace.meta_impasse.get("is_impasse"):
        return "IMPASSE"
    return "UNKNOWN"


def run_arc_with_trace(
    model: GridWorldModel,
    dataset: ARCDataset,
    n_tasks: int,
    device: torch.device,
    smoke: bool = False,
) -> Tuple[List[TaskTrace], Dict]:
    """Avalia N tasks ARC-AGI com rastreabilidade completa."""
    traces: List[TaskTrace] = []
    summary = {
        "tasks": 0,
        "exact_match": 0,
        "avg_pixel_accuracy": 0.0,
        "total_time_s": 0.0,
        "bottleneck_counts": defaultdict(int),
    }

    if model is not None:
        model.eval()
    B = 1  # Uma task por vez para rastreabilidade individual

    for task_idx in range(min(n_tasks, len(dataset.tasks))):
        trace = TaskTrace()
        t_start = time.time()

        # Itera direto nas tasks do dataset (sem random)
        name, task = dataset.tasks[task_idx % len(dataset.tasks)]
        train_pairs = task['train']
        test_pairs = task['test']
        K = min(3, len(train_pairs))

        # Suporte: K pares input/output
        xs_list, ys_list = [], []
        for k in range(K):
            pair = train_pairs[k % len(train_pairs)]
            xs_list.append(dataset.pad_grid(pair['input']))
            ys_list.append(dataset.pad_grid(pair['output']))
        xs = torch.stack(xs_list).unsqueeze(0).to(device)  # [1, K, 900]
        ys = torch.stack(ys_list).unsqueeze(0).to(device)

        # Query: primeiro teste
        if test_pairs:
            q_pair = test_pairs[0]
            xq = dataset.pad_grid(q_pair['input']).unsqueeze(0).unsqueeze(0).to(device)
            yq = dataset.pad_grid(q_pair['output']).unsqueeze(0).unsqueeze(0).to(device)
        else:
            continue

        # Converte para numpy
        xq_np = xq[0, 0].cpu().numpy().reshape(MAX_GRID, MAX_GRID)
        yq_np = yq[0, 0].cpu().numpy().reshape(MAX_GRID, MAX_GRID)

        # Suporte: grids de entrada
        input_grids = []
        output_grids = []
        for k in range(xs.shape[1]):
            inp = xs[0, k].cpu().numpy().reshape(MAX_GRID, MAX_GRID)
            out = ys[0, k].cpu().numpy().reshape(MAX_GRID, MAX_GRID)
            input_grids.append(inp)
            output_grids.append(out)

        trace.task_id = f"{name.replace('.json','')}"
        trace.n_support = len(input_grids)
        trace.input_grids = [g.tolist() for g in input_grids]
        trace.output_grids = [g.tolist() for g in output_grids]
        trace.expected_output = yq_np.tolist()

        # --- PERCEPCAO ---
        t_percept = time.time()
        pattern = extract_pattern_name(input_grids, output_grids)
        trace.pattern_detected = pattern
        trace.perception_time_ms = (time.time() - t_percept) * 1000

        # Similaridade entre grids de entrada (mede consistencia do padrao)
        sims = []
        for i in range(len(input_grids)):
            for j in range(i + 1, len(input_grids)):
                if input_grids[i].shape == input_grids[j].shape:
                    sim = 1.0 - np.abs(input_grids[i] - input_grids[j]).mean() / 9.0
                    sims.append(sim)
        trace.input_hdv_similarity = float(np.mean(sims)) if sims else 0.0

        # --- HIPOTESE (Modelo Transformer ou dummy) ---
        t_hyp = time.time()
        if model is not None:
            with torch.no_grad():
                logits = model(xq, (xs, ys))
                pred = logits.argmax(dim=2).reshape(1, 1, MAX_GRID * MAX_GRID)
                pred_np = pred[0, 0].cpu().numpy().reshape(MAX_GRID, MAX_GRID)

            # Probabilidades (softmax) para medir confianca
            probs = torch.softmax(logits, dim=2)
            max_probs = probs.max(dim=2).values
            avg_confidence = float(max_probs.mean().cpu())
        else:
            # Modo dummy: copia o ultimo grid de suporte como predicao
            pred_np = output_grids[-1].copy() if output_grids else np.zeros((MAX_GRID, MAX_GRID), dtype=int)
            avg_confidence = 0.5

        # Monta hipoteses para o CognitionPipeline
        hypotheses_list = []
        for cls_idx in range(N_COLORS):
            class_prob = float(probs[0, 0, 0, cls_idx].cpu()) if probs.shape[-1] > cls_idx else 0.0
            if class_prob < 0.05:
                continue
            h = Hypothesis(
                name=f"color_{cls_idx}_{COLOR_NAMES[cls_idx]}",
                category="recoloring",
                description=f"grid uses color {COLOR_NAMES[cls_idx]} (prob={class_prob:.2f})",
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
            hypotheses_list.append(h)

        # Competicao das hipoteses (INSTRUMENTADA)
        competition = Competition(iterations=3)
        comp_result = competition.run(hypotheses_list)

        trace.hypothesis = comp_result.winner or "unknown"
        trace.hypothesis_confidence = comp_result.winner_confidence if comp_result.winner else avg_confidence
        trace.hypothesis_entropy = comp_result.entropy
        trace.hypothesis_margin = comp_result.margin
        trace.hypothesis_stability = comp_result.stability
        trace.hypotheses_ranked = [h.to_dict() for h in comp_result.hypotheses[:5]]
        trace.counterfactuals = comp_result.counterfactuals

        # Evidencias
        trace.evidence_counts = defaultdict(int)
        trace.evidence_by_source = defaultdict(float)
        for h in comp_result.hypotheses:
            for ev in h.evidence_for:
                trace.evidence_counts[ev.source.value] += 1
                trace.evidence_by_source[ev.source.value] += ev.confidence

        # ─── Novos campos de instrumentacao ──────────
        # Contribution breakdown da vencedora
        if comp_result.hypotheses:
            trace.contribution_breakdown = [
                cb.to_dict() for cb in comp_result.hypotheses[0].contribution_breakdown
            ]
            # Breakdown do runner-up tambem
            if len(comp_result.hypotheses) > 1:
                runner_cb = [
                    cb.to_dict() for cb in comp_result.hypotheses[1].contribution_breakdown
                ]
                if runner_cb:
                    trace.contribution_breakdown.append({"runner_up": runner_cb})

        # Convergence steps
        trace.convergence_steps = [cs.to_dict() for cs in comp_result.convergence_steps]

        # Metrica de diversidade
        trace.diversity_score = comp_result.diversity_score
        trace.num_clusters = comp_result.num_clusters
        trace.winner_reversals = comp_result.winner_reversals
        trace.convergence_velocity = comp_result.convergence_velocity

        # Meta-cognicao
        meta = MetaCognition()
        meta_obs = meta.observe(name, comp_result, step=0)
        trace.meta_impasse = meta_obs.get("impasse", {})
        trace.meta_reasoning_style = meta_obs.get("reasoning_style", {})
        trace.meta_knowledge_gaps = meta_obs.get("knowledge_gaps", [])

        # --- DECISAO ---
        trace.generated_output = pred_np.tolist()
        acc, wrong, total = compare_grids(pred_np, yq_np)
        trace.pixel_accuracy = acc
        trace.exact_match = wrong == 0
        trace.error_pixels = wrong
        trace.total_pixels = total

        # --- GARGALO ---
        trace.total_time_ms = (time.time() - t_start) * 1000
        trace.bottleneck = classify_bottleneck(trace)

        # --- ATUALIZA SUMMARY ---
        summary["tasks"] += 1
        if trace.exact_match:
            summary["exact_match"] += 1
        summary["avg_pixel_accuracy"] = (
            (summary["avg_pixel_accuracy"] * (summary["tasks"] - 1) + trace.pixel_accuracy) / summary["tasks"]
        )
        summary["total_time_s"] += trace.total_time_ms / 1000
        summary["bottleneck_counts"][trace.bottleneck] += 1

        traces.append(trace)

        if (task_idx + 1) % 10 == 0:
            em = summary["exact_match"]
            pa = summary["avg_pixel_accuracy"]
            print(f"  [{task_idx+1}/{n_tasks}] EM={em} Acc={pa:.2%} "
                  f"Btlnk={dict(summary['bottleneck_counts'])}", flush=True)

    return traces, dict(summary)


def print_trace_report(traces: List[TaskTrace], summary: Dict):
    """Imprime relatorio detalhado de rastreabilidade."""
    print("\n" + "=" * 70)
    print("RELATORIO DE RASTREABILIDADE ARC-AGI")
    print("=" * 70)

    total = summary["tasks"]
    exact = summary["exact_match"]
    print(f"\nTasks: {total}  |  Exact Match: {exact}/{total} ({exact/max(1,total):.1%})")
    print(f"Pixel Accuracy Media: {summary['avg_pixel_accuracy']:.2%}")
    print(f"Tempo Total: {summary['total_time_s']:.1f}s")
    print(f"Media por Task: {summary['total_time_s']/max(1,total)*1000:.0f}ms")

    # Metricas agregadas de instrumentacao
    avg_div = sum(t.diversity_score for t in traces) / max(1, total)
    avg_clust = sum(t.num_clusters for t in traces) / max(1, total)
    avg_rev = sum(t.winner_reversals for t in traces) / max(1, total)
    avg_vel = sum(t.convergence_velocity for t in traces) / max(1, total)
    impasses = sum(1 for t in traces if t.meta_impasse.get("is_impasse"))
    print(f"\nInstrumentacao:")
    print(f"  Diversidade media: {avg_div:.3f} | Clusters medio: {avg_clust:.1f}")
    print(f"  Reversoes media: {avg_rev:.2f} | Vel. Convergencia: {avg_vel:.4f}")
    print(f"  Impasses: {impasses}/{total} ({impasses/max(1,total)*100:.0f}%)")

    print(f"\n--- Distribuicao de Gargalos ---")
    for bt, count in sorted(summary["bottleneck_counts"].items(), key=lambda x: -x[1]):
        pct = count / max(1, total) * 100
        print(f"  {bt:25s}: {count:3d} ({pct:.0f}%)")

    # Amostra de traces com falha
    print(f"\n--- Top 5 Erros (Menor Pixel Accuracy) ---")
    sorted_by_error = sorted(traces, key=lambda t: t.pixel_accuracy)[:5]
    for t in sorted_by_error:
        print(f"\n  Task: {t.task_id}")
        print(f"  Padrao: {t.pattern_detected}")
        print(f"  Hipotese: {t.hypothesis} (conf={t.hypothesis_confidence:.2f})")
        print(f"  Pixel Acc: {t.pixel_accuracy:.2%} | EM={t.exact_match}")
        print(f"  Entropia: {t.hypothesis_entropy:.3f} | Margem: {t.hypothesis_margin:.3f}")
        print(f"  Diversidade: {t.diversity_score:.3f} | Clusters: {t.num_clusters}")
        print(f"  Reversoes: {t.winner_reversals} | Vel. Conv.: {t.convergence_velocity:.3f}")
        print(f"  Evidencias: {dict(t.evidence_counts)}")
        print(f"  Contribution: {t.contribution_breakdown[:2]}")
        print(f"  Estilo: {t.meta_reasoning_style}")
        print(f"  Impasse: {t.meta_impasse}")
        print(f"  Gargalo: {t.bottleneck}")
        print(f"  Tempo: {t.total_time_ms:.0f}ms")

    # Acertos
    exact_traces = [t for t in traces if t.exact_match]
    if exact_traces:
        print(f"\n--- Acertos ({len(exact_traces)}) ---")
        for t in exact_traces[:3]:
            print(f"  Task: {t.task_id} | Conf={t.hypothesis_confidence:.2f} | "
                  f"Margem={t.hypothesis_margin:.3f} | Estabilidade={t.hypothesis_stability}")


def save_trace_json(traces: List[TaskTrace], summary: Dict, path: str):
    """Salva rastreabilidade em JSON."""
    data = {
        "summary": summary,
        "traces": [],
    }
    for t in traces:
        d = {
            "task_id": t.task_id,
            "pattern_detected": t.pattern_detected,
            "hypothesis": t.hypothesis,
            "hypothesis_confidence": round(t.hypothesis_confidence, 4),
            "hypothesis_entropy": round(t.hypothesis_entropy, 4),
            "hypothesis_margin": round(t.hypothesis_margin, 4),
            "hypothesis_stability": [round(s, 4) for s in t.hypothesis_stability],
            "pixel_accuracy": round(t.pixel_accuracy, 4),
            "exact_match": t.exact_match,
            "error_pixels": t.error_pixels,
            "total_pixels": t.total_pixels,
            "bottleneck": t.bottleneck,
            "total_time_ms": round(t.total_time_ms, 2),
            "perception_time_ms": round(t.perception_time_ms, 2),
            "input_hdv_similarity": round(t.input_hdv_similarity, 4),
            "evidence_counts": dict(t.evidence_counts),
            "evidence_by_source": {k: round(v, 4) for k, v in t.evidence_by_source.items()},
            "counterfactuals": t.counterfactuals,
            "hypotheses_ranked": t.hypotheses_ranked[:3],
            # Novos campos
            "contribution_breakdown": t.contribution_breakdown[:4],
            "convergence_steps": t.convergence_steps,
            "diversity_score": round(t.diversity_score, 4),
            "num_clusters": t.num_clusters,
            "winner_reversals": t.winner_reversals,
            "convergence_velocity": round(t.convergence_velocity, 4),
            "meta_impasse": t.meta_impasse,
            "meta_reasoning_style": t.meta_reasoning_style,
            "meta_knowledge_gaps": t.meta_knowledge_gaps,
        }
        data["traces"].append(d)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=lambda o: float(o) if hasattr(o, '__float__') else str(o))
    print(f"\nTrace salvo em: {path}")


def main():
    parser = argparse.ArgumentParser(description="ARC-AGI Full Trace Benchmark")
    parser.add_argument("--tasks", type=int, default=10, help="Numero de tasks (max 400)")
    parser.add_argument("--smoke", action="store_true", help="50 steps de treino")
    parser.add_argument("--trace-only", action="store_true", help="So trace, sem modelo (dummy)")
    parser.add_argument("--output", default="arc_full_trace_results.json", help="Arquivo de saida")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.trace_only:
        print("\n[MODO TRACE-ONLY] Usando dummy model (sem treino)")
        n_tasks = min(args.tasks, 400)
        dataset = ARCDataset(data_dir=ARC_EVAL_DIR, max_h=MAX_GRID, max_w=MAX_GRID, pad_val=10, dev=device)
        traces, summary = run_arc_with_trace(None, dataset, n_tasks, device, smoke=args.smoke)
    else:
        # Carrega modelo pre-treinado ou treina do zero
        train_steps = 50 if args.smoke else 300
        n_tasks = min(args.tasks, 400)

        print(f"\n[INIT] Carregando dataset...")
        dataset = ARCDataset(data_dir=ARC_TRAIN_DIR, max_h=MAX_GRID, max_w=MAX_GRID, pad_val=10, dev=device)
        eval_dataset = ARCDataset(data_dir=ARC_EVAL_DIR, max_h=MAX_GRID, max_w=MAX_GRID, pad_val=10, dev=device)

        model = GridWorldModel(d=128, heads=4, layers=3, max_hw=900, colors=11).to(device)
        print(f"  Parametros: {sum(p.numel() for p in model.parameters()):,}")

        # Tenta carregar checkpoint
        model_path = "arc_model_trained.pt"
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=device)
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            elif isinstance(ckpt, dict):
                model.load_state_dict(ckpt, strict=False)
            print(f"  Checkpoint carregado: {model_path}")
        else:
            print(f"  Nenhum checkpoint encontrado, treinando do zero...")
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
            for step in range(1, train_steps + 1):
                (xs, ys), (xq, yq) = dataset.sample_batch(16, 3, 1)
                logits = model(xq, (xs, ys))
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(16, 11, 900), yq.reshape(16, 900), ignore_index=10
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if step % 50 == 0:
                    print(f"    Step {step}/{train_steps} Loss={loss.item():.4f}", flush=True)
            torch.save({"model_state_dict": model.state_dict()}, model_path)
            print(f"  Modelo salvo em {model_path}")

        print(f"\n[AVALIACAO] {n_tasks} tasks com rastreabilidade...")
        traces, summary = run_arc_with_trace(model, eval_dataset, n_tasks, device, smoke=args.smoke)

    print_trace_report(traces, summary)
    save_trace_json(traces, summary, args.output)


if __name__ == "__main__":
    main()
