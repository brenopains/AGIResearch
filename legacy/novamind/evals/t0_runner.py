"""
T0 Runner — Baseline do Sistema Atual (Antes de F2-F4)
======================================================
Executa T1 (Caixa de Pandora), T3 (SyntLog), T4 (Causal) e coleta T2 (ARC).
Nenhuma intervenção no sistema — apenas carga de estado + query.

Uso:
  cd novamind
  python evals/t0_runner.py
"""
import sys, os, json, time, random, pickle, math, re
from pathlib import Path
from datetime import datetime

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)
os.chdir(_BASE)

import torch
import numpy as np

# ---------------------------------------------------------------------------
# 1. Load saved system state
# ---------------------------------------------------------------------------
print("=" * 70)
print("T0 RUNNER — Baseline do Sistema NovaMind")
print("=" * 70)

SAVE_DIR = Path(_BASE) / "save"
STATE_PATH = SAVE_DIR / "agi_state_live.pkl"
BRAIN_PATH = SAVE_DIR / "brain_live.pth"
POOL_DIR = Path(_BASE) / "evals" / "pool"

def load_saved_state():
    print("\n[LOAD] Loading agi_state_live.pkl...")
    state = pickle.load(open(STATE_PATH, "rb"))
    print(f"[LOAD]  cge_nodes: {len(state.get('cge_nodes', {}))}")
    print(f"[LOAD]  cge_edges: {len(state.get('cge_edges', []))}")
    if 'self_model' in state:
        print(f"[LOAD]  self_model: present")
    print(f"[LOAD]  timestamp: {state.get('timestamp', 'N/A')}")
    return state

def load_cge_from_state(state):
    import torch
    from core.cognitive_graph_engine import CognitiveGraphEngine
    from core.bge_encoder import BGEEncoder
    
    bge = BGEEncoder()
    cge = CognitiveGraphEngine(bge_encoder=bge)
    
    # Restore nodes — move vectors to correct device
    cge_nodes = state.get("cge_nodes", {})
    for node_id, node_data in cge_nodes.items():
        if not isinstance(node_id, str) or len(node_id) > 50:
            continue
        try:
            if isinstance(node_data, dict):
                # Deep copy to avoid modifying original state
                nd = dict(node_data)
                if "vector" in nd and isinstance(nd["vector"], torch.Tensor):
                    nd["vector"] = nd["vector"].to(cge.device)
                if "semantic" in nd and isinstance(nd["semantic"], torch.Tensor):
                    nd["semantic"] = nd["semantic"].to(cge.device)
                if "perceptual" in nd and isinstance(nd["perceptual"], torch.Tensor):
                    nd["perceptual"] = nd["perceptual"].to(cge.device)
                cge.nodes[node_id] = nd
        except Exception as e:
            pass
    
    # Restore edges — handle various storage formats
    cge_edges = state.get("cge_edges", [])
    edge_type_map = {
        "causes": cge.EDGE_CAUSES,
        "caused_by": cge.EDGE_CAUSED_BY,
        "right_of": cge.EDGE_RIGHT_OF,
        "left_of": cge.EDGE_LEFT_OF,
        "above": cge.EDGE_ABOVE,
        "below": cge.EDGE_BELOW,
        "part_of": cge.EDGE_PART_OF,
        "before": cge.EDGE_BEFORE,
        "after": cge.EDGE_AFTER,
    }
    for edge in cge_edges:
        try:
            if isinstance(edge, list) and len(edge) >= 3:
                frm, to = str(edge[0]), str(edge[1])
                etype = edge[2]
                if isinstance(etype, torch.Tensor):
                    etype = etype.to(cge.device)
                elif isinstance(etype, str):
                    etype = edge_type_map.get(etype.lower(), cge.EDGE_CAUSES)
                else:
                    etype = cge.EDGE_CAUSES
                cge.add_edge(frm, to, etype)
            elif isinstance(edge, dict):
                frm = str(edge.get("from", "unknown"))
                to = str(edge.get("to", "unknown"))
                etype = edge.get("type", edge.get("edge_type", "causes"))
                if isinstance(etype, torch.Tensor):
                    etype = etype.to(cge.device)
                elif isinstance(etype, str):
                    etype = edge_type_map.get(etype.lower(), cge.EDGE_CAUSES)
                else:
                    etype = cge.EDGE_CAUSES
                cge.add_edge(frm, to, etype)
        except Exception as e:
            pass
    
    print(f"[LOAD] CGE restored: {len(cge.nodes)} nodes, {len(cge.edges)} edges")
    return cge

# ---------------------------------------------------------------------------
# 2. Test T3 — SyntLog (Syllogisms)
# ---------------------------------------------------------------------------
def run_t3_syntlog(cge, bge, syllogism_path):
    print("\n" + "=" * 70)
    print("T3 — SYNTLOG (Silogismos)")
    print("=" * 70)
    
    with open(syllogism_path) as f:
        pool = json.load(f)
    
    print(f"[T3] Pool: {len(pool)} syllogisms ({len(set(p['category'] for p in pool))} categories)")
    
    results = []
    for item in pool[:200]:  # subset de 200 para T0 (time limit)
        category = item["category"]
        premises = item["premises"]
        correct = item["correct_conclusion"]
        distractors = item["distractors"]
        
        # Build query text from premises
        query = " ".join(premises)
        
        # Method 1: BGE encode query, cosine similarity to possible answers
        all_options = [correct] + distractors
        try:
            query_vec = bge.encode_bipolar([query])[0]
            option_vecs = bge.encode_bipolar(all_options)
            sims = torch.cosine_similarity(
                query_vec.unsqueeze(0).expand(len(all_options), -1),
                option_vecs
            )
            chosen_idx = sims.argmax().item()
            chosen = all_options[chosen_idx]
            is_correct = (chosen == correct)
            
            # Also compute margin
            sorted_sims = sims.sort(descending=True)
            margin = (sorted_sims.values[0] - sorted_sims.values[1]).item()
        except Exception as e:
            chosen = "ERROR"
            is_correct = False
            margin = 0.0
        
        results.append({
            "id": item["id"],
            "category": category,
            "premises": premises,
            "correct": correct,
            "chosen": chosen,
            "is_correct": is_correct,
            "margin": margin,
        })
    
    # Aggregate
    from collections import defaultdict
    cat_stats = defaultdict(list)
    for r in results:
        cat_stats[r["category"]].append(r["is_correct"])
    
    print(f"\n[T3] Results (n={len(results)}):")
    overall_correct = sum(1 for r in results if r["is_correct"])
    print(f"  Overall accuracy: {overall_correct}/{len(results)} = {overall_correct/len(results)*100:.1f}%")
    print(f"  Chance baseline (MP/MT): 50%, (Barbara): 12.5%, (RelChain): 50%")
    
    for cat, outcomes in sorted(cat_stats.items()):
        n = len(outcomes)
        acc = sum(outcomes) / n * 100
        chance = {"modus_ponens": 50, "modus_tollens": 50, "barbara": 12.5, "relational_chain": 50}
        ch = chance.get(cat, 50)
        print(f"  {cat:20s}: {acc:.1f}% ({sum(outcomes)}/{n})  [chance={ch}%]")
    
    return results

# ---------------------------------------------------------------------------
# 3. Test T1 — Caixa de Pandora (Composition)
# ---------------------------------------------------------------------------
def run_t1_pandora(cge, bge, pandora_path, n_triples=20):
    print("\n" + "=" * 70)
    print("T1 — CAIXA DE PANDORA (Composição)")
    print("=" * 70)
    
    with open(pandora_path) as f:
        pool = json.load(f)
    
    print(f"[T1] Pool: {len(pool)} triples")
    
    # Use a subset for T0 (time/GPU/energy constraint)
    device = cge.device
    results = []
    for item in pool[:n_triples]:
        A, B, C = item["A"], item["B"], item["C"]
        
        try:
            vec_A = bge.encode_bipolar([A])[0].to(device)
            vec_B = bge.encode_bipolar([B])[0].to(device)
            vec_C = bge.encode_bipolar([C])[0].to(device)
            
            # Bind: A ⊗ B ⊗ C
            invention = torch.sign(vec_A * vec_B * vec_C)
            
            # Compute novelty: similarity to each component
            sim_A = torch.cosine_similarity(invention.unsqueeze(0), vec_A.unsqueeze(0)).item()
            sim_B = torch.cosine_similarity(invention.unsqueeze(0), vec_B.unsqueeze(0)).item()
            sim_C = torch.cosine_similarity(invention.unsqueeze(0), vec_C.unsqueeze(0)).item()
            
            # Find nearest neighbor in concept space to understand "decode"
            # Compare invention to all concepts in CGE
            best_match = "unknown"
            best_sim = -1.0
            for nid, ndata in list(cge.nodes.items())[:500]:
                if 'vector' in ndata:
                    nvec = ndata['vector']
                    if isinstance(nvec, torch.Tensor):
                        sim = torch.cosine_similarity(
                            invention.unsqueeze(0),
                            nvec.flatten().unsqueeze(0)
                        ).item()
                        if sim > best_sim:
                            best_sim = sim
                            best_match = nid
            
            results.append({
                "id": item["id"],
                "A": A, "B": B, "C": C,
                "sim_A": sim_A,
                "sim_B": sim_B,
                "sim_C": sim_C,
                "novelty_ok": (sim_A < 0.3 and sim_B < 0.3 and sim_C < 0.3),
                "best_decode_match": best_match,
                "best_decode_sim": best_sim,
            })
        except Exception as e:
            print(f"[T1] Error on {item['id']}: {e}")
            results.append({"id": item["id"], "A": A, "B": B, "C": C, "error": str(e)})
    
    novelty_ok_count = sum(1 for r in results if r.get("novelty_ok") and not r.get("error"))
    print(f"\n[T1] Results (n={len(results)}):")
    print(f"  Novelty OK (all sim < 0.3): {novelty_ok_count}/{len(results)}")
    
    if results and "sim_A" in results[0]:
        avg_sims = (
            np.mean([r["sim_A"] for r in results if "sim_A" in r]),
            np.mean([r["sim_B"] for r in results if "sim_B" in r]),
            np.mean([r["sim_C"] for r in results if "sim_C" in r]),
        )
        print(f"  Avg sim to A: {avg_sims[0]:.4f}")
        print(f"  Avg sim to B: {avg_sims[1]:.4f}")
        print(f"  Avg sim to C: {avg_sims[2]:.4f}")
        print(f"  Best decode match (CGE): {results[0].get('best_decode_match', 'N/A')} sim={results[0].get('best_decode_sim', 0):.4f}")
    
    return results

# ---------------------------------------------------------------------------
# 4. Test T4 — Causal (Tübingen-style)
# ---------------------------------------------------------------------------
def run_t4_causal(cge, bge, n_pairs=50):
    print("\n" + "=" * 70)
    print("T4 — CAUSAL (Tübingen-style)")
    print("=" * 70)
    
    # Generate synthetic causal pairs (since we don't have Tübingen downloaded)
    # Generate pairs with known causal direction
    random.seed(20260712)
    pairs = []
    for i in range(n_pairs):
        cause = random.choice(SUBJECTS)
        effect = random.choice([s for s in SUBJECTS if s != cause])
        pairs.append({"cause": cause, "effect": effect, "id": f"CAUSAL-{i:03d}"})
    
    print(f"[T4] Generated {len(pairs)} causal pairs")
    
    device = cge.device
    results = []
    for pair in pairs:
        cause = pair["cause"]
        effect = pair["effect"]
        
        try:
            vec_cause = bge.encode_bipolar([cause])[0].to(device)
            vec_effect = bge.encode_bipolar([effect])[0].to(device)
            
            # Create nodes in CGE
            if cause not in cge.nodes:
                cge.create_node(cause, vec_cause)
            if effect not in cge.nodes:
                cge.create_node(effect, vec_effect)
            
            # Add ONE causal edge: cause -> effect
            cge.add_edge(cause, effect, cge.EDGE_CAUSES)
            
            # Causal intervention: remove cause, measure if effect is predicted to change
            intervention_preds = cge.causal_intervention(cause, remove=True)
            effect_in_preds = effect in intervention_preds
            pred_change = intervention_preds.get(effect, 0)
            
            # Test causal model: if we intervene on cause, effect must change
            # test_causal_model(intervention_dict, observation_dict) -> error
            intervention_dict = {cause: -1.0}  # cause is removed
            observation_dict = {effect: -1.0}   # effect should also go away
            model_error = cge.test_causal_model(intervention_dict, observation_dict)
            model_accurate = (model_error < 0.5)  # 0 = perfect
            
            # Baseline: random guess
            random_guess = random.choice([True, False])
            
            results.append({
                "id": pair["id"],
                "pair": f"{cause} -> {effect}",
                "cause": cause,
                "effect": effect,
                "intervention_detected_effect": effect_in_preds,
                "intervention_pred_change": pred_change,
                "model_error": model_error,
                "model_accurate": model_accurate,
                "random_guess_correct": random_guess,
            })
        except Exception as e:
            print(f"[T4] Error on {pair['id']}: {e}")
            results.append({
                "id": pair["id"],
                "pair": f"{cause} -> {effect}",
                "error": str(e),
                "intervention_detected_effect": False,
                "model_accurate": False,
                "random_guess_correct": random.choice([True, False]),
            })
    
    causal_ok = sum(1 for r in results if r.get("intervention_detected_effect"))
    model_ok = sum(1 for r in results if r.get("model_accurate"))
    random_ok = sum(1 for r in results if r.get("random_guess_correct"))
    
    print(f"\n[T4] Results (n={len(results)}):")
    print(f"  Intervention detects effect: {causal_ok}/{len(results)} = {causal_ok/len(results)*100:.1f}%")
    print(f"  Causal model accurate:       {model_ok}/{len(results)} = {model_ok/len(results)*100:.1f}%")
    print(f"  Random baseline:             {random_ok}/{len(results)} = {random_ok/len(results)*100:.1f}% (expected ~50%)")
    
    return results

SUBJECTS = [
    "grom", "fex", "blarg", "snorp", "quib", "zork", "mlep", "tronk",
    "flarg", "snerp", "gleep", "bork", "zilp", "quarg", "freb", "slin",
    "dronk", "plerg", "snirk", "torb", "glurp", "frend", "blix", "treek",
    "sprog", "murk", "grenk", "florp", "snert", "trill", "gorb", "pleen",
]

# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    start_time = time.time()
    
    # Load state
    state = load_saved_state()
    
    # Initialize core components
    from core.bge_encoder import BGEEncoder
    bge = BGEEncoder()
    cge = load_cge_from_state(state)
    
    # Initialize goal system for error-driven learning tests
    from core.goal_system import GoalManager
    gm = GoalManager()
    
    # Run T3 — SyntLog
    syllogism_path = POOL_DIR / "syllogism_pool.json"
    t3_results = run_t3_syntlog(cge, bge, syllogism_path)
    
    # Run T1 — Caixa de Pandora
    pandora_path = POOL_DIR / "pandora_pool.json"
    t1_results = run_t1_pandora(cge, bge, pandora_path, n_triples=50)
    
    # Run T4 — Causal
    t4_results = run_t4_causal(cge, bge, n_pairs=50)
    
    # Collect T2 — ARC baseline from existing results
    print("\n" + "=" * 70)
    print("T2 — ARC-AGI (from existing results)")
    print("=" * 70)
    try:
        arc_results = json.load(open(Path(_BASE) / "_benchmark.py"))
        print("[T2] _benchmark.py loaded")
    except:
        arc_results = {}
    try:
        arc_json = json.load(open(Path(_BASE) / "arc_integrated_results.json"))
        print(f"[T2] ARC results: pixel_acc={arc_json.get('pixel_accuracy', 'N/A')}, exact_match={arc_json.get('exact_match', 'N/A')}")
    except:
        arc_json = {}
        print("[T2] No ARC results file found")
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("T0 — SUMMARY")
    print("=" * 70)
    
    t3_acc = sum(1 for r in t3_results if r["is_correct"]) / len(t3_results) * 100 if t3_results else 0
    t1_novelty = sum(1 for r in t1_results if r.get("novelty_ok")) / len(t1_results) * 100 if t1_results else 0
    t4_intervention = sum(1 for r in t4_results if r.get("intervention_detected_effect")) / len(t4_results) * 100 if t4_results else 0
    t4_model = sum(1 for r in t4_results if r.get("model_accurate")) / len(t4_results) * 100 if t4_results else 0
    
    print(f"  T1 (Pandora - novelty): {t1_novelty:.1f}%") 
    print(f"  T2 (ARC - existing data): pixel_acc={arc_json.get('pixel_accuracy', 'N/A')}")
    print(f"  T3 (SyntLog): {t3_acc:.1f}%")
    print(f"  T4 (Intervention): {t4_intervention:.1f}%")
    print(f"  T4 (Causal Model): {t4_model:.1f}%")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    
    # Save results
    report = {
        "timestamp": datetime.now().isoformat(),
        "git_sha": "aacd9cf+modified",
        "t0_version": "2026-07-12T00:41",
        "tests": {
            "T1_pandora": {
                "n": len(t1_results),
                "novelty_rate": t1_novelty,
                "details": t1_results[:10],
            },
            "T3_syntlog": {
                "n": len(t3_results),
                "overall_accuracy": t3_acc,
                "details": t3_results[:20],
            },
            "T4_causal": {
                "n": len(t4_results),
                "intervention_detects_effect": t4_intervention,
                "causal_model_accuracy": t4_model,
                "details": t4_results[:10],
            },
        },
        "arc_existing": {
            "pixel_accuracy": arc_json.get("pixel_accuracy"),
            "exact_match": arc_json.get("exact_match"),
        },
        "elapsed_seconds": elapsed,
    }
    
    report_path = Path(_BASE) / "evals" / "results" / "t0_report.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[T0] Report saved to {report_path}")
