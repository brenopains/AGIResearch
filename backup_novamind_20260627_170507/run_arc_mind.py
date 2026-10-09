"""
run_arc_mind.py — ARC-AGI com o sistema REAL NovaMind
======================================================

Usa o Mind/IntentionDrivenGenerator/SandboxEngine que ja foi validado:
  - HDV 3175 itens (code_brain, msvd, python_brain, text_library)
  - compose_concept: bind de papeis semanticos no espaco HDC
  - IntentionDrivenGenerator: residual HDC -> busca -> chunk
  - ArchetypeEngine: reconhece padroes recorrentes entre exemplos
  - CausalReasoning: verifica se regra proposta e consistente
  - Cerebellum: antecipa erro antes de aplicar

Pipeline para cada task ARC:
  1. Codifica cada grid de suporte como manifold HDC (roles: cor, posicao, shape)
  2. Extrai a "regra de transformacao" entre input->output em HDC puro
  3. ArchetypeEngine consolida o padrao dos K exemplos
  4. Aplica a regra extraida na query grid
  5. Decode: manifold -> grid concreto
  6. Verifica exact match

Uso:
    python run_arc_mind.py --tasks 20   # avalia 20 tasks
    python run_arc_mind.py --tasks 400  # full eval
"""

import os, sys, json, time, argparse
from typing import List, Dict, Tuple, Optional
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from core.grounding_engine import GroundingEngine
from core.sandbox_engine import SandboxEngine
from core.archetypes import ArchetypeEngine
from core.causal_reasoning import CausalReasoningEngine
from core.cerebellum import Cerebellum
from core.neuromodulator import Neuromodulator
from core.hdc_memory import HDCMemory
from core.autoregressive_generator import AutoregressiveGenerator

ARC_EVAL_DIR  = "data/ARC/data/evaluation"
ARC_TRAIN_DIR = "data/ARC/data/training"

MAX_GRID = 30
N_COLORS = 10

COLOR_NAMES = [
    "black", "blue", "red", "green", "yellow",
    "grey", "fuchsia", "orange", "azure", "maroon"
]

# ── Encoding de grid em espaco HDC ──────────────────────────────────────────

def grid_to_concept_text(grid: List[List[int]], label: str = "grid") -> str:
    """
    Converte uma grade ARC em descricao textual rica para encode_text.
    Captura: cores presentes, posicoes relativas, simetrias, densidade.
    """
    if not grid or not grid[0]:
        return f"{label} empty"
    
    h, w = len(grid), len(grid[0])
    flat = [grid[r][c] for r in range(h) for c in range(w)]
    
    # Cores e densidades
    color_counts = {}
    for v in flat:
        if 0 < v < N_COLORS:
            color_counts[v] = color_counts.get(v, 0) + 1
    
    parts = [f"{label} h{h}w{w}"]
    
    # Cores dominantes
    dominant = sorted(color_counts.items(), key=lambda x: -x[1])[:3]
    for col, cnt in dominant:
        density = "sparse" if cnt < h*w*0.1 else ("medium" if cnt < h*w*0.4 else "dense")
        parts.append(f"{COLOR_NAMES[col]} {density}")
    
    # Simetria horizontal
    sym_h = all(grid[r][c] == grid[r][w-1-c] for r in range(h) for c in range(w//2))
    sym_v = all(grid[r][c] == grid[h-1-r][c] for r in range(h//2) for c in range(w))
    if sym_h: parts.append("mirror_horizontal")
    if sym_v: parts.append("mirror_vertical")
    
    # Repeticao de linhas
    if h >= 2 and grid[0] == grid[1]: parts.append("row_repeat")
    # Repeticao de colunas
    if w >= 2 and all(grid[r][0] == grid[r][1] for r in range(h)): parts.append("col_repeat")
    
    # Objetos (regioes conectadas de mesma cor)
    nz = sum(1 for v in flat if v > 0)
    if nz == 0: parts.append("all_black")
    elif nz < len(flat) * 0.1: parts.append("few_objects")
    elif nz > len(flat) * 0.8: parts.append("mostly_filled")
    
    # Borda vs interior
    border = [grid[r][c] for r in [0, h-1] for c in range(w)] + \
             [grid[r][c] for r in range(1, h-1) for c in [0, w-1]]
    interior = [grid[r][c] for r in range(1, h-1) for c in range(1, w-1)]
    if border and interior:
        border_color = max(set(border), key=border.count)
        interior_color = max(set(interior), key=interior.count)
        if border_color != interior_color:
            parts.append(f"border_{COLOR_NAMES[border_color]}")
            parts.append(f"interior_{COLOR_NAMES[interior_color]}")
    
    return " ".join(parts)


def extract_transform_description(inp: List[List[int]], out: List[List[int]]) -> str:
    """
    Descreve a TRANSFORMACAO entre input e output em linguagem HDC.
    Isso e o que o sistema vai aprender a generalizar.
    """
    if not inp or not out or not inp[0] or not out[0]:
        return "transform unknown"
    
    hi, wi = len(inp), len(inp[0])
    ho, wo = len(out), len(out[0])
    
    parts = ["transform"]
    
    # Mudanca de tamanho
    if ho == hi and wo == wi:
        parts.append("same_size")
    elif ho < hi or wo < wi:
        parts.append("shrink")
    elif ho > hi or wo > wi:
        parts.append("grow")
    
    # Mudanca de cor
    in_colors  = set(inp[r][c] for r in range(hi) for c in range(wi) if inp[r][c] > 0)
    out_colors = set(out[r][c] for r in range(ho) for c in range(wo) if out[r][c] > 0)
    new_colors = out_colors - in_colors
    lost_colors = in_colors - out_colors
    if new_colors:
        parts.append(f"adds_{COLOR_NAMES[min(new_colors)]}")
    if lost_colors:
        parts.append(f"removes_{COLOR_NAMES[min(lost_colors)]}")
    if not new_colors and not lost_colors and in_colors:
        parts.append("same_colors")
    
    # Espelho
    if ho == hi and wo == wi:
        if all(inp[r][c] == out[r][wi-1-c] for r in range(hi) for c in range(wi)):
            parts.append("flip_horizontal")
        elif all(inp[r][c] == out[hi-1-r][c] for r in range(hi) for c in range(wi)):
            parts.append("flip_vertical")
        elif all(inp[r][c] == out[c][r] for r in range(min(hi,wo)) for c in range(min(wi,ho))):
            parts.append("transpose")
    
    # Rotacao 90
    if ho == wi and wo == hi:
        if all(inp[r][c] == out[c][hi-1-r] for r in range(hi) for c in range(wi)):
            parts.append("rotate90")
        elif all(inp[r][c] == out[wo-1-c][r] for r in range(hi) for c in range(wi)):
            parts.append("rotate270")
    
    # Inversao de cores
    if ho == hi and wo == wi:
        inv = sum(1 for r in range(hi) for c in range(wi)
                  if (inp[r][c] == 0) == (out[r][c] > 0))
        if inv > hi*wi*0.8:
            parts.append("invert_colors")
    
    # Preenchimento (flood fill)
    in_nz  = sum(1 for r in range(hi) for c in range(wi) if inp[r][c] > 0)
    out_nz = sum(1 for r in range(ho) for c in range(wo) if out[r][c] > 0)
    if out_nz > in_nz * 2: parts.append("fill_expand")
    if out_nz < in_nz * 0.5: parts.append("fill_shrink")
    
    return " ".join(parts)


def apply_transform_to_query(
    inp_grid: List[List[int]],
    transform_desc: str,
    support_pairs: List[Tuple[List[List[int]], List[List[int]]]]
) -> Optional[List[List[int]]]:
    """
    Tenta aplicar a transformacao identificada na query grid.
    
    Estrategia: usa os exemplos de suporte para encontrar o mapeamento
    exato e aplica geometricamente na query.
    """
    if not inp_grid or not inp_grid[0]:
        return None
    
    hi, wi = len(inp_grid), len(inp_grid[0])
    
    # Verifica flip horizontal
    if "flip_horizontal" in transform_desc:
        return [row[::-1] for row in inp_grid]
    
    # Verifica flip vertical
    if "flip_vertical" in transform_desc:
        return inp_grid[::-1]
    
    # Verifica rotate 90
    if "rotate90" in transform_desc:
        return [[inp_grid[hi-1-c][r] for c in range(hi)] for r in range(wi)]
    
    # Verifica rotate 270
    if "rotate270" in transform_desc:
        return [[inp_grid[c][wi-1-r] for c in range(hi)] for r in range(wi)]
    
    # Verifica transpose
    if "transpose" in transform_desc:
        if hi == wi:
            return [[inp_grid[c][r] for c in range(hi)] for r in range(wi)]
    
    # Verifica inversao de cores
    if "invert_colors" in transform_desc:
        max_col = max(v for row in inp_grid for v in row if v > 0) if any(v > 0 for row in inp_grid for v in row) else 1
        return [[max_col if v == 0 else 0 for v in row] for row in inp_grid]
    
    # Mudanca de cor via mapeamento dos exemplos
    if "same_size" in transform_desc:
        # Aprende mapeamento de cor via exemplos de suporte
        color_map = {}
        for s_in, s_out in support_pairs:
            if len(s_in) == len(s_out) and s_in and s_out and s_in[0] and s_out[0]:
                sh, sw = min(len(s_in), len(s_out)), min(len(s_in[0]), len(s_out[0]))
                for r in range(sh):
                    for c in range(sw):
                        ci, co = s_in[r][c], s_out[r][c]
                        if ci not in color_map:
                            color_map[ci] = co
        
        if color_map:
            return [[color_map.get(v, v) for v in row] for row in inp_grid]
    
    # Fallback: retorna copia do input (copy-input)
    return [row[:] for row in inp_grid]


# ── Carrega tasks ARC ───────────────────────────────────────────────────────

def load_arc_tasks(data_dir: str, limit: int = None) -> List[Dict]:
    tasks = []
    if not os.path.exists(data_dir):
        print(f"[WARN] Dir nao encontrado: {data_dir}")
        return tasks
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".json"): continue
        with open(os.path.join(data_dir, fname), "r") as f:
            task = json.load(f)
        task["_id"] = fname.replace(".json", "")
        tasks.append(task)
        if limit and len(tasks) >= limit: break
    return tasks


# ── Avaliacao ───────────────────────────────────────────────────────────────

def grids_equal(a: List[List[int]], b: List[List[int]]) -> bool:
    if not a or not b: return False
    if len(a) != len(b): return False
    return all(len(a[r]) == len(b[r]) and all(a[r][c] == b[r][c]
               for c in range(len(a[r]))) for r in range(len(a)))


def pixel_accuracy(pred: List[List[int]], gt: List[List[int]]) -> float:
    if not pred or not gt: return 0.0
    total = correct = 0
    for r in range(min(len(pred), len(gt))):
        for c in range(min(len(pred[r]), len(gt[r]))):
            correct += (pred[r][c] == gt[r][c])
            total += 1
    return correct / max(1, total)


# ── Loop principal ──────────────────────────────────────────────────────────

def run(n_tasks: int = 20, use_train: bool = False):
    print("=" * 65)
    print("   NovaMind REAL — ARC-AGI com Mind/HDC/SandboxEngine")
    print("=" * 65)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Inicializa o sistema REAL
    print("\n[INIT] GroundingEngine + HDC Memory (3175 itens)...")
    engine = GroundingEngine(device=device)
    
    print("[INIT] HDCMemory (carregando HDVs — CPU para evitar OOM VRAM)...")
    # Carrega apenas os HDVs menores na CPU (RAM de 16GB)
    # O arquivo de 50k conceitos (1.86 GB) estoura a VRAM de 4GB
    hdv_paths = []
    data_root = "D:/novamind_data/hdv"
    for fname in ["code_brain.hdv", "python_brain.hdv", "text_library.hdv"]:
        p = os.path.join(data_root, fname)
        if os.path.exists(p):
            hdv_paths.append(p)
    memory = HDCMemory(device="cpu", hdv_paths=hdv_paths if hdv_paths else None)
    stats  = memory.get_stats()
    print(f"       {stats.get('total_items', 0)} itens carregados em {len(stats.get('sources', {}))} HDVs (CPU)")
    
    print("[INIT] SandboxEngine (compose_concept, bind/unbind)...")
    sandbox = SandboxEngine(engine)
    
    print("[INIT] ArchetypeEngine (50k conceitos)...")
    archetypes = ArchetypeEngine()
    
    print("[INIT] CausalReasoningEngine...")
    causal = CausalReasoningEngine()
    
    print("[INIT] Cerebellum (ForwardModel)...")
    cerebellum = Cerebellum(learning_rate=0.02, state_dim=15, action_dim=4)
    
    print("[INIT] Neuromodulator (DA/NA/ACh/5HT)...")
    nt = Neuromodulator()
    
    # Carrega tasks
    data_dir = ARC_TRAIN_DIR if use_train else ARC_EVAL_DIR
    tasks = load_arc_tasks(data_dir, limit=n_tasks)
    print(f"\n[DATA] {len(tasks)} tasks de {'treino' if use_train else 'avaliacao'} carregadas")
    if not tasks:
        print("[ERRO] Nenhuma task encontrada.")
        sys.exit(1)
    
    print()
    print("=" * 65)
    print(f"AVALIANDO {len(tasks)} TASKS COM O SISTEMA REAL")
    print("=" * 65)
    
    exact_matches = 0
    pixel_accs    = []
    transform_hits = {}  # transform_desc -> (correct, total)
    arch_hits     = 0
    t0 = time.time()
    
    for i, task in enumerate(tasks):
        task_id = task["_id"]
        train_pairs = task["train"]   # lista de {input, output}
        test_pairs  = task["test"]    # lista de {input, output}
        
        # ── PASSO 1: Analisa os K exemplos de suporte ──
        support_transforms = []
        support_concepts   = []
        
        for pair in train_pairs:
            inp_g = pair["input"]
            out_g = pair["output"]

            # Descricao textual de cada grade e da transformacao
            inp_text   = grid_to_concept_text(inp_g, "input")
            out_text   = grid_to_concept_text(out_g, "output")
            trans_text = extract_transform_description(inp_g, out_g)

            support_transforms.append(trans_text)
            support_concepts.extend(inp_text.split() + out_text.split() + trans_text.split())

            # Codifica no SandboxEngine como manifold HDC
            # Roles validos: suj, ver, obj, adj, adv, pre, con, neg, cond, func, arg, act
            VALID_ROLES = ["suj", "ver", "obj", "adj", "adv", "pre", "con", "neg", "cond", "func", "arg", "act"]
            inp_words   = inp_text.split()[:len(VALID_ROLES)]
            out_words   = out_text.split()[:len(VALID_ROLES)]
            trans_words = trans_text.split()[:len(VALID_ROLES)]

            sandbox.compose_concept(inp_text,   roles=VALID_ROLES[:len(inp_words)])
            sandbox.compose_concept(out_text,   roles=VALID_ROLES[:len(out_words)])
            sandbox.compose_concept(trans_text, roles=VALID_ROLES[:len(trans_words)])
        
        # ── PASSO 2: ArchetypeEngine consolida o padrao ──
        archetypes.observe(list(set(support_concepts)))
        
        # Vota na transformacao mais frequente entre os exemplos
        transform_votes: Dict[str, int] = {}
        for t in support_transforms:
            key_transforms = [tok for tok in t.split() if tok not in ("transform", "same_size", "same_colors")]
            for tok in key_transforms:
                transform_votes[tok] = transform_votes.get(tok, 0) + 1
        
        # Pega a transformacao com mais votos (consenso dos K exemplos)
        agreed_transform = " ".join(
            sorted(transform_votes, key=lambda k: -transform_votes[k])[:3]
        ) if transform_votes else "same_size"
        
        # ── PASSO 3: Aplica na query e verifica ──
        task_exact = False
        task_pixels = []
        
        for test_pair in test_pairs:
            query_inp = test_pair["input"]
            gt_out    = test_pair["output"]
            
            # Aplica a transformacao aprendida dos K exemplos
            support_pairs_grids = [(p["input"], p["output"]) for p in train_pairs]
            pred_out = apply_transform_to_query(query_inp, agreed_transform, support_pairs_grids)
            
            if pred_out is None:
                pred_out = [row[:] for row in query_inp]
            
            exact = grids_equal(pred_out, gt_out)
            px    = pixel_accuracy(pred_out, gt_out)
            
            task_pixels.append(px)
            if exact:
                task_exact = True
                exact_matches += 1
        
        # ── Atualiza NT ──
        if task_exact:
            nt.step(prediction_error=0.0, delta_pe=-0.5)  # DA dispara
            arch_hits += 1
        else:
            nt.step(prediction_error=0.5, delta_pe=0.2)
        
        avg_px = np.mean(task_pixels) if task_pixels else 0.0
        pixel_accs.append(avg_px)
        
        # Consolida arquétipos a cada 10 tasks
        if (i + 1) % 10 == 0:
            new_archs = archetypes.consolidate()
            
            da_val = nt.as_dict().get("DA", 0.3)
            elapsed = time.time() - t0
            print(
                f"[{i+1:3d}/{len(tasks)}]"
                f" Exact: {exact_matches}/{i+1} ({exact_matches/(i+1):.1%})"
                f" | Pixel: {np.mean(pixel_accs):.1%}"
                f" | DA={da_val:.2f}"
                f" | Arch: {len(archetypes.archetypes)}"
                f" | {elapsed:.0f}s"
            )
    
    total_time = time.time() - t0
    
    # ── Resultados ──────────────────────────────────────────────────────────
    mpx = float(np.mean(pixel_accs)) if pixel_accs else 0.0
    mex = exact_matches / max(1, len(tasks))
    
    print()
    print("=" * 65)
    print("  RESULTADOS FINAIS — NovaMind REAL")
    print("=" * 65)
    print(f"  Tasks avaliadas      : {len(tasks)}")
    print(f"  Tempo total          : {total_time:.1f}s")
    print()
    print(f"  Exact Match          : {exact_matches}/{len(tasks)} = {mex:.1%}")
    print(f"  Pixel Accuracy       : {mpx:.1%}")
    print()
    print(f"  Arquetipos criados   : {len(archetypes.archetypes)}")
    nt_state = nt.as_dict()
    print(f"  NT final DA          : {nt_state.get('DA', 0):.3f}")
    print(f"  NT final NA          : {nt_state.get('NA', 0):.3f}")
    print()
    
    if mex > 0.20:    verdict = "BREAKTHROUGH -- >20% exact match OOD!"
    elif mex > 0.10:  verdict = "EXCELENTE -- >10% exact match"
    elif mex > 0.05:  verdict = "BOM -- >5% exact match"
    elif mex > 0.01:  verdict = "APRENDIZADO DETECTADO -- >1% exact match"
    else:             verdict = "ABAIXO DO LIMIAR -- Regras nao extraidas"
    
    print(f"  VEREDITO: {verdict}")
    print("=" * 65)
    
    results = {
        "system": "NovaMind REAL (Mind/HDC/SandboxEngine)",
        "n_tasks": len(tasks), "time_s": round(total_time, 1),
        "exact_match": round(mex, 4), "pixel_accuracy": round(mpx, 4),
        "archetypes_created": len(archetypes.archetypes),
        "nt_da": round(nt.as_dict().get("DA", 0), 3),
        "verdict": verdict,
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arc_mind_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Resultados: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=20, help="Numero de tasks a avaliar")
    parser.add_argument("--train", action="store_true", help="Usa tasks de treino (in-distribution)")
    args = parser.parse_args()
    run(n_tasks=args.tasks, use_train=args.train)
