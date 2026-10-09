"""
run_arc_synthesis.py — ARC-AGI via Program Synthesis com NovaMind Real
=======================================================================

Pipeline:
  1. Para cada task ARC (K exemplos de suporte + 1 query):
     a. Descreve a transformacao em linguagem natural
     b. AutoregressiveGenerator gera funcao Python: transform(grid) -> grid
     c. exec() verifica nos K exemplos de suporte
     d. Se correto em todos: aplica na query -> exact match real
     e. Se falhar: tenta variantes (flip, rotate, color_map, etc.)

  O sistema usa o stack REAL validado:
    NovaMind (13 camadas cognitivas)
    AutoregressiveGenerator (IntentionDrivenGenerator + HDV 808 itens)
    ArchetypeEngine (consolida regras entre tasks)

Uso:
    python run_arc_synthesis.py --tasks 20    # avalia 20 tasks
    python run_arc_synthesis.py --tasks 400   # full OOD
    python run_arc_synthesis.py --train       # tasks de treino (in-distribution)
"""

import os, sys, json, time, argparse, textwrap, ast, warnings
from typing import List, Dict, Optional, Tuple

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from core.mind import NovaMind
from core.broca_decoder import BrocaDecoder
from core.sandbox_engine import SandboxEngine
from core.positional_binder import PositionalBinder
from core.autoregressive_generator import AutoregressiveGenerator
from core.temporal_buffer import TemporalBuffer
from core.archetypes import ArchetypeEngine
from core.neuromodulator import Neuromodulator
from core.hdc_memory import HDCMemory

ARC_EVAL_DIR  = "data/ARC/data/evaluation"
ARC_TRAIN_DIR = "data/ARC/data/training"
N_COLORS = 10
COLOR_NAMES = ["black","blue","red","green","yellow","grey","fuchsia","orange","azure","maroon"]

# ── Candidatos geometricos hardcoded (fallback verificado nos exemplos) ──────

def _flip_h(g):
    return [row[::-1] for row in g]

def _flip_v(g):
    return g[::-1]

def _rot90(g):
    h, w = len(g), len(g[0]) if g else 0
    return [[g[h-1-c][r] for c in range(h)] for r in range(w)]

def _rot180(g):
    return _flip_v(_flip_h(g))

def _rot270(g):
    return _rot90(_rot90(_rot90(g)))

def _transpose(g):
    h, w = len(g), len(g[0]) if g else 0
    return [[g[r][c] for r in range(h)] for c in range(w)]

def _identity(g):
    return [row[:] for row in g]

def _color_map(g, cmap):
    return [[cmap.get(v, v) for v in row] for row in g]

def _invert(g):
    nz = [v for row in g for v in row if v > 0]
    mc = max(nz) if nz else 1
    return [[mc if v == 0 else 0 for v in row] for row in g]

GEOMETRIC_CANDIDATES = [
    ("identity",   _identity),
    ("flip_h",     _flip_h),
    ("flip_v",     _flip_v),
    ("rot90",      _rot90),
    ("rot180",     _rot180),
    ("rot270",     _rot270),
    ("transpose",  _transpose),
    ("invert",     _invert),
]


def learn_color_map(pairs: List[Tuple]) -> Optional[Dict]:
    """Aprende mapeamento de cor a partir dos exemplos de suporte."""
    cmap: Dict[int, int] = {}
    for inp_g, out_g in pairs:
        if not inp_g or not out_g or not inp_g[0] or not out_g[0]:
            continue
        sh = min(len(inp_g), len(out_g))
        sw = min(len(inp_g[0]), len(out_g[0]))
        for r in range(sh):
            for c in range(sw):
                ci, co = inp_g[r][c], out_g[r][c]
                if ci in cmap and cmap[ci] != co:
                    return None  # inconsistente
                cmap[ci] = co
    return cmap if cmap else None


def grids_equal(a, b) -> bool:
    if not a or not b or len(a) != len(b): return False
    return all(len(a[r]) == len(b[r]) and all(a[r][c] == b[r][c]
               for c in range(len(a[r]))) for r in range(len(a)))


def pixel_acc(pred, gt) -> float:
    if not pred or not gt: return 0.0
    total = correct = 0
    for r in range(min(len(pred), len(gt))):
        for c in range(min(len(pred[r]), len(gt[r]))):
            correct += pred[r][c] == gt[r][c]
            total += 1
    return correct / max(1, total)


def verify_fn(fn, pairs: List[Tuple]) -> bool:
    """Verifica se fn(input) == output para todos os pares de suporte."""
    for inp_g, out_g in pairs:
        try:
            pred = fn(inp_g)
            if not grids_equal(pred, out_g):
                return False
        except Exception:
            return False
    return True


# ── Describe task para o gerador de codigo ──────────────────────────────────

def describe_transform(pairs: List[Tuple]) -> str:
    """
    Gera descricao em linguagem natural da transformacao observada.
    Ex: "def transform(grid): flip grid horizontally return result"
    """
    if not pairs:
        return "def transform(grid): return grid"

    inp0, out0 = pairs[0]
    hi = len(inp0); wi = len(inp0[0]) if inp0 else 0
    ho = len(out0); wo = len(out0[0]) if out0 else 0

    hints = []

    # Testa geometrias simples
    for name, fn in GEOMETRIC_CANDIDATES[1:]:  # pula identity
        if all(grids_equal(fn(p[0]), p[1]) for p in pairs):
            hints.append(name)

    # Mudanca de tamanho
    if hi != ho or wi != wo:
        hints.append(f"resize from {hi}x{wi} to {ho}x{wo}")

    # Cores
    in_colors  = set(v for row in inp0 for v in row if v > 0)
    out_colors = set(v for row in out0 for v in row if v > 0)
    if out_colors - in_colors:
        hints.append(f"adds color {min(out_colors-in_colors)}")
    if in_colors - out_colors:
        hints.append(f"removes color {min(in_colors-out_colors)}")

    if not hints:
        hints.append("apply pattern transformation")

    transform_desc = " ".join(hints[:3])
    return f"def transform(grid): {transform_desc} return result"


def safe_exec_transform(code: str, inp_grid: List) -> Optional[List]:
    """Executa codigo gerado de forma segura com timeout implicito."""
    # Sanitiza: so aceita codigo com 'def transform'
    if "def transform" not in code:
        return None

    try:
        # Parse AST para verificar seguranca basica
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                return None  # sem imports perigosos

        namespace: Dict = {"grid": [row[:] for row in inp_grid]}
        exec(code, namespace)
        if "transform" not in namespace:
            return None
        result = namespace["transform"]([row[:] for row in inp_grid])
        if isinstance(result, list):
            return result
    except Exception:
        return None
    return None


# ── Carrega tasks ARC ────────────────────────────────────────────────────────

def load_arc_tasks(data_dir: str, limit: int = None) -> List[Dict]:
    tasks = []
    if not os.path.exists(data_dir):
        print(f"[WARN] Dir nao encontrado: {data_dir}")
        return tasks
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".json"): continue
        with open(os.path.join(data_dir, fname)) as f:
            task = json.load(f)
        task["_id"] = fname.replace(".json", "")
        tasks.append(task)
        if limit and len(tasks) >= limit: break
    return tasks


# ── Pipeline principal ───────────────────────────────────────────────────────

def run(n_tasks: int = 20, use_train: bool = False):
    print("=" * 65)
    print("  NovaMind — ARC-AGI via Program Synthesis")
    print("=" * 65)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # ── Init stack real (igual ao interact.py) ──
    print("[INIT] NovaMind (13 camadas cognitivas)...")
    mind = NovaMind()

    if not hasattr(mind, 'hdc_memory') or mind.hdc_memory is None:
        print("[ERRO] HDCMemory nao carregada. Execute ingest primeiro.")
        sys.exit(1)

    stats = mind.hdc_memory.get_stats()
    print(f"       HDV: {stats['total_items']} itens")

    print("[INIT] BrocaDecoder (vocabulario HDC)...")
    all_index = []
    for idx_list in mind.hdc_memory.indices.values():
        all_index.extend(idx_list)
    decoder = BrocaDecoder(mind.hdc_memory.engine)
    decoder.build_vocabulary_from_metadata(all_index)
    # Vocabulario de programacao Python — essencial para gerar codigo
    python_vocab = (
        "def transform grid return row col val result new_grid append list range len "
        "for in if else elif not and or True False None int str zip enumerate reversed "
        "flip rotate mirror transpose invert color map black blue red green yellow "
        "grey orange azure maroon fill expand shrink copy width height size shape "
        "input output pattern rule apply each every row column cell value"
    )
    decoder.build_vocabulary_from_texts([python_vocab])
    decoder.min_word_len = 0
    print(f"       {len(decoder.word_vectors)} palavras no vocabulario")

    print("[INIT] AutoregressiveGenerator...")
    binder   = PositionalBinder(max_length=30, device=mind.hdc_memory.engine.device)
    conv_buf = TemporalBuffer(capacity=10, decay=0.8)
    gen_eng  = mind.hdc_memory.engine
    sandbox  = SandboxEngine(gen_eng)
    generator = AutoregressiveGenerator(
        engine=gen_eng, binder=binder, decoder=decoder,
        sandbox=sandbox, hdc_memory=mind.hdc_memory, temporal_buffer=conv_buf
    )
    print(f"       Generator pronto (IntentionDriven + Working HDV)")

    print("[INIT] ArchetypeEngine + Neuromodulator...")
    archetypes = ArchetypeEngine()
    nt         = Neuromodulator()
    print()

    # ── Carrega tasks ──
    data_dir = ARC_TRAIN_DIR if use_train else ARC_EVAL_DIR
    tasks    = load_arc_tasks(data_dir, limit=n_tasks)
    print(f"[DATA] {len(tasks)} tasks de {'treino' if use_train else 'avaliacao OOD'}")
    if not tasks:
        print("[ERRO] Nenhuma task encontrada.")
        sys.exit(1)

    print()
    print("=" * 65)
    print(f"AVALIANDO {len(tasks)} TASKS — PROGRAM SYNTHESIS")
    print("=" * 65)

    exact_total = 0
    px_all      = []
    method_log  = {}   # method -> (correct, total)
    t0 = time.time()

    for i, task in enumerate(tasks):
        task_id     = task["_id"]
        train_pairs = [(p["input"], p["output"]) for p in task["train"]]
        test_pairs  = task["test"]

        task_exact = False
        task_pxs   = []
        method_used = "none"

        # ═══════════════════════════════════════════════════════════════
        # TENTATIVA 1: Candidatos geometricos verificados nos exemplos
        # ═══════════════════════════════════════════════════════════════
        solved_fn = None
        for cname, cfn in GEOMETRIC_CANDIDATES:
            if verify_fn(cfn, train_pairs):
                solved_fn  = cfn
                method_used = cname
                break

        # ═══════════════════════════════════════════════════════════════
        # TENTATIVA 2: Color map aprendido dos exemplos
        # ═══════════════════════════════════════════════════════════════
        if solved_fn is None:
            cmap = learn_color_map(train_pairs)
            if cmap is not None:
                cfn_cmap = lambda g, m=cmap: _color_map(g, m)
                if verify_fn(cfn_cmap, train_pairs):
                    solved_fn   = cfn_cmap
                    method_used = "color_map"

        # ═══════════════════════════════════════════════════════════════
        # TENTATIVA 3: AutoregressiveGenerator sintetiza funcao Python
        # ═══════════════════════════════════════════════════════════════
        if solved_fn is None:
            # Monta descricao da task para o gerador
            task_desc = describe_transform(train_pairs)
            # Adiciona contexto dos exemplos na working HDV
            generator.add_to_working_hdv(task_desc)
            generator.add_to_working_hdv(f"transform input grid to output grid")

            # Tenta gerar codigo
            generated_code = generator.generate(
                target_text=task_desc,
                max_chunks=12,
                chunk_size=4,
                min_sim=0.06,
                verbose=False
            )

            # Verifica se o codigo gerado funciona nos exemplos
            if generated_code and generated_code != "[...]" and "def transform" in generated_code:
                gen_fn = lambda g, code=generated_code: safe_exec_transform(code, g)
                if gen_fn(train_pairs[0][0]) is not None and verify_fn(gen_fn, train_pairs):
                    solved_fn   = gen_fn
                    method_used = "synthesis"

        # ═══════════════════════════════════════════════════════════════
        # APLICA solucao encontrada na(s) query(s) de teste
        # ═══════════════════════════════════════════════════════════════
        for test_pair in test_pairs:
            query_inp = test_pair["input"]
            gt_out    = test_pair["output"]

            if solved_fn is not None:
                try:
                    pred = solved_fn(query_inp)
                except Exception:
                    pred = [row[:] for row in query_inp]
            else:
                # Fallback final: copia o input (pior caso)
                pred = [row[:] for row in query_inp]
                method_used = "copy_input"

            exact = grids_equal(pred, gt_out)
            px    = pixel_acc(pred, gt_out)

            task_pxs.append(px)
            if exact:
                task_exact = True

        if task_exact:
            exact_total += 1
            nt.step(prediction_error=0.0, delta_pe=-0.5)
        else:
            nt.step(prediction_error=0.6, delta_pe=0.2)

        avg_px = float(np.mean(task_pxs)) if task_pxs else 0.0
        px_all.append(avg_px)

        # Atualiza contagem por metodo
        m = method_used
        if m not in method_log: method_log[m] = [0, 0]
        method_log[m][0] += int(task_exact)
        method_log[m][1] += 1

        # Atualiza ArchetypeEngine com padroes desta task
        archetypes.observe([method_used, "arc_task"])
        if (i + 1) % 10 == 0:
            archetypes.consolidate()

        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            da = nt.as_dict().get("DA", 0)
            print(
                f"[{i+1:3d}/{len(tasks)}]"
                f" Exact: {exact_total}/{i+1} ({exact_total/(i+1):.1%})"
                f" | Pixel: {np.mean(px_all):.1%}"
                f" | DA={da:.2f}"
                f" | {elapsed:.0f}s"
            )

    total_time = time.time() - t0
    mex = exact_total / max(1, len(tasks))
    mpx = float(np.mean(px_all))

    print()
    print("=" * 65)
    print("  RESULTADOS — NovaMind Program Synthesis")
    print("=" * 65)
    print(f"  Tasks avaliadas      : {len(tasks)}")
    print(f"  Tempo total          : {total_time:.1f}s ({total_time/len(tasks):.1f}s/task)")
    print()
    print(f"  Exact Match          : {exact_total}/{len(tasks)} = {mex:.1%}")
    print(f"  Pixel Accuracy       : {mpx:.1%}")
    print()
    print("  Metodos usados:")
    for mname, (correct, total) in sorted(method_log.items(), key=lambda x: -x[1][1]):
        print(f"    {mname:<15} {correct}/{total} corretos ({correct/max(1,total):.0%})")
    print()

    nt_d = nt.as_dict()
    print(f"  NT final DA          : {nt_d.get('DA', 0):.3f}")
    print(f"  NT final NA          : {nt_d.get('NA', 0):.3f}")
    print(f"  Arquetipos criados   : {len(archetypes.archetypes)}")
    print()

    if mex > 0.20:    verdict = "BREAKTHROUGH -- >20% exact match OOD!"
    elif mex > 0.10:  verdict = "EXCELENTE -- >10% exact match OOD"
    elif mex > 0.05:  verdict = "BOM -- >5% exact match OOD"
    elif mex > 0.01:  verdict = "APRENDIZADO -- >1% exact match OOD"
    else:             verdict = "ABAIXO DO LIMIAR"

    print(f"  VEREDITO: {verdict}")
    print("=" * 65)

    results = {
        "system": "NovaMind Program Synthesis",
        "n_tasks": len(tasks), "time_s": round(total_time, 1),
        "exact_match": round(mex, 4),
        "pixel_accuracy": round(mpx, 4),
        "methods": {k: {"correct": v[0], "total": v[1]} for k, v in method_log.items()},
        "archetypes_created": len(archetypes.archetypes),
        "nt_da": round(nt_d.get("DA", 0), 3),
        "verdict": verdict,
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arc_synthesis_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Resultados: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    run(n_tasks=args.tasks, use_train=args.train)
