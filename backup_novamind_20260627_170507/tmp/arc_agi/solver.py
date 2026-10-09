"""
ARC-NOVAMIND — solver que aplica a arquitetura emergente (personas + composicao nao-comutativa +
drives darwinianos + biblioteca de estrategias acumulada) por cima de um LLM (GLM/Venice) como
fonte de BREADTH, com VERIFICACAO POR EXECUCAO (o programa proposto roda nos pares de treino;
ou reproduz a saida ou nao — impossivel falsificar).

A aposta AGI-relevante: drives + biblioteca de estrategias acumulam conhecimento ENTRE tasks.
Mede-se TRANSFERENCIA via A/B: COM acumulo (quente) vs SEM acumulo (frio por task).
"""
import sys, os, json, time, re, threading, random, pathlib, traceback
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "sci"))
from dotenv import load_dotenv
load_dotenv(pathlib.Path(__file__).parent.parent.parent / "sci" / ".env")
from shared.llm_client import ask
import numpy as np

ARC_DIR = pathlib.Path(__file__).parent.parent / "arc_data" / "data" / "training"
OUT_DIR = pathlib.Path(__file__).parent / "runs"
OUT_DIR.mkdir(exist_ok=True)

# ---------------- PERSONAS (vieses indutivos distintos p/ ARC) ----------------
PERSONAS = {
    "geometra":   "Voce pensa em SIMETRIA e GEOMETRIA: rotacoes (90/180/270), reflexoes (horizontal/vertical/diagonal), translacoes, simetrias espelhadas, completar padroes simetricos.",
    "pintor":     "Voce pensa em CORES: mapeamento de cor->cor, contagem de cores, recolorir regioes, preencher areas fechadas (flood fill), substituir a cor mais/menos frequente.",
    "objetista":  "Voce pensa em OBJETOS: componentes conexos, detectar formas, mover/copiar/contar/ordenar objetos, manter o maior/menor, remover ruido, isolar o objeto unico diferente.",
    "escalador":  "Voce pensa em ESCALA e REPETICAO: tiling (repetir o grid), scaling (ampliar NxN), crop (recortar), padding, espelhar-e-colar, fractais (repetir o grid dentro das celulas nao-zero).",
    "relacional": "Voce pensa em RELACOES: gravidade (objetos caem), alinhamento, bordas/molduras, conectar pontos com linhas, projetar raios, relacao dentro/fora, contornos.",
}

import numpy as _np, math as _math, collections as _collections, itertools as _itertools
def _safe_import(name, *a, **k):
    """permite SO modulos seguros e uteis p/ ARC (numpy/math/collections/itertools + submodulos numpy)."""
    allowed = {"numpy": _np, "np": _np, "math": _math, "collections": _collections, "itertools": _itertools}
    if name in allowed: return allowed[name]
    if name.startswith("numpy") or name.startswith("math") or name.startswith("collections") or name.startswith("itertools"):
        import importlib
        return importlib.import_module(name)   # submodulos internos (ex: numpy.core._methods)
    raise ImportError(f"import bloqueado: {name}")

_bset = ["range","len","list","dict","set","tuple","enumerate","zip","min","max",
         "sum","abs","sorted","map","filter","any","all","int","float","bool","str",
         "reversed","round","frozenset","slice","isinstance","print","type","getattr",
         "setattr","hasattr","repr","Exception","ValueError","IndexError","KeyError",
         "TypeError","ZeroDivisionError","StopIteration","next","iter","divmod","pow",
         "ord","chr","format","__build_class__","property","staticmethod","classmethod","object"]
SAFE_BUILTINS = {k: (__builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k))
                 for k in _bset if (k in __builtins__ if isinstance(__builtins__, dict) else hasattr(__builtins__, k))}
SAFE_BUILTINS["__import__"] = _safe_import
SAFE_BUILTINS["True"]=True; SAFE_BUILTINS["False"]=False; SAFE_BUILTINS["None"]=None

def grid_to_str(g):
    return "\n".join("".join(str(c) for c in row) for row in g)

def make_prompt(task, persona_key, library_hints, allow_compose=False):
    p = PERSONAS[persona_key]
    pairs = ""
    for i, ex in enumerate(task["train"]):
        pairs += f"\n# Exemplo {i+1}\nINPUT:\n{grid_to_str(ex['input'])}\nOUTPUT:\n{grid_to_str(ex['output'])}\n"
    hints = ""
    if library_hints:
        hints = "\n\nESTRATEGIAS QUE FUNCIONARAM EM TASKS ANTERIORES (pode reaproveitar a ideia):\n- " + "\n- ".join(library_hints[:6])
    sys_prompt = f"""Voce resolve puzzles ARC-AGI. {p}

Cada grid e uma matriz de inteiros 0-9 (0 normalmente = fundo). Voce recebe pares INPUT->OUTPUT de treino e deve DESCOBRIR a transformacao.

Responda APENAS com uma funcao Python pura:

def transform(g):
    # g: lista de listas de int (0-9). Retorne a nova grade (lista de listas de int).
    # Pode usar numpy como np. NAO use imports, open, arquivos, rede.
    ...
    return resultado

A funcao DEVE reproduzir EXATAMENTE o OUTPUT de TODOS os exemplos de treino. Pense na regra geral, nao decore. Codigo entre ```python e ```. SEM texto fora do bloco."""
    user = f"PARES DE TREINO:{pairs}{hints}\n\nEscreva transform(g) que reproduz todos os OUTPUTs. APENAS o bloco de codigo Python."
    return sys_prompt, user

CODE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)
def extract_code(raw):
    m = CODE_RE.search(raw)
    code = m.group(1) if m else raw
    # pega do primeiro 'def transform' em diante
    idx = code.find("def transform")
    return code[idx:] if idx >= 0 else code

def run_with_timeout(fn, args, timeout=4.0):
    result = {}
    def worker():
        try: result["v"] = fn(*args)
        except Exception as e: result["e"] = e
    t = threading.Thread(target=worker, daemon=True); t.start(); t.join(timeout)
    if t.is_alive(): return None, "timeout"
    if "e" in result: return None, result["e"]
    return result.get("v"), None

def compile_transform(code):
    ns = {"np": np, "numpy": np, "__builtins__": SAFE_BUILTINS}
    try:
        exec(code, ns)
    except Exception as e:
        return None, f"compile:{e}"
    fn = ns.get("transform")
    if not callable(fn): return None, "sem transform"
    return fn, None

def normalize_grid(out):
    """converte saida (np.array/list) p/ lista de listas de int python, ou None se invalido."""
    try:
        if isinstance(out, np.ndarray): out = out.tolist()
        if not isinstance(out, list) or not out: return None
        norm = []
        for row in out:
            if isinstance(row, np.ndarray): row = row.tolist()
            if not isinstance(row, list): return None
            norm.append([int(c) for c in row])
        return norm
    except Exception:
        return None

def verify_on_train(fn, task):
    """retorna (resolve_todos, n_acertos, n_total)."""
    ok = 0
    for ex in task["train"]:
        out, err = run_with_timeout(fn, ([list(r) for r in ex["input"]],))
        if err is not None: return False, ok, len(task["train"])
        g = normalize_grid(out)
        if g is not None and g == ex["output"]: ok += 1
    return ok == len(task["train"]), ok, len(task["train"])

def apply_to_test(fn, task):
    preds = []
    for ex in task["test"]:
        out, err = run_with_timeout(fn, ([list(r) for r in ex["input"]],))
        preds.append(normalize_grid(out) if err is None else None)
    return preds

def test_correct(preds, task):
    """so funciona se a task tem 'output' no test (training set tem)."""
    for p, ex in zip(preds, task["test"]):
        if "output" not in ex: return None
        if p != ex["output"]: return False
    return True
