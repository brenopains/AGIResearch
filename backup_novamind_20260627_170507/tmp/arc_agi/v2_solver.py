"""
ARC-NOVAMIND v2 — fusao da arquitetura emergente (NovaMind) com o estado-da-arte ARC 2025.

SOTA incorporado (pesquisado): refinement loops (tema vencedor 2025), self-consistency/
Product-of-Experts voting (Franzen 71.6%), wake-sleep program library (Pang), debate
multi-agente (Berman 79.6%), D8 augmentation. Tudo VERIFICADO POR EXECUCAO.

Nosso (unico): personas por drive darwiniano + COMPOSICAO NAO-COMUTATIVA + biblioteca de
ABSTRACOES de codigo acumulada entre tasks (breadth por experiencia).

Sinais de AGI medidos: (1) transferencia (biblioteca ON vs OFF), (2) calibracao do consenso
(quando K experts concordam, acerta mais?), (3) tasks resolvidas SO por composicao/refinamento.
"""
import sys, json, time, re, threading, random, pathlib, os as _os
from collections import Counter
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "sci"))
from dotenv import load_dotenv
load_dotenv(pathlib.Path(__file__).parent.parent.parent / "sci" / ".env")
from shared.llm_client import ask
import numpy as np
from solver import (PERSONAS, SAFE_BUILTINS, grid_to_str, extract_code, compile_transform,
                    run_with_timeout, normalize_grid, verify_on_train, apply_to_test,
                    test_correct, ARC_DIR, OUT_DIR)

# ---------- D8 augmentation (8 simetrias) p/ checar se a regra e ROBUSTA (nao overfit) ----------
def d8_variants(g):
    a = np.array(g)
    outs = []
    for k in range(4):
        r = np.rot90(a, k)
        outs.append(r); outs.append(np.fliplr(r))
    return [o.tolist() for o in outs]

def program_robustness(fn, task, n_perm=3):
    """quao robusta a regra e sob permutacao de cores? (overfit quebra; regra real aguenta).
    Aplica permutacao de cores aos pares de treino e checa se a funcao ainda mapeia certo."""
    rng = random.Random(12345)
    robust = 0
    for _ in range(n_perm):
        perm = list(range(10)); rng.shuffle(perm)
        pm = {i: perm[i] for i in range(10)}
        ok = True
        for ex in task["train"]:
            gin = [[pm[c] for c in row] for row in ex["input"]]
            gout_expected = [[pm[c] for c in row] for row in ex["output"]]
            res, err = run_with_timeout(fn, ([list(r) for r in gin],))
            if err is not None: ok = False; break
            ng = normalize_grid(res)
            if ng != gout_expected: ok = False; break
        if ok: robust += 1
    return robust / n_perm   # 1.0 = regra invariante a cor (forte sinal de regra real)

# ---------- prompts ----------
def rule_of(code):
    """extrai a linha de REGRA do comentario, se houver."""
    m = re.search(r"#\s*REGRA:\s*(.+)", code)
    return m.group(1).strip()[:160] if m else ""

def gen_prompt(task, persona_key, library_code, failed_feedback=None):
    p = PERSONAS[persona_key]
    pairs = ""
    for i, ex in enumerate(task["train"]):
        pairs += f"\n# Exemplo {i+1}\nINPUT:\n{grid_to_str(ex['input'])}\nOUTPUT:\n{grid_to_str(ex['output'])}\n"
    lib = ""
    if library_code:
        # library_code = lista de PROGRAMAS INTEIROS (transform) que resolveram tasks passadas.
        # Few-shot: mostra a tecnica/padrao p/ reaproveitar (wake-sleep de programas, nao so helpers).
        ex_blocks = []
        for item in library_code[:3]:
            code = item["code"] if isinstance(item, dict) else item
            rule = (item.get("rule") if isinstance(item, dict) else rule_of(code)) or ""
            head = f"# (regra: {rule})\n" if rule else ""
            ex_blocks.append(head + code[:700])
        lib = ("\n\nTRANSFORMACOES QUE RESOLVERAM TASKS ANTERIORES (reaproveite TECNICAS/PADROES de codigo "
               "que se encaixem nesta task — adapte, nao copie cego):\n```python\n" + "\n\n# ---\n\n".join(ex_blocks) + "\n```")
    fb = ""
    if failed_feedback:
        fb = f"\n\n[REFINAMENTO] Sua tentativa anterior FALHOU nestes exemplos de treino:\n{failed_feedback}\nCorrija a REGRA GERAL — nao decore os exemplos."
    generalize_note = ""
    if _os.environ.get("V3_GENERALIZE") == "1":
        generalize_note = ("\n\n[GENERALIZACAO] Escreva a primitiva o MAIS GERAL possivel: detecte os parametros "
                           "a partir do proprio grid (ex: em vez de 'rotaciona 180 fixo', detecte qual rotacao/cor/escala "
                           "mapeia os exemplos e aplique). A funcao deve funcionar tambem em OUTRAS tasks da MESMA familia "
                           "de transformacao, nao so nesta. Auto-configuravel a partir de g.")
    sysp = f"""Voce resolve ARC-AGI. {p}

Grids = matrizes de int 0-9 (0=fundo geralmente). Descubra a REGRA que transforma INPUT->OUTPUT.
Primeiro pense na regra em 1 frase (comentario), depois implemente.{generalize_note}

Responda APENAS:
```python
def transform(g):
    # REGRA: <descreva a regra geral em 1 frase>
    # g: lista de listas int. Retorne a nova grade. Use numpy as np. SEM imports/open/arquivos.
    ...
    return resultado
```
A funcao DEVE reproduzir EXATAMENTE o OUTPUT de TODOS os exemplos. Pense na regra geral que generaliza, NAO decore os exemplos (isso e overfit e falha no teste)."""
    user = f"PARES DE TREINO:{pairs}{lib}{fb}\n\nEscreva transform(g) com a regra GERAL. APENAS o bloco Python."
    return sysp, user

def refute_prompt(task, programs_desc):
    pairs = ""
    for i, ex in enumerate(task["train"][:3]):
        pairs += f"\nINPUT:\n{grid_to_str(ex['input'])}\nOUTPUT:\n{grid_to_str(ex['output'])}\n"
    sysp = """Voce e o REFUTADOR (advogado do diabo) num debate cientifico sobre a regra de um puzzle ARC.
Recebe varias REGRAS candidatas que TODAS passam nos exemplos de treino. Seu trabalho: identificar qual e mais
provavel de ser a regra GERAL VERDADEIRA (generaliza) vs qual e OVERFIT (decorou). Considere: simplicidade,
invariancia, se a regra explica o PORQUE. Responda JSON: {"best_index": <int>, "reason": "1 frase"}."""
    user = f"EXEMPLOS DE TREINO:{pairs}\n\nREGRAS CANDIDATAS (todas passam no treino):\n{programs_desc}\n\nQual generaliza melhor? JSON."
    return sysp, user

LLM_MODEL_ENV = None  # usa o default do env (LLM_MODEL)

import os as _os
def llm_code(task, persona, library_code, failed_feedback, temperature, max_tokens=None):
    # com thinking ligado, precisa de teto alto (raciocinio + codigo); senao 1300 basta
    if max_tokens is None:
        max_tokens = 6000 if _os.environ.get("VENICE_THINKING") == "1" else 1300
    sysp, user = gen_prompt(task, persona, library_code, failed_feedback)
    try:
        raw = ask([{"role":"user","content":user}], system=sysp, temperature=temperature, max_tokens=max_tokens)
    except Exception as e:
        return None, None, f"llm:{e}"
    code = extract_code(raw)
    fn, err = compile_transform(code)
    return fn, code, err

def failing_examples_str(fn, task, limit=2):
    """descreve quais exemplos de treino a funcao erra (p/ refinamento)."""
    out = []
    for i, ex in enumerate(task["train"]):
        res, err = run_with_timeout(fn, ([list(r) for r in ex["input"]],))
        ng = normalize_grid(res) if err is None else None
        if ng != ex["output"]:
            got = grid_to_str(ng) if ng else f"(erro: {str(err)[:40]})"
            out.append(f"Exemplo {i+1}:\nINPUT:\n{grid_to_str(ex['input'])}\nESPERADO:\n{grid_to_str(ex['output'])}\nSEU OUTPUT:\n{got}")
        if len(out) >= limit: break
    return "\n".join(out)

def extract_helpers(code):
    """wake-sleep: extrai funcoes auxiliares (def diferentes de transform) p/ a biblioteca."""
    helpers = []
    for m in re.finditer(r"(def\s+(\w+)\s*\([^)]*\):(?:\n(?:    .*|\s*)?)+)", code):
        name = m.group(2)
        if name != "transform":
            helpers.append(m.group(1).rstrip())
    return helpers
