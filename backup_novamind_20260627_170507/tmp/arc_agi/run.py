"""
Orquestrador ARC-NOVAMIND. Resolve um stream de tasks ARC com:
  - personas amostradas por DRIVE (peso evolutivo darwiniano)
  - VERIFICACAO POR EXECUCAO (programa roda nos pares de treino)
  - COMPOSICAO NAO-COMUTATIVA: se 2 programas acertam parcial, tenta f2(f1(g)) e f1(f2(g))
  - BIBLIOTECA de estrategias que funcionaram (injetada em tasks futuras = acumulo de breadth)
  - A/B de TRANSFERENCIA: modo 'quente' (acumula) vs 'frio' (zera por task)

Autosave incremental. Mede solve-rate e se a 2a metade supera a 1a (transferencia).
"""
import sys, os, json, time, random, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from solver import (PERSONAS, make_prompt, extract_code, compile_transform,
                    verify_on_train, apply_to_test, test_correct, ARC_DIR, OUT_DIR, ask)

def llm_program(task, persona, library, temperature):
    sysp, user = make_prompt(task, persona, library)
    try:
        raw = ask([{"role":"user","content":user}], system=sysp, temperature=temperature, max_tokens=1100)
    except Exception as e:
        return None, None, f"llm_err:{e}"
    code = extract_code(raw)
    fn, err = compile_transform(code)
    if err: return None, code, err
    return fn, code, None

def sample_persona(drives):
    keys = list(drives); w = [max(0.05, drives[k]) for k in keys]
    tot = sum(w); r = random.random()*tot; acc = 0
    for k, wi in zip(keys, w):
        acc += wi
        if r <= acc: return k
    return keys[-1]

def solve_task(task, drives, library, attempts_per_task, hot, emit):
    """tenta resolver 1 task. retorna dict com resultado."""
    tried_codes = []
    partial_fns = []   # (fn, persona, n_acertos) de programas que acertaram >0 mas nao tudo
    used_personas = []
    for attempt in range(attempts_per_task):
        persona = sample_persona(drives)
        used_personas.append(persona)
        temp = 0.2 + 0.25*attempt    # esquenta a cada tentativa (mais diversidade)
        fn, code, err = llm_program(task, persona, library if hot else [], temp)
        if fn is None:
            emit(f"    [{persona}] falha: {str(err)[:60]}")
            continue
        solved, ok, total = verify_on_train(fn, task)
        emit(f"    [{persona}] treino {ok}/{total}" + ("  <<< RESOLVEU" if solved else ""))
        if solved:
            preds = apply_to_test(fn, task)
            return {"solved_train": True, "fn_code": code, "persona": persona,
                    "preds": preds, "attempts": attempt+1, "used_personas": used_personas}
        if ok > 0:
            partial_fns.append((fn, code, persona, ok))
        tried_codes.append(code)

    # COMPOSICAO NAO-COMUTATIVA: combina os 2 melhores parciais nas 2 ordens
    if len(partial_fns) >= 2:
        partial_fns.sort(key=lambda x: -x[3])
        (f1, c1, p1, _), (f2, c2, p2, _) = partial_fns[0], partial_fns[1]
        for (fa, fb, order) in [(f1, f2, "A>>B"), (f2, f1, "B>>A")]:
            def composed(g, fa=fa, fb=fb):
                import numpy as np
                r1 = fa([list(r) for r in g])
                if hasattr(r1, "tolist"): r1 = r1.tolist()
                return fb([list(r) for r in r1])
            solved, ok, total = verify_on_train(composed, task)
            emit(f"    [compose {order}] treino {ok}/{total}" + ("  <<< RESOLVEU" if solved else ""))
            if solved:
                preds = apply_to_test(composed, task)
                return {"solved_train": True, "fn_code": f"# COMPOSED {order}\n{c1}\n{c2}",
                        "persona": "composto", "preds": preds, "attempts": attempts_per_task,
                        "used_personas": used_personas, "compose_order": order}
    return {"solved_train": False, "used_personas": used_personas}

def run(n_tasks=40, attempts=3, hot=True, seed=0, tag="hot", task_ids=None):
    random.seed(seed)
    all_tasks = sorted(p.name for p in ARC_DIR.glob("*.json"))
    if task_ids is None:
        random.Random(seed).shuffle(all_tasks)
        task_ids = all_tasks[:n_tasks]
    drives = {k: 1.0 for k in PERSONAS}
    library = []   # descricoes/codigos de estrategias que funcionaram
    results = []
    run_name = f"arc_{tag}_{int(time.time())}"
    t0 = time.time()
    def emit(s): print(s, flush=True)
    emit(f"=== ARC-NOVAMIND [{tag}] {'QUENTE(acumula)' if hot else 'FRIO(zera)'} | {len(task_ids)} tasks | {attempts} tentativas/task ===")
    solved_ids = []
    for i, tid in enumerate(task_ids, 1):
        task = json.load(open(ARC_DIR / tid))
        emit(f"\n[{i}/{len(task_ids)}] {tid}  (train pairs={len(task['train'])})  drives={ {k:round(v,2) for k,v in drives.items()} }")
        r = solve_task(task, drives, library, attempts, hot, emit)
        r["task"] = tid; r["index"] = i
        # verifica no test (training set tem gabarito)
        if r.get("solved_train"):
            correct = test_correct(r.get("preds"), task)
            r["test_correct"] = correct
            persona = r["persona"]
            # DRIVE darwiniano: persona vencedora ganha peso
            if hot:
                drives[persona if persona in drives else "geometra"] = drives.get(persona, 1.0) + 0.6
                tot = sum(drives.values())
                for k in drives: drives[k] = drives[k]/tot*len(drives)
                # BIBLIOTECA: registra a ideia que funcionou (1a linha de comentario ou resumo do codigo)
                desc = f"task {tid}: persona {persona}" + (f" (compose {r['compose_order']})" if r.get("compose_order") else "")
                library.append(desc)
            if correct: solved_ids.append(tid)
            emit(f"  => RESOLVEU TREINO via {persona}. TEST correto: {correct}")
        else:
            emit(f"  => nao resolvido")
        results.append(r)
        # autosave incremental
        payload = {"tag":tag,"hot":hot,"n_tasks":len(task_ids),"attempts":attempts,
                   "duration_s":time.time()-t0,"drives":drives,"library_size":len(library),
                   "solved_train":[x["task"] for x in results if x.get("solved_train")],
                   "solved_test":solved_ids,"results":results,"task_ids":task_ids}
        json.dump(payload, open(OUT_DIR / f"{run_name}.json","w"), indent=1)

    n = len(task_ids)
    st = len([x for x in results if x.get("solved_train")])
    sc = len(solved_ids)
    # TRANSFERENCIA: solve-rate 1a metade vs 2a metade
    half = n//2
    first = [x for x in results[:half] if x.get("test_correct")]
    second = [x for x in results[half:] if x.get("test_correct")]
    emit(f"\n{'='*60}")
    emit(f"[{tag}] RESOLVEU TREINO: {st}/{n} = {st/n:.1%} | TEST correto: {sc}/{n} = {sc/n:.1%}")
    emit(f"[{tag}] TRANSFERENCIA: 1a metade {len(first)}/{half} vs 2a metade {len(second)}/{n-half}")
    emit(f"[{tag}] drives finais: { {k:round(v,2) for k,v in drives.items()} }")
    payload["final"] = {"solved_train":st,"solved_test":sc,"rate_test":sc/n,
                        "first_half":len(first),"second_half":len(second),"half":half}
    json.dump(payload, open(OUT_DIR / f"{run_name}.json","w"), indent=1)
    emit(f"salvo: {OUT_DIR / (run_name+'.json')}")
    return payload

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--cold", action="store_true", help="modo frio (sem acumulo) p/ A/B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    hot = not a.cold
    tag = a.tag or ("hot" if hot else "cold")
    run(n_tasks=a.n, attempts=a.attempts, hot=hot, seed=a.seed, tag=tag)
