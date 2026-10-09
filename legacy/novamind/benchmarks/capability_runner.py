"""Runner para benchmarks de capability (minutos)."""
import time
from typing import Dict, List, Optional


def run_arc(system_fn, tasks: int = 10) -> Dict:
    """ARC-AGI benchmark: resolucao de grids visuais."""
    start = time.time()
    passed = 0
    total = tasks
    results = []

    for i in range(tasks):
        t0 = time.time()
        result = system_fn(f"arc_task_{i}")
        elapsed = time.time() - t0
        ok = result.get("passed", False)
        if ok:
            passed += 1
        results.append({
            "task": i,
            "passed": ok,
            "confidence": result.get("confidence", 0),
            "elapsed": round(elapsed, 3),
        })

    return {
        "name": "arc",
        "passed": passed > total * 0.3,
        "metrics": {
            "accuracy": round(passed / total, 3),
            "total": total,
            "passed": passed,
            "elapsed_total": round(time.time() - start, 1),
        },
        "details": results,
    }


def run_vision(system_fn, samples: int = 10) -> Dict:
    """Benchmark de percepcao visual (DINOv2)."""
    start = time.time()
    passed = 0
    results = []

    for i in range(samples):
        t0 = time.time()
        result = system_fn(f"vision_sample_{i}")
        elapsed = time.time() - t0
        ok = result.get("fidelity", 0) > 0.3
        if ok:
            passed += 1
        results.append({
            "sample": i,
            "passed": ok,
            "fidelity": result.get("fidelity", 0),
            "psnr": result.get("psnr", 0),
            "elapsed": round(elapsed, 3),
        })

    return {
        "name": "vision",
        "passed": passed > samples * 0.3,
        "metrics": {
            "accuracy": round(passed / samples, 3),
            "total": samples,
            "passed": passed,
            "elapsed_total": round(time.time() - start, 1),
        },
        "details": results,
    }


def run_noise(system_fn) -> Dict:
    """Benchmark de ruido linguistico: variacoes da mesma pergunta."""
    queries = [
        "what is whale",
        "tell me about whales",
        "describe a whale",
        "whale information",
        "what are whales",
        "explain whale",
        "what is a whale",
    ]
    start = time.time()
    results = []
    main_answer = None
    consistent = 0

    for q in queries:
        t0 = time.time()
        result = system_fn(q)
        elapsed = time.time() - t0
        answer = result.get("answer", "")
        if main_answer is None:
            main_answer = answer
            consistent = 1
        elif answer == main_answer:
            consistent += 1
        results.append({
            "query": q,
            "answer_snippet": answer[:60],
            "confidence": result.get("confidence", 0),
            "elapsed": round(elapsed, 3),
        })

    consistency = round(consistent / len(queries), 3)

    return {
        "name": "noise_robustness",
        "passed": consistency > 0.4,
        "metrics": {
            "consistency": consistency,
            "queries": len(queries),
            "elapsed_total": round(time.time() - start, 1),
        },
        "details": results,
    }


def run_all_capability(system_fn) -> List[Dict]:
    return [
        run_arc(system_fn),
        run_vision(system_fn),
        run_noise(system_fn),
    ]
