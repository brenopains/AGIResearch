"""Runner para benchmarks de regressao (<10s cada)."""
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Optional


HERE = Path(__file__).parent
REGRESSION_DIR = HERE / "regression"


def load_tests(name: str) -> List[Dict]:
    path = REGRESSION_DIR / name
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def run_grounding(system_fn) -> Dict:
    """Testa se o sistema consegue identificar relacoes semanticas basicas."""
    tests = load_tests("grounding.json")
    if not tests:
        return {"name": "grounding", "passed": False, "error": "grounding.json not found"}

    passed = 0
    results = []
    for t in tests:
        start = time.time()
        result = system_fn(t["query"])
        elapsed = time.time() - start

        ok = result.get("relation") == t["expected"] and result.get("confidence", 0) >= t["min_confidence"]
        if ok:
            passed += 1
        results.append({
            "query": t["query"],
            "expected": t["expected"],
            "got": result.get("relation"),
            "confidence": result.get("confidence", 0),
            "elapsed": round(elapsed, 3),
            "passed": ok,
        })

    return {
        "name": "grounding",
        "passed": passed == len(tests),
        "metrics": {"accuracy": round(passed / len(tests), 3), "total": len(tests), "passed": passed},
        "details": results,
    }


def run_inference(system_fn) -> Dict:
    """Testa capacidade de inferencia (A>B, B>C -> A>C)."""
    tests = load_tests("inference.json")
    if not tests:
        return {"name": "inference", "passed": False, "error": "inference.json not found"}

    passed = 0
    results = []
    for t in tests:
        start = time.time()
        result = system_fn(t["query"], context=t.get("premises", []))
        elapsed = time.time() - start

        inferred = result.get("relation") == t["expected"] if isinstance(t["expected"], str) else (
            result.get("confidence", 0) >= t["min_confidence"] if t["expected"] else result.get("confidence", 0) < t["min_confidence"]
        )
        if isinstance(t["expected"], bool):
            ok = (result.get("confidence", 0) >= t["min_confidence"]) == t["expected"]
        else:
            ok = inferred
        if ok:
            passed += 1
        results.append({**t, "got_confidence": result.get("confidence", 0), "elapsed": round(elapsed, 3), "passed": ok})

    return {
        "name": "inference",
        "passed": passed == len(tests),
        "metrics": {"accuracy": round(passed / len(tests), 3), "total": len(tests), "passed": passed},
        "details": results,
    }


def run_confidence_stability(system_fn) -> Dict:
    """Testa estabilidade: mesma query 10x, mede std da confianca."""
    config = load_tests("confidence.json")
    if isinstance(config, dict) and "runs" in config:
        runs = config["runs"]
        query = config["query"]
    else:
        runs = 10
        query = "whale is mammal"

    confidences = []
    results = []
    for i in range(runs):
        start = time.time()
        result = system_fn(query)
        elapsed = time.time() - start
        conf = result.get("confidence", 0)
        confidences.append(conf)
        results.append({"run": i, "confidence": conf, "elapsed": round(elapsed, 3)})

    mean_conf = sum(confidences) / len(confidences)
    variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)
    std_conf = variance ** 0.5

    max_std = config.get("max_confidence_std", 0.15) if isinstance(config, dict) else 0.15

    return {
        "name": "confidence_stability",
        "passed": std_conf <= max_std,
        "metrics": {
            "mean_confidence": round(mean_conf, 4),
            "std_confidence": round(std_conf, 4),
            "max_std_allowed": max_std,
            "runs": runs,
        },
        "details": results,
    }


def run_episodic(system_fn) -> Dict:
    """Testa com conceitos inventados (deve retornar baixa confianca)."""
    tests = load_tests("episodic.json")
    if not tests:
        return {"name": "episodic", "passed": False, "error": "episodic.json not found"}

    passed = 0
    results = []
    for t in tests:
        start = time.time()
        result = system_fn(t["query"])
        elapsed = time.time() - start

        conf = result.get("confidence", 0)
        ok = conf <= t["max_confidence"]
        if ok:
            passed += 1
        results.append({**t, "got_confidence": conf, "elapsed": round(elapsed, 3), "passed": ok})

    return {
        "name": "episodic",
        "passed": passed == len(tests),
        "metrics": {"accuracy": round(passed / len(tests), 3), "total": len(tests), "passed": passed},
        "details": results,
    }


def run_all(system_fn, include_capability: bool = False) -> List[Dict]:
    results = [
        run_grounding(system_fn),
        run_inference(system_fn),
        run_confidence_stability(system_fn),
        run_episodic(system_fn),
    ]

    if include_capability:
        try:
            from .capability_runner import run_all_capability
            results.extend(run_all_capability(system_fn))
        except ImportError:
            pass

    return results


def print_report(results: List[Dict]):
    total = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    print(f"\n{'='*60}")
    print(f"REGRESSION BENCHMARKS: {passed}/{total} passed")
    print(f"{'='*60}")
    for r in results:
        status = "PASS" if r.get("passed") else "FAIL"
        metrics = r.get("metrics", {})
        metrics_str = " | ".join(f"{k}={v}" for k, v in metrics.items()) if metrics else ""
        print(f"  [{status}] {r['name']}  {metrics_str}")
    print()
