#!/usr/bin/env python3
"""Runner principal de benchmarks.

Uso:
    python benchmarks/run_benchmarks.py                    # regression only
    python benchmarks/run_benchmarks.py --capability       # regression + capability
    python benchmarks/run_benchmarks.py --arc              # ARC only
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.regression_runner import run_all as run_regression, print_report as print_regression
from benchmarks.capability_runner import run_all_capability, run_arc as run_arc_bench


def mock_system(query: str, context=None):
    """Mock que simula respostas para testar os benchmarks."""
    query_lower = query.lower()
    result = {"relation": "is_a", "confidence": 0.6, "answer": f"about {query}"}

    if "whale" in query_lower:
        result = {"relation": "is_a", "confidence": 0.72, "answer": "A whale is a mammal"}
    elif "dog" in query_lower:
        result = {"relation": "is_a", "confidence": 0.85, "answer": "A dog is an animal"}
    elif "cat" in query_lower:
        result = {"relation": "is_a", "confidence": 0.80, "answer": "A cat is an animal"}
    elif "sparrow" in query_lower:
        result = {"relation": "is_a", "confidence": 0.65, "answer": "A sparrow is a bird"}
    elif "rose" in query_lower:
        result = {"relation": "is_a", "confidence": 0.55, "answer": "A rose is a flower"}
    elif "water" in query_lower:
        result = {"relation": "has_property", "confidence": 0.50, "answer": "Water is liquid"}
    elif "fire" in query_lower:
        result = {"relation": "causes", "confidence": 0.45, "answer": "Fire causes heat"}
    elif "rain" in query_lower:
        result = {"relation": "has_property", "confidence": 0.40, "answer": "Rain is wet"}
    elif "sun" in query_lower:
        result = {"relation": "is_a", "confidence": 0.60, "answer": "The sun is a star"}
    elif "tree" in query_lower:
        result = {"relation": "is_a", "confidence": 0.50, "answer": "A tree is a plant"}
    elif "florp" in query_lower or "dax" in query_lower or "wug" in query_lower or "zorp" in query_lower or "blarf" in query_lower:
        result = {"relation": "unknown", "confidence": 0.05, "answer": "I don't know"}
    elif "arc_task" in query_lower:
        result = {"passed": True, "confidence": 0.6}
    elif "vision_sample" in query_lower:
        result = {"fidelity": 0.7, "psnr": 28.5}

    if context:
        pass  # inference context handling

    return result


def main():
    use_capability = "--capability" in sys.argv or "--arc" in sys.argv
    arc_only = "--arc" in sys.argv

    print("NovaMind Benchmark Runner")
    print("=" * 60)

    if arc_only:
        print("\n[ARC-AGI Benchmark]")
        result = run_arc_bench(mock_system, tasks=10)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  [{status}] {result['name']}: {result['metrics']}")
        return

    results = run_regression(mock_system, include_capability=use_capability)
    print_regression(results)

    if use_capability:
        print("\n[Capability Benchmarks]")
        cap_results = run_all_capability(mock_system)
        for r in cap_results:
            status = "PASS" if r["passed"] else "FAIL"
            print(f"  [{status}] {r['name']}: {r['metrics']}")


if __name__ == "__main__":
    main()
