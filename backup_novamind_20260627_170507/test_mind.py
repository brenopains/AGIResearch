import sys
print("=== NovaMind Test ===", flush=True)
try:
    from core.mind import NovaMind
    print("Import OK", flush=True)

    mind = NovaMind()
    print("Mind created", flush=True)

    result = mind.think("Hello, this is a test input about artificial intelligence and learning")
    print(f"Cycle: {result['cycle']}", flush=True)
    print(f"Response: {result['response']['text'][:300]}", flush=True)
    print(f"Concepts: {result['layers']['perception']['total_concepts_known']}", flush=True)
    print(f"Confidence: {result['layers']['metacognition']['confidence']['overall']:.2f}", flush=True)
    print(f"Cycle time: {result['cycle_time_ms']:.0f}ms", flush=True)

    print("--- Testing Cultural Rules ---", flush=True)
    cultural_result = mind.think("Água contaminada pode causar doença.")
    print(f"Cultural Result: {cultural_result['response']['text'][:300]}", flush=True)
    print(f"Facts extracted: {[f.get('display', '') for f in mind.semantic_facts]}")

    print("=== TEST PASSED ===", flush=True)
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}", flush=True)
    import traceback
    traceback.print_exc()
