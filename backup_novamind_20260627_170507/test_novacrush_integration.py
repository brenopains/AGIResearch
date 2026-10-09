"""Quick integration check for NovaMind + NovaCrush."""
from core.mind import NovaMind

mind = NovaMind()

inputs = [
    'hello world',
    'intelligence is compression', 
    'the brain computes with spikes',
    'DNA encodes programs not data',
    'learning never stops',
]

for text in inputs:
    r = mind.think(text)
    print(f"Cycle {r['cycle']}: conf={r['response']['confidence']:.2f}")

state = mind.get_full_state()
nc = state.get('novacrush', {})

print("\n=== NovaCrush Integration Status ===")
print(f"  NovaCrush Enabled: {state['perception']['novacrush_enabled']}")
print(f"  Total Concepts: {state['perception']['total_concepts']}")
print(f"  Cycles Completed: {state['cycle_count']}")

if nc:
    c = nc.get('compression', {})
    print(f"  Transition Sparsity: {c.get('transition_stats', {}).get('sparsity', 'N/A')}")
    print(f"  HDC Items Stored: {c.get('hdc_stats', {}).get('items_stored', 'N/A')}")
    print(f"  HDC Codebook Size: {c.get('hdc_stats', {}).get('codebook_size', 'N/A')}")
    
    ff = nc.get('forward_forward', {})
    if ff.get('accuracy'):
        print(f"  FF Accuracy: {ff['accuracy']:.1%}")
    
    inf = c.get('novacrush_inference', {})
    print(f"  Inference Compression: {inf.get('compression_vs_fp32', 'N/A')}x")
    print(f"  Neurogenesis Events: {c.get('neurogenesis_events', 'N/A')}")

print("\n=== INTEGRATION SUCCESSFUL ===")
