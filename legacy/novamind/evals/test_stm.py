"""Teste: SpatialTemporalMemory com objetos sinteticos em movimento"""
import sys, os, time, numpy as np
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)

from core.spatial_temporal_memory import SpatialTemporalMemory
from core.typed_graph import TypedCognitiveGraph

# Create TCG and STM
tcg = TypedCognitiveGraph(dim=20, curvature=1.0)
stm = SpatialTemporalMemory(tcg=tcg)

print("=" * 70)
print("STM Test: Synthetic moving objects")
print("=" * 70)

# Generate synthetic frames: 2 moving circles
H, W = 240, 320
frames = []
for t in range(60):
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Object 1: horizontal line (simulates a moving bar)
    x1 = int(30 + t * 3)  # moving right
    y1 = 120
    frame[max(0, y1-10):min(H, y1+10), max(0, x1-5):min(W, x1+5)] = [0, 255, 0]
    
    # Object 2: vertical bar moving down
    x2 = 160
    y2 = int(20 + t * 2.5)  # moving down
    frame[max(0, y2-5):min(H, y2+5), max(0, x2-10):min(W, x2+10)] = [255, 0, 0]
    
    frames.append(frame)

print(f"Generated {len(frames)} synthetic frames")

# Process through STM
for i, frame in enumerate(frames):
    result = stm.process_frame(frame, timestamp=i * 0.033)  # ~30fps

    if i == 10:
        print(f"\nFrame 10 state:")
        print(f"  Tracked objects: {result['n_objects']}")
        for obj_id in result['objects']:
            o = stm.objects[obj_id]
            print(f"    {obj_id}: centroid={o.centroid()}, "
                  f"v={o.velocity}, speed={o.speed():.1f}")

    if i == 30:
        print(f"\nFrame 30 state:")
        print(f"  Spatial relations:")
        for f, t, r, c in stm._spatial_relations[:10]:
            print(f"    {f} --[{r}]--> {t} (c={c:.2f})")
        print(f"  Events:")
        for ev in stm._events[-5:]:
            print(f"    [{ev['type']}] {ev}")

    if i == 59:
        print(f"\nFinal frame state:")
        print(f"  Objects tracked: {result['n_objects']}")
        print(f"  Total spatial relations: {result['spatial_relations']}")
        print(f"  Events: {result['events']}")
        
        print(f"\n  TCG concepts: {len(tcg.nodes)}")
        print(f"  TCG edges: {len(tcg.edges)}")
        # Show spatial edges in TCG
        spatial_edges = [(k, e) for k, e in tcg.edges.items() 
                        if k[2] in ('LEFT_OF', 'RIGHT_OF', 'ABOVE', 'BELOW')]
        print(f"  Spatial edges in TCG: {len(spatial_edges)}")
        for k, e in spatial_edges[:10]:
            print(f"    {k[0]} --[{k[2]}]--> {k[1]} (conf={e['confidence']:.3f})")
        
        # Test prediction
        for oid, obj in stm.objects.items():
            future = obj.predict_position(frames_ahead=10)
            print(f"\n  Prediction: {oid} in 10 frames: {future} "
                  f"(current: {obj.centroid()})")

print(f"\n{'='*70}")
print(f"STM test complete: {len(tcg.nodes)} TCG concepts, {len(tcg.edges)} edges")
print("=" * 70)
