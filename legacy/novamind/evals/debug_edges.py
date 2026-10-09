"""Debug edge format in saved state"""
import pickle, torch
state = pickle.load(open(r'save/agi_state_live.pkl', 'rb'))
edges = state['cge_edges']
print(f'Total edges: {len(edges)}')
print(f'Type of edges container: {type(edges).__name__}')

for i, e in enumerate(edges[:3]):
    print(f'\nEdge {i}: type={type(e).__name__}')
    if isinstance(e, list):
        print(f'  len={len(e)}')
        for j, v in enumerate(e):
            t = type(v).__name__
            if isinstance(v, torch.Tensor):
                print(f'  [{j}] Tensor shape={v.shape} dtype={v.dtype} device={v.device}')
                print(f'        mean={v.mean().item():.4f} min={v.min().item():.4f} max={v.max().item():.4f}')
            elif isinstance(v, str):
                print(f'  [{j}] str: "{v[:60]}"')
            else:
                print(f'  [{j}] {t}: {str(v)[:60]}')
    elif isinstance(e, dict):
        for k, v in e.items():
            t = type(v).__name__
            if isinstance(v, torch.Tensor):
                print(f'  {k}: Tensor shape={v.shape}')
            elif isinstance(v, str):
                print(f'  {k}: str "{v[:60]}"')
            else:
                print(f'  {k}: {t} = {str(v)[:60]}')
