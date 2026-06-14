import torch
ckpt = torch.load('checkpoints/substrate_distilled.pt', map_location='cpu', weights_only=True)
print('embeddings shape:', ckpt['embeddings'].shape)
t = ckpt['transition_weight']
if isinstance(t, dict):
    print('transition_weight is dict with keys:', list(t.keys()))
    for k,v in t.items():
        if hasattr(v, 'shape'):
            print(f'  transition.{k}: shape={v.shape}')
else:
    print(f'transition_weight shape: {t.shape}')
print('vocab_size:', ckpt['vocab_size'])
tokens = list(ckpt['token_to_id'].keys())
print('Total tokens:', len(tokens))
print('First 40 tokens:', tokens[:40])
full_words = [t for t in tokens if len(t) > 3]
print('Tokens > 3 chars:', len(full_words))
print('Sample:', full_words[:30])
