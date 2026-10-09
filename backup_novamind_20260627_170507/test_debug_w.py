"""Debug W matrix decode"""
import sys, os, warnings, torch
sys.path.insert(0, ('C:\\Users\\usuario\\Desktop\\AGI 2\\novamind').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~')))
warnings.filterwarnings('ignore')

from core.mind import NovaMind
from core.broca_decoder import BrocaDecoder
from core.positional_binder import PositionalBinder
from core.coglang import Manifold

mind = NovaMind()
engine = mind.hdc_memory.engine
decoder = BrocaDecoder(engine)
all_idx = []
for v in mind.hdc_memory.indices.values():
    all_idx.extend(v)
decoder.build_vocabulary_from_metadata(all_idx)
decoder.min_word_len = 0
binder = PositionalBinder(max_length=30, device=engine.device)

W = torch.load(r'D:\novamind_data\transition_matrix.pt', map_location='cuda', weights_only=False)
W = W['W'].float()

# Simula estado como no generator
target_mf = binder.encode_sequence(['once', 'upon', 'time'], engine)
working_memory = Manifold(torch.ones(1, 10000, device='cuda'))
state = Manifold(torch.sign((working_memory + target_mf).tensor), device='cuda')

# W matrix prediction
raw = W @ state.tensor
p = Manifold(torch.sign(raw), device='cuda')

# Decode
words = binder.decode_sequence(p, decoder.vocab_matrix, decoder.word_list, min_sim=0.08, max_words=4)
print(f'W decode (min_sim=0.08): {words}')

words2 = binder.decode_sequence(p, decoder.vocab_matrix, decoder.word_list, min_sim=0.01, max_words=4)
print(f'W decode (min_sim=0.01): {words2}')
