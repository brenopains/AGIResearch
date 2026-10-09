"""Debug decode_sequence"""
import sys, os, warnings
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

vm = decoder.vocab_matrix
print(f'vocab_matrix shape: {vm.shape if vm is not None else "None"}')
print(f'word_list len: {len(decoder.word_list)}')
print(f'word_vectors: {len(decoder.word_vectors)}')

binder = PositionalBinder(max_length=30, device=engine.device)
target_mf = binder.encode_sequence(['once', 'upon', 'a', 'time'], engine)
print(f'target_mf shape: {target_mf.tensor.shape}')

words = binder.decode_sequence(target_mf, decoder.vocab_matrix, decoder.word_list, min_sim=0.08, max_words=4)
print(f'decode_sequence(min_sim=0.08): {words}')

words2 = binder.decode_sequence(target_mf, decoder.vocab_matrix, decoder.word_list, min_sim=0.001, max_words=4)
print(f'decode_sequence(min_sim=0.001): {words2}')
