import pytest
import os
import random
from novamind.v10.tokenizer import Tokenizer
from datasets import load_dataset
import pickle

def test_vocab_size():
    tokenizer = Tokenizer()
    assert tokenizer.vocab_size == 16384, f"Expected exactly 16384, got {tokenizer.vocab_size}"

def test_special_tokens():
    tokenizer = Tokenizer()
    # Check if special tokens exist in vocabulary map
    assert tokenizer.sp.piece_to_id('<PAD>') == tokenizer.pad_id
    assert tokenizer.sp.piece_to_id('<BOS>') == tokenizer.bos_id
    assert tokenizer.sp.piece_to_id('<EOS>') == tokenizer.eos_id
    assert tokenizer.sp.piece_to_id('<UNK>') == tokenizer.unk_id

def test_picklable():
    tokenizer = Tokenizer()
    pickled = pickle.dumps(tokenizer)
    unpickled = pickle.loads(pickled)
    assert unpickled.vocab_size == 16384

def test_round_trip_preservation():
    tokenizer = Tokenizer()
    # Load a small batch from wikitext
    ds = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
    
    lines = []
    # Fetch 1000 non-empty random lines
    for item in ds:
        text = item['text'].strip()
        if len(text) > 20: 
            lines.append(text)
        if len(lines) >= 1000:
            break
            
    assert len(lines) == 1000
    
    # Test round trip
    passed = 0
    for line in lines:
        encoded = tokenizer.encode(line)
        decoded = tokenizer.decode(encoded)
        
        d_clean = ''.join(c for c in decoded if c.isalnum())
        l_clean = ''.join(c for c in line if c.isalnum())
        if d_clean == l_clean:
            passed += 1
            
    assert (passed / len(lines)) >= 0.99, f"Round trip preserved only {passed}/{len(lines)} lines"
