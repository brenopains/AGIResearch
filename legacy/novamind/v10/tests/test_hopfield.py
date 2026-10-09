import torch
import pytest
from novamind.v10.models.hopfield import HopfieldMemory

def test_hopfield_retrieval():
    embed_dim = 256
    memory = HopfieldMemory(embed_dim=embed_dim)
    
    # Store explicit patterns
    pattern1 = torch.randn(embed_dim)
    pattern2 = torch.randn(embed_dim)
    
    memory.store(pattern1)
    memory.store(pattern2)
    
    assert memory.memory_store.shape[0] == 2
    
    # Query with a noisy pattern1
    noisy_query = pattern1 + torch.randn(embed_dim) * 0.1
    
    # Acceptance criteria: Retrieval should be closer to pattern1 than pattern2
    retrieved = memory.retrieve(noisy_query.unsqueeze(0)) # [1, 256]
    
    dist_to_1 = torch.norm(retrieved.squeeze(0) - pattern1)
    dist_to_2 = torch.norm(retrieved.squeeze(0) - pattern2)
    
    assert dist_to_1 < dist_to_2, "Hopfield network failed to retrieve the closest associative memory"

def test_hopfield_capacity():
    memory = HopfieldMemory(embed_dim=32, max_capacity=100)
    
    # Store exactly 100 items
    memory.store(torch.randn(100, 32))
    assert memory.count == 100
    
    # Store 10 more, pushing out old ones if FIFO
    memory.store(torch.randn(10, 32))
    assert len(memory.memory_store) == 100
    assert memory.count == 110 # lifetime count
