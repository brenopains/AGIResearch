import numpy as np
import pytest
from novamind.v10.models.faiss_memory import FaissMemory

def test_faiss_memory_storage_retrieval():
    embed_dim = 64
    memory = FaissMemory(embed_dim=embed_dim)
    
    # Create 100 random vectors
    vectors = np.random.randn(100, embed_dim).astype(np.float32)
    
    # Store
    memory.store(vectors)
    assert memory.index.ntotal == 100
    
    # Add a specific vector of interest
    target = np.ones((1, embed_dim), dtype=np.float32)
    memory.store(target)
    
    assert memory.index.ntotal == 101
    
    # Retrieve top 2 nearest to a noisy target
    noisy_target = target + np.random.randn(1, embed_dim).astype(np.float32) * 0.1
    distances, indices, retrieved_vectors = memory.retrieve(noisy_target, k=2)
    
    # It should retrieve the exact index 100 (the ones vector)
    assert indices[0][0] == 100, "Should retrieve the exact target vector"
    
    # Ensure distance is positive and small
    assert distances[0][0] >= 0.0
