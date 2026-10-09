import torch
import torch.nn as nn
import torch.nn.functional as F

class HopfieldMemory(nn.Module):
    """
    Modern Continuous Hopfield Network.
    Paper: "Hopfield Networks is All You Need" (Ramsauer et al. 2020).
    Acts as an exponential capacity associative memory where queries attract to stored states.
    """
    def __init__(self, embed_dim=256, max_capacity=16384, beta=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_capacity = max_capacity
        self.beta = beta if beta is not None else (1.0 / (embed_dim ** 0.5))
        
        # Buffer to hold raw memory embeddings
        self.register_buffer('memory_store', torch.empty(0, embed_dim))
        self.count = 0
        
    def store(self, patterns: torch.Tensor):
        """
        Stores temporal patterns into memory. Shape can be [D] or [B, D].
        Uses a simple FIFO replacement if max capacity is exceeded.
        """
        if patterns.dim() == 1:
            patterns = patterns.unsqueeze(0)
            
        with torch.no_grad():
            self.memory_store = torch.cat([self.memory_store, patterns], dim=0)
            
            # Trim if exceeded
            if self.memory_store.size(0) > self.max_capacity:
                self.memory_store = self.memory_store[-self.max_capacity:]
                
            self.count += patterns.size(0)

    def retrieve(self, queries: torch.Tensor):
        """
        Retrieves associative memory.
        queries: [Batch, embed_dim]
        Returns: [Batch, embed_dim]
        """
        if self.memory_store.size(0) == 0:
            return queries # No memory to retrieve
            
        # Attention logic: softmax(beta * Q * K^T) * V
        # Where K and V are the memory_store
        
        # Q * K^T: [Batch, embed_dim] x [embed_dim, NumMem] -> [Batch, NumMem]
        scores = torch.matmul(queries, self.memory_store.t())
        
        # Attn: [Batch, NumMem]
        attn = F.softmax(self.beta * scores, dim=-1)
        
        # Retrieve: [Batch, NumMem] x [NumMem, embed_dim] -> [Batch, embed_dim]
        retrieved = torch.matmul(attn, self.memory_store)
        
        return retrieved
