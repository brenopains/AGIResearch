import faiss
import numpy as np

class FaissMemory:
    """
    Hypermassive memory wrapper relying on FAISS for ultra-fast, robust nearest neighbor clustering in CPU RAM/Disk.
    Used for retrieving relevant prior experiences during reasoning.
    """
    def __init__(self, embed_dim=256, use_hnsw=False):
        self.embed_dim = embed_dim
        
        if use_hnsw:
            # Hierarchical Navigable Small World Graph for billion-scale retrieval
            self.index = faiss.IndexHNSWFlat(embed_dim, 32)
        else:
            # Standard exact L2 search
            self.index = faiss.IndexFlatL2(embed_dim)
            
        # Optional metadata mapping could be implemented here
        self.vectors_stored = []
        
    def store(self, vectors: np.ndarray):
        """
        vectors: shape [num_items, embed_dim]
        """
        assert vectors.shape[1] == self.embed_dim
        # FAISS expects float32
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
            
        self.index.add(vectors)
        # In a real deployed version, we save vector to text/payload mappings
        self.vectors_stored.extend(list(vectors))
        
    def retrieve(self, queries: np.ndarray, k: int = 5):
        """
        queries: [batch, embed_dim]
        k: nearest neighbors
        """
        if self.index.ntotal == 0:
            return None, None, None
            
        if queries.dtype != np.float32:
            queries = queries.astype(np.float32)
            
        distances, indices = self.index.search(queries, k)
        
        # reconstruct vectors from stored base for utility
        retrieved_vectors = []
        for row in indices:
            row_items = []
            for idx in row:
                if idx != -1:
                    row_items.append(self.vectors_stored[idx])
                else:
                    row_items.append(np.zeros(self.embed_dim, dtype=np.float32))
            retrieved_vectors.append(row_items)
            
        return distances, indices, np.array(retrieved_vectors)
