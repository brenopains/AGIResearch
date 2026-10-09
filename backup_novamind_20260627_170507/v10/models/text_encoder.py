import torch
import torch.nn as nn
import math

class TextEncoder(nn.Module):
    """
    Transformer-based text encoder with positional encoding.
    Converts token sequences into CONTEXTUALIZED semantic representations.
    
    Previously this was a bare nn.Embedding (lookup table) which had ZERO
    understanding of word order or context. Now it uses multi-head self-attention
    to understand relationships between words in a sentence.
    """
    def __init__(self, vocab_size: int = 16384, d_model: int = 256, num_heads: int = 8, num_layers: int = 4, max_seq_len: int = 512):
        super().__init__()
        self.d_model = d_model
        
        # Token embedding (same as before)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=0
        )
        
        # Learnable positional encoding — tells the model WHERE each word is in the sentence
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer self-attention layers — THIS is what was missing.
        # This is the same architecture used in GPT/BERT/Llama for understanding language.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            activation='gelu',
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Scale factor for stable training (Vaswani et al. 2017)
        self.scale = math.sqrt(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Maps a sequence of token IDs to contextualized dense vectors.
        
        Args:
            x: Tensor of shape [batch_size, seq_len] containing token ints.
            
        Returns:
            Tensor of shape [batch_size, seq_len, d_model] with context-aware embeddings.
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        
        # Token embedding + positional encoding
        emb = self.embedding(x) * self.scale + self.pos_embedding(positions)
        
        # Self-attention: each word attends to all other words in the sentence
        out = self.transformer(emb)
        return self.layer_norm(out)
