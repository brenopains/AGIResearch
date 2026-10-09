import torch
import torch.nn as nn
from torch.distributions import Categorical

class PrimitiveBase:
    pass

class SymbolicHead(nn.Module):
    """
    Acts as the Recognition Model in DreamCoder.
    Maps continuous embeddings from the MoE / RSSM into probabilities over a DSL (Domain Specific Language) of primitives.
    """
    def __init__(self, embed_dim=256, primitives=None):
        super().__init__()
        if primitives is None:
            primitives = ['ADD', 'SUB', 'MUL', 'IF', 'FOLD']
            
        self.primitives = primitives
        self.vocab = self.primitives + ['<EOS>']
        self.vocab_size = len(self.vocab)
        
        # RNN to emit sequence of primitives conditioned on the embedding
        self.rnn = nn.GRUCell(embed_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, self.vocab_size)
        
        # Learnable start token
        self.start_token = nn.Parameter(torch.randn(1, embed_dim))
        
    def generate_program(self, latent_embedding: torch.Tensor, max_len=10):
        """
        Samples a symbolic sequence from the recognition model.
        latent_embedding: [Batch, embed_dim]
        """
        batch_size = latent_embedding.size(0)
        device = latent_embedding.device
        
        hidden = latent_embedding
        curr_input = self.start_token.expand(batch_size, -1)
        
        programs = [[] for _ in range(batch_size)]
        log_probs = torch.zeros(batch_size, device=device)
        
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        eos_idx = self.vocab.index('<EOS>')
        
        for _ in range(max_len):
            if not active_mask.any():
                break
                
            hidden = self.rnn(curr_input, hidden)
            logits = self.classifier(hidden)
            
            dist = Categorical(logits=logits)
            action = dist.sample()
            
            # Add to log probability for REINFORCE training later
            log_probs[active_mask] += dist.log_prob(action)[active_mask]
            
            for i in range(batch_size):
                if active_mask[i]:
                    token = self.vocab[action[i]]
                    programs[i].append(token)
                    if token == '<EOS>':
                        active_mask[i] = False
                        
            # Next input would actually embed the sampled token, but for simplified
            # DSL generation we feed the hidden state or embedding back.
            curr_input = hidden
            
        return programs, log_probs
