import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, x):
        return self.net(x)

class SparseMoE(nn.Module):
    """
    Sparse Mixture-of-Experts Layer.
    Uses Top-K routing to combine outputs from a subset of experts.
    """
    def __init__(self, d_model: int, num_experts: int = 64, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model
        
        self.experts = nn.ModuleList([Expert(d_model) for _ in range(num_experts)])
        self.router = nn.Linear(d_model, num_experts)
        
    def forward(self, x: torch.Tensor):
        original_shape = x.shape
        x_flat = x.view(-1, self.d_model) # [B*S, D]
        
        # 1. Routing
        router_logits = self.router(x_flat) # [B*S, num_experts]
        routing_probs = F.softmax(router_logits, dim=-1)
        
        # 2. Select top-k experts
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1) # [B*S, top_k]
        
        # Normalize top-k probabilities (so they sum to 1.0 roughly, or retain magnitude)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # 3. Output aggregation
        final_output = torch.zeros_like(x_flat)
        
        # In a real distributed MoE, we scatter to devices. 
        # Here we loop over the top_k indices for simplicity in formulation.
        for i in range(self.top_k):
            expert_indices = top_k_indices[:, i] # [B*S]
            expert_probs = top_k_probs[:, i].unsqueeze(1) # [B*S, 1]
            
            # Since experts are PyTorch modules, we have to process them. 
            # Vectorization across experts is generally done via torch.bmm or specialized kernels.
            # Local iteration:
            for expert_idx in range(self.num_experts):
                # Find which items in the batch use this expert
                mask = (expert_indices == expert_idx)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[expert_idx](expert_input)
                    # Add to final output weighted by routing probability
                    final_output[mask] += expert_output * expert_probs[mask]
                    
        # 4. Auxillary Loss for Load Balancing
        # Minimize variance of routing probs across batch to encourage using all experts
        # simple implementation: variance of mean probs
        mean_probs = routing_probs.mean(dim=0)
        aux_loss = torch.var(mean_probs)
        
        return final_output.view(original_shape), aux_loss
