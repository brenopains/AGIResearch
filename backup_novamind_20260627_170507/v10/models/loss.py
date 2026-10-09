import torch
import torch.nn.functional as F

def compute_expected_free_energy(reward_pred, prior_logits, posterior_logits, beta=1.0):
    """
    Computes the Expected Free Energy (EFE) which combines Pragmatic Value (Reward)
    and Epistemic Value / Risk (KL Divergence).
    
    Args:
        reward_pred: Estimated pragmatic rewards [Batch] or [Batch, Time]
        prior_logits: Logits from the prior dynamics model [Batch, Classes, Dims]
        posterior_logits: Logits from the posterior model [Batch, Classes, Dims]
        beta: Scaling factor for the information gain.
        
    Returns:
        Scalar EFE sum/mean.
    """
    # 1. Pragmatic Value: Maximize expected reward (minimize negative reward)
    pragmatic_value = -torch.mean(reward_pred)
    
    # 2. Epistemic Value / Risk: KL Divergence between Posterior and Prior
    # We want to minimize the divergence (prevent model from over-fitting posterior far from prior)
    # Both are categorical distributions: KL = sum(p * log(p / q))
    
    probs_post = F.softmax(posterior_logits, dim=-1)
    # log of posterior and prior
    log_post = F.log_softmax(posterior_logits, dim=-1)
    log_prior = F.log_softmax(prior_logits, dim=-1)
    
    # KL(post || prior)
    kl_div = torch.sum(probs_post * (log_post - log_prior), dim=-1)
    # Mean across categorical variables and batch
    kl_mean = torch.mean(kl_div)
    
    # EFE is bound: -Reward + \beta KL
    efe = pragmatic_value + beta * kl_mean
    
    return efe
