import torch
import torch.nn as nn
import torch.nn.functional as F

class RSSM(nn.Module):
    """
    Recurrent State Space Model (DreamerV3 style).
    Maintains a deterministic state h (GRU) and a stochastic state z (categorical).
    """
    def __init__(self, action_dim=10, embed_dim=256, stoch_dim=32, stoch_classes=32, deter_dim=512, hidden_dim=512):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.stoch_classes = stoch_classes
        self.deter_dim = deter_dim
        
        # 1. Update deterministic state (GRU cell substitute / basic RNN layer)
        self.cell = nn.GRUCell(self.stoch_dim * self.stoch_classes + action_dim, deter_dim)
        
        # 2. Prior dynamics: predict next stoch state given ONLY deter state
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, stoch_dim * stoch_classes)
        )
        
        # 3. Posterior dynamics: predict next stoch state given deter state AND observation
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, stoch_dim * stoch_classes)
        )

    def initial_state(self, batch_size, device='cpu'):
        return {
            'deter': torch.zeros(batch_size, self.deter_dim, device=device),
            'stoch': torch.zeros(batch_size, self.stoch_dim, self.stoch_classes, device=device),
            'logits': torch.zeros(batch_size, self.stoch_dim, self.stoch_classes, device=device)
        }

    def _sample(self, logits):
        """ Straight-through Gumbel-Softmax or simple argmax for sampling """
        # We use a simple straight-through sample (argmax during forward, softmax during backward)
        probs = F.softmax(logits, dim=-1)
        index = torch.multinomial(probs.view(-1, self.stoch_classes), 1)
        sample = torch.zeros_like(logits).view(-1, self.stoch_classes)
        sample.scatter_(1, index, 1.0)
        sample = sample.view(-1, self.stoch_dim, self.stoch_classes)
        # Straight through estimator
        sample = sample + probs - probs.detach()
        return sample

    def get_stoch_state(self, net_output):
        logits = net_output.view(-1, self.stoch_dim, self.stoch_classes)
        stoch = self._sample(logits)
        return {'logits': logits, 'stoch': stoch}

    def step(self, prev_state, prev_action, embed=None):
        """
        Advances the sequence by one step.
        If embed is None, acts as imagination (using prior).
        """
        flat_stoch = prev_state['stoch'].view(-1, self.stoch_dim * self.stoch_classes)
        
        # 1. Deterministic update (physics simulation)
        x = torch.cat([flat_stoch, prev_action], dim=-1)
        deter = self.cell(x, prev_state['deter'])
        
        # 2. Prior prediction (blind dreaming)
        prior = self.get_stoch_state(self.prior_net(deter))
        
        if embed is not None:
            # 3. Posterior update (waking state / observation correction)
            post_in = torch.cat([deter, embed], dim=-1)
            posterior = self.get_stoch_state(self.posterior_net(post_in))
        else:
            posterior = prior

        return prior, posterior, {'deter': deter, 'stoch': posterior['stoch'], 'logits': posterior['logits']}

    def imagine(self, start_state, action_seq):
        """ Unroll the model without observations (H-step rollout) """
        states = []
        priors = []
        curr_state = start_state
        
        # action_seq is [horizon, batch, action_dim]
        for t in range(action_seq.shape[0]):
            prior, _, curr_state = self.step(curr_state, action_seq[t], embed=None)
            states.append(curr_state)
            priors.append(prior)
            
        return states, priors
