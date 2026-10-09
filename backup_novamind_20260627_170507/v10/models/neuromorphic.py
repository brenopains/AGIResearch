import torch
import torch.nn as nn

class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) Spiking Neuron.
    Maintains a membrane potential that leaks over time. Decides to emit a binary
    Spike (1.0) when the voltage crosses the threshold threshold, resetting the voltage.
    """
    def __init__(self, action_dim=10, threshold=1.0, leak_decay=0.85):
        super().__init__()
        self.action_dim = action_dim
        self.threshold = threshold
        self.leak_decay = leak_decay
        self.register_buffer('membrane_potential', torch.zeros(1, action_dim))

    def reset_state(self, batch_size=1, device='cpu'):
        self.membrane_potential = torch.zeros(batch_size, self.action_dim, device=device)

    def forward(self, continuous_input):
        """
        Receives continuous input currents (e.g. from dense ActorCritic outputs)
        and integrates them into the membrane potential.
        """
        # Leaky Integration of current
        self.membrane_potential = (self.membrane_potential * self.leak_decay) + continuous_input
        
        # Fire mechanisms
        spikes = (self.membrane_potential >= self.threshold).float()
        
        # Reset voltage of fired neurons (Soft reset, subtracting threshold)
        self.membrane_potential = self.membrane_potential - (spikes * self.threshold)
        
        # Ensure voltage doesn't go extremely negative unconditionally
        self.membrane_potential = torch.clamp(self.membrane_potential, min=-0.5)
        
        return spikes, self.membrane_potential.clone()
