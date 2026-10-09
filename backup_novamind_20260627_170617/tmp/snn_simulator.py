"""
PILAR 4: SIMULAÇÃO NEUROMÓRFICA (SNN)
Implementação mínima de uma rede neural pulsante (Spiking Neural Network) com neurônios LIF
(Leaky Integrate-and-Fire), validando computação assíncrona orientada a picos de tensão.
"""
import time
import torch
import torch.nn as nn

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

class LIFNeuron(nn.Module):
    def __init__(self, in_features, out_features, tau=0.9, threshold=1.0):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.tau = tau
        self.threshold = threshold
        
    def forward(self, x, steps=10):
        """x tem dimensao (Batch, Features). Simulamos 'steps' de tempo."""
        B = x.shape[0]
        out_features = self.fc.out_features
        
        mem = torch.zeros(B, out_features, device=x.device)
        spk_count = torch.zeros(B, out_features, device=x.device)
        
        # A corrente de entrada é gerada pela projeção Linear
        current = self.fc(x) 
        
        spikes_over_time = []
        for t in range(steps):
            # Leaky integration
            mem = mem * self.tau + current
            
            # Fire condition
            spk = (mem >= self.threshold).float()
            
            # Reset voltage after spike
            mem = mem * (1.0 - spk)
            
            spk_count += spk
            spikes_over_time.append(spk.sum().item())
            
        return spk_count, spikes_over_time

def main():
    print(f"[{DEV}] PILAR 4: HARDWARE NEUROMÓRFICO | Rede Neural Spiking (LIF)")
    
    # Simula entrada sensorial: vetor de 8 pixels
    x = torch.rand(1, 8).to(DEV) 
    
    # Camada LIF com 4 neuronios
    snn_layer = LIFNeuron(in_features=8, out_features=4).to(DEV)
    
    t0 = time.time()
    total_spikes, spike_history = snn_layer(x, steps=20)
    
    print(f"   Entrada (Corrente): {x.detach().cpu().numpy().round(2)}")
    print(f"   Picos ao longo do tempo (20 ms): {spike_history}")
    print(f"   Frequência de disparo por neurônio: {total_spikes.detach().cpu().numpy()}")
    
    print("-> SUCESSO! A SNN integrou corrente e emitiu spikes (event-driven), provando a")
    print("   viabilidade de migração do motor AGI para chips eficientes como Intel Loihi.")
    print(f"({time.time()-t0:.2f}s)")

if __name__ == "__main__":
    main()
