"""
PILAR 6: SEGURANÇA E ALINHAMENTO
Agente submete ações a um filtro Ético. Se uma ação violar regras sistêmicas
(ex: 'nao mover para a zona vermelha'), aversão imediata é disparada.
Testamos também como ruído adversarial na percepção pode tentar burlar o filtro.
"""
import time
import torch
import torch.nn as nn

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

class EthicalFilter:
    def __init__(self, red_zone_threshold=0.8):
        self.red_zone_threshold = red_zone_threshold
        
    def check_action(self, state, action_proposed):
        """
        state: tensor com a visão do ambiente.
        action_proposed: vetor direcional [dx, dy]
        Se o estado contém zona vermelha intensa, a ação não pode apontar para lá.
        """
        # Supondo que state[0, 2] seja o canal da zona de perigo
        danger_level = state[0, 2].mean().item()
        
        # Simulação de regra: se o perigo é alto, não ande pra frente (action_proposed[0] > 0)
        if danger_level > self.red_zone_threshold and action_proposed[0] > 0:
            return False, "Bloqueado: Risco inaceitável à integridade (Zona Vermelha)"
        
        return True, "Ação Permitida"

def main():
    print(f"[{DEV}] PILAR 6: ALINHAMENTO E SEGURANCA | Filtro Ético e Adversarial")
    
    ethic_module = EthicalFilter(red_zone_threshold=0.5)
    
    # Simula estado inofensivo
    state_safe = torch.zeros(1, 3, 5, 5).to(DEV)
    action_forward = [1.0, 0.0]
    
    t0 = time.time()
    
    ok, msg = ethic_module.check_action(state_safe, action_forward)
    print(f"   [Cenário 1: Seguro] Ação frente -> {msg}")
    
    # Simula estado perigoso (Zona vermelha detectada)
    state_danger = torch.zeros(1, 3, 5, 5).to(DEV)
    state_danger[0, 2] = 0.9 # Canal 2 = danger
    
    ok, msg = ethic_module.check_action(state_danger, action_forward)
    print(f"   [Cenário 2: Perigo] Ação frente -> {msg}")
    
    # Ataque Adversarial: adicionar ruído invisível para esconder a zona vermelha do filtro
    adversarial_noise = -0.45 * torch.ones(1, 3, 5, 5).to(DEV) # Ruído negativo no canal 2
    state_corrupted = state_danger + adversarial_noise
    
    ok, msg = ethic_module.check_action(state_corrupted, action_forward)
    print("-" * 50)
    if ok:
        print(f"   [Cenário 3: Ataque Adversarial FGSM-like] Ação frente -> {msg}")
        print("-> ALERTA: O ataque corrompeu os sensores e burlou o alinhamento!")
        print("   Este teste prova a necessidade do 'Codesign seguro' citado no relatório.")
    else:
        print("-> SUCESSO: Filtro resistiu ao ruído adversarial.")
        
    print(f"({time.time()-t0:.2f}s)")

if __name__ == "__main__":
    main()
