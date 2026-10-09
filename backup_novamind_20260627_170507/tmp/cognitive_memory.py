"""
PILAR 3: SISTEMA DE MEMÓRIA COGNITIVA
Separação entre Buffer de Curto Prazo e Banco de Longo Prazo (Vector DB) imitando a
consolidação hipocampal. O modelo guarda "experiências" e as recupera por similaridade.
"""
import time
import numpy as np
import torch
import torch.nn.functional as F

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

class CognitiveMemory:
    def __init__(self, emb_dim=64, st_capacity=5):
        self.emb_dim = emb_dim
        self.st_capacity = st_capacity
        self.short_term = [] # list of (embedding, data)
        
        # Long-term memory is a tensor of consolidated embeddings
        self.lt_embeddings = torch.empty((0, emb_dim), device=DEV)
        self.lt_data = []
        
    def observe(self, embedding, data):
        """Salva a experiência no buffer de curto prazo."""
        self.short_term.append((embedding, data))
        print(f"      [Curto-Prazo] Guardado: {data}")
        if len(self.short_term) >= self.st_capacity:
            self.consolidate()
            
    def consolidate(self):
        """Transfere do curto prazo para longo prazo (Vector DB) e limpa o buffer."""
        if not self.short_term: return
        
        embs = torch.stack([x[0] for x in self.short_term]).to(DEV)
        self.lt_embeddings = torch.cat([self.lt_embeddings, embs], dim=0)
        self.lt_data.extend([x[1] for x in self.short_term])
        
        print(f"      >> [Hipocampo] Consolidados {len(self.short_term)} memórias para Longo Prazo.")
        self.short_term.clear()
        
    def retrieve(self, query_emb, k=1):
        """Recupera do Banco de Longo Prazo usando Similaridade de Cosseno."""
        if self.lt_embeddings.shape[0] == 0:
            return None
            
        query_emb = query_emb.to(DEV)
        # Cosine similarity
        sims = F.cosine_similarity(query_emb.unsqueeze(0), self.lt_embeddings)
        best_idx = sims.argmax().item()
        
        return self.lt_data[best_idx], sims[best_idx].item()

def main():
    print(f"[{DEV}] PILAR 3: MEMÓRIA COGNITIVA | Gravacao, Consolidação e Recuperacao Vetorial")
    
    memory = CognitiveMemory(emb_dim=16, st_capacity=3)
    
    t0 = time.time()
    
    # Simula experiencias (cada uma geraria um embedding representativo na AGI)
    exp_list = [
        ("Agente encontrou maçã", torch.randn(16)),
        ("Agente abriu a porta", torch.randn(16)),
        ("Agente fugiu do fogo", torch.randn(16)), # Atinge capacidade e consolida
        ("Agente encontrou chave", torch.randn(16)),
        ("Agente dormiu", torch.randn(16))
    ]
    
    for txt, emb in exp_list:
        memory.observe(emb, txt)
        
    # Forçar a consolidação final
    memory.consolidate()
    
    print("-" * 50)
    print("-> Teste de Recuperação (Atenção Baseada em Conteúdo):")
    
    # Criamos uma query que é levemente alterada da experiência 3 (Fogo)
    query_emb = exp_list[2][1] + torch.randn(16) * 0.1 
    
    retrieved_data, sim_score = memory.retrieve(query_emb)
    
    print(f"   Query: Lembrar do 'Fogo'")
    print(f"   Recuperado: '{retrieved_data}' (Similaridade: {sim_score:.2f})")
    
    if "fogo" in retrieved_data:
        print("-> SUCESSO! Sistema recuperou com precisão o conhecimento do Longo Prazo sem retreinar pesos.")
    
    print(f"({time.time()-t0:.2f}s)")

if __name__ == "__main__":
    main()
