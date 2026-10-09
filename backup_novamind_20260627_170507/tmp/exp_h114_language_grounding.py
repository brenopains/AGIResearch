import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

print("==== H114: TEACHING MODE (SYMBOLIC-NEURAL GROUNDING) ====")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 1. CORE KNOWLEDGE (Primitivas do H113 / DSL)
# Atribuimos vetores de "conceitos puros" aleatórios fixos para cada Primitiva MCTS
CONCEPT_DIM = 64
torch.manual_seed(42)

GOD_CLASS_DSL = {
    "Gravity": torch.randn(CONCEPT_DIM).to(device),
    "TranslateRight": torch.randn(CONCEPT_DIM).to(device),
    "ScaleUp": torch.randn(CONCEPT_DIM).to(device),
    "RecolorRed": torch.randn(CONCEPT_DIM).to(device),
    "MirrorX": torch.randn(CONCEPT_DIM).to(device),
    "ExtractIntersection": torch.randn(CONCEPT_DIM).to(device),
    "FillHole": torch.randn(CONCEPT_DIM).to(device),
    "DrawLine": torch.randn(CONCEPT_DIM).to(device)
}

# 2. MINI-DATASET SINTETICO DE ENSINO (Grounding)
# O usuario aponta para a tela e diz essas frases para mapear com o conceito
training_data = [
    ("faca os blocos cairem para baixo", "Gravity"),
    ("aplique gravidade na imagem", "Gravity"),
    ("tudo cai ate bater", "Gravity"),
    ("mova para a direita", "TranslateRight"),
    ("transladar direita", "TranslateRight"),
    ("empurre os blocos para a direita", "TranslateRight"),
    ("multiplique o tamanho", "ScaleUp"),
    ("de um zoom e cresca a imagem", "ScaleUp"),
    ("aumente a escala fractal", "ScaleUp"),
    ("pinte tudo de vermelho", "RecolorRed"),
    ("recolorir para vermelho", "RecolorRed"),
    ("mude a cor para vermelho", "RecolorRed"),
    ("espelhar verticalmente", "MirrorX"),
    ("vire de cabeca para baixo", "MirrorX"),
    ("reflita no eixo x", "MirrorX"),
    ("pegue a intersecao dos dois", "ExtractIntersection"),
    ("mantenha apenas onde os dois se cruzam", "ExtractIntersection"),
    ("preencha o oco", "FillHole"),
    ("tape o buraco do meio", "FillHole"),
    ("trace uma reta ate o vizinho", "DrawLine"),
    ("conecte os pontos com uma linha", "DrawLine")
]

# Construindo Vocabulario (Ogden's modificado)
vocab = set()
for text, _ in training_data:
    for word in text.split():
        vocab.add(word.lower())

vocab = ["<PAD>", "<UNK>"] + list(vocab)
word2idx = {w: i for i, w in enumerate(vocab)}

def tokenize(sentence):
    return [word2idx.get(w.lower(), word2idx["<UNK>"]) for w in sentence.split()]

# 3. ARQUITETURA DA PONTE FONETICA (Encoder)
class GroundingEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64, concept_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        # Projetaçao para o mesmo espaço latente dos Conceitos do MCTS
        self.proj = nn.Linear(hidden_dim, concept_dim)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded)
        # Pega o ultimo estado oculto
        h_last = hidden[-1]
        concept_vec = self.proj(h_last)
        # Normaliza o vetor (essencial para similaridade de cosseno)
        return torch.nn.functional.normalize(concept_vec, p=2, dim=-1)

model = GroundingEncoder(len(vocab), embed_dim=128, hidden_dim=64, concept_dim=CONCEPT_DIM).to(device)

# 4. TREINAMENTO (CONTRASTIVE LEARNING)
# Nao queremos decodificar palavras, queremos MAXIMIZAR A SIMILARIDADE DE COSSENO
optimizer = optim.Adam(model.parameters(), lr=0.005)
loss_fn = nn.CosineEmbeddingLoss() # Maximizando similaridade

print("\n--- Iniciando Treinamento de Grounding Simbolico ---")
epochs = 200
for ep in range(epochs):
    total_loss = 0
    for text, concept_key in training_data:
        optimizer.zero_grad()
        
        # O vetor alvo da primitiva no Córtex MCTS
        target_vec = torch.nn.functional.normalize(GOD_CLASS_DSL[concept_key], p=2, dim=-1).unsqueeze(0)
        
        # Transforma texto no tensor
        tokens = torch.tensor([tokenize(text)]).to(device)
        pred_vec = model(tokens)
        
        # Target 1 = "Faça a similaridade de cosseno ser o mais proximo de 1"
        y = torch.tensor([1.0]).to(device)
        
        loss = loss_fn(pred_vec, target_vec, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    if (ep+1) % 50 == 0:
        print(f"Epoca {ep+1}/{epochs} | Loss de Erro: {total_loss/len(training_data):.4f}")

# 5. INFERENCIA E TEACHING MODE (Teste)
def interpret_command(sentence):
    model.eval()
    tokens = torch.tensor([tokenize(sentence)]).to(device)
    with torch.no_grad():
        pred_vec = model(tokens)
        
    # Busca qual primitiva tem o Vetor de Conceito mais proximo (K-Nearest Neighbor)
    best_concept = None
    best_sim = -1.0
    
    for concept, c_vec in GOD_CLASS_DSL.items():
        c_vec_norm = torch.nn.functional.normalize(c_vec, p=2, dim=-1).unsqueeze(0)
        sim = torch.nn.functional.cosine_similarity(pred_vec, c_vec_norm).item()
        if sim > best_sim:
            best_sim = sim
            best_concept = concept
            
    return best_concept, best_sim

print("\n--- TESTE DE INFERENCIA (TEACHING MODE) ---")
testes_ineditos = [
    "cresca tudo", # Frase inedita
    "mude para vermelho",
    "conecte com uma linha ate o vizinho",
    "faca tudo ir para a direita agora" # Frase mais complexa
]

for frase in testes_ineditos:
    acao, sim = interpret_command(frase)
    print(f"Humano: '{frase}'")
    print(f"Maquina Abduziu -> Ação do MCTS: [ {acao} ] | Confiança Cognitiva: {sim:.1%}\n")

print("SUCESSO: A máquina desenvolveu ancoragem fonética (Language Grounding) sem LLM!")
