import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from scipy.ndimage import binary_fill_holes

app = Flask(__name__)
CORS(app)

print("==== INICIANDO CORTEX MESTRE INTERATIVO (H116) ====")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. NEURAL CORE (Language Grounding Dinâmico) ---
CONCEPT_DIM = 64
torch.manual_seed(42)

GOD_CLASS_DSL = {
    "Gravity": torch.randn(CONCEPT_DIM).to(device),
    "TranslateRight": torch.randn(CONCEPT_DIM).to(device),
    "TranslateLeft": torch.randn(CONCEPT_DIM).to(device),
    "TranslateUp": torch.randn(CONCEPT_DIM).to(device),
    "TranslateDown": torch.randn(CONCEPT_DIM).to(device),
    "ScaleUp": torch.randn(CONCEPT_DIM).to(device),
    "RecolorRed": torch.randn(CONCEPT_DIM).to(device),
    "MirrorX": torch.randn(CONCEPT_DIM).to(device),
    "ExtractIntersection": torch.randn(CONCEPT_DIM).to(device),
    "FillHole": torch.randn(CONCEPT_DIM).to(device),
    "DrawLine": torch.randn(CONCEPT_DIM).to(device)
}

# Hashing Vocab (O segredo para palavras infinitas OOV)
VOCAB_SIZE = 10000

def tokenize(sentence):
    # Processo de hashing permite que qualquer palavra nova receba um indice unico
    return [(hash(w.lower()) % (VOCAB_SIZE - 1)) + 1 for w in sentence.split()]

class GroundingEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64, concept_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, concept_dim)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded)
        h_last = hidden[-1]
        concept_vec = self.proj(h_last)
        return torch.nn.functional.normalize(concept_vec, p=2, dim=-1)

model = GroundingEncoder(VOCAB_SIZE, embed_dim=128, hidden_dim=64, concept_dim=CONCEPT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CosineEmbeddingLoss()

# Treino basal minúsculo so para alinhar os vetores iniciais
training_data = [
    ("faca os blocos cairem para baixo", "Gravity"),
    ("mova para a direita", "TranslateRight"),
    ("pinte tudo de vermelho", "RecolorRed")
]

print("Treinando instintos basais...")
model.train()
for ep in range(30):
    for text, concept_key in training_data:
        optimizer.zero_grad()
        target_vec = torch.nn.functional.normalize(GOD_CLASS_DSL[concept_key], p=2, dim=-1).unsqueeze(0)
        tokens = torch.tensor([tokenize(text)]).to(device)
        pred_vec = model(tokens)
        y = torch.tensor([1.0]).to(device)
        loss = loss_fn(pred_vec, target_vec, y)
        loss.backward()
        optimizer.step()
print("Rede Pronta para Ensino Online.")

def interpret_command(sentence):
    model.eval()
    tokens = torch.tensor([tokenize(sentence)]).to(device)
    with torch.no_grad():
        pred_vec = model(tokens)
        
    best_concept = None
    best_sim = -1.0
    
    for concept, c_vec in GOD_CLASS_DSL.items():
        c_vec_norm = torch.nn.functional.normalize(c_vec, p=2, dim=-1).unsqueeze(0)
        sim = torch.nn.functional.cosine_similarity(pred_vec, c_vec_norm).item()
        if sim > best_sim:
            best_sim = sim
            best_concept = concept
            
    # Garantir que a confianca reflita entre 0 e 100%
    confidence = max(0.0, sim)
    return best_concept, confidence

# --- 2. O MUNDO FÍSICO (Simulador ARC) ---
working_grid = np.zeros((10, 10), dtype=int)
working_grid[3:6, 3:6] = 1 
working_grid[4, 4] = 0
working_grid[1, 8] = 3

def apply_physical_rule(action_name, grid):
    new_g = grid.copy()
    H, W = new_g.shape
    
    if action_name == "RecolorRed":
        new_g[new_g != 0] = 2
    elif action_name == "TranslateRight":
        shifted = np.zeros_like(new_g)
        shifted[:, 1:] = new_g[:, :-1]
        new_g = shifted
    elif action_name == "TranslateLeft":
        shifted = np.zeros_like(new_g)
        shifted[:, :-1] = new_g[:, 1:]
        new_g = shifted
    elif action_name == "TranslateUp":
        shifted = np.zeros_like(new_g)
        shifted[:-1, :] = new_g[1:, :]
        new_g = shifted
    elif action_name == "TranslateDown":
        shifted = np.zeros_like(new_g)
        shifted[1:, :] = new_g[:-1, :]
        new_g = shifted
    elif action_name == "Gravity":
        for c in range(W):
            col = new_g[:, c]
            non_zeros = col[col != 0]
            col[:] = 0
            if len(non_zeros) > 0:
                col[H-len(non_zeros):] = non_zeros
            new_g[:, c] = col
    elif action_name == "ScaleUp":
        # Simula um fractal crescendo mas croppando pro tamanho original (simples)
        scaled = np.repeat(np.repeat(new_g, 2, axis=0), 2, axis=1)
        new_g = scaled[:H, :W]
    elif action_name == "MirrorX":
        new_g = np.flipud(new_g)
    elif action_name == "FillHole":
        for color in np.unique(new_g):
            if color == 0: continue
            mask = (new_g == color)
            filled = binary_fill_holes(mask)
            new_g[filled] = color
    return new_g

# --- 3. ENDPOINTS DA API ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/state", methods=["GET"])
def get_state():
    return jsonify({"grid": working_grid.tolist()})

@app.route("/api/interact", methods=["POST"])
def interact():
    data = request.json
    sentence = data.get("command", "")
    action, confidence = interpret_command(sentence)
    return jsonify({
        "action": action,
        "confidence": float(confidence),
    })

@app.route("/api/execute", methods=["POST"])
def execute_action():
    global working_grid
    data = request.json
    action = data.get("action")
    working_grid = apply_physical_rule(action, working_grid)
    return jsonify({"grid": working_grid.tolist()})

@app.route("/api/teach", methods=["POST"])
def teach():
    """Endpoint Mágico: Onde a AGI aprende uma nova sinapse ao vivo"""
    data = request.json
    sentence = data.get("command", "")
    correct_action = data.get("correct_action", "")
    
    if correct_action not in GOD_CLASS_DSL:
        return jsonify({"status": "error", "message": "Action not found"})
        
    model.train()
    optimizer.zero_grad()
    
    target_vec = torch.nn.functional.normalize(GOD_CLASS_DSL[correct_action], p=2, dim=-1).unsqueeze(0)
    tokens = torch.tensor([tokenize(sentence)]).to(device)
    pred_vec = model(tokens)
    y = torch.tensor([1.0]).to(device)
    
    loss = loss_fn(pred_vec, target_vec, y)
    loss.backward()
    
    # Da um passo otimizado mais forte para aprender rapido (One-shot learning)
    for _ in range(5):
        optimizer.step()
        optimizer.zero_grad()
        pred_vec = model(tokens)
        loss = loss_fn(pred_vec, target_vec, y)
        loss.backward()
        
    return jsonify({"status": "success", "message": f"Rede atualizada: '{sentence}' -> {correct_action}"})

@app.route("/api/reset", methods=["POST"])
def reset_grid():
    global working_grid
    working_grid = np.zeros((10, 10), dtype=int)
    working_grid[3:6, 3:6] = random.randint(1, 9)
    working_grid[4, 4] = 0
    working_grid[1, 8] = random.randint(1, 9)
    return jsonify({"grid": working_grid.tolist()})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
