import torch
import torch.nn as nn
import torch.optim as optim
import math
import os
import pickle

print("==== CÓRTEX BILINGUE DE ALTA ESCALA (H118) ====")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA Dispositivo: {device}")

# 1. CARREGAR DATASET (Corpus Mestre)
corpus_path = "dataset/corpus_mestre.txt"
if not os.path.exists(corpus_path):
    print("ERRO: Dataset nao encontrado. Rode 'python data_downloader.py' primeiro.")
    exit()

with open(corpus_path, "r", encoding="utf-8") as f:
    text = f.read()

print(f"Livro carregado. Total de Caracteres (Bytes): {len(text)}")

# 2. TOKENIZADOR CHARACTER-LEVEL (Sub-word puro, suporta PT e EN perfeitamente)
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulario de Caracteres (AGI Sub-word): {vocab_size}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Salvar o vocabulario para o Benchmark script
os.makedirs("checkpoints", exist_ok=True)
with open("checkpoints/vocab.pkl", "wb") as f:
    pickle.dump((stoi, itos, chars), f)

data = torch.tensor(encode(text), dtype=torch.long)

# 3. SPLIT DE TREINO E TESTE (10% reservado para provar Generalização)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Hiperparametros para a RTX 3050 (Ajuste para caber em 4GB VRAM)
block_size = 128    # Tamanho do contexto (Quantos caracteres ele lembra para tras)
batch_size = 64     # Quantidade de textos em paralelo
n_embd = 256        # Dimensoes do espaço latente
n_head = 8          # Quantidade de cabecas de atencao
n_layer = 4         # Profundidade do pensamento (Transformer layers)
dropout = 0.1
learning_rate = 3e-4

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# 4. ARQUITETURA DO GPT (Generative Pre-trained Transformer)
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = torch.nn.functional.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BilingualAGI(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = nn.functional.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            
            # HONESTIDADE INTELECTUAL (SEM HACKS DE LLM)
            # O cérebro biológico não rola dados com Top-K ou Temperature.
            # Ele executa o caminho neural de maior ativação (Convicção Absoluta).
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BilingualAGI().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 5. TREINAMENTO MASSIVO E CHECKPOINTS
if __name__ == "__main__":
    print("\nIniciando o Forno (Deep Learning). Pressione CTRL+C para parar e salvar a qualquer momento.")
    max_iters = 300
    eval_interval = 50

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(50)
            for k in range(50):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    try:
        for iter in range(max_iters):
            if iter % eval_interval == 0:
                losses = estimate_loss()
                print(f"Passo {iter} | Loss de Treino: {losses['train']:.4f} | Loss de Validação (O Benchmark): {losses['val']:.4f}")
                # Checkpoint (Salvar cerebro na maquina)
                torch.save(model.state_dict(), "checkpoints/brain_v1.pt")
            
            xb, yb = get_batch('train')
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
        print("\nTREINAMENTO CONCLUIDO!")
        torch.save(model.state_dict(), "checkpoints/brain_v1.pt")
    except KeyboardInterrupt:
        print("\nTreinamento Interrompido pelo Usuário. Salvando Checkpoint Atual...")
        torch.save(model.state_dict(), "checkpoints/brain_v1.pt")
        print("Salvo em 'checkpoints/brain_v1.pt'")
