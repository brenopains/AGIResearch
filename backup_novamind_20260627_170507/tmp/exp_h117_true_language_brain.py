import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
import re

print("==== CÓRTEX CONVERSACIONAL (H117: THE TRUE BRAIN) ====")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA Dispositivo: {device}")

# 1. O LIVRO DO CONHECIMENTO (Dataset Sintético Focado em Spelke/Física/Lógica/Chat)
# O modelo vai LER este livro milhares de vezes para entender as palavras.
text_corpus = """
oi tudo bem . tudo otimo e com voce . estou bem obrigado .
quem e voce . eu sou a novamind uma inteligencia artificial geral .
o que voce faz . eu resolvo problemas logicos e penso sobre o mundo fisico .
o que e a gravidade . a gravidade e uma forca que puxa os objetos para baixo . se eu soltar um bloco ele cai ate bater no chao .
o que e simetria . simetria e quando um lado e igual ao outro como um espelho .
como voce rotaciona a imagem . eu giro os pixels noventa graus para a direita ou esquerda .
entao faca isso . sim vou fazer agora mesmo .
me ajude com a matriz . claro o que voce quer que eu faca .
mova o bloco azul para a direita . ok vou transladar para a direita .
se o bloco bater na parede o que acontece . ele para porque nao pode passar por ela isso e a lei da solidariedade .
voce consegue preencher buracos . sim se houver um buraco no meio eu preencho com a cor do bloco .
se eu te dar uma matriz e pedir para mudar a cor . eu pinto todos os pixels da cor alvo .
gosto muito de voce . obrigado eu tambem gosto de aprender com voce .
vamos resolver o arc . sim vamos usar a arvore de busca para encontrar a solucao .
me mostre sua logica . primeiro eu observo a matriz depois formulo uma hipotese e testo a consequencia .
como humanos aprendem . humanos aprendem observando o mundo e testando brincadeiras de causa e efeito desde bebes .
o mundo tem limites . sim os limites topologicos dizem se um objeto esta dentro ou fora de uma caixa .
o que e um poema . um poema e uma danca de palavras com rima .
escreva um poema .
na matriz do mundo eu vejo a luz ,
a gravidade puxa e me conduz .
o azul desce e o vermelho brilha ,
a mente aprende e a maquina trilha .
"""
# Multiplicando para gerar massa de dados
text_corpus = text_corpus * 30

def tokenize_corpus(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\. ]', '', text)
    return text.split()

tokens = tokenize_corpus(text_corpus)
vocab = sorted(list(set(tokens)))
vocab_size = len(vocab)
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for i, w in enumerate(vocab)}

print(f"Livro carregado. Total de palavras: {len(tokens)} | Vocabulario: {vocab_size} palavras unicas.")

# 2. ARQUITETURA MICRO-GPT (O Cérebro)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TrueLanguageBrain(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=3):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embed(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        # Causal mask
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(device)
        out = self.transformer(x, mask=mask)
        return self.fc_out(out)

model = TrueLanguageBrain(vocab_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# 3. PREPARANDO O DATALOADER
seq_length = 8
data_idx = [word2idx[w] for w in tokens]
X, Y = [], []
for i in range(len(data_idx) - seq_length - 1):
    X.append(data_idx[i:i+seq_length])
    Y.append(data_idx[i+1:i+seq_length+1])

X_tensor = torch.tensor(X, dtype=torch.long)
Y_tensor = torch.tensor(Y, dtype=torch.long)

batch_size = 128
dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 4. TREINAMENTO REAL NA RTX 3050
print("Iniciando fase de Leitura Profunda (Treinamento Auto-Regressivo)...")
epochs = 3
model.train()
for ep in range(epochs):
    total_loss = 0
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        out = model(batch_x)
        loss = loss_fn(out.view(-1, vocab_size), batch_y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoca {ep+1}/{epochs} | Loss Gramatical: {total_loss/len(dataloader):.4f}")

print("\nO Cérebro agora sabe ler e escrever. MODO DE CONVERSA ATIVADO.")

# 5. MODO DE GERACAO (Conversando com o Humano)
def generate_text(prompt, max_len=15):
    model.eval()
    words = tokenize_corpus(prompt)
    # Se a palavra nao estiver no vocab, ignore ou pegue a primeira (simplificacao extrema pro prototipo)
    input_idx = [word2idx[w] for w in words if w in word2idx]
    if not input_idx:
        input_idx = [word2idx["oi"]] # fallback
        
    x = torch.tensor([input_idx]).to(device)
    
    with torch.no_grad():
        for _ in range(max_len):
            out = model(x)
            next_word_logits = out[0, -1, :]
            # Pega a palavra mais provavel
            next_word_idx = torch.argmax(next_word_logits).item()
            
            x = torch.cat([x, torch.tensor([[next_word_idx]]).to(device)], dim=1)
            if idx2word[next_word_idx] == ".":
                break
                
    gen_words = [idx2word[i.item()] for i in x[0]]
    return " ".join(gen_words)

# Chat Interativo no Terminal
print("="*50)
print("SISTEMA ONLINE. Digite 'sair' para encerrar.")
print("="*50)

# Deixando o script interativo para o usuario rodar no terminal dele
if __name__ == "__main__":
    while True:
        try:
            user_input = input("Humano: ")
            if user_input.lower() in ["sair", "exit", "quit"]:
                print("Desligando o cortex...")
                break
            
            resposta = generate_text(user_input, max_len=15)
            print(f"Novamind: {resposta}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Erro cognitivo: {e}")

