import torch
import pickle
import os
from exp_h118_bilingual_agi import BilingualAGI, device

print("==== BENCHMARK AGI: TESTE ZERO-SHOT (H118) ====")

if not os.path.exists("checkpoints/brain_v1.pt"):
    print("ERRO: Checkpoint nao encontrado. Treine a maquina primeiro com 'python exp_h118_bilingual_agi.py'")
    exit()

# Carregar o Vocabulario
with open("checkpoints/vocab.pkl", "rb") as f:
    stoi, itos, chars = pickle.load(f)

vocab_size = len(chars)
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])

# Instanciar e Carregar o Cérebro
model = BilingualAGI()
model.load_state_dict(torch.load("checkpoints/brain_v1.pt"))
model.to(device)
model.eval()

print("Cérebro AGI carregado na memoria com sucesso.")

def test_prompt(text_prompt, max_tokens=100):
    context = torch.tensor((encode(text_prompt)), dtype=torch.long, device=device).unsqueeze(0)
    print(f"\n[PROMPT ZERO-SHOT]: {text_prompt}")
    # Gerando predicao sem nunca ter visto a resposta especifica
    prediction = model.generate(context, max_new_tokens=max_tokens)
    print(f"[RESPOSTA AGI]: {decode(prediction[0].tolist())}")

# 1. Teste de Generalização (Adivinhar continuação de um texto fora da curva)
test_prompt("O que e mais pesado, um kilo de")
test_prompt("Alice was beginning to get very")
test_prompt("Era uma vez um menino chamado Bras Cubas que")

# 2. Terminal Live Demo
print("\n" + "="*50)
print("INTERROGATORIO AO VIVO. Digite algo e pressione Enter. (Digite 'sair' para encerrar)")
print("="*50)
while True:
    try:
        user_input = input("Voce: ")
        if user_input.lower() in ["sair", "exit", "quit"]:
            break
        test_prompt(user_input, 200)
    except KeyboardInterrupt:
        break
