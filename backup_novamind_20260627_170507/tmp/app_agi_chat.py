from flask import Flask, render_template_string, request, jsonify
import torch
import pickle
import os
from exp_h118_bilingual_agi import BilingualAGI, device

app = Flask(__name__)

# Carregar o Modelo
model_loaded = False
try:
    if os.path.exists("checkpoints/brain_v1.pt"):
        with open("checkpoints/vocab.pkl", "rb") as f:
            stoi, itos, chars = pickle.load(f)

        vocab_size = len(chars)
        encode = lambda s: [stoi[c] for c in s if c in stoi]
        decode = lambda l: ''.join([itos[i] for i in l])

        model = BilingualAGI()
        model.load_state_dict(torch.load("checkpoints/brain_v1.pt"))
        model.to(device)
        model.eval()
        model_loaded = True
        print("Cortex AGI Carregado com Sucesso no Servidor Web!")
except Exception as e:
    print(f"Erro ao carregar modelo: {e}")

HTML_PAGE = """
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <title>Novamind AGI - Córtex Interativo</title>
    <style>
        body { font-family: 'Courier New', Courier, monospace; background-color: #0a0a0a; color: #00ff00; margin: 0; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; border: 1px solid #333; padding: 20px; border-radius: 10px; background-color: #111; box-shadow: 0 0 20px rgba(0,255,0,0.1); }
        h1 { text-align: center; border-bottom: 1px solid #333; padding-bottom: 10px; }
        #chatbox { height: 400px; overflow-y: auto; background-color: #000; padding: 10px; border: 1px solid #333; margin-bottom: 20px; border-radius: 5px; }
        .msg-humano { color: #00ffff; margin-bottom: 10px; }
        .msg-agi { color: #00ff00; margin-bottom: 20px; font-weight: bold; }
        input[type="text"] { width: 80%; padding: 10px; background-color: #222; color: #00ff00; border: 1px solid #00ff00; border-radius: 5px; outline: none; }
        button { width: 15%; padding: 10px; background-color: #00ff00; color: #000; border: none; font-weight: bold; cursor: pointer; border-radius: 5px; }
        button:hover { background-color: #00cc00; }
        /* Barra de rolagem customizada */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #111; }
        ::-webkit-scrollbar-thumb { background: #00ff00; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>NOVAMIND AGI v1.0</h1>
        <div id="chatbox">
            <div class="msg-agi">SISTEMA ONLINE. Pronta para receber raciocínios.</div>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <input type="text" id="userInput" placeholder="Faça uma pergunta lógica ou de física..." onkeypress="if(event.key === 'Enter') sendMessage()">
            <button onclick="sendMessage()">ENVIAR</button>
        </div>
    </div>

    <script>
        function sendMessage() {
            let input = document.getElementById("userInput");
            let chatbox = document.getElementById("chatbox");
            let text = input.value;
            
            if (text.trim() === "") return;
            
            chatbox.innerHTML += `<div class="msg-humano">> Humano: ${text}</div>`;
            input.value = "";
            chatbox.scrollTop = chatbox.scrollHeight;

            fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: text })
            })
            .then(response => response.json())
            .then(data => {
                chatbox.innerHTML += `<div class="msg-agi">[AGI]: ${data.resposta}</div>`;
                chatbox.scrollTop = chatbox.scrollHeight;
            })
            .catch(err => {
                chatbox.innerHTML += `<div class="msg-agi" style="color:red;">[ERRO]: Falha na rede neural.</div>`;
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/api/chat', methods=['POST'])
def chat():
    if not model_loaded:
        return jsonify({"resposta": "ERRO FATAL: Cérebro não encontrado. Treinamento na RTX 3050 foi concluído?"})
    
    data = request.json
    prompt = data.get("prompt", "")
    
    try:
        context = torch.tensor((encode(prompt)), dtype=torch.long, device=device).unsqueeze(0)
        prediction = model.generate(context, max_new_tokens=150)
        resposta = decode(prediction[0].tolist())
        return jsonify({"resposta": resposta})
    except Exception as e:
        return jsonify({"resposta": f"Erro de Inferência: {str(e)}"})

if __name__ == '__main__':
    print("Iniciando Front-End Localhost do Córtex...")
    app.run(host='127.0.0.1', port=5000)
