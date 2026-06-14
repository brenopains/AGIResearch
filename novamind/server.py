"""
NovaMind — HTTP API Server
"""

import os
import json
import time
import threading
import glob
import re
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import torch
if not torch.cuda.is_available():
    print("[WARNING] GPU não encontrada ou CUDA não instalado. Rodando em CPU mode (Vai ser mais lento, mas funciona).")
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    print(f"[INIT] Processamento estritamente na GPU: {torch.cuda.get_device_name(0)}")

from core.mind import NovaMind

# Initialize the Mind
mind = NovaMind()

# Flask app
app = Flask(__name__, static_folder='web', static_url_path='')
CORS(app)

# Dreaming State
is_dreaming = False
dream_thread = None

def dream_loop():
    global is_dreaming
    
    # Extract sentences from training data
    sentences = []
    for filepath in glob.glob(os.path.join("data", "train", "*.txt")):
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                # Simple sentence split
                raw_sentences = re.split(r'[.!?]+', text)
                for s in raw_sentences:
                    s = s.strip()
                    if len(s.split()) >= 3: # Only feed reasonable chunks
                        sentences.append(s)
        except Exception as e:
            print(f"[DREAM] Erro ao ler {filepath}: {e}")
            
    if not sentences:
        print("[DREAM] Nenhum texto encontrado em data/train/. Coloque um .txt lá.")
        is_dreaming = False
        return

    print(f"[DREAM] Iniciando sonho profundo com {len(sentences)} sequências de memória...")
    
    # Loop over sentences while dreaming
    idx = 0
    while is_dreaming:
        # Pega a frase, usa ate 20 palavras pra treinar bigrams reais
        sentence_words = sentences[idx % len(sentences)].split()[:20]
        sentence = " ".join(sentence_words)
        idx += 1
        
        try:
            # The mind processes the input autonomously
            mind.think(sentence)
            
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"[DREAM] Erro cognitivo durante o sonho: {e}")
            
        # Pausa curta - o batch training na GPU e rapido agora
        for _ in range(4):
            if not is_dreaming:
                break
            time.sleep(0.25)

    print("[DREAM] Córtex retornou ao estado de vigília (Dreaming pausado).")


@app.route('/')
def index():
    return send_from_directory('web', 'index.html')


@app.route('/api/think', methods=['POST'])
def think():
    """Submit input to the mind and get a complete thought trace."""
    data = request.get_json()
    raw_input = data.get('input', '')
    if not raw_input:
        return jsonify({'error': 'No input provided'}), 400

    thought = mind.think(raw_input)
    return jsonify(thought)

@app.route('/api/dream', methods=['POST'])
def toggle_dream():
    """Toggle background autonomous reading/training mode."""
    global is_dreaming, dream_thread
    
    data = request.get_json() or {}
    enable = data.get('enable', not is_dreaming)
    
    if enable and not is_dreaming:
        is_dreaming = True
        dream_thread = threading.Thread(target=dream_loop, daemon=True)
        dream_thread.start()
        return jsonify({'status': 'dreaming', 'message': 'Área de Broca iniciou consolidação autônoma de linguagem.'})
    elif not enable and is_dreaming:
        is_dreaming = False
        return jsonify({'status': 'awake', 'message': 'Córtex despertou.'})
        
    return jsonify({'status': 'dreaming' if is_dreaming else 'awake'})

@app.route('/api/state', methods=['GET'])
def get_state():
    """Get the complete mind state."""
    state = mind.get_full_state()
    return jsonify(state)


@app.route('/api/graph', methods=['GET'])
def get_graph():
    """Get the knowledge graph data."""
    return jsonify(mind.world_model.get_graph_data())


@app.route('/api/memory', methods=['GET'])
def get_memory():
    """Get memory contents and stats."""
    return jsonify({
        'stats': mind.memory.get_stats(),
        'contents': mind.memory.get_all_memories(),
    })


@app.route('/api/metacognition', methods=['GET'])
def get_metacognition():
    """Get metacognitive state."""
    return jsonify(mind.metacognition.get_full_state())


@app.route('/api/goals', methods=['GET'])
def get_goals():
    """Get goal system state."""
    return jsonify(mind.goals.get_state())


@app.route('/api/learning', methods=['GET'])
def get_learning():
    """Get learning statistics."""
    return jsonify(mind.learning.get_stats())


@app.route('/api/concepts', methods=['GET'])
def get_concepts():
    """Get all known concepts."""
    return jsonify(mind.perception.get_all_concepts())


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  >> NovaMind — 7-Layer AGI Architecture Prototype")
    print("="*60)
    print(f"  Dashboard: http://localhost:5000")
    print(f"  API:       http://localhost:5000/api/")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
