"""NovaMind Dashboard — localhost web interface."""
import sys, os, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
gc.collect(); torch.cuda.empty_cache()

from flask import Flask, render_template, request, jsonify
from core.mind import NovaMind

app = Flask(__name__)
_mind = None

def get_mind():
    global _mind
    if _mind is None:
        _mind = NovaMind()
    return _mind

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/think', methods=['POST'])
def api_think():
    mind = get_mind()
    data = request.get_json()
    text = data.get('text', '')
    if not text.strip():
        return jsonify({'error': 'empty input'})
    r = mind.think(text)
    layers = r['layers']
    meta_layers = layers.get('metacognition', {})
    perception = layers.get('perception', {})
    learning = layers.get('learning', {})

    recall = layers.get('memory', {}).get('recalled_episodes', [])
    recalled_texts = [ep.get('content', {}).get('input', '') for ep in recall if ep.get('content')]

    concepts = [c['label'] for c in perception.get('concepts', [])]
    is_novel = learning.get('novelty_analysis', {}).get('is_novel', False)

    # Handle both think-layer and full-state metacognition structures
    if 'confidence' in meta_layers and isinstance(meta_layers['confidence'], dict):
        conf = meta_layers['confidence'].get('overall', 0.5)
    elif 'confidence' in meta_layers and isinstance(meta_layers['confidence'], (int, float)):
        conf = meta_layers['confidence']
    else:
        conf = 0.5

    trace = {
        'concepts': concepts[:8],
        'confidence': f"{conf:.0%}",
        'novel': is_novel,
        'recall_count': len(recalled_texts),
        'recalled_texts': recalled_texts[:3],
    }

    return jsonify({
        'response': r['response']['text'],
        'trace': trace,
        'inferences': r.get('inferences', [])[:3],
        'goal_satisfied': r.get('goal_satisfied'),
        'contradiction': r.get('contradiction'),
        'cycle': r['cycle'],
        'cycle_time_ms': r.get('cycle_time_ms', 0),
    })

@app.route('/api/state')
def api_state():
    mind = get_mind()
    s = mind.get_full_state()

    # Metacognition from full state
    mc = s.get('metacognition', {}).get('current_state', {})
    emotion = mc.get('emotional_state', {})
    meta = {
        'confidence': mc.get('confidence', 0),
        'coherence': mc.get('coherence', 0),
        'surprise': mc.get('surprise_level', 0),
        'curiosity': emotion.get('curiosity', 0),
        'certainty': emotion.get('certainty', 0),
        'confusion': emotion.get('confusion', 0),
    }

    # Semantic rules
    sem_rules = []
    for rule in mind.memory.semantic:
        c = rule.content
        if isinstance(c, dict):
            sem_rules.append({
                'display': c.get('display', ''),
                'relation': c.get('relation', ''),
                'source': c.get('source_text', '')[:60],
                'confidence': getattr(rule, 'confidence', 1.0),
            })

    # Causal chains
    chains = mind.get_causal_chains_with_labels(min_length=2)

    # Recent episodes
    episodes = []
    for ep in mind.memory.episodic[-8:]:
        ec = ep.content or {}
        episodes.append({
            'input': ec.get('input', '')[:80],
            'type': ec.get('type', ''),
            'activation': getattr(ep, 'activation', 0),
        })

    # Goals — mind.goals.goals is a dict of Goal objects
    goals_dict = getattr(mind.goals, 'goals', {})
    active_count = sum(1 for g in goals_dict.values() if getattr(g, 'status', '') == 'active')
    total_goals = len(goals_dict)

    return jsonify({
        'cycle': s.get('cycle_count', 0),
        'uptime': s.get('uptime_seconds', 0),
        'metacognition': meta,
        'world_model': {
            'nodes': s.get('world_model', {}).get('node_count', 0),
            'edges': s.get('world_model', {}).get('edge_count', 0),
            'chains': chains[:10],
        },
        'memory': {
            'episodic_count': len(mind.memory.episodic),
            'semantic_count': len(mind.memory.semantic),
            'recent_episodes': episodes,
        },
        'semantic_rules': sem_rules[:20],
        'goals': {
            'active': active_count,
            'total': total_goals,
            'unanswered': len(getattr(mind, 'unanswered_questions', [])),
        },
        'concepts_known': s.get('perception', {}).get('total_concepts', 0),
        'system': {
            'vram_mb': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
            'vram_total': 4096,
        },
    })

if __name__ == '__main__':
    print(f"  NovaMind Dashboard → http://localhost:5001")
    app.run(host='localhost', port=5001, debug=False, threaded=False, use_reloader=False)
