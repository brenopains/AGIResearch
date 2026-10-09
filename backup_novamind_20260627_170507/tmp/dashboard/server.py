"""NovaMind Dashboard — localhost web interface using only stdlib."""
import sys, os, gc, json, time, http.server, threading, urllib.parse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
gc.collect(); torch.cuda.empty_cache()

from core.mind import NovaMind

_mind = None
_lock = threading.Lock()

def get_mind():
    global _mind
    if _mind is None:
        with _lock:
            if _mind is None:
                _mind = NovaMind()
    return _mind

def json_resp(data, status=200):
    body = json.dumps(data, ensure_ascii=False).encode('utf-8')
    return (status, {'Content-Type': 'application/json; charset=utf-8'}, body)

def html_resp(html):
    body = html.encode('utf-8')
    return (200, {'Content-Type': 'text/html; charset=utf-8'}, body)

TEMPLATE = None
def get_template():
    global TEMPLATE
    if TEMPLATE is None:
        tpl_path = os.path.join(os.path.dirname(__file__), 'templates', 'index.html')
        with open(tpl_path, 'r', encoding='utf-8') as f:
            TEMPLATE = f.read()
    return TEMPLATE

STATIC_CACHE = {}
def get_static(path):
    if path not in STATIC_CACHE:
        spath = os.path.join(os.path.dirname(__file__), 'static', path)
        with open(spath, 'rb') as f:
            STATIC_CACHE[path] = f.read()
    return STATIC_CACHE[path]

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == '/':
            self._send(*html_resp(get_template()))
        elif path == '/api/state':
            self._send(*json_resp(self._get_state()))
        elif path.startswith('/static/'):
            rel = path[8:]
            try:
                data = get_static(rel)
                ctype = 'text/css' if rel.endswith('.css') else 'application/javascript'
                self.send_response(200)
                self.send_header('Content-Type', ctype)
                self.send_header('Content-Length', len(data))
                self.end_headers()
                self.wfile.write(data)
            except FileNotFoundError:
                self._send(404, {}, b'Not found')
        else:
            self._send(404, {}, b'Not found')

    def do_POST(self):
        if self.path == '/api/think':
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length) if length else b'{}'
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                self._send(*json_resp({'error': 'invalid json'}, 400))
                return
            result = self._do_think(data.get('text', ''))
            self._send(*json_resp(result))
        else:
            self._send(404, {}, b'Not found')

    def _send(self, status, headers, body):
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def _get_state(self):
        mind = get_mind()
        s = mind.get_full_state()
        mc = s.get('metacognition', {}).get('current_state', {})
        emotion = mc.get('emotional_state', {})

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

        chains = mind.get_causal_chains_with_labels(min_length=2)

        episodes = []
        for ep in mind.memory.episodic[-8:]:
            ec = ep.content or {}
            episodes.append({
                'input': ec.get('input', '')[:80],
                'type': ec.get('type', ''),
                'activation': getattr(ep, 'activation', 0),
            })

        goals_dict = getattr(mind.goals, 'goals', {})
        active_count = sum(1 for g in goals_dict.values() if getattr(g, 'status', '') == 'active')

        return {
            'cycle': s.get('cycle_count', 0),
            'uptime': s.get('uptime_seconds', 0),
            'metacognition': {
                'confidence': mc.get('confidence', 0),
                'coherence': mc.get('coherence', 0),
                'surprise': mc.get('surprise_level', 0),
                'curiosity': emotion.get('curiosity', 0),
                'certainty': emotion.get('certainty', 0),
                'confusion': emotion.get('confusion', 0),
            },
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
                'total': len(goals_dict),
                'unanswered': len(getattr(mind, 'unanswered_questions', [])),
            },
            'concepts_known': s.get('perception', {}).get('total_concepts', 0),
            'system': {
                'vram_mb': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
                'vram_total': 4096,
            },
        }

    def _do_think(self, text):
        if not text.strip():
            return {'error': 'empty input'}
        mind = get_mind()
        r = mind.think(text)
        layers = r['layers']
        meta_layers = layers.get('metacognition', {})
        perception = layers.get('perception', {})
        learning = layers.get('learning', {})

        recall = layers.get('memory', {}).get('recalled_episodes', [])
        recalled_texts = [ep.get('content', {}).get('input', '') for ep in recall if ep.get('content')]

        concepts = [c['label'] for c in perception.get('concepts', [])]
        is_novel = learning.get('novelty_analysis', {}).get('is_novel', False)

        if 'confidence' in meta_layers and isinstance(meta_layers['confidence'], dict):
            conf = meta_layers['confidence'].get('overall', 0.5)
        elif 'confidence' in meta_layers and isinstance(meta_layers['confidence'], (int, float)):
            conf = meta_layers['confidence']
        else:
            conf = 0.5

        return {
            'response': r['response']['text'],
            'trace': {
                'concepts': concepts[:8],
                'confidence': f"{conf:.0%}",
                'novel': is_novel,
                'recall_count': len(recalled_texts),
                'recalled_texts': recalled_texts[:3],
            },
            'inferences': r.get('inferences', [])[:3],
            'goal_satisfied': r.get('goal_satisfied'),
            'contradiction': r.get('contradiction'),
            'cycle': r['cycle'],
            'cycle_time_ms': r.get('cycle_time_ms', 0),
        }

if __name__ == '__main__':
    port = 5001
    server = http.server.HTTPServer(('localhost', port), Handler)
    print(f"  NovaMind Dashboard → http://localhost:{port}")
    server.serve_forever()
