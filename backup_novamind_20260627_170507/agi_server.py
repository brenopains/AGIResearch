"""
AGI Server — SERVINDO DADOS REAIS do sistema cognitivo
Usa StateBridge como fonte unica de verdade. Sem mock.
"""
import os, sys, json, time, threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(__file__))

HOST, PORT = "127.0.0.1", 8765

# ---- Importa StateBridge e modulos reais ----
from core.state_bridge import StateBridge
bridge = StateBridge.instance()

# Tenta carregar CGE real
try:
    from core.cognitive_graph_engine import CognitiveGraphEngine
    _cge = CognitiveGraphEngine(device="cpu")
    bridge.set_from_cge(_cge)
    bridge.add_thought("CGE carregado com sucesso (cpu)", "system")
except Exception as e:
    _cge = None
    bridge.add_thought(f"CGE indisponivel: {e}", "warn")

# Tenta carregar SelfModel real
try:
    from core.self_model import SelfModel
    _sm = SelfModel()
    bridge.set_from_self_model(_sm)
except:
    _sm = None

# Tenta carregar OperatorSystem
try:
    from core.cognitive_operators import OperatorSystem
    _ops = OperatorSystem()
    bridge.set_from_operators(_ops)
except:
    _ops = None

bridge.update(running=True, uptime=0.0)
bridge.add_thought("Servidor AGI iniciado — aguardando agente conectar", "system")


class AGIHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        p = urlparse(self.path).path
        if p == "/" or p == "/index.html":
            self._serve_html()
        elif p == "/state":
            self._serve_sse()
        elif p == "/data":
            self._serve_json()
        else:
            self.send_error(404)

    def _serve_html(self):
        path = os.path.join(os.path.dirname(__file__), "agiview_live.html")
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(content.encode("utf-8"))
        except FileNotFoundError:
            self.send_error(404, "agiview_live.html not found")

    def _serve_sse(self):
        """SSE com estado REAL da StateBridge."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        cycle = 0
        try:
            while True:
                state = bridge.get_state()
                state["cycle"] = cycle
                state["uptime"] = round(time.time() - state.get("last_update", time.time()) + 1, 1)

                # Atualiza CGE se disponivel
                if _cge and cycle % 3 == 0:
                    bridge.set_from_cge(_cge)
                    bridge.add_concept_history(cycle, len(_cge.nodes), len(_cge.edges))
                    state = bridge.get_state()
                    state["cycle"] = cycle

                if _sm and cycle % 5 == 0:
                    bridge.set_from_self_model(_sm)

                if _ops and cycle % 5 == 0:
                    bridge.set_from_operators(_ops)

                msg = f"data: {json.dumps(state)}\n\n"
                self.wfile.write(msg.encode("utf-8"))
                self.wfile.flush()

                cycle += 1
                time.sleep(0.3)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _serve_json(self):
        state = bridge.get_state()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(state).encode("utf-8"))

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    print(f"\n{'='*60}")
    print(f"  AGI Server (DADOS REAIS via StateBridge)")
    print(f"  http://{HOST}:{port}")
    print(f"  JSON: http://{HOST}:{port}/data")
    print(f"  SSE:  http://{HOST}:{port}/state")
    print(f"{'='*60}")
    print(f"  CGE: {'OK' if _cge else 'indisponivel'}")
    print(f"  SelfModel: {'OK' if _sm else 'indisponivel'}")
    print(f"  Operators: {'OK' if _ops else 'indisponivel'}")
    print(f"  Aguardando agente escrever na StateBridge...")
    print(f"{'='*60}\n")

    srv = HTTPServer((HOST, port), AGIHandler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor parado.")
