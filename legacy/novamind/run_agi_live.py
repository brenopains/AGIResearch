"""
AGI Live Runner — Benchmark MTB-1 + Servidor em paralelo

Roda o benchmark enquanto o servidor expoe o estado REAL
via StateBridge. Abra http://127.0.0.1:8765 no navegador
para ver a representacao interna AO VIVO enquanto a AGI
resolve os mundos.

Uso: python run_agi_live.py [n_worlds] [porta]
"""
import sys, os, time, threading

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

from core.state_bridge import StateBridge
bridge = StateBridge.instance()

n_worlds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
port = int(sys.argv[2]) if len(sys.argv) > 2 else 8765


def run_server():
    """Inicia servidor HTTP em thread."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from urllib.parse import urlparse
    import json

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            p = urlparse(self.path).path
            if p == "/" or p == "/index.html":
                fp = os.path.join(os.path.dirname(__file__), "agiview_live.html")
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        content = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(content.encode("utf-8"))
                except:
                    self.send_error(404)
            elif p == "/data":
                state = bridge.get_state()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(state).encode("utf-8"))
            else:
                self.send_error(404)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", port), Handler)
    print(f"\n[SERVER] http://127.0.0.1:{port}")
    bridge.add_thought("Servidor iniciado", "system")
    srv.serve_forever()


def run_benchmark():
    """Roda MTB-1 e escreve na StateBridge."""
    from mtb_world_generator import MTBWorld
    from core.cognitive_graph_engine import CognitiveGraphEngine
    from core.goal_system import GoalManager
    from mtb_sandbox_salto2 import MTBSandboxSalto2

    bridge.add_thought("Iniciando benchmark MTB-1...", "system")

    try:
        from core.bge_encoder import BGEEncoder
        bge = BGEEncoder()
        bridge.add_thought("BGE 768D encoder ativo", "system")
    except:
        bge = None
        bridge.add_thought("BGE indisponivel (usando hash fallback)", "warn")

    cge = CognitiveGraphEngine(device="cpu")
    gm = GoalManager(cge=cge)
    sandbox = MTBSandboxSalto2(cge, gm, bge)

    bridge.add_thought(f"MTB-1: {n_worlds} mundos + transferencia", "system")

    # Gerar mundos
    worlds = []
    for i in range(n_worlds):
        seed = 42 + i * 137
        w = MTBWorld(world_id=i + 1, seed=seed)
        worlds.append(w)

    solved = 0
    for i, w in enumerate(worlds):
        bridge.add_thought(f"Iniciando Mundo {i+1}: {w.physics['energy_flow']}+{w.physics['gravity']}", "system")
        result = sandbox.run_world(w, max_cycles=60)

        # Forcar write final
        bridge.set_from_cge(cge)
        bridge.set_from_mtb_sandbox(sandbox)
        bridge.add_concept_history(sandbox.cycle, len(cge.nodes), len(cge.edges))

        if result["solved"]:
            solved += 1
            bridge.add_thought(f"Mundo {i+1} RESOLVIDO! energia={result['final_energy']:.1f} cge_nos={result['n_cge_nodes']}", "success")
        else:
            bridge.add_thought(f"Mundo {i+1} FALHOU. energia={result['final_energy']:.1f}", "error")
        time.sleep(1.0)  # pausa entre mundos para visualizacao

    # Transferencia
    import random as rnd
    rng = rnd.Random(9999)
    physics = {
        "energy_flow": rng.choice(["liquid", "pressure", "vibration"]),
        "gravity": rng.choice(["normal", "inverted", "local", "none"]),
        "time_modifier": "decaying",
        "friction_inverse": False,
    }
    from mtb_world_generator import MATERIAL_NAMES, MTBMaterial
    available = [m for m in MATERIAL_NAMES if all(m not in w.materials for w in worlds)][:10] or MATERIAL_NAMES[:7]
    tw = MTBWorld(world_id=999, seed=9999)
    tw.physics = physics
    tw.materials = {}
    for name in rng.sample(available, min(7, len(available))):
        tw.materials[name] = MTBMaterial(name, tw._generate_properties(rng))
    tw.grid = [[None for _ in range(tw.width)] for _ in range(tw.height)]
    for y in range(tw.height):
        for x in range(tw.width):
            if rng.random() < 0.4:
                tw.grid[y][x] = rng.choice(list(tw.materials.keys()))

    bridge.add_thought(f"TRANSFERENCIA: mundo novo com materiais ineditos", "system")
    tr = sandbox.run_world(tw, max_cycles=80)
    bridge.set_from_cge(cge)
    bridge.set_from_mtb_sandbox(sandbox)
    bridge.add_concept_history(sandbox.cycle, len(cge.nodes), len(cge.edges))

    if tr["solved"]:
        bridge.add_thought(f"TRANSFERENCIA RESOLVIDA em {tr['cycles']} ciclos! energia={tr['final_energy']:.1f}", "success")
    else:
        bridge.add_thought(f"TRANSFERENCIA FALHOU", "error")

    bridge.add_thought(f"FIM: {solved}/{n_worlds} resolvidos + transferencia {'OK' if tr['solved'] else 'FAIL'}", "system")
    bridge.update(running=True)

    # Manter servidor vivo
    while True:
        time.sleep(1)


if __name__ == "__main__":
    print("=" * 60)
    print("  AGI LIVE — MTB-1 Benchmark + Visualizacao em Tempo Real")
    print(f"  Mundos: {n_worlds} + transferencia")
    print(f"  Abra http://127.0.0.1:{port} no navegador")
    print("=" * 60)

    # Iniciar servidor em thread
    server_thread = threading.Thread(target=run_server, daemon=False)
    server_thread.start()
    time.sleep(1)

    # Rodar benchmark
    try:
        run_benchmark()
    except Exception as e:
        bridge.add_thought(f"ERRO no benchmark: {e}", "error")
        import traceback
        traceback.print_exc()
        # Manter servidor vivo mesmo com erro
        while True:
            time.sleep(1)
