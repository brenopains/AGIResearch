"""
HyperView — Tomografia 3D da Mente da AGI (para humanos observarem)
=====================================================================
Servidor FastAPI + WebSocket que expõe:
  - Vetores HDC como pontos 3D (PCA/UMAP) em tempo real
  - Conceitos ativos no Maestro
  - Free Energy trajectory
  - Conexões entre conceitos (similaridade > threshold)
  - Neuromoduladores atuais

Frontend Three.js embutido: navegação 3D, clique para ver detalhes.

Uso:
    python -m core.hyperview
    # Abre http://localhost:8768

Não é um "painel de controle" — é uma TOMOGRAFIA.
O humano pode ver a AGI pensar em tempo real.
"""

import os
import sys
import json
import time
import math
import asyncio
import threading
from typing import Optional, List, Dict, Any
from pathlib import Path

_NOVAMIND = str(Path(__file__).parent.parent)
if _NOVAMIND not in sys.path:
    sys.path.insert(0, _NOVAMIND)

import torch
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="HyperView — Tomografia da NovaMind")

# ─── Estado compartilhado (alimentado pela AGI) ──────────────

class HyperViewState:
    """Estado global que a AGI alimenta e o frontend consome."""

    def __init__(self):
        self._lock = threading.Lock()

        # Pontos 3D (conceitos)
        self.points: List[dict] = []          # [{id, label, x, y, z, color, strength, domain}]

        # Conexões entre pontos (arestas)
        self.edges: List[dict] = []            # [{source, target, weight}]

        # Neuromoduladores atuais
        self.neuromodulators: Dict[str, float] = {
            "DA": 0.5, "NA": 0.3, "ACh": 0.3, "5HT": 0.5,
        }

        # Free Energy trajectory
        self.fe_trajectory: List[float] = []

        # Conceito ativo no momento
        self.active_concept: Optional[str] = None
        self.active_text: str = ""

        # Timestamp
        self.updated_at: float = time.time()

        # Conexões WebSocket ativas
        self.websockets: List[WebSocket] = []

        self._dim_reducer = DimensionReducer()

    def update_points(self, vectors: List[tuple]):
        """
        Atualiza pontos 3D a partir de vetores HDC.
        vectors: [(name, vec_10000d, domain, strength), ...]
        """
        with self._lock:
            if not vectors:
                return

            # Reduz dimensionalidade: 10000D → 3D
            names = [v[0] for v in vectors]
            vecs = np.array([v[1].cpu().numpy() if isinstance(v[1], torch.Tensor) else v[1]
                            for v in vectors])
            domains = [v[2] for v in vectors]
            strengths = [v[3] for v in vectors]

            coords_3d = self._dim_reducer.fit_transform(vecs)

            # Mapeia domínios para cores
            domain_colors = self._domain_colors(domains)

            self.points = [
                {
                    "id": names[i],
                    "label": names[i][:30],
                    "x": float(coords_3d[i, 0]),
                    "y": float(coords_3d[i, 1]),
                    "z": float(coords_3d[i, 2]),
                    "color": domain_colors[i],
                    "strength": float(strengths[i]),
                    "domain": domains[i],
                }
                for i in range(len(names))
            ]

            # Conecta pontos com similaridade > 0.6
            self.edges = []
            if len(vecs) < 50:
                for i in range(len(vecs)):
                    for j in range(i + 1, len(vecs)):
                        vi = torch.from_numpy(vecs[i]).float()
                        vj = torch.from_numpy(vecs[j]).float()
                        sim = float(torch.cosine_similarity(
                            vi.flatten().unsqueeze(0), vj.flatten().unsqueeze(0)))
                        if sim > 0.6:
                            self.edges.append({
                                "source": names[i],
                                "target": names[j],
                                "weight": float(sim),
                            })

            self.updated_at = time.time()

    def update_neuromodulators(self, nm: dict):
        with self._lock:
            self.neuromodulators.update(nm)
            self.updated_at = time.time()

    def add_fe_point(self, fe: float):
        with self._lock:
            self.fe_trajectory.append(fe)
            if len(self.fe_trajectory) > 500:
                self.fe_trajectory = self.fe_trajectory[-500:]
            self.updated_at = time.time()

    def set_active_concept(self, concept: str, text: str = ""):
        with self._lock:
            self.active_concept = concept
            self.active_text = text
            self.updated_at = time.time()

    def get_state(self) -> dict:
        with self._lock:
            return {
                "points": self.points,
                "edges": self.edges,
                "neuromodulators": self.neuromodulators,
                "fe_trajectory": self.fe_trajectory,
                "active_concept": self.active_concept,
                "active_text": self.active_text,
                "updated_at": self.updated_at,
                "n_points": len(self.points),
                "n_edges": len(self.edges),
            }

    def _domain_colors(self, domains: List[str]) -> List[str]:
        color_map = {
            "code": "#ff6b6b",
            "physics": "#4ecdc4",
            "medicine": "#45b7d1",
            "biology": "#96ceb4",
            "math": "#ffeaa7",
            "art": "#dda0dd",
            "music": "#ff9ff3",
            "general": "#a8a8a8",
            "psychology": "#feca57",
            "ethics": "#ff9f43",
        }
        return [color_map.get(d, "#a8a8a8") for d in domains]


class DimensionReducer:
    """Reduz 10000D → 3D para visualização."""

    def __init__(self):
        self.proj = None

    def fit_transform(self, vecs: np.ndarray) -> np.ndarray:
        if vecs.shape[0] == 0:
            return np.zeros((0, 3))

        if vecs.shape[0] == 1:
            return np.zeros((1, 3))

        # PCA via SVD
        mean = vecs.mean(axis=0, keepdims=True)
        centered = vecs - mean
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        # 3 primeiras componentes
        coords = centered @ Vt[:3, :].T
        # Normaliza
        max_val = np.abs(coords).max()
        if max_val > 0:
            coords = coords / max_val
        return coords


# ─── Singleton ────────────────────────────────────────────────

state = HyperViewState()


# ─── API ──────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def get_hyperview():
    html = _generate_html()
    return HTMLResponse(html)


@app.get("/api/state")
async def get_state():
    return state.get_state()


@app.post("/api/feed")
async def feed_data(data: dict):
    """Endpoint para a AGI alimentar dados."""
    if "points" in data:
        vectors = [(p["id"], torch.tensor(p.get("vector", [0]*10000)),
                    p.get("domain", "general"), p.get("strength", 0.5))
                   for p in data["points"]]
        state.update_points(vectors)
    if "nm" in data:
        state.update_neuromodulators(data["nm"])
    if "fe" in data:
        state.add_fe_point(data["fe"])
    if "active" in data:
        state.set_active_concept(data["active"], data.get("text", ""))
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state.websockets.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json(state.get_state())
    except WebSocketDisconnect:
        if websocket in state.websockets:
            state.websockets.remove(websocket)


def _generate_html() -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>HyperView — Tomografia NovaMind</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0a0a1a; color: #ccc; font-family: 'Courier New', monospace; overflow: hidden; }}
  #canvas {{ display: block; }}
  #info {{ position: absolute; top: 10px; left: 10px; color: #888; font-size: 12px; z-index: 10; }}
  #nm {{ position: absolute; bottom: 20px; left: 20px; z-index: 10; display: flex; gap: 20px; }}
  .nm-bar {{ width: 40px; text-align: center; }}
  .nm-bar .label {{ font-size: 10px; color: #666; }}
  .nm-bar .value {{ font-size: 14px; font-weight: bold; }}
  .nm-bar .bar {{ width: 10px; height: 60px; background: #1a1a3a; margin: 4px auto; border-radius: 5px; position: relative; }}
  .nm-bar .fill {{ position: absolute; bottom: 0; width: 100%; border-radius: 5px; transition: height 0.3s; }}
  .da .fill {{ background: #ffd700; }}
  .na .fill {{ background: #ff6b6b; }}
  .ach .fill {{ background: #4ecdc4; }}
  .sht .fill {{ background: #45b7d1; }}
  #fe {{ position: absolute; bottom: 20px; right: 20px; z-index: 10; color: #888; font-size: 11px; text-align: right; }}
  #fe canvas {{ margin-top: 4px; border: 1px solid #1a1a3a; }}
  #tooltip {{ position: absolute; background: rgba(10,10,30,0.9); border: 1px solid #333; padding: 8px 12px; border-radius: 4px; font-size: 11px; display: none; z-index: 20; pointer-events: none; max-width: 300px; }}
</style>
</head>
<body>
<div id="info">HYPERVIEW · tomografia neural · <span id="npoints">0</span> conceitos · <span id="nedges">0</span> conexões</div>
<div id="nm"></div>
<div id="fe"><div>FREE ENERGY</div><canvas id="feCanvas" width="200" height="60"></canvas></div>
<div id="tooltip"></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.1, 1000);
camera.position.set(3, 2, 5);
const renderer = new THREE.WebGLRenderer({{ antialias: true, canvas: document.getElementById('canvas') }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x0a0a1a);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.rotateSpeed = 0.5;

// Lighting
const ambient = new THREE.AmbientLight(0x333366, 0.4);
scene.add(ambient);
const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
dirLight.position.set(1, 2, 1);
scene.add(dirLight);

// Grid
const gridHelper = new THREE.GridHelper(4, 20, 0x333366, 0x222244);
scene.add(gridHelper);

// Stars background
const stars = new THREE.BufferGeometry();
const starVerts = [];
for (let i = 0; i < 2000; i++) {{
    starVerts.push((Math.random() - 0.5) * 200, (Math.random() - 0.5) * 200, (Math.random() - 0.5) * 200 - 50);
}}
stars.setAttribute('position', new THREE.Float32BufferAttribute(starVerts, 3));
const starMat = new THREE.PointsMaterial({{ color: 0x444466, size: 0.05 }});
scene.add(new THREE.Points(stars, starMat));

// Data structures
let pointMeshes = [];
let edgeLines = [];
let tooltip = document.getElementById('tooltip');

function hexColor(c) {{
    return new THREE.Color(parseInt(c.slice(1), 16) / 255);
}}

function updateScene(data) {{
    // Clear old
    pointMeshes.forEach(m => scene.remove(m));
    edgeLines.forEach(l => scene.remove(l));
    pointMeshes = [];
    edgeLines = [];

    if (!data.points) return;

    document.getElementById('npoints').textContent = data.n_points || 0;
    document.getElementById('nedges').textContent = data.n_edges || 0;

    // Points
    data.points.forEach(p => {{
        const size = 0.03 + p.strength * 0.12;
        const geo = new THREE.SphereGeometry(size, 16, 16);
        const mat = new THREE.MeshPhongMaterial({{
            color: hexColor(p.color),
            emissive: hexColor(p.color),
            emissiveIntensity: 0.3 + p.strength * 0.5,
            transparent: true,
            opacity: 0.8 + p.strength * 0.2,
        }});
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(p.x, p.y, p.z);
        mesh.userData = {{ label: p.label, domain: p.domain, strength: p.strength }};
        scene.add(mesh);
        pointMeshes.push(mesh);

        // Glow
        if (p.strength > 0.6) {{
            const glowGeo = new THREE.SphereGeometry(size * 2, 16, 16);
            const glowMat = new THREE.MeshBasicMaterial({{
                color: hexColor(p.color),
                transparent: true,
                opacity: 0.15,
            }});
            const glow = new THREE.Mesh(glowGeo, glowMat);
            glow.position.copy(mesh.position);
            scene.add(glow);
            pointMeshes.push(glow);
        }}
    }});

    // Edges (connections)
    if (data.edges) {{
        data.edges.forEach(e => {{
            const p1 = data.points.find(p => p.id === e.source);
            const p2 = data.points.find(p => p.id === e.target);
            if (!p1 || !p2) return;
            const points = [new THREE.Vector3(p1.x, p1.y, p1.z), new THREE.Vector3(p2.x, p2.y, p2.z)];
            const geo = new THREE.BufferGeometry().setFromPoints(points);
            const mat = new THREE.LineBasicMaterial({{
                color: 0x444488,
                transparent: true,
                opacity: e.weight * 0.4,
            }});
            const line = new THREE.Line(geo, mat);
            scene.add(line);
            edgeLines.push(line);
        }});
    }}

    // Neuromodulators
    updateNM(data.neuromodulators || {{}});

    // FE trajectory
    updateFE(data.fe_trajectory || []);
}}

function updateNM(nm) {{
    const colors = {{ DA: '#ffd700', NA: '#ff6b6b', ACh: '#4ecdc4', '5HT': '#45b7d1' }};
    const labels = {{ DA: 'DA', NA: 'NA', ACh: 'ACh', '5HT': '5HT' }};
    let html = '';
    for (const [key, val] of Object.entries(nm)) {{
        const pct = Math.round(val * 100);
        const color = colors[key] || '#888';
        html += '<div class="nm-bar ' + key.toLowerCase() + '"><div class="label">' + (labels[key] || key) + '</div>';
        html += '<div class="bar"><div class="fill" style="height:' + pct + '%;background:' + color + '"></div></div>';
        html += '<div class="value" style="color:' + color + '">' + pct + '%</div></div>';
    }}
    document.getElementById('nm').innerHTML = html;
}}

function updateFE(traj) {{
    const canvas = document.getElementById('feCanvas');
    if (!canvas || traj.length < 2) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, 200, 60);
    const w = 200, h = 60;
    const maxVal = Math.max(0.1, ...traj);
    const step = w / traj.length;
    ctx.beginPath();
    traj.forEach((v, i) => {{
        const x = i * step;
        const y = h - (v / maxVal) * h;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }});
    ctx.strokeStyle = '#4ecdc4';
    ctx.lineWidth = 1.5;
    ctx.stroke();
}}

// Raycaster for hover
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

renderer.domElement.addEventListener('mousemove', (e) => {{
    mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(pointMeshes);
    if (intersects.length > 0) {{
        const obj = intersects[0].object;
        const d = obj.userData;
        tooltip.style.display = 'block';
        tooltip.style.left = (e.clientX + 15) + 'px';
        tooltip.style.top = (e.clientY - 10) + 'px';
        tooltip.innerHTML = '<b>' + d.label + '</b><br>dominio: ' + d.domain + '<br>forca: ' + Math.round(d.strength * 100) + '%';
    }} else {{
        tooltip.style.display = 'none';
    }}
}});

// Animation loop
function animate() {{
    requestAnimationFrame(animate);
    controls.update();
    pointMeshes.forEach(m => {{
        if (m.geometry.type === 'SphereGeometry' && m.material.emissive) {{
            const pulse = 0.8 + Math.sin(Date.now() * 0.001 + m.position.x * 10) * 0.2;
            m.material.emissiveIntensity = m.userData.strength * pulse * 0.5;
        }}
    }});
    renderer.render(scene, camera);
}}
animate();

// Fetch state periodically
async function fetchState() {{
    try {{
        const resp = await fetch('/api/state');
        const data = await resp.json();
        updateScene(data);
    }} catch(e) {{}}
    setTimeout(fetchState, 2000);
}}
fetchState();

// Resize
window.addEventListener('resize', () => {{
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}});
</script>
</body>
</html>"""


# ─── Server ───────────────────────────────────────────────────

def run(host: str = "0.0.0.0", port: int = 8768):
    """Inicia servidor HyperView."""
    print(f"[HYPERVIEW] Tomografia neural em http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    run()
