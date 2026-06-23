"""
HyperView Server — FastAPI + WebSocket backend
Expoe a AGI NovaMind como servidor em tempo real.
"""

import sys, os, json, asyncio, time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent_bridge import AgentBridge


# ─── Modelos REST ──────────────────────────────────────────────

class QueryRequest(BaseModel):
    text: str
    max_chunks: int = 12

class QueryResponse(BaseModel):
    output: str
    intention: str
    residual_history: List[float]

class StatusResponse(BaseModel):
    intention: str
    output: str
    memory_items: int
    memory_sources: List[str]
    working_hdv_size: int
    pca_points: int


# ─── Gerenciador de Conexoes ─────────────────────────────────

class ConnectionManager:
    """Gerencia multiplos clientes WebSocket."""
    def __init__(self):
        self.active: List[WebSocket] = []
    
    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
    
    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)
    
    async def broadcast(self, event: dict):
        """Envia evento para todos os clientes conectados."""
        msg = json.dumps(event, default=str)
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    async def safe_broadcast(self, event: dict):
        """broadcast com proteção contra WebSockets não aceitos."""
        try:
            await self.broadcast(event)
        except Exception:
            pass


# ─── App Lifecycle ────────────────────────────────────────────

manager = ConnectionManager()
bridge: AgentBridge = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global bridge
    print("[HyperView] Inicializando AGI NovaMind...")
    bridge = AgentBridge(emit_callback=lambda e: manager.broadcast(e))
    print("[HyperView] AGI pronta. Servidor ouvindo.")
    yield
    print("[HyperView] Servidor encerrado.")

app = FastAPI(title="HyperView - NovaMind Visualizer", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── WebSocket ────────────────────────────────────────────────

_pca_cache = None  # cache global: (points, edges)

@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    await manager.connect(ws)
    try:
        # Envia estado inicial
        if bridge:
            status = bridge.get_status()
            await ws.send_text(json.dumps({"type": "status", **status}))
        
        # PCA: usa cache se disponível, senão computa uma vez
        global _pca_cache
        if _pca_cache is None and bridge:
            loop = asyncio.get_event_loop()
            pca_points = await loop.run_in_executor(None, bridge.build_pca_map)
            _pca_cache = pca_points
        elif _pca_cache is not None:
            pca_points = _pca_cache
        else:
            pca_points = []
        
        if pca_points:
            await ws.send_text(json.dumps({"type": "pca_update", "points": pca_points}))
        
        # Mantem conexao viva, processa comandos
        while True:
            try:
                data = await ws.receive_text()
            except Exception:
                break  # cliente desconectou durante comando longo
            msg = json.loads(data)
            
            if msg.get("action") == "query":
                text = msg.get("text", "")
                max_chunks = msg.get("max_chunks", 12)
                if bridge and text:
                    await bridge.query(text, max_chunks)
            
            elif msg.get("action") == "compositional":
                text = msg.get("text", "")
                if bridge and text:
                    await bridge.query_compositional(text)

            elif msg.get("action") == "analogical":
                a = msg.get("concept_a", "")
                b = msg.get("concept_b", "")
                if bridge and a and b:
                    await bridge.query_analogical(a, b)

            elif msg.get("action") == "ping":
                try:
                    await ws.send_text(json.dumps({"type": "pong"}))
                except Exception:
                    break
    
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception as e:
        manager.disconnect(ws)
        if "accept" not in str(e):
            print(f"[HyperView] Erro WS: {e}")


# ─── REST Endpoints ───────────────────────────────────────────

@app.post("/api/query", response_model=QueryResponse)
async def api_query(req: QueryRequest):
    """Envia uma query e aguarda o resultado completo."""
    if not bridge:
        raise HTTPException(503, "AGI nao inicializada")
    
    loop = asyncio.get_event_loop()
    output = await bridge.query(req.text, req.max_chunks)
    
    return QueryResponse(
        output=output,
        intention=bridge.current_intention,
        residual_history=bridge.residual_history[-50:],
    )

@app.post("/api/compositional", response_model=QueryResponse)
async def api_compositional(req: QueryRequest):
    """Query usando CognitiveAnswerer (composicional)."""
    if not bridge:
        raise HTTPException(503, "AGI nao inicializada")
    output = await bridge.query_compositional(req.text)
    return QueryResponse(
        output=output,
        intention=bridge.current_intention,
        residual_history=[],
    )

@app.get("/api/status", response_model=StatusResponse)
async def api_status():
    if not bridge:
        raise HTTPException(503, "AGI nao inicializada")
    s = bridge.get_status()
    return StatusResponse(
        intention=s["intention"],
        output=s["output"],
        memory_items=s["memory_items"],
        memory_sources=s["memory_sources"],
        working_hdv_size=s["working_hdv_size"],
        pca_points=s["pca_points"],
    )

class AnalogicalRequest(BaseModel):
    concept_a: str
    concept_b: str

@app.post("/api/analogical", response_model=QueryResponse)
async def api_analogical(req: AnalogicalRequest):
    """Transferência analógica entre dois conceitos."""
    if not bridge:
        raise HTTPException(503, "AGI nao inicializada")
    output = await bridge.query_analogical(req.concept_a, req.concept_b)
    return QueryResponse(
        output=output,
        intention=f"{req.concept_a} ⊕ {req.concept_b}",
        residual_history=[],
    )

@app.get("/api/pca_map")
async def api_pca():
    """Retorna o mapa PCA completo do HDV."""
    if not bridge:
        raise HTTPException(503, "AGI nao inicializada")
    return {"points": bridge.build_pca_map()}

@app.get("/api/subconcepts")
async def api_subconcepts(label: str = ""):
    """Retorna sub-conceitos para um label."""
    if not bridge:
        raise HTTPException(503, "AGI nao inicializada")
    if not label:
        return {"subconcepts": []}
    return {"subconcepts": bridge.get_subconcepts(label)}

@app.post("/api/fly_to")
async def api_fly_to(req: QueryRequest):
    """Navega ate um conceito e mostra seus sub-conceitos."""
    if not bridge:
        raise HTTPException(503, "AGI nao inicializada")
    subs = bridge.get_subconcepts(req.text)
    return {"concept": req.text, "subconcepts": subs}


# ─── Entrypoint ───────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("[HyperView] Iniciando servidor em http://localhost:8765")
    uvicorn.run(app, host="0.0.0.0", port=8765)
