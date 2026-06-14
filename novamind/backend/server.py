import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from core.sensory import SensoryFlow
from core.episodic import EpisodicMemory
from core.world_model import WorldModel
from core.representations import RepresentationLibrary
from core.curiosity import CuriosityLayer
from core.persistence import CognitivePersistence
from env.textual_grounding import TextualGroundingEnv

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Cognitive Architecture
persistence = CognitivePersistence()
saved_state = persistence.load_state()

sensory = SensoryFlow()
episodic = EpisodicMemory()
world_model = WorldModel()
representations = RepresentationLibrary()
curiosity = CuriosityLayer()
env = TextualGroundingEnv()

if saved_state:
    # Restore State
    episodic.timeline = saved_state.get("timeline", [])
    world_model.entities = saved_state.get("entities", {})
    world_model.causal_links = saved_state.get("causal_links", [])
    world_model.beliefs = saved_state.get("beliefs", {})
    representations.library = saved_state.get("representations", {})
    curiosity.active_questions = saved_state.get("active_questions", [])
    curiosity.attention_distribution = saved_state.get("attention_distribution", {})

active_websockets = []

async def cognitive_loop():
    """
    The main loop of the Cognitive OS.
    Reads from environment -> sensory -> episodic -> world model.
    """
    while True:
        # 1. Environment tick
        raw_obs = env.step()
        
        # 2. Sensory processing
        event = sensory.process_input(raw_obs)
        
        # 3. Episodic memory recording
        episodic.record_event(event)
        
        # 4. World Model compression
        world_model.compress_episodes(episodic)
        
        # 5. Broadcast state to UI
        state = {
            "sensory_recent": sensory.get_recent(5),
            "episodic_uncompressed": episodic.get_uncompressed_episodes()[:5],
            "world_model": world_model.get_graph(),
            "representations": representations.get_dag(),
            "curiosity": curiosity.get_state()
        }
        
        for ws in active_websockets:
            try:
                await ws.send_json(state)
            except Exception:
                pass
                
        await asyncio.sleep(1.0) # 1 second tick

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cognitive_loop())

@app.on_event("shutdown")
def shutdown_event():
    # Save State
    state_to_save = {
        "timeline": episodic.timeline,
        "entities": world_model.entities,
        "causal_links": world_model.causal_links,
        "beliefs": world_model.beliefs,
        "representations": representations.library,
        "active_questions": curiosity.active_questions,
        "attention_distribution": curiosity.attention_distribution
    }
    persistence.save_state(state_to_save)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle UI commands if needed (e.g. changing env scenario)
    except WebSocketDisconnect:
        active_websockets.remove(websocket)

@app.get("/")
def read_root():
    return {"status": "Cognitive OS Online"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
