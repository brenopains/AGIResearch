import asyncio
import random
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from core.sensory import SensoryFlow
from core.attention import AttentionLayer
from core.episodic import EpisodicMemory
from core.world_model import WorldModel
from core.skill_library import SkillLibrary
from core.representations import RepresentationLibrary
from core.predictor import Predictor
from core.mental_simulation import MentalSimulation
from core.planner import Planner
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
attention = AttentionLayer()
episodic = EpisodicMemory()
world_model = WorldModel()
skill_library = SkillLibrary()
representations = RepresentationLibrary()
predictor = Predictor(world_model, skill_library)
mental_sim = MentalSimulation(world_model, skill_library)
planner = Planner(mental_sim)
curiosity = CuriosityLayer()
env = TextualGroundingEnv()

if saved_state:
    # Restore State
    episodic.timeline = saved_state.get("timeline", [])
    world_model.entities = saved_state.get("entities", {})
    world_model.causal_links = saved_state.get("causal_links", [])
    world_model.beliefs = saved_state.get("beliefs", {})
    skill_library.skills = saved_state.get("skills", {})
    representations.library = saved_state.get("representations", {})
    curiosity.active_questions = saved_state.get("active_questions", [])

active_websockets = []

async def cognitive_loop():
    """
    The main loop of the full Cognitive OS:
    Sensory -> Attention -> Episodic -> WorldModels -> Predictor -> Simulation -> Planner -> Action
    """
    # Just to give it a goal occasionally
    goals = ["obter_madeira", "craftar_tábuas"]
    tick = 0
    
    while True:
        tick += 1
        
        # 1. Sense Environment
        raw_obs = env.get_observation()
        event = sensory.process_input(raw_obs)
        
        # 2. Attention Filter
        passed, score = attention.filter_event(event)
        
        if passed:
            # 3. Episodic Memory
            episodic.record_event(event)
            
            # 4. Predictor evaluates if this event matches any expectation (Error update)
            predictor.evaluate_observations([event])
            
            # 5. World Model Compression
            world_model.compress_episodes(episodic)
            
        # 6. Action cycle
        # Try to form a plan every 5 ticks if we don't have one
        if not planner.current_plan and tick % 5 == 0:
            goal = random.choice(goals)
            planner.create_plan(env.state, goal)
            
        # Execute next action in plan
        next_action = planner.get_next_action()
        if next_action:
            # Predict outcome before acting
            expected = "madeira" if next_action["action"] == "bater" else "tábuas" # Simplification
            predictor.make_prediction(next_action["action"], next_action["target"], expected)
            
            # Act in environment
            action_result = env.act(next_action["action"], next_action["target"])
            
            # Sense the result immediately
            result_evt = sensory.process_input(action_result)
            passed, _ = attention.filter_event(result_evt)
            if passed:
                episodic.record_event(result_evt)
                # Predictor evaluates the action outcome
                predictor.evaluate_observations([result_evt])
                world_model.compress_episodes(episodic)

        # Broadcast state
        state = {
            "sensory_recent": sensory.get_recent(5),
            "episodic_uncompressed": episodic.get_uncompressed_episodes()[:5],
            "world_model": world_model.get_graph(),
            "skills": skill_library.skills,
            "representations": representations.get_dag(),
            "predictions": predictor.active_predictions,
            "simulations": mental_sim.get_recent_simulations(),
            "curiosity": curiosity.get_state()
        }
        
        for ws in active_websockets:
            try:
                await ws.send_json(state)
            except Exception:
                pass
                
        await asyncio.sleep(2.0) # Slower tick to observe the UI

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cognitive_loop())

@app.on_event("shutdown")
def shutdown_event():
    state_to_save = {
        "timeline": episodic.timeline,
        "entities": world_model.entities,
        "causal_links": world_model.causal_links,
        "beliefs": world_model.beliefs,
        "skills": skill_library.skills,
        "representations": representations.library,
        "active_questions": curiosity.active_questions
    }
    persistence.save_state(state_to_save)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_websockets.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
