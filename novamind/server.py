import asyncio
import json
import os

# Redirect caches to D: drive to save C: space
os.environ["HF_HOME"] = r"D:\novamind_data\cache\huggingface"
os.environ["TORCH_HOME"] = r"D:\novamind_data\cache\torch"

import sys
import glob
import re
import time
import tempfile
import threading
import traceback

# ─── STDOUT/STDERR CAPTURE: proxy completo que não quebra isatty() etc ───
_stdout = sys.stdout
_stderr = sys.stderr
class StreamInterceptor:
    def __init__(self, orig, stream_name):
        self.orig = orig
        self.stream_name = stream_name
    def __getattr__(self, name):
        return getattr(self.orig, name)
    def write(self, msg):
        if msg.strip() and ("image.png" in msg or "model does not support" in msg.lower()):
            self.orig.write(f"[{self.stream_name}-CAPTURE] {{{msg.strip()}}}\n")
            stack = "".join(traceback.format_stack()[:-1])
            self.orig.write(f"[{self.stream_name}-CAPTURE] Stack:\n{stack}\n")
        self.orig.write(msg)
sys.stdout = StreamInterceptor(sys.stdout, "STDOUT")
sys.stderr = StreamInterceptor(sys.stderr, "STDERR")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

import torch
if not torch.cuda.is_available():
    print("[WARNING] GPU não encontrada. Rodando em CPU mode.")
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    print(f"[INIT] GPU: {torch.cuda.get_device_name(0)}")

try:
    from core.mind import NovaMind
except Exception as e:
    print(f"[BOOT] ERRO importando NovaMind: {e}")
    traceback.print_exc()
    raise

try:
    from knowledge.manager import KnowledgeManager
except Exception as e:
    print(f"[BOOT] ERRO importando KnowledgeManager: {e}")
    traceback.print_exc()
    raise

from brain_persistence import load_brain, save_brain, start_autosave
from core.colony import Colony
from core.playground import PlaygroundState
from core.maestro import CognitiveMaestro


# ═══════════════════════════════════════════════
# GLOBAL STATE
# ═══════════════════════════════════════════════
try:
    mind = NovaMind()
except Exception as e:
    print(f"[BOOT] ERRO criando NovaMind: {e}")
    traceback.print_exc()
    raise

try:
    knowledge_manager = KnowledgeManager()
except Exception as e:
    print(f"[BOOT] ERRO criando KnowledgeManager: {e}")
    traceback.print_exc()
    raise

# ═══════════════════════════════════════════════
# AGENTIAL PERMISSIONS & SANDBOX CODE EXECUTION
# ═══════════════════════════════════════════════
import uuid
import types

pending_permissions = []
pending_permissions_lock = threading.Lock()

def request_permission_sync(action_type: str, details: str) -> bool:
    req_id = f"perm_{int(time.time() * 1000)}"
    event = threading.Event()
    req = {
        "id": req_id,
        "type": action_type,
        "details": details,
        "event": event,
        "approved": False,
        "resolved": False
    }
    with pending_permissions_lock:
        pending_permissions.append(req)
    print(f"[PERM] Novo pedido de permissão [{action_type}]: {details}")
    event.wait()
    return req["approved"]

mind.permission_handler = request_permission_sync

def execute_sandbox_code(self, code_string: str) -> str:
    approved = True
    if hasattr(self, 'permission_handler') and self.permission_handler:
        approved = self.permission_handler("execute_code", code_string)
    if not approved:
        return "Execução de código rejeitada pelo usuário."
    
    os.makedirs("tmp/sandbox", exist_ok=True)
    filepath = "tmp/sandbox/script.py"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code_string)
        
    import subprocess
    try:
        result = subprocess.run(
            ["python", filepath],
            capture_output=True,
            text=True,
            timeout=5
        )
        output = result.stdout
        if result.stderr:
            output += "\n[ERRO]\n" + result.stderr
        return output if output.strip() else "[Sucesso, sem output]"
    except subprocess.TimeoutExpired:
        return "[ERRO] Tempo limite de execução (5s) excedido."
    except Exception as e:
        return f"[ERRO] Falha ao rodar o script: {e}"

mind.execute_sandbox_code = types.MethodType(execute_sandbox_code, mind)

# ═══════════════════════════════════════════════
# SPEECH SYNTHESIS (VRAM-free)
# ═══════════════════════════════════════════════
def text_to_speech_wav(text: str, filepath: str):
    import numpy as np
    import struct
    from core.vocal_synthesizer import synthesize, formant_for_vowel
    
    words = [w.strip() for w in re.split(r'\W+', text.lower()) if w.strip()]
    if not words:
        words = ["olá"]
        
    vowel_map = {
        'a': 'aa', 'e': 'ee', 'i': 'ii', 'o': 'oo', 'u': 'uu',
        'm': 'mmm', 'n': 'mmm'
    }
    
    waveforms = []
    sample_rate = 22050
    
    for word in words:
        vowel = 'aa'
        for char in word:
            if char in vowel_map:
                vowel = vowel_map[char]
                break
        
        f1, f2, f3 = formant_for_vowel(vowel)
        pitch = 110 + (len(word) % 5) * 10
        duration = 0.12 + (len(word) % 3) * 0.04
        
        wav_bytes = synthesize(pitch=pitch, f1=f1, f2=f2, f3=f3, duration=duration)
        raw_data = np.frombuffer(wav_bytes[44:], dtype=np.int16).astype(np.float32) / 32768.0
        waveforms.append(raw_data)
        
    silence = np.zeros(int(sample_rate * 0.04), dtype=np.float32)
    full_audio_list = []
    for w in waveforms:
        full_audio_list.append(w)
        full_audio_list.append(silence)
        
    if not full_audio_list:
        full_audio = np.zeros(sample_rate, dtype=np.float32)
    else:
        full_audio = np.concatenate(full_audio_list[:-1])
        
    peak = np.max(np.abs(full_audio))
    if peak > 1e-10:
        full_audio = full_audio / peak
    signal_int = (full_audio * 0.4 * 32767).astype(np.int16)
    
    data_size = len(signal_int) * 2
    with open(filepath, 'wb') as f:
        f.write(struct.pack('<4sI4s', b'RIFF', 36 + data_size, b'WAVE'))
        f.write(struct.pack('<4sIHHIIHH', b'fmt ', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
        f.write(struct.pack('<4sI', b'data', data_size))
        f.write(signal_int.tobytes())

active_websockets: list = []

# Colony (multi-mind classroom) — activated with --colony flag
colony = None
if "--colony" in sys.argv[1:]:
    colony = Colony(max_minds=5)
    colony.start()
    print("[BOOT] Colonia AGI (sala de aula) iniciada")

# Playground — embodied environment for motor actions
playground = PlaygroundState()

# Try to restore brain from previous session
try:
    restored = load_brain(mind)
    if restored:
        print("[BOOT] Cérebro restaurado. Continuando de onde parou.")
    else:
        print("[BOOT] Cérebro novo. Iniciando aprendizado do zero.")
except Exception as e:
    print(f"[BOOT] ERRO carregando cérebro: {e}")
    traceback.print_exc()
    restored = None

# Maestro (orquestrador paralelo) — criado no lifespan
maestro = None

# Fallback state para offline_training (quando maestro não está ativo)
_fallback_state = {
    "status": "booting",
    "current_topic": "",
    "current_source": "",
    "current_sentence": "",
    "sentences_processed": 0,
    "total_sentences": 0,
    "topics_completed": 0,
    "stream_vision": "Standby",
    "stream_audio": "Standby",
    "stream_reading": "Standby",
    "media_type": None,
    "media_audio_url": None,
    "vocal_audio_url": None,
    "media_frame_url": None,
    "media_caption": None,
    "media_timestamp": None,
}


# ═══════════════════════════════════════════════
# OFFLINE TRAINING (from data/train/*.txt)
# ═══════════════════════════════════════════════
def offline_training():
    """
    Se não tiver internet ou quiser treinar offline,
    alimenta o NovaMind com os .txt em data/train/.
    """
    txt_files = glob.glob(os.path.join("data", "train", "*.txt"))
    if not txt_files:
        return

    print(f"[OFFLINE] Encontrados {len(txt_files)} arquivos de treinamento local.")
    _fallback_state["status"] = "offline_training"

    for filepath in txt_files:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()

            sentences = KnowledgeManager.split_into_sentences(text)
            fname = os.path.basename(filepath)
            _fallback_state["current_source"] = f"Local: {fname}"
            _fallback_state["stream_reading"] = f"Lendo: {fname}"
            _fallback_state["total_sentences"] = len(sentences)

            for i, sentence in enumerate(sentences):
                _fallback_state["current_sentence"] = sentence[:100]
                _fallback_state["sentences_processed"] = i + 1
                try:
                    mind.think(sentence)
                except Exception as e:
                    print(f"[OFFLINE] Erro: {e}")
                time.sleep(0.1)

        except Exception as e:
            print(f"[OFFLINE] Erro ao ler {filepath}: {e}")

    print(f"[OFFLINE] Treinamento local concluído.")


# ═══════════════════════════════════════════════
# MAESTRO TICK LOOP (substitui o loop sequencial)
# ═══════════════════════════════════════════════
def maestro_tick_loop():
    """Heartbeat do Maestro — chama tick() continuamente."""
    tick_count = 0
    while True:
        try:
            if maestro is not None:
                maestro.tick()
            tick_count += 1
            if tick_count % 20 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(0.02)
        except Exception as e:
            print(f"[MAESTRO] Erro no tick: {e}")
            traceback.print_exc()
            time.sleep(1)


# ═══════════════════════════════════════════════
# WEBSOCKET BROADCAST
# ═══════════════════════════════════════════════
async def broadcast_state():
    while True:
        if active_websockets:
            try:
                state = mind.get_full_state()
            except Exception as e:
                await asyncio.sleep(2)
                continue

            # Project concept positions to 3D and classify modalities
            import numpy as np
            np.random.seed(42)
            proj_matrix_3d = np.random.randn(32, 3) / np.sqrt(3)
            
            all_concepts = state.get("perception", {}).get("all_concepts", [])
            for c in all_concepts:
                pos = c.get("position")
                if pos is not None and len(pos) == 32:
                    p_arr = np.array(pos)
                    proj = np.dot(p_arr, proj_matrix_3d) * 12.0
                    c["coords3d"] = proj.tolist()
                else:
                    c["coords3d"] = [0.0, 0.0, 0.0]
                
                lbl = c.get("label", "").lower()
                tp = c.get("type", "")
                if tp == "archetype" or "gravity" in lbl or "rule" in lbl:
                    c["modality"] = "archetype"
                elif "visual" in lbl or "frame" in lbl or "vendo" in lbl or "olha" in lbl:
                    c["modality"] = "visual"
                elif "audio" in lbl or "som" in lbl or "bark" in lbl or "ouvindo" in lbl or "latido" in lbl:
                    c["modality"] = "audio"
                else:
                    c["modality"] = "text"
            
            # Send pending permissions
            with pending_permissions_lock:
                state["pending_permissions"] = [
                    {
                        "id": p["id"],
                        "type": p["type"],
                        "details": p["details"],
                        "resolved": p["resolved"],
                        "approved": p["approved"]
                    }
                    for p in pending_permissions if not p["resolved"]
                ]


            # Lê do maestro (ou fallback state se maestro não iniciou)
            if maestro is not None:
                ls = maestro.get_learning_state()
                _p = maestro.playground
            else:
                ls = _fallback_state
                _p = playground

            state["streams"] = {
                "vision": ls.get("stream_vision", "Standby"),
                "audio": ls.get("stream_audio", "Standby"),
                "reading": ls.get("stream_reading", "Standby"),
            }
            state["autonomous_learning"] = {
                "status": ls.get("status", "booting"),
                "current_topic": ls.get("current_topic", ""),
                "current_source": ls.get("current_source", ""),
                "current_sentence": ls.get("current_sentence", ""),
                "sentences_processed": ls.get("sentences_processed", 0),
                "total_sentences": ls.get("total_sentences", 0),
                "topics_completed": ls.get("topics_completed", 0),
                "attention_weight": mind.predictor.get_attention_weight(),
                "prediction_error": mind.predictor.prediction_error,
                "enabled": maestro.autonomous_learning_enabled if maestro is not None else False,
            }
            state["media"] = {
                "type": ls.get("media_type"),
                "audio_url": ls.get("media_audio_url"),
                "vocal_audio_url": ls.get("vocal_audio_url"),
                "frame_url": ls.get("media_frame_url"),
                "caption": ls.get("media_caption"),
                "timestamp": ls.get("media_timestamp"),
            }

            # Colony state if active
            if colony and colony._running:
                state["colony"] = colony.get_state()
                state["colony"]["discussion"] = colony.get_recent_discussion(8)

            # Archetypes & Cognitive Primitives
            state["archetypes"] = mind.archetypes.get_state()

            # Playground (embodied environment)
            state["playground"] = {
                "position": _p.position,
                "held_object": _p.held_object,
                "nearby": _p.nearby_objects[:3],
                "last_action": _p.last_action_result[:100] if _p.last_action_result else "",
                "recent_actions": [
                    {"action": a["action"], "result": a["result"][:60]}
                    for a in list(_p.action_history)[-5:]
                ],
            }

            dead = []
            for ws in active_websockets:
                try:
                    await ws.send_json(state)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                if ws in active_websockets:
                    active_websockets.remove(ws)

        await asyncio.sleep(1.0)


# ═══════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    global maestro

    # Startup
    asyncio.create_task(broadcast_state())
    start_autosave(mind, interval=60)

    # Cleanup leftover YouTube temp files from previous runs
    if os.path.exists(yt_temp_dir):
        import shutil
        try:
            shutil.rmtree(yt_temp_dir)
            print("[BOOT] YouTube temp directory cleaned")
        except Exception:
            pass
    os.makedirs(yt_temp_dir, exist_ok=True)

    # Inicia o Maestro (orquestrador paralelo)
    maestro = CognitiveMaestro(mind, knowledge_manager, playground)
    maestro.start()

    # Thread do tick do maestro (heartbeat)
    tick_thread = threading.Thread(target=maestro_tick_loop, daemon=True)
    tick_thread.start()
    print("[BOOT] Maestro cognitivo paralelo: Percepção | Cognição | Motor | Sono")

    yield

    # Shutdown — save brain before dying
    print("[SHUTDOWN] Salvando cérebro...")
    if maestro is not None:
        maestro.stop()
    save_brain(mind)
    print("[SHUTDOWN] Cérebro salvo. Até a próxima.")


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

yt_temp_dir = r"D:\novamind_data\temp\novamind_yt"
vocal_temp_dir = r"D:\novamind_data\temp\novamind_vocal"
os.makedirs(yt_temp_dir, exist_ok=True)
os.makedirs(vocal_temp_dir, exist_ok=True)
app.mount("/static/yt", StaticFiles(directory=yt_temp_dir), name="yt")
app.mount("/static/vocal", StaticFiles(directory=vocal_temp_dir), name="vocal")


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    import shutil
    os.makedirs(r"D:\novamind_data\temp\uploads", exist_ok=True)
    filepath = os.path.join(r"D:\novamind_data\temp\uploads", file.filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    ext = os.path.splitext(file.filename)[1].lower()
    
    text_content = ""
    sensory_desc = ""
    
    if ext == ".pdf":
        import pypdf
        try:
            reader = pypdf.PdfReader(filepath)
            extracted = []
            for page in reader.pages[:10]: # limit to 10 pages to avoid context bloat
                t = page.extract_text()
                if t:
                    extracted.append(t)
            text_content = "\n".join(extracted)
            sensory_desc = f"[Anexo PDF: {file.filename}]"
        except Exception as e:
            text_content = f"Erro ao ler PDF: {e}"
            
    elif ext in [".mp3", ".wav", ".webm"]:
        yt_adapter = knowledge_manager.adapters["youtube"]
        try:
            if ext == ".webm":
                wav_path = filepath + ".wav"
                import subprocess
                subprocess.run(["ffmpeg", "-y", "-i", filepath, "-ar", "16000", "-ac", "1", wav_path], capture_output=True)
                audio_file = wav_path
            else:
                audio_file = filepath
                
            transcription_data = yt_adapter._transcribe(audio_file)
            text_content = transcription_data.get("text", "")
            
            import librosa
            audio_raw, audio_sr = librosa.load(audio_file, sr=16000, mono=True)
            if hasattr(mind, 'auditory_cortex'):
                mind.auditory_cortex.process(audio_raw, audio_sr)
                
            sensory_desc = f"[Anexo Áudio: {file.filename}]"
        except Exception as e:
            text_content = f"Erro ao processar áudio: {e}"
            
    elif ext == ".mp4":
        yt_adapter = knowledge_manager.adapters["youtube"]
        try:
            audio_path = filepath + ".wav"
            import subprocess
            subprocess.run(["ffmpeg", "-y", "-i", filepath, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path], capture_output=True)
            
            transcription_data = yt_adapter._transcribe(audio_path)
            text_content = transcription_data.get("text", "")
            
            import librosa
            audio_raw, audio_sr = librosa.load(audio_path, sr=16000, mono=True)
            if hasattr(mind, 'auditory_cortex'):
                mind.auditory_cortex.process(audio_raw, audio_sr)
                
            frames_dir = os.path.join(yt_temp_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            for f in os.listdir(frames_dir):
                try: os.remove(os.path.join(frames_dir, f))
                except: pass
                
            subprocess.run([
                "ffmpeg", "-y", "-i", filepath,
                "-vf", "fps=1/20", "-frames:v", "8",
                os.path.join(frames_dir, "frame_%03d.jpg")
            ], capture_output=True)
            
            frames = sorted([
                os.path.join(frames_dir, f)
                for f in os.listdir(frames_dir)
                if f.endswith(".jpg")
            ])
            
            frame_descriptions = yt_adapter._describe_frames(frames)
            sensory_desc = f"[Anexo Vídeo: {file.filename}] " + " | ".join(frame_descriptions)
            
        except Exception as e:
            text_content = f"Erro ao processar vídeo: {e}"
            
    full_prompt = f"{sensory_desc}\nConteúdo: {text_content}" if sensory_desc else text_content
    thought = await asyncio.to_thread(mind.think, full_prompt)
    response_text = thought.get("response", {}).get("text", "...")
    
    resp_wav_name = f"response_{int(time.time() * 1000)}.wav"
    resp_wav_path = os.path.join(vocal_temp_dir, resp_wav_name)
    try:
        await asyncio.to_thread(text_to_speech_wav, response_text, resp_wav_path)
        vocal_url = f"http://localhost:8000/static/vocal/{resp_wav_name}"
        if maestro is not None:
            maestro.state["vocal_audio_url"] = vocal_url
    except Exception as ex:
        print(f"Speech synthesis error: {ex}")
        vocal_url = None
        
    return {
        "text": response_text,
        "vocal_audio_url": vocal_url,
        "filename": file.filename
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)
    try:
        while True:
            msg_str = await websocket.receive_text()
            try:
                data = json.loads(msg_str)
            except Exception:
                # ignore non-JSON or pings
                continue
                
            msg_type = data.get("type")
            if msg_type == "user_chat":
                text = data.get("text", "")
                
                # Check for code execution in user message
                if "```python" in text:
                    code = text.split("```python")[1].split("```")[0].strip()
                    output = await asyncio.to_thread(mind.execute_sandbox_code, code)
                    user_input_prompt = f"Código rodado pelo usuário:\n{code}\nOutput:\n{output}"
                else:
                    user_input_prompt = text
                    
                thought = await asyncio.to_thread(mind.think, user_input_prompt)
                response = thought.get("response", {})
                resp_text = response.get("text", "...")
                
                resp_wav_name = f"response_{int(time.time() * 1000)}.wav"
                resp_wav_path = os.path.join(vocal_temp_dir, resp_wav_name)
                try:
                    await asyncio.to_thread(text_to_speech_wav, resp_text, resp_wav_path)
                    vocal_url = f"http://localhost:8000/static/vocal/{resp_wav_name}"
                    if maestro is not None:
                        maestro.state["vocal_audio_url"] = vocal_url
                except Exception as ex:
                    print(f"Speech synthesis error: {ex}")
                    vocal_url = None
                    
                await websocket.send_json({
                    "type": "chat_response",
                    "text": resp_text,
                    "vocal_audio_url": vocal_url
                })
                
            elif msg_type == "permission_response":
                req_id = data.get("id")
                approved = data.get("approved", False)
                with pending_permissions_lock:
                    for p in pending_permissions:
                        if p["id"] == req_id and not p["resolved"]:
                            p["approved"] = approved
                            p["resolved"] = True
                            p["event"].set()
                            print(f"[PERM] Pedido {req_id} resolvido pelo usuario. Aprovado: {approved}")
                            break
                            
            elif msg_type == "toggle_learning":
                enabled = data.get("enabled", False)
                if maestro is not None:
                    maestro.autonomous_learning_enabled = enabled
                    print(f"[MAESTRO] Aprendizado autônomo alterado pelo usuario para: {enabled}")
                    if enabled:
                        maestro.percepcao.send("next_topic")
    except WebSocketDisconnect:
        if websocket in active_websockets:
            active_websockets.remove(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in active_websockets:
            active_websockets.remove(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
