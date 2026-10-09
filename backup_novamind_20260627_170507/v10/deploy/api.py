import os
import sys
import asyncio
import json
import base64
import io
import random
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from datasets import load_dataset

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from novamind.v10.tokenizer import Tokenizer
from novamind.v10.models.actor_critic import ActorCritic
from novamind.v10.models.symbolic import SymbolicHead
from novamind.v10.models.faiss_memory import FaissMemory
from novamind.v10.models.text_encoder import TextEncoder
from novamind.v10.models.jepa import JEPATrunk
from novamind.v10.models.rssm import RSSM
from novamind.v10.models.moe import SparseMoE
from novamind.v10.models.neuromorphic import LIFNeuron
from novamind.v10.models.vision_encoder import VisionVQVAE
from novamind.v10.models.audio_encoder import AudioEncoder
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(autonomous_mind_loop())
    yield
    task.cancel()

app = FastAPI(title="NovaMind v10 AGI Multimodal", lifespan=lifespan)
ui_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui")
app.mount("/static", StaticFiles(directory=ui_dir), name="static")

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(ui_dir, "index.html"))

print("[Backend] Booting NovaMind AGI Multimodal Cortex (A40 Massive Mode)...")
if not torch.cuda.is_available():
    raise SystemError("GPU não encontrada, o processamento está estritamente limitado para a GPU ativa.")
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = 'cuda'

# Laptop / RTX 3050 friendly constraints (Scaled down to prevent system freeze)
D_MODEL = 512
STOCH_DIM = 32
STOCH_CLASSES = 32

tokenizer = Tokenizer()
text_encoder = TextEncoder(vocab_size=16384, d_model=D_MODEL).to(device)

# --- SENSORY ENCODERS ---
vision_encoder_core = VisionVQVAE(num_embeddings=512, embedding_dim=64).to(device)
# Map 512 discrete vision codes into standard D_MODEL sequence elements
vision_projection = torch.nn.Embedding(512, D_MODEL).to(device)

audio_encoder_core = AudioEncoder(in_channels=1, d_model=D_MODEL).to(device)

jepa_trunk = JEPATrunk(embed_dim=D_MODEL).to(device)
rssm = RSSM(action_dim=10, embed_dim=D_MODEL, deter_dim=D_MODEL, stoch_dim=STOCH_DIM, stoch_classes=STOCH_CLASSES, hidden_dim=D_MODEL).to(device)
moe = SparseMoE(d_model=D_MODEL, num_experts=8, top_k=2).to(device) # expanded experts for A40
actor_critic = ActorCritic(deter_dim=D_MODEL, stoch_dim=STOCH_DIM, stoch_classes=STOCH_CLASSES, action_dim=10).to(device)
symbolic_head = SymbolicHead(embed_dim=D_MODEL).to(device)
memory = FaissMemory(embed_dim=D_MODEL)
snn_adapter = LIFNeuron(action_dim=10, threshold=0.75, leak_decay=0.85).to(device)

emit_head = torch.nn.Sequential(
    torch.nn.Linear(D_MODEL, D_MODEL),
    torch.nn.LayerNorm(D_MODEL),
    torch.nn.GELU(),
    torch.nn.Linear(D_MODEL, D_MODEL // 2),
    torch.nn.GELU(),
    torch.nn.Linear(D_MODEL // 2, 16384)
).to(device)

all_params = (
    list(text_encoder.parameters()) + list(jepa_trunk.parameters()) +
    list(rssm.parameters()) + list(moe.parameters()) +
    list(actor_critic.parameters()) + list(symbolic_head.parameters()) +
    list(emit_head.parameters()) + list(vision_encoder_core.parameters()) +
    list(vision_projection.parameters()) + list(audio_encoder_core.parameters())
)
optimizer = torch.optim.AdamW(all_params, lr=3e-4, weight_decay=0.01)

batch_size = 1
current_state = rssm.initial_state(batch_size, device=device)
last_action = torch.zeros(1, 10).to(device)

# --- CHECKPOINT LOADING (Resume Training Across Restarts) ---
checkpoint_path = "best_a40_multimodal.pt"
if os.path.exists(checkpoint_path):
    print(f"[MIND] Loading checkpoint from {checkpoint_path}...")
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        text_encoder.load_state_dict(ckpt.get('text_enc', text_encoder.state_dict()))
        vision_encoder_core.load_state_dict(ckpt.get('vis_enc', vision_encoder_core.state_dict()))
        audio_encoder_core.load_state_dict(ckpt.get('aud_enc', audio_encoder_core.state_dict()))
        jepa_trunk.load_state_dict(ckpt.get('jepa', jepa_trunk.state_dict()))
        rssm.load_state_dict(ckpt.get('rssm', rssm.state_dict()))
        actor_critic.load_state_dict(ckpt.get('actor', actor_critic.state_dict()))
        if 'emit_head' in ckpt:
            try: emit_head.load_state_dict(ckpt['emit_head'])
            except: print("[MIND] emit_head architecture changed, skipping.")
        if 'moe' in ckpt:
            moe.load_state_dict(ckpt['moe'])
        if 'symbolic' in ckpt:
            symbolic_head.load_state_dict(ckpt['symbolic'])
        if 'vis_proj' in ckpt:
            vision_projection.load_state_dict(ckpt['vis_proj'])
        _step = ckpt.get('global_step', 0)
        _loss = ckpt.get('best_loss', 999.0)
        print(f"[MIND] Checkpoint loaded! Step: {_step}, Best Loss: {_loss:.4f}")
    except Exception as e:
        print(f"[MIND] Checkpoint load failed (architecture change?): {e}. Starting fresh.")
else:
    print("[MIND] No checkpoint found. Starting fresh training.")

def tensor_step(input_text: str, input_image: torch.Tensor = None, input_audio: torch.Tensor = None, is_autonomous: bool = False, target_text: str = None) -> dict:
    global current_state, last_action
    
    # 1. TEXT STREAM
    tokens = tokenizer.encode(input_text)
    if not tokens: tokens = [tokenizer.unk_id]
    input_ids = torch.tensor([tokens]).to(device)
    embeddings = text_encoder(input_ids) # [1, seq, D_MODEL]
    
    sequence_tensors = [embeddings]
    
    # 2. VISION STREAM
    if input_image is not None:
        # Expected image: [B, 3, 64, 64]
        input_image = input_image.to(device)
        quantized_indices, _ = vision_encoder_core(input_image) # [B, 64]
        vis_emb = vision_projection(quantized_indices) # [B, 64, D_MODEL]
        sequence_tensors.append(vis_emb)
        
    # 3. AUDIO STREAM
    if input_audio is not None:
        # Expected audio: [B, 1, T] 
        input_audio = input_audio.to(device)
        aud_emb = audio_encoder_core(input_audio) # [B, seq_a, D_MODEL]
        sequence_tensors.append(aud_emb)

    # UNIFIED MULTIMODAL CONCATENATION ALONG TIME (SEQUENCE)
    fused_embeddings = torch.cat(sequence_tensors, dim=1)
    
    # FORWARD PASS INTO CORTEX
    jepa_out = jepa_trunk(fused_embeddings)
    obs_embed = jepa_out.mean(dim=1)
    
    prior, posterior, current_state = rssm.step(current_state, last_action, obs_embed)
    dense_latent = current_state['deter']
    
    # MOE Expert routing requires [B, SeqLength, D_MODEL]
    # In earlier versions it was [B, 1, D_MODEL].
    moe_out, _ = moe(dense_latent.unsqueeze(1))
    moe_out = moe_out.squeeze(1)
    
    stoch_flat = current_state['stoch'].view(current_state['stoch'].size(0), -1)
    continuous_action, value = actor_critic(moe_out, stoch_flat)
    
    spikes, membrane_potentials = snn_adapter(continuous_action.abs())
    action = continuous_action
    
    # Latent Memory Renderer
    latent_geom = current_state['deter'][0].detach().cpu().numpy()
    latent_geom = ((latent_geom - latent_geom.min()) / (latent_geom.max() - latent_geom.min() + 1e-5) * 255).astype('uint8')
    im = Image.fromarray(latent_geom.reshape(16, 32), mode='L').convert("P")
    im.putpalette([int(x) for x in range(256)] * 3)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    b64_latent_img = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    loss_val = 0.0
    emitted_text = ""
    emitted_modality = "text"
    emitted_media_b64 = None
    
    # 1. Modality Decision (Free Will Selection Disabled per user request)
    emitted_modality = "text"
    
    if is_autonomous:
        # Fast Training Mode (Single Token)
        gen_logits = emit_head(obs_embed + moe_out) # Current observation + expert-routed temporal context
        sampled_id = gen_logits.argmax(-1).item()
        emitted_text = tokenizer.decode([sampled_id])
        
        if target_text is not None:
            target_tokens = tokenizer.encode(target_text)
            if not target_tokens: target_tokens = [tokenizer.unk_id]
            target_ids = torch.tensor(target_tokens[-1:]).to(device)
            loss = F.cross_entropy(gen_logits, target_ids)
            loss_val = float(loss.item())
            # NÃO faz backward aqui! Acumula e faz 1 backward gigante no fim da frase!

        last_action = action.detach()
        current_state = {k: v.detach() for k, v in current_state.items()}
    else:
        # Interactive Mode (Autoregressive Generation & Media)
        if emitted_modality == "text":
            gen_words = []
            loop_state = {k: v.detach() for k, v in current_state.items()}
            loop_action = action.detach()
            
            for _ in range(8): # Generate 8 token sentence
                loop_moe, _ = moe(loop_state['deter'].unsqueeze(1))
                gen_logits = emit_head(obs_embed + loop_moe.squeeze(1))
                sampled_id = gen_logits.argmax(-1).item()
                word = tokenizer.decode([sampled_id])
                gen_words.append(word)
                
                # Feedback loop
                next_input_ids = torch.tensor([[sampled_id]]).to(device)
                next_emb = text_encoder(next_input_ids)
                next_prior, next_posterior, loop_state = rssm.step(loop_state, loop_action, next_emb.squeeze(1))
                
                next_moe, _ = moe(loop_state['deter'].unsqueeze(1))
                stoch_flat = loop_state['stoch'].view(loop_state['stoch'].size(0), -1)
                next_action, _ = actor_critic(next_moe.squeeze(1), stoch_flat)
                loop_action = next_action.detach()
                
            emitted_text = " ".join(gen_words)
            
        last_action = action.detach()
        current_state = {k: v.detach() for k, v in current_state.items()}
    
    programs, _ = symbolic_head.generate_program(moe_out.detach(), max_len=3)
    program_emitted = "".join(programs[0][:-1])
    ast_nodes = [{"id": f"n{i}", "label": p} for i, p in enumerate(programs[0][:-1])]
    ast_edges = [{"from": f"n{i}", "to": f"n{i+1}"} for i in range(len(ast_nodes)-1)]
    
    return {
        "is_autonomous": is_autonomous,
        "input_tokens": tokens,
        "rssm_prior_mean": float(prior['logits'].mean().item()),
        "moe_expert_idx": 0, 
        "symbolic_prog": program_emitted,
        "symbolic_ast": {"nodes": ast_nodes, "edges": ast_edges},
        "minds_eye_b64": b64_latent_img,
        "mem_index": 0,
        "action_vector": [float(x) for x in action[0][:5].detach().cpu().numpy()], 
        "spikes": [int(x) for x in spikes[0][:5].cpu().numpy()],
        "membrane_potentials": [float(x) for x in membrane_potentials[0][:5].detach().cpu().numpy()],
        "emitted_word": emitted_text,
        "emitted_modality": emitted_modality,
        "emitted_media_b64": emitted_media_b64,
        "loss_val": loss_val
    }

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.autonomous_loop_running = False

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

manager = ConnectionManager()

async def autonomous_mind_loop():
    global current_state, last_action
    manager.autonomous_loop_running = True
    print("[MIND] Initializing MASSIVE Multimodal HuggingFace Streams (FineWeb / MNIST / LibriSpeech)...")
    
    try:
        # Puxando o FineWeb (O mesmo que treinou o Llama 3) para Inteligência de texto real
        print("[MIND] Inicializando requests assincronas para dados do HuggingFace (não vai travar a porta 5000)...")
        ds_text = await asyncio.to_thread(load_dataset, "HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
        ds_vision = await asyncio.to_thread(load_dataset, "mnist", split="train", streaming=True)
        ds_audio = await asyncio.to_thread(load_dataset, "openslr/librispeech_asr", "default", split="validation", revision="refs/convert/parquet", streaming=True)
        
        text_iter = iter(ds_text)
        vision_iter = iter(ds_vision)
        audio_iter = iter(ds_audio)
    except Exception as e:
        print(f"[MIND] Stream boot error: {e}. Ensure libraries are installed.")
        return

    trans_vis = T.Compose([T.Resize((64, 64)), T.Grayscale(num_output_channels=3), T.ToTensor()])
    
    global_step = 0
    best_loss = 999.0
    
    # Resume training counters from checkpoint
    if os.path.exists("best_a40_multimodal.pt"):
        try:
            ckpt = torch.load("best_a40_multimodal.pt", map_location=device, weights_only=False)
            global_step = ckpt.get('global_step', 0)
            best_loss = ckpt.get('best_loss', 999.0)
            print(f"[TRAIN] Resuming from step {global_step}, best loss {best_loss:.4f}")
        except:
            pass
    
    while True:
        await asyncio.sleep(0.01)
        # TRAIN CONTINUOUSLY - NO WebSocket barrier! AGI trains even when nobody watches.
            
        try:
            text_sample_data = await asyncio.to_thread(next, text_iter)
            text_sample = text_sample_data['text'].strip()
            if not text_sample: continue
                
            img_sample_data = await asyncio.to_thread(next, vision_iter)
            img_sample = img_sample_data['image']
            img_tensor = trans_vis(img_sample).unsqueeze(0).to(device)
            
            try:
                audio_sample_data = await asyncio.to_thread(next, audio_iter)
                audio_sample = audio_sample_data['audio']['array']
                audio_cropped = torch.tensor(audio_sample[:2000]).float().unsqueeze(0).unsqueeze(0).to(device)
            except Exception:
                audio_cropped = torch.randn((1, 1, 2000)).float().to(device)
                
        except StopIteration:
            text_iter = iter(ds_text); vision_iter = iter(ds_vision); audio_iter = iter(ds_audio)
            continue
        except Exception as e:
            print(f"[MIND EXCEPTION]: {e}")
            await asyncio.sleep(1.0)
            continue
            
        words = text_sample.split()
        if len(words) < 10: continue
        
        optimizer.zero_grad()
        
        # Fresh recurrent state per sentence (clean computation graph)
        train_state = rssm.initial_state(1, device=device)
        train_action = torch.zeros(1, 10, device=device)
        
        total_loss = torch.tensor(0.0, device=device)
        num_preds = 0
        last_brain_data = None
        ctx_window = 8
        max_steps = 16  # Cap to prevent VRAM overflow on A40
        step_count = 0
        
        # Precompute vision/audio embeddings (same for all steps in this sentence)
        q_idx, _ = vision_encoder_core(img_tensor)
        vis_emb = vision_projection(q_idx)
        aud_emb = audio_encoder_core(audio_cropped)
        
        for i in range(0, len(words) - ctx_window - 1, 2):
            if step_count >= max_steps:
                break
                
            context = " ".join(words[i:i+ctx_window])
            target = words[i+ctx_window]
            
            # === END-TO-END FORWARD PASS WITH FULL GRADIENT FLOW ===
            # Every component receives gradients: TextEncoder -> JEPA -> RSSM -> MoE -> emit_head
            tokens = tokenizer.encode(context)
            if not tokens: tokens = [tokenizer.unk_id]
            input_ids = torch.tensor([tokens], device=device)
            embeddings = text_encoder(input_ids)
            
            fused = torch.cat([embeddings, vis_emb, aud_emb], dim=1)
            jepa_out = jepa_trunk(fused)
            obs_embed = jepa_out.mean(dim=1)
            
            prior, posterior, train_state = rssm.step(train_state, train_action, obs_embed)
            dense_latent = train_state['deter']
            
            moe_out, _ = moe(dense_latent.unsqueeze(1))
            moe_out = moe_out.squeeze(1)
            stoch_flat = train_state['stoch'].view(1, -1)
            train_action, value = actor_critic(moe_out, stoch_flat)
            
            # Predict next word using CURRENT observation + temporal context
            gen_logits = emit_head(obs_embed + moe_out)
            predicted_word = tokenizer.decode([gen_logits.argmax(-1).item()])
            
            # Cross-entropy loss CONNECTED through ENTIRE computation graph!
            t_tokens = tokenizer.encode(target)
            if not t_tokens: t_tokens = [tokenizer.unk_id]
            t_ids = torch.tensor(t_tokens[-1:], device=device)
            step_loss = F.cross_entropy(gen_logits, t_ids)
            total_loss = total_loss + step_loss
            num_preds += 1
            step_count += 1
            
            # Build UI metrics (detached, no graph impact)
            with torch.no_grad():
                spk, memb = snn_adapter(train_action.abs())
                last_brain_data = {
                    "is_autonomous": True, "input_tokens": tokens,
                    "rssm_prior_mean": float(prior['logits'].mean()),
                    "moe_expert_idx": 0, "symbolic_prog": "",
                    "symbolic_ast": {"nodes": [], "edges": []},
                    "minds_eye_b64": "", "mem_index": global_step,
                    "action_vector": [0.0]*5,
                    "spikes": [int(x) for x in spk[0][:5].cpu().numpy()],
                    "membrane_potentials": [float(x) for x in memb[0][:5].cpu().numpy()],
                    "emitted_word": predicted_word, "emitted_modality": "text",
                    "emitted_media_b64": None, "loss_val": float(step_loss.item())
                }
            
            if step_count % 4 == 0 and manager.active_connections:
                payload = json.dumps({"type": "brain_cycle", "data": last_brain_data})
                for conn in manager.active_connections.copy():
                    try: await conn.send_text(payload)
                    except: pass
            
            await asyncio.sleep(0.005)
        
        # === SINGLE BACKWARD THROUGH ENTIRE SENTENCE ===
        if num_preds > 0:
            (total_loss / num_preds).backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()
        
        global_step += 1
        avg_loss_val = float(total_loss.item()) / max(1, num_preds) if num_preds > 0 else 999.0
        if avg_loss_val > 0 and avg_loss_val < best_loss:
            best_loss = avg_loss_val
        
        if global_step % 10 == 0:
            print(f"[TRAIN] Step {global_step} | Loss: {avg_loss_val:.4f} | Best: {best_loss:.4f}")
            
        # Save COMPLETE checkpoint (all components + optimizer for resume)
        if global_step == 1 or global_step % 5 == 0:
            torch.save({
                'text_enc': text_encoder.state_dict(),
                'vis_enc': vision_encoder_core.state_dict(),
                'aud_enc': audio_encoder_core.state_dict(),
                'jepa': jepa_trunk.state_dict(),
                'rssm': rssm.state_dict(),
                'moe': moe.state_dict(),
                'actor': actor_critic.state_dict(),
                'emit_head': emit_head.state_dict(),
                'symbolic': symbolic_head.state_dict(),
                'vis_proj': vision_projection.state_dict(),
                'optimizer': optimizer.state_dict(),
                'global_step': global_step,
                'best_loss': best_loss
            }, "best_a40_multimodal.pt")
        
        # Update global state for interactive mode
        current_state = {k: v.detach() for k, v in train_state.items()}
        last_action = train_action.detach()
        
        # Broadcast final state to viewers
        if last_brain_data and manager.active_connections:
            payload = json.dumps({"type": "brain_cycle", "data": last_brain_data})
            for conn in manager.active_connections.copy():
                try: await conn.send_text(payload)
                except: pass
        
        await asyncio.sleep(0.05)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data_str = await websocket.receive_text()
            try:
                payload = json.loads(data_str)
                text_input = payload.get("text", "")
                media_b64 = payload.get("media")
                media_type = payload.get("media_type", "")
            except:
                text_input = data_str
                media_b64 = None
                media_type = ""
                
            input_img_tensor = None
            input_aud_tensor = None
            
            if media_b64:
                try:
                    if "base64," in media_b64:
                        _, encoded = media_b64.split("base64,", 1)
                    else:
                        encoded = media_b64
                    
                    raw_bytes = base64.b64decode(encoded)
                    
                    if "image" in media_type:
                        im = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                        trans_vis = T.Compose([T.Resize((64, 64)), T.ToTensor()])
                        input_img_tensor = trans_vis(im).unsqueeze(0).to(device)
                    elif "audio" in media_type or "video" in media_type:
                        # Attempt torchaudio for microphone
                        import torchaudio
                        # torchaudio can often process from BytesIO if backend supports it. Better to write to temp file.
                        tmp_path = "tmp_audio.webm"
                        with open(tmp_path, "wb") as f:
                            f.write(raw_bytes)
                        try:
                            waveform, _ = torchaudio.load(tmp_path)
                            if waveform.shape[0] > 1:
                                waveform = waveform.mean(dim=0, keepdim=True)
                            audio_cropped = waveform[:, :2000]
                            # pad if too short
                            if audio_cropped.shape[1] < 2000:
                                audio_cropped = F.pad(audio_cropped, (0, 2000 - audio_cropped.shape[1]))
                            input_aud_tensor = audio_cropped.unsqueeze(0).to(device)
                        finally:
                            if os.path.exists(tmp_path): os.remove(tmp_path)
                except Exception as e:
                    print(f"Media Decode Error: {e}")

            thinking_msg = json.dumps({"type": "status", "message": "processing", "input": text_input})
            await websocket.send_text(thinking_msg)
            
            await asyncio.sleep(0.1)
            brain_state = tensor_step(
                input_text=text_input, 
                input_image=input_img_tensor, 
                input_audio=input_aud_tensor, 
                is_autonomous=False
            )
            
            payload = json.dumps({"type": "brain_cycle", "data": brain_state})
            await websocket.send_text(payload)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
