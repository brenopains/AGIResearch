"""
Teste: PerceptualStream completo (video + audio em tempo real).

Processa um video do MSVD com audio, frame a frame.
Atualiza CGE em tempo real. Reconstroi frames e audio do buffer.
Verifica fidelidade da memoria vs percepcao original.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HOME"] = r"D:\novamind_data\cache\huggingface"
os.environ["TORCH_HOME"] = r"D:\novamind_data\cache\torch"

import torch
import cv2
import numpy as np
from core.grounding_engine import GroundingEngine
from core.visual_pipeline import VisualPipeline
from core.cognitive_graph_engine import CognitiveGraphEngine
from core.hdc_memory import HDCMemory
from core.perceptual_stream import PerceptualStream
from core.audio_pipeline import AudioPipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
print("="*60)
print("TESTE: PerceptualStream (video + audio em tempo real)")
print("="*60)

eng = GroundingEngine(device=torch.device(DEVICE))
vp = VisualPipeline(eng.proj_visual, device=DEVICE)

DECODER_PATH = r"D:\novamind_data\models\visual_decoder.pt"
if os.path.exists(DECODER_PATH):
    vp.decoder.load_state_dict(torch.load(DECODER_PATH, map_location=DEVICE))
    vp.decoder.eval()
    print(f"Decoder carregado ({os.path.getsize(DECODER_PATH)/1024:.0f} KB)")
else:
    print("[FATAL] Decoder nao encontrado.")
    sys.exit(1)

mem = HDCMemory(device=DEVICE)
cge = CognitiveGraphEngine(hdc_memory=mem, device=DEVICE)
stream = PerceptualStream(cge, vp, eng, device=DEVICE)

# --- Selecionar video com audio ---
META_PATH = r"D:\novamind_data\MSVD\msvd_train.json"
CLIPS_DIR = r"D:\novamind_data\MSVD\YouTubeClips"
with open(META_PATH, "r") as f:
    metadata = json.load(f)

# Achar video com dog + possivel audio
test_rec = metadata[0]
for r in metadata:
    c = r["caption"][0].lower()
    if "dog" in c:
        test_rec = r
        break

video_path = os.path.join(CLIPS_DIR, test_rec["video"])
print(f"\nVideo: {test_rec['video']}")
print(f"Legenda: {test_rec['caption'][0]}")

# --- Processar video inteiro frame a frame ---
print("\n--- Processando video (stream visual) ---")
t0 = time.time()
n_frames = stream.process_video(video_path, max_frames=60)
t_vis = time.time() - t0
print(f"Frames processados: {n_frames} em {t_vis:.1f}s ({n_frames/t_vis:.0f} FPS)")

print(f"\nEstado do CGE apos stream:")
print(cge.get_scene_text())

# --- Reconstruir frames do buffer e verificar fidelidade ---
print("\n--- Verificando fidelidade visual ---")
cap = cv2.VideoCapture(video_path)
psnrs = []
for i in range(min(10, n_frames)):
    ret, frame = cap.read()
    if not ret:
        break
    node_id = f"frame_{i}"
    metrics = stream.verify_visual_fidelity(node_id, frame)
    if "psnr" in metrics:
        psnrs.append(metrics["psnr"])
cap.release()

if psnrs:
    print(f"PSNR medio (10 frames): {np.mean(psnrs):.1f} dB")
    print(f"PSNR max: {max(psnrs):.1f} dB")

# --- Processar audio ---
print("\n--- Processando audio ---")
t0 = time.time()
try:
    import torchaudio
    wav, sr = torchaudio.load(video_path.replace(".avi", ".wav")
                                .replace(".mp4", ".wav"))
    if wav.shape[-1] > 0:
        if sr != 22050:
            resampler = torchaudio.transforms.Resample(sr, 22050)
            wav = resampler(wav)
        ap = AudioPipeline(eng, device=DEVICE)

        # Codificar 2 segundos de audio
        chunk = wav[0, :44100]
        mf_audio, codes = ap.encode_audio(chunk)
        print(f"Audio codificado: codes shape={codes.shape}")

        # Reconstruir fiel (dos codes)
        wav_rec = ap.decode_from_codes(codes)
        metrics_a = ap.verify_fidelity(chunk[:wav_rec.shape[-1]], wav_rec)
        print(f"Reconstrucao fiel (codes): PSNR={metrics_a['psnr']:.1f} dB")

        # Reconstruir HDC (do bipolar, via unbinding)
        wav_rec_hdc = ap.decode_audio(mf_audio)
        # So compara os primeiros N samples que o decode produz
        min_len = min(chunk.shape[-1], wav_rec_hdc.shape[-1])
        metrics_hdc = ap.verify_fidelity(chunk[:min_len], wav_rec_hdc[:min_len])
        print(f"Reconstrucao HDC (unbinding): PSNR={metrics_hdc['psnr']:.1f} dB")
    else:
        print("Audio vazio ou inexistente")
except Exception as e:
    print(f"Audio nao disponivel: {e}")

# --- Salvar frames reconstruidos ---
out_dir = r"D:\novamind_data\debug_stream"
os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
for i in range(min(5, n_frames)):
    ret, frame = cap.read()
    if not ret:
        break
    node_id = f"frame_{i}"
    img_rec = stream.reconstruct_visual(node_id)
    if img_rec is not None:
        t = img_rec.detach().cpu().float()
        arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"frame_{i}_rec.png"),
                    cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
cap.release()

print(f"\nFrames salvos em {out_dir}/")
print(f"\n=== TESTE CONCLUIDO ===")
