"""
Gera outputs visuais da AGI: o que ela VE, INVENTA e RECONSTROI.
Salva em D:\novamind_data\agi_outputs/
"""
import sys, os, time, json, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HOME"] = r"D:\novamind_data\cache\huggingface"
os.environ["TORCH_HOME"] = r"D:\novamind_data\cache\torch"

import torch
import cv2
from core.grounding_engine import GroundingEngine
from core.visual_pipeline import VisualPipeline
from core.hdc_memory import HDCMemory
from core.coglang import Manifold

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIDEO_DIR = r"D:\novamind_data\MSVD\YouTubeClips"
OUT_DIR = r"D:\novamind_data\agi_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Inicializando... Device={DEVICE}")
eng = GroundingEngine(device=torch.device(DEVICE))
vp = VisualPipeline(eng.proj_visual, device=DEVICE)
ckpt = r"D:\novamind_data\models\visual_decoder.pt"
if os.path.exists(ckpt):
    vp.decoder.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
vp.decoder.eval()

def save_img(tensor, name):
    """Salva tensor (C,H,W) float32 0-1 como PNG."""
    arr = (tensor.detach().cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUT_DIR, name), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    print(f"  Salvo: {name}")

def get_frame(path, frame_idx=15):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret: return None, None, None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_t = eng.transform(frame_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats = eng.feature_extractor(img_t).view(-1)
    mf, presign, _ = vp.encode_frame(feats)
    f_rec = torch.matmul(vp.pinv, mf.tensor.float())
    img_rec = vp.decoder(f_rec.unsqueeze(0)).squeeze(0)
    orig = torch.from_numpy(cv2.resize(frame_rgb, (224,224)).astype(np.float32)/255.0).permute(2,0,1)
    return orig, img_rec, mf

# ============================================================
# 1. CAIXA DE PANDORA: O que a AGI inventa?
# ============================================================
print("\n" + "="*60)
print("CAIXA DE PANDORA: A AGI INVENTA UMA IMAGEM NOVA")
print("="*60)

videos = [v for v in os.listdir(VIDEO_DIR) if v.endswith(".avi")]
random.seed(42)
pair = random.sample(videos, 2)
va, vb = pair
print(f"Video A: {va}")
print(f"Video B: {vb}")

orig_a, rec_a, mf_a = get_frame(os.path.join(VIDEO_DIR, va), 15)
orig_b, rec_b, mf_b = get_frame(os.path.join(VIDEO_DIR, vb), 15)

if orig_a is not None and orig_b is not None:
    save_img(orig_a, "caixa_A_original.png")
    save_img(orig_b, "caixa_B_original.png")
    save_img(rec_a, "caixa_A_decoded.png")
    save_img(rec_b, "caixa_B_decoded.png")

    # Forja bind: A * B
    bound = Manifold(mf_a.tensor * mf_b.tensor, device=DEVICE)
    f_inv = torch.matmul(vp.pinv, bound.tensor.float())
    img_inv = vp.decoder(f_inv.unsqueeze(0)).squeeze(0)
    save_img(img_inv, "caixa_INVENCAO_AxB.png")

    sim_a = float(torch.cosine_similarity(bound.tensor.float().unsqueeze(0), mf_a.tensor.float().unsqueeze(0)))
    sim_b = float(torch.cosine_similarity(bound.tensor.float().unsqueeze(0), mf_b.tensor.float().unsqueeze(0)))
    print(f"\n  Similaridade invencao vs A: {sim_a:.4f}")
    print(f"  Similaridade invencao vs B: {sim_b:.4f}")
    print(f"  Invencao salva em: {OUT_DIR}/caixa_INVENCAO_AxB.png")

# ============================================================
# 2. CLOSED-LOOP: A AGI RECONSTROI O QUE VE?
# ============================================================
print("\n" + "="*60)
print("CLOSED-LOOP: AGI TENTA RECONSTRUIR 5 VIDEOS")
print("="*60)

random.seed(123)
test_vids = random.sample(videos, min(5, len(videos)))
psnrs = []
for i, vid in enumerate(test_vids):
    vpath = os.path.join(VIDEO_DIR, vid)
    orig, rec, _ = get_frame(vpath, 15)
    if orig is None: continue
    psnr = vp.verify_fidelity(orig, rec)["psnr"]
    psnrs.append(psnr)
    save_img(orig, f"closed_{i}_original_{vid}.png")
    save_img(rec, f"closed_{i}_reconstruido_{vid}.png")
    print(f"  [{i}] {vid}: PSNR={psnr:.1f} dB")

print(f"\n  PSNR medio: {np.mean(psnrs):.1f} dB")

# ============================================================
# 3. TEMPORAL: A AGI VE O TEMPO PASSAR?
# ============================================================
print("\n" + "="*60)
print("COERENCIA TEMPORAL: FRAMES CONSECUTIVOS")
print("="*60)

va = os.path.join(VIDEO_DIR, random.choice(videos))
cap = cv2.VideoCapture(va)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
print(f"Video: {os.path.basename(va)}, {total} frames")

frames_mf = []
for i in range(0, min(total, 20), 2):
    orig, _, mf = get_frame(va, i)
    if mf is not None:
        frames_mf.append((i, mf))
        print(f"  Frame {i} processado")

if len(frames_mf) >= 3:
    print("\n  Matriz de similaridade temporal:")
    for i in range(min(5, len(frames_mf))):
        row = []
        for j in range(min(5, len(frames_mf))):
            s = float(torch.cosine_similarity(frames_mf[i][1].tensor.float().unsqueeze(0),
                                              frames_mf[j][1].tensor.float().unsqueeze(0)))
            row.append(f"{s:.3f}")
        print(f"    t={frames_mf[i][0]:3d}: " + "  ".join(row))

# ============================================================
# 4. ONE-SHOT: A AGI RECONHECE O MESMO VIDEO?
# ============================================================
print("\n" + "="*60)
print("ONE-SHOT: A AGI DISTINGUE VIDEOS DIFERENTES?")
print("="*60)

random.seed(42)
va, vb = random.sample(videos, 2)

# Frame 0 e 15 do video A
orig_a0, _, mf_a0 = get_frame(os.path.join(VIDEO_DIR, va), 0)
orig_a15, _, mf_a15 = get_frame(os.path.join(VIDEO_DIR, va), 15)
# Frame 0 do video B
orig_b0, _, mf_b0 = get_frame(os.path.join(VIDEO_DIR, vb), 0)

sim_same = float(torch.cosine_similarity(mf_a0.tensor.float().unsqueeze(0), mf_a15.tensor.float().unsqueeze(0)))
sim_diff = float(torch.cosine_similarity(mf_a0.tensor.float().unsqueeze(0), mf_b0.tensor.float().unsqueeze(0)))

print(f"  Video A: {va}")
print(f"  Video B: {vb}")
print(f"  Similaridade intra-video (A0 vs A15): {sim_same:.4f}")
print(f"  Similaridade inter-video (A0 vs B0):  {sim_diff:.4f}")
print(f"  Margem: {sim_same - sim_diff:.4f}")
save_img(orig_a0, "oneshot_A_frame0.png")
save_img(orig_a15, "oneshot_A_frame15.png")
save_img(orig_b0, "oneshot_B_frame0.png")

print(f"\n  Todas as imagens salvas em: {OUT_DIR}/")
print("="*60)
