"""
Extracao unica de dados de treinamento do MSVD.
Salva features (float16) e imagens (uint8) em disco para treino rapido GPU.
Uso: python extract_training_data.py (uma vez, ~5-10 min CPU)
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HOME"] = r"D:\novamind_data\cache\huggingface"
os.environ["TORCH_HOME"] = r"D:\novamind_data\cache\torch"

import torch
import cv2
import numpy as np
from core.grounding_engine import GroundingEngine

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRAMES_POR_VIDEO = 10
DATASET_DIR = r"D:\novamind_data\MSVD"
CLIPS_DIR = os.path.join(DATASET_DIR, "YouTubeClips")
META_PATH = os.path.join(DATASET_DIR, "msvd_train.json")
OUT_PATH = r"D:\novamind_data\models\training_data.pt"

print(f"Device: {DEVICE}")
eng = GroundingEngine(device=torch.device(DEVICE))

with open(META_PATH, "r") as f:
    metadata = json.load(f)
print(f"Total de videos no MSVD: {len(metadata)}")

feats_list, imgs_list = [], []
t0 = time.time()
n_ok, n_fail = 0, 0

for idx, record in enumerate(metadata):
    vpath = os.path.join(CLIPS_DIR, record["video"])
    if not os.path.exists(vpath):
        n_fail += 1; continue

    cap = cv2.VideoCapture(vpath)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release(); n_fail += 1; continue

    sample_indices = np.linspace(0, total-1, min(FRAMES_POR_VIDEO, total), dtype=int)
    for fi in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret: continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = eng.transform(frame_rgb).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feats = eng.feature_extractor(img_tensor).view(-1).half().cpu()  # float16
        img_resized = cv2.resize(frame_rgb, (224, 224))  # uint8 (0-255)
        img_t = torch.from_numpy(img_resized).permute(2, 0, 1).byte()  # uint8
        feats_list.append(feats)
        imgs_list.append(img_t)
        n_ok += 1
    cap.release()
    if (idx + 1) % 200 == 0:
        print(f"  [{idx+1}/{len(metadata)}] {n_ok} frames OK, {n_fail} falhas, {time.time()-t0:.0f}s")

# Salvar compactado
feats = torch.stack(feats_list)   # (N, 512) float16
imgs = torch.stack(imgs_list)     # (N, 3, 224, 224) uint8
torch.save({"feats": feats, "imgs": imgs}, OUT_PATH)
gb = (feats.numel() * feats.element_size() + imgs.numel() * imgs.element_size()) / 1e9
print(f"\nExtraido: {n_ok} frames, {n_fail} falhas, {time.time()-t0:.0f}s")
print(f"Salvo em: {OUT_PATH} ({gb:.1f} GB)")
