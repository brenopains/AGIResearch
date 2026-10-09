"""Download STL-10, extrai features ResNet18, salva para treino do decoder."""
import sys, os, time, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_BASE)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = r"D:\AGI2\datasets"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs("save", exist_ok=True)

print(f"[EXTRACT] Device: {DEVICE}")
print(f"[EXTRACT] Data: {DATA_DIR}")

# Carrega GroundingEngine para ResNet18
from core.grounding_engine import GroundingEngine
eng = GroundingEngine(device=DEVICE)
eng.feature_extractor.eval()

import torchvision.transforms as T
import torchvision

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("[EXTRACT] Downloading STL-10...")
t0 = time.time()
trainset = torchvision.datasets.STL10(root=DATA_DIR, split="train", download=True, transform=transform)
testset = torchvision.datasets.STL10(root=DATA_DIR, split="test", download=True, transform=transform)
print(f"[EXTRACT] STL-10: {len(trainset)} train, {len(testset)} test (em {(time.time()-t0):.0f}s)")

from torch.utils.data import DataLoader

def extract(dataset, desc="extract"):
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    feats, imgs = [], []
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(loader):
            f = eng.feature_extractor(images.to(DEVICE)).flatten(start_dim=1)
            feats.append(f.cpu())
            # Para imagens de treino, salvamos versao to_tensor (sem normalize)
            imgs.append(images.cpu())  # ja normalizadas
            if (batch_idx + 1) % 10 == 0:
                perc = (batch_idx + 1) * 128 / len(dataset) * 100
                print(f"  {desc}: {min((batch_idx+1)*128, len(dataset))}/{len(dataset)} ({perc:.0f}%)", flush=True)
    return torch.cat(feats), torch.cat(imgs)

print("[EXTRACT] Extraindo features train...")
train_f, train_i = extract(trainset, "train")
print(f"[EXTRACT] Train features: {train_f.shape}")

print("[EXTRACT] Extraindo features test...")
test_f, test_i = extract(testset, "test")
print(f"[EXTRACT] Test features: {test_f.shape}")

save_path = os.path.join(DATA_DIR, "stl10_features.pt")
torch.save({"train_f": train_f, "test_f": test_f, "train_i": train_i, "test_i": test_i}, save_path)
print(f"[EXTRACT] Salvo: {save_path} ({os.path.getsize(save_path)/1024/1024:.0f} MB)")
