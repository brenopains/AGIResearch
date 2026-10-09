"""
Retreina VisualDecoder v2. Carrega dados pre-extraidos do disco.
RAM: ~1.8 GB (uint8), GPU: batches 64, 200 epochs.
"""
import sys, os, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HOME"] = r"D:\novamind_data\cache\huggingface"
os.environ["TORCH_HOME"] = r"D:\novamind_data\cache\torch"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from core.visual_pipeline import VisualDecoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 200
LR = 1e-3
DATA_PATH = r"D:\novamind_data\models\training_data.pt"
SAVE_PATH = r"D:\novamind_data\models\visual_decoder.pt"

print(f"Device: {DEVICE}")

# Carregar dados pre-extraidos
t0 = time.time()
data = torch.load(DATA_PATH, map_location="cpu", weights_only=True)
feats_all = data["feats"]  # (N, 512) float16
imgs_all = data["imgs"]    # (N, 3, 224, 224) uint8
print(f"Carregado: {len(feats_all)} amostras em {time.time()-t0:.0f}s")
print(f"  RAM: feats={feats_all.numel()*feats_all.element_size()/1e9:.1f}GB, imgs={imgs_all.numel()*imgs_all.element_size()/1e9:.1f}GB")

# Split treino/val
n_total = len(feats_all)
n_val = max(1, n_total // 20)
indices = list(range(n_total))
random.seed(42)
random.shuffle(indices)
train_idx, val_idx = indices[n_val:], indices[:n_val]

class FeatDataset(Dataset):
    def __init__(self, feats, imgs, idxs):
        self.feats = feats[idxs].float()  # float32 on CPU
        self.imgs = imgs[idxs].float() / 255.0  # uint8->float32 on CPU
    def __len__(self):
        return len(self.feats)
    def __getitem__(self, idx):
        f = self.feats[idx]
        img = self.imgs[idx]
        if random.random() < 0.5:
            img = torch.flip(img, dims=[2])  # flip horizontal (so CPU, leve)
        return f, img

train_ds = FeatDataset(feats_all, imgs_all, train_idx)
val_ds = FeatDataset(feats_all, imgs_all, val_idx)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Treino: {len(train_ds)}, Val: {len(val_ds)}")

decoder = VisualDecoder().to(DEVICE)
optimizer = torch.optim.AdamW(decoder.parameters(), lr=LR, weight_decay=1e-5)
loss_fn = nn.MSELoss()
best_val_loss = float("inf")
t_start = time.time()

for epoch in range(EPOCHS):
    decoder.train()
    train_loss = 0
    for feats, imgs in train_dl:
        pred = decoder(feats.to(DEVICE))
        loss = loss_fn(pred, imgs.to(DEVICE))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_dl)

    decoder.eval()
    val_loss = 0
    psnrs = []
    with torch.no_grad():
        for feats, imgs in val_dl:
            pred = decoder(feats.to(DEVICE))
            loss = loss_fn(pred, imgs.to(DEVICE))
            val_loss += loss.item()
            for i in range(len(imgs)):
                mse = float(F.mse_loss(pred[i].cpu(), imgs[i]))
                psnrs.append(20 * np.log10(1.0 / np.sqrt(mse + 1e-10)))
    val_loss /= len(val_dl)
    psnr_avg = np.mean(psnrs)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(decoder.state_dict(), SAVE_PATH)
        print(f"  Ep {epoch+1}/{EPOCHS} train={train_loss:.6f} val={val_loss:.6f} PSNR={psnr_avg:.1f} (melhor, salvo)")
    elif (epoch + 1) % 10 == 0:
        print(f"  Ep {epoch+1}/{EPOCHS} train={train_loss:.6f} val={val_loss:.6f} PSNR={psnr_avg:.1f} ({time.time()-t_start:.0f}s)")

print(f"\nConcluido em {time.time()-t_start:.0f}s, melhor val_loss={best_val_loss:.6f}")
print(f"Decoder salvo em: {SAVE_PATH}")
