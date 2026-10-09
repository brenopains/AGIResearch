"""
COCO 2017 — Treino do VisualDecoder com ResNet18 features.

Pipeline:
  1. Extrai features ResNet18 (512D) das imagens COCO
  2. Salva features + indice de arquivos em D:\AGI2\treino\
  3. Cria DataLoader que carrega imagens do disco sob demanda
  4. Treina VisualDecoder (30 epochs, ~5h GPU)
  5. Salva checkpoints

Uso:
  python train_coco.py
"""

import os, sys, time, json, pickle
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np

# ─── Config ──────────────────────────────────────────────────
COCO_ROOT = r"D:\AGI2\datasets\coco2017"
FEATURE_DIR = r"D:\AGI2\treino"
SAVE_DIR = r"D:\AGI2\treino\save"
MODEL_DEST = ('C:\\Users\\usuario\\Desktop\\AGI 2\\novamind\\save').replace('C:\\Users\\usuario', __import__('os').path.expanduser('~'))
LOG_FILE = r"D:\AGI2\coco_log.txt"
BATCH_SIZE_FEAT = 64
BATCH_SIZE_TRAIN = 64
EPOCHS = 30
LR = 1e-3
NUM_TRAIN = 50000
NUM_VAL = 5000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(MODEL_DEST, exist_ok=True)


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ─── Extração de features (APENAS features, sem acumular imagens) ──
def extract_features(split, num_images, feat_file, index_file):
    img_dir = os.path.join(COCO_ROOT, split)
    files = sorted([f for f in os.listdir(img_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))])[:num_images]

    log(f"Extraindo features {split} ({len(files)} imagens)...")

    # Salvar indice (ordem dos arquivos)
    with open(index_file, "w") as f:
        for name in files:
            f.write(name + "\n")

    # ResNet18
    import torchvision.models as models
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(DEVICE)
    resnet.eval()
    feat_ext = nn.Sequential(*list(resnet.children())[:-1])

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Memmap para salvar features sem acumular em RAM
    mm = np.memmap(feat_file, dtype=np.float32, mode='write',
                   shape=(len(files), 512))

    batch_f = []
    batch_i = 0
    start = time.time()
    last_log = 0

    for idx, fname in enumerate(files):
        try:
            path = os.path.join(img_dir, fname)
            img = Image.open(path).convert("RGB")
            batch_f.append(transform(img))
        except Exception as e:
            log(f"  AVISO: pulando {fname}: {e}")
            continue

        if len(batch_f) == BATCH_SIZE_FEAT or idx == len(files) - 1:
            x = torch.stack(batch_f).to(DEVICE)
            with torch.no_grad():
                feats = feat_ext(x).squeeze(-1).squeeze(-1)
            mm[batch_i:batch_i + len(batch_f)] = feats.cpu().numpy()
            mm.flush()
            batch_i += len(batch_f)
            batch_f = []

            done = min(batch_i, len(files))
            if done - last_log >= 3200 or idx == len(files) - 1:
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0
                log(f"  {split}: {done}/{len(files)} [{elapsed:.0f}s, {rate:.0f} img/s]")
                last_log = done

    mm.flush()
    del mm
    elapsed = time.time() - start
    log(f"  {split} concluido: {len(files)} imagens em {elapsed:.0f}s ({len(files)/elapsed:.0f} img/s)")


# ─── Dataset de treino (carrega imagens do disco sob demanda) ──
class COCOTrainDataset(Dataset):
    def __init__(self, split, feat_file, index_file, num_images):
        self.img_dir = os.path.join(COCO_ROOT, split)
        with open(index_file) as f:
            self.files = [line.strip() for line in f if line.strip()][:num_images]
        self.features = np.memmap(feat_file, dtype=np.float32, mode='r', shape=(num_images, 512))
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        feat = torch.from_numpy(self.features[idx].copy()).float()
        path = os.path.join(self.img_dir, self.files[idx])
        img = Image.open(path).convert("RGB")
        img_t = self.transform(img)
        return feat, img_t


# ─── VisualDecoder ──────────────────────────────────────────
from core.visual_pipeline import VisualPipeline


def train_decoder():
    log("Inicializando VisualDecoder...")
    proj = torch.randn(10000, 512, device=DEVICE) / np.sqrt(512)
    pipeline = VisualPipeline(proj, DEVICE)

    train_ds = COCOTrainDataset(
        "train2017",
        os.path.join(FEATURE_DIR, "coco_train_features.dat"),
        os.path.join(FEATURE_DIR, "coco_train_index.txt"),
        NUM_TRAIN
    )
    val_ds = COCOTrainDataset(
        "val2017",
        os.path.join(FEATURE_DIR, "coco_val_features.dat"),
        os.path.join(FEATURE_DIR, "coco_val_index.txt"),
        NUM_VAL
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_TRAIN,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE_TRAIN,
                            shuffle=False, num_workers=2, pin_memory=True)

    optimizer = torch.optim.Adam(pipeline.decoder.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    best_psnr = 0.0
    start = time.time()

    for epoch in range(EPOCHS):
        pipeline.decoder.train()
        total_loss = 0
        n_batches = 0
        for feats, imgs in train_loader:
            f = feats.to(DEVICE)
            i = imgs.to(DEVICE)
            pred = pipeline.decoder(f)
            loss = loss_fn(pred, i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        epoch_psnr = 20 * np.log10(1.0 / np.sqrt(avg_loss + 1e-10))

        pipeline.decoder.eval()
        val_loss = 0
        v_batches = 0
        with torch.no_grad():
            for feats, imgs in val_loader:
                f = feats.to(DEVICE)
                i = imgs.to(DEVICE)
                pred = pipeline.decoder(f)
                val_loss += loss_fn(pred, i).item()
                v_batches += 1
        val_avg = val_loss / max(v_batches, 1)
        val_psnr = 20 * np.log10(1.0 / np.sqrt(val_avg + 1e-10))
        elapsed = time.time() - start
        remaining = (elapsed / (epoch + 1)) * (EPOCHS - epoch - 1) / 60

        log(f"  Epoch {epoch+1}/{EPOCHS} | loss={avg_loss:.6f} PSNR={epoch_psnr:.1f} | val_loss={val_avg:.6f} val_PSNR={val_psnr:.1f} | {elapsed:.0f}s | ~{remaining:.0f}min restantes")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            ckpt = {
                "epoch": epoch, "model_state": pipeline.decoder.state_dict(),
                "optimizer_state": optimizer.state_dict(), "val_psnr": val_psnr,
                "train_psnr": epoch_psnr
            }
            torch.save(ckpt, os.path.join(SAVE_DIR, "coco_decoder_best.pt"))
            torch.save(ckpt, os.path.join(MODEL_DEST, "visual_decoder.pt"))
            log(f"  >> NOVO BEST: val_PSNR={val_psnr:.1f}")

    total_time = time.time() - start
    log(f"Treino concluido em {total_time:.0f}s ({total_time/60:.1f} min)")
    log(f"Melhor val_PSNR: {best_psnr:.1f}")
    return best_psnr


# ─── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    log("=" * 60)
    log("COCO 2017 — TREINO DO VISUAL DECODER")
    log(f"Device: {DEVICE}")
    log(f"Train: {NUM_TRAIN} | Val: {NUM_VAL} | Epochs: {EPOCHS}")
    log("=" * 60)

    # Extrair features (se não existirem)
    train_feat = os.path.join(FEATURE_DIR, "coco_train_features.dat")
    train_idx = os.path.join(FEATURE_DIR, "coco_train_index.txt")
    val_feat = os.path.join(FEATURE_DIR, "coco_val_features.dat")
    val_idx = os.path.join(FEATURE_DIR, "coco_val_index.txt")

    if not os.path.exists(train_feat):
        extract_features("train2017", NUM_TRAIN, train_feat, train_idx)
    else:
        log("Features de treino ja existem.")

    if not os.path.exists(val_feat):
        extract_features("val2017", NUM_VAL, val_feat, val_idx)
    else:
        log("Features de validacao ja existem.")

    best = train_decoder()
    log(f"\nRESULTADO FINAL: Melhor PSNR = {best:.1f} dB")
    log("=" * 60)
