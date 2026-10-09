"""
Treino do Broca V2 — LSTM Decoder HDC → Texto
================================================
Treina o modelo no simplewiki.txt (487MB) com teacher forcing.

Uso:
    python -m core.broca_v2.train

Isso vai:
  1. Treinar tokenizador BPE no corpus
  2. Treinar modelo LSTM por ~5 épocas
  3. Salvar checkpoint em D:/temp/broca_v2/
"""

import os
import sys
import time
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.broca_v2.tokenizer import BPETokenizer
from core.broca_v2.model import BrocaV2Model

# ─── Configuração ──────────────────────────────────────────────

CORPUS_PATH = "D:/novamind_data/temp/simplewiki_clean.txt"
OUTPUT_DIR = "D:/temp/broca_v2"
TOKENIZER_PATH = os.path.join(OUTPUT_DIR, "tokenizer.json")
MODEL_PATH = os.path.join(OUTPUT_DIR, "broca_v2.pt")
CONFIG_PATH = os.path.join(OUTPUT_DIR, "config.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
SEQ_LENGTH = 128
VOCAB_SIZE = 32768
HDC_DIM = 10000
EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.3
LR = 3e-4
EPOCHS = 8
MAX_TRAIN_LINES = 240000
MAX_EVAL_LINES = 10000
LOG_INTERVAL = 100
SAVE_INTERVAL = 1000

# ─── Dataset ───────────────────────────────────────────────────

class TextDataset(Dataset):
    """Dataset de texto para teacher forcing."""

    def __init__(self, corpus_path: str, tokenizer: BPETokenizer,
                 seq_length: int = 128, max_lines: int = 400000, offset: int = 0):
        self.seq_length = seq_length
        self.tokenizer = tokenizer

        lines = []
        with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i < offset:
                    continue
                if len(lines) >= max_lines:
                    break
                line = line.strip()
                if len(line) < 20:
                    continue
                lines.append(line)

        self.encodings = []
        for line in lines:
            ids = tokenizer.encode(line, add_special=True)
            self.encodings.append(torch.tensor(ids, dtype=torch.long))

        print(f"[DATASET] {len(self.encodings)} sequencias (max_lines={max_lines}, offset={offset})")

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        ids = self.encodings[idx]
        if len(ids) > self.seq_length:
            start = torch.randint(0, len(ids) - self.seq_length, (1,)).item()
            ids = ids[start:start + self.seq_length]
        else:
            pad_len = self.seq_length - len(ids)
            ids = torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)])

        input_ids = ids[:-1].clone()
        target_ids = ids[1:].clone()
        return input_ids, target_ids


def collate_fn(batch):
    inputs = torch.stack([b[0] for b in batch])
    targets = torch.stack([b[1] for b in batch])
    return inputs, targets


# ─── Treino ────────────────────────────────────────────────────

class Trainer:
    def __init__(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.device = torch.device(DEVICE)
        print(f"[BROCA] Device: {self.device}")
        print(f"[BROCA] Output: {OUTPUT_DIR}")

        # Tokenizador
        self.tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE)
        if os.path.exists(TOKENIZER_PATH):
            self.tokenizer = BPETokenizer.load(TOKENIZER_PATH)
            print(f"[BROCA] Tokenizador carregado: {len(self.tokenizer)} tokens")
        else:
            self.tokenizer.train(CORPUS_PATH, max_lines=MAX_TRAIN_LINES)
            self.tokenizer.save(TOKENIZER_PATH)

        # Modelo
        self.model = BrocaV2Model(
            vocab_size=self.tokenizer.vocab_size_real,
            hdc_dim=HDC_DIM,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        ).to(self.device)

        # Otimizador
        self.optimizer = AdamW(self.model.parameters(), lr=LR, weight_decay=0.01)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=EPOCHS * 1000)

        self.start_epoch = 0
        self.global_step = 0

        # Carregar checkpoint se existir
        if os.path.exists(MODEL_PATH):
            self._load_checkpoint()

        # Dataset
        self.train_dataset = TextDataset(
            CORPUS_PATH, self.tokenizer, SEQ_LENGTH, MAX_TRAIN_LINES, offset=0)
        self.eval_dataset = TextDataset(
            CORPUS_PATH, self.tokenizer, SEQ_LENGTH, MAX_EVAL_LINES, offset=MAX_TRAIN_LINES)

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            collate_fn=collate_fn, num_workers=0, pin_memory=True)
        self.eval_loader = DataLoader(
            self.eval_dataset, batch_size=BATCH_SIZE, shuffle=False,
            collate_fn=collate_fn, num_workers=0)

    def _load_checkpoint(self):
        print(f"[BROCA] Carregando checkpoint: {MODEL_PATH}")
        ckpt = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.global_step = ckpt.get("step", 0)
        print(f"[BROCA] Retomando do epoch {self.start_epoch}, step {self.global_step}")

    def _save_checkpoint(self, epoch, step, loss):
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "config": {
                "vocab_size": self.tokenizer.vocab_size_real,
                "hdc_dim": HDC_DIM,
                "embed_dim": EMBED_DIM,
                "hidden_dim": HIDDEN_DIM,
                "num_layers": NUM_LAYERS,
            },
        }
        torch.save(ckpt, MODEL_PATH)
        print(f"[BROCA] Checkpoint salvo: {MODEL_PATH} (step={step}, loss={loss:.4f})")

    def train(self):
        print(f"[BROCA] Iniciando treino por {EPOCHS} épocas")
        print(f"[BROCA] Train: {len(self.train_dataset)} amostras, Eval: {len(self.eval_dataset)}")
        print(f"[BROCA] Parametros: {sum(p.numel() for p in self.model.parameters()):,}")

        total_steps = len(self.train_loader) * EPOCHS
        best_eval_loss = float("inf")

        for epoch in range(self.start_epoch, EPOCHS):
            self.model.train()
            epoch_loss = 0.0
            epoch_start = time.time()

            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # HDC vector dummy durante treino (será substituído na inferência real)
                # Durante treino, usamos um vetor zero (o modelo aprenderá a gerar
                # texto condicionado apenas no contexto linguístico)
                batch_size = inputs.shape[0]
                hdc_vec = torch.zeros(batch_size, HDC_DIM, device=self.device)
                hdc_vec[:, 0] = 1.0  # marcador de "modo texto livre"

                self.optimizer.zero_grad()

                logits, _ = self.model(hdc_vec, inputs)
                loss = nn.CrossEntropyLoss(ignore_index=0)(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += loss.item()
                self.global_step += 1

                if batch_idx % LOG_INTERVAL == 0 and batch_idx > 0:
                    avg = epoch_loss / (batch_idx + 1)
                    lr_now = self.scheduler.get_last_lr()[0]
                    elapsed = time.time() - epoch_start
                    progress = self.global_step / total_steps * 100
                    print(
                        f"  E{epoch+1}/{EPOCHS} "
                        f"B{batch_idx}/{len(self.train_loader)} "
                        f"loss={avg:.4f} "
                        f"lr={lr_now:.2e} "
                        f"step={self.global_step} "
                        f"({progress:.1f}%) "
                        f"elapsed={elapsed:.0f}s"
                    )

                if batch_idx > 0 and batch_idx % SAVE_INTERVAL == 0:
                    self._save_checkpoint(epoch, self.global_step, avg)

            avg_epoch_loss = epoch_loss / len(self.train_loader)

            # Avaliação
            eval_loss = self._evaluate()

            print(
                f"[BROCA] Epoch {epoch+1}: "
                f"train_loss={avg_epoch_loss:.4f} "
                f"eval_loss={eval_loss:.4f} "
                f"elapsed={time.time()-epoch_start:.0f}s"
            )

            self._save_checkpoint(epoch, self.global_step, avg_epoch_loss)

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                torch.save(self.model.state_dict(),
                           os.path.join(OUTPUT_DIR, "broca_v2_best.pt"))
                print(f"[BROCA] Novo melhor modelo! eval_loss={eval_loss:.4f}")

        # Salvar configuração final
        with open(CONFIG_PATH, "w") as f:
            json.dump({
                "vocab_size": self.tokenizer.vocab_size_real,
                "hdc_dim": HDC_DIM,
                "embed_dim": EMBED_DIM,
                "hidden_dim": HIDDEN_DIM,
                "num_layers": NUM_LAYERS,
                "epochs": EPOCHS,
                "final_train_loss": avg_epoch_loss,
                "final_eval_loss": eval_loss,
            }, f, indent=2)

        print(f"[BROCA] Treino completo! Modelo em {MODEL_PATH}")
        print(f"[BROCA] Tokenizador em {TOKENIZER_PATH}")

    def _evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for inputs, targets in self.eval_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                batch_size = inputs.shape[0]
                hdc_vec = torch.zeros(batch_size, HDC_DIM, device=self.device)
                hdc_vec[:, 0] = 1.0

                logits, _ = self.model(hdc_vec, inputs)
                loss = nn.CrossEntropyLoss(ignore_index=0)(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                )
                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(1, n_batches)

    def test_generation(self, prompt: str = "The meaning of life is", max_length: int = 50):
        """Testa geração após treino."""
        self.model.eval()
        ids = self.tokenizer.encode(prompt, add_special=True)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)

        hdc_vec = torch.zeros(1, HDC_DIM, device=self.device)
        hdc_vec[:, 0] = 1.0

        generated = self.model.generate(
            hdc_vec, max_length=max_length, temperature=0.7, top_k=30)

        text = self.tokenizer.decode(generated[0].tolist())
        print(f"\n[TEST] Prompt: {prompt}")
        print(f"[TEST] Gerado: {text}")
        return text


# ─── Entrypoint ────────────────────────────────────────────────

def main():
    trainer = Trainer()
    trainer.train()
    trainer.test_generation()
    trainer.test_generation("Artificial intelligence is")
    trainer.test_generation("The theory of evolution states")


if __name__ == "__main__":
    main()
