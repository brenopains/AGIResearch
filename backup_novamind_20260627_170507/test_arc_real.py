import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import time
import sys

from core.grid_cognition import GridWorldModel
from core.arc_dataset import ARCDataset

def main():
    print("=== NovaMind Real ARC Evaluation ===", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    
    # Init domains and model
    # H=30, W=30 => 900 tokens. C=11 (0-9 for ARC colors, 10 for PAD)
    dataset = ARCDataset(data_dir="data/ARC/data/training", max_h=30, max_w=30, pad_val=10, dev=device)
    
    model = GridWorldModel(d=128, heads=4, layers=3, max_hw=900, colors=11).to(device)
    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    B = 16   # Tasks per batch
    K = 3    # Support pairs
    Q = 1    # Query pairs
    STEPS = 500 # We need enough steps to learn the embeddings and attention patterns for real ARC tasks
    
    model.train()
    start_time = time.time()
    
    print("\n--- Training on ARC Real Subset ---", flush=True)
    for step in range(1, STEPS + 1):
        (xs, ys), (xq, yq) = dataset.sample_batch(B, K, Q)
        
        logits = model(xq, (xs, ys)) # [B, Q, C, hw]
        
        # Cross entropy with ignore_index=10 (padding)
        loss = F.cross_entropy(logits.reshape(B*Q, 11, 900), yq.reshape(B*Q, 900), ignore_index=10)
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        if step % 50 == 0 or step == 1:
            pred = logits.argmax(2).reshape(B, Q, 900)
            
            # Mask out padding for accuracy calculation
            valid_mask = (yq != 10)
            
            acc = ((pred == yq) & valid_mask).float().sum() / valid_mask.float().sum().clamp(min=1)
            
            # Exact match (all valid pixels match)
            correct_pixels = (pred == yq) | ~valid_mask
            exact = correct_pixels.all(dim=2).float().mean().item()
            
            print(f"[Train] Step {step:3d}/{STEPS} | Loss: {loss.item():.4f} | Valid Acc: {acc:.1%} | Exact: {exact:.1%}", flush=True)
            
    train_time = time.time() - start_time
    print(f"\nTraining finished in {train_time:.1f}s", flush=True)
    
    print("\n=== EVALUATING ON REAL ARC (Validation Set) ===", flush=True)
    model.eval()
    val_dataset = ARCDataset(data_dir="data/ARC/data/evaluation", max_h=30, max_w=30, pad_val=10, dev=device)
    
    with torch.no_grad():
        TEST_B = 32
        TEST_K = 3
        (xs, ys), (xq, yq) = val_dataset.sample_batch(TEST_B, TEST_K, Q)
        
        logits = model(xq, (xs, ys))
        loss = F.cross_entropy(logits.reshape(TEST_B*Q, 11, 900), yq.reshape(TEST_B*Q, 900), ignore_index=10)
        
        pred = logits.argmax(2).reshape(TEST_B, Q, 900)
        
        valid_mask = (yq != 10)
        acc = ((pred == yq) & valid_mask).float().sum() / valid_mask.float().sum().clamp(min=1)
        
        # Baselines
        acc_chance = 1.0 / 10 # 10 colors
        acc_copy = ((xq == yq) & valid_mask).float().sum() / valid_mask.float().sum().clamp(min=1)
        
        correct_pixels = (pred == yq) | ~valid_mask
        exact = correct_pixels.all(dim=2).float().mean().item()
        
        print(f"OOD Loss (Evaluation Set): {loss.item():.4f}", flush=True)
        print(f"Valid Pixel Accuracy: {acc:.1%} (vs Chance: {acc_chance:.1%} | vs Copy Input: {acc_copy:.1%})", flush=True)
        print(f"Exact Grade Match (Task Solved): {exact:.1%}", flush=True)
        
        if exact > 0:
            print("=> BREAKTHROUGH: System solved at least one real ARC task out-of-distribution without symbolic solver!", flush=True)
        elif acc > acc_copy * 1.1:
            print("=> Model shows structural fluid intelligence but needs symbolic solver for exact logic.", flush=True)
        else:
            print("=> Model struggling with real ARC tasks. A hybrid neuro-symbolic engine is required.", flush=True)

if __name__ == "__main__":
    main()
