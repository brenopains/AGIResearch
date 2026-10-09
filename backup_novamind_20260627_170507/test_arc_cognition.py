import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import time
import sys
import random

from core.grid_cognition import GridWorldModel, GridDomains, TRAIN_FAMS, HELDOUT_FAM, C, H, W, HW

def main():
    print("=== NovaMind ARC Cognition Autonomous Test ===", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    
    # Init domains and model
    domains = GridDomains(dev=device)
    model = GridWorldModel(d=128, heads=4, layers=3).to(device)
    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    B = 16   # Tasks per batch
    K = 3    # Support pairs
    Q = 1    # Query pairs
    STEPS = 200
    
    print(f"Training on base families: {TRAIN_FAMS}", flush=True)
    print(f"Holding out family for OOD test: {HELDOUT_FAM}", flush=True)
    
    model.train()
    start_time = time.time()
    
    for step in range(1, STEPS + 1):
        fam = random.choice(TRAIN_FAMS)
        (xs, ys), (xq, yq) = domains.episode(fam, B, K, Q)
        
        logits = model(xq, (xs, ys)) # [B, Q, C, H, W]
        loss = F.cross_entropy(logits.reshape(B*Q, C, H, W), yq.reshape(B*Q, H, W))
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        if step % 50 == 0 or step == 1:
            # Eval on training batch
            pred = logits.argmax(2).reshape(B, Q, HW)
            acc = (pred == yq).float().mean().item()
            exact = (pred == yq).all(dim=2).float().mean().item()
            print(f"[Train] Step {step:3d}/{STEPS} | Loss: {loss.item():.4f} | Acc: {acc:.1%} | Exact: {exact:.1%}", flush=True)
    
    train_time = time.time() - start_time
    print(f"\nTraining finished in {train_time:.1f}s", flush=True)
    
    print("\n=== ZERO-SHOT GENERALIZATION TEST ===", flush=True)
    model.eval()
    with torch.no_grad():
        # Test on the held-out task (gravity)
        TEST_B = 32
        TEST_K = 3
        (xs, ys), (xq, yq) = domains.episode(HELDOUT_FAM, TEST_B, TEST_K, Q)
        
        logits = model(xq, (xs, ys))
        loss = F.cross_entropy(logits.reshape(TEST_B*Q, C, H, W), yq.reshape(TEST_B*Q, H, W))
        
        pred = logits.argmax(2).reshape(TEST_B, Q, HW)
        
        # Baselines: 
        # Chance: 1/C = 1/6 = 16.6%
        # Copy input: xq == yq
        acc_chance = 1.0 / C
        acc_copy = (xq == yq).float().mean().item()
        
        acc = (pred == yq).float().mean().item()
        exact = (pred == yq).all(dim=2).float().mean().item()
        
        print(f"Domain: {HELDOUT_FAM}", flush=True)
        print(f"OOD Loss: {loss.item():.4f}", flush=True)
        print(f"Pixel Accuracy: {acc:.1%} (vs Chance: {acc_chance:.1%} | vs Copy Input: {acc_copy:.1%})", flush=True)
        print(f"Exact Grade Match: {exact:.1%}", flush=True)
        
        if acc > acc_copy * 1.1:
            print("=> STRONG EVIDENCE of fluid intelligence and rule inference! Model outperformed simple copying.", flush=True)
        else:
            print("=> Model struggles with OOD transfer. Further architectural improvements needed (e.g. neurosymbolic integration).", flush=True)
            
if __name__ == "__main__":
    main()
