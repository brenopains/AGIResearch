"""
Treino do GridWorldModel com ARC real + datasets de geometria/transformacao.

Uso:
    python train_geometry_arc.py --steps 300     # treino completo
    python train_geometry_arc.py --smoke          # 50 steps rapido
    python train_geometry_arc.py --eval-only      # so avalia o modelo atual
"""
import os, sys, time, json, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.grid_cognition import GridWorldModel
from core.arc_dataset import ARCDataset
from core.geometry_dataset import GeometryTransformDataset, RelationalDataset


def pad_grid_for_model(grid: np.ndarray, max_h: int = 30, max_w: int = 30, pad_val: int = 10):
    """Padding de grid numpy para tensor do modelo."""
    h, w = grid.shape
    padded = np.full((max_h, max_w), pad_val, dtype=np.int64)
    h_end = min(h, max_h)
    w_end = min(w, max_w)
    padded[:h_end, :w_end] = grid[:h_end, :w_end]
    return padded.reshape(-1)  # (900,)


def train_step(model, opt, xs, ys, xq, yq):
    """Um passo de treino."""
    logits = model(xq, (xs, ys))
    loss = F.cross_entropy(
        logits.reshape(-1, 11, 900),
        yq.reshape(-1, 900),
        ignore_index=10,
    )
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item()


def evaluate(model, dataset, n_tasks: int = 50) -> dict:
    """Avalia o modelo em N tasks ARC."""
    model.eval()
    exact = 0
    total_pixels = 0
    correct_pixels = 0
    total_loss = 0.0

    with torch.no_grad():
        for task_idx in range(min(n_tasks, len(dataset.tasks))):
            name, task = dataset.tasks[task_idx]
            train_pairs = task["train"]
            test_pairs = task["test"]
            K = min(3, len(train_pairs))

            xs_list = [dataset.pad_grid(train_pairs[k % len(train_pairs)]["input"]) for k in range(K)]
            ys_list = [dataset.pad_grid(train_pairs[k % len(train_pairs)]["output"]) for k in range(K)]
            xs = torch.stack(xs_list).unsqueeze(0).to(next(model.parameters()).device)
            ys = torch.stack(ys_list).unsqueeze(0).to(next(model.parameters()).device)

            if test_pairs:
                q_pair = test_pairs[0]
                xq = dataset.pad_grid(q_pair["input"]).unsqueeze(0).unsqueeze(0).to(next(model.parameters()).device)
                yq = dataset.pad_grid(q_pair["output"]).unsqueeze(0).unsqueeze(0).to(next(model.parameters()).device)

                logits = model(xq, (xs, ys))
                loss = F.cross_entropy(
                    logits.reshape(1, 11, 900), yq.reshape(1, 900), ignore_index=10
                )
                total_loss += loss.item()

                pred = logits.argmax(dim=2).reshape(1, 1, 900)
                pred_np = pred[0, 0].cpu().numpy()
                yq_np = yq[0, 0].cpu().numpy()

                valid = yq_np != 10
                correct = (pred_np[valid] == yq_np[valid]).sum()
                total_pixels += valid.sum()
                correct_pixels += correct

                if (pred_np == yq_np).all():
                    exact += 1

    acc = correct_pixels / max(1, total_pixels)
    avg_loss = total_loss / max(1, n_tasks)
    return {"pixel_accuracy": float(acc), "exact_match": exact, "avg_loss": avg_loss, "n_tasks": min(n_tasks, len(dataset.tasks))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300, help="Numero de steps de treino")
    parser.add_argument("--smoke", action="store_true", help="50 steps rapidos")
    parser.add_argument("--eval-only", action="store_true", help="So avalia, nao treina")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--geo-ratio", type=float, default=0.3, help="Proporcao de dados sinteticos")
    parser.add_argument("--output", default="arc_training_report.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Datasets
    print("\n[Datasets]")
    arc_train = ARCDataset(data_dir="data/ARC/data/training", max_h=30, max_w=30, pad_val=10, dev=device)
    arc_eval = ARCDataset(data_dir="data/ARC/data/evaluation", max_h=30, max_w=30, pad_val=10, dev=device)
    geo = GeometryTransformDataset(size=5000, h=10, w=10)
    rel = RelationalDataset(size=2000, h=10, w=10)
    print(f"  ARC train: {len(arc_train.tasks)} tasks")
    print(f"  ARC eval:  {len(arc_eval.tasks)} tasks")
    print(f"  Geometry:  {len(geo)} grids")
    print(f"  Relational: {len(rel)} grids")

    # Model
    model = GridWorldModel(d=128, heads=4, layers=3, max_hw=900, colors=11).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"\n[Model] {params:,} parameters")

    model_path = "arc_model_trained.pt"
    if os.path.exists(model_path) and not args.eval_only:
        ckpt = torch.load(model_path, map_location=device)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif isinstance(ckpt, dict):
            model.load_state_dict(ckpt, strict=False)
        print(f"  Checkpoint loaded: {model_path}")

    if args.eval_only:
        results = evaluate(model, arc_eval, n_tasks=50)
        print(f"\n[Eval Only] Pixel Acc: {results['pixel_accuracy']:.2%} | Exact: {results['exact_match']}")
        return

    # Training
    steps = 50 if args.smoke else args.steps
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    geo_ratio = args.geo_ratio

    print(f"\n[Training] {steps} steps, geo_ratio={geo_ratio}")
    t_start = time.time()

    for step in range(1, steps + 1):
        model.train()

        # Decide se usa ARC real ou dado sintetico
        if random.random() < geo_ratio:
            # Batch sintetico
            batch_size = 32
            geo_batch = geo.sample_batch(batch_size // 2)
            rel_batch = rel.sample_batch(batch_size // 4)

            xs_list, ys_list, xq_list, yq_list = [], [], [], []
            for inp, out, label, _ in geo_batch:
                inp_pad = pad_grid_for_model(inp)
                out_pad = pad_grid_for_model(out)
                xs_list.append(torch.tensor(inp_pad, dtype=torch.long))
                ys_list.append(torch.tensor(out_pad, dtype=torch.long))
                xq_list.append(torch.tensor(inp_pad, dtype=torch.long))
                yq_list.append(torch.tensor(out_pad, dtype=torch.long))

            for inp, out, label, _ in rel_batch:
                inp_pad = pad_grid_for_model(inp)
                out_pad = pad_grid_for_model(out)
                xs_list.append(torch.tensor(inp_pad, dtype=torch.long))
                ys_list.append(torch.tensor(out_pad, dtype=torch.long))
                xq_list.append(torch.tensor(inp_pad, dtype=torch.long))
                yq_list.append(torch.tensor(out_pad, dtype=torch.long))

            if not xs_list:
                continue

            K = 3
            xs = torch.stack(xs_list).unsqueeze(1).repeat(1, K, 1).to(device)
            ys = torch.stack(ys_list).unsqueeze(1).repeat(1, K, 1).to(device)
            xq = torch.stack(xq_list).unsqueeze(1).to(device)
            yq = torch.stack(yq_list).unsqueeze(1).to(device)
        else:
            # Batch ARC real
            (xs_b, ys_b), (xq_b, yq_b) = arc_train.sample_batch(16, 3, 1)
            xs, ys = xs_b.to(device), ys_b.to(device)
            xq, yq = xq_b.to(device), yq_b.to(device)

        loss = train_step(model, opt, xs, ys, xq, yq)

        if step % 50 == 0 or step == 1:
            elapsed = time.time() - t_start
            print(f"  Step {step:3d}/{steps} | Loss: {loss:.4f} | {elapsed:.0f}s", flush=True)

    train_time = time.time() - t_start
    print(f"\nTraining completed in {train_time:.1f}s")

    # Save
    torch.save({"model_state_dict": model.state_dict()}, model_path)
    print(f"Model saved: {model_path}")

    # Evaluate
    print("\n[Evaluation]")
    eval_results = evaluate(model, arc_eval, n_tasks=50)
    print(f"  Pixel Accuracy: {eval_results['pixel_accuracy']:.2%}")
    print(f"  Exact Match: {eval_results['exact_match']}/{eval_results['n_tasks']}")
    print(f"  Avg Loss: {eval_results['avg_loss']:.4f}")

    # Train accuracy for comparison
    train_results = evaluate(model, arc_train, n_tasks=50)
    print(f"\n  Train Pixel Acc: {train_results['pixel_accuracy']:.2%}")
    print(f"  Train Exact: {train_results['exact_match']}/{train_results['n_tasks']}")

    # Report
    report = {
        "steps": steps,
        "train_time_s": round(train_time, 1),
        "geo_ratio": geo_ratio,
        "eval": {k: round(v, 4) if isinstance(v, float) else v for k, v in eval_results.items()},
        "train": {k: round(v, 4) if isinstance(v, float) else v for k, v in train_results.items()},
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {args.output}")


if __name__ == "__main__":
    import random
    main()
