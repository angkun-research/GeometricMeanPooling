"""R2: Local cell log-mean regression — parallel across GPUs.

Dataset: single lognormal log(x_i) ~ N(0, 1).
Target: mean_j(exp(mean(log(|x_j|)))) over cells.
Sweep: cell_size ∈ [2, 4, 8, 16], seq_len ∈ [32, 64, 128, 256].
Network: GeometricPool(kernel=cell, stride=cell) → FC(32) → ReLU → FC(1).
Baselines: max/avg/geo pooling.
"""
import subprocess
import sys
import os

cell_sizes = [2, 4, 8, 16]
seq_lens = [32, 64, 128, 256]
seeds = [42, 123, 456]
poolings = ["max", "avg", "geo"]

# Verify divisibility
for cs in cell_sizes:
    for sl in seq_lens:
        assert sl % cs == 0, f"seq_len={sl} not divisible by cell_size={cs}"

tasks = []
for cell_size in cell_sizes:
    for seq_len in seq_lens:
        for seed in seeds:
            for p in poolings:
                tasks.append((cell_size, seq_len, seed, p))

n_gpus = 2
print(f"R2: {len(tasks)} tasks across {n_gpus} GPUs")
for gpu in range(n_gpus):
    gpu_tasks = [t for t in tasks if t[2] % n_gpus == gpu]  # use seed as proxy for distribution
    print(f"  GPU {gpu}: {len(gpu_tasks)} tasks")

# Build code for one GPU
base_code = """
import sys; sys.path.insert(0, '.')
import torch
from torch.utils.data import DataLoader
from tasks.seq1d_regression import LocalCellDataset, LocalCellNetwork

def train_one(cell_size, seq_len, pooling, seed, device):
    train_ds = LocalCellDataset(2000, seq_len, cell_size, sigma=1.0, seed=seed)
    val_ds = LocalCellDataset(500, seq_len, cell_size, sigma=1.0, seed=seed + 1)
    test_ds = LocalCellDataset(1000, seq_len, cell_size, sigma=1.0, seed=seed + 2)

    model = LocalCellNetwork(cell_size=cell_size, pooling=pooling, num_cells=seq_len // cell_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.functional.mse_loss
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=500)

    best_val_mse = float("inf")
    best_state = None
    for epoch in range(1, 201):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out.squeeze(), y.squeeze())
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.eval()
        vloss, vt = 0.0, 0
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            vloss += criterion(model(x).squeeze(), y.squeeze()).item() * x.size(0)
            vt += x.size(0)
        val_mse = vloss / vt
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    tc, vt = 0.0, 0
    preds, trues = [], []
    test_loader = DataLoader(test_ds, batch_size=500)
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        tc += criterion(out.squeeze(), y.squeeze()).item() * x.size(0)
        vt += x.size(0)
        preds.append(out.detach().cpu().squeeze())
        trues.append(y.detach().cpu().squeeze())
    test_mse = tc / vt

    preds_t = torch.cat(preds, dim=0)
    trues_t = torch.cat(trues, dim=0)
    residual_var = torch.var(preds_t - trues_t)
    target_var = torch.var(trues_t)
    r2 = (1 - residual_var / target_var).item() if target_var > 0 else 0.0
    return {"test_r2": r2, "test_mse": test_mse}

results = []
for cell_size, seq_len, seed, pooling, gpu in TASKS_PLACEHOLDER:
    device = torch.device("cuda:" + str(gpu))
    print("cs=" + str(cell_size) + " L=" + str(seq_len) + " p=" + pooling, flush=True)
    r = train_one(cell_size, seq_len, pooling, seed, device)
    results.append((cell_size, seq_len, seed, pooling, r["test_r2"], r["test_mse"]))
    print("  => R2=" + str(round(r["test_r2"], 3)) + " MSE=" + str(round(r["test_mse"], 4)), flush=True)

# Compute means across seeds
print("\\n--- Mean R^2 across seeds ---", flush=True)
for cs in CS_PLACEHOLDER:
    row = "cs=" + str(cs)
    for sl in SL_PLACEHOLDER:
        row += "  "
        for p in POOL_PLACEHOLDER:
            vals = [v[4] for v in results if v[0]==cs and v[1]==sl and v[2] in SEEDS_PLACEHOLDER and v[3]==p]
            if vals:
                row += (str(round(sum(vals)/len(vals), 3)) + ", ")
    print(row, flush=True)

print("\\n--- Mean MSE across seeds ---", flush=True)
for cs in CS_PLACEHOLDER:
    row = "cs=" + str(cs)
    for sl in SL_PLACEHOLDER:
        row += "  "
        for p in POOL_PLACEHOLDER:
            vals = [v[5] for v in results if v[0]==cs and v[1]==sl and v[2] in SEEDS_PLACEHOLDER and v[3]==p]
            if vals:
                row += (str(round(sum(vals)/len(vals), 4)) + ", ")
    print(row, flush=True)
"""

procs = []
for gpu in range(n_gpus):
    gpu_tasks = [(cs, sl, s, p, gpu) for cs in cell_sizes for sl in seq_lens for s in seeds for p in poolings if s % n_gpus == gpu]
    code = base_code.replace("TASKS_PLACEHOLDER", str(gpu_tasks)).replace("SEEDS_PLACEHOLDER", str(seeds)).replace("CS_PLACEHOLDER", str(cell_sizes)).replace("SL_PLACEHOLDER", str(seq_lens)).replace("POOL_PLACEHOLDER", str(poolings))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    logf = open(f"/tmp/r2_gpu{gpu}.log", "w")
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=logf, stderr=logf, env=env, text=True)
    procs.append(proc)
    print(f"GPU {gpu}: {len(gpu_tasks)} tasks (PID {proc.pid})")

for i, proc in enumerate(procs):
    proc.wait()
    with open(f"/tmp/r2_gpu{i}.log") as f:
        print(f"\n=== GPU {i} ===")
        print(f.read())

print("\n=== R2 Complete ===")
