"""R4: Launch one process per GPU, each runs its share of tasks."""
import subprocess
import sys
import os

seq_lens = [8, 16, 32, 64, 128, 256]
seeds = [42, 123, 456]
poolings = ["max", "avg", "geo"]
sigma, sp = 1.0, 0.25

tasks = []
for seed in seeds:
    for seq_len in seq_lens:
        for p in poolings:
            tasks.append((seed, seq_len, p))

n_gpus = 2
print(f"R4: {len(tasks)} tasks across {n_gpus} GPUs")

# Build one Python script that runs a subset of tasks
base_code = """
import sys; sys.path.insert(0, '.')
import torch
from run_seq1d_regression import _train_one
from tasks.seq1d_regression import GaussianProductDataset

torch.cuda.set_device(0)
results = []
for seed, seq_len, pooling, sigma, sp in TASKS_PLACEHOLDER:
    device = torch.device("cuda:0")
    print("L=" + str(seq_len) + " p=" + pooling, flush=True)
    train_ds = GaussianProductDataset(2000, seq_len, seed, noise_sigma=sigma, noise_sparsity=sp, unsigned=True)
    val_ds = GaussianProductDataset(500, seq_len, seed + 1, noise_sigma=sigma, noise_sparsity=sp, unsigned=True)
    test_ds = GaussianProductDataset(1000, seq_len, seed + 2, noise_sigma=sigma, noise_sparsity=sp, unsigned=True)
    r = _train_one(pooling, train_ds, val_ds, test_ds, epochs=200, lr=0.01, device=device)
    results.append((seed, seq_len, pooling, r["test_r2"], r["test_mse"]))
    print("  => R2=" + str(round(r["test_r2"], 3)) + " MSE=" + str(round(r["test_mse"], 4)), flush=True)

# Print table
seeds = SEEDS_PLACEHOLDER
seq_lens = SL_PLACEHOLDER
poolings = POOL_PLACEHOLDER
print("\\n--- R^2 ---")
for s in seeds:
    row = "seed=" + str(s)
    for sl in seq_lens:
        row += "  "
        for p in poolings:
            val = next((v for v in results if v[0]==s and v[1]==sl and v[2]==p), None)
            row += "R2=" + (str(round(val[3], 3)) if val else "???)")
    print(row, flush=True)

print("\\n--- MSE ---")
for s in seeds:
    row = "seed=" + str(s)
    for sl in seq_lens:
        row += "  "
        for p in poolings:
            val = next((v for v in results if v[0]==s and v[1]==sl and v[2]==p), None)
            row += "MSE=" + (str(round(val[4], 4)) if val else "????")
    print(row, flush=True)
"""

procs = []
for gpu in range(n_gpus):
    gpu_tasks = [(s, sl, p, sigma, sp) for i, (s, sl, p) in enumerate(tasks) if i % n_gpus == gpu]
    code = base_code.replace("TASKS_PLACEHOLDER", str(gpu_tasks)).replace("SEEDS_PLACEHOLDER", str(seeds)).replace("SL_PLACEHOLDER", str(seq_lens)).replace("POOL_PLACEHOLDER", str(poolings))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    logf = open(f"/tmp/r4_gpu{gpu}.log", "w")
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=logf, stderr=logf, env=env, text=True)
    procs.append(proc)
    print(f"GPU {gpu}: {len(gpu_tasks)} tasks (PID {proc.pid})")

for i, proc in enumerate(procs):
    proc.wait()
    with open(f"/tmp/r4_gpu{i}.log") as f:
        print(f"\n=== GPU {i} ===")
        print(f.read())

print("\n=== R4 Complete ===")
