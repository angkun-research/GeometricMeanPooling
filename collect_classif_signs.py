"""Ablation: embedding-dim-32 pointwise Conv1d + (signed vs unsigned) geometric pooling.

Case A: global signed-product classification (GaussianProductDataset).
Case B: local signed-product-sum classification (SeqDatasetB3, matches paper Case B).

Architecture: Conv1d(1, 32, kernel_size=1) -> Pool(global for A / local for B) -> FC(., 2)
Pooling: geometric mean, with or without the sign(prod(x)) factor.

Results (10 seeds x 2 cases x 2 pooling variants = 40 runs) are saved to
plots/embed_sign_ablation_results.json
"""
import json
import os
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tasks.seq1d_global import GaussianProductDataset
from tasks.seq1d_classif import SeqDatasetB3
from utils import set_seed, pick_best_device

SEEDS = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
SEQ_LEN = 120
UNIT_CELL = 3  # Case B cell size
EMBED_DIM = 32
EPOCHS = 100
LR = 0.01
NUM_WORKERS = 4
RESULTS_FILE = "plots/embed_sign_ablation_results.json"


class GeometricPool1dOptSign(nn.Module):
    """Signed or unsigned geometric mean pooling (global if kernel_size in {None, 1})."""

    def __init__(self, kernel_size: int | None = None, stride: int | None = None,
                 eps: float = 1e-6, signed: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.eps = float(eps)
        self.signed = signed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size is None or self.kernel_size == 1:
            w = x.view(x.size(0), x.size(1), -1)
            log_abs = torch.log(torch.clamp(torch.abs(w), min=self.eps))
            geom = torch.exp(torch.mean(log_abs, dim=-1))
            if self.signed:
                sign = torch.prod(torch.sign(w), dim=-1)
                geom = sign * geom
            return geom.unsqueeze(-1)

        k = self.kernel_size
        s = self.stride if self.stride is not None else k
        w = x.unfold(2, k, s)  # (B, C, L_out, k)
        log_abs = torch.log(torch.clamp(torch.abs(w), min=self.eps))
        geom = torch.exp(torch.mean(log_abs, dim=-1))
        if self.signed:
            sign = torch.prod(torch.sign(w), dim=-1)
            geom = sign * geom
        return geom


class EmbedGlobalNet(nn.Module):
    """Case A: Conv1d(1, embed_dim, k=1) -> global GMP -> FC."""

    def __init__(self, embed_dim: int = EMBED_DIM, n_classes: int = 2, signed: bool = True):
        super().__init__()
        self.embed = nn.Conv1d(1, embed_dim, kernel_size=1)
        self.pool = GeometricPool1dOptSign(kernel_size=1, signed=signed)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)          # (B, 1, L)
        x = self.embed(x)              # (B, embed_dim, L)
        pooled = self.pool(x)          # (B, embed_dim, 1)
        return self.classifier(pooled.flatten(1))


class EmbedLocalNet(nn.Module):
    """Case B: Conv1d(1, embed_dim, k=1) -> local GMP(k=unit_cell) -> Flatten -> FC."""

    def __init__(self, embed_dim: int = EMBED_DIM, n_classes: int = 2,
                 kernel_size: int = UNIT_CELL, seq_len: int = SEQ_LEN, signed: bool = True):
        super().__init__()
        self.embed = nn.Conv1d(1, embed_dim, kernel_size=1)
        self.pool = GeometricPool1dOptSign(kernel_size, kernel_size, signed=signed)
        l_after = seq_len // kernel_size
        self.classifier = nn.Linear(embed_dim * l_after, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)          # (B, 1, L)
        x = self.embed(x)              # (B, embed_dim, L)
        pooled = self.pool(x)          # (B, embed_dim, L_out)
        return self.classifier(pooled.flatten(1))


def _train_eval(model: nn.Module, train_ds, val_ds, test_ds, device, epochs=EPOCHS, lr=LR) -> float:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=500)
    test_loader = DataLoader(test_ds, batch_size=1000)

    best_acc, best_state = 0.0, None
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                vc += (model(x).argmax(1) == y).sum().item()
                vt += x.size(0)
        va = vc / vt
        if va > best_acc:
            best_acc = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    tc, tt = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            tc += (model(x).argmax(1) == y).sum().item()
            tt += x.size(0)
    return tc / tt


def run_case_a(seed: int, signed: bool, device) -> float:
    set_seed(seed)
    train_ds = GaussianProductDataset(2000, SEQ_LEN, rho=0.0, seed=seed)
    val_ds = GaussianProductDataset(500, SEQ_LEN, rho=0.0, seed=seed + 1)
    test_ds = GaussianProductDataset(1000, SEQ_LEN, rho=0.0, seed=seed + 2)
    model = EmbedGlobalNet(signed=signed)
    return _train_eval(model, train_ds, val_ds, test_ds, device)


def run_case_b(seed: int, signed: bool, device) -> float:
    set_seed(seed)
    train_ds = SeqDatasetB3(2000, SEQ_LEN, sigma=1.0, unit_cell=UNIT_CELL, seed=seed)
    val_ds = SeqDatasetB3(500, SEQ_LEN, sigma=1.0, unit_cell=UNIT_CELL, seed=seed + 1)
    test_ds = SeqDatasetB3(1000, SEQ_LEN, sigma=1.0, unit_cell=UNIT_CELL, seed=seed + 2)
    model = EmbedLocalNet(kernel_size=UNIT_CELL, seq_len=SEQ_LEN, signed=signed)
    return _train_eval(model, train_ds, val_ds, test_ds, device)


RUN_FNS = {"A": run_case_a, "B": run_case_b}


def task_key(case: str, signed: bool, seed: int) -> str:
    return f"case{case}_signed{signed}_seed{seed}"

def run_single_experiment(case: str, signed: bool, seed: int) -> dict:
    device = pick_best_device()
    acc = RUN_FNS[case](seed, signed, device)
    return {"case": case, "signed": signed, "seed": seed, "test_acc": acc}

def main():
    os.makedirs("plots", exist_ok=True)

    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = {}

    tasks = []
    for case in ["A", "B"]:
        for signed in [True, False]:
            for seed in SEEDS:
                key = task_key(case, signed, seed)
                if key not in results:
                    tasks.append((key, case, signed, seed))

    print(f"Pending tasks: {len(tasks)}")
    print(f"Using {NUM_WORKERS} workers")

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_map = {
            executor.submit(run_single_experiment, case, signed, seed): (
                key, case, signed, seed
            )
            for key, case, signed, seed in tasks
        }

        for future in as_completed(future_map):
            key, case, signed, seed = future_map[future]
            try:
                result = future.result()
                results[key] = result

                with open(RESULTS_FILE, "w") as f:
                    json.dump(results, f, indent=2)

                print(f"done  case={case} signed={signed} seed={seed:>6} test_acc={result['test_acc']:.4f}")
            except Exception as exc:
                print(f"failed case={case} signed={signed} seed={seed}: {exc}")

    print(f"Saved results to {RESULTS_FILE}")

    print("\n=== Summary (mean +/- std over 10 seeds) ===")
    print(f"{'Case':<6}{'Pooling':<10}{'Mean Acc':<12}{'Std':<10}")
    for case in ["A", "B"]:
        for signed in [True, False]:
            accs = [r["test_acc"] for r in results.values() if r["case"] == case and r["signed"] == signed]
            if not accs:
                continue
            label = "Signed" if signed else "Unsigned"
            print(f"{case:<6}{label:<10}{statistics.mean(accs):<12.4f}{statistics.stdev(accs):<10.4f}")


if __name__ == "__main__":
    main()
    # to run without interruption:
    # nohup python collect_classif_signs.py > plots/embed_sign_ablation.log 2>&1 &