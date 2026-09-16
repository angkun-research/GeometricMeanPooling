import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tasks.seq1d_regression import LocalCellNetwork
from utils import set_seed


RESULTS_FILE = "./plots/fig2_r2_uc_results.json"
SEEDS = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
SEQ_LEN = 120
UNIT_CELLS = [2, 3, 4, 5, 6, 8, 10]
POOLINGS = ["max", "avg", "geo"]

SIGNAL_SIGMA = 1.0
NOISE_SIGMA = 0.05
EPOCHS = 100
LR = 0.01
NUM_WORKERS = 10

TRAIN_N = 2000
VAL_N = 500
TEST_N = 1000
EPS = 1e-12


class NoisyLocalCellRegressionDataset(Dataset):
    """Local regression dataset with additive Gaussian noise on X only.

    Target is computed from the clean sequence.
    Model receives noisy sequence.
    """

    def __init__(
        self,
        n_samples: int,
        seq_len: int,
        sigma: float,
        noise_sigma: float,
        unit_cell: int,
        seed: int,
    ):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.sigma = sigma
        self.noise_sigma = noise_sigma
        self.unit_cell = unit_cell
        self.num_cells = seq_len // unit_cell
        assert seq_len % unit_cell == 0, "seq_len must be divisible by unit_cell"
        self.rng = torch.Generator().manual_seed(seed)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        log_feat = torch.randn(self.seq_len, generator=self.rng) * self.sigma
        x_clean = torch.exp(log_feat)

        x_cells = x_clean.view(self.num_cells, self.unit_cell)
        log_abs = torch.log(torch.clamp(torch.abs(x_cells), min=EPS))
        cell_signal = torch.exp(log_abs.mean(dim=1))
        target = cell_signal.mean().unsqueeze(-1)

        noise = torch.randn(self.seq_len, generator=self.rng) * self.noise_sigma
        x_noisy = x_clean + noise

        return x_noisy.unsqueeze(-1), target


def make_noisy_local_reg_dataloaders(seq_len: int, unit_cell: int, seed: int):
    train_ds = NoisyLocalCellRegressionDataset(
        n_samples=TRAIN_N,
        seq_len=seq_len,
        sigma=SIGNAL_SIGMA,
        noise_sigma=NOISE_SIGMA,
        unit_cell=unit_cell,
        seed=seed,
    )
    val_ds = NoisyLocalCellRegressionDataset(
        n_samples=VAL_N,
        seq_len=seq_len,
        sigma=SIGNAL_SIGMA,
        noise_sigma=NOISE_SIGMA,
        unit_cell=unit_cell,
        seed=seed + 1,
    )
    test_ds = NoisyLocalCellRegressionDataset(
        n_samples=TEST_N,
        seq_len=seq_len,
        sigma=SIGNAL_SIGMA,
        noise_sigma=NOISE_SIGMA,
        unit_cell=unit_cell,
        seed=seed + 2,
    )

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=500)
    test_loader = DataLoader(test_ds, batch_size=1000)
    return train_ds, train_loader, val_loader, test_loader


def run_single_experiment(pooling: str, unit_cell: int, seed: int) -> dict:
    set_seed(seed)
    device = torch.device("cpu")

    train_ds, train_loader, val_loader, test_loader = make_noisy_local_reg_dataloaders(
        seq_len=SEQ_LEN,
        unit_cell=unit_cell,
        seed=seed,
    )

    model = LocalCellNetwork(
        cell_size=unit_cell,
        pooling=pooling,
        num_cells=SEQ_LEN // unit_cell,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_mse = float("inf")
    best_state = None

    for _ in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            loss = F.mse_loss(out.squeeze(), y.squeeze())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        vloss = 0.0
        vt = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                out = model(x)
                vloss += F.mse_loss(out.squeeze(), y.squeeze()).item() * x.size(0)
                vt += x.size(0)

        val_mse = vloss / vt
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    total_se = 0.0
    total_n = 0
    preds, trues = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)

            total_se += F.mse_loss(out.squeeze(), y.squeeze()).item() * x.size(0)
            total_n += x.size(0)

            preds.append(out.detach().cpu().squeeze())
            trues.append(y.detach().cpu().squeeze())

    test_mse = total_se / total_n

    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)
    residual_var = torch.var(preds - trues)
    target_var = torch.var(trues)
    test_r2 = (1 - residual_var / target_var).item() if target_var > 0 else 0.0

    return {
        "pooling": pooling,
        "seq_len": SEQ_LEN,
        "seed": seed,
        "unit_cell": unit_cell,
        "signal_sigma": SIGNAL_SIGMA,
        "noise_sigma": NOISE_SIGMA,
        "best_val_mse": float(f"{best_val_mse:.6g}"),
        "test_mse": float(f"{test_mse:.6g}"),
        "test_r2": float(f"{test_r2:.6g}"),
    }


def task_key(pooling: str, unit_cell: int, seed: int) -> str:
    return f"r2_local_uc{unit_cell}_L{SEQ_LEN}_{pooling}_seed{seed}"


def main():
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = {}

    tasks = []
    for unit_cell in UNIT_CELLS:
        for pooling in POOLINGS:
            for seed in SEEDS:
                key = task_key(pooling, unit_cell, seed)
                if key not in results:
                    tasks.append((key, pooling, unit_cell, seed))

    print(f"Pending tasks: {len(tasks)}")
    print(f"Using {NUM_WORKERS} CPU workers")

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_map = {
            executor.submit(run_single_experiment, pooling, unit_cell, seed): (key, pooling, unit_cell, seed)
            for key, pooling, unit_cell, seed in tasks
        }

        for future in as_completed(future_map):
            key, pooling, unit_cell, seed = future_map[future]
            try:
                result = future.result()
                results[key] = result

                with open(RESULTS_FILE, "w") as f:
                    json.dump(results, f, indent=2)

                print(
                    f"done  uc={unit_cell:<2} pool={pooling:<3} seed={seed:<3} "
                    f"r2={result['test_r2']:.4f} mse={result['test_mse']:.4f}"
                )
            except Exception as exc:
                print(f"failed uc={unit_cell} pool={pooling} seed={seed}: {exc}")

    print(f"Saved results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
    # nohup python collect_fig2_r2_uc.py > plots/fig2_r2_uc.log 2>&1 &

