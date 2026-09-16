import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tasks.seq1d_classif import Small1DCNN
from utils import set_seed


RESULTS_FILE = "./plots/fig2_b3_results.json"
SEEDS = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
SEQ_LENS = [18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
POOLINGS = ["max", "avg", "geo"]

UNIT_CELL = 3
SIGNAL_SIGMA = 1.0
NOISE_SIGMA = 0.05
EPOCHS = 100
LR = 0.01
NUM_WORKERS = 10

TRAIN_N = 2000
VAL_N = 500
TEST_N = 1000


class NoisySeqDatasetB3(Dataset):
    """B3 dataset with additive Gaussian noise on X only.

    Label is computed from the clean sequence.
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
        self.rng = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        seq_clean = torch.randn(self.seq_len, generator=self.rng) * self.sigma

        n_cells = self.seq_len // self.unit_cell
        S = 0.0
        for j in range(n_cells):
            cell = seq_clean[j * self.unit_cell : (j + 1) * self.unit_cell]
            sign = torch.prod(torch.sign(cell))
            log_abs = torch.log(torch.clamp(torch.abs(cell), min=1e-12))
            val = torch.exp(torch.mean(log_abs))
            S += sign * val

        label = 0 if S >= 0 else 1

        noise = torch.randn(self.seq_len, generator=self.rng) * self.noise_sigma
        seq_noisy = seq_clean + noise

        return seq_noisy.unsqueeze(-1), label


def make_b3_noisy_dataloaders(seq_len: int, seed: int):
    train_ds = NoisySeqDatasetB3(
        n_samples=TRAIN_N,
        seq_len=seq_len,
        sigma=SIGNAL_SIGMA,
        noise_sigma=NOISE_SIGMA,
        unit_cell=UNIT_CELL,
        seed=seed,
    )
    val_ds = NoisySeqDatasetB3(
        n_samples=VAL_N,
        seq_len=seq_len,
        sigma=SIGNAL_SIGMA,
        noise_sigma=NOISE_SIGMA,
        unit_cell=UNIT_CELL,
        seed=seed + 1,
    )
    test_ds = NoisySeqDatasetB3(
        n_samples=TEST_N,
        seq_len=seq_len,
        sigma=SIGNAL_SIGMA,
        noise_sigma=NOISE_SIGMA,
        unit_cell=UNIT_CELL,
        seed=seed + 2,
    )

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=500)
    test_loader = DataLoader(test_ds, batch_size=1000)
    return train_ds, train_loader, val_loader, test_loader


def run_single_experiment(pooling: str, seq_len: int, seed: int) -> dict:
    set_seed(seed)
    device = torch.device("cpu")

    train_ds, train_loader, val_loader, test_loader = make_b3_noisy_dataloaders(
        seq_len=seq_len,
        seed=seed,
    )

    n_feat = train_ds[0][0].shape[-1]
    model = Small1DCNN(
        n_feat=n_feat,
        pooling=pooling,
        kernel_size=UNIT_CELL,
        seq_len=seq_len,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_acc = 0.0
    best_state = None

    for _ in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            loss = F.cross_entropy(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        vc = 0
        vt = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x).argmax(1)
                vc += (pred == y).sum().item()
                vt += x.size(0)

        val_acc = vc / vt
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    tc = 0
    tt = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x).argmax(1)
            tc += (pred == y).sum().item()
            tt += x.size(0)

    return {
        "pooling": pooling,
        "seq_len": seq_len,
        "seed": seed,
        "unit_cell": UNIT_CELL,
        "signal_sigma": SIGNAL_SIGMA,
        "noise_sigma": NOISE_SIGMA,
        "best_val_acc": float(f"{best_acc:.8g}"),
        "test_acc": float(f"{(tc / tt):.8g}"),
    }


def task_key(pooling: str, seq_len: int, seed: int) -> str:
    return f"b3_uc{UNIT_CELL}_L{seq_len}_{pooling}_seed{seed}"


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
    for seq_len in SEQ_LENS:
        for pooling in POOLINGS:
            for seed in SEEDS:
                key = task_key(pooling, seq_len, seed)
                if key not in results:
                    tasks.append((key, pooling, seq_len, seed))
        #         if len(tasks) >= NUM_WORKERS:
        #             break
        #     if len(tasks) >= NUM_WORKERS:
        #         break
        # if len(tasks) >= NUM_WORKERS:
        #     break

    print(f"Pending tasks: {len(tasks)}")
    print(f"Using {NUM_WORKERS} CPU workers")

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_map = {
            executor.submit(run_single_experiment, pooling, seq_len, seed): (key, pooling, seq_len, seed)
            for key, pooling, seq_len, seed in tasks
        }

        for future in as_completed(future_map):
            key, pooling, seq_len, seed = future_map[future]
            try:
                result = future.result()
                results[key] = result

                with open(RESULTS_FILE, "w") as f:
                    json.dump(results, f, indent=2)

                print(
                    f"done  L={seq_len:<3} pool={pooling:<3} seed={seed:<3} "
                    f"test={result['test_acc']:.6f}"
                )
            except Exception as exc:
                print(f"failed L={seq_len} pool={pooling} seed={seed}: {exc}")

    print(f"Saved results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
    # to run without interruption: nohup python collect_fig2_b3_data.py > plots/fig2_b3.log 2>&1 &