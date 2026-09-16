import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tasks.seq1d_regression import Global1DCNN
from utils import set_seed


RESULTS_FILE = "./plots/fig3_r3_results.json"

SEED = 42
SEQ_LEN = 120
POOLINGS = ["max", "avg", "geo"]

MUL_SIGMA_LIST = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]
ADD_SIGMA_LIST = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
NOISE_SPARSITY = 1.0

EPOCHS = 200
LR = 0.01
D_EMBED = 32
NUM_WORKERS = 9

TRAIN_N = 2000
VAL_N = 500
TEST_N = 1000
EPS = 1e-12
UNSIGNED = True #False

class CombinedNoiseGaussianProductDataset(Dataset):
    """Global Gaussian-product regression with multiplicative noise followed by additive noise.

    Clean target is computed from x_true.
    Input to the model is x_noisy = (x_true * exp(mult_noise)) + add_noise.
    Multiplicative noise is applied to all features because NOISE_SPARSITY = 1.0.
    """

    def __init__(
        self,
        n_samples: int,
        seq_len: int,
        seed: int,
        multiplicative_sigma: float,
        additive_sigma: float,
        noise_sparsity: float = 1.0,
        unsigned: bool = False,
    ):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.multiplicative_sigma = multiplicative_sigma
        self.additive_sigma = additive_sigma
        self.noise_sparsity = noise_sparsity
        self.unsigned = unsigned
        self.rng = torch.Generator().manual_seed(seed)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        if self.unsigned:
            # Lognormal features: log(x_i) ~ N(0, 1) -> all x_i > 0
            log_feat = torch.randn(self.seq_len, generator=self.rng)
            x_true = torch.exp(log_feat)
        else:
            # Standard Gaussian features: x_i ~ N(0, 1)
            x_true = torch.randn(self.seq_len, generator=self.rng)

        sign = torch.prod(torch.sign(x_true + EPS))
        log_mag = torch.mean(torch.log(torch.abs(x_true) + EPS))
        target = sign * torch.exp(log_mag)

        x_noisy = x_true.clone()

        if self.multiplicative_sigma > 0:
            if self.noise_sparsity >= 1.0:
                noise_mask = torch.ones(self.seq_len, dtype=torch.bool)
            else:
                noise_mask = torch.rand(self.seq_len, generator=self.rng) < self.noise_sparsity

            mult_noise = torch.randn(self.seq_len, generator=self.rng) * self.multiplicative_sigma
            x_noisy[noise_mask] *= torch.exp(mult_noise[noise_mask])

        if self.additive_sigma > 0:
            add_noise = torch.randn(self.seq_len, generator=self.rng) * self.additive_sigma
            x_noisy = x_noisy + add_noise

        return x_noisy.unsqueeze(-1), target.unsqueeze(-1)


def make_dataloaders(
    seq_len: int,
    seed: int,
    multiplicative_sigma: float,
    additive_sigma: float,
):
    #print(f"Has sign: {not UNSIGNED}")
    train_ds = CombinedNoiseGaussianProductDataset(
        n_samples=TRAIN_N,
        seq_len=seq_len,
        seed=seed,
        multiplicative_sigma=multiplicative_sigma,
        additive_sigma=additive_sigma,
        noise_sparsity=NOISE_SPARSITY,
        unsigned=UNSIGNED,
    )
    val_ds = CombinedNoiseGaussianProductDataset(
        n_samples=VAL_N,
        seq_len=seq_len,
        seed=seed + 1,
        multiplicative_sigma=multiplicative_sigma,
        additive_sigma=additive_sigma,
        noise_sparsity=NOISE_SPARSITY,
        unsigned=UNSIGNED,
    )
    test_ds = CombinedNoiseGaussianProductDataset(
        n_samples=TEST_N,
        seq_len=seq_len,
        seed=seed + 2,
        multiplicative_sigma=multiplicative_sigma,
        additive_sigma=additive_sigma,
        noise_sparsity=NOISE_SPARSITY,
        unsigned=UNSIGNED,
    )

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=500)
    test_loader = DataLoader(test_ds, batch_size=1000)
    return train_ds, train_loader, val_loader, test_loader


def run_single_experiment(pooling: str, multiplicative_sigma: float, additive_sigma: float) -> dict:
    set_seed(SEED)
    device = torch.device("cpu")

    train_ds, train_loader, val_loader, test_loader = make_dataloaders(
        seq_len=SEQ_LEN,
        seed=SEED,
        multiplicative_sigma=multiplicative_sigma,
        additive_sigma=additive_sigma,
    )
    print("d_embed:", D_EMBED)
    model = Global1DCNN(n_feat=1, pooling=pooling, d_embed=D_EMBED).to(device)
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
        "seed": SEED,
        "multiplicative_sigma": multiplicative_sigma,
        "additive_sigma": additive_sigma,
        "noise_sparsity": NOISE_SPARSITY,
        "best_val_mse": float(f"{best_val_mse:.6g}"),
        "test_mse": float(f"{test_mse:.6g}"),
        "test_r2": float(f"{test_r2:.6g}"),
    }


def task_key(pooling: str, multiplicative_sigma: float, additive_sigma: float) -> str:
    return f"r3_global_L{SEQ_LEN}_mul{multiplicative_sigma:.3f}_add{additive_sigma:.3f}_{pooling}_seed{SEED}"


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
    for multiplicative_sigma in MUL_SIGMA_LIST:
        for additive_sigma in ADD_SIGMA_LIST:
            for pooling in POOLINGS:
                key = task_key(pooling, multiplicative_sigma, additive_sigma)
                if key not in results:
                    tasks.append((key, pooling, multiplicative_sigma, additive_sigma))
        #         if len(tasks) >= NUM_WORKERS:
        #             break
        #     if len(tasks) >= NUM_WORKERS:
        #         break
        # if len(tasks) >= NUM_WORKERS:
        #     break
    print(f"Pending tasks: {len(tasks)}")
    print(f"Using {NUM_WORKERS} CPU workers")
    #exit(0)

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_map = {
            executor.submit(run_single_experiment, pooling, multiplicative_sigma, additive_sigma): (
                key,
                pooling,
                multiplicative_sigma,
                additive_sigma,
            )
            for key, pooling, multiplicative_sigma, additive_sigma in tasks
        }

        for future in as_completed(future_map):
            key, pooling, multiplicative_sigma, additive_sigma = future_map[future]
            try:
                result = future.result()
                results[key] = result

                with open(RESULTS_FILE, "w") as f:
                    json.dump(results, f, indent=2)

                print(
                    f"done  mul={multiplicative_sigma:<3.3f} add={additive_sigma:<3.3f} "
                    f"pool={pooling:<3} seed={SEED:<3} "
                    f"r2={result['test_r2']:.4f} mse={result['test_mse']:.4f}"
                )
            except Exception as exc:
                print(
                    f"failed mul={multiplicative_sigma} add={additive_sigma} "
                    f"pool={pooling}: {exc}"
                )

    print(f"Saved results to {RESULTS_FILE}")

    print("\n=== R^2 TABLE ===")
    print(f"{'mul':>6}  {'add':>6}  " + "  ".join(f"{p:>10}" for p in POOLINGS))
    print("-" * 52)
    for multiplicative_sigma in MUL_SIGMA_LIST:
        for additive_sigma in ADD_SIGMA_LIST:
            print(f"{multiplicative_sigma:>6.3f}  {additive_sigma:>6.3f}  ", end="")
            for pooling in POOLINGS:
                r = results[task_key(pooling, multiplicative_sigma, additive_sigma)]
                print(f"{r['test_r2']:>10.3f}  ", end="")
            print()

    print("\n=== MSE TABLE ===")
    print(f"{'mul':>6}  {'add':>6}  " + "  ".join(f"{p:>10}" for p in POOLINGS))
    print("-" * 52)
    for multiplicative_sigma in MUL_SIGMA_LIST:
        for additive_sigma in ADD_SIGMA_LIST:
            print(f"{multiplicative_sigma:>6.3f}  {additive_sigma:>6.3f}  ", end="")
            for pooling in POOLINGS:
                r = results[task_key(pooling, multiplicative_sigma, additive_sigma)]
                print(f"{r['test_mse']:>10.4f}  ", end="")
            print()


if __name__ == "__main__":
    main()
    # nohup python collect_fig3_r3.py > plots/fig3_r3.log 2>&1 &