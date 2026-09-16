"""Runner for R1 (global regression), R3 (noise robustness sweep), and R4 (scaling behavior)."""
import sys
sys.path.insert(0, '.')

import torch
from torch.utils.data import DataLoader

from tasks.seq1d_regression import (
    Global1DCNN,
    GaussianProductDataset,
    run_experiment,
)


def _train_one(pooling: str, train_ds, val_ds, test_ds,
               epochs: int = 50, lr: float = 0.01, device=None):
    """Train a single pooling model and return test R^2 + MSE."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Global1DCNN(n_feat=1, pooling=pooling).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.functional.mse_loss
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=500)

    best_val_mse = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x).squeeze(), y.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

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
    ss_res = torch.sum((trues_t - preds_t) ** 2)
    ss_tot = torch.sum((trues_t - torch.mean(trues_t)) ** 2)
    r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0
    return {"test_mse": test_mse, "test_r2": r2}


def run_r4_scaling(
    seq_len_list: list = None,
    seed_list: list = None,
    noise_sigma: float = 1.0,
    noise_sparsity: float = 0.25,
):
    """R4: Scaling behavior — R^2 vs seq_len for global regression under fixed moderate noise.

    Dataset: lognormal features (all x_i > 0), fixed multiplicative noise.
    Target: exp(mean(log(x_true_i))) — unsigned geometric mean from clean features.
    Sweep: seq_len in [8, 16, 32, 64, 128, 256], 3 seeds.
    Shows how geo's advantage grows with sequence length under realistic noise.
    """
    seq_len_list = seq_len_list or [8, 16, 32, 64, 128, 256]
    seed_list = seed_list or [42, 123, 456]
    poolings = ["max", "avg", "geo"]

    print(f"\n=== R4: Scaling under fixed noise (sigma={noise_sigma}, sparsity={noise_sparsity}) ===")
    print(f"Dataset: unsigned lognormal, target = exp(mean(log(x_true)))")

    # Print R^2 table
    print(f"\n--- R^2 ---")
    for seed in seed_list:
        header = f"{'seq_len':>8}  " + "  ".join(f"{p:>10}" for p in poolings)
        print(f"\n--- seed={seed} ---")
        print(f"{header}\n{'-' * len(header)}")
        for seq_len in seq_len_list:
            train_ds = GaussianProductDataset(2000, seq_len, seed,
                                               noise_sigma=noise_sigma,
                                               noise_sparsity=noise_sparsity,
                                               unsigned=True)
            val_ds = GaussianProductDataset(500, seq_len, seed + 1,
                                            noise_sigma=noise_sigma,
                                            noise_sparsity=noise_sparsity,
                                            unsigned=True)
            test_ds = GaussianProductDataset(1000, seq_len, seed + 2,
                                             noise_sigma=noise_sigma,
                                             noise_sparsity=noise_sparsity,
                                             unsigned=True)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"{seq_len:>8}  ", end="")
            for p in poolings:
                results = _train_one(p, train_ds, val_ds, test_ds,
                                       epochs=200, lr=0.01, device=device)
                print(f"{results['test_r2']:>10.3f}  ", end="")
            print()

    # Print MSE table
    print(f"\n--- MSE ---")
    for seed in seed_list:
        header = f"{'seq_len':>8}  " + "  ".join(f"{p:>10}" for p in poolings)
        print(f"\n--- seed={seed} ---")
        print(f"{header}\n{'-' * len(header)}")
        for seq_len in seq_len_list:
            train_ds = GaussianProductDataset(2000, seq_len, seed,
                                               noise_sigma=noise_sigma,
                                               noise_sparsity=noise_sparsity,
                                               unsigned=True)
            val_ds = GaussianProductDataset(500, seq_len, seed + 1,
                                            noise_sigma=noise_sigma,
                                            noise_sparsity=noise_sparsity,
                                            unsigned=True)
            test_ds = GaussianProductDataset(1000, seq_len, seed + 2,
                                             noise_sigma=noise_sigma,
                                             noise_sparsity=noise_sparsity,
                                             unsigned=True)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"{seq_len:>8}  ", end="")
            for p in poolings:
                results = _train_one(p, train_ds, val_ds, test_ds,
                                       epochs=200, lr=0.01, device=device)
                print(f"{results['test_mse']:>10.4f}  ", end="")
            print()


def run_r3_unsigned_sweep(
    sigma_list: list = None,
    sparsity_list: list = None,
    seq_len: int = 32,
    seed: int = 42,
):
    """R3-unsigned: Noise robustness with lognormal features (all x_i > 0).

    Same noise sweep as R3 but with unsigned=True in the dataset.
    Tests whether geo's noise robustness holds for all-positive features.
    """
    sigma_list = sigma_list or [0.0, 0.5, 1.0, 2.0, 4.0]
    sparsity_list = sparsity_list or [0.0, 0.1, 0.25, 0.5]
    poolings = ["max", "avg", "geo"]

    # Print R^2 table
    print(f"\n=== R3-unsigned: Noise Robustness (lognormal) -- R^2 (seq_len={seq_len}, seed={seed}) ===")
    header = f"{'sigma':>6}  {'sparsity':>10}  " + "  ".join(f"{p:>10}" for p in poolings)
    print(f"{header}\n{'-' * len(header)}")
    for sigma in sigma_list:
        for sparsity in sparsity_list:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            train_ds = GaussianProductDataset(2000, seq_len, seed,
                                               noise_sigma=sigma,
                                               noise_sparsity=sparsity,
                                               unsigned=True)
            val_ds = GaussianProductDataset(500, seq_len, seed + 1,
                                            noise_sigma=sigma,
                                            noise_sparsity=sparsity,
                                            unsigned=True)
            test_ds = GaussianProductDataset(1000, seq_len, seed + 2,
                                             noise_sigma=sigma,
                                             noise_sparsity=sparsity,
                                             unsigned=True)
            print(f"{sigma:>6.1f}  {sparsity:>10.2f}  ", end="")
            for p in poolings:
                results = _train_one(p, train_ds, val_ds, test_ds,
                                       epochs=200, lr=0.01, device=device)
                print(f"{results['test_r2']:>10.3f}  ", end="")
            print()

    # Print MSE table
    print(f"\n=== R3-unsigned: Noise Robustness (lognormal) -- MSE (seq_len={seq_len}, seed={seed}) ===")
    header = f"{'sigma':>6}  {'sparsity':>10}  " + "  ".join(f"{p:>10}" for p in poolings)
    print(f"{header}\n{'-' * len(header)}")
    for sigma in sigma_list:
        for sparsity in sparsity_list:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            train_ds = GaussianProductDataset(2000, seq_len, seed,
                                               noise_sigma=sigma,
                                               noise_sparsity=sparsity,
                                               unsigned=True)
            val_ds = GaussianProductDataset(500, seq_len, seed + 1,
                                            noise_sigma=sigma,
                                            noise_sparsity=sparsity,
                                            unsigned=True)
            test_ds = GaussianProductDataset(1000, seq_len, seed + 2,
                                             noise_sigma=sigma,
                                             noise_sparsity=sparsity,
                                             unsigned=True)
            print(f"{sigma:>6.1f}  {sparsity:>10.2f}  ", end="")
            for p in poolings:
                results = _train_one(p, train_ds, val_ds, test_ds,
                                       epochs=200, lr=0.01, device=device)
                print(f"{results['test_mse']:>10.4f}  ", end="")
            print()

def run_r1(
    seq_len_list=None,
    seed=42,
    epochs=200,
    lr=0.01,
):
    """R1: global regression on clean positive lognormal inputs."""
    seq_len_list = seq_len_list or [16, 32, 64, 128]
    poolings = ["max", "avg", "geo"]

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    print("\n=== R1: Global geometric-mean regression ===")
    print("Dataset: x_i ~ LogNormal(0, 1)")
    print("Target: exp(mean(log(x_i)))")
    print(f"seed={seed}, epochs={epochs}, lr={lr}")

    header = (
        f"{'N':>8}  "
        + "  ".join(f"{pooling:>10}" for pooling in poolings)
    )

    print("\n--- R^2 ---")
    print(header)
    print("-" * len(header))

    for seq_len in seq_len_list:
        print(f"{seq_len:>8}  ", end="")

        for pooling in poolings:
            # Re-seed model initialization and DataLoader shuffling.
            torch.manual_seed(seed)

            # Recreate each dataset so all pooling methods receive
            # identical generated examples.
            train_ds = GaussianProductDataset(
                n_samples=2000,
                seq_len=seq_len,
                seed=seed,
                noise_sigma=0.0,
                noise_sparsity=0.0,
                unsigned=True,
            )
            val_ds = GaussianProductDataset(
                n_samples=500,
                seq_len=seq_len,
                seed=seed + 1,
                noise_sigma=0.0,
                noise_sparsity=0.0,
                unsigned=True,
            )
            test_ds = GaussianProductDataset(
                n_samples=1000,
                seq_len=seq_len,
                seed=seed + 2,
                noise_sigma=0.0,
                noise_sparsity=0.0,
                unsigned=True,
            )

            result = _train_one(
                pooling,
                train_ds,
                val_ds,
                test_ds,
                epochs=epochs,
                lr=lr,
                device=device,
            )
            print(f"{result['test_r2']:>10.3f}  ", end="")

        print()

    print("\n--- MSE (x 10^-3) ---")
    print(header)
    print("-" * len(header))

    for seq_len in seq_len_list:
        print(f"{seq_len:>8}  ", end="")

        for pooling in poolings:
            torch.manual_seed(seed)

            train_ds = GaussianProductDataset(
                n_samples=2000,
                seq_len=seq_len,
                seed=seed,
                noise_sigma=0.0,
                noise_sparsity=0.0,
                unsigned=True,
            )
            val_ds = GaussianProductDataset(
                n_samples=500,
                seq_len=seq_len,
                seed=seed + 1,
                noise_sigma=0.0,
                noise_sparsity=0.0,
                unsigned=True,
            )
            test_ds = GaussianProductDataset(
                n_samples=1000,
                seq_len=seq_len,
                seed=seed + 2,
                noise_sigma=0.0,
                noise_sparsity=0.0,
                unsigned=True,
            )

            result = _train_one(
                pooling,
                train_ds,
                val_ds,
                test_ds,
                epochs=epochs,
                lr=lr,
                device=device,
            )
            print(f"{result['test_mse'] * 1e3:>10.3f}  ", end="")

        print()

if __name__ == "__main__":
    #run_r4_scaling()
    # run_r3_unsigned_sweep()
    run_r1()