"""R3: Noise robustness sweep. Runs all 60 (sigma, sparsity, pooling) combos."""
import sys
sys.path.insert(0, '.')

import json
import torch
from torch.utils.data import DataLoader
from tasks.seq1d_regression import Global1DCNN, GaussianProductDataset


def _train_one(pooling, seq_len, seed, noise_sigma, noise_sparsity, epochs=50, lr=0.01):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_ds = GaussianProductDataset(2000, seq_len, seed,
                                      noise_sigma=noise_sigma,
                                      noise_sparsity=noise_sparsity)
    val_ds = GaussianProductDataset(500, seq_len, seed + 1,
                                    noise_sigma=noise_sigma,
                                    noise_sparsity=noise_sparsity)
    test_ds = GaussianProductDataset(1000, seq_len, seed + 2,
                                     noise_sigma=noise_sigma,
                                     noise_sparsity=noise_sparsity)
    model = Global1DCNN(n_feat=1, pooling=pooling).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.functional.mse_loss
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=500)

    best_val_mse = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x).squeeze(), y.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        vloss, vt = 0.0, 0
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            vloss += criterion(model(x).squeeze(), y.squeeze()).item() * x.size(0)
            vt += x.size(0)
        if vloss / vt < best_val_mse:
            best_val_mse = vloss / vt
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    tc, vt = 0.0, 0
    preds, trues = [], []
    for x, y in DataLoader(test_ds, batch_size=500):
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
    return {"test_mse": test_mse, "test_r2": r2}


if __name__ == "__main__":
    sigma_list = [0.0, 0.5, 1.0, 2.0, 4.0]
    sparsity_list = [0.0, 0.1, 0.25, 0.5]
    poolings = ["max", "avg", "geo"]
    seq_len = 32
    seed = 42

    results = {}
    total = len(sigma_list) * len(sparsity_list) * len(poolings)
    count = 0

    for sigma in sigma_list:
        for sparsity in sparsity_list:
            for p in poolings:
                count += 1
                r = _train_one(p, seq_len, seed, sigma, sparsity, epochs=50, lr=0.01)
                results[(sigma, sparsity, p)] = r
                print(f"[{count:>3}/{total}] [{sigma:.1f}] [{sparsity:.2f}] [{p:>3}] "
                      f"r2={r['test_r2']:>12.4f}  mse={r['test_mse']:>12.4f}", flush=True)

    # Save to JSON
    serializable = {}
    for (s, sp, p), r in results.items():
        k = f"{s:.1f}_{sp:.2f}_{p}"
        serializable[k] = r
    with open("r3_results.json", "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved {len(results)} results to r3_results.json")

    # Print summary tables
    print("\n=== R^2 TABLE ===")
    print(f"{'sigma':>6}  {'sparsity':>10}  " + "  ".join(f"{p:>10}" for p in poolings))
    print("-" * 50)
    for sigma in sigma_list:
        for sparsity in sparsity_list:
            print(f"{sigma:>6.1f}  {sparsity:>10.2f}  ", end="")
            for p in poolings:
                r = results[(sigma, sparsity, p)]
                print(f"{r['test_r2']:>10.3f}  ", end="")
            print()

    print("\n=== MSE TABLE ===")
    print(f"{'sigma':>6}  {'sparsity':>10}  " + "  ".join(f"{p:>10}" for p in poolings))
    print("-" * 50)
    for sigma in sigma_list:
        for sparsity in sparsity_list:
            print(f"{sigma:>6.1f}  {sparsity:>10.2f}  ", end="")
            for p in poolings:
                r = results[(sigma, sparsity, p)]
                print(f"{r['test_mse']:>10.4f}  ", end="")
            print()
