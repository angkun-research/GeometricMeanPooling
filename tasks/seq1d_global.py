"""Type 1 classification: degenerate marginals, class from product of ALL sites.

No unit cells. Class = sign of the product of all features.
Each feature x_i ~ N(0, 1) i.i.d. — marginals are identical under both classes.
The class signal is entirely in the joint multiplicative structure.

For degenerate marginals, max/avg see identical features under both classes.
Only the product structure carries the signal.

No conv layer, no activation — pooling operates on raw features so the product is preserved.

Architectural choices:
  Global pooling (kernel=1): pool entire sequence to 1 value
  No BatchNorm, no conv, no activation: preserves product structure
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

RHO = 0.0  # correlation strength. rho=0 = fully degenerate (independent)


class GaussianProductDataset(Dataset):
    """Type 1 data: degenerate marginals, class = sign of product over all sites.

    Each sample: x_i ~ N(0, 1) i.i.d.
    Class 0: prod(x_i) > 0 (even number of negative features)
    Class 1: prod(x_i) < 0 (odd number of negative features)
    Balanced classes by symmetry.
    """

    def __init__(self, n_samples: int = 2000, seq_len: int = 32,
                 rho: float = RHO, seed: int = 42):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.rho = rho
        self.rng = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        seq = torch.randn(self.seq_len, generator=self.rng)

        # Add pair-wise correlation (controlled by rho)
        if self.rho > 0:
            seq[1::2] = self.rho * seq[::2] + (1 - self.rho**2)**0.5 * seq[1::2]
            seq[::2] = self.rho * seq[1::2] + (1 - self.rho**2)**0.5 * seq[::2]

        # Class = sign of product over all sites
        label = 0 if torch.prod(seq) >= 0 else 1

        return seq.unsqueeze(-1), label  # [seq_len, 1]


class Global1DCNN(nn.Module):
    """Type 1 network: GlobalPool -> FC.

    No conv, no activation. Pooling operates on raw features so the product is preserved.
    GlobalPool: computes product/mean over all raw features
    FC: linear classifier on pooled output
    """

    def __init__(self, n_feat: int = 1, n_classes: int = 2, pooling: str = "geo"):
        super().__init__()

        if pooling == "max":
            pool = nn.AdaptiveMaxPool1d(1)
        elif pooling == "avg":
            pool = nn.AdaptiveAvgPool1d(1)
        elif pooling == "geo":
            from GeometricPool1d import GeometricPool1d
            pool = GeometricPool1d(1)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        self.pool = pool
        # Global pooling always outputs (B, 1, 1) -> FC sees 1 feature
        self.classifier = nn.Linear(1, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, L, C) -> (B, C, L)
        pooled = self.pool(x)  # (B, 1, 1)
        return self.classifier(pooled.flatten(1))


def make_dataloaders(
    train_n: int = 2000, val_n: int = 500, test_n: int = 1000,
    rho: float = RHO, seed: int = 42, seq_len: int = 32,
) -> tuple[Dataset, Dataset, Dataset]:
    return (GaussianProductDataset(train_n, seq_len, rho, seed),
            GaussianProductDataset(val_n, seq_len, rho, seed + 1),
            GaussianProductDataset(test_n, seq_len, rho, seed + 2))


def run_experiment(
    pooling: str, rho: float = RHO, epochs: int = 50, lr: float = 0.01,
    seed: int = 42, seq_len: int = 32,
) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds, test_ds = make_dataloaders(
        rho=rho, seed=seed, seq_len=seq_len)
    n_feat = train_ds[0][0].shape[-1]
    model = Global1DCNN(n_feat=n_feat, pooling=pooling).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=500)
    test_loader = DataLoader(test_ds, batch_size=1000)

    best_acc = 0.0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = F.cross_entropy(out, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)

        model.eval()
        vc, vt = 0, 0
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
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        tc += (model(x).argmax(1) == y).sum().item()
        tt += x.size(0)
    return {"pooling": pooling, "rho": rho, "seq_len": seq_len,
            "best_val_acc": best_acc, "test_acc": tc / tt}


if __name__ == "__main__":
    poolings = ["max", "avg", "geo"]
    for seed in [42, 123, 456]:
        print(f"\n--- seed={seed} ---")
        for p in poolings:
            r = run_experiment(p, seed=seed, epochs=50)
            print(f"{p:>6}: val={r['best_val_acc']:.4f} test={r['test_acc']:.4f}")
