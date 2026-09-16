"""Type 1 regression: geometric mean (signed) of product on zero-mean Gaussian.

No unit cells. Target = sign(prod) * exp(mean(log(|x|))) — the true geometric mean.
Each feature x_i ~ N(0, 1) i.i.d.

Target = prod(sign(x_i)) * exp(mean(log(|x_i|)))
This is exactly what GeometricPool1d(1) computes internally.
The network's job: use its own pooling to regress this value.

For geo: the pooling directly outputs the target → FC learns identity → R² ≈ 1
For max/avg: they produce different scalars → can't recover geo mean → lower R²

No conv layer, no activation — pooling operates on raw features.

Architectural choices:
  Global pooling (kernel=1): pool entire sequence to 1 value
  No BatchNorm, no conv, no activation: preserves product structure
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

EPS = 1e-8


class GaussianProductDataset(Dataset):
    """Type 1 data: features x_i ~ N(0, 1) i.i.d., target = geo_mean = sign * exp(mean(log(|x|))).

    Each sample: x_i ~ N(0, 1) i.i.d.
    Target: signed geometric mean = prod(sign(x_i)) * exp(mean(log(|x_i|)))
    The sign = prod(sign(x_i)) is the binary signal from Case A.

    For R3 noise robustness: some features can be corrupted by multiplicative noise.
    x_noisy_i = x_true_i * exp(N(0, sigma²)) with probability p (sparsity)
    Network receives x_noisy, regresses target from x_true.
    """

    def __init__(self, n_samples: int = 2000, seq_len: int = 32,
                 seed: int = 42, noise_sigma: float = 0.0, noise_sparsity: float = 0.0,
                 unsigned: bool = False):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.noise_sigma = noise_sigma
        self.noise_sparsity = noise_sparsity  # fraction of features corrupted
        self.unsigned = unsigned  # if True, use lognormal (all positive) features
        self.rng = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.unsigned:
            # Lognormal features: log(x_i) ~ N(0, 1) -> all x_i > 0
            log_feat = torch.randn(self.seq_len, generator=self.rng)
            x_true = torch.exp(log_feat)
        else:
            # Zero-mean Gaussian: features can be positive or negative
            seq = torch.randn(self.seq_len, generator=self.rng)
            x_true = seq.clone()

        # Apply sparse multiplicative noise (multiplicative, works for both)
        if self.noise_sigma > 0 and self.noise_sparsity > 0:
            noise_mask = torch.rand(self.seq_len, generator=self.rng) < self.noise_sparsity
            noise = torch.randn(self.seq_len, generator=self.rng) * self.noise_sigma
            x_clean = x_true.clone()
            x_true[noise_mask] *= torch.exp(noise[noise_mask])
        else:
            x_clean = x_true

        sign = torch.prod(torch.sign(x_clean + EPS))
        log_mag = torch.mean(torch.log(torch.abs(x_clean) + EPS))
        target = sign * torch.exp(log_mag)
        return x_true.unsqueeze(-1), target.unsqueeze(-1)


class Global1DCNN(nn.Module):
    """Type 1 network: GlobalPool -> FC.

    No conv, no activation. Pooling operates on raw features so the product is preserved.
    GlobalPool: computes product/mean over all raw features.
    FC: linear regressor on pooled output.
    """

    def __init__(self, n_feat: int = 1, pooling: str = "geo", d_embed: int = 1):
        super().__init__()
        self.d_embed = d_embed

        if pooling == "max":
            pool = nn.AdaptiveMaxPool1d(1)
        elif pooling == "avg":
            pool = nn.AdaptiveAvgPool1d(1)
        elif pooling == "geo":
            from GeometricPool1d import GeometricPool1d
            pool = GeometricPool1d(1)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        self.embedding = nn.Conv1d(1,self.d_embed, kernel_size=1)
        self.pool = pool
        self.regressor = nn.Linear(self.d_embed, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — pool raw x, FC maps scalar to log_mag.

        Input x: (B, L, 1)
        Output: (B, 1)
        """
        x = x.transpose(1, 2)  # (B, L, C) -> (B, C, L)
        x = self.embedding(x)  # (B, d_embed, L)
        pooled = self.pool(x)  # (B, d_embed, 1)
        return self.regressor(pooled.flatten(1))  # (B, 1)


def make_dataloaders(
    train_n: int = 2000, val_n: int = 500, test_n: int = 1000,
    seed: int = 42, seq_len: int = 32,
) -> tuple[Dataset, Dataset, Dataset]:
    return (GaussianProductDataset(train_n, seq_len, seed),
            GaussianProductDataset(val_n, seq_len, seed + 1),
            GaussianProductDataset(test_n, seq_len, seed + 2))


def run_experiment(
    pooling: str, epochs: int = 50, lr: float = 0.01,
    seed: int = 42, seq_len: int = 32,
    d_embed: int = 1
) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds, test_ds = make_dataloaders(
        seed=seed, seq_len=seq_len)
    n_feat = train_ds[0][0].shape[-1]
    model = Global1DCNN(n_feat=n_feat, pooling=pooling, d_embed=d_embed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=500)
    test_loader = DataLoader(test_ds, batch_size=1000)

    best_val_mse = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = F.mse_loss(out.squeeze(), y.squeeze())
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * x.size(0)

        model.eval()
        vloss, vt = 0.0, 0
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            vloss += F.mse_loss(out.squeeze(), y.squeeze()).item() * x.size(0)
            vt += x.size(0)
        va = vloss / vt
        if va < best_val_mse:
            best_val_mse = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    tc, vt = 0.0, 0
    preds, trues = [], []
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        tc += F.mse_loss(out.squeeze(), y.squeeze()).item() * x.size(0)
        vt += x.size(0)
        preds.append(out.detach().cpu().squeeze())
        trues.append(y.detach().cpu().squeeze())
    test_mse = tc / vt

    # Compute R²
    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)
    residual_var = torch.var(preds - trues)
    target_var = torch.var(trues)
    r2 = (1 - residual_var / target_var).item() if target_var > 0 else 0.0

    return {"pooling": pooling, "seq_len": seq_len,
            "best_val_mse": best_val_mse, "test_mse": test_mse, "test_r2": r2}


class LocalCellDataset(Dataset):
    """R2 dataset: single lognormal, local geometric mean per cell, target = mean of cell signals.

    Features: log(x_i) ~ N(0, sigma²) i.i.d. (default sigma=1.0).
    For each sample, features are divided into cells of size `cell_size`.
    Cell signal: S_j = exp(mean(log(|x_j|))) for features in cell j.
    Target: mean_j(S_j) = (1/num_cells) * sum_j S_j.

    This tests whether the pooling primitive can recover the cell signal.
    The FC head must reconstruct the target from pooled summaries.
    """

    def __init__(self, n_samples: int = 2000, seq_len: int = 32, cell_size: int = 4,
                 sigma: float = 1.0, seed: int = 42):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.cell_size = cell_size
        self.sigma = sigma
        self.num_cells = seq_len // cell_size
        assert seq_len % cell_size == 0, "seq_len must be divisible by cell_size"
        self.rng = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Single lognormal features: log(x_i) ~ N(0, sigma^2)
        log_feat = torch.randn(self.seq_len, generator=self.rng) * self.sigma
        x = torch.exp(log_feat)  # x > 0

        # Compute cell signals: S_j = exp(mean(log(|x_j|))) for each cell
        x = x.view(self.num_cells, self.cell_size)
        log_abs = torch.log(torch.abs(x) + EPS)
        cell_signal = torch.exp(log_abs.mean(dim=1))  # (num_cells,)
        target = cell_signal.mean()  # scalar

        return x.view(self.seq_len, 1), target.unsqueeze(-1)


class LocalCellNetwork(nn.Module):
    """R2 network: local pooling on unit cells → FC head.

    GeometricPool(kernel=cell_size, stride=cell_size) → FC(32) → ReLU → FC(1).
    Pooling operates on raw features; no conv, no BatchNorm.
    The head in_features = num_cells = seq_len // cell_size, computed dynamically.
    """

    def __init__(self, cell_size: int = 4, pooling: str = "geo", num_cells: int = 8):
        super().__init__()
        if pooling == "max":
            pool = nn.MaxPool1d(kernel_size=cell_size, stride=cell_size)
        elif pooling == "avg":
            pool = nn.AvgPool1d(kernel_size=cell_size, stride=cell_size)
        elif pooling == "geo":
            from GeometricPool1d import GeometricPool1d
            pool = GeometricPool1d(kernel_size=cell_size, stride=cell_size)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        self.pool = pool
        self.head = nn.Sequential(
            nn.Linear(num_cells, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, L, 1) → (B, 1, L)
        pooled = self.pool(x)  # (B, 1, num_cells)
        return self.head(pooled.flatten(1))  # (B, 1)


if __name__ == "__main__":
    poolings = ["max", "avg", "geo"]
    for seed in [42, 123, 456]:
        print(f"\n--- seed={seed} ---")
        for p in poolings:
            r = run_experiment(p, seed=seed, epochs=50)
            print(f"  {p:>6}: val_mse={r['best_val_mse']:.4f} "
                  f"test_mse={r['test_mse']:.4f} test_r2={r['test_r2']:.4f}")
