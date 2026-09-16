"""Type 2 classification: lognormal unit-cell data.

Two construction strategies:
- Variant B1 (single lognormal): order parameter split at median
- Variant B2 (two lognormals): pre-assigned labels from two lognormal distributions
- Variant B3 (Gaussian local product sum): sum of signed geometric means over cells
Pooling operates directly on raw features — geometric mean within cells recovers the signal.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from GeometricPool1d import GeometricPool1d
from utils import pick_best_device, set_seed
# import numpy as np

# # ------ Reproducibility ------
# def set_seed(seed):
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

#     # deterministic behavior
#     torch.use_deterministic_algorithms(True)   # raise error on nondeterministic ops
#     torch.backends.cudnn.deterministic = True # disable nondeterministic CuDNN algorithms
#     torch.backends.cudnn.benchmark = False # disable CuDNN auto-tuning (introduces nondeterminism)

# Dataset defaults for Type2 (lognormal).
MU = 0.5
SIGMA = 1.0

def _compute_threshold(seq_len: int, sigma: float, unit_cell: int, seed: int = 42) -> float:
    """Compute median of order parameter S = sum of cell geometric means."""
    n_ref = 100000
    rng = torch.Generator().manual_seed(seed)
    log_feat = torch.randn(n_ref * seq_len, generator=rng) * sigma
    feat = torch.exp(log_feat).view(n_ref, seq_len)
    n_cells = seq_len // unit_cell
    S = torch.zeros(n_ref)
    for j in range(n_cells):
        cell_prod = torch.ones(n_ref)
        for k in range(unit_cell):
            cell_prod *= feat[:, j * unit_cell + k]
        S += cell_prod
    return float(S.median())


class SeqDataset(Dataset):
    """Lognormal unit-cell dataset.

    Single lognormal distribution generates all data: log(x_i) ~ N(0, sigma^2).
    Order parameter S = sum_j geo_mean(cell_j). Samples split at median:
    class 0 = high S (above median), class 1 = low S (below median).
    """

    def __init__(self, n_samples: int = 2000, seq_len: int = 32,
                 sigma: float = SIGMA,
                 unit_cell: int = 2, seed: int = 42,
                 threshold: float | None = None):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.sigma = sigma
        self.unit_cell = unit_cell
        self.rng = torch.Generator().manual_seed(seed)
        self.threshold = threshold if threshold is not None else _compute_threshold(seq_len, sigma, unit_cell, seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # Generate from the single lognormal (log-mean = 0)
        log_feat = torch.randn(self.seq_len, generator=self.rng) * self.sigma
        feat = torch.exp(log_feat)
        seq = feat + torch.randn(self.seq_len, generator=self.rng) * 0.01

        # Compute order parameter S for this sample
        n_cells = self.seq_len // self.unit_cell
        S = 0.0
        for j in range(n_cells):
            cs = 1.0
            for k in range(self.unit_cell):
                cs *= feat[j * self.unit_cell + k]
            S += cs

        label = 0 if S >= self.threshold else 1
        return seq.unsqueeze(-1), label  # [seq_len, 1]


class SeqDatasetTwoLognormal(Dataset):
    """Lognormal unit-cell dataset with two lognormal distributions.

    Class 0: log(x_i) ~ N(+mu, sigma^2) -> features concentrated > 1
    Class 1: log(x_i) ~ N(-mu, sigma^2) -> features concentrated < 1
    Label is pre-assigned via idx % 2, determines the distribution.
    mu controls marginal overlap: small mu = heavy overlap, large mu = easy separation.
    """

    def __init__(self, n_samples: int = 2000, seq_len: int = 32,
                 mu: float = MU, sigma: float = SIGMA,
                 seed: int = 42):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.mu = mu
        self.sigma = sigma
        self.rng = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        label = idx % 2
        log_mean = self.mu if label == 0 else -self.mu
        log_feat = torch.randn(self.seq_len, generator=self.rng) * self.sigma + log_mean
        feat = torch.exp(log_feat)
        seq = feat + torch.randn(self.seq_len, generator=self.rng) * 0.01
        return seq.unsqueeze(-1), label  # [seq_len, 1]


class SeqDatasetB3(Dataset):
    """Variant B3: Gaussian features with local product signal.

    Features x_i ~ N(0, sigma^2).
    Local cell signal: s_j = sign(prod(x)) * exp(mean(log|x|)).
    Global Signal: S = sum(s_j). Label = sign(S).
    """
    def __init__(self, n_samples: int = 2000, seq_len: int = 32,
                 sigma: float = SIGMA, unit_cell: int = 2, seed: int = 42):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.sigma = sigma
        self.unit_cell = unit_cell
        self.rng = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # Generate x ~ N(0, sigma^2)
        seq = torch.randn(self.seq_len, generator=self.rng) * self.sigma

        # Compute local signals s_j
        n_cells = self.seq_len // self.unit_cell
        S = 0.0
        for j in range(n_cells):
            cell = seq[j * self.unit_cell : (j + 1) * self.unit_cell]
            # local signal: sign(prod) * exp(mean(log|x|))
            sign = torch.prod(torch.sign(cell))
            # Using eps=1e-12 to match GeometricPool1d
            log_abs = torch.log(torch.clamp(torch.abs(cell), min=1e-12))
            val = torch.exp(torch.mean(log_abs))
            S += sign * val

        label = 0 if S >= 0 else 1
        return seq.unsqueeze(-1), label


class Small1DCNN(nn.Module):
    """Pooling-only network for local geometric pooling comparison.

    Pool(kernel=unit_cell, stride=unit_cell) on raw features -> FC.
    Pooling operates directly on the features where the class signal lives.
    No conv, no BatchNorm, no ReLU.
    """

    def __init__(
        self, n_feat: int = 1, n_classes: int = 2, pooling: str = "geo",
        kernel_size: int = 3, seq_len: int = 32,
    ):
        super().__init__()

        pool_k = min(kernel_size, max(2, seq_len - 1))
        pool_s = min(pool_k, seq_len - 1)
        L_after_pool = max(1, (seq_len - pool_k) // pool_s + 1)

        if pooling == "max":
            pool = nn.MaxPool1d(pool_k, stride=pool_s)
        elif pooling == "avg":
            pool = nn.AvgPool1d(pool_k, stride=pool_s)
        elif pooling == "geo":
            pool = GeometricPool1d(pool_k, stride=pool_s)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        self.features = nn.Sequential(pool)
        self.classifier = nn.Linear(n_feat * L_after_pool, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x.transpose(1, 2))
        return self.classifier(out.flatten(1))


def make_dataloaders(
    train_n: int = 2000, val_n: int = 500, test_n: int = 1000,
    sigma: float = SIGMA, seed: int = 42, unit_cell: int = 2, seq_len: int = 32,
) -> tuple[Dataset, Dataset, Dataset]:
    # Shared threshold across train/val/test
    threshold = _compute_threshold(seq_len, sigma, unit_cell, seed)
    return (SeqDataset(n_samples=train_n, seq_len=seq_len, sigma=sigma, seed=seed, unit_cell=unit_cell, threshold=threshold),
            SeqDataset(n_samples=val_n, seq_len=seq_len, sigma=sigma, seed=seed + 1, unit_cell=unit_cell, threshold=threshold),
            SeqDataset(n_samples=test_n, seq_len=seq_len, sigma=sigma, seed=seed + 2, unit_cell=unit_cell, threshold=threshold))


def make_dataloaders_two_lognormal(
    train_n: int = 2000, val_n: int = 500, test_n: int = 1000,
    mu: float = MU, sigma: float = SIGMA, seed: int = 42, seq_len: int = 32,
) -> tuple[Dataset, Dataset, Dataset]:
    """Dataloaders for two-lognormal variant (pre-assigned labels)."""
    return (SeqDatasetTwoLognormal(n_samples=train_n, seq_len=seq_len, mu=mu, sigma=sigma, seed=seed),
            SeqDatasetTwoLognormal(n_samples=val_n, seq_len=seq_len, mu=mu, sigma=sigma, seed=seed + 1),
            SeqDatasetTwoLognormal(n_samples=test_n, seq_len=seq_len, mu=mu, sigma=sigma, seed=seed + 2))


def make_dataloaders_b3(
    train_n: int = 2000, val_n: int = 500, test_n: int = 1000,
    sigma: float = SIGMA, seed: int = 42, unit_cell: int = 2, seq_len: int = 32,
) -> tuple[Dataset, Dataset, Dataset]:
    """Dataloaders for Variant B3 (Local Product Sum)."""
    return (SeqDatasetB3(n_samples=train_n, seq_len=seq_len, sigma=sigma, seed=seed, unit_cell=unit_cell),
            SeqDatasetB3(n_samples=val_n, seq_len=seq_len, sigma=sigma, seed=seed + 1, unit_cell=unit_cell),
            SeqDatasetB3(n_samples=test_n, seq_len=seq_len, sigma=sigma, seed=seed + 2, unit_cell=unit_cell))


def run_experiment(
    pooling: str, mu: float = MU, sigma: float = SIGMA,
    epochs: int = 50, lr: float = 0.01, seed: int = 42,
    unit_cell: int = 2, variant: str = "single", seq_len: int = 32, 
    use_final_model: bool = False, device: torch.device | None = None
) -> dict:
    """Run experiment for Case B.

    Args:
        variant: 'single' for order-parameter split, 'two' for two-lognormal, 'b3' for local product sum
    """
    #torch.manual_seed(seed)
    set_seed(42)
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device is None:
        device = pick_best_device()
    print(f"Using device: {device}")
    
    if variant == "single":
        make_dl = make_dataloaders
        train_ds, val_ds, test_ds = make_dl(
            sigma=sigma, seed=seed, unit_cell=unit_cell, seq_len=seq_len)
    elif variant == "two":
        make_dl = make_dataloaders_two_lognormal
        train_ds, val_ds, test_ds = make_dl(
            mu=mu, sigma=sigma, seed=seed, seq_len=seq_len)
    elif variant == "b3":
        make_dl = make_dataloaders_b3
        train_ds, val_ds, test_ds = make_dl(
            sigma=sigma, seed=seed, unit_cell=unit_cell, seq_len=seq_len)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=500)
    test_loader = DataLoader(test_ds, batch_size=1000)

    n_feat = train_ds[0][0].shape[-1]
    model = Small1DCNN(n_feat=n_feat, pooling=pooling, kernel_size=unit_cell, seq_len=seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

    model.load_state_dict(best_state) if not use_final_model else None
    tc, tt = 0, 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        tc += (model(x).argmax(1) == y).sum().item()
        tt += x.size(0)
    
    # best_acc = 0.0
    # for epoch in range(1, epochs + 1):
    #     model.train()
    #     for x, y in train_loader:
    #         x, y = x.to(device), y.to(device)
    #         out = model(x)
    #         loss = F.cross_entropy(out, y)
    #         optimizer.zero_grad(); loss.backward(); optimizer.step()
    
    # model.eval()
    # tc = 0
    # tt = 0
    # with torch.no_grad():
    #     for x, y in test_loader:
    #         x, y = x.to(device), y.to(device)
    #         out = model(x)
    #         tc += (out.argmax(1) == y).sum().item()
    #         tt += x.size(0)

    return {"pooling": pooling, "mu": mu, "sigma": sigma,
            "best_val_acc": best_acc, "test_acc": tc / tt, "variant": variant}


if __name__ == "__main__":
    poolings = ["max", "avg", "geo"]
    for seed in [42, 123, 456]:
        print(f"\n--- seed={seed} ---")
        for p in poolings:
            r = run_experiment(p, seed=seed, epochs=50)
            print(f"{p:>6}: val={r['best_val_acc']:.4f} test={r['test_acc']:.4f}")
