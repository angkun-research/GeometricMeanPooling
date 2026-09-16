import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from GeometricPool1d import GeometricPool1d

def generate_data(n_samples=2000, seq_len=256, seed=42, noise_sigma=0.0, noise_sparsity=0.0, unsigned=True):
    """
    Generates data for RG stability testing.
    Returns: x_noisy (B, 1, L), targets (B,)
    """
    rng = torch.Generator().manual_seed(seed)

    # Ground truth features
    if unsigned:
        # log(x_i) ~ N(0, 1) -> all x_i > 0
        log_feat = torch.randn(n_samples, seq_len, generator=rng)
        x_clean = torch.exp(log_feat)
    else:
        # Zero-mean Gaussian: features can be positive or negative
        x_clean = torch.randn(n_samples, seq_len, generator=rng)

    # Compute global geometric mean as target (the fixed point)
    if unsigned:
        # Target = exp(mean(log(x)))
        target = torch.exp(torch.mean(torch.log(x_clean + 1e-12), dim=1))
    else:
        # Signed Geometric Mean: sign(prod) * exp(mean(log|x|))
        sign = torch.prod(torch.sign(x_clean + 1e-12), dim=1)
        log_mag = torch.mean(torch.log(torch.abs(x_clean) + 1e-12), dim=1)
        target = sign * torch.exp(log_mag)

    # Apply sparse multiplicative noise to get x_observed
    x_noisy = x_clean.clone()
    if noise_sigma > 0 and noise_sparsity > 0:
        noise_mask = torch.rand(n_samples, seq_len, generator=rng) < noise_sparsity
        noise = torch.randn(n_samples, seq_len, generator=rng) * noise_sigma
        x_noisy[noise_mask] *= torch.exp(noise[noise_mask])

    # Shape for pooling: (B, C, L) where C=1
    return x_noisy.unsqueeze(1), target

def run_rg_experiment(pooling_type="geo", sigma=1.0, sparsity=0.25, unsigned=True, n_samples=2000, seq_len=256):
    """
    Performs iterative pooling and measures drift from target at each step.
    """
    # Data generation
    x, targets = generate_data(n_samples=n_samples, seq_len=seq_len, noise_sigma=sigma, noise_sparsity=sparsity, unsigned=unsigned)

    # Setup pooler (k=2, s=2)
    if pooling_type == "geo":
        pooler = GeometricPool1d(kernel_size=2, stride=2)
    elif pooling_type == "max":
        pooler = nn.MaxPool1d(kernel_size=2, stride=2)
    elif pooling_type == "avg":
        pooler = nn.AvgPool1d(kernel_size=2, stride=2)
    else:
        raise ValueError("Unknown pooling type")

    # Iterative RG flow
    current_x = x
    results = []

    # We expect 8 iterations for L=256 -> 1 (2^8 = 256)
    for step in range(1, 9):
        current_x = pooler(current_x)

        # Final result at this scale: since we want to compare it to the GLOBAL target,
        # and current_x has length L / (2^step), we can either take a global pool of the remainder
        # or just wait until step 8. But for RG stability, we want to see if the 'representative'
        # value is preserved. So at each step, we compute the CURRENT geometric mean of the remaining sites.

        # unified global geometric-mean computation (works for geo/max/avg)
        eps = 1e-12
        sign = torch.prod(torch.sign(current_x + eps), dim=-1)
        log_mag = torch.mean(torch.log(torch.abs(current_x) + eps), dim=-1)
        val = sign * torch.exp(log_mag)

        # Metric: Mean Absolute Error from the ground truth target
        mae = torch.mean(torch.abs(val - targets)).item()
        results.append(mae)

    return results

if __name__ == "__main__":
    # Settings
    SIGMA = 1.0
    SPARSITY = 0.25
    UNSIGNED = True # Main case: Lognormal
    SAMPLES = 2000
    L = 256

    print(f"Running Iterative RG Stability Test (L={L}, sigma={SIGMA}, sparsity={SPARSITY})")

    poolings = ["geo", "max", "avg"]
    final_results = {}

    for p in poolings:
        print(f"Testing {p}...")
        errs = run_rg_experiment(pooling_type=p, sigma=SIGMA, sparsity=SPARSITY, unsigned=UNSIGNED, n_samples=SAMPLES, seq_len=L)
        final_results[p] = errs

    # Format as DataFrame for easy viewing/saving
    df = pd.DataFrame(final_results, index=[f"Step {i+1}" for i in range(8)])
    print("\nMean Absolute Error vs RG Iteration Level:")
    print(df)

    df.to_csv("data/rg_stability_results.csv")
    print("\nResults saved to data/rg_stability_results.csv")
