"""Case A runner: Type 1 data (zero-mean Gaussian) with global pooling."""
from tasks.seq1d_global import run_experiment


def run_default():
    """Default run: 3 seeds."""
    poolings = ["max", "avg", "geo"]
    for seed in [42, 123, 456]:
        print(f"\n--- seed={seed} ---")
        for p in poolings:
            r = run_experiment(p, seed=seed, epochs=100)
            print(f"  {p:>6}: val={r['best_val_acc']:.4f} test={r['test_acc']:.4f}")


def run_rho_sweep():
    """Sweep rho to probe how correlation affects pooling performance."""
    poolings = ["max", "avg", "geo"]
    for rho in [0.0, 0.3, 0.5, 0.7, 0.9]:
        print(f"\n=== rho={rho} ===")
        for seed in [42]:
            for p in poolings:
                r = run_experiment(p, rho=rho, seed=seed, epochs=100)
                print(f"  {p:>6}: val={r['best_val_acc']:.4f} test={r['test_acc']:.4f}")


def run_seq_len_sweep():
    """Sweep seq_len to probe scaling with input size."""
    poolings = ["max", "avg", "geo"]
    for seq_len in [16, 32, 64, 128]:
        print(f"\n=== seq_len={seq_len} ===")
        for seed in [42]:
            for p in poolings:
                r = run_experiment(p, seq_len=seq_len, seed=seed, epochs=100)
                print(f"  {p:>6}: val={r['best_val_acc']:.4f} test={r['test_acc']:.4f}")


if __name__ == "__main__":
    # Default run: 3 seeds
    run_default()
    # rho sweep
    run_rho_sweep()
    # seq_len sweep
    run_seq_len_sweep()
