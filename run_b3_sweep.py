import torch
from tasks.seq1d_classif import run_experiment

def run_sweep():
    unit_cells = [2, 3, 4, 5, 6]
    seq_len = 120
    poolings = ["max", "avg", "geo"]
    seed = 42

    print(f"=== Variant B3 Sweep (seq_len={seq_len}, seed={seed}) ===")
    print(f"{'Cell':<6} {'Pool':<6} {'Test Acc':<10}")
    print("-" * 25)

    for uc in unit_cells:
        for p in poolings:
            # Run a single experiment for each combination
            r = run_experiment(
                pooling=p, sigma=1.0, seed=seed, 
                epochs=100, unit_cell=uc, variant="b3", seq_len=seq_len,
                device = 'cpu'
            )
            print(f"{uc:<6} {p:<6} {r['test_acc']:.4f}")

if __name__ == "__main__":
    run_sweep()
