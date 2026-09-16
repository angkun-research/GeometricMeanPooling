"""Runner for Case B: lognormal unit-cell classification (both variants).

Variant B1: Single lognormal, order parameter split at median
Variant B2: Two lognormals, pre-assigned labels
"""
from tasks.seq1d_classif import run_experiment

def run_mu_sweep(mu_values, poolings, seeds, unit_cell, variant):
    """Run mu sweep for a given variant."""
    print(f"\n=== Variant {variant}: mu sweep (unit_cell={unit_cell}) ===")
    for mu in mu_values:
        print(f"\n--- mu={mu} ---")
        for seed in seeds:
            for p in poolings:
                r = run_experiment(
                    pooling=p, mu=mu, sigma=1.0, seed=seed,
                    epochs=200, unit_cell=unit_cell, variant=variant)
                print(f"  mu={mu:>4} seed={seed:>3} {p:>6}: test={r['test_acc']:.4f}")


def run_default(unit_cell, seeds, variant):
    """Run default experiment for a given variant."""
    print(f"\n=== Variant {variant}: default (unit_cell={unit_cell}) ===")
    for seed in seeds:
        print(f"\n--- seed={seed} ---")
        for p in ["max", "avg", "geo"]:
            r = run_experiment(
                pooling=p, mu=0.5, sigma=1.0, seed=seed,
                epochs=50, unit_cell=unit_cell, variant=variant)
            print(f"  {p:>6}: val={r['best_val_acc']:.4f} test={r['test_acc']:.4f}")


if __name__ == "__main__":
    poolings = ["max", "avg", "geo"]
    seeds = [42, 123, 456]
    mu_values = [0.1, 0.2, 0.3, 0.5, 1.0]

    # Variant B1: single lognormal
    print("=== VARIANT B1: Single lognormal (order parameter split) ===")
    run_default(unit_cell=2, seeds=seeds, variant="single")

    # Variant B2: two lognormals
    print("\n\n=== VARIANT B2: Two lognormals (pre-assigned labels) ===")
    run_default(unit_cell=2, seeds=seeds, variant="two")
    run_mu_sweep(mu_values, poolings, seeds, unit_cell=2, variant="two")

    # Summary
    print("\n\n=== SUMMARY: Variant comparison ===")
    print("B1: Single lognormal, order parameter split at median")
    print("B2: Two lognormals, pre-assigned labels")
    print("mu controls: B1 -> feature spread, B2 -> marginal overlap")
