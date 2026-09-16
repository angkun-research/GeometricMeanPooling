import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import seaborn as sns

# STRICT FONT ADHERENCE
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"

RESULTS_FILE = "./plots/fig3_r3_results.json"

POOLINGS = ["geo", "avg", "max"]
TITLES = {
    "geo": "Geometric Pooling",
    "avg": "Average Pooling",
    "max": "Max Pooling",
}

def load_results(path):
    with open(path, "r") as f:
        return json.load(f)

def infer_axes(results):
    mul_vals = sorted({v["multiplicative_sigma"] for v in results.values()})
    add_vals = sorted({v["additive_sigma"] for v in results.values()})
    return mul_vals, add_vals

def build_matrix(results, pooling, mul_vals, add_vals):
    # rows = multiplicative sigma, cols = additive sigma
    mat = np.full((len(mul_vals), len(add_vals)), np.nan, dtype=float)

    for i, mul in enumerate(mul_vals):
        for j, add in enumerate(add_vals):
            # robust lookup by fields (instead of key-string formatting)
            hit = None
            for rec in results.values():
                if (
                    rec["pooling"] == pooling
                    and rec["multiplicative_sigma"] == mul
                    and rec["additive_sigma"] == add
                ):
                    hit = rec
                    break
            if hit is not None:
                mat[i, j] = hit["test_r2"]

    # same clipping convention as your original script
    return np.clip(mat, 0, 1)

def generate_fig():
    results = load_results(RESULTS_FILE)
    mul_vals, add_vals = infer_axes(results)

    data_mats = [build_matrix(results, p, mul_vals, add_vals) for p in POOLINGS]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=1000, layout="constrained")

    for ax, pooling, mat in zip(axes, POOLINGS, data_mats):
        sns.heatmap(
            mat,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            xticklabels=add_vals,
            yticklabels=mul_vals,
            ax=ax,
            cbar=False,
            vmin=0,
            vmax=1,
        )
        ax.set_title(TITLES[pooling], fontsize=14)
        ax.set_xlabel(r"Additive noise $\sigma_{\mathrm{add}}$", fontsize=12)
        ax.set_ylabel(r"Multiplicative noise $\sigma_{\mathrm{mul}}$", fontsize=12)
        ax.invert_yaxis()

    sm = ScalarMappable(cmap="viridis", norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, shrink=1.0, label=r"$R^2$", pad=0.02)

    fig_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(fig_dir, "..", "plots", "fig3_noise_heatmaps.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    generate_fig()