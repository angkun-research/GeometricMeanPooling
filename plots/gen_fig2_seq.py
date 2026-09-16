import json
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"

B3_RESULTS_FILE = "./plots/fig2_b3_seq_results.json"
R2_RESULTS_FILE = "./plots/fig2_r2_seq_results.json"

POOLINGS = ["geo", "max", "avg"]
SEQ_LENS = [18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
SEEDS = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
UNIT_CELL = 3

COLORS = {"geo": "#1f77b4", "max": "#ff7f0e", "avg": "#2ca02c"}
LABELS = {"geo": "Geometric", "max": "Max", "avg": "Average"}


def load_b3_data() -> np.ndarray:
    """Returns array of shape (3, 10, 10): (pooling, seq_len, seed)."""
    with open(B3_RESULTS_FILE, "r") as f:
        results = json.load(f)

    data = np.zeros((3, len(SEQ_LENS), len(SEEDS)))
    for pi, pooling in enumerate(POOLINGS):
        for li, seq_len in enumerate(SEQ_LENS):
            for si, seed in enumerate(SEEDS):
                key = f"b3_uc{UNIT_CELL}_L{seq_len}_{pooling}_seed{seed}"
                data[pi, li, si] = results[key]["test_acc"]
    return data


def load_r2_data() -> np.ndarray:
    """Returns array of shape (3, 10, 10): (pooling, seq_len, seed)."""
    with open(R2_RESULTS_FILE, "r") as f:
        results = json.load(f)

    data = np.zeros((3, len(SEQ_LENS), len(SEEDS)))
    for pi, pooling in enumerate(POOLINGS):
        for li, seq_len in enumerate(SEQ_LENS):
            for si, seed in enumerate(SEEDS):
                key = f"r2_local_uc{UNIT_CELL}_L{seq_len}_{pooling}_seed{seed}"
                data[pi, li, si] = results[key]["test_r2"]
    return data


def plot_combined(b3_data: np.ndarray, r2_data: np.ndarray):
    b3_means = b3_data.mean(axis=2)
    b3_stds = b3_data.std(axis=2)

    r2_means = r2_data.mean(axis=2)
    r2_stds = r2_data.std(axis=2)

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(6, 4),
        dpi=300,
        sharex=True,
        gridspec_kw={"hspace": 0.0},
    )

    for pi, pooling in enumerate(POOLINGS):
        ax_top.plot(
            SEQ_LENS,
            b3_means[pi],
            marker="o",
            linewidth=1.6,
            label=LABELS[pooling],
            color=COLORS[pooling],
        )
        ax_top.fill_between(
            SEQ_LENS,
            b3_means[pi] - b3_stds[pi],
            b3_means[pi] + b3_stds[pi],
            alpha=0.2,
            color=COLORS[pooling],
        )

    for pi, pooling in enumerate(POOLINGS):
        ax_bottom.plot(
            SEQ_LENS,
            r2_means[pi],
            marker="o",
            linewidth=1.6,
            label=LABELS[pooling],
            color=COLORS[pooling],
        )
        ax_bottom.fill_between(
            SEQ_LENS,
            r2_means[pi] - r2_stds[pi],
            r2_means[pi] + r2_stds[pi],
            alpha=0.2,
            color=COLORS[pooling],
        )

    ax_top.set_ylabel("Test accuracy", fontsize=12)
    ax_bottom.set_ylabel(r"Test $R^2$", fontsize=12)
    ax_bottom.set_xlabel(r"Sequence length $L$", fontsize=12)

    ax_top.set_ylim(0.35, 1.05)
    ax_top.set_yticks(np.arange(0.4, 1.05, 0.2))
    ax_top.set_yticklabels([f"{y:.1f}" for y in np.arange(0.4, 1.05, 0.2)], fontsize=11)

    ax_bottom.set_ylim(0.2, 1.05)
    ax_bottom.set_yticks(np.arange(0.2, 1.05, 0.2))
    ax_bottom.set_yticklabels([f"{y:.1f}" for y in np.arange(0.2, 1.05, 0.2)], fontsize=11)

    ax_bottom.set_xlim(15, 185)
    ax_bottom.set_xticks([30, 60, 90, 120, 150, 180])
    ax_bottom.set_xticklabels([30, 60, 90, 120, 150, 180], fontsize=12)

    ax_top.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax_bottom.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax_top.legend(loc="best", fontsize=11)

    # Use manual spacing instead of tight_layout
    fig.subplots_adjust(left=0.14, right=0.98, top=0.97, bottom=0.10, hspace=0.0)
    
    fig_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(fig_dir, "..", "plots", "fig2_seq.pdf")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    b3_data = load_b3_data()
    r2_data = load_r2_data()

    print("B3 shape:", b3_data.shape)
    print("R2 shape:", r2_data.shape)

    for pi, pooling in enumerate(POOLINGS):
        print(f"{pooling} B3 mean acc = {b3_data[pi].mean(axis=1).round(3)}")
        print(f"{pooling} R2 mean R^2 = {r2_data[pi].mean(axis=1).round(3)}")

    plot_combined(b3_data, r2_data)