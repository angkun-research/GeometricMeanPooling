import os
import matplotlib.pyplot as plt
import numpy as np

# STRICT FONT ADHERENCE
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

def generate_fig3():
    # Panel A: Accuracy vs mu
    mu = [0.1, 0.2, 0.3, 0.5, 1.0]
    geo_acc = [0.7, 0.843, 0.945, 0.992, 1]
    avg_acc = [0.662, 0.807, 0.906, 0.986, 1]
    max_acc = [0.633, 0.753, 0.849, 0.957, 0.999]

    # Panel B: R2 vs cell_size
    cell_size = [2, 4, 8, 16]
    geo_r2 = [1, 1, 1, 1]
    avg_r2 = [0.798, 0.729, 0.686, 0.655]
    max_r2 = [0.618, 0.435, 0.303, 0.204]

    # -------- Figure 3a --------
    fig_a, ax1 = plt.subplots(figsize=(6, 4.5), dpi=1000)
    ax1.plot(mu, geo_acc, color='#1f77b4', marker='o', linewidth=2, markersize=6, label='Geometric Pooling')
    ax1.plot(mu, avg_acc, color='#2ca02c', marker='s', linewidth=2, markersize=6, label='Average Pooling')
    ax1.plot(mu, max_acc, color='#d62728', marker='^', linewidth=2, markersize=6, label='Max Pooling')
    ax1.set_xlabel(r'Marginal Shift $\mu$', fontsize=12)
    ax1.set_ylabel(r'Accuracy', fontsize=12)
    ax1.set_title('(a) Classification Accuracy vs. Marginal Overlap', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(frameon=False, fontsize=14)
    ax1.set_ylim(0.6, 1.02)
    fig_a.tight_layout()
    fig_dir = os.path.dirname(os.path.abspath(__file__))
    out_a = os.path.join(fig_dir, '..', 'paper', 'figs', 'fig3_transition_a.pdf')
    fig_a.savefig(out_a, bbox_inches='tight')
    plt.close(fig_a)

    # -------- Figure 3b --------
    fig_b, ax2 = plt.subplots(figsize=(6, 4.5), dpi=1000)
    ax2.plot(cell_size, geo_r2, color='#1f77b4', marker='o', linewidth=2, markersize=6, label='Geometric Pooling')
    ax2.plot(cell_size, avg_r2, color='#2ca02c', marker='s', linewidth=2, markersize=6, label='Average Pooling')
    ax2.plot(cell_size, max_r2, color='#d62728', marker='^', linewidth=2, markersize=6, label='Max Pooling')
    ax2.set_xlabel(r'Cell Size', fontsize=12)
    ax2.set_ylabel(r'$R^2$ Score', fontsize=12)
    ax2.set_title('(b) Regression $R^2$ vs. Coarse-Graining Scale', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(frameon=False, fontsize=14)
    ax2.set_ylim(0, 1.05)
    fig_b.tight_layout()
    out_b = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'paper', 'figs', 'fig3_transition_b.pdf')
    fig_b.savefig(out_b, bbox_inches='tight')
    plt.close(fig_b)

if __name__ == "__main__":
    generate_fig3()
