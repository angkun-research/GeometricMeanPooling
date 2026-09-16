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

def generate_fig1():
    # Data
    seq_len = [8, 16, 32, 64, 128, 256]
    geo = [0.786, 0.795, 0.787, 0.782, 0.789, 0.678]
    avg = [0.276, 0.277, 0.269, 0.253, 0.218, 0.245]
    max_val = [0.085, 0.042, 0.023, 0.014, 0.005, -0.009]

    plt.figure(figsize=(6, 4.5))

    plt.plot(seq_len, geo, color='#1f77b4', marker='o', linewidth=2, markersize=6, label='Geometric Pooling')
    plt.plot(seq_len, avg, color='#2ca02c', marker='s', linewidth=2, markersize=6, label='Average Pooling')
    plt.plot(seq_len, max_val, color='#d62728', marker='^', linewidth=2, markersize=6, label='Max Pooling')

    plt.xlabel(r'Sequence Length ($L$)', fontsize=12)
    plt.ylabel(r'$R^2$ Score', fontsize=12)
    #plt.title(r'Scaling of $R^2$ with Sequence Length ($\sigma=1.0, p=0.25$)', fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=False, fontsize=14, loc='right')
    plt.xlim(6, 260)
    plt.ylim(-0.1, 1.0)
    plt.tight_layout()
    fig_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(fig_dir, '..', 'paper', 'figs', 'fig1_scaling.pdf')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    generate_fig1()
