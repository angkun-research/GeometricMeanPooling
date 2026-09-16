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

def generate_fig4():
    # Data
    steps = np.arange(1, 9)
    geo_mae = [0.077] * 8
    avg_mae = [0.289, 0.507, 0.664, 0.766, 0.831, 0.871, 0.895, 0.909]
    max_mae = [0.874, 2.149, 3.959, 6.459, 9.964, 14.965, 22.560, 34.881]

    plt.figure(figsize=(6, 4.5), dpi=1000)

    plt.plot(steps, geo_mae, color='#1f77b4', marker='o', linewidth=2, markersize=6, label='Geometric Pooling')
    plt.plot(steps, avg_mae, color='#2ca02c', marker='s', linewidth=2, markersize=6, label='Average Pooling')
    plt.plot(steps, max_mae, color='#d62728', marker='^', linewidth=2, markersize=6, label='Max Pooling')

    plt.yscale('log')
    plt.xlabel('RG Step (Iteration)', fontsize=12)
    plt.ylabel('Mean Absolute Error (MAE)', fontsize=12)
    #plt.title('Iterative RG Stability (L=256, $\sigma=1.0$)', fontsize=13)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend(frameon=False, fontsize=14)
    plt.xticks(steps)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'paper', 'figs', 'fig4_rg_flow.pdf')
    plt.savefig(out, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    generate_fig4()
