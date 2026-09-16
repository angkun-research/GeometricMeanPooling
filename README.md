# GeometricPool

PyTorch implementations and experiments for **signed geometric-mean pooling** (GMP), a parameter-free pooling operation designed for signals with multiplicative structure.

For a pooling window \(W = \{x_1, \ldots, x_k\}\), GMP is

\[
\operatorname{GMP}(W) =
\left(\prod_{i=1}^{k}\operatorname{sign}(x_i)\right)
\exp\left(\frac{1}{k}\sum_{i=1}^{k}
\log\left(\max\left(|x_i|, \varepsilon\right)\right)\right).
\]

The implementation computes the magnitude in log space for numerical stability and preserves the product of the input signs. It can be used as either a local sliding-window operator or a global pooling layer. This repository compares that multiplicative inductive bias with average and max pooling on synthetic sequences, image classification, iterative coarse-graining, and molecular lipophilicity regression.

## Repository layout

```text
GeometricPool1d.py             Signed local/global 1D pooling layer
poolings/geometric_pool2d.py   Signed local/global 2D pooling layer
models/                        CNN and molecular model definitions
tasks/                         Datasets, preprocessing, and train/eval helpers
run_*.py                       Individual experiments and sweeps
collect_*.py                   Multi-run result collection scripts
plots/                         Saved JSON results and figure generators
requirements.txt               Core Python dependencies
```

## Installation

Python 3.10 or newer is recommended because the source uses modern type-hint syntax.

```bash
git clone <repository-url>
cd GeometricMeanPooling
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Some experiment families need additional packages:

```bash
# MNIST, Fashion-MNIST, and CIFAR-10
python -m pip install torchvision

# Figure generation
python -m pip install matplotlib seaborn

# XGBoost molecular baseline
python -m pip install xgboost
```

PyTorch should be installed using the build appropriate for the target CPU or CUDA environment. CUDA is optional for the pooling layers and most single-run scripts, although several large sweep/collector scripts are configured explicitly for two NVIDIA GPUs.

## Quick start

### 1D pooling

```python
import torch

from GeometricPool1d import GeometricPool1d

x = torch.tensor([[[-1.0, 4.0, -9.0, 16.0]]])  # (batch, channels, length)

local_pool = GeometricPool1d(kernel_size=2, stride=2)
global_pool = GeometricPool1d()  # kernel_size=None means global pooling

print(local_pool(x))   # tensor([[[-2., -12.]]])
print(global_pool(x))  # tensor([[[4.8990]]])
```

### 2D pooling

```python
import torch

from poolings.geometric_pool2d import GeometricPool2d

x = torch.randn(8, 32, 14, 14)

local_pool = GeometricPool2d(kernel_size=2, stride=2)
global_pool = GeometricPool2d(kernel_size=1)

print(local_pool(x).shape)   # torch.Size([8, 32, 7, 7])
print(global_pool(x).shape)  # torch.Size([8, 32, 1, 1])
```

Note that this repository uses `kernel_size=None` or `1` as the global-pooling sentinel in 1D, and `kernel_size=1` in 2D. Other kernel sizes select local pooling. Local windows do not use padding, so the usual valid-window output shape applies.

## Running experiments

Run commands from the repository root so that local imports and relative output paths resolve correctly. Pooling choices are `geo`, `max`, and `avg`.

### Synthetic sequence tasks

The synthetic experiments test product-derived classification and regression targets, local unit-cell structure, noise robustness, and scaling with sequence length.

```bash
python run_seq1d_classif.py
python run_seq1d_global.py
python run_seq1d_regression.py
python run_rg_iterative.py
python run_r3_noise.py
```

Several scripts run full grids over pooling methods, seeds, and problem settings and can therefore take substantially longer than a single training run.

### Image classification

MNIST has separate entry points for global, local, and combined (`All`) pooling configurations:

```bash
python run_mnist_config1.py geo  # global pooling
python run_mnist_config2.py geo  # local pooling
python run_mnist_config3.py geo  # local and global pooling
```

Fashion-MNIST and CIFAR-10 accept a model configuration followed by a pooling method:

```bash
python run_fashion_mnist.py Global geo
python run_fashion_mnist.py Local avg
python run_cifar10.py All max
```

Valid model configurations are `Global`, `Local`, and `All`. Torchvision downloads these datasets into `./data` on first use.

### Molecular lipophilicity

Molecular scripts expect:

```text
data/lipophilicity.csv
```

The CSV must contain `SMILES` and `label` columns. The preprocessing code tokenizes SMILES strings, constructs an 80/20 Bemis-Murcko scaffold split, and can also generate 2,048-bit Morgan fingerprints.

Examples:

```bash
python tasks/lipophilicity_preprocess.py
python run_baseline_rf.py
python run_lipophilicity_sweep.py
python run_lipophilicity_baselines.py
```

The dataset is not included in this repository.

## Reproducing saved figures

Precomputed experiment results are stored as JSON files under `plots/`. Figure scripts should also be launched from the repository root:

```bash
python plots/gen_fig1_mnist.py
python plots/gen_fig2_seq.py
python plots/gen_fig2_uc.py
python plots/gen_fig3_heatnoise.py
python plots/gen_fig4.py
python plots/gen_fig5_lipo.py
python plots/gen_fig5_lipo_morgan.py
```

The corresponding publication-ready PDFs are kept in `paper/figs/`. Collector scripts such as `collect_fig1_data.py`, `collect_fig2_*.py`, `collect_fig3_r3.py`, and `collect_lipo_*.py` regenerate the underlying results. Review their constants (`SEEDS`, `EPOCHS`, GPU IDs, worker counts, and output paths) before launching them; the defaults are research-scale workloads and some assume two CUDA devices.

## Implementation details

- Inputs to `GeometricPool1d` have shape `(batch, channels, length)`.
- Inputs to `GeometricPool2d` have shape `(batch, channels, height, width)`.
- Local pooling uses `Tensor.unfold` and supports non-overlapping or overlapping windows through `stride`.
- Global pooling reduces the full sequence or spatial map to one value per channel.
- The magnitude clamp defaults to `1e-6` in 1D and `1e-12` in 2D.
- Since `torch.sign(0) == 0`, any exact zero in a window makes that window's signed output zero.
- For complete, equally sized, non-overlapping groups using the same clamp, hierarchical GMP preserves the corresponding global signed geometric statistic.

## License

This project is available under the [MIT License](LICENSE).
