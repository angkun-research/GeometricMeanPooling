"""Signed geometric-mean pooling with configurable kernel and stride.

Poolings computes exp(mean(log(|x|))) * sign(prod(x)) over sliding windows
along the sequence dimension (L). Zero entries produce zero output because
clamp(eps) in the log-domain prevents NaN propagation.

Parameters:
    kernel_size: window width in sites. None or 1 -> pool entire sequence to 1
                 (global geometric pooling, like nn.AdaptiveAvgPool1d(1)).
                 Otherwise -> sliding window over the sequence.
    stride:      step between successive windows. None -> equals kernel_size
                 (non-overlapping, full coverage). Only used when kernel_size > 1.
    eps:         lower bound for clamp(|x|) inside log.

Formula (local mode):  L_out = 1 + (L - kernel_size) // stride
Formula (global mode): L_out = 1

Input shape:  (B, C, L)
Output shape: (B, C, L_out)  -  (B, C) when kernel_size is None or 1
"""

import torch


class GeometricPool1d(torch.nn.Module):
    def __init__(self, kernel_size: int | None = None, stride: int | None = None, eps: float = 1e-6) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool along the sequence dimension.

        Args:
            x: (B, C, L)

        Returns:
            (B, C, L_out)
        """
        L = x.size(-1)

        # Global pooling: kernel_size=1 or None -> pool entire sequence to 1
        if self.kernel_size is None or self.kernel_size == 1:
            w = x.view(x.size(0), x.size(1), -1)  # (B, C, L)
            sign = torch.prod(torch.sign(w), dim=-1)  # (B, C)
            log_abs = torch.log(torch.clamp(torch.abs(w), min=self.eps))
            geom = torch.exp(torch.mean(log_abs, dim=-1))
            return sign.unsqueeze(-1) * geom.unsqueeze(-1)  # (B, C, 1)

        k = self.kernel_size
        s = self.stride if self.stride is not None else k

        if k > L:
            raise ValueError(f"kernel_size ({k}) must be <= input length ({L})")
        if s < 1:
            raise ValueError(f"stride must be >= 1, got {s}")

        # Sliding windows: (B, C, L) -> (B, C, L_out, k)
        w = x.unfold(2, k, s)

        # Signed geometric mean per window
        sign = torch.prod(torch.sign(w), dim=-1)  # (B, C, L_out)
        log_abs = torch.log(torch.clamp(torch.abs(w), min=self.eps))
        geom = torch.exp(torch.mean(log_abs, dim=-1))
        return sign * geom  # (B, C, L_out)
