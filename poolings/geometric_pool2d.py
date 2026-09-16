"""Geometric pooling for 2D spatial inputs."""

import torch
import torch.nn as nn


class GeometricPool2d(nn.Module):
    """Signed geometric-mean pooling over 2D spatial windows.

    For each kernel window within a feature map:
        output = sign(prod(w)) * exp(mean(log(clamp(|w|, eps))))

    Parameters:
        kernel_size: window size (H, W) or int for square. (1,1) -> global pooling
                     (like nn.AdaptiveAvgPool2d(1)). Otherwise -> sliding window.
        stride:      step size, default = kernel_size. Only used when kernel_size > 1.
        eps:         lower bound for clamp(|x|) inside log.

    Input shape:  (B, C, H, W)
    Output shape: (B, C, H_out, W_out)  -  (B, C, 1, 1) when kernel_size=(1,1)
    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.kernel_h, self.kernel_w = _pair(kernel_size)
        self.stride_h = self.stride_w = self.kernel_h if stride is None else _pair(stride)[0] if isinstance(stride, tuple) else stride
        # Handle stride tuple
        if isinstance(stride, tuple):
            self.stride_h, self.stride_w = stride
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool 2D spatial windows.

        Args:
            x: (B, C, H, W)

        Returns:
            (B, C, H_out, W_out)
        """
        B, C, H, W = x.size()
        k_h, k_w = self.kernel_h, self.kernel_w
        s_h, s_w = self.stride_h, self.stride_w

        # Global pooling: kernel_size=1 -> pool entire spatial map to (B, C, 1, 1)
        if k_h == 1 and k_w == 1:
            w = x.view(B, C, -1)  # (B, C, H*W)
            sign = torch.prod(torch.sign(w), dim=-1)  # (B, C)
            log_abs = torch.log(torch.clamp(torch.abs(w), min=self.eps))
            geom = torch.exp(torch.mean(log_abs, dim=-1))
            return sign.unsqueeze(-1).unsqueeze(-1) * geom.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)

        if k_h > H or k_w > W:
            raise ValueError(f"kernel_size ({k_h}x{k_w}) must be <= input size ({H}x{W})")

        # Sliding windows via double unfold: (B, C, H, W) -> (B, C, H_out, W_out, k_h, k_w)
        w = x.unfold(2, k_h, s_h).unfold(3, k_w, s_w)

        # Signed geometric mean per window (reduce last two dims sequentially)
        sign = torch.prod(torch.sign(w), dim=-1)  # (B, C, H_out, W_out, k_h)
        sign = torch.prod(sign, dim=-1)  # (B, C, H_out, W_out)
        log_abs = torch.log(torch.clamp(torch.abs(w), min=self.eps))
        log_geom = torch.mean(log_abs, dim=-1)  # (B, C, H_out, W_out, k_h)
        log_geom = torch.mean(log_geom, dim=-1)  # (B, C, H_out, W_out)
        geom = torch.exp(log_geom)
        return sign * geom  # (B, C, H_out, W_out)


def _pair(x):
    if isinstance(x, int):
        return (x, x)
    return tuple(x)
