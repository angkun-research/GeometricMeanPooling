"""CNN architecture with injectable pooling layer."""

import torch
import torch.nn as nn

from poolings.geometric_pool2d import GeometricPool2d


def get_pooling(name: str, kernel_size: int, stride: int | None = None) -> nn.Module:
    """Create a pooling layer by name."""
    if name == "max":
        return nn.MaxPool2d(kernel_size, stride=stride or kernel_size)
    if name == "avg":
        return nn.AvgPool2d(kernel_size, stride=stride or kernel_size)
    if name == "geo":
        return GeometricPool2d(kernel_size, stride=stride or kernel_size)
    raise ValueError(f"Unknown pooling: {name}")


class MinimalCNNGlobal(nn.Module):
    """Minimal CNN for Global Pooling test (Config 1).

    Conv(1->32, 3x3, pad=1) -> ReLU -> Conv(32->32, 3x3, pad=1) -> ReLU -> GlobalPool -> FC(32->10)
    """
    def __init__(self, pooling: str = "geo") -> None:
        super().__init__()
        self.pooling_type = pooling
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        B, C, H, W = x.size()

        if self.pooling_type == "geo":
            pool = GeometricPool2d(kernel_size=1)
            x = pool(x) # (B, C, 1, 1)
        elif self.pooling_type == "max": # same as nn.AdaptiveMaxPool2d(1)
            x = torch.max(x, dim=-1)[0] # max over W
            x = torch.max(x, dim=-1)[0] # max over H
            x = x.unsqueeze(-1).unsqueeze(-1) # (B, C, 1, 1)
        elif self.pooling_type == "avg": # same as nn.AdaptiveAvgPool2d(1)
            x = torch.mean(x, dim=(-1, -2), keepdim=True) # (B, C, 1, 1)
        else:
            raise ValueError(f"Unsupported pooling {self.pooling_type}")

        x = x.view(B, C)
        return self.classifier(x)


class MinimalCNNLocal(nn.Module):
    """Minimal CNN for Local Coarse-Graining test (Config 2).

    Conv(1->32, 3x3, pad=1) -> ReLU -> LocalPool(2x2) -> Conv(32->32, 3x3, pad=1)
    -> ReLU -> GlobalMaxPool -> FC(32->10)
    """
    def __init__(self, pooling: str = "geo") -> None:
        super().__init__()
        self.pooling_type = pooling
        self.local_pool = get_pooling(pooling, kernel_size=2)

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.local_pool(x) # Local coarse-graining (B, 16, 14, 14)
        x = self.conv2(x)      # (B, 32, 14, 14)

        # Fixed Global Pool: Max Pooling as "Standard Anchor"
        x = torch.max(x, dim=-1)[0]
        x = torch.max(x, dim=-1)[0] # (B, 32)

        return self.classifier(x)


class DeepCNNGlobal(nn.Module):
    """Deep CNN for Global Pooling test (Rigorous Config 1 - Extended).

    Conv1 -> ReLU -> MaxPool(2x2) -> Conv2 -> ReLU -> MaxPool(2x2)
    -> Conv3 -> ReLU -> MaxPool(2x2) -> Conv4 -> ReLU -> GlobalPool -> FC(32->10)
    """
    def __init__(self, pooling: str = "geo") -> None:
        super().__init__()
        self.pooling_type = pooling

        self.features = nn.Sequential(
            # Layer 1: 28x28 -> 14x14
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 2: 14x14 -> 7x7
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 3: 7x7 -> 3x3
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 4: 3x3 -> 3x3
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        B, C, H, W = x.size()

        if self.pooling_type == "geo":
            pool = GeometricPool2d(kernel_size=1)
            x = pool(x) # (B, C, 1, 1)
        elif self.pooling_type == "max":
            x = torch.max(x, dim=-1)[0]
            x = torch.max(x, dim=-1)[0]
            x = x.unsqueeze(-1).unsqueeze(-1) # (B, C, 1, 1)
        elif self.pooling_type == "avg":
            x = torch.mean(x, dim=(-1, -2), keepdim=True) # (B, C, 1, 1)
        else:
            raise ValueError(f"Unsupported pooling {self.pooling_type}")

        x = x.view(B, C)
        return self.classifier(x)

class DeepCNNLocal(nn.Module):
    """Deep CNN for Local Coarse-Graining test (Rigorous Config 2).

    Conv1 -> ReLU -> LocalPool(2x2) -> Conv2 -> ReLU -> LocalPool(2x2)
    -> Conv3 -> ReLU -> LocalPool(2x2) -> Conv4 -> ReLU -> GlobalMaxPool -> FC(32->10)
    """
    def __init__(self, pooling: str = "geo") -> None:
        super().__init__()
        self.pooling_type = pooling
        # Use the same pooling operator for all local stages
        self.local_pool = get_pooling(pooling, kernel_size=2)

        self.features = nn.Sequential(
            # Stage 1: 28x28 -> 14x14
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            self.local_pool,

            # Stage 2: 14x14 -> 7x7
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            self.local_pool,

            # Stage 3: 7x7 -> 3x3
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            self.local_pool,

            # Final refine: 3x3 -> 3x3
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)

        # Fixed Global Pool: Max Pooling as "Standard Anchor"
        x = torch.max(x, dim=-1)[0]
        x = torch.max(x, dim=-1)[0] # (B, 32)

        return self.classifier(x)


class DeepCNNAll(nn.Module):
    """Deep CNN for Synchronized Local & Global Pooling test (Config 3).

    Conv1 -> ReLU -> LocalPool(2x2) -> Conv2 -> ReLU -> LocalPool(2x2)
    -> Conv3 -> ReLU -> LocalPool(2x2) -> Conv4 -> ReLU -> GlobalPool -> FC(32->10)
    Where LocalPool and GlobalPool are the same primitive.
    """
    def __init__(self, pooling: str = "geo") -> None:
        super().__init__()
        self.pooling_type = pooling
        # Use the same pooling operator for all local stages
        self.local_pool = get_pooling(pooling, kernel_size=2)

        self.features = nn.Sequential(
            # Stage 1: 28x28 -> 14x14
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            self.local_pool,

            # Stage 2: 14x14 -> 7x7
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            self.local_pool,

            # Stage 3: 7x7 -> 3x3
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            self.local_pool,

            # Final refine: 3x3 -> 3x3
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        B, C, H, W = x.size()

        if self.pooling_type == "geo":
            pool = GeometricPool2d(kernel_size=1)
            x = pool(x) # (B, C, 1, 1)
        elif self.pooling_type == "max":
            x = torch.max(x, dim=-1)[0]
            x = torch.max(x, dim=-1)[0]
            x = x.unsqueeze(-1).unsqueeze(-1) # (B, C, 1, 1)
        elif self.pooling_type == "avg":
            x = torch.mean(x, dim=(-1, -2), keepdim=True) # (B, C, 1, 1)
        else:
            raise ValueError(f"Unsupported pooling {self.pooling_type}")

        x = x.view(B, C)
        return self.classifier(x)

class CIFARBase(nn.Module):
    """Shared feature extractor for CIFAR-10 models."""
    def __init__(self, local_pooling: str = "max"):
        super().__init__()
        self.local_pool = get_pooling(local_pooling, kernel_size=2)

        self.layer1 = nn.Sequential(nn.Conv2d(3, 48, 3, padding=1), nn.ReLU(True), self.local_pool)   # 32->16
        self.layer2 = nn.Sequential(nn.Conv2d(48, 64, 3, padding=1), nn.ReLU(True), self.local_pool)  # 16->8
        self.layer3 = nn.Sequential(nn.Conv2d(64, 96, 3, padding=1), nn.ReLU(True), self.local_pool)  # 8->4
        self.layer4 = nn.Sequential(nn.Conv2d(96, 128, 3, padding=1), nn.ReLU(True), self.local_pool) # 4->2
        self.layer5 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(True))               # 2->2

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

class CIFARCNNGlobal(nn.Module):
    """CIFAR-10: Local=Max, Global=Variable."""
    def __init__(self, pooling: str = "geo"):
        super().__init__()
        self.pooling_type = pooling
        self.features = CIFARBase(local_pooling="max")
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        x = self.features(x)
        B, C, H, W = x.size()
        if self.pooling_type == "geo":
            x = GeometricPool2d(kernel_size=1)(x).view(B, C)
        elif self.pooling_type == "max":
            x = torch.max(torch.max(x, dim=-1)[0], dim=-1)[0]
        elif self.pooling_type == "avg":
            x = torch.mean(x, dim=(-1, -2))
        return self.classifier(x)

class CIFARCNNLocal(nn.Module):
    """CIFAR-10: Local=Variable, Global=Max."""
    def __init__(self, pooling: str = "geo"):
        super().__init__()
        self.features = CIFARBase(local_pooling=pooling)
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.max(torch.max(x, dim=-1)[0], dim=-1)[0]
        return self.classifier(x)

class CIFARCNNAll(nn.Module):
    """CIFAR-10: Local=Variable, Global=Variable."""
    def __init__(self, pooling: str = "geo"):
        super().__init__()
        self.pooling_type = pooling
        self.features = CIFARBase(local_pooling=pooling)
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        x = self.features(x)
        B, C, H, W = x.size()
        if self.pooling_type == "geo":
            x = GeometricPool2d(kernel_size=1)(x).view(B, C)
        elif self.pooling_type == "max":
            x = torch.max(torch.max(x, dim=-1)[0], dim=-1)[0]
        elif self.pooling_type == "avg":
            x = torch.mean(x, dim=(-1, -2))
        return self.classifier(x)
