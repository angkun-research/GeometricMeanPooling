"""MNIST data loading, training loop, and evaluation."""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def make_dataloaders(batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    """Create MNIST train/val dataloaders."""
    train_transform = transforms.Compose([
        transforms.RandomCrop(28, padding=2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST("./data", train=True, download=True, transform=train_transform)
    test = datasets.MNIST("./data", train=False, download=True, transform=test_transform)
    return DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0), \
           DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0)


def train_one_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = F.cross_entropy(out, y)
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def run_experiment(
    pooling: str,
    epochs: int = 20,
    lr: float = 0.01,
    batch_size: int = 128,
    seed: int = 42,
) -> dict:
    """Run a full MNIST training experiment.

    Returns a dict with training curves and final metrics.
    """
    torch.manual_seed(seed)
    device = torch.device("cpu")

    from models.cnn import DeepCNNGlobal
    model = DeepCNNGlobal(pooling=pooling).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader, val_loader = make_dataloaders(batch_size)

    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    for epoch in range(1, epochs + 1):
        tl, ta = train_one_epoch(model, train_loader, optimizer, device)
        vl, va = evaluate(model, val_loader, device)
        train_losses.append(tl)
        train_accs.append(ta)
        val_losses.append(vl)
        val_accs.append(va)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Pooling={pooling:>6} | Epoch {epoch:>2d} | train_loss={tl:.4f} train_acc={ta:.4f} | val_loss={vl:.4f} val_acc={va:.4f}")

    # Final evaluation
    _, final_val_acc = evaluate(model, val_loader, device)

    return {
        "pooling": pooling,
        "epochs": epochs,
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_losses": val_losses,
        "val_accs": val_accs,
        "final_val_acc": final_val_acc,
    }
