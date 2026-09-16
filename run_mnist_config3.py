import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.cnn import DeepCNNAll
from tasks.mnist import make_dataloaders, train_one_epoch, evaluate
from utils import pick_best_device, set_seed

def run_config3(pooling: str, epochs=100, lr=0.001, batch_size=128, seed=42):
    print(f"\n--- Running Config 3 (Synchronized Local & Global) with Pooling: {pooling} ---")
    #torch.manual_seed(seed)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = pick_best_device()
    set_seed(seed)
    print(f"Using device: {device}")

    model = DeepCNNAll(pooling=pooling).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader, val_loader = make_dataloaders(batch_size)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Loss: {val_loss:.4f}")

    print(f"Final Best Val Acc for {pooling}: {best_val_acc:.4f}")
    return best_val_acc

if __name__ == "__main__":
    poolings = ['geo', 'max', 'avg']
    results = {}

    for p in poolings:
        acc = run_config3(p)
        results[p] = acc

    print("\n" + "="*30)
    print("Config 3 Results (Accuracy):")
    for p, acc in results.items():
        print(f"{p:>6}: {acc:.4f}")
    print("="*30)
