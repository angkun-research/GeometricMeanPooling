import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.cnn import DeepCNNGlobal
from tasks.mnist import make_dataloaders, train_one_epoch, evaluate
from utils import pick_best_device, set_seed

def run_config1(pooling: str, epochs=100, lr=0.001, batch_size=128, seed=42):
    print(f"\n--- Running Config 1 with Pooling: {pooling} ---")
    #torch.manual_seed(seed)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = pick_best_device()
    set_seed(seed)
    print(f"Using device: {device}")

    model = DeepCNNGlobal(pooling=pooling).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader, val_loader = make_dataloaders(batch_size)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Loss: {val_loss:.4f}")

    print(f"Final Best Val Acc for {pooling}: {best_val_acc:.4f}")
    return best_val_acc

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python run_mnist_config1.py <pooling>")
        print("Poolings: geo, max, avg")
        sys.exit(1)

    p = sys.argv[1]
    if p not in ['geo', 'max', 'avg']:
        print(f"Invalid pooling {p}. Use geo, max, or avg.")
        sys.exit(1)

    acc = run_config1(p, epochs=100)
    print("\n" + "="*30)
    print(f"Config 1 Test ({p}): {acc:.4f}")
    print("="*30)
