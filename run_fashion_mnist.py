import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.cnn import DeepCNNAll, DeepCNNLocal, DeepCNNGlobal
from tasks.fashion_mnist import make_dataloaders, train_one_epoch, evaluate
from utils import pick_best_device, set_seed

def run_experiment(model_class, pooling: str, epochs=100, lr=0.001, batch_size=128, seed=42):
    model_name = model_class.__name__
    print(f"\n--- Running Fashion-MNIST | Model: {model_name} | Pooling: {pooling} ---")
    #torch.manual_seed(seed)
    set_seed(seed)
    device = pick_best_device()
    print(f"Using device: {device}")

    model = model_class(pooling=pooling).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader, val_loader = make_dataloaders(batch_size)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Loss: {val_loss:.4f}")

    print(f"Final Best Val Acc for {model_name} ({pooling}): {best_val_acc:.4f}")
    return best_val_acc

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python run_fashion_mnist.py <model> <pooling>")
        print("Models: All, Local, Global")
        print("Poolings: geo, max, avg")
        sys.exit(1)

    model_map = {
        "All": DeepCNNAll,
        "Local": DeepCNNLocal,
        "Global": DeepCNNGlobal
    }
    pooling_options = ['geo', 'max', 'avg']

    m_arg = sys.argv[1]
    p_arg = sys.argv[2]

    if m_arg not in model_map:
        print(f"Invalid model {m_arg}. Use All, Local, or Global.")
        sys.exit(1)
    if p_arg not in pooling_options:
        print(f"Invalid pooling {p_arg}. Use geo, max, or avg.")
        sys.exit(1)

    model_cls = model_map[m_arg]
    acc = run_experiment(model_cls, p_arg, epochs=100, lr=0.001) # Using updated research hyperparameters
    print("\n" + "="*30)
    print(f"Fashion-MNIST | {m_arg} | {p_arg}: {acc:.4f}")
    print("="*30)

