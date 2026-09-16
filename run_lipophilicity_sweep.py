import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tasks.lipophilicity_preprocess import preprocess_data
from models.lipophilicity_model import LipophilicityGlobal, LipophilicityLocal, LipophilicityCombined, SMILESDataset
import numpy as np
from sklearn.metrics import mean_squared_error

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y.unsqueeze(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y.unsqueeze(-1))
            total_loss += loss.item() * x.size(0)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    return total_loss / len(loader.dataset), rmse

if __name__ == "__main__":
    # Settings
    csv_path = "data/lipophilicity.csv"
    batch_size = 64
    lr = 0.001
    epochs = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rf_baseline = 0.8718
    
    # Preprocess
    train_raw, test_raw, vocab = preprocess_data(csv_path)
    train_ds = SMILESDataset(train_raw, vocab)
    test_ds = SMILESDataset(test_raw, vocab)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    configs = {
        "Global": LipophilicityGlobal,
        "Local": LipophilicityLocal,
        "Combined": LipophilicityCombined
    }
    poolings = ["geo", "max", "avg"]
    results = {}

    print(f"Starting Sweep on {device}...")
    print(f"RF Baseline RMSE: {rf_baseline:.4f}\n")

    for cfg_name, model_class in configs.items():
        results[cfg_name] = {}
        for p_type in poolings:
            print(f"Training {cfg_name} with {p_type} pooling...", end=" ")
            model = model_class(vocab_size=len(vocab), pooling_type=p_type).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()
            
            for epoch in range(1, epochs + 1):
                train(model, train_loader, optimizer, criterion, device)
            
            _, rmse = evaluate(model, test_loader, criterion, device)
            results[cfg_name][p_type] = rmse
            print(f"RMSE: {rmse:.4f}")

    # Final Table
    print("\n" + "="*60)
    print(f"{'Config':<15} | {'Geo (Ours)':<12} | {'Max':<12} | {'Avg':<12}")
    print("-" * 60)
    for cfg_name in configs.keys():
        res = results[cfg_name]
        print(f"{cfg_name:<15} | {res['geo']:<12.4f} | {res['max']:<12.4f} | {res['avg']:<12.4f}")
    print("-" * 60)
    print(f"RF Baseline: {rf_baseline:.4f}")
    print("="*60)
