import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import itertools

from tasks.lipophilicity_preprocess import preprocess_data, get_scaffold_split, create_vocab, smiles_to_sequence, get_morgan_fps
from models.lipophilicity_model import LipoMorganCNN, LipoMorganCNN2D, SMILESDataset

def train(model, loader, optimizer, criterion, device, epochs=200):
    model.train()
    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y.unsqueeze(-1))
            if torch.isnan(loss): continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    return np.sqrt(mean_squared_error(all_targets, all_preds))

if __name__ == "__main__":
    csv_path = "data/lipophilicity.csv"
    batch_size = 64
    lr = 0.001 # Use the stable learning rate found in LipoCNN sweep
    epochs = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocess
    df = pd.read_csv(csv_path)
    train_idx, test_idx = get_scaffold_split(df)

    # Prepare Morgan Data
    X_train_morgan = get_morgan_fps(df, train_idx)
    y_train_morgan = df['label'].iloc[train_idx].values.astype(np.float32)
    X_test_morgan = get_morgan_fps(df, test_idx)
    y_test_morgan = df['label'].iloc[test_idx].values.astype(np.float32)

    # Create a simple dataset for Morgan FPs
    class MorganDataset(torch.utils.data.Dataset):
        def __init__(self, x, y):
            self.x = torch.tensor(x, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
        def __len__(self): return len(self.x)
        def __getitem__(self, idx): return self.x[idx], self.y[idx]

    train_loader = DataLoader(MorganDataset(X_train_morgan, y_train_morgan), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(MorganDataset(X_test_morgan, y_test_morgan), batch_size=batch_size)

    # Sweep Configuration
    poolings = ["geo", "max", "avg"]
    combinations = list(itertools.product(poolings, poolings))

    model_configs = [
        {"name": "1D CNN", "class": LipoMorganCNN},
        {"name": "2D CNN", "class": LipoMorganCNN2D}
    ]

    all_results = {}
    print(f"Starting LipoMorganCNN Pool Sweep on {device}...")
    print(f"Grid: 3 Local x 3 Global pools. Total variants per model: {len(combinations)}\n")

    for cfg in model_configs:
        model_name = cfg["name"]
        model_class = cfg["class"]
        results = []

        print(f"\n--- Testing Model: {model_name} ---")
        for local_type, global_type in combinations:
            print(f"Training [Local: {local_type}, Global: {global_type}]...", end=" ")

            model = model_class(pooling_type_local=local_type, pooling_type_global=global_type).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            train(model, train_loader, optimizer, criterion, device, epochs=epochs)
            rmse = evaluate(model, test_loader, device)

            results.append({
                "local": local_type,
                "global": global_type,
                "rmse": rmse
            })
            print(f"RMSE: {rmse:.4f}")

        all_results[model_name] = results

    # Final Table Output
    for model_name, results in all_results.items():
        print("\n" + "="*50)
        print(f"Results for {model_name}")
        print("="*50)
        print(f"{'Local':<10} | {'Global':<10} | {'RMSE':<10}")
        print("-" * 50)
        for res in results:
            print(f"{res['local']:<10} | {res['global']:<10} | {res['rmse']:<10.4f}")
        print("-" * 50)

        best = min(results, key=lambda x: x["rmse"])
        print(f"Best for {model_name}: Local={best['local']}, Global={best['global']} with RMSE={best['rmse']:.4f}")
