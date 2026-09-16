import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import itertools

from tasks.lipophilicity_preprocess import preprocess_data, get_scaffold_split, create_vocab, smiles_to_sequence
from models.lipophilicity_model import LipoCNN, SMILESDataset

def train(model, loader, optimizer, criterion, device, epochs=200):
    model.train()
    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y.unsqueeze(-1))
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch}. Skipping batch.")
                continue
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
    # Settings
    csv_path = "data/lipophilicity.csv"
    batch_size = 64
    lr = 0.001  # Updated from 0.01 to test convergence speed/stability
    epochs = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocess
    df = pd.read_csv(csv_path)
    train_idx, test_idx = get_scaffold_split(df)
    vocab = create_vocab(df)

    def process_row(row):
        return smiles_to_sequence(row['SMILES'], vocab), row['label']

    train_raw = [process_row(df.iloc[i]) for i in train_idx]
    test_raw = [process_row(df.iloc[i]) for i in test_idx]

    train_ds = SMILESDataset(train_raw, vocab)
    test_ds = SMILESDataset(test_raw, vocab)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Sweep Configuration
    poolings = ["geo", "max", "avg"]
    combinations = list(itertools.product(poolings, poolings)) # (local, global)

    results = []

    print(f"Starting LipoCNN Pool Sweep on {device}...")
    print(f"Grid: 3 Local x 3 Global pools. Total variants: {len(combinations)}\n")

    for local_type, global_type in combinations:
        print(f"Training [Local: {local_type}, Global: {global_type}]...", end=" ")

        model = LipoCNN(vocab_size=len(vocab), pooling_type_local=local_type, pooling_type_global=global_type).to(device)
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

    # Final Table Output
    print("\n" + "="*50)
    print(f"{'Local':<10} | {'Global':<10} | {'RMSE':<10}")
    print("-" * 50)
    for res in results:
        print(f"{res['local']:<10} | {res['global']:<10} | {res['rmse']:<10.4f}")
    print("="*50)

    # Find best combination
    best = min(results, key=lambda x: x["rmse"])
    print(f"\nBest Configuration: Local={best['local']}, Global={best['global']} with RMSE={best['rmse']:.4f}")
