import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

from tasks.lipophilicity_preprocess import preprocess_data, get_morgan_fps, get_scaffold_split, create_vocab, smiles_to_sequence
from models.lipophilicity_model import LipoMorganFC, LipoEmbeddingFC

class SMILESDataset(torch.utils.data.Dataset):
    """Dataset for loading and padding SMILES sequences."""
    def __init__(self, data, vocab, max_len=128):
        self.samples = []
        for seq, label in data:
            if len(seq) > max_len:
                seq = seq[:max_len]
            else:
                seq = seq + [0] * (max_len - len(seq))
            self.samples.append((torch.tensor(seq), torch.tensor(label, dtype=torch.float32)))
        self.vocab_size = len(vocab)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def train_nn(model, loader, optimizer, criterion, device, epochs=200):
    model.train()
    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y.unsqueeze(-1))
            loss.backward()
            optimizer.step()

def evaluate_nn(model, loader, device):
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
    return np.sqrt(mean_squared_error(all_targets, all_preds)), r2_score(all_targets, all_preds)

if __name__ == "__main__":
    # Settings
    csv_path = "data/lipophilicity.csv"
    batch_size = 64
    lr = 0.001
    epochs = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocess
    df = pd.read_csv(csv_path)
    train_idx, test_idx = get_scaffold_split(df)
    vocab = create_vocab(df)

    # Prepare Sequence Data (for Embedding + FC)
    def process_row(row):
        return smiles_to_sequence(row['SMILES'], vocab), row['label']

    train_raw = [process_row(df.iloc[i]) for i in train_idx]
    test_raw = [process_row(df.iloc[i]) for i in test_idx]

    train_ds = SMILESDataset(train_raw, vocab)
    test_ds = SMILESDataset(test_raw, vocab)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Prepare Morgan Data (for XGBoost and Morgan FC)
    X_train_morgan = get_morgan_fps(df, train_idx)
    y_train_morgan = df['label'].iloc[train_idx].values.astype(np.float32)
    X_test_morgan = get_morgan_fps(df, test_idx)
    y_test_morgan = df['label'].iloc[test_idx].values.astype(np.float32)

    results = {}

    # --- Baseline 1: Morgan FP + XGBoost ---
    print("Training Morgan FP + XGBoost...", end=" ")
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train_morgan, y_train_morgan)
    preds_xgb = xgb_model.predict(X_test_morgan)
    rmse_xgb = np.sqrt(mean_squared_error(y_test_morgan, preds_xgb))
    r2_xgb = r2_score(y_test_morgan, preds_xgb)
    results["Morgan + XGBoost"] = (rmse_xgb, r2_xgb)
    print(f"RMSE: {rmse_xgb:.4f}, R^2: {r2_xgb:.4f}")

    # --- Baseline 2: Morgan FP + FC ---
    print("Training Morgan FP + FC...", end=" ")
    model_morgan_fc = LipoMorganFC().to(device)
    optimizer_mfc = torch.optim.Adam(model_morgan_fc.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Convert Morgan data to tensors for NN
    train_morgan_tensor = torch.tensor(X_train_morgan).to(device)
    train_y_tensor = torch.tensor(y_train_morgan).to(device)
    test_morgan_tensor = torch.tensor(X_test_morgan).to(device)
    test_y_tensor = torch.tensor(y_test_morgan).to(device)

    # Simple loop for Morgan FC since it's small
    model_morgan_fc.train()
    for epoch in range(epochs):
        perm = torch.randperm(train_morgan_tensor.size(0))
        for i in range(0, train_morgan_tensor.size(0), batch_size):
            idx = perm[i:i+batch_size]
            x_b, y_b = train_morgan_tensor[idx], train_y_tensor[idx]
            optimizer_mfc.zero_grad()
            p = model_morgan_fc(x_b)
            loss = criterion(p, y_b.unsqueeze(-1))
            loss.backward()
            optimizer_mfc.step()

    model_morgan_fc.eval()
    with torch.no_grad():
        preds_mfc = model_morgan_fc(test_morgan_tensor).cpu().numpy().flatten()
    rmse_mfc = np.sqrt(mean_squared_error(y_test_morgan, preds_mfc))
    r2_mfc = r2_score(y_test_morgan, preds_mfc)
    results["Morgan + FC"] = (rmse_mfc, r2_mfc)
    print(f"RMSE: {rmse_mfc:.4f}, R^2: {r2_mfc:.4f}")

    # --- Baseline 3: Embedding + FC ---
    print("Training Embedding + FC...", end=" ")
    model_emb_fc = LipoEmbeddingFC(vocab_size=len(vocab)).to(device)
    optimizer_efc = torch.optim.Adam(model_emb_fc.parameters(), lr=lr)

    train_nn(model_emb_fc, train_loader, optimizer_efc, criterion, device, epochs=epochs)
    rmse_efc = evaluate_nn(model_emb_fc, test_loader, device)
    rmse_efc, r2_efc = rmse_efc
    results["Embedding + FC"] = (rmse_efc, r2_efc)
    print(f"RMSE: {rmse_efc:.4f}, R^2: {r2_efc:.4f}")

    # Final Summary
    print("\n" + "="*40)
    print(f"{'Baseline':<25} | {'RMSE':<10} | {'R^2':<10}")
    print("-" * 50)
    for name, (rmse, r2) in results.items():
        print(f"{name:<25} | {rmse:<10.4f} | {r2:<10.4f}")
    print("="*40)
