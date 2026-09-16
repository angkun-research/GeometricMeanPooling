import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset

from tasks.lipophilicity_preprocess import get_scaffold_split, get_morgan_fps
from models.lipophilicity_model import LipoMorganCNN, LipoMorganCNN2D
from utils import set_seed, pick_best_device

CSV_PATH = "data/lipophilicity.csv"
SEEDS = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
POOLINGS = ["geo", "max", "avg"]
COMBINATIONS = list(itertools.product(POOLINGS, POOLINGS))
MODEL_CLASSES = {"1D": LipoMorganCNN, "2D": LipoMorganCNN2D}
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 200
NUM_WORKERS = 4
ACTIVATION = "tanh"  # "relu", "sigmoid", "tanh", "softplus", "none"
RESULTS_FILE = f"./plots/lipo_morgan_cnn_expy_{ACTIVATION}.json"

class MorganDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def train(model, loader, optimizer, criterion, device):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y.unsqueeze(-1))
        if torch.isnan(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    rmse = float(np.sqrt(mean_squared_error(all_targets, all_preds)))
    r2 = float(r2_score(all_targets, all_preds))
    return rmse, r2

def run_single_experiment(model_name: str, local_type: str, global_type: str, seed: int) -> dict:
    set_seed(seed)
    device = pick_best_device() #torch.device("cpu")

    # scaffold split and Morgan FPs are deterministic; recomputed per worker (cheap vs. training)
    df = pd.read_csv(CSV_PATH)
    train_idx, test_idx = get_scaffold_split(df)

    X_train = get_morgan_fps(df, train_idx)
    y_train = df["label"].iloc[train_idx].values.astype(np.float32)
    X_test = get_morgan_fps(df, test_idx)
    y_test = df["label"].iloc[test_idx].values.astype(np.float32)
    y_train = np.exp(df["label"].iloc[train_idx].to_numpy(dtype=np.float32))
    y_test = np.exp(df["label"].iloc[test_idx].to_numpy(dtype=np.float32))
    # scale all y values to [0, 1] range
    # ymax = 91
    # y_train /= ymax
    # y_test /= ymax

    train_loader = DataLoader(MorganDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(MorganDataset(X_test, y_test), batch_size=BATCH_SIZE)

    model_class = MODEL_CLASSES[model_name]
    model = model_class(pooling_type_local=local_type, 
        pooling_type_global=global_type,activation=ACTIVATION).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    for _ in range(EPOCHS):
        train(model, train_loader, optimizer, criterion, device)

    rmse, r2 = evaluate(model, test_loader, device)

    return {
        "model": model_name,
        "local": local_type,
        "global": global_type,
        "seed": seed,
        "rmse": rmse,
        "r2": r2,
    }


def task_key(model_name: str, local_type: str, global_type: str, seed: int) -> str:
    return f"{model_name}_local{local_type}_global{global_type}_seed{seed}"


def main():
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = {}

    tasks = []
    for model_name in MODEL_CLASSES:
        for local_type, global_type in COMBINATIONS:
            for seed in SEEDS:
                key = task_key(model_name, local_type, global_type, seed)
                if key not in results:
                    tasks.append((key, model_name, local_type, global_type, seed))

    print(f"Pending tasks: {len(tasks)}")
    print(f"Using {NUM_WORKERS} CPU workers")

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_map = {
            executor.submit(run_single_experiment, model_name, local_type, global_type, seed): (
                key, model_name, local_type, global_type, seed
            )
            for key, model_name, local_type, global_type, seed in tasks
        }

        for future in as_completed(future_map):
            key, model_name, local_type, global_type, seed = future_map[future]
            try:
                result = future.result()
                results[key] = result

                with open(RESULTS_FILE, "w") as f:
                    json.dump(results, f, indent=2)

                print(
                    f"done  model={model_name:<3} local={local_type:<4} global={global_type:<4} "
                    f"seed={seed:<3} rmse={result['rmse']:.4f} r2={result['r2']:.4f}"
                )
            except Exception as exc:
                print(f"failed model={model_name} local={local_type} global={global_type} seed={seed}: {exc}")

    print(f"Saved results to {RESULTS_FILE}")


def peak_at_data():
    df = pd.read_csv(CSV_PATH)
    train_idx, test_idx = get_scaffold_split(df)

    X_train = get_morgan_fps(df, train_idx)
    y_train = df["label"].iloc[train_idx].values.astype(np.float32)
    X_test = get_morgan_fps(df, test_idx)
    y_test = df["label"].iloc[test_idx].values.astype(np.float32)
    y_train = np.exp(df["label"].iloc[train_idx].to_numpy(dtype=np.float32))
    y_test = np.exp(df["label"].iloc[test_idx].to_numpy(dtype=np.float32))
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"max y train: {np.max(y_train)}, min y_train: {np.min(y_train)}")
    print(f"max y test: {np.max(y_test)}, min y test: {np.min(y_test)}")

if __name__ == "__main__":
    main()
    # to run without interruption: 
    # nohup python collect_lipo_morgan.py > plots/lipo_morgan_cnn.log 2>&1 &