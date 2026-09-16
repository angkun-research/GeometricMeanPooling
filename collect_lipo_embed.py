import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error

from tasks.lipophilicity_preprocess import (
    SMILESDataset,
    collate_smiles_batch,
    preprocess_data,
)
from models.lipophilicity_model import (
    LipophilicityGlobal,
    LipophilicityLocal,
    LipophilicityCombined,
)
from utils import set_seed, pick_best_device

RESULTS_FILE = "./plots/lipo_embedding_masked.json"
CSV_PATH = "data/lipophilicity.csv"
SEEDS = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
CONFIGS = ["Global", "Local", "Combined"]
POOLINGS = ["geo", "max", "avg"]
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 200
NUM_WORKERS = 4

MODEL_CLASSES = {
    "Global": LipophilicityGlobal,
    "Local": LipophilicityLocal,
    "Combined": LipophilicityCombined,
}

def train(model, loader, optimizer, criterion, device):
    model.train()

    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        predictions = model(input_ids, attention_mask)
        loss = criterion(predictions, labels.unsqueeze(-1))

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            predictions = model(input_ids, attention_mask)

            if not torch.isfinite(predictions).all():
                raise RuntimeError("Non-finite evaluation predictions")

            all_preds.append(predictions.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()

    return float(np.sqrt(mean_squared_error(all_targets, all_preds)))


def run_single_experiment(cfg_name: str, pooling: str, seed: int) -> dict:
    set_seed(seed)
    device = pick_best_device()

    # scaffold split is fixed (random_state=42); seed only affects init/shuffle
    train_raw, test_raw, vocab = preprocess_data(CSV_PATH)

    train_ds = SMILESDataset(train_raw)
    test_ds = SMILESDataset(test_raw)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_smiles_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_smiles_batch,
    )

    model_class = MODEL_CLASSES[cfg_name]
    model = model_class(vocab_size=len(vocab), pooling_type=pooling).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    for _ in range(EPOCHS):
        train(model, train_loader, optimizer, criterion, device)

    rmse = evaluate(model, test_loader, device)

    return {
        "config": cfg_name,
        "pooling": pooling,
        "seed": seed,
        "rmse": rmse,
    }


def task_key(cfg_name: str, pooling: str, seed: int) -> str:
    return f"{cfg_name}_{pooling}_seed{seed}"


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
    for cfg_name in CONFIGS:
        for pooling in POOLINGS:
            for seed in SEEDS:
                key = task_key(cfg_name, pooling, seed)
                if key not in results:
                    tasks.append((key, cfg_name, pooling, seed))

    print(f"Pending tasks: {len(tasks)}")
    print(f"Using {NUM_WORKERS} CPU workers")

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_map = {
            executor.submit(run_single_experiment, cfg_name, pooling, seed): (key, cfg_name, pooling, seed)
            for key, cfg_name, pooling, seed in tasks
        }

        for future in as_completed(future_map):
            key, cfg_name, pooling, seed = future_map[future]
            try:
                result = future.result()
                results[key] = result

                with open(RESULTS_FILE, "w") as f:
                    json.dump(results, f, indent=2)

                print(
                    f"done  cfg={cfg_name:<9} pool={pooling:<4} seed={seed:<3} "
                    f"rmse={result['rmse']:.4f}"
                )
            except Exception as exc:
                print(f"failed cfg={cfg_name} pool={pooling} seed={seed}: {exc}")

    print(f"Saved results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
    # to run without interruption: 
    # nohup python collect_lipo_embed.py > plots/lipo_embedding.log 2>&1 &