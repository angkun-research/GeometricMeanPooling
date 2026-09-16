import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader

from models.lipophilicity_model import LipoCNN
from tasks.lipophilicity_preprocess import (
    SMILESDataset,
    collate_smiles_batch,
    preprocess_data,
)
from utils import pick_best_device, set_seed


RESULTS_FILE = "./plots/lipo_embedding_cnn_exp_tanhx3.json"
CSV_PATH = "data/lipophilicity.csv"

SEEDS = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
CONFIGS = ["Global", "Local", "Combined"]
POOLINGS = ["geo", "max", "avg"]

BATCH_SIZE = 64
EMBED_DIM = 48
LR = 0.001
EPOCHS = 200

# Two workers suit the two available GPUs better than launching many
# experiments that all select the same currently-idle device.
NUM_WORKERS = 4

# Three k=2, stride=2 local pooling layers require at least 2^3 valid tokens.
MIN_CNN_LENGTH = 8


def config_to_poolings(cfg_name: str, pooling: str):
    if cfg_name == "Global":
        return "max", pooling
    if cfg_name == "Local":
        return pooling, "max"
    if cfg_name == "Combined":
        return pooling, pooling
    raise ValueError(f"Unknown config: {cfg_name}")


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


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_predictions = []
    all_targets = []

    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        predictions = model(input_ids, attention_mask)

        if not torch.isfinite(predictions).all():
            raise RuntimeError("Non-finite evaluation predictions")

        all_predictions.append(predictions.squeeze(-1).cpu().numpy())
        all_targets.append(labels.cpu().numpy())

    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)

    rmse = float(np.sqrt(mean_squared_error(targets, predictions)))
    r2 = float(r2_score(targets, predictions))
    return rmse, r2

def run_single_experiment(
    cfg_name: str,
    pooling: str,
    seed: int,
) -> dict:
    set_seed(seed)
    device = pick_best_device()

    local_pooling, global_pooling = config_to_poolings(cfg_name, pooling)

    train_raw, test_raw, vocab = preprocess_data(CSV_PATH)

    # # Use the same eligible molecule subset in every pooling condition.
    # train_raw = [
    #     (sequence, label)
    #     for sequence, label in train_raw
    #     if len(sequence) >= MIN_CNN_LENGTH
    # ]
    # test_raw = [
    #     (sequence, label)
    #     for sequence, label in test_raw
    #     if len(sequence) >= MIN_CNN_LENGTH
    # ]
    # Recover the multiplicative relation: train on D = exp(logD), not logD directly.
    train_raw = [
        (sequence, np.exp(label))
        for sequence, label in train_raw
        if len(sequence) >= MIN_CNN_LENGTH
    ]
    test_raw = [
        (sequence, np.exp(label))
        for sequence, label in test_raw
        if len(sequence) >= MIN_CNN_LENGTH
    ]

    if not train_raw or not test_raw:
        raise RuntimeError(
            f"No samples remain after requiring {MIN_CNN_LENGTH} tokens."
        )

    train_dataset = SMILESDataset(train_raw)
    test_dataset = SMILESDataset(test_raw)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_smiles_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_smiles_batch,
    )

    model = LipoCNN(
        vocab_size=len(vocab),
        pooling_type_local=local_pooling,
        pooling_type_global=global_pooling,
        embed_dim=EMBED_DIM,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    for _ in range(EPOCHS):
        train(model, train_loader, optimizer, criterion, device)

    rmse, r2 = evaluate(model, test_loader, device)

    return {
        "config": cfg_name,
        "pooling": pooling,
        "seed": seed,
        "rmse": rmse,
        "r2": r2,
        # Optional metadata, safe for plotting code that only reads config/pooling/rmse.
        "local_pooling": local_pooling,
        "global_pooling": global_pooling,
        "train_samples": len(train_raw),
        "test_samples": len(test_raw),
    }


def task_key(cfg_name: str, pooling: str, seed: int) -> str:
    return f"{cfg_name}_{pooling}_seed{seed}"


def main():
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as results_file:
            try:
                results = json.load(results_file)
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
    print(f"Using {NUM_WORKERS} workers")

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_map = {
            executor.submit(
                run_single_experiment,
                cfg_name,
                pooling,
                seed,
            ): (key, cfg_name, pooling, seed)
            for key, cfg_name, pooling, seed in tasks
        }

        for future in as_completed(future_map):
            key, cfg_name, pooling, seed = future_map[future]

            try:
                result = future.result()
                results[key] = result

                with open(RESULTS_FILE, "w") as results_file:
                    json.dump(results, results_file, indent=2)

                print(
                    f"done  cfg={cfg_name:<9} "
                    f"pool={pooling:<4} "
                    f"seed={seed:<3} "
                    f"rmse={result['rmse']:.4f} "
                    f"r2={result['r2']:.4f}"
                )
            except Exception as exception:
                print(
                    f"failed cfg={cfg_name} "
                    f"pool={pooling} "
                    f"seed={seed}: {exception}"
                )

    print(f"Saved results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
    # to run without interruption:
    # nohup python collect_lipo_cnn.py > plots/collect_lipo_cnn.log 2>&1 &