import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

from models.lipophilicity_model import LipophilicityLocal,LipoCNN
from tasks.lipophilicity_preprocess import (
    SMILESDataset,
    collate_smiles_batch,
    preprocess_data,
)


CSV_PATH = "data/lipophilicity.csv"
SEED = 42
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
EMBED_DIM = 48


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_examples = 0

    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        predictions = model(input_ids, attention_mask)
        loss = criterion(predictions, labels.unsqueeze(-1))

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss: {loss.item()}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / total_examples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_predictions = []
    all_targets = []

    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        predictions = model(input_ids, attention_mask)
        loss = criterion(predictions, labels.unsqueeze(-1))

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite evaluation loss: {loss.item()}")

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

        all_predictions.append(predictions.squeeze(-1).cpu())
        all_targets.append(labels.cpu())

    predictions = torch.cat(all_predictions).numpy()
    targets = torch.cat(all_targets).numpy()

    rmse = float(np.sqrt(mean_squared_error(targets, predictions)))
    mean_loss = total_loss / total_examples

    return mean_loss, rmse


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_raw, test_raw, vocab = preprocess_data(CSV_PATH)

    train_lengths = [len(sequence) for sequence, _ in train_raw]
    test_lengths = [len(sequence) for sequence, _ in test_raw]

    print(
        f"Vocabulary size: {len(vocab)} | "
        f"train={len(train_raw)} | test={len(test_raw)}"
    )
    print(
        f"Token lengths: "
        f"train=[{min(train_lengths)}, {max(train_lengths)}], "
        f"test=[{min(test_lengths)}, {max(test_lengths)}]"
    )

    # Local pooling with k=3 needs at least three real SMILES tokens.
    if min(train_lengths + test_lengths) < 3:
        raise ValueError(
            "Found a sequence shorter than the local pooling kernel size (3)."
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

    # model = LipophilicityLocal(
    #     vocab_size=len(vocab),
    #     pooling_type="geo",
    #     embed_dim=EMBED_DIM,
    #     kernel_size=2,
    #     stride=2,
    # ).to(device)
    model = LipoCNN(
        vocab_size=len(vocab),
        pooling_type_local="geo",
        pooling_type_global="max",
        embed_dim=EMBED_DIM,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print(
        f"\nTraining LipophilicityLocal "
        f"(local=max, global=max) for {EPOCHS} epochs..."
    )

    for epoch in range(1, EPOCHS + 1):
        train_mse = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            test_mse, test_rmse = evaluate(
                model,
                test_loader,
                criterion,
                device,
            )
            print(
                f"Epoch {epoch:3d}/{EPOCHS} | "
                f"train_mse={train_mse:.5f} | "
                f"test_mse={test_mse:.5f} | "
                f"test_rmse={test_rmse:.4f}"
            )

    _, final_rmse = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal scaffold-split test RMSE: {final_rmse:.4f}")


if __name__ == "__main__":
    main()