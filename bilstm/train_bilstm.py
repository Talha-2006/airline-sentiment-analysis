"""Train the BiLSTM airline sentiment classifier."""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from bilstm_model import BiLSTMClassifier
from build_bilstm_data import make_loaders


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "bilstm_results_unweighted"
SEED = 42
LEARNING_RATE = 1e-3


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    for token_ids, lengths, labels, _ in loader:
        token_ids = token_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(token_ids, lengths)
        loss = loss_fn(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(logits.argmax(dim=1).detach().cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    for token_ids, lengths, labels, _ in loader:
        token_ids = token_ids.to(device)
        labels = labels.to(device)
        logits = model(token_ids, lengths)
        loss = loss_fn(logits, labels)

        total_loss += loss.item() * len(labels)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1


def save_curves(history, path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping the training-curve image.")
        return

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss")
    axes[0].legend()

    axes[1].plot(epochs, [row["train_macro_f1"] for row in history], label="Train")
    axes[1].plot(epochs, [row["val_macro_f1"] for row in history], label="Validation")
    axes[1].set(xlabel="Epoch", ylabel="Macro-F1", title="Macro-F1", ylim=(0, 1))
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(epochs=15, patience=3):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS_DIR.mkdir(exist_ok=True)

    train_loader, val_loader, _, vocab = make_loaders()
    model_config = {"vocab_size": len(vocab), "embedding_dim": 128,
                    "hidden_dim": 128, "dropout": 0.3}
    model = BiLSTMClassifier(**model_config).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Device:", device)
    print("Train/validation:", len(train_loader.dataset), len(val_loader.dataset))
    print("Parameters:", f"{sum(p.numel() for p in model.parameters()):,}")

    history = []
    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint_path = RESULTS_DIR / "best_bilstm.pt"

    for epoch in range(1, epochs + 1):
        train_loss, train_f1 = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device
        )
        val_loss, val_f1 = evaluate(model, val_loader, loss_fn, device)

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "train_macro_f1": train_f1, "val_loss": val_loss,
            "val_macro_f1": val_f1,
        })

        improved = val_f1 > best_f1 + 1e-6
        if improved:
            best_f1 = val_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": model_config,
                "vocab": vocab,
                "best_epoch": best_epoch,
                "val_macro_f1": best_f1,
                "seed": SEED,
                "label_map": {"negative": 0, "neutral": 1, "positive": 2},
            }, checkpoint_path)
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:2d}/{epochs} | "
            f"train loss {train_loss:.4f}, F1 {train_f1:.4f} | "
            f"val loss {val_loss:.4f}, F1 {val_f1:.4f}"
            + ("  [best]" if improved else ""),
            flush=True,
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping: no validation F1 improvement for {patience} epochs.")
            break

    with (RESULTS_DIR / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    save_curves(history, RESULTS_DIR / "training_curves.png")
    with (RESULTS_DIR / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump({
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_f1,
            "epochs_run": len(history),
            "max_epochs": epochs,
            "patience": patience,
            "learning_rate": LEARNING_RATE,
            "batch_size": train_loader.batch_size,
            "seed": SEED,
            "model_config": model_config,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "loss": "unweighted_cross_entropy",
            "optimizer": "Adam",
        }, f, indent=2)

    print(f"\nBest epoch: {best_epoch}, validation macro-F1: {best_f1:.4f}")
    print("Saved results in:", RESULTS_DIR)
    print("Test set has not been evaluated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the airline sentiment BiLSTM")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1:
        parser.error("--epochs and --patience must be positive")
    main(epochs=args.epochs, patience=args.patience)
