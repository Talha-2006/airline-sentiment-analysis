"""Train the same BiLSTM with class-weighted cross-entropy loss."""

import argparse
import csv
import json
from pathlib import Path

import torch
from torch import nn

from bilstm_model import BiLSTMClassifier
from build_bilstm_data import make_loaders
from train_bilstm import evaluate, save_curves, set_seed, train_one_epoch


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "bilstm_results_weighted"
SEED = 42
LEARNING_RATE = 0.001


def main(epochs=15, patience=3):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _, vocab = make_loaders()

    # compute class weights from train split only
    counts = torch.bincount(torch.tensor(train_loader.dataset.labels), minlength=3)
    weights = counts.sum() / (3 * counts.float())
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))

    config = {"vocab_size": len(vocab), "embedding_dim": 128,
              "hidden_dim": 128, "dropout": 0.3}
    model = BiLSTMClassifier(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Device:", device)
    print("Train/validation:", len(train_loader.dataset), len(val_loader.dataset))
    print("Class counts [negative, neutral, positive]:", counts.tolist())
    print("Class weights:", [round(x, 3) for x in weights.tolist()])
    print("Parameters:", f"{sum(p.numel() for p in model.parameters()):,}")

    history = []
    best_f1 = -1.0
    best_epoch = 0
    no_improvement = 0

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
            no_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": config,
                "vocab": vocab,
                "best_epoch": best_epoch,
                "val_macro_f1": best_f1,
                "seed": SEED,
                "label_map": {"negative": 0, "neutral": 1, "positive": 2},
                "class_weights": weights.tolist(),
            }, RESULTS_DIR / "best_bilstm.pt")
        else:
            no_improvement += 1

        print(
            f"Epoch {epoch:2d}/{epochs} | "
            f"train loss {train_loss:.4f}, F1 {train_f1:.4f} | "
            f"val loss {val_loss:.4f}, F1 {val_f1:.4f}"
            + ("  [best]" if improved else ""), flush=True
        )
        if no_improvement >= patience:
            print(f"Early stopping after {patience} epochs without improvement.")
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
            "model_config": config,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "loss": "class_weighted_cross_entropy",
            "class_counts": counts.tolist(),
            "class_weights": weights.tolist(),
            "optimizer": "Adam",
        }, f, indent=2)

    print(f"\nBest epoch: {best_epoch}, validation macro-F1: {best_f1:.4f}")
    print("Saved results in:", RESULTS_DIR)
    print("Test set has NOT been evaluated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train class-weighted airline BiLSTM")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1:
        parser.error("--epochs and --patience must be positive")
    main(epochs=args.epochs, patience=args.patience)
