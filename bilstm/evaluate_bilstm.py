"""Evaluate the selected BiLSTM checkpoint and export group prediction files."""

import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from bilstm_model import BiLSTMClassifier
from build_bilstm_data import make_loaders


BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_FILE = BASE_DIR / "bilstm_results_unweighted" / "best_bilstm.pt"
OUTPUT_DIR = BASE_DIR / "bilstm_results_final"
CLASS_NAMES = ["negative", "neutral", "positive"]


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    tweet_ids, true_labels, pred_labels = [], [], []

    for token_ids, lengths, labels, ids in loader:
        logits = model(token_ids.to(device), lengths)
        tweet_ids.extend(ids)
        true_labels.extend(labels.tolist())
        pred_labels.extend(logits.argmax(dim=1).cpu().tolist())

    return pd.DataFrame({
        "tweet_id": tweet_ids,
        "true_label": true_labels,
        "pred_label": pred_labels,
    })


def check_predictions(predictions, csv_path):
    """Make sure every prediction matches the right input tweet and label."""
    original = pd.read_csv(csv_path, dtype={"tweet_id": "string"})
    assert len(predictions) == len(original)
    assert predictions["tweet_id"].astype(str).tolist() == original["tweet_id"].astype(str).tolist()
    assert predictions["true_label"].tolist() == original["true_label"].astype(int).tolist()
    assert predictions["tweet_id"].is_unique
    assert predictions["pred_label"].isin([0, 1, 2]).all()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=device, weights_only=True)

    # make sure vocab and label map match saved checkpoint
    _, val_loader, test_loader, vocab = make_loaders()
    assert checkpoint["vocab"] == vocab, "vocab doesn't match saved model"
    assert checkpoint.get("label_map") == {"negative": 0, "neutral": 1, "positive": 2}, "unexpected label encoding"

    model = BiLSTMClassifier(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    OUTPUT_DIR.mkdir(exist_ok=True)

    # sanity check: verify reloaded model gets same val F1
    val_predictions = predict(model, val_loader, device)
    check_predictions(val_predictions, BASE_DIR / "shared_splits" / "val.csv")
    val_f1 = f1_score(val_predictions["true_label"], val_predictions["pred_label"],
                      labels=[0, 1, 2], average="macro", zero_division=0)
    assert abs(val_f1 - checkpoint["val_macro_f1"]) <= 1e-6, "reloaded model val macro-F1 mismatch"
    val_predictions.to_csv(OUTPUT_DIR / "bilstm_val_predictions.csv", index=False)
    print(f"Selected checkpoint: epoch {checkpoint['best_epoch']}")
    print(f"Validation macro-F1 (verified): {val_f1:.4f}")

    # run final test eval (no tuning or checkpoint selection)
    test_predictions = predict(model, test_loader, device)
    check_predictions(test_predictions, BASE_DIR / "shared_splits" / "test.csv")
    test_predictions.to_csv(OUTPUT_DIR / "bilstm_test_predictions.csv", index=False)

    true_labels = test_predictions["true_label"]
    pred_labels = test_predictions["pred_label"]
    report = classification_report(true_labels, pred_labels, labels=[0, 1, 2],
                                   target_names=CLASS_NAMES, zero_division=0)
    confusion = confusion_matrix(true_labels, pred_labels, labels=[0, 1, 2])
    cm_df = pd.DataFrame(confusion, index=CLASS_NAMES, columns=CLASS_NAMES)
    macro_f1 = f1_score(true_labels, pred_labels, labels=[0, 1, 2],
                        average="macro", zero_division=0)
    accuracy = accuracy_score(true_labels, pred_labels)

    (OUTPUT_DIR / "test_classification_report.txt").write_text(report, encoding="utf-8")
    cm_df.to_csv(OUTPUT_DIR / "test_confusion_matrix.csv", index_label="actual\\predicted")
    with (OUTPUT_DIR / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump({
            "selected_checkpoint": "bilstm_results_unweighted/best_bilstm.pt",
            "best_epoch": checkpoint["best_epoch"],
            "validation_macro_f1": val_f1,
            "test_macro_f1": macro_f1,
            "test_accuracy": accuracy,
            "test_examples": len(test_predictions),
            "label_map": checkpoint["label_map"],
        }, file, indent=2)

    print(f"\nTest macro-F1: {macro_f1:.4f} | Accuracy: {accuracy:.4f}\n")
    print(report)
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(cm_df)
    print(f"\nFiles saved in: {OUTPUT_DIR}")
    print("Group deliverables: bilstm_val_predictions.csv, bilstm_test_predictions.csv")


if __name__ == "__main__":
    main()
