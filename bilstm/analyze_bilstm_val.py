from pathlib import Path
import argparse
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from bilstm_model import BiLSTMClassifier
from build_bilstm_data import make_loaders

BASE_DIR = Path(__file__).resolve().parent
LABELS = ["negative", "neutral", "positive"]


def main(results_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        results_dir / "best_bilstm.pt", map_location=device, weights_only=True
    )
    model = BiLSTMClassifier(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, val_loader, _, vocab = make_loaders()
    assert checkpoint["vocab"] == vocab, "vocab doesn't match saved checkpoint"

    tweet_ids, true_labels, predictions = [], [], []
    with torch.no_grad():
        for token_ids, lengths, labels, ids in val_loader:
            logits = model(token_ids.to(device), lengths)
            tweet_ids.extend(ids)
            true_labels.extend(labels.tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())

    f1 = f1_score(true_labels, predictions, average="macro")
    print(f"Saved checkpoint: epoch {checkpoint['best_epoch']}")
    print(f"Validation macro-F1: {f1:.4f}\n")
    print(classification_report(
        true_labels, predictions, labels=[0, 1, 2],
        target_names=LABELS, zero_division=0
    ))

    confusion = confusion_matrix(true_labels, predictions, labels=[0, 1, 2])
    cm_df = pd.DataFrame(confusion, index=LABELS, columns=LABELS)
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(cm_df)

    # val loader doesn't shuffle so predictions line up with val.csv
    results = pd.DataFrame({
        "tweet_id": tweet_ids,
        "true_label": true_labels,
        "pred_label": predictions,
    })
    results.to_csv(results_dir / "validation_predictions.csv", index=False)
    cm_df.to_csv(results_dir / "validation_confusion_matrix.csv")

    original = pd.read_csv(BASE_DIR / "shared_splits" / "val.csv", dtype={"tweet_id": str})
    assert original["tweet_id"].tolist() == tweet_ids, "val tweet IDs out of order"
    original["pred_label"] = predictions
    errors = original[results["true_label"] != results["pred_label"]]
    errors.head(30).to_csv(results_dir / "validation_error_examples.csv", index=False)
    print("\nSaved validation predictions, confusion matrix, and error examples.")
    print("The test set was not evaluated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze validation-set predictions")
    parser.add_argument("--results-dir", type=Path,
                        default=BASE_DIR / "bilstm_results_unweighted")
    args = parser.parse_args()
    main(args.results_dir)
