"""Train-only TF-IDF/logistic regression; select on validation, then test once."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import platform
import shlex
import subprocess
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
LABELS = [0, 1, 2]
LABEL_MAPPING = {0: "negative", 1: "neutral", 2: "positive"}
SEED = 42
PREDICTION_COLUMNS = ["tweet_id", "true_label", "pred_label"]


class IntegrityError(ValueError):
    """Invalid data or failed artifact verification; never repair implicitly."""


@dataclass
class Split:
    path: Path
    ids: list[str]
    texts: list[str]
    labels: np.ndarray


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_split(path: Path) -> Split:
    ids, texts, labels = [], [], []
    seen = set()
    # csv reads all fields as strings, preserving large IDs and leading zeros.
    # Literal strings such as "NA" are valid tweet text, not missing values.
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        columns = reader.fieldnames or []
        required = {"tweet_id", "text", "true_label"}
        if not required.issubset(columns) or len(columns) != len(set(columns)):
            raise IntegrityError(f"{path}: required unique columns {sorted(required)}; got {columns}")
        for row_number, row in enumerate(reader, start=1):
            location = f"{path}: data row {row_number}"
            if None in row or any(value is None for value in row.values()):
                raise IntegrityError(f"{location}: field count differs from the header")
            for column in required:
                if not row[column].strip():
                    raise IntegrityError(f"{location}: missing/blank {column}")
            if row["true_label"] not in {"0", "1", "2"}:
                raise IntegrityError(f"{location}: invalid true_label {row['true_label']!r}; expected 0, 1, 2")
            tweet_id = row["tweet_id"]
            if tweet_id in seen:
                raise IntegrityError(f"{location}: duplicate tweet_id {tweet_id!r}")
            seen.add(tweet_id)
            ids.append(tweet_id)
            texts.append(row["text"])
            labels.append(int(row["true_label"]))
    if not ids:
        raise IntegrityError(f"{path}: empty split")
    return Split(path, ids, texts, np.asarray(labels, dtype=np.int64))


def validate_splits(splits: dict[str, Split]) -> None:
    for (name_a, a), (name_b, b) in itertools.combinations(splits.items(), 2):
        overlap = set(a.ids).intersection(b.ids)
        if overlap:
            raise IntegrityError(
                f"{name_a}/{name_b}: {len(overlap)} overlapping tweet IDs; "
                f"examples: {sorted(overlap)[:5]}"
            )
    train_labels = set(splits["train"].labels.tolist())
    if train_labels != set(LABELS):
        raise IntegrityError(f"Training split must contain all three classes; got {sorted(train_labels)}")


def make_pipeline(c: float, class_weight: str | None) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words=None)),
        ("logreg", LogisticRegression(
            C=c, class_weight=class_weight, penalty="l2", solver="lbfgs",
            random_state=SEED, max_iter=2000,
        )),
    ])


def select_best(results: list[dict]) -> dict:
    # Exact ties: smaller C, then unweighted. Test scores never enter this key.
    return min(results, key=lambda r: (-r["macro_f1"], r["C"], r["class_weight"] is not None))


def evaluate(truth: np.ndarray, predictions: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, labels=LABELS, zero_division=0,
    )
    return {
        "macro_f1": float(f1_score(truth, predictions, labels=LABELS, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(truth, predictions)),
        "per_class": {
            str(label): {
                "name": LABEL_MAPPING[label],
                "precision": float(precision[i]), "recall": float(recall[i]),
                "f1": float(f1[i]), "support": int(support[i]),
            } for i, label in enumerate(LABELS)
        },
        "confusion_matrix": confusion_matrix(truth, predictions, labels=LABELS).tolist(),
    }


def write_predictions(path: Path, split: Split, predictions: np.ndarray) -> None:
    if len(predictions) != len(split.ids):
        raise IntegrityError(f"{path}: prediction count differs from source")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(PREDICTION_COLUMNS)
        writer.writerows((tweet_id, int(truth), int(pred)) for tweet_id, truth, pred
                         in zip(split.ids, split.labels, predictions, strict=True))


def verify_predictions(path: Path, split: Split, expected: np.ndarray) -> None:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if reader.fieldnames != PREDICTION_COLUMNS:
            raise IntegrityError(f"{path}: incorrect prediction columns/order")
        rows = list(reader)
    if len(rows) != len(split.ids):
        raise IntegrityError(f"{path}: incorrect row count")
    for i, (row, tweet_id, truth, pred) in enumerate(
        zip(rows, split.ids, split.labels, expected, strict=True), start=1
    ):
        if (set(row) != set(PREDICTION_COLUMNS) or row["tweet_id"] != tweet_id
                or row["true_label"] != str(int(truth))
                or row["pred_label"] not in {"0", "1", "2"}
                or row["pred_label"] != str(int(pred))):
            raise IntegrityError(f"{path}: schema, ID/order, truth, or prediction mismatch at row {i}")


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")


def reproduce_command(data_dir: Path, output_dir: Path) -> str:
    args = [sys.executable, str(Path(__file__).resolve()),
            "--data-dir", str(data_dir), "--output-dir", str(output_dir)]
    return subprocess.list2cmdline(args) if sys.platform == "win32" else shlex.join(args)


def write_readme(output_dir: Path, command: str, selected: dict, metrics: dict) -> None:
    val, test = metrics["validation"], metrics["test"]
    text = f"""# TF-IDF + logistic regression baseline

Tweet text is the sole input. Labels: 0 = negative, 1 = neutral, 2 = positive.
The supplied train/validation/test files are used in their original order and never modified.
All input columns, missing/blank required values, labels, within-split ID uniqueness,
and between-split ID disjointness are checked before training. IDs are read as strings.

## Reproduce

From the project root in PowerShell (Python 3.13.12 was used):

```powershell
python -m venv baseline/.venv
.\\baseline\\.venv\\Scripts\\python.exe -m pip install -r baseline/requirements.txt
.\\baseline\\.venv\\Scripts\\python.exe baseline/train_logreg.py --data-dir baseline/data --output-dir baseline/outputs/logreg
```

Exact command for this run (absolute paths):

```text
{command}
```

The CLI defaults resolve to `baseline/data/` and `baseline/outputs/logreg/`, regardless
of working directory. Dependencies and versions are pinned in `baseline/requirements.txt`.

## Method and selection

Each of six Pipelines fits TF-IDF and logistic regression on training text/labels only.
TF-IDF uses word unigrams/bigrams, lowercase=True, stop_words=None, and scikit-learn
defaults: input='content', encoding='utf-8', decode_error='strict', strip_accents=None,
preprocessor=None, tokenizer=None, analyzer='word', token_pattern=r'(?u)\\b\\w\\w+\\b',
max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False, dtype=float64,
norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False.
No custom cleaning is applied. The default tokenizer ignores single-character tokens.
The classifier uses L2, lbfgs, random_state=42, max_iter=2000, tol=1e-4,
fit_intercept=True and remaining scikit-learn 1.7.2 defaults. With three classes,
lbfgs optimizes multinomial loss (no deprecated multi_class argument is passed).
Numerical libraries are limited to one thread for consistent runs and to avoid oversubscription.
Full fitted-component settings and convergence information are recorded in run_info.json.

Grid: C in [0.1, 1.0, 10.0], class_weight in [None, 'balanced']; all else fixed.
Select by validation macro-F1 over labels [0, 1, 2]; exact ties prefer smaller C,
then None. The chosen model stays trained on training alone. Test predictions and
metrics are computed only after selection; test scores do not change any settings.
No shared evaluator exists in this folder, so metrics use scikit-learn with
labels=[0, 1, 2] and zero_division=0. Confusion rows are truth; columns are predictions.

Selected: **C={selected['C']}, class_weight={selected['class_weight']}**.

| Split | Macro-F1 | Accuracy |
| --- | ---: | ---: |
| Validation | {val['macro_f1']:.9f} | {val['accuracy']:.9f} |
| Test | {test['macro_f1']:.9f} | {test['accuracy']:.9f} |

## Outputs

- `logreg_val_predictions.csv`, `logreg_test_predictions.csv`: exactly
  `tweet_id,true_label,pred_label`, one row per source row in the same order; no index.
  Read tweet_id as a string in downstream tools (for pandas: dtype={{'tweet_id': str}}).
  Labels are integer strings in {{0, 1, 2}}. These are the shared evaluation handoff.
- `validation_results.csv`: all six settings and validation macro-F1/accuracy;
  class_weight=None is represented by the literal `None`.
- `metrics.json`: selected settings, label order, aggregate/per-class metrics and
  confusion matrices for validation and test.
- `model.joblib`: selected fitted Pipeline, including its training-only TF-IDF.
- `run_info.json`: paths, source SHA-256 hashes, split sizes and class counts,
  seed, versions, settings, fit warnings/iteration counts, and verification status.

The run reloads both prediction CSVs and the saved model, checks exact IDs/order,
truth and prediction values, confirms predictions reproduce, and verifies that the
source CSV hashes are unchanged. Convergence warnings stop the run before test evaluation.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def run(data_dir: Path, output_dir: Path) -> dict:
    data_dir, output_dir = data_dir.resolve(), output_dir.resolve()
    paths = {name: data_dir / f"{name}.csv" for name in ("train", "val", "test")}
    initial_hashes = {name: sha256(path) for name, path in paths.items()}
    splits = {name: read_split(path) for name, path in paths.items()}
    validate_splits(splits)
    split_info = {
        name: {
            "filename": split.path.name, "path": str(split.path), "rows": len(split.ids),
            "class_counts": {str(label): int(Counter(split.labels.tolist())[label]) for label in LABELS},
            "sha256": initial_hashes[name],
        } for name, split in splits.items()
    }
    for name, info in split_info.items():
        print(f"{name}: {info['rows']} rows; classes {info['class_counts']}", flush=True)
    train, val, test = (splits[name] for name in ("train", "val", "test"))
    results, candidates, fit_info = [], [], []
    # BLAS scheduling is fixed for all six settings and artifact verification.
    with threadpool_limits(limits=1):
        for c, weight in itertools.product((0.1, 1.0, 10.0), (None, "balanced")):
            model = make_pipeline(c, weight)
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                model.fit(train.texts, train.labels)
            fit_warnings = [{"category": w.category.__name__, "message": str(w.message)} for w in captured]
            if any(issubclass(w.category, ConvergenceWarning) for w in captured):
                raise RuntimeError(
                    f"C={c}, class_weight={weight}: convergence failed at max_iter=2000; "
                    f"test has not been evaluated. Warnings: {fit_warnings}"
                )
            predictions = model.predict(val.texts)
            row = {
                "C": c, "class_weight": weight,
                "macro_f1": float(f1_score(val.labels, predictions, labels=LABELS, average="macro", zero_division=0)),
                "accuracy": float(accuracy_score(val.labels, predictions)),
            }
            results.append(row)
            candidates.append((row, model, predictions))
            fit_info.append({"C": c, "class_weight": weight, "warnings": fit_warnings,
                             "n_iter": model.named_steps["logreg"].n_iter_.tolist()})
            print(f"C={c:4g}, class_weight={str(weight):8s}: val macro-F1={row['macro_f1']:.9f}, "
                  f"accuracy={row['accuracy']:.9f}", flush=True)

        best = select_best(results)
        _, selected_model, val_predictions = next(item for item in candidates if item[0] is best)
        selected = {key: best[key] for key in ("C", "class_weight")}
        print(f"Selected using validation only: {selected}", flush=True)
        # Selection is now final. No fits or changes to settings occur below.
        test_predictions = selected_model.predict(test.texts)
        metrics = {
            "selected_settings": selected, "class_order": LABELS,
            "confusion_matrix_axes": {"rows": "true_label", "columns": "pred_label"},
            "validation": evaluate(val.labels, val_predictions),
            "test": evaluate(test.labels, test_predictions),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "validation_results.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["C", "class_weight", "macro_f1", "accuracy"])
            writer.writeheader()
            for row in results:
                writer.writerow({**row, "class_weight": "None" if row["class_weight"] is None else row["class_weight"]})
        for name, split, predictions in (("val", val, val_predictions), ("test", test, test_predictions)):
            path = output_dir / f"logreg_{name}_predictions.csv"
            write_predictions(path, split, predictions)
            verify_predictions(path, split, predictions)
        joblib.dump(selected_model, output_dir / "model.joblib")
        reloaded = joblib.load(output_dir / "model.joblib")
        for split, predictions in ((val, val_predictions), (test, test_predictions)):
            if not np.array_equal(reloaded.predict(split.texts), predictions):
                raise IntegrityError(f"Saved model predictions differ for {split.path}")
    final_hashes = {name: sha256(path) for name, path in paths.items()}
    if final_hashes != initial_hashes:
        raise IntegrityError("Source CSV hashes changed during the run")
    command = reproduce_command(data_dir, output_dir)
    run_info = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "reproduction_command": command,
        "splits": split_info, "label_mapping": LABEL_MAPPING, "class_order": LABELS,
        "random_seed": SEED, "python_version": platform.python_version(),
        "python_executable": sys.executable, "platform": platform.platform(),
        "package_versions": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")},
        "selected_settings": selected,
        "selection_rule": "maximum validation macro-F1; exact ties: smaller C, then class_weight=None",
        "fitted_on": "train only; no train+validation refit", "numerical_threads": 1,
        "fit_diagnostics": fit_info,
        "vectorizer_parameters": selected_model.named_steps["tfidf"].get_params(),
        "classifier_parameters": selected_model.named_steps["logreg"].get_params(),
        "verification": {
            "prediction_csv_columns_counts_order_ids_truth_ranges": True,
            "reloaded_model_predictions_match_both_exports": True,
            "source_sha256_unchanged": True,
            "all_six_fits_without_convergence_warnings": True,
        },
        "final_source_sha256": final_hashes,
    }
    # dtype is the sole non-JSON scalar among vectorizer parameters.
    run_info["vectorizer_parameters"]["dtype"] = np.dtype(run_info["vectorizer_parameters"]["dtype"]).name
    save_json(output_dir / "metrics.json", metrics)
    save_json(output_dir / "run_info.json", run_info)
    write_readme(output_dir, command, selected, metrics)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Verified all exports, saved model, convergence, and unchanged sources: {output_dir}", flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "logreg")
    args = parser.parse_args()
    try:
        run(args.data_dir, args.output_dir)
    except (IntegrityError, OSError, csv.Error, RuntimeError, ValueError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
