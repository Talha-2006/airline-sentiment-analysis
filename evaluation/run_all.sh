#!/usr/bin/env bash
# Runs the shared evaluation on every model's saved predictions.
# DATA_DIR must hold the val.csv and test.csv produced by
# baseline/data/generate_splits.py (not committed; they contain tweet text).
set -euo pipefail
cd "$(dirname "$0")"

DATA_DIR="${DATA_DIR:-../data}"
PY="${PYTHON:-python}"
OUT="${OUT:-results}"

BASE=../baseline/outputs/logreg
BI=../bilstm/bilstm_results_final
DB=../distilBERT/distilbert_outputs

for f in "$DATA_DIR/val.csv" "$DATA_DIR/test.csv"; do
  [ -f "$f" ] || { echo "Missing $f (set DATA_DIR)"; exit 1; }
done

echo "=== Validation ==="
"$PY" evaluate.py --split "$DATA_DIR/val.csv" --majority --outdir "$OUT/val" \
  --preds tfidf_lr=$BASE/logreg_val_predictions.csv \
  --preds bilstm=$BI/bilstm_val_predictions.csv \
  --preds distilbert=$DB/validation_predictions.csv

echo; echo "=== Test ==="
"$PY" evaluate.py --split "$DATA_DIR/test.csv" --majority --outdir "$OUT/test" \
  --preds tfidf_lr=$BASE/logreg_test_predictions.csv \
  --preds bilstm=$BI/bilstm_test_predictions.csv \
  --preds distilbert=$DB/test_predictions.csv

echo; echo "=== Bootstrap (test) ==="
"$PY" significance.py --outdir "$OUT/test" \
  --preds distilbert=$DB/test_predictions.csv \
  --preds tfidf_lr=$BASE/logreg_test_predictions.csv \
  --preds bilstm=$BI/bilstm_test_predictions.csv

echo; echo "=== Error analysis (test) ==="
"$PY" error_analysis.py --split "$DATA_DIR/test.csv" --outdir "$OUT/test" \
  --preds tfidf_lr=$BASE/logreg_test_predictions.csv \
  --preds bilstm=$BI/bilstm_test_predictions.csv \
  --preds distilbert=$DB/test_predictions.csv \
  --review distilbert:negative:positive
