# TF-IDF + logistic regression baseline

Tweet text is the sole input. Labels: 0 = negative, 1 = neutral, 2 = positive.
The supplied train/validation/test files are used in their original order and never modified.
All input columns, missing/blank required values, labels, within-split ID uniqueness,
and between-split ID disjointness are checked before training. IDs are read as strings.

## Reproduce

From the project root in PowerShell (Python 3.13.12 was used):

```powershell
python -m venv baseline/.venv
.\baseline\.venv\Scripts\python.exe -m pip install -r baseline/requirements.txt
.\baseline\.venv\Scripts\python.exe baseline/train_logreg.py --data-dir baseline/data --output-dir baseline/outputs/logreg
```

Exact command for this run (absolute paths):

```text
C:\Users\Admin\Desktop\Projects\413\final_project\baseline\.venv\Scripts\python.exe C:\Users\Admin\Desktop\Projects\413\final_project\baseline\train_logreg.py --data-dir C:\Users\Admin\Desktop\Projects\413\final_project\baseline\data --output-dir C:\Users\Admin\Desktop\Projects\413\final_project\baseline\outputs\logreg
```

The CLI defaults resolve to `baseline/data/` and `baseline/outputs/logreg/`, regardless
of working directory. Dependencies and versions are pinned in `baseline/requirements.txt`.

## Method and selection

Each of six Pipelines fits TF-IDF and logistic regression on training text/labels only.
TF-IDF uses word unigrams/bigrams, lowercase=True, stop_words=None, and scikit-learn
defaults: input='content', encoding='utf-8', decode_error='strict', strip_accents=None,
preprocessor=None, tokenizer=None, analyzer='word', token_pattern=r'(?u)\b\w\w+\b',
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

Selected: **C=10.0, class_weight=balanced**.

| Split | Macro-F1 | Accuracy |
| --- | ---: | ---: |
| Validation | 0.754037078 | 0.811111111 |
| Test | 0.756144435 | 0.808466343 |

## Outputs

- `logreg_val_predictions.csv`, `logreg_test_predictions.csv`: exactly
  `tweet_id,true_label,pred_label`, one row per source row in the same order; no index.
  Read tweet_id as a string in downstream tools (for pandas: dtype={'tweet_id': str}).
  Labels are integer strings in {0, 1, 2}. These are the shared evaluation handoff.
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
