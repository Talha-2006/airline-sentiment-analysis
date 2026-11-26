"""Paired bootstrap confidence intervals for differences in macro-F1.

Each resample draws test tweets with replacement and scores every model on the
same draw, so the interval reflects test-set sampling variation only.
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

LABELS = [0, 1, 2]


def macro_f1(y, p):
    return f1_score(y, p, labels=LABELS, average="macro", zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", action="append", required=True,
                    help="name=path to a prediction CSV (repeatable)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    data, names = None, []
    for spec in args.preds:
        name, path = spec.split("=", 1)
        names.append(name)
        df = pd.read_csv(path)[["tweet_id", "true_label", "pred_label"]]
        df = df.rename(columns={"pred_label": name})
        data = df if data is None else data.merge(
            df, on=["tweet_id", "true_label"], how="inner")
    for spec in args.preds:
        if len(pd.read_csv(spec.split("=", 1)[1])) != len(data):
            raise ValueError("prediction files do not cover the same tweets "
                             "with the same labels")

    y = data["true_label"].to_numpy()
    preds = {n: data[n].to_numpy() for n in names}
    rng = np.random.default_rng(args.seed)
    boot = {n: np.empty(args.n_boot) for n in names}
    for b in range(args.n_boot):
        idx = rng.integers(0, len(y), len(y))
        for n in names:
            boot[n][b] = macro_f1(y[idx], preds[n][idx])

    rows = []
    for a, b in itertools.combinations(names, 2):
        obs = macro_f1(y, preds[a]) - macro_f1(y, preds[b])
        lo, hi = np.percentile(boot[a] - boot[b], [2.5, 97.5])
        rows.append({"model_a": a, "model_b": b, "diff_macro_f1": obs,
                     "ci_low": lo, "ci_high": hi,
                     "excludes_zero": bool(lo > 0 or hi < 0)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.outdir, "bootstrap.csv"), index=False)
    print(f"Paired bootstrap, {len(y)} tweets, {args.n_boot} resamples, "
          f"seed {args.seed}")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
