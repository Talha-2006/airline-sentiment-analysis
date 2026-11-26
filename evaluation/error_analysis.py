import argparse
import os

import pandas as pd

NAMES = ["negative", "neutral", "positive"]
NAME_TO_ID = {n: i for i, n in enumerate(NAMES)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", action="append", required=True,
                    help="name=path to a prediction CSV (repeatable)")
    ap.add_argument("--split", required=True,
                    help="split CSV with tweet_id and text columns")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--review", action="append", default=[],
                    help="model:true:pred, e.g. distilbert:negative:positive")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    split = pd.read_csv(args.split).rename(columns={args.text_col: "text"})
    data = split[["tweet_id", "text"]]
    names = []
    for spec in args.preds:
        name, path = spec.split("=", 1)
        names.append(name)
        df = pd.read_csv(path)[["tweet_id", "true_label", "pred_label"]]
        df = df.rename(columns={"pred_label": name})
        if "true_label" in data.columns:
            data = data.merge(df, on=["tweet_id", "true_label"], how="inner")
        else:
            data = data.merge(df, on="tweet_id", how="inner")
    y = data["true_label"]
    has_q = data["text"].str.contains(r"\?", regex=True)

    print(f"{len(data)} tweets\n")
    for n in names:
        wrong = data[data[n] != y]
        pairs = (wrong.groupby(["true_label", n]).size()
                 .sort_values(ascending=False).head(3))
        desc = ", ".join(f"{NAMES[t]}->{NAMES[p]}: {c}"
                         for (t, p), c in pairs.items())
        print(f"{n}: {len(wrong)} errors; most common {desc}")

    all_wrong = data[pd.concat([data[n] != y for n in names], axis=1).all(axis=1)]
    same = (all_wrong[names].nunique(axis=1) == 1).sum()
    by_label = {NAMES[k]: v for k, v in
                all_wrong["true_label"].value_counts().sort_index().items()}
    print(f"\nMisclassified by all models: {len(all_wrong)} "
          f"(all predict the same wrong label: {same}); by true label {by_label}")
    all_wrong.to_csv(os.path.join(args.outdir, "all_models_wrong.csv"), index=False)

    neutral = y == NAME_TO_ID["neutral"]
    print(f"\nQuestion marks: {has_q[neutral].mean():.0%} of all neutral tweets")
    for n in names:
        nn = neutral & (data[n] == NAME_TO_ID["negative"])
        if nn.sum():
            print(f"  {n}: {has_q[nn].sum()} of {nn.sum()} neutral->negative "
                  f"errors ({has_q[nn].mean():.0%})")

    for spec in args.review:
        model, t, p = spec.split(":")
        sel = data[(y == NAME_TO_ID[t]) & (data[model] == NAME_TO_ID[p])]
        path = os.path.join(args.outdir, f"review_{model}_{t}_as_{p}.csv")
        sel[["tweet_id", "text", "true_label", *names]].to_csv(path, index=False)
        print(f"\nWrote {len(sel)} tweets to {path} for manual review")


if __name__ == "__main__":
    main()
