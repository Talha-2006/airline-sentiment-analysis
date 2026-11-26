import argparse
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

LABELS = [0, 1, 2]
NAMES = ["negative", "neutral", "positive"]
NAME_TO_ID = {n: i for i, n in enumerate(NAMES)}


def load_preds(path):
    df = pd.read_csv(path)
    missing = {"tweet_id", "true_label", "pred_label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    bad = set(df["true_label"]).union(df["pred_label"]) - set(LABELS)
    if bad:
        raise ValueError(f"{path} has labels outside 0/1/2: {sorted(bad)}")
    if df["tweet_id"].duplicated().any():
        raise ValueError(f"{path} has duplicate tweet_ids")
    return df


HANDLES = {
    "united": "United", "usairways": "US Airways", "southwestair": "Southwest",
    "americanair": "American", "jetblue": "JetBlue",
    "virginamerica": "Virgin America", "deltaassist": "Delta", "delta": "Delta",
}
HANDLE_RE = re.compile(
    r"@(" + "|".join(sorted(HANDLES, key=len, reverse=True)) + r")\b", re.I)


def infer_airline(text):
    found = {HANDLES[h.lower()] for h in HANDLE_RE.findall(str(text))}
    if len(found) == 1:
        return found.pop()
    return "Multiple" if found else "Unknown"


def load_split(path, id_col, label_col, text_col, airline_col):
    df = pd.read_csv(path)
    df = df.rename(columns={id_col: "tweet_id", label_col: "label",
                            text_col: "text", airline_col: "airline"})
    if "airline" not in df.columns:
        print(f"Note: {path} has no airline column, inferring it from "
              "the @handles in the tweet text.")
        df["airline"] = df["text"].apply(infer_airline)
        counts = df["airline"].value_counts()
        small = counts[counts < 20].index
        df.loc[df["airline"].isin(small), "airline"] = "Other"
    if not pd.api.types.is_numeric_dtype(df["label"]):
        df["label"] = df["label"].astype(str).str.strip().str.lower() \
            .map(NAME_TO_ID)
        if df["label"].isna().any():
            raise ValueError(f"{path}: unrecognized sentiment labels")
    return df[["tweet_id", "label", "text", "airline"]]


def metrics(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0)
    row = {
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS,
                             average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    for i, n in enumerate(NAMES):
        row[f"{n}_precision"] = p[i]
        row[f"{n}_recall"] = r[i]
        row[f"{n}_f1"] = f[i]
    return row


def plot_confusion(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ConfusionMatrixDisplay(cm, display_labels=NAMES).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(os.path.splitext(path)[0] + ".png", dpi=150)
    plt.close(fig)


def evaluate_model(name, df, split, outdir):
    y_true, y_pred = df["true_label"], df["pred_label"]
    row = {"model": name, "n": len(df), **metrics(y_true, y_pred)}

    plot_confusion(y_true, y_pred, name,
                   os.path.join(outdir, f"confusion_{name}.pdf"))

    wrong = df[y_true != y_pred].copy()
    if split is not None:
        wrong = wrong.merge(split[["tweet_id", "airline", "text"]],
                            on="tweet_id", how="left")
    wrong["true_name"] = wrong["true_label"].map(dict(enumerate(NAMES)))
    wrong["pred_name"] = wrong["pred_label"].map(dict(enumerate(NAMES)))
    wrong.to_csv(os.path.join(outdir, f"errors_{name}.csv"), index=False)

    airline_rows = []
    if split is not None and split["airline"].notna().any():
        merged = df.merge(split[["tweet_id", "airline"]], on="tweet_id")
        for airline, g in merged.groupby("airline"):
            airline_rows.append({
                "model": name, "airline": airline, "n": len(g),
                "macro_f1": f1_score(g["true_label"], g["pred_label"],
                                     labels=LABELS, average="macro",
                                     zero_division=0),
                "accuracy": accuracy_score(g["true_label"], g["pred_label"]),
            })
    return row, airline_rows


def check_against_split(name, df, split):
    merged = df.merge(split[["tweet_id", "label"]], on="tweet_id", how="left")
    if merged["label"].isna().any():
        raise ValueError(f"{name}: some tweet_ids are not in the split file")
    if (merged["label"] != merged["true_label"]).any():
        raise ValueError(f"{name}: true_label disagrees with the split file "
                         "(check the label encoding)")
    if len(df) != len(split):
        raise ValueError(f"{name}: has {len(df)} rows but the split has "
                         f"{len(split)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", action="append", required=False, default=[],
                    help="name=path to a prediction CSV (repeatable)")
    ap.add_argument("--split", help="split CSV with tweet text and airline")
    ap.add_argument("--id-col", default="tweet_id")
    ap.add_argument("--label-col", default="true_label")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--airline-col", default="airline")
    ap.add_argument("--majority", action="store_true",
                    help="add an always-negative reference row")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    split = None
    if args.split:
        split = load_split(args.split, args.id_col, args.label_col,
                           args.text_col, args.airline_col)

    models = {}
    for spec in args.preds:
        name, path = spec.split("=", 1)
        models[name] = load_preds(path)
        if split is not None:
            check_against_split(name, models[name], split)

    if args.majority:
        if split is None:
            raise ValueError("--majority needs --split")
        models["majority"] = pd.DataFrame({
            "tweet_id": split["tweet_id"],
            "true_label": split["label"],
            "pred_label": 0,
        })

    if not models:
        raise ValueError("nothing to evaluate: pass --preds and/or --majority")

    rows, airline_rows = [], []
    for name, df in models.items():
        row, arows = evaluate_model(name, df, split, args.outdir)
        rows.append(row)
        airline_rows.extend(arows)

    summary = pd.DataFrame(rows).set_index("model")
    summary.to_csv(os.path.join(args.outdir, "summary.csv"))

    table = summary[["macro_f1", "accuracy", "negative_f1",
                     "neutral_f1", "positive_f1"]]
    table.columns = ["Macro-F1", "Accuracy", "Neg. F1", "Neu. F1", "Pos. F1"]
    with open(os.path.join(args.outdir, "summary_table.tex"), "w") as f:
        f.write(table.to_latex(float_format="%.3f"))

    if airline_rows:
        by_airline = pd.DataFrame(airline_rows)
        by_airline.to_csv(os.path.join(args.outdir, "by_airline.csv"),
                          index=False)
        pivot = by_airline.pivot(index="airline", columns="model",
                                 values="macro_f1")
        print("\nMacro-F1 by airline:")
        print(pivot.round(3).to_string())

    print("\nSummary:")
    print(table.round(3).to_string())
    print(f"\nSaved results to {args.outdir}/")


if __name__ == "__main__":
    main()
