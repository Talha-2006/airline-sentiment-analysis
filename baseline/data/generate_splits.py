from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
INPUT = Path("Tweets.csv")
OUTPUT = Path("shared_splits")
OUTPUT.mkdir(exist_ok=True)

# load raw csv, keep tweet_id as string so digits don't get rounded
raw = pd.read_csv(INPUT, dtype={"tweet_id": "string"})
required = {"tweet_id", "text", "airline_sentiment", "airline"}
assert required.issubset(raw.columns), f"Missing columns: {required - set(raw.columns)}"

# cleaning steps
data = raw[["tweet_id", "text", "airline_sentiment", "airline"]].copy()
data = data.dropna(subset=["text", "airline_sentiment"]).copy()
data["text"] = data["text"].astype(str).str.strip()
data = data.loc[data["text"].ne("")].copy()

# drop rows with conflicting labels for identical text
num_labels = data.groupby("text")["airline_sentiment"].nunique()
conflicting_texts = num_labels[num_labels > 1].index
n_conflicting = int(data["text"].isin(conflicting_texts).sum())
data = data.loc[~data["text"].isin(conflicting_texts)].copy()

# for remaining duplicate texts keep first occurrence
rows_before_dedup = len(data)
data = data.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
n_duplicates = rows_before_dedup - len(data)

assert data["tweet_id"].is_unique
assert data["text"].is_unique
assert data["airline_sentiment"].isin(LABEL_MAP).all()

# stratified 80/10/10 split (two-step split)
train, holdout = train_test_split(
    data,
    test_size=0.20,
    stratify=data["airline_sentiment"],
    random_state=SEED,
)
val, test = train_test_split(
    holdout,
    test_size=0.50,
    stratify=holdout["airline_sentiment"],
    random_state=SEED,
)

# export id, text, and encoded label only (no metadata)
for name, split in [("train", train), ("val", val), ("test", test)]:
    export = split[["tweet_id", "text"]].copy()
    export["true_label"] = split["airline_sentiment"].map(LABEL_MAP).astype(int)
    export.to_csv(OUTPUT / f"{name}.csv", index=False)
    print(f"{name}.csv: {len(export):,} rows; label counts {export['true_label'].value_counts().sort_index().to_dict()}")

assert len(train) + len(val) + len(test) == len(data)
assert set(train["text"]).isdisjoint(val["text"])
assert set(train["text"]).isdisjoint(test["text"])
assert set(val["text"]).isdisjoint(test["text"])

print(f"\nRaw: {len(raw):,}; conflicting label rows removed: {n_conflicting:,}; "
      f"additional duplicate rows removed: {n_duplicates:,}; cleaned: {len(data):,}")
print(f"Label encoding: {LABEL_MAP}")
print(f"Saved files to: {OUTPUT.resolve()}")
