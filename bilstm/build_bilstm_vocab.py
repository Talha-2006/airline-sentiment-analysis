from collections import Counter
from pathlib import Path
import json
import re

import pandas as pd


TRAIN_FILE = Path(__file__).resolve().parent / "shared_splits" / "train.csv"
VOCAB_FILE = Path(__file__).resolve().parent / "bilstm_vocab.json"
MIN_FREQ = 2


def tokenize(text):
    text = text.lower()
    # Keep @mentions/hashtags/contractions/punctuation as tokens
    return re.findall(r"[@#]?\w+(?:['’]\w+)*|[^\w\s]", text)


def encode_text(text, vocab):
    return [vocab.get(word, vocab["<UNK>"]) for word in tokenize(text)]


def build_vocab(texts):
    counts = Counter()
    for text in texts:
        counts.update(tokenize(text))

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, count in counts.most_common():
        if count >= MIN_FREQ:
            vocab[word] = len(vocab)
    return vocab


def main():
    train = pd.read_csv(TRAIN_FILE, dtype={"tweet_id": "string"})
    vocab = build_vocab(train["text"].dropna())

    with VOCAB_FILE.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    example = train.iloc[0]["text"]
    print(f"Training tweets: {len(train):,}")
    print(f"Vocabulary size: {len(vocab):,}")
    print(f"Saved: {VOCAB_FILE}")
    print("Example tokens:", tokenize(example)[:15])
    print("Example IDs:", encode_text(example, vocab)[:15])


if __name__ == "__main__":
    main()
