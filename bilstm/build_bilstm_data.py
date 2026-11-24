"""Load the airline tweet splits as padded batches for the BiLSTM."""

import json
from pathlib import Path

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from build_bilstm_vocab import encode_text


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "shared_splits"
VOCAB_FILE = BASE_DIR / "bilstm_vocab.json"
BATCH_SIZE = 32


class TweetDataset(Dataset):
    def __init__(self, csv_file, vocab):
        data = pd.read_csv(csv_file, dtype={"tweet_id": "string"})
        self.tweet_ids = data["tweet_id"].tolist()
        self.labels = data["true_label"].astype(int).tolist()
        self.sequences = [encode_text(text, vocab) for text in data["text"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.sequences[index], self.labels[index], self.tweet_ids[index]


def collate_batch(batch):
    sequences, labels, tweet_ids = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    tensors = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
    padded = pad_sequence(tensors, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)

    return padded, lengths, labels, list(tweet_ids)


def make_loaders(data_dir=DATA_DIR, vocab_file=VOCAB_FILE, batch_size=BATCH_SIZE):
    with open(vocab_file, encoding="utf-8") as file:
        vocab = json.load(file)

    train_set = TweetDataset(data_dir / "train.csv", vocab)
    val_set = TweetDataset(data_dir / "val.csv", vocab)
    test_set = TweetDataset(data_dir / "test.csv", vocab)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_batch,
        generator=torch.Generator().manual_seed(42)
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )

    return train_loader, val_loader, test_loader, vocab


if __name__ == "__main__":
    train_loader, val_loader, test_loader, vocab = make_loaders()
    print(f"Vocabulary size: {len(vocab):,}")
    print(f"Train: {len(train_loader.dataset):,} tweets")
    print(f"Validation: {len(val_loader.dataset):,} tweets")
    print(f"Test: {len(test_loader.dataset):,} tweets")

    token_ids, lengths, labels, tweet_ids = next(iter(train_loader))
    print("\nFirst training batch:")
    print("Token tensor shape:", tuple(token_ids.shape))
    print("Length tensor shape:", tuple(lengths.shape))
    print("Label tensor shape:", tuple(labels.shape))
    print("First tweet ID:", tweet_ids[0])
    print("First tweet length:", lengths[0].item())
    print("First tweet label:", labels[0].item())
    print("First 10 token IDs:", token_ids[0, :10].tolist())
