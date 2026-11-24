"""BiLSTM classifier for airline tweet sentiment."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from build_bilstm_data import make_loaders


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, 3)

    def forward(self, token_ids, lengths):
        embeddings = self.embedding(token_ids)
        packed = pack_padded_sequence(
            embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)

        # concat final forward and backward hidden states
        tweet_representation = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(self.dropout(tweet_representation))


if __name__ == "__main__":
    train_loader, _, _, vocab = make_loaders()
    model = BiLSTMClassifier(len(vocab))
    model.eval()

    token_ids, lengths, labels, tweet_ids = next(iter(train_loader))
    with torch.no_grad():
        logits = model(token_ids, lengths)

    print("Tokens:", tuple(token_ids.shape))
    print("Lengths:", tuple(lengths.shape))
    print("Logits:", tuple(logits.shape))
    print("Predicted labels:", logits.argmax(dim=1)[:10].tolist())
    print("Trainable parameters:", f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
