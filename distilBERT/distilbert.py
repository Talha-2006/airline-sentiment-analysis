import os
import random
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Initialize constants

SEED = 42

MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3

TRAIN_FILE = "data/train.csv"
VAL_FILE = "data/val.csv"
TEST_FILE = "data/test.csv"

OUTPUT_FOLDER = "distilbert_outputs"
MODEL_FOLDER = os.path.join(OUTPUT_FOLDER, "best_model")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

device = torch.device("cpu")

print("Using device:", device)

# Use seed to make results more reproducible
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)
test_df = pd.read_csv(TEST_FILE)

ID_COLUMN = "tweet_id"
TEXT_COLUMN = "text"
LABEL_COLUMN = "true_label"

label_mapping = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

def prepare_data(df):

    df = df[[ID_COLUMN, TEXT_COLUMN, LABEL_COLUMN]].copy()

    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)

    # Convert sentiment labels to the shared encoding
    if df[LABEL_COLUMN].dtype == object:
        df[LABEL_COLUMN] = (
            df[LABEL_COLUMN]
            .str.lower()
            .str.strip()
            .map(label_mapping)
        )

    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

    if not df[LABEL_COLUMN].isin([0, 1, 2]).all():
        raise ValueError("Invalid sentiment labels found.")

    return df.reset_index(drop=True)

train_df = prepare_data(train_df)
val_df = prepare_data(val_df)
test_df = prepare_data(test_df)

# Calculate class weights using the training set
class_counts = np.bincount(
    train_df[LABEL_COLUMN],
    minlength=3
)

class_weights = len(train_df) / (3 * class_counts)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float
).to(device)

print("Class counts:", class_counts)
print("Class weights:", class_weights)

print("Training examples:", len(train_df))
print("Validation examples:", len(val_df))
print("Test examples:", len(test_df))

# Check that the expected shared splits are being used
assert len(train_df) == 11521
assert len(val_df) == 1440
assert len(test_df) == 1441

# Make sure no tweet appears in multiple splits
assert set(train_df[ID_COLUMN]).isdisjoint(val_df[ID_COLUMN])
assert set(train_df[ID_COLUMN]).isdisjoint(test_df[ID_COLUMN])
assert set(val_df[ID_COLUMN]).isdisjoint(test_df[ID_COLUMN])

# Load the DistilBERT tokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Create a PyTorch dataset

class TweetDataset(Dataset):

    def __init__(self, dataframe):
        self.texts = dataframe[TEXT_COLUMN].tolist()
        self.labels = dataframe[LABEL_COLUMN].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):

        encoding = tokenizer(
            self.texts[index],
            truncation=True,
            max_length=MAX_LENGTH
        )

        return {
            "input_ids": torch.tensor(encoding["input_ids"]),
            "attention_mask": torch.tensor(
                encoding["attention_mask"]
            ),
            "labels": torch.tensor(
                self.labels[index],
                dtype=torch.long
            )
        }

# Pad tweets to the longest sequence in each batch
def collate_batch(batch):

    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )

    attention_masks = torch.nn.utils.rnn.pad_sequence(
        attention_masks,
        batch_first=True,
        padding_value=0
    )

    labels = torch.stack([item["labels"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels
    }

train_dataset = TweetDataset(train_df)
val_dataset = TweetDataset(val_df)
test_dataset = TweetDataset(test_df)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_batch
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_batch
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_batch
)

# Load DistilBERT for classification
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3
)

model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Weighted cross-entropy loss
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

def evaluate(model, data_loader):

    model.eval()

    predictions = []
    actual_labels = []

    total_loss = 0

    with torch.no_grad():

        for batch in data_loader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()

            batch_predictions = torch.argmax(outputs.logits, dim=1)

            predictions.extend(
                batch_predictions.cpu().tolist()
            )

            actual_labels.extend(
                labels.cpu().tolist()
            )

    avg_loss = total_loss / len(data_loader)

    macro_f1 = f1_score(
        actual_labels,
        predictions,
        average="macro"
    )

    accuracy = accuracy_score(
        actual_labels,
        predictions
    )

    return avg_loss, macro_f1, accuracy


# Train DistilBERT

best_val_f1 = -1

training_history = []

for epoch in range(EPOCHS):

    model.train()

    total_train_loss = 0

    for batch in train_loader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = criterion(outputs.logits, labels)

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    train_loss = total_train_loss / len(train_loader)

    # Check performance on the validation set.
    val_loss, val_f1, val_accuracy = evaluate(
        model,
        val_loader
    )

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"Training loss: {train_loss:.4f}")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Validation macro-F1: {val_f1:.4f}")

    training_history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "val_macro_f1": val_f1
    })

    # Keep the model with the best validation macro-F1
    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        model.save_pretrained(MODEL_FOLDER)
        tokenizer.save_pretrained(MODEL_FOLDER)

        print("Best model saved.")

# Save the training history
history_df = pd.DataFrame(training_history)

history_df.to_csv(
    os.path.join(OUTPUT_FOLDER, "training_history.csv"),
    index=False
)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_FOLDER)

model.to(device)

print("\nBest validation macro-F1:", best_val_f1)

# Generate prediction CSV files

def save_predictions(dataframe, data_loader, filename):

    model.eval()

    predictions = []

    with torch.no_grad():

        for batch in data_loader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            batch_predictions = torch.argmax(
                outputs.logits,
                dim=1
            )

            predictions.extend(batch_predictions.cpu().tolist())

    results = pd.DataFrame({
        "tweet_id": dataframe[ID_COLUMN].values,
        "true_label": dataframe[LABEL_COLUMN].values,
        "pred_label": predictions
    })

    results.to_csv(
        os.path.join(OUTPUT_FOLDER, filename),
        index=False
    )

    print(f"Saved {filename}")

    return results

val_results = save_predictions(
    val_df,
    val_loader,
    "validation_predictions.csv"
)

test_results = save_predictions(
    test_df,
    test_loader,
    "test_predictions.csv"
)

# Calculate final test results

y_true = test_results["true_label"]
y_pred = test_results["pred_label"]

test_accuracy = accuracy_score(y_true, y_pred)

test_macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro"
)

print("\nFinal Test Results")
print("------------------")

print("Accuracy:", round(test_accuracy, 4))
print("Macro-F1:", round(test_macro_f1, 4))

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=[
            "Negative",
            "Neutral",
            "Positive"
        ],
        digits=4
    )
)

# Generate and save the confusion matrix as a CSV

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1, 2]
)

cm_df = pd.DataFrame(
    cm,
    index=["Negative", "Neutral", "Positive"],
    columns=["Negative", "Neutral", "Positive"]
)

cm_df.index.name = "Actual"
cm_df.columns.name = "Predicted"

cm_df.to_csv(os.path.join(OUTPUT_FOLDER, "confusion_matrix.csv"))

print("\nConfusion matrix saved.")
print("\nFinished!")
print("All results saved in:", OUTPUT_FOLDER)
