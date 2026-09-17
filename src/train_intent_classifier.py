"""
Fine-tunes the DistilBERT intent classifier from text_pipeline.py on
data/text_queries.csv. CLIP and Whisper stay frozen elsewhere in this
project; this is the one place actual gradient-based training happens, and
it's deliberately small — ~70 training sentences, 14 classes, 8 epochs,
CPU-only, a few minutes total.

Run: python src/train_intent_classifier.py
Outputs:
  models/intent_classifier/model.pt
  models/intent_classifier/labels.json
  evaluation/outputs/intent_training_history.json
  evaluation/plots/intent_loss_curve.png
  evaluation/plots/intent_confusion_matrix.png
"""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_pipeline import IntentClassifier, clean_text, DISTILBERT_NAME  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "intent_classifier"
PLOTS_DIR = ROOT / "evaluation" / "plots"
OUT_DIR = ROOT / "evaluation" / "outputs"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


class IntentDataset(Dataset):
    def __init__(self, rows, tokenizer, label2idx, max_len=32):
        self.rows = rows
        self.tokenizer = tokenizer
        self.label2idx = label2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        enc = self.tokenizer(
            clean_text(row["text"]), truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "label": self.label2idx[row["intent"]],
        }


def load_rows():
    with open(ROOT / "data" / "text_queries.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    return train_rows, val_rows


def main():
    train_rows, val_rows = load_rows()
    labels = sorted({r["intent"] for r in train_rows + val_rows})
    label2idx = {l: i for i, l in enumerate(labels)}
    print(f"Training on {len(train_rows)} examples, validating on {len(val_rows)}, {len(labels)} intents")

    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_NAME)
    train_ds = IntentDataset(train_rows, tokenizer, label2idx)
    val_ds = IntentDataset(val_rows, tokenizer, label2idx)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)

    model = IntentClassifier(n_labels=len(labels))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    EPOCHS = 8

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch["label"])
        train_loss = total_loss / len(train_ds)

        model.eval()
        val_loss, correct = 0.0, 0
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch["input_ids"], batch["attention_mask"])
                loss = criterion(logits, batch["label"])
                val_loss += loss.item() * len(batch["label"])
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == batch["label"]).sum().item()
                all_preds.extend(preds.tolist())
                all_true.extend(batch["label"].tolist())
        val_loss /= len(val_ds)
        val_acc = correct / len(val_ds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch+1}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

    # --- save model + labels ---
    torch.save(model.state_dict(), MODEL_DIR / "model.pt")
    json.dump(labels, open(MODEL_DIR / "labels.json", "w"))

    # --- final metrics ---
    precision, recall, f1, _ = precision_recall_fscore_support(all_true, all_preds, average="macro", zero_division=0)
    metrics = {"val_accuracy": val_acc, "macro_precision": precision, "macro_recall": recall, "macro_f1": f1}
    print("Final validation metrics:", metrics)
    json.dump({"history": history, "final_metrics": metrics}, open(OUT_DIR / "intent_training_history.json", "w"), indent=2)

    # --- loss curve plot ---
    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], label="Train loss")
    plt.plot(history["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Intent Classifier Fine-tuning: Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "intent_loss_curve.png", dpi=150)
    plt.close()

    # --- confusion matrix plot ---
    cm = confusion_matrix(all_true, all_preds, labels=list(range(len(labels))))
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted intent")
    plt.ylabel("True intent")
    plt.title("Intent Classifier: Validation Confusion Matrix")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "intent_confusion_matrix.png", dpi=150)
    plt.close()

    print(f"Saved model -> {MODEL_DIR}")
    print(f"Saved plots -> {PLOTS_DIR}")


if __name__ == "__main__":
    main()
