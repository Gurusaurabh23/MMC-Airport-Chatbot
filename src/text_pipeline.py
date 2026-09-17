"""
Text understanding: intent classification + semantic KB retrieval, both
lightweight enough to run on CPU.

- IntentClassifier: DistilBERT encoder + a small trainable head, fine-tuned
  on data/text_queries.csv. Gives a hard intent label for routing.
- semantic_kb_search: frozen sentence-transformers embeddings, used as a
  fallback when the intent classifier isn't confident, or for free-text
  queries that don't map cleanly onto one of the fixed intents.

Typed queries and Whisper transcriptions both go through clean_text() ->
extract_entities() -> predict_intent(), so voice and text input are handled
identically downstream.
"""
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DISTILBERT_NAME = "distilbert-base-uncased"
SBERT_NAME = "all-MiniLM-L6-v2"

GATE_RE = re.compile(r"\b([A-C]\d{1,2})\b", re.IGNORECASE)
TERMINAL_RE = re.compile(r"\bterminal\s*([12])\b", re.IGNORECASE)
FLIGHT_RE = re.compile(r"\b([A-Z]{2}\d{2,4})\b")


def clean_text(text: str) -> str:
    """Lowercasing + whitespace normalisation shared by typed and transcribed
    input. Stop words are deliberately NOT removed: DistilBERT/SBERT are
    context-sensitive transformer encoders, and stripping function words
    (e.g. "is", "near") would remove information they rely on and would not
    improve, and could harm, downstream accuracy."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_entities(text: str) -> dict:
    """Lightweight rule-based entity extraction (gate / terminal / flight
    number). A regex approach is justified here: airport entity formats are
    highly regular (e.g. 'B12', 'Terminal 2', 'LH441'), so a trained NER
    model would add complexity without a meaningful accuracy gain on this
    scale of dataset."""
    entities = {}
    m = GATE_RE.search(text)
    if m:
        entities["gate"] = m.group(1).upper()
    m = TERMINAL_RE.search(text)
    if m:
        entities["terminal"] = f"Terminal {m.group(1)}"
    m = FLIGHT_RE.search(text.upper())
    if m:
        entities["flight"] = m.group(1)
    return entities


class IntentClassifier(nn.Module):
    """DistilBERT encoder + linear head. The encoder can be frozen or
    fine-tuned end-to-end (see src/train_intent_classifier.py); this class
    only defines the forward pass used at both training and inference time."""

    def __init__(self, n_labels: int, encoder_name: str = DISTILBERT_NAME):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, n_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]  # [CLS] token representation
        return self.classifier(self.dropout(cls))


class TextPipeline:
    def __init__(self, model_dir: Path = MODELS_DIR / "intent_classifier"):
        self.tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_NAME)
        self.sbert = SentenceTransformer(SBERT_NAME)

        self.labels = None
        self.model = None
        if (model_dir / "labels.json").exists():
            self.labels = json.load(open(model_dir / "labels.json"))
            self.model = IntentClassifier(n_labels=len(self.labels))
            state = torch.load(model_dir / "model.pt", map_location="cpu")
            self.model.load_state_dict(state)
            self.model.eval()

        self._build_kb_retrieval_index()

    def _build_kb_retrieval_index(self):
        from knowledge_base import KnowledgeBase
        self.kb = KnowledgeBase()
        docs = [f"{r.name}. {r.description}" for r in self.kb.records]
        self.kb_embeddings = self.sbert.encode(docs, normalize_embeddings=True)

    def predict_intent(self, text: str) -> Optional[dict]:
        if self.model is None:
            return None
        clean = clean_text(text)
        enc = self.tokenizer(clean, return_tensors="pt", truncation=True, padding=True, max_length=32)
        with torch.no_grad():
            # Only pass what IntentClassifier.forward() accepts; some
            # tokenizer configs also emit token_type_ids, which DistilBERT's
            # encoder does not use.
            logits = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            probs = torch.softmax(logits, dim=-1)[0]
        top_idx = int(torch.argmax(probs))
        return {
            "intent": self.labels[top_idx],
            "confidence": float(probs[top_idx]),
            "distribution": {self.labels[i]: float(probs[i]) for i in range(len(self.labels))},
        }

    def semantic_kb_search(self, text: str, top_k: int = 3):
        """Fallback path: embed the query with SBERT and cosine-match it
        directly against KB record descriptions, bypassing intent labels
        entirely. Used when intent confidence is low (see fusion.py)."""
        clean = clean_text(text)
        q_emb = self.sbert.encode([clean], normalize_embeddings=True)[0]
        sims = self.kb_embeddings @ q_emb
        order = np.argsort(-sims)[:top_k]
        return [(self.kb.records[i], float(sims[i])) for i in order]

    def process(self, text: str) -> dict:
        clean = clean_text(text)
        entities = extract_entities(clean)
        intent_result = self.predict_intent(clean)
        retrieval_result = self.semantic_kb_search(clean, top_k=3)
        return {
            "raw_text": text,
            "clean_text": clean,
            "entities": entities,
            "intent": intent_result,
            "retrieval": [(r.id, r.category, score) for r, score in retrieval_result],
        }


if __name__ == "__main__":
    tp = TextPipeline()
    for q in ["Where is gate B12?", "How do I get to baggage claim?", "Is there a lounge near terminal 2?"]:
        print(q, "->", tp.process(q))
