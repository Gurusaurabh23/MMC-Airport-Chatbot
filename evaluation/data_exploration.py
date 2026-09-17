"""
Produces the data-exploration artefacts the brief explicitly asks for and
that weren't yet in the report: class distribution plots, a visual
similarity comparison, a concrete preprocessing before/after example, and
worked text tokenisation/entity-extraction examples.

Run: python evaluation/data_exploration.py   (after generate_images.py and
generate_text_data.py)
Outputs: evaluation/plots/*.png, evaluation/outputs/text_exploration.json
"""
import ast
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor

ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = ROOT / "evaluation" / "plots"
OUT_DIR = ROOT / "evaluation" / "outputs"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))
from text_pipeline import clean_text, extract_entities  # noqa: E402


def image_class_distribution():
    rows = list(csv.DictReader(open(ROOT / "data" / "image_labels.csv", newline="", encoding="utf-8")))
    counts = Counter(r["category"] for r in rows)
    cats = sorted(counts)
    plt.figure(figsize=(9, 4.5))
    plt.bar(cats, [counts[c] for c in cats], color="#2b6cb0")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Number of images")
    plt.title("Visual Dataset: Class Distribution (104 images, 13 categories)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "image_class_distribution.png", dpi=150)
    plt.close()
    print("saved image_class_distribution.png")


def text_intent_distribution():
    rows = list(csv.DictReader(open(ROOT / "data" / "text_queries.csv", newline="", encoding="utf-8")))
    counts = Counter(r["intent"] for r in rows)
    cats = sorted(counts)
    plt.figure(figsize=(9, 4.5))
    plt.bar(cats, [counts[c] for c in cats], color="#805ad5")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Number of queries")
    plt.title("Text Dataset: Intent Distribution (84 queries, 14 intents)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "text_intent_distribution.png", dpi=150)
    plt.close()
    print("saved text_intent_distribution.png")


def visual_similarity_comparison():
    """Side-by-side figure backing the report's claim that lounge/restaurant
    pictograms are the most visually similar pair, and gate/security signs
    are clearly dissimilar (different colour + pictogram shape)."""
    def load(cat, idx=0):
        return Image.open(ROOT / "data" / "images" / f"{cat}_{idx:02d}.png").convert("RGB")

    pairs = [("lounge", "restaurant", "Most visually similar pair"),
             ("gate", "security", "Clearly dissimilar pair")]
    fig, axes = plt.subplots(2, 2, figsize=(6, 7.2))
    for row, (a, b, label) in enumerate(pairs):
        axes[row, 0].imshow(load(a))
        axes[row, 0].set_title(a, fontsize=10)
        axes[row, 0].axis("off")
        axes[row, 1].imshow(load(b))
        axes[row, 1].set_title(b, fontsize=10)
        axes[row, 1].axis("off")
    fig.suptitle("", fontsize=1)  # reserve top margin before overlaying labels
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    for row, (a, b, label) in enumerate(pairs):
        fig.text(0.5, axes[row, 0].get_position().y1 + 0.035, label, ha="center", fontsize=11, style="italic")
    plt.savefig(PLOTS_DIR / "visual_similarity_comparison.png", dpi=150)
    plt.close()
    print("saved visual_similarity_comparison.png")


def preprocessing_sample_output():
    """Shows the actual image preprocessing pipeline (load -> resize ->
    CLIP normalise -> tensor) with a real before/after image, rather than
    just describing it in prose."""
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    original = Image.open(ROOT / "data" / "images" / "baggage_claim_00.png").convert("RGB")
    inputs = processor(images=original, return_tensors="pt")
    tensor = inputs["pixel_values"][0]  # (3, 224, 224), CLIP-normalised

    # un-normalise for visualisation using CLIP's published mean/std
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    unnorm = (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    axes[0].imshow(original)
    axes[0].set_title(f"1. Original\n{original.size[0]}x{original.size[1]}", fontsize=10)
    axes[0].axis("off")
    axes[1].imshow(np.array(original.resize((224, 224))))
    axes[1].set_title("2. Resized\n224x224", fontsize=10)
    axes[1].axis("off")
    axes[2].imshow(unnorm)
    axes[2].set_title("3. CLIP-normalised tensor\n(shown un-normalised)", fontsize=10)
    axes[2].axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(PLOTS_DIR / "preprocessing_sample_output.png", dpi=150)
    plt.close()
    print("saved preprocessing_sample_output.png")
    print(f"  tensor shape: {tuple(tensor.shape)}, dtype: {tensor.dtype}, "
          f"value range: [{tensor.min():.2f}, {tensor.max():.2f}]")


def text_exploration_examples():
    """Worked examples of the text pipeline: cleaning, tokenisation and
    entity extraction on a handful of real queries from the dataset."""
    rows = list(csv.DictReader(open(ROOT / "data" / "text_queries.csv", newline="", encoding="utf-8")))
    with_entities = [r for r in rows if r["entities"] != "{}"]
    # pick a diverse spread rather than the first N (which skew toward
    # whichever intent happens to appear first in the CSV)
    seen_entity_types = set()
    samples = []
    for r in with_entities:
        ent_keys = tuple(sorted(ast.literal_eval(r["entities"]).keys()))
        if ent_keys not in seen_entity_types:
            samples.append(r)
            seen_entity_types.add(ent_keys)
    while len(samples) < 5 and len(samples) < len(with_entities):
        for r in with_entities:
            if r not in samples:
                samples.append(r)
                break
    samples = samples[:5]
    examples = []
    for r in samples:
        clean = clean_text(r["text"])
        entities = extract_entities(clean)
        examples.append({
            "raw_text": r["text"],
            "clean_text": clean,
            "tokens": clean.split(),
            "entities": entities,
            "intent_label": r["intent"],
        })
    json.dump(examples, open(OUT_DIR / "text_exploration.json", "w"), indent=2)
    print(f"saved text_exploration.json ({len(examples)} worked examples)")

    # vocabulary size for the report narrative
    vocab = set()
    for r in rows:
        vocab.update(clean_text(r["text"]).split())
    print(f"  vocabulary size across {len(rows)} queries: {len(vocab)} unique tokens")
    json.dump({"n_queries": len(rows), "vocab_size": len(vocab)},
               open(OUT_DIR / "text_vocab_stats.json", "w"), indent=2)


if __name__ == "__main__":
    image_class_distribution()
    text_intent_distribution()
    visual_similarity_comparison()
    preprocessing_sample_output()
    text_exploration_examples()
