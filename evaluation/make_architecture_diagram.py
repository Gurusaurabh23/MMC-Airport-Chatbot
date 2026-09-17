"""
Draws the system architecture diagram with matplotlib so it stays
reproducible and version-controlled alongside the code, rather than a
hand-made external image.

Run: python evaluation/make_architecture_diagram.py
Output: evaluation/plots/architecture_diagram.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "plots" / "architecture_diagram.png"


def box(ax, xy, w, h, text, color="#2b6cb0", fontsize=9, text_color="white"):
    x, y = xy
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                            linewidth=1.2, edgecolor="#1a365d", facecolor=color)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
             color=text_color, wrap=True, fontweight="bold")
    return (x + w / 2, y, x + w / 2, y + h, x, y + h / 2, x + w, y + h / 2)


def arrow(ax, start, end, color="#4a5568"):
    a = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                         linewidth=1.4, color=color)
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(13, 8.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8.5)
    ax.axis("off")
    ax.set_title("Smart Airport Passenger Assistance Multimodal Chatbot — System Architecture",
                  fontsize=13, fontweight="bold", pad=14)

    # --- Input layer ---
    _, _, _, _, _, _, _, img_top = box(ax, (0.3, 7.0), 2.3, 0.9, "Image input\n(sign / boarding pass photo)", "#2c7a7b")
    box(ax, (3.0, 7.0), 2.3, 0.9, "Voice input\n(spoken query)", "#2c7a7b")
    box(ax, (5.7, 7.0), 2.3, 0.9, "Text input\n(typed query)", "#2c7a7b")

    # --- Preprocessing layer ---
    box(ax, (0.3, 5.6), 2.3, 0.9, "Image preprocessing\n(resize, normalise, tensor)", "#2b6cb0")
    box(ax, (3.0, 5.6), 2.3, 0.9, "Audio preprocessing\n(16kHz mono, Whisper STT)", "#2b6cb0")
    box(ax, (5.7, 5.6), 2.3, 0.9, "Text cleaning\n(lowercase, normalise)", "#2b6cb0")

    # --- Model layer ---
    box(ax, (0.3, 4.2), 2.3, 0.9, "Vision model\nCLIP + FAISS retrieval", "#2b6cb0")
    box(ax, (3.65, 4.2), 2.3, 0.9, "Entity extraction\n(gate/terminal/flight regex)", "#2b6cb0")
    box(ax, (6.35, 4.2), 2.3, 0.9, "Intent classifier\nDistilBERT (fine-tuned)", "#2b6cb0")
    box(ax, (9.05, 4.2), 2.3, 0.9, "Semantic retrieval\nSBERT cosine similarity", "#2b6cb0")

    # --- Fusion layer ---
    box(ax, (3.3, 2.8), 5.4, 0.9, "Multimodal Fusion & Routing\n(rule-based, weighted-confidence agreement check)", "#c05621")

    # --- Knowledge base ---
    box(ax, (9.3, 2.8), 2.9, 0.9, "Airport Knowledge Base\n(JSON, 20 records)", "#6b46c1")

    # --- Output layer ---
    box(ax, (3.3, 1.4), 5.4, 0.9, "Response generation\nlocation, directions, hours, accessibility, confidence score", "#276749")

    # --- Deployment ---
    box(ax, (3.3, 0.1), 5.4, 0.9, "Streamlit UI\n(image upload, mic/audio input, text box, response panel)", "#718096")

    # arrows: inputs -> preprocessing
    arrow(ax, (1.45, 7.0), (1.45, 6.5))
    arrow(ax, (4.15, 7.0), (4.15, 6.5))
    arrow(ax, (6.85, 7.0), (6.85, 6.5))

    # preprocessing -> models
    arrow(ax, (1.45, 5.6), (1.45, 5.1))
    arrow(ax, (4.15, 5.6), (4.8, 5.1))
    arrow(ax, (4.15, 5.6), (7.5, 5.1))
    arrow(ax, (6.85, 5.6), (4.8, 5.1))
    arrow(ax, (6.85, 5.6), (7.5, 5.1))
    arrow(ax, (6.85, 5.6), (10.2, 5.1))

    # models -> fusion
    arrow(ax, (1.45, 4.2), (4.2, 3.7))
    arrow(ax, (4.8, 4.2), (5.2, 3.7))
    arrow(ax, (7.5, 4.2), (6.5, 3.7))
    arrow(ax, (10.2, 4.2), (7.9, 3.7))

    # fusion <-> KB
    arrow(ax, (8.7, 3.25), (9.3, 3.25))

    # fusion -> response
    arrow(ax, (6.0, 2.8), (6.0, 2.3))

    # response -> UI
    arrow(ax, (6.0, 1.4), (6.0, 1.0))

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
