"""
CLIP + FAISS vision pipeline for matching an airport sign photo to a
knowledge-base category. CLIP stays frozen and projects both images and
short text descriptions into the same embedding space, so a new sign
category can be added just by writing one more description below instead
of retraining anything. Chosen over training a CNN from scratch because
~100 images per category isn't enough to avoid overfitting.
"""
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_NAME = "openai/clip-vit-base-patch32"

# Natural-language descriptions used as the "class prototypes" CLIP matches
# uploaded images against — this is what lets CLIP+FAISS generalise to
# categories with very few training images.
CATEGORY_DESCRIPTIONS = {
    "gate": "an airport gate sign showing an aircraft boarding gate number",
    "baggage_claim": "an airport baggage claim sign showing a suitcase on a carousel",
    "check_in": "an airport check-in desk sign showing a clock and counter",
    "security": "an airport security checkpoint sign with a shield and checkmark",
    "information_desk": "an airport information desk sign with a letter i in a circle",
    "lost_and_found": "an airport lost and found office sign with a circle and cross",
    "lounge": "an airport lounge sign showing an armchair",
    "restaurant": "an airport restaurant sign showing a fork and a cup",
    "transport": "an airport transport sign showing a train or bus",
    "prayer_room": "an airport prayer room sign showing a crescent moon",
    "accessibility": "an airport accessibility sign showing a wheelchair symbol",
    "shopping": "an airport duty free shopping sign showing a shopping bag",
    "medical": "an airport medical or pharmacy sign showing a red cross",
}


class VisionPipeline:
    def __init__(self, images_csv: Path = None, device: str = "cpu"):
        self.device = device
        self.model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)

        self.categories = list(CATEGORY_DESCRIPTIONS.keys())
        self._build_text_prototypes()

        self.image_paths: List[Path] = []
        self.image_labels: List[str] = []
        self.index: faiss.Index = None
        if images_csv is None:
            images_csv = DATA_DIR / "image_labels.csv"
        if images_csv.exists():
            self._build_image_index(images_csv)

    @staticmethod
    def _unwrap_embeds(output):
        """CLIPModel.get_text_features/get_image_features are documented to
        return a plain tensor, but this is defensive against transformers
        versions (observed with 5.17.0) that instead return a ModelOutput
        wrapper, whose projected embedding lives under a differently-named
        attribute depending on version."""
        if torch.is_tensor(output):
            return output
        for attr in ("text_embeds", "image_embeds", "pooler_output"):
            if hasattr(output, attr) and getattr(output, attr) is not None:
                return getattr(output, attr)
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state[:, 0]
        raise TypeError(f"Unexpected CLIP output type: {type(output)}")

    def _build_text_prototypes(self):
        texts = [CATEGORY_DESCRIPTIONS[c] for c in self.categories]
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            feats = self._unwrap_embeds(self.model.get_text_features(**inputs))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        self.text_prototypes = feats.cpu().numpy().astype("float32")

    @torch.no_grad()
    def embed_image(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        feat = self._unwrap_embeds(self.model.get_image_features(**inputs))
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy().astype("float32")

    def _build_image_index(self, images_csv: Path):
        import csv
        rows = list(csv.DictReader(open(images_csv, newline="", encoding="utf-8")))
        embeddings = []
        for row in rows:
            path = DATA_DIR.parent / row["filename"]
            img = Image.open(path).convert("RGB")
            emb = self.embed_image(img)[0]
            embeddings.append(emb)
            self.image_paths.append(path)
            self.image_labels.append(row["category"])
        mat = np.stack(embeddings).astype("float32")
        self.index = faiss.IndexFlatIP(mat.shape[1])  # cosine sim via normalised inner product
        self.index.add(mat)
        self._image_matrix = mat

    def classify_by_text_prototype(self, image: Image.Image) -> List[Tuple[str, float]]:
        """Zero-shot classification: cosine similarity between the image and
        each category's text prototype. Returns categories ranked by score."""
        emb = self.embed_image(image)
        sims = (emb @ self.text_prototypes.T)[0]
        order = np.argsort(-sims)
        return [(self.categories[i], float(sims[i])) for i in order]

    def retrieve_similar_images(self, image: Image.Image, top_k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve the most similar labelled images from the FAISS index,
        used for the vision evaluation (top-1 / top-3 accuracy)."""
        emb = self.embed_image(image)
        scores, idx = self.index.search(emb, top_k)
        return [(self.image_labels[i], float(scores[0][j])) for j, i in enumerate(idx[0])]

    def predict(self, image: Image.Image, top_k: int = 3) -> dict:
        """Combines both signals: FAISS nearest-neighbour vote (primary,
        grounded in real labelled examples) with the zero-shot text-prototype
        score as a confidence cross-check."""
        neighbours = self.retrieve_similar_images(image, top_k=top_k)
        top_category = neighbours[0][0]
        top_score = neighbours[0][1]
        prototype_scores = dict(self.classify_by_text_prototype(image))
        return {
            "category": top_category,
            "confidence": top_score,
            "neighbours": neighbours,
            "prototype_score": prototype_scores.get(top_category, 0.0),
        }


if __name__ == "__main__":
    vp = VisionPipeline()
    test_img = Image.open(DATA_DIR / "images" / "gate_00.png").convert("RGB")
    result = vp.predict(test_img)
    print("Prediction:", result)
