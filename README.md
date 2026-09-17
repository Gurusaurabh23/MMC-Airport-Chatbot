# Smart Airport Passenger Assistance Multimodal Chatbot

Proof-of-concept multimodal chatbot for the MSc AI *Multi-Modal Chatbots* set
exercise. Combines a frozen CLIP vision model, a fine-tuned
DistilBERT intent classifier, a frozen Whisper speech-to-text model, and a
structured airport knowledge base behind a rule-based multimodal fusion
layer, deployed as a Streamlit app.

## 1. Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

> Tested with Python 3.10 on CPU only — no GPU required. First run downloads
> CLIP (`openai/clip-vit-base-patch32`), DistilBERT (`distilbert-base-uncased`),
> Whisper (`base`) and a sentence-transformer (`all-MiniLM-L6-v2`) from
> Hugging Face / OpenAI (~1.5GB total, cached after first download).

## 2. Generate the synthetic dataset

Uses self-curated/synthetic images and TTS-generated voice data to avoid
copyright/privacy issues with real airport photography or recordings of
the public. Run once, in order:

```bash
python data/generate_images.py       # -> data/images/*.png (104 sign mockups, 13 categories)
python data/generate_text_data.py    # -> data/text_queries.csv (84 labelled passenger queries, 14 intents)
python data/generate_audio.py        # -> data/audio/*.wav (20 TTS voice queries)
```

## 3. Train the intent classifier

```bash
python src/train_intent_classifier.py
```
Saves `models/intent_classifier/{model.pt,labels.json}` and writes the loss
curve + confusion matrix to `evaluation/plots/`.

## 4. Run the evaluation suite

```bash
python src/evaluate.py
```
Writes vision top-1/top-3 accuracy, Whisper WER, text/retrieval accuracy, and
five multimodal fusion test scenarios to `evaluation/outputs/*.json` and
`evaluation/plots/*.png`.

## 5. Launch the chatbot

```bash
streamlit run app.py
```
Opens a browser UI with image upload, microphone/audio-file input, a text
box, a response panel and a confidence indicator.

## Project structure

```
data/                    synthetic dataset generators + generated data
  generate_images.py       -> data/images/, data/image_labels.csv
  generate_text_data.py    -> data/text_queries.csv
  generate_audio.py        -> data/audio/, data/audio_manifest.csv
  knowledge_base.json       20-record structured airport knowledge base
src/
  knowledge_base.py         KB loader / category & entity lookups
  vision_pipeline.py        CLIP + FAISS image retrieval & zero-shot classification
  text_pipeline.py          cleaning, entity extraction, DistilBERT intent head, SBERT retrieval
  speech_pipeline.py        Whisper transcription (frozen)
  train_intent_classifier.py fine-tunes the DistilBERT head
  fusion.py                 rule-based multimodal routing/fusion
  evaluate.py               full evaluation suite
app.py                    Streamlit deployment prototype
evaluation/                generated metrics (outputs/) and plots (plots/)
models/                    saved fine-tuned model weights (generated)
report/                   final Word report
```

## Notes on design choices

- **Vision:** CLIP+FAISS rather than training a CNN from scratch, because
  the dataset (~100 images) is far too small to train a classifier without
  severe overfitting.
- **Speech:** Whisper-base used frozen; audio is decoded with
  librosa/soundfile rather than Whisper's default ffmpeg path since ffmpeg
  isn't available in this environment.
- **Text:** DistilBERT fine-tuned as a lightweight intent classifier, with
  SBERT semantic retrieval as a fallback when intent confidence is low, and
  regex-based entity extraction for highly-regular fields (gate codes,
  terminal numbers, flight numbers).
- **Fusion:** rule-based routing with a weighted-confidence agreement check,
  chosen over a learned MLP/transformer fusion layer because of the small
  dataset and because transparency matters for a passenger-facing tool.
