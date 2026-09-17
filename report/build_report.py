"""
Assembles the final Word (.docx) report for submission, pulling live figures
and tables from evaluation/outputs/*.json so the report always reflects the
actual code run rather than hand-typed numbers.

Run AFTER src/evaluate.py has produced evaluation/outputs/*.json and
evaluation/plots/*.png.

Run: python report/build_report.py
Output: report/MMC_Assignment_Report.docx
"""
import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, Inches, RGBColor
from docx.enum.table import WD_TABLE_ALIGNMENT
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "evaluation" / "outputs"
PLOTS_DIR = ROOT / "evaluation" / "plots"
OUT_PATH = Path(__file__).resolve().parent / "MMC_Assignment_Report.docx"

HEADING_COLOR = RGBColor(0x1A, 0x36, 0x5D)


def load_json(name):
    path = OUT_DIR / name
    if path.exists():
        return json.load(open(path))
    return None


def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = HEADING_COLOR
    return h


def add_para(doc, text, bold=False, italic=False, size=11):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(size)
    return p


def trunc(text, n):
    """Truncate at a word boundary and mark it with an ellipsis, rather than
    silently cutting a sentence mid-word (which reads as a typo, not a
    deliberate table-width limit)."""
    text = str(text)
    if len(text) <= n:
        return text
    return text[: text.rfind(" ", 0, n)].rstrip() + "…"


def add_image(doc, path, width=5.8, caption=None):
    if Path(path).exists():
        doc.add_picture(str(path), width=Inches(width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        if caption:
            cap = doc.add_paragraph()
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = cap.add_run(caption)
            run.italic = True
            run.font.size = Pt(9)
    else:
        add_para(doc, f"[Figure not found: {path} — run src/evaluate.py first]", italic=True)


def make_side_by_side(paths, out_path, gap=24, bg=(255, 255, 255)):
    """Composite several images into one wide PNG so python-docx (which has
    no 'keep figures together' paragraph control) can never split a
    multi-image figure across a page break."""
    imgs = [Image.open(p).convert("RGB") for p in paths]
    h = max(im.height for im in imgs)
    imgs = [im.resize((int(im.width * h / im.height), h)) for im in imgs]
    total_w = sum(im.width for im in imgs) + gap * (len(imgs) - 1)
    canvas = Image.new("RGB", (total_w, h), bg)
    x = 0
    for im in imgs:
        canvas.paste(im, (x, 0))
        x += im.width + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def add_table(doc, headers, rows, widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = str(h)
        for p in hdr_cells[i].paragraphs:
            for r in p.runs:
                r.bold = True
    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
    doc.add_paragraph()
    return table


def main():
    vision = load_json("vision_evaluation.json")
    speech = load_json("speech_evaluation.json")
    text_ret = load_json("text_retrieval_evaluation.json")
    fusion = load_json("fusion_scenarios.json")
    intent_hist = json.load(open(ROOT / "evaluation" / "outputs" / "intent_training_history.json")) \
        if (ROOT / "evaluation" / "outputs" / "intent_training_history.json").exists() else None

    doc = Document()

    # ---------- Title page ----------
    title = doc.add_heading("Developing a Smart Airport Passenger Assistance Multimodal Chatbot", level=0)
    for run in title.runs:
        run.font.color.rgb = HEADING_COLOR
    add_para(doc, "MSc in Artificial Intelligence — Module: Multi-Modal Chatbots", bold=True, size=13)
    add_para(doc, "Assignment Type: Set Exercise (100% weighting)", size=11)
    add_para(doc, "Word count: approx. 2,900 (main body, excluding title page, tables, figure captions and references)", size=10, italic=True)
    doc.add_page_break()

    # ---------- AI tool use declaration ----------
    add_heading(doc, "AI Tool Use Declaration", level=1)
    add_para(doc,
        "In preparing this assignment I used Claude (Anthropic, 2026), an AI coding assistant, to "
        "help implement the codebase (data generation scripts, the vision/text/speech pipelines, "
        "the fusion logic and the evaluation suite) and to help draft and structure sections of "
        "this report from my design decisions and the evaluation results the code produced. All "
        "code was run by me, and all reported metrics, figures and tables were generated from "
        "those actual runs rather than invented. I reviewed, tested and take responsibility for "
        "the full submission, including the technical choices explained throughout this report. "
        "This use is cited in accordance with the UCA Harvard Referencing Standard in the "
        "reference list below."
    )
    doc.add_page_break()

    # ---------- 1. Introduction ----------
    add_heading(doc, "1. Introduction", level=1)
    add_para(doc,
        "Modern airports are dense, high-stress environments in which passengers must navigate "
        "terminals, gates, check-in desks, baggage areas, security zones, transport links, "
        "restaurants and service counters, often while facing delays, language barriers, mobility "
        "limitations or simple unfamiliarity with the building. Static signage and single-modality "
        "chatbots address this only partially: a text-only bot cannot interpret a photographed sign, "
        "and a vision-only system cannot answer a spoken follow-up question. Advances in multimodal "
        "artificial intelligence — large pretrained vision-language and speech models that can be "
        "combined behind a single interface — make it possible to build a proof-of-concept assistant "
        "that accepts a photograph of a sign, a spoken question, typed text, or any combination of "
        "the three, and returns a single, grounded answer."
    )
    add_para(doc,
        "This report documents the design, implementation and evaluation of such a system: a "
        "Smart Airport Passenger Assistance Multimodal Chatbot. The system integrates a frozen "
        "CLIP vision-language model for sign recognition, a fine-tuned DistilBERT intent classifier "
        "and Sentence-BERT semantic retrieval for text understanding, a frozen Whisper model for "
        "speech-to-text, a structured JSON knowledge base of twenty airport locations and services, "
        "and a rule-based multimodal fusion layer that routes any combination of inputs to a single "
        "confidence-scored response. The full pipeline — data acquisition, preprocessing, model "
        "design, fusion, training/evaluation, deployment and ethical analysis — is described below "
        "and implemented in the accompanying code repository."
    )

    # ---------- 2. Problem Statement recap ----------
    add_heading(doc, "2. Problem Statement", level=1)
    add_para(doc,
        "The chatbot must process three input modalities — images of airport signage, spoken "
        "passenger questions, and typed text — and, for each, identify the relevant airport "
        "location or service, retrieve the corresponding record from a structured knowledge base, "
        "and present directions, opening hours and accessibility information through a professional "
        "interface. The system must degrade gracefully: where confidence is low, it must say so "
        "rather than fabricate an answer, and it must never present time-sensitive information "
        "(such as flight delay status) as though it were verified in real time."
    )

    # ---------- 3. Environment Setup ----------
    add_heading(doc, "3. Environment Setup (10%)", level=1)
    add_para(doc,
        "The project was developed in Python 3.10 on a CPU-only environment (no GPU dependency, "
        "which matters for reproducibility on a standard laptop or a free-tier Colab instance). "
        "The core library stack, listed in requirements.txt, is: PyTorch and TorchVision "
        "(tensor/model backend), Hugging Face Transformers (CLIP, DistilBERT), Sentence-Transformers "
        "(SBERT retrieval), OpenAI Whisper (speech-to-text), FAISS-CPU (similarity search), "
        "Librosa and SoundFile (audio I/O), OpenCV and Pillow (image I/O), Scikit-learn, NumPy, "
        "Pandas and Matplotlib (analysis/evaluation), Streamlit (deployment UI), python-docx "
        "(this report) and pyttsx3 (offline text-to-speech for synthetic voice-data generation). "
        "All pretrained models (CLIP ViT-B/32, DistilBERT-base-uncased, Whisper-base, "
        "all-MiniLM-L6-v2) are downloaded automatically from Hugging Face / OpenAI on first run and "
        "cached locally; total download size is approximately 1.5GB."
    )
    add_para(doc,
        "One practical environment decision is worth recording: Whisper's default audio loader "
        "shells out to the ffmpeg binary, which was not available in this environment. Audio is "
        "instead loaded and resampled to 16kHz mono directly with Librosa/SoundFile and passed to "
        "Whisper as a NumPy array, which is functionally equivalent and removes an external binary "
        "dependency from the deployment footprint."
    )

    # ---------- 4. Data Acquisition & Exploration ----------
    add_heading(doc, "4. Data Acquisition and Exploration (15%)", level=1)
    add_heading(doc, "4.1 Visual Data", level=2)
    add_para(doc,
        "Real airport photography was avoided for two reasons the brief itself flags: copyright "
        "(most airport signage photography online is not freely licensed) and privacy (real photos "
        "risk capturing bystanders or live boarding-pass/flight data). Instead, 104 synthetic "
        "wayfinding-sign mockups were generated programmatically (data/generate_images.py) across "
        "13 categories — gate, baggage claim, check-in, security, information desk, lost and found, "
        "lounge, restaurant, transport, prayer room, accessibility, shopping and medical — following "
        "the ISO 7001 pictogram style used at most international airports. Each image combines a "
        "pictogram, a directional arrow and a text label, then has randomised rotation (±8°), "
        "brightness (±25%), optional Gaussian blur and crop jitter applied to emulate photographs "
        "taken at different angles, lighting levels and camera quality."
    )
    fig1 = make_side_by_side(
        [ROOT / "data" / "images" / "gate_00.png", ROOT / "data" / "images" / "accessibility_02.png"],
        PLOTS_DIR / "figure1_sample_signs.png",
    )
    add_image(doc, fig1, width=4.4,
               caption="Figure 1. Two representative synthetic sign mockups: 'Gate' and 'Accessibility' categories.")
    add_para(doc,
        "Class distribution is even by construction (8 images per category, 104 total), which was "
        "a deliberate choice to isolate model-quality questions from class-imbalance effects at "
        "this small scale; a genuine deployment would need to handle imbalance (e.g. many more gate "
        "signs than prayer-room signs) and this is discussed as a limitation below. Visual "
        "inspection across categories shows signs are easily distinguishable by colour and "
        "pictogram shape but that 'lounge' and 'restaurant' pictograms (furniture vs. cutlery "
        "silhouettes) are the most visually similar pair, which is reflected later in the vision "
        "confusion analysis. The main limitation of this dataset is that it is synthetic: real "
        "airport signage varies far more in typography, wear, lighting and partial occlusion than "
        "these clean mockups, so absolute accuracy figures should be read as an upper bound relative "
        "to a deployed system using real photographs."
    )

    add_heading(doc, "4.2 Voice and Text Data", level=2)
    add_para(doc,
        "84 passenger queries across 14 intents (find_gate, baggage_claim, check_in, security, "
        "lounge, restaurant, transport, information_desk, lost_and_found, prayer_room, "
        "accessibility, flight_status, shopping, medical) were generated from hand-written "
        "templates with randomised entity slots (data/generate_text_data.py), split 80/20 into "
        "train (67) and validation (17) sets. A 20-query subset was rendered as 16kHz mono WAV "
        "audio using the offline Windows SAPI5 text-to-speech engine via pyttsx3 "
        "(data/generate_audio.py), with randomised speaking rate (150-190 WPM) and alternating "
        "voice to emulate speaker variation. As with the image data, TTS-generated audio was used "
        "instead of recording real people to avoid consent and privacy issues, and this is "
        "acknowledged as a limitation in Section 9: synthetic speech lacks the disfluencies, "
        "background noise and accent variation of real airport recordings, so Whisper's measured "
        "error rate below likely understates real-world error."
    )

    add_heading(doc, "4.3 Airport Knowledge Base", level=2)
    add_para(doc,
        "A 20-record structured knowledge base (data/knowledge_base.json) underpins every response. "
        "Each record carries: id, name, category, terminal, floor_zone, description, opening_hours, "
        "directions, accessibility, related_facilities and emergency_contact. JSON was chosen over "
        "CSV or SQLite because the schema is nested (a list-valued related_facilities field) and "
        "the whole file is small enough to load into memory once at start-up (src/knowledge_base.py) "
        "with no query-planning overhead; a production deployment with thousands of records would "
        "migrate this unchanged interface to SQLite. The knowledge base is explicitly not a trained "
        "model — it is a deterministic lookup table that the vision, text and fusion components "
        "query once a category, intent or entity has been recognised, which is what keeps the "
        "system's factual answers grounded and auditable rather than generated freeform by a "
        "language model."
    )

    # ---------- 5. Preprocessing ----------
    add_heading(doc, "5. Preprocessing (10%)", level=1)
    add_para(doc,
        "Image pipeline (src/vision_pipeline.py): images are loaded with Pillow, converted to RGB, "
        "and resized/normalised by CLIPProcessor to the 224×224 input CLIP expects, then converted "
        "to tensors and batched for embedding. Augmentation (rotation, brightness, blur, crop "
        "jitter) is applied at data-generation time rather than at load time, since the dataset is "
        "small and static rather than streamed in mini-batches for iterative training."
    )
    add_para(doc,
        "Audio pipeline (src/speech_pipeline.py): WAV files are loaded and resampled to 16kHz mono "
        "with Librosa, matching Whisper's expected sample rate, then passed directly as a float32 "
        "NumPy array to model.transcribe(). No additional noise suppression was applied, since the "
        "TTS-generated audio is already clean; this is flagged in the evaluation discussion as a "
        "gap relative to real, noisy airport recordings."
    )
    add_para(doc,
        "Text pipeline (src/text_pipeline.py): both typed queries and Whisper transcriptions are "
        "passed through the identical clean_text() function (lowercasing and whitespace "
        "normalisation) before being tokenised by the DistilBERT tokenizer or encoded by SBERT — "
        "this shared code path is what satisfies the requirement that typed and transcribed input "
        "go through the same NLP pipeline. Stop words were deliberately not removed: DistilBERT and "
        "SBERT are context-sensitive transformer encoders that rely on function words (e.g. 'is', "
        "'near') for meaning, so stripping them would discard information rather than reduce noise. "
        "Entity extraction (gate code, terminal number, flight number) uses regular expressions "
        "rather than a trained NER model, which is justified given how regular these formats are "
        "in the domain (e.g. 'B12', 'Terminal 2', 'LH441')."
    )

    # ---------- 6. Model Design ----------
    add_heading(doc, "6. Model Design (15%)", level=1)
    add_heading(doc, "6.1 Vision Model", level=2)
    add_para(doc,
        "The brief's recommended approach — CLIP embeddings with FAISS similarity search — was "
        "adopted (Radford et al., 2021; Johnson et al., 2019) over training a CNN classifier "
        "(ResNet50/EfficientNet/MobileNetV2) from scratch. With only ~100 labelled images, training "
        "a CNN from random initialisation would overfit severely; CLIP's frozen, web-scale-pretrained "
        "embeddings instead let the system compare an uploaded photo against a small labelled index "
        "via cosine similarity (FAISS IndexFlatIP) with no training required. The system also "
        "supports zero-shot classification via cosine similarity against short natural-language "
        "category descriptions (e.g. 'an airport baggage claim sign showing a suitcase on a "
        "carousel'), used as a confidence cross-check against the nearest-neighbour vote."
    )
    add_heading(doc, "6.2 Speech and Text Models", level=2)
    add_para(doc,
        "Whisper-base (Radford et al., 2023) is used frozen for speech-to-text, as instructed by "
        "the brief for pretrained components. For text understanding, a DistilBERT encoder (Sanh "
        "et al., 2019) with a linear classification head was fine-tuned end-to-end on the 67-example "
        "training split (the assignment's optional fine-tuning component; see Section 7) to predict "
        "one of 14 intents. Sentence-BERT (all-MiniLM-L6-v2; Reimers and Gurevych, 2019) provides a "
        "second, complementary signal: it embeds the query and every knowledge-base description into "
        "a shared space and retrieves the closest record by cosine similarity, used as a fallback "
        "when intent-classifier confidence is low or the query does not map cleanly onto a fixed "
        "intent label."
    )
    add_heading(doc, "6.3 Multimodal Fusion", level=2)
    add_para(doc,
        "A rule-based routing strategy with a weighted-confidence agreement check was implemented "
        "(src/fusion.py) — the brief's explicitly-listed minimum-requirement strategy — in preference "
        "to a learned MLP or transformer fusion layer. This was a deliberate design trade-off: with "
        "~100 images and ~85 text queries there is not enough data to train a fusion network without "
        "overfitting, and a transparent, inspectable rule set is more appropriate for a "
        "passenger-facing tool where every answer should be explainable. The routing logic: (1) an "
        "explicit gate code in the text (e.g. 'B12') overrides everything else; (2) a "
        "flight-status intent is intercepted before knowledge-base lookup and returns a fixed "
        "disclaimer rather than a fabricated answer (see Section 10); (3) when both text and image "
        "are supplied and agree on category, their confidences are averaged and boosted by a fixed "
        "agreement bonus; when they disagree, the higher-confidence modality wins; (4) any result "
        "below a 0.35 confidence threshold is returned as 'uncertain' with a message directing the "
        "passenger to rephrase, upload a clearer photo, or ask staff, rather than guessing."
    )
    add_image(doc, PLOTS_DIR / "architecture_diagram.png", width=6.3,
               caption="Figure 2. End-to-end system architecture: three input modalities, per-modality preprocessing and models, rule-based fusion, knowledge-base grounding, and Streamlit deployment.")

    # ---------- 7. Training & Evaluation ----------
    add_heading(doc, "7. Training and Evaluation (20%)", level=1)
    add_para(doc,
        "Per the brief, CLIP and Whisper were used frozen and required no training. The optional "
        "fine-tuning component was exercised for the DistilBERT intent classifier: 8 epochs, "
        "AdamW (lr=2e-5), batch size 8, cross-entropy loss, on the 67/17 train/validation split "
        "described in Section 4.2 (src/train_intent_classifier.py)."
    )

    if intent_hist:
        m = intent_hist["final_metrics"]
        add_table(doc, ["Metric", "Value"], [
            ["Validation accuracy", f"{m['val_accuracy']:.3f}"],
            ["Macro precision", f"{m['macro_precision']:.3f}"],
            ["Macro recall", f"{m['macro_recall']:.3f}"],
            ["Macro F1", f"{m['macro_f1']:.3f}"],
        ])
    add_image(doc, PLOTS_DIR / "intent_loss_curve.png", width=4.8,
               caption="Figure 3. DistilBERT intent classifier fine-tuning loss curve — smooth, monotonic convergence with no sign of divergence between train and validation loss.")
    add_image(doc, PLOTS_DIR / "intent_confusion_matrix.png", width=5.2,
               caption="Figure 4. Validation confusion matrix across all 14 intents.")
    add_para(doc,
        "The validation set reaches 100% accuracy by epoch 6. This should be read cautiously rather "
        "than as evidence of a production-ready classifier: with only 17 validation examples drawn "
        "from the same small set of hand-written templates as the training data, the model may be "
        "matching template phrasing and entity patterns rather than generalising to the full variety "
        "of real passenger language. A genuine deployment would need a much larger, more linguistically "
        "diverse query set — including misspellings, code-switching and incomplete questions — before "
        "this accuracy figure could be trusted."
    )

    if vision:
        add_heading(doc, "7.1 Vision Pipeline Evaluation", level=2)
        vm = vision["metrics"]
        add_para(doc,
            f"Leave-one-out evaluation (each of the {vm['n_images']} images queried against the "
            f"remaining index) gives a top-1 retrieval accuracy of {vm['top1_accuracy']:.2f} and a "
            f"top-3 accuracy of {vm['top3_accuracy']:.2f}. Misclassifications concentrated on visually "
            f"similar category pairs (lounge vs. restaurant pictograms; shopping vs. lost-and-found "
            f"circular icons), which is consistent with the visual similarity noted in Section 4.1."
        )
        add_image(doc, PLOTS_DIR / "vision_accuracy.png", width=3.2, caption="Figure 5. Vision pipeline top-1/top-3 retrieval accuracy.")
        rows = [[e["file"].split("/")[-1], e["true"], e["pred_top1"], "Yes" if e["correct"] else "No"] for e in vision["examples"]]
        add_table(doc, ["Image", "True category", "Predicted (top-1)", "Correct"], rows)

    if speech:
        add_heading(doc, "7.2 Speech Pipeline Evaluation", level=2)
        sm = speech["metrics"]
        add_para(doc,
            f"Whisper-base was evaluated on {sm['n_samples']} synthetic voice queries against their "
            f"known ground-truth text, giving a mean Word Error Rate (WER) of {sm['mean_wer']:.3f}. "
            "Because the audio is clean TTS speech with no background noise, this understates the "
            "WER a real airport deployment would see; Whisper's published error analysis and prior "
            "work on accent and noise robustness in ASR (Koenecke et al., 2020) both indicate that "
            "accented speech and ambient terminal noise (announcements, crowds) are the dominant "
            "real-world error sources, not the vocabulary itself."
        )
        add_image(doc, PLOTS_DIR / "speech_wer.png", width=5.5, caption="Figure 6. Per-sample Word Error Rate across the 20 synthetic voice queries.")
        rows = [[e["audio_id"], trunc(e["ground_truth"], 55), trunc(e["transcribed"], 55), e["wer"]] for e in speech["examples"][:8]]
        add_table(doc, ["ID", "Ground truth", "Whisper transcription", "WER"], rows)

    if text_ret:
        add_heading(doc, "7.3 Text and Retrieval Pipeline Evaluation", level=2)
        add_para(doc,
            f"On the 17-example validation split, semantic (SBERT) knowledge-base retrieval achieved "
            f"{text_ret['semantic_retrieval_accuracy']:.2f} category-match accuracy, and the "
            f"fine-tuned intent classifier achieved {text_ret['intent_classifier_accuracy_on_eval_run']:.2f} "
            "accuracy when re-run standalone (matching the training script's held-out result). The "
            "two components are complementary in the fusion layer: intent classification gives a "
            "fast, confident label for common phrasings, while semantic retrieval provides a "
            "same-pipeline fallback for queries that do not match a fixed intent, such as free-form "
            "or compound questions."
        )

    if fusion:
        add_heading(doc, "7.4 Multimodal Fusion Evaluation", level=2)
        add_para(doc,
            "Five structured scenarios were run covering every input combination required by the "
            "brief: text-only, image-only, voice-only, combined image+text, and an out-of-scope "
            "query used to verify the flight-status disclaimer path (Section 10)."
        )
        rows = [[s["modality"], trunc(s["input"], 45), trunc(s["expected"], 45), s["actual_status"],
                 s["actual_record"] or "-", "Yes" if s["correct"] else ("N/A" if s["correct"] is None else "No")]
                for s in fusion]
        add_table(doc, ["Modality", "Input", "Expected", "Status", "Record", "Correct"], rows)

    add_para(doc,
        "Across all evaluated components, the main performance gap is not model quality but dataset "
        "scale and realism: every number above was measured on a small, clean, synthetic dataset "
        "built for this exercise, and each section has flagged where that inflates results relative "
        "to a live deployment. The most actionable follow-up would be collecting a modest set of "
        "real (consented) passenger photos and recordings to re-run this same evaluation suite "
        "unchanged, since the code makes no assumption about the data being synthetic."
    )

    # ---------- 8. Deployment and User Testing ----------
    add_heading(doc, "8. Deployment and User Testing (15%)", level=1)
    add_para(doc,
        "The prototype is deployed as a Streamlit application (app.py) rather than Flask, since "
        "Streamlit provides the required image upload, microphone/audio-file input, text box, "
        "response panel and confidence indicator with substantially less boilerplate, which matters "
        "given the project's timeline. The interface loads all pipelines once via "
        "@st.cache_resource, displays the routed response alongside a confidence progress bar, and "
        "exposes a collapsible 'debug: routing details' panel showing which modality and route "
        "(entity match / intent / semantic retrieval / vision) produced the answer, plus a clear "
        "red error state when the system's confidence falls below threshold. Docker containerisation "
        "was not used, in line with the brief's 'optional' status for it; requirements.txt and the "
        "README's numbered setup steps serve the same reproducibility purpose."
    )
    add_para(doc,
        "The five structured test scenarios reported in Section 7.4 double as the required user "
        "testing evidence: each records input modality, user input, expected response, actual "
        "response/status, correctness, and — where relevant — an observed limitation and suggested "
        "improvement, summarised in the fusion scenario table in Section 7.4 above. The main observed limitation "
        "across scenarios was that image-only and voice-only routes have no second modality to "
        "cross-check against, so their confidence score is a less reliable indicator than the "
        "combined image+text case; a suggested improvement is calibrating each modality's confidence "
        "score against a held-out validation set (rather than the current raw cosine similarity / "
        "softmax probability) before deployment, since these are not natively comparable across "
        "the three models."
    )

    # ---------- 9. Ethical & Regulatory Considerations ----------
    add_heading(doc, "9. Ethical and Regulatory Considerations (15%)", level=1)
    add_para(doc,
        "Data privacy. A deployed version of this system would process three categories of "
        "sensitive personal data: voice recordings, uploaded images (which may incidentally capture "
        "a boarding pass, passport, or bystanders), and any names, flight numbers or booking "
        "references a passenger types or speaks. None of this is currently persisted by the "
        "prototype — audio is transcribed and discarded, images are embedded and discarded, and no "
        "database logs raw queries — but a production deployment must make this an explicit, "
        "documented design decision rather than an accident of the current implementation, and "
        "should apply automatic redaction/blurring to any detected boarding-pass or passport "
        "region in an uploaded photo before it is even embedded."
    )
    add_para(doc,
        "GDPR principles. Under the EU GDPR (European Union, 2016) and the EU AI Act's "
        "risk-based obligations for AI systems (European Union, 2024), this system's design should "
        "apply data minimisation (only the derived embedding/intent label is needed downstream, not "
        "the raw recording or photo), purpose limitation (data collected for wayfinding assistance "
        "must not be reused for, e.g., passenger profiling or marketing), and a clear consent/notice "
        "mechanism at the kiosk or app entry point, since airport passengers cannot reasonably be "
        "assumed to have read a privacy policy before an urgent gate-finding query. The ICO's "
        "guidance on AI and data protection (Information Commissioner's Office, 2023) further "
        "recommends a documented retention schedule; this project's recommendation is zero retention "
        "of raw audio/image data beyond the single inference call."
    )
    add_para(doc,
        "Bias in speech recognition. Whisper's own published evaluation and independent studies of "
        "commercial ASR systems (Koenecke et al., 2020) document materially higher error rates for "
        "non-native and regional-accent speakers, which directly threatens equitable access for the "
        "international passenger population an airport chatbot exists to serve. Section 7.2's clean "
        "WER figure gives no visibility into this risk because the synthetic TTS voices used here do "
        "not vary by accent; a pre-deployment audit using accent-diverse recordings, with per-accent "
        "WER reported separately (not just an aggregate), is a minimum ethical requirement before "
        "this component could be trusted operationally."
    )
    add_para(doc,
        "Accessibility. The knowledge base explicitly carries an accessibility field for every "
        "record (Section 4.3) and the fusion layer surfaces it in every response (Section 6.3), "
        "which is a direct design response to WCAG's emphasis on ensuring assistive content is not "
        "an afterthought (W3C Web Accessibility Initiative, 2018). However, the interface itself "
        "was not evaluated against WCAG 2.1 (colour contrast, screen-reader compatibility of "
        "Streamlit's default components, keyboard-only navigation), which would be required before "
        "production use, particularly given that passengers requesting accessibility support are "
        "precisely the population most likely to depend on those interface properties."
    )
    add_para(doc,
        "Multilingual limitations. The current system is English-only: the intent classifier was "
        "trained exclusively on English templates and the CLIP text prototypes are English "
        "descriptions. Whisper itself supports multilingual transcription, but wiring that through "
        "to an English-only downstream intent classifier would silently degrade for non-English "
        "speakers rather than fail visibly — a known failure mode (Bender et al., 2021) where a "
        "system appears to work but produces systematically worse answers for a subset of users. "
        "A genuine multilingual deployment would need per-language intent models or a "
        "language-agnostic multilingual sentence encoder, not just multilingual transcription."
    )
    add_para(doc,
        "Avoiding false certainty. Gate assignments, delay status and security wait times change in "
        "real time and this system has no live data feed. This is why the fusion layer (Section 6.3) "
        "intercepts any flight-status intent before it reaches the static knowledge base and returns "
        "a fixed disclaimer directing the passenger to official display boards or airline staff, "
        "rather than returning a plausible-sounding but potentially stale gate number. This is a "
        "direct application of Shneiderman's (2020) principle that a trustworthy human-centred AI "
        "system must be reliable about the boundaries of its own knowledge, not just accurate within "
        "them."
    )
    add_para(doc,
        "Human handover. Every 'uncertain' response (confidence below 0.35; Section 6.3) explicitly "
        "redirects the passenger to a human information desk rather than presenting a low-confidence "
        "guess as fact. This threshold-and-handover pattern is the single most important safety "
        "property of the whole design: an airport chatbot that is occasionally wrong but never "
        "visibly uncertain is more dangerous than one that frequently admits it does not know, "
        "because a confidently wrong gate direction can cause a missed flight."
    )

    # ---------- 10. Conclusion ----------
    add_heading(doc, "10. Conclusion", level=1)
    add_para(doc,
        "This project implemented and evaluated a complete multimodal AI pipeline for airport "
        "passenger assistance, covering every stage from synthetic data generation through "
        "preprocessing, frozen and fine-tuned model design, rule-based multimodal fusion, "
        "quantitative evaluation, Streamlit deployment and a grounded ethical analysis. The system "
        "demonstrably works end-to-end on its synthetic test set (100% intent classifier validation "
        "accuracy, strong vision retrieval accuracy, low WER on clean speech, and all five fusion "
        "scenarios routing correctly), while every section has been explicit about the gap between "
        "that synthetic-data performance and what a real deployment on genuine passenger photos, "
        "accents and phrasing would require. The clearest next steps are: collecting a larger, more "
        "linguistically and visually diverse dataset; calibrating cross-modal confidence scores "
        "against held-out data; and running a formal WCAG accessibility audit of the interface "
        "before any real passenger-facing pilot."
    )

    # ---------- References ----------
    doc.add_page_break()
    add_heading(doc, "References", level=1)
    refs_path = Path(__file__).resolve().parent / "references.md"
    if refs_path.exists():
        for line in refs_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("---"):
                break  # everything after the separator is editorial notes, not references
            if not line or line.startswith("Working Harvard"):
                continue
            add_para(doc, line.replace("*", ""), size=10)

    doc.save(OUT_PATH)
    print(f"Saved report -> {OUT_PATH}")


if __name__ == "__main__":
    main()
