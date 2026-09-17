"""
Evaluates all four pipeline components and saves metrics/plots:
  1. Vision       - top-1/top-3 retrieval accuracy, leave-one-out FAISS search.
  2. Speech       - Whisper transcription vs. ground truth, Word Error Rate.
  3. Text/retrieval - intent classifier accuracy + semantic KB-retrieval accuracy.
  4. Fusion       - five scripted scenarios (text/voice/image/image+text/
                     voice+image), each logged with expected vs. actual result.

Run: python src/evaluate.py   (after generate_images.py, generate_audio.py
and train_intent_classifier.py)
Outputs: evaluation/outputs/*.json and evaluation/plots/*.png
"""
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vision_pipeline import VisionPipeline  # noqa: E402
from text_pipeline import TextPipeline  # noqa: E402
from speech_pipeline import SpeechPipeline  # noqa: E402
from fusion import MultimodalFusion  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "evaluation" / "outputs"
PLOTS_DIR = ROOT / "evaluation" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def word_error_rate(ref: str, hyp: str) -> float:
    """Standard WER via Levenshtein distance on word sequences."""
    r, h = ref.lower().split(), hyp.lower().split()
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = 1 + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    return d[len(r), len(h)] / max(1, len(r))


def evaluate_vision(vp: VisionPipeline):
    print("\n=== Vision pipeline evaluation (leave-one-out top-1/top-3) ===")
    rows = list(csv.DictReader(open(ROOT / "data" / "image_labels.csv", newline="", encoding="utf-8")))
    top1_correct, top3_correct = 0, 0
    examples = []
    for i, row in enumerate(rows):
        img = Image.open(ROOT / row["filename"]).convert("RGB")
        emb = vp.embed_image(img)
        scores, idx = vp.index.search(emb, 4)  # 4 because index[0] is the query image itself
        neighbours = [(vp.image_labels[j], float(scores[0][k])) for k, j in enumerate(idx[0]) if j != i][:3]
        true_label = row["category"]
        pred_top1 = neighbours[0][0]
        pred_top3 = [n[0] for n in neighbours]
        is_top1 = pred_top1 == true_label
        is_top3 = true_label in pred_top3
        top1_correct += is_top1
        top3_correct += is_top3
        if len(examples) < 6:
            examples.append({"file": row["filename"], "true": true_label, "pred_top1": pred_top1,
                              "correct": bool(is_top1), "top3": pred_top3})

    n = len(rows)
    metrics = {"top1_accuracy": top1_correct / n, "top3_accuracy": top3_correct / n, "n_images": n}
    print(json.dumps(metrics, indent=2))
    json.dump({"metrics": metrics, "examples": examples}, open(OUT_DIR / "vision_evaluation.json", "w"), indent=2)

    plt.figure(figsize=(4, 4))
    plt.bar(["Top-1", "Top-3"], [metrics["top1_accuracy"], metrics["top3_accuracy"]], color=["#2b6cb0", "#63b3ed"])
    plt.ylim(0, 1.15)  # headroom so value labels never collide with the title
    plt.ylabel("Accuracy")
    plt.title("Vision Pipeline: Retrieval Accuracy")
    for i, v in enumerate([metrics["top1_accuracy"], metrics["top3_accuracy"]]):
        plt.text(i, v + 0.03, f"{v:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "vision_accuracy.png", dpi=150)
    plt.close()
    return metrics


def evaluate_speech(sp: SpeechPipeline):
    print("\n=== Speech pipeline evaluation (Whisper WER) ===")
    manifest_path = ROOT / "data" / "audio_manifest.csv"
    if not manifest_path.exists():
        print("No audio manifest found, skipping speech evaluation.")
        return None
    rows = list(csv.DictReader(open(manifest_path, newline="", encoding="utf-8")))
    results = []
    total_wer = 0.0
    for row in rows:
        audio_path = ROOT / row["filename"]
        t0 = time.time()
        out = sp.transcribe(audio_path)
        dt = time.time() - t0
        wer = word_error_rate(row["transcript_ground_truth"], out["text"])
        total_wer += wer
        results.append({
            "audio_id": row["audio_id"], "ground_truth": row["transcript_ground_truth"],
            "transcribed": out["text"], "wer": round(wer, 3), "latency_sec": round(dt, 2),
        })
        print(f"  {row['audio_id']}: WER={wer:.2f}  ref={row['transcript_ground_truth']!r}  hyp={out['text']!r}")

    mean_wer = total_wer / len(rows)
    metrics = {"mean_wer": mean_wer, "n_samples": len(rows)}
    print(json.dumps(metrics, indent=2))
    json.dump({"metrics": metrics, "examples": results}, open(OUT_DIR / "speech_evaluation.json", "w"), indent=2)

    plt.figure(figsize=(7, 4))
    wers = [r["wer"] for r in results]
    plt.bar(range(len(wers)), wers, color="#805ad5")
    plt.axhline(mean_wer, color="red", linestyle="--", label=f"Mean WER = {mean_wer:.2f}")
    plt.xlabel("Sample")
    plt.ylabel("Word Error Rate")
    plt.title("Speech Pipeline: Per-sample WER (Whisper-base)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "speech_wer.png", dpi=150)
    plt.close()
    return metrics


def evaluate_text_retrieval(tp: TextPipeline):
    print("\n=== Text / retrieval pipeline evaluation ===")
    rows = list(csv.DictReader(open(ROOT / "data" / "text_queries.csv", newline="", encoding="utf-8")))
    val_rows = [r for r in rows if r["split"] == "val"]
    correct_semantic = 0
    correct_intent = 0
    for row in val_rows:
        result = tp.process(row["text"])
        top_kb_category = result["retrieval"][0][1]
        if top_kb_category == row["intent"].replace("find_gate", "gate"):
            correct_semantic += 1
        if result["intent"] and result["intent"]["intent"] == row["intent"]:
            correct_intent += 1
    metrics = {
        "semantic_retrieval_accuracy": correct_semantic / len(val_rows),
        "intent_classifier_accuracy_on_eval_run": correct_intent / len(val_rows),
        "n_val_samples": len(val_rows),
    }
    print(json.dumps(metrics, indent=2))
    json.dump(metrics, open(OUT_DIR / "text_retrieval_evaluation.json", "w"), indent=2)
    return metrics


def evaluate_fusion(fusion: MultimodalFusion):
    print("\n=== Multimodal fusion scenario testing ===")
    scenarios = []

    # 1. Text only
    r = fusion.process(text="Where is gate B12?")
    scenarios.append({"modality": "text", "input": "Where is gate B12?", "expected": "Gate B12 record",
                       "actual_status": r.status, "actual_record": r.kb_record.name if r.kb_record else None,
                       "confidence": round(r.confidence, 3), "correct": bool(r.kb_record and r.kb_record.name == "Gate B12")})

    # 2. Image only
    img_path = ROOT / "data" / "images" / "baggage_claim_00.png"
    img = Image.open(img_path).convert("RGB")
    r = fusion.process(image=img)
    scenarios.append({"modality": "image", "input": "photo of a baggage claim sign", "expected": "Baggage Claim Hall record",
                       "actual_status": r.status, "actual_record": r.kb_record.name if r.kb_record else None,
                       "confidence": round(r.confidence, 3),
                       "correct": bool(r.kb_record and r.kb_record.category == "baggage_claim")})

    # 3. Voice only
    manifest_path = ROOT / "data" / "audio_manifest.csv"
    voice_scenario_done = False
    if manifest_path.exists():
        rows = list(csv.DictReader(open(manifest_path, newline="", encoding="utf-8")))
        # Select by the manifest's own intent label (find_gate), not a
        # substring match on the transcript text — a lounge query like
        # "close to gate A11" mentions the word "gate" without being a gate
        # lookup, which previously caused a false "incorrect" scenario.
        gate_rows = [row for row in rows if row["intent"] == "find_gate"]
        if gate_rows:
            row = gate_rows[0]
            r = fusion.process(audio_path=ROOT / row["filename"])
            scenarios.append({"modality": "voice", "input": row["transcript_ground_truth"],
                               "expected": "Gate-related KB record", "actual_status": r.status,
                               "actual_record": r.kb_record.name if r.kb_record else None,
                               "confidence": round(r.confidence, 3),
                               "correct": bool(r.kb_record and r.kb_record.category == "gate")})
            voice_scenario_done = True
    if not voice_scenario_done:
        scenarios.append({"modality": "voice", "input": "(no audio available)", "expected": "N/A",
                           "actual_status": "skipped", "actual_record": None, "confidence": 0.0, "correct": None})

    # 4. Combined image + text
    img_path = ROOT / "data" / "images" / "security_01.png"
    img = Image.open(img_path).convert("RGB")
    r = fusion.process(text="Where is airport security?", image=img)
    scenarios.append({"modality": "image+text", "input": "\"Where is airport security?\" + photo of a security sign",
                       "expected": "Security Control (agreement boost)", "actual_status": r.status,
                       "actual_record": r.kb_record.name if r.kb_record else None,
                       "confidence": round(r.confidence, 3),
                       "correct": bool(r.kb_record and r.kb_record.category == "security")})

    # 5. Uncertain / out-of-scope input (flight status disclaimer)
    r = fusion.process(text="Is my flight delayed?")
    scenarios.append({"modality": "text", "input": "Is my flight delayed?",
                       "expected": "Disclaimer to check live flight boards",
                       "actual_status": r.status, "actual_record": None, "confidence": round(r.confidence, 3),
                       "correct": r.status == "flight_status_disclaimer"})

    print(json.dumps(scenarios, indent=2))
    json.dump(scenarios, open(OUT_DIR / "fusion_scenarios.json", "w"), indent=2)
    return scenarios


def main():
    print("Loading pipelines (this downloads CLIP/Whisper on first run)...")
    vp = VisionPipeline()
    tp = TextPipeline()
    sp = SpeechPipeline()
    fusion = MultimodalFusion(vp, tp, sp)

    vision_metrics = evaluate_vision(vp)
    speech_metrics = evaluate_speech(sp)
    text_metrics = evaluate_text_retrieval(tp)
    fusion_scenarios = evaluate_fusion(fusion)

    summary = {
        "vision": vision_metrics,
        "speech": speech_metrics,
        "text_retrieval": text_metrics,
        "fusion_scenarios_pass_rate": sum(1 for s in fusion_scenarios if s["correct"]) / len([s for s in fusion_scenarios if s["correct"] is not None]),
    }
    json.dump(summary, open(OUT_DIR / "evaluation_summary.json", "w"), indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
