"""
Synthesises voice queries with TTS instead of recording real people, which
would raise consent/privacy issues. Shells out to _tts_worker.py once per
query (offline Windows SAPI5 engine via pyttsx3 — no internet, no API cost)
with randomised speaking rate and voice to emulate speaker variation.

Run: python data/generate_audio.py
Output: data/audio/<id>.wav  (+ data/audio_manifest.csv)
"""
import csv
import random
import subprocess
import sys
from pathlib import Path

import soundfile as sf

random.seed(11)

N_SAMPLES = 20
HERE = Path(__file__).resolve().parent


def resample_to_16k(path):
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        sr = 16000
    sf.write(path, data, sr, subtype="PCM_16")


def main():
    with open(HERE / "text_queries.csv", newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    sample = random.sample(all_rows, N_SAMPLES)

    rows = []
    for i, row in enumerate(sample):
        out_path = HERE / "audio" / f"{row['id']}.wav"
        rate = random.randint(150, 190)
        voice_idx = i % 2
        print(f"[{i+1}/{len(sample)}] synthesising {row['id']}: {row['text'][:50]!r}")
        result = subprocess.run(
            [sys.executable, str(HERE / "_tts_worker.py"), row["text"], str(out_path), str(rate), str(voice_idx)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0 or not out_path.exists():
            print(f"  FAILED: {result.stderr[:300]}")
            continue
        try:
            resample_to_16k(out_path)
        except Exception as e:
            print(f"  resample warning: {e}")
        rows.append({
            "audio_id": row["id"],
            "filename": f"data/audio/{row['id']}.wav",
            "transcript_ground_truth": row["text"],
            "intent": row["intent"],
            "tts_rate_wpm": rate,
        })

    with open(HERE / "audio_manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_id", "filename", "transcript_ground_truth", "intent", "tts_rate_wpm"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)}/{len(sample)} audio queries -> data/audio_manifest.csv")


if __name__ == "__main__":
    main()
