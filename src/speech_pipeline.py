"""
Whisper-based speech-to-text, used frozen (no fine-tuning).

Audio is resampled to 16kHz mono with librosa and handed to Whisper as a
numpy array instead of a file path, so this doesn't depend on ffmpeg being
installed — Whisper's default loader shells out to it, but librosa/soundfile
already cover WAV decoding.
"""
from pathlib import Path

import librosa
import numpy as np
import whisper

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
WHISPER_MODEL_SIZE = "base"


class SpeechPipeline:
    def __init__(self, model_size: str = WHISPER_MODEL_SIZE):
        self.model = whisper.load_model(model_size)

    def load_audio_array(self, path: Path) -> np.ndarray:
        audio, sr = librosa.load(str(path), sr=16000, mono=True)
        return audio.astype(np.float32)

    def transcribe(self, path: Path) -> dict:
        audio = self.load_audio_array(path)
        result = self.model.transcribe(audio, language="en", fp16=False)
        return {
            "text": result["text"].strip(),
            "segments": result.get("segments", []),
            "language": result.get("language", "en"),
        }


if __name__ == "__main__":
    sp = SpeechPipeline()
    sample = DATA_DIR / "audio"
    files = sorted(sample.glob("*.wav"))[:2]
    for f in files:
        out = sp.transcribe(f)
        print(f.name, "->", out["text"])
