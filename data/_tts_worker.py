"""Single-shot TTS worker: synthesises ONE utterance per process invocation.
pyttsx3's SAPI5 backend on Windows is unreliable when an engine instance is
reused for many save_to_file()/runAndWait() calls in a loop (it silently
hangs after the first couple of calls); running one clean process per
utterance avoids that entirely at the cost of process-startup overhead,
which is acceptable for a ~20-file dataset.
"""
import sys
import pyttsx3

text, out_path, rate, voice_idx = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
engine = pyttsx3.init()
voices = engine.getProperty("voices")
if voices:
    engine.setProperty("voice", voices[voice_idx % len(voices)].id)
engine.setProperty("rate", rate)
engine.save_to_file(text, out_path)
engine.runAndWait()
