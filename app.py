"""
Streamlit front end for the airport passenger assistance chatbot.
Chosen over Flask + hand-rolled HTML/JS for the multi-input UI (image
upload, mic/audio input, text box, response panel, confidence score) with
much less boilerplate.

Run:  streamlit run app.py
"""
import io
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from vision_pipeline import VisionPipeline  # noqa: E402
from text_pipeline import TextPipeline  # noqa: E402
from speech_pipeline import SpeechPipeline  # noqa: E402
from fusion import MultimodalFusion  # noqa: E402

st.set_page_config(page_title="Airport Passenger Assistant", page_icon="✈️", layout="centered")


@st.cache_resource(show_spinner="Loading multimodal AI pipelines (CLIP, DistilBERT, Whisper)...")
def load_pipelines():
    vp = VisionPipeline()
    tp = TextPipeline()
    sp = SpeechPipeline()
    return MultimodalFusion(vp, tp, sp)


fusion = load_pipelines()

st.title("✈️ Smart Airport Passenger Assistant")
st.caption(
    "Proof-of-concept multimodal chatbot (MSc AI — Multi-Modal Chatbots assignment). "
    "Ask about gates, baggage claim, check-in, security, transport and more — by text, "
    "voice, a photo of a sign, or any combination."
)

with st.expander("ℹ️ About this prototype / limitations", expanded=False):
    st.markdown(
        "- This system uses a small, **self-curated synthetic** dataset of airport sign "
        "mockups and template-generated passenger queries — it is a proof of concept, not "
        "a production system connected to live airport data.\n"
        "- It **cannot** provide real-time flight delay or gate-change information; for "
        "that, always check the official flight information boards or airline staff.\n"
        "- If the system is not confident in an answer, it will say so rather than guess."
    )

col1, col2 = st.columns(2)
with col1:
    uploaded_image = st.file_uploader("📷 Upload a photo of an airport sign", type=["png", "jpg", "jpeg"])
with col2:
    audio_value = st.audio_input("🎙️ Record a voice question")
    uploaded_audio = st.file_uploader("or upload an audio file", type=["wav", "mp3", "m4a"])

text_query = st.text_input("💬 Or type your question", placeholder="e.g. Where is gate B12?")

if uploaded_image:
    st.image(uploaded_image, caption="Uploaded sign", width=220)

submit = st.button("Ask the assistant", type="primary")

if submit:
    image = None
    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")

    audio_path = None
    audio_source = audio_value or uploaded_audio
    if audio_source is not None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(audio_source.getvalue())
        tmp.close()
        audio_path = Path(tmp.name)

    if not text_query and image is None and audio_path is None:
        st.warning("Please provide at least one input: text, an image, or a voice recording.")
    else:
        with st.spinner("Thinking..."):
            result = fusion.process(
                text=text_query if text_query else None,
                image=image,
                audio_path=audio_path,
            )

        st.divider()
        if result.debug.get("transcript"):
            st.markdown(f"**🎙️ Transcribed:** _{result.debug['transcript']}_")

        if result.status == "answered":
            st.success("Here's what I found:")
            st.markdown(result.message)
            st.progress(min(1.0, result.confidence), text=f"Confidence: {result.confidence:.0%}")
        elif result.status == "flight_status_disclaimer":
            st.info(result.message)
        else:  # uncertain
            st.error(result.message)
            st.progress(min(1.0, max(0.0, result.confidence)), text=f"Confidence: {result.confidence:.0%} (below threshold)")

        with st.expander("🔍 Debug: routing details"):
            st.json({
                "source": result.source,
                "status": result.status,
                "confidence": result.confidence,
                "text_route": result.debug.get("text_result"),
                "image_route": {
                    k: v for k, v in (result.debug.get("image_result") or {}).items() if k != "record"
                } if result.debug.get("image_result") else None,
            })

st.divider()
st.caption("Proof-of-concept only — not for real passenger use.")
