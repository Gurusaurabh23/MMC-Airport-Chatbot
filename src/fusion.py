"""
Combines text/image/voice inputs into one grounded answer.

Uses rule-based routing with a weighted-confidence agreement check, rather
than a learned fusion layer — the dataset (~100 images, ~85 text queries) is
too small to train a fusion network without overfitting, and a rule set is
easy to explain and debug when a route picks the "wrong" modality.

Handles text-only, image-only, voice-only (via speech_pipeline ->
text_pipeline), image+text and voice+image.
"""
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

INTENT_TO_CATEGORY = {
    "find_gate": "gate",
    "baggage_claim": "baggage_claim",
    "check_in": "check_in",
    "security": "security",
    "lounge": "lounge",
    "restaurant": "restaurant",
    "transport": "transport",
    "information_desk": "information_desk",
    "lost_and_found": "lost_and_found",
    "prayer_room": "prayer_room",
    "accessibility": "accessibility",
    "shopping": "shopping",
    "medical": "medical",
    # flight_status has no static KB category — handled as a special case
    # below to avoid presenting stale/incorrect live flight data as fact.
}

CONFIDENCE_UNCERTAIN_THRESHOLD = 0.35
AGREEMENT_BONUS = 0.15


@dataclass
class FusionResult:
    status: str  # "answered" | "uncertain" | "flight_status_disclaimer"
    kb_record: Optional[object] = None
    confidence: float = 0.0
    source: str = ""  # "text" | "image" | "text+image" | "voice" | "voice+image"
    message: str = ""
    debug: dict = field(default_factory=dict)


class MultimodalFusion:
    def __init__(self, vision_pipeline, text_pipeline, speech_pipeline=None):
        self.vision = vision_pipeline
        self.text = text_pipeline
        self.speech = speech_pipeline

    def _text_route(self, text: str) -> dict:
        result = self.text.process(text)
        entities = result["entities"]

        # Gate lookups are entity-driven, not intent-driven (a specific gate
        # code overrides a generic "find_gate" category match).
        if "gate" in entities:
            record = self.text.kb.get_by_gate(entities["gate"])
            if record:
                return {"record": record, "confidence": 0.95, "via": "entity:gate"}

        intent_info = result["intent"]
        if intent_info and intent_info["intent"] == "flight_status":
            return {"record": None, "confidence": intent_info["confidence"], "via": "flight_status"}

        if intent_info and intent_info["confidence"] >= CONFIDENCE_UNCERTAIN_THRESHOLD:
            category = INTENT_TO_CATEGORY.get(intent_info["intent"])
            if category:
                record = self.text.kb.get_by_category(category, terminal=entities.get("terminal"))
                if record:
                    return {"record": record, "confidence": intent_info["confidence"], "via": "intent"}

        # fallback: semantic retrieval directly against KB descriptions
        top_record, score = result["retrieval"][0][0], result["retrieval"][0][2]
        record = self.text.kb.get_by_id(top_record)
        return {"record": record, "confidence": score, "via": "semantic_retrieval"}

    def _image_route(self, image: Image.Image) -> dict:
        pred = self.vision.predict(image)
        record = self.text.kb.get_by_category(pred["category"])
        return {"record": record, "confidence": pred["confidence"], "via": "vision", "category": pred["category"]}

    def process(self, text: Optional[str] = None, image: Optional[Image.Image] = None,
                audio_path: Optional[str] = None) -> FusionResult:
        source_parts = []
        transcript = None

        if audio_path is not None:
            if self.speech is None:
                raise RuntimeError("SpeechPipeline not provided but audio_path was given")
            transcript = self.speech.transcribe(audio_path)["text"]
            text = transcript if not text else f"{text} {transcript}"
            source_parts.append("voice")
        elif text:
            source_parts.append("text")

        if image is not None:
            source_parts.append("image")

        text_result = self._text_route(text) if text else None
        image_result = self._image_route(image) if image is not None else None

        if text_result and text_result.get("via") == "flight_status":
            return FusionResult(
                status="flight_status_disclaimer",
                confidence=text_result["confidence"],
                source="+".join(source_parts),
                message=(
                    "I can't provide live flight delay or gate-change status — that information "
                    "changes in real time and must come from the official airline or airport "
                    "display boards. Please check the nearest Flight Information Display Board or "
                    "ask airline staff at the check-in desk for your current flight status."
                ),
                debug={"transcript": transcript},
            )

        # --- fusion decision ---
        if text_result and image_result:
            if text_result["record"] and image_result["record"] and \
               text_result["record"].category == image_result["record"].category:
                # both modalities agree -> boosted confidence, weighted average
                combined_conf = min(1.0, 0.5 * text_result["confidence"] + 0.5 * image_result["confidence"] + AGREEMENT_BONUS)
                chosen = text_result["record"]
            else:
                # disagreement -> trust the higher-confidence modality
                chosen_result = text_result if text_result["confidence"] >= image_result["confidence"] else image_result
                chosen = chosen_result["record"]
                combined_conf = chosen_result["confidence"]
            best = {"record": chosen, "confidence": combined_conf}
        elif text_result:
            best = text_result
        elif image_result:
            best = image_result
        else:
            return FusionResult(status="uncertain", source="+".join(source_parts),
                                 message="No input received.", debug={})

        if best["record"] is None or best["confidence"] < CONFIDENCE_UNCERTAIN_THRESHOLD:
            return FusionResult(
                status="uncertain",
                confidence=best["confidence"],
                source="+".join(source_parts),
                message=(
                    "I'm not confident I understood that correctly. Could you rephrase, upload a "
                    "clearer photo of the sign, or ask a member of airport staff at the nearest "
                    "information desk?"
                ),
                debug={"transcript": transcript, "text_result": text_result, "image_result": image_result},
            )

        return FusionResult(
            status="answered",
            kb_record=best["record"],
            confidence=best["confidence"],
            source="+".join(source_parts),
            message=best["record"].to_response_text(),
            debug={"transcript": transcript, "text_result": text_result, "image_result": image_result},
        )
