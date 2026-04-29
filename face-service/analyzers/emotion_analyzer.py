"""
HSEmotion — EfficientNet-B0 fine-tuned for 8-class emotion recognition.

Uses the published HSEmotion checkpoint (Savchenko et al., enet_b0_8_best_afew),
which has actual fine-tuned weights for the 8 emotion classes. The previous
version asked timm for a 1000-class ImageNet checkpoint and reset the head to
8 randomly-initialized neurons, so the outputs were softmax-over-noise.

Classes: Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise.

Also provides valence (positive/negative) and arousal (calm/excited) scores
derived from the emotion distribution.

Install: pip install hsemotion
"""

from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

try:
    from hsemotion.facial_emotions import HSEmotionRecognizer
    HAS_HSEMOTION = True
except ImportError:
    HAS_HSEMOTION = False

# Order MUST match the class order produced by enet_b0_8_best_afew.
# HSEmotion's 8-class AffectNet/AFEW models use alphabetical order.
EMOTION_LABELS = [
    "anger", "contempt", "disgust", "fear",
    "happiness", "neutral", "sadness", "surprise",
]

# Valence weights for each emotion (-1 to +1)
VALENCE_MAP = {
    "anger": -0.6,
    "contempt": -0.3,
    "disgust": -0.7,
    "fear": -0.6,
    "happiness": 0.9,
    "neutral": 0.0,
    "sadness": -0.7,
    "surprise": 0.3,
}

# Arousal weights for each emotion (0 to 1)
AROUSAL_MAP = {
    "anger": 0.8,
    "contempt": 0.3,
    "disgust": 0.5,
    "fear": 0.9,
    "happiness": 0.7,
    "neutral": 0.1,
    "sadness": 0.3,
    "surprise": 0.9,
}

HSEMOTION_MODEL_NAME = "enet_b0_8_best_afew"


@contextmanager
def _legacy_torch_load():
    """Temporarily make torch.load default to weights_only=False.

    PyTorch 2.6 changed the default to weights_only=True. The HSEmotion
    checkpoint is pickled as a full timm.models.efficientnet.EfficientNet
    object (not a clean state dict), so the safe unpickler refuses to
    deserialize it. We trust this checkpoint (it comes from the published
    HSEmotion repo and was already vetted by the pip install), so we opt
    back into legacy loading — scoped to just the HSEmotion init so the
    rest of the process keeps the safer default.
    """
    original_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = _patched_load
    try:
        yield
    finally:
        torch.load = original_load


class EmotionAnalyzer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.recognizer = self._load_model()

    def _load_model(self):
        if not HAS_HSEMOTION:
            print(
                "[EmotionAnalyzer] hsemotion not installed — emotion outputs "
                "will be 'unknown'. Install with: pip install hsemotion"
            )
            return None

        try:
            with _legacy_torch_load():
                return HSEmotionRecognizer(
                    model_name=HSEMOTION_MODEL_NAME,
                    device=self.device,
                )
        except Exception as exc:
            print(f"[EmotionAnalyzer] Could not load HSEmotion: {exc}")
            return None

    def analyze(self, img_rgb: np.ndarray) -> dict[str, Any]:
        if self.recognizer is None:
            return self._empty_result()

        try:
            # logits=False → returns post-softmax probabilities.
            # HSEmotionRecognizer handles its own resize/normalize/preproc.
            _, scores = self.recognizer.predict_emotions(img_rgb, logits=False)
        except Exception as exc:
            print(f"[EmotionAnalyzer] Inference failed: {exc}")
            return self._empty_result()

        probs = np.asarray(scores, dtype=float).flatten()
        if probs.size != len(EMOTION_LABELS):
            print(
                f"[EmotionAnalyzer] Unexpected score length: {probs.size} "
                f"(expected {len(EMOTION_LABELS)}). Check that "
                f"{HSEMOTION_MODEL_NAME} still produces 8 classes in this order."
            )
            return self._empty_result()

        # Defensive renormalization. With logits=False this is a no-op, but
        # guards against future API drift in the hsemotion package.
        total = probs.sum()
        if total > 0:
            probs = probs / total

        emotion_scores = {
            label: round(float(probs[i]), 3)
            for i, label in enumerate(EMOTION_LABELS)
        }

        primary_idx = int(np.argmax(probs))
        primary_emotion = EMOTION_LABELS[primary_idx]
        primary_confidence = float(probs[primary_idx])

        # Secondary emotion (second highest)
        sorted_idx = np.argsort(probs)[::-1]
        secondary_emotion = EMOTION_LABELS[int(sorted_idx[1])]

        # Calculate valence and arousal
        valence = sum(
            probs[i] * VALENCE_MAP[label]
            for i, label in enumerate(EMOTION_LABELS)
        )
        arousal = sum(
            probs[i] * AROUSAL_MAP[label]
            for i, label in enumerate(EMOTION_LABELS)
        )

        return {
            "primary_emotion": primary_emotion,
            "emotion_confidence": round(primary_confidence, 3),
            "secondary_emotion": secondary_emotion,
            "emotion_scores": emotion_scores,
            "valence": round(float(valence), 3),
            "arousal": round(float(arousal), 3),
            "mood": (
                "positive" if valence > 0.2
                else "negative" if valence < -0.2
                else "neutral"
            ),
        }

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        return {
            "primary_emotion": "unknown",
            "emotion_confidence": 0.0,
            "secondary_emotion": "unknown",
            "emotion_scores": {label: 0.0 for label in EMOTION_LABELS},
            "valence": 0.0,
            "arousal": 0.0,
            "mood": "unknown",
        }