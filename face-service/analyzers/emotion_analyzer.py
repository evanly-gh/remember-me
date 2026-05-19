"""
EmotionAnalyzer — HSEmotion 8-class facial emotion recognition.

Model
-----
- Architecture : EfficientNet-B0
- Checkpoint   : enet_b0_8_best_afew (Savchenko et al.)
                 published by the hsemotion PyPI package
- Classes (8)  : anger, contempt, disgust, fear, happiness,
                 neutral, sadness, surprise
- License      : Apache 2.0 (hsemotion package)
- Source       : https://github.com/HSE-asavchenko/face-emotion-recognition

Inputs
------
img_rgb : np.ndarray (H, W, 3) uint8. HSEmotionRecognizer handles its
own resize/normalise internally.

Outputs (dict)
--------------
primary_emotion, emotion_confidence, secondary_emotion,
emotion_scores (full distribution), valence (-1..+1), arousal (0..1),
mood (positive | negative | neutral).

Notes
-----
Valence and arousal are derived from the emotion distribution using
hand-set per-emotion weights (VALENCE_MAP / AROUSAL_MAP) — they are
weighted sums, not separate model outputs.

PyTorch 2.6 changed torch.load to weights_only=True by default. The
HSEmotion checkpoint is pickled as a full timm EfficientNet object
(not a clean state dict), so the safe unpickler refuses to load it.
We scope a legacy weights_only=False just around the HSEmotion init
to keep the rest of the process on the safer default.
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

# Per-emotion valence weights. Used to project the 8-class distribution
# down to a single scalar in [-1, 1] (negative = sad/angry, positive = happy).
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

# Per-emotion arousal weights, scalar in [0, 1] (0 = calm, 1 = intense).
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
    """Temporarily switch torch.load back to weights_only=False.

    Scoped via a context manager so only the HSEmotion init runs with
    the legacy default; everything else keeps PyTorch 2.6's safer
    weights_only=True behaviour.
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
        # Without the hsemotion package installed there's no model to
        # load. We log once and the rest of the service still works —
        # the emotion fields just stay "unknown".
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
            # The recognizer handles its own resize/normalize/preproc,
            # so we hand it the raw RGB ndarray.
            _, scores = self.recognizer.predict_emotions(img_rgb, logits=False)
        except Exception as exc:
            print(f"[EmotionAnalyzer] Inference failed: {exc}")
            return self._empty_result()

        # Flatten to a 1D numpy array and sanity-check its length matches
        # the class list. Mismatch likely means the upstream package
        # changed its class count.
        probs = np.asarray(scores, dtype=float).flatten()
        if probs.size != len(EMOTION_LABELS):
            print(
                f"[EmotionAnalyzer] Unexpected score length: {probs.size} "
                f"(expected {len(EMOTION_LABELS)}). Check that "
                f"{HSEMOTION_MODEL_NAME} still produces 8 classes in this order."
            )
            return self._empty_result()

        # Defensive renormalisation. With logits=False this is a no-op,
        # but it guards against future API drift in the hsemotion package.
        total = probs.sum()
        if total > 0:
            probs = probs / total

        # Build the {emotion: probability} dict for downstream display.
        emotion_scores = {
            label: round(float(probs[i]), 3)
            for i, label in enumerate(EMOTION_LABELS)
        }

        # Primary = argmax of the distribution; secondary = second-highest.
        # These are the two most-likely emotions, useful when the model
        # is genuinely uncertain between two similar classes.
        primary_idx = int(np.argmax(probs))
        primary_emotion = EMOTION_LABELS[primary_idx]
        primary_confidence = float(probs[primary_idx])

        sorted_idx = np.argsort(probs)[::-1]
        secondary_emotion = EMOTION_LABELS[int(sorted_idx[1])]

        # Valence and arousal: weighted sums over the distribution. A
        # confidently-happy face gives valence ~0.9; a fearful one drops
        # into negative territory with high arousal.
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
        """Stub used when HSEmotion isn't available or inference fails."""
        return {
            "primary_emotion": "unknown",
            "emotion_confidence": 0.0,
            "secondary_emotion": "unknown",
            "emotion_scores": {label: 0.0 for label in EMOTION_LABELS},
            "valence": 0.0,
            "arousal": 0.0,
            "mood": "unknown",
        }
