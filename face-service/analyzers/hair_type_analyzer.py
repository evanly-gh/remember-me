"""
HairTypeAnalyzer — hair texture classifier.

Model
-----
- Architecture : Vision Transformer (ViT-B/16)
- HF repo      : dima806/hair_type_image_detection
- License      : Apache 2.0
- Classes (5)  : curly, dreadlocks, kinky, straight, wavy
- Reported acc : 93% overall.
                 Per-class F1: dreadlocks 0.978, kinky 0.949,
                 straight 0.927, curly 0.902, wavy 0.884.

Inputs
------
img_rgb : np.ndarray (H, W, 3) uint8

Outputs (dict)
--------------
hair_type            — argmax label
hair_type_confidence — argmax softmax score
hair_type_scores     — full {class: score} dict

Notes
-----
This is the authoritative hair-texture output. The Laplacian-std-
based `hair_texture` field from ColorAnalyzer is a coarse fallback
that runs even when this model is unavailable.
"""

from typing import Any

from PIL import Image
from transformers import pipeline


MODEL_ID = "dima806/hair_type_image_detection"

# Canonical class names in lowercase. Pipeline output is normalised
# to these on the way out.
_KNOWN = {"curly", "dreadlocks", "kinky", "straight", "wavy"}


class HairTypeAnalyzer:
    def __init__(self):
        self.classifier = None
        try:
            self.classifier = pipeline("image-classification", model=MODEL_ID)
        except Exception as exc:
            print(f"[HairTypeAnalyzer] Failed to load {MODEL_ID}: {exc}")

    def analyze(self, img_rgb) -> dict[str, Any]:
        if self.classifier is None:
            return self._empty_result()

        try:
            pil = Image.fromarray(img_rgb)
            # Pull all five class probabilities so downstream code can
            # inspect the full distribution (e.g. wavy-vs-curly margin).
            preds = self.classifier(pil, top_k=len(_KNOWN))
        except Exception as exc:
            print(f"[HairTypeAnalyzer] Prediction failed: {exc}")
            return self._empty_result()

        # Normalise label casing and build the score map.
        scores = {label: 0.0 for label in _KNOWN}
        for pred in preds:
            label = str(pred["label"]).strip().lower()
            if label in scores:
                scores[label] = round(float(pred["score"]), 3)

        top_label = max(scores, key=scores.get)
        top_score = scores[top_label]

        return {
            "hair_type": top_label,
            "hair_type_confidence": top_score,
            "hair_type_scores": scores,
        }

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        return {
            "hair_type": "unknown",
            "hair_type_confidence": 0.0,
            "hair_type_scores": {label: 0.0 for label in _KNOWN},
        }
