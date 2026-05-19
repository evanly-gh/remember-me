"""
ObstructionAnalyzer — face obstruction classifier.

Model
-----
- Architecture : Vision Transformer (ViT-B/16)
- HF repo      : dima806/face_obstruction_image_detection
- License      : Apache 2.0
- Classes (6)  : sunglasses, glasses, mask, hand, other, none
- Reported acc : ~91% overall.
                 99.7% / 99.85% precision/recall on sunglasses
                 99.0% / 99.7%  precision/recall on glasses
                 99.7% / 99.85% precision/recall on mask
                 Hand and "other" are much weaker (~71-75%); we don't
                 surface those as booleans.

Inputs
------
img_rgb : np.ndarray (H, W, 3) uint8

Outputs (dict)
--------------
obstruction_top        — argmax label
obstruction_confidence — argmax softmax score
obstruction_scores     — full {class: score} dict
wearing_glasses        — bool (true when glasses OR sunglasses > 0.5)
wearing_sunglasses     — bool
wearing_mask           — bool

Notes
-----
Same author as the FairFace age/gender models already in
DemographicAnalyzer. Built specifically for the glasses/sunglasses/mask
case, which is why precision/recall on those three classes is so high.
"""

from typing import Any

from PIL import Image
from transformers import pipeline


MODEL_ID = "dima806/face_obstruction_image_detection"

# Canonical labels in lowercase. The pipeline may return any casing —
# we normalise on the way out so downstream code keys consistently.
_KNOWN = {"sunglasses", "glasses", "mask", "hand", "other", "none"}


class ObstructionAnalyzer:
    def __init__(self):
        self.classifier = None
        try:
            # HF image-classification pipeline. Weights lazy-load from
            # the Hub on first instantiation and cache locally.
            self.classifier = pipeline("image-classification", model=MODEL_ID)
        except Exception as exc:
            print(f"[ObstructionAnalyzer] Failed to load {MODEL_ID}: {exc}")

    def analyze(self, img_rgb) -> dict[str, Any]:
        # Empty stub when the model failed to load — keeps the result
        # dict shape stable so the merge in app.py never sees missing keys.
        if self.classifier is None:
            return self._empty_result()

        try:
            pil = Image.fromarray(img_rgb)
            # top_k=len(_KNOWN) → full softmax across all six classes.
            preds = self.classifier(pil, top_k=len(_KNOWN))
        except Exception as exc:
            print(f"[ObstructionAnalyzer] Prediction failed: {exc}")
            return self._empty_result()

        # Flatten predictions into a {label: score} dict, normalising
        # label casing as we go. Unseen labels stay at 0.
        scores = {label: 0.0 for label in _KNOWN}
        for pred in preds:
            label = str(pred["label"]).strip().lower()
            if label in scores:
                scores[label] = round(float(pred["score"]), 3)

        # Top class wins.
        top_label = max(scores, key=scores.get)
        top_score = scores[top_label]

        return {
            "obstruction_top": top_label,
            "obstruction_confidence": top_score,
            "obstruction_scores": scores,
            # Specific boolean flags the UI consumes directly.
            # `wearing_glasses` is True for any kind of eyewear — the
            # caller can branch on `wearing_sunglasses` if it cares
            # about tinted vs clear lenses.
            "wearing_glasses": scores["glasses"] > 0.5 or scores["sunglasses"] > 0.5,
            "wearing_sunglasses": scores["sunglasses"] > 0.5,
            "wearing_mask": scores["mask"] > 0.5,
        }

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        return {
            "obstruction_top": "unknown",
            "obstruction_confidence": 0.0,
            "obstruction_scores": {label: 0.0 for label in _KNOWN},
            "wearing_glasses": False,
            "wearing_sunglasses": False,
            "wearing_mask": False,
        }
