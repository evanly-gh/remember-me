"""
AgeAnalyzer — FairFace age classifier with softmax-weighted estimate.

Model
-----
- HF repo  : dima806/fairface_age_image_detection
- Arch     : Vision Transformer (ViT-B/16)
- Trained  : FairFace dataset (race-balanced)
- Reported : ~59% top-1 accuracy across 9 age buckets
- License  : Apache 2.0

Why this and not InsightFace's bundled genderage head
-----------------------------------------------------
InsightFace's age regression head systematically over-predicts for
certain face types — strong jaw, brow ridge, beard shadow, or just
poor lighting can make it call a 20-year-old "52". Piecewise
calibration helps with mild overshoot but can't recover when the
raw prediction is already 50+ years off.

FairFace uses softmax classification across 9 age buckets. Even when
wrong it's wrong by ~5-10 years, not 30+. We take the softmax-weighted
expected value across all buckets to get a smooth continuous number
that moves with confidence — rather than always snapping to a fixed
bucket midpoint.

Inputs
------
img_rgb : np.ndarray (H, W, 3) uint8. Typically a face crop produced
          by `_crop_to_face` in app.py.

Outputs (dict)
--------------
age_estimate     : softmax-weighted expected age (float, years)
age_range        : argmax bucket as a string (e.g. "20-29")
age_confidence   : argmax softmax score
age_distribution : full {bucket: prob} dict over all 9 buckets
"""

from typing import Any

from PIL import Image
from transformers import pipeline


MODEL_ID = "dima806/fairface_age_image_detection"

AGE_LABELS = [
    "0-2", "3-9", "10-19", "20-29", "30-39",
    "40-49", "50-59", "60-69", "70+",
]

# Midpoint per bucket; used to compute the softmax-weighted expected
# age. The 70+ bucket midpoint is a guess — there's no upper bound in
# the FairFace label space.
AGE_MIDPOINTS = {
    "0-2": 1.0,
    "3-9": 6.0,
    "10-19": 14.5,
    "20-29": 24.5,
    "30-39": 34.5,
    "40-49": 44.5,
    "50-59": 54.5,
    "60-69": 64.5,
    "70+": 75.0,
}


class AgeAnalyzer:
    def __init__(self):
        self.classifier = None
        try:
            self.classifier = pipeline("image-classification", model=MODEL_ID)
        except Exception as exc:
            print(f"[AgeAnalyzer] Failed to load {MODEL_ID}: {exc}")

    def analyze(self, img_rgb) -> dict[str, Any]:
        if self.classifier is None:
            return self._empty_result()

        try:
            pil = Image.fromarray(img_rgb)
            # Pull all 9 buckets so we can compute the weighted estimate.
            preds = self.classifier(pil, top_k=len(AGE_LABELS))
        except Exception as exc:
            print(f"[AgeAnalyzer] Prediction failed: {exc}")
            return self._empty_result()

        if not preds:
            return self._empty_result()

        # Normalise label casing and build the {bucket: prob} dict.
        distribution = {label: 0.0 for label in AGE_LABELS}
        for pred in preds:
            label = self._normalize_label(pred["label"])
            if label in distribution:
                distribution[label] = round(float(pred["score"]), 3)

        # Softmax-weighted expected age. Sum over (midpoint × prob).
        # Lets the number slide between buckets when the model is
        # uncertain — e.g. 80% confident 20-29, 20% 30-39 → ~26.5
        # instead of snapping to either bucket's midpoint.
        total_weight = sum(distribution.values()) or 1.0
        weighted_age = sum(
            AGE_MIDPOINTS[label] * prob
            for label, prob in distribution.items()
        ) / total_weight

        # Argmax bucket = the model's top guess; report that as
        # `age_range` for legacy UI compatibility.
        top = max(distribution.items(), key=lambda kv: kv[1])
        top_label, top_score = top

        return {
            "age_estimate": round(float(weighted_age), 1),
            "age_range": top_label,
            "age_confidence": round(float(top_score), 3),
            "age_distribution": distribution,
        }

    @staticmethod
    def _normalize_label(label: str) -> str:
        """Map model output to canonical AGE_LABELS entry."""
        normalized = label.strip().lower()
        if normalized == "more than 70":
            return "70+"
        return label if label in AGE_LABELS else label.strip()

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        return {
            "age_estimate": 0.0,
            "age_range": "unknown",
            "age_confidence": 0.0,
            "age_distribution": {label: 0.0 for label in AGE_LABELS},
        }
