"""
EthnicityAnalyzer — single-model ethnicity classifier.

Model
-----
- HF repo  : cledoux42/Ethnicity_Test_v003
- Arch     : Vision Transformer
- Classes  : african, asian, caucasian, hispanic, indian (5)
- Reported : 79.6% accuracy, macro-F1 0.797
- License  : Apache 2.0
- Source   : https://huggingface.co/cledoux42/Ethnicity_Test_v003

Inputs
------
img_rgb : np.ndarray (H, W, 3) uint8

Outputs (dict)
--------------
ethnicity              : canonical label (legacy 7-bucket schema)
ethnicity_confidence   : argmax softmax score
ethnicity_distribution : full {label: prob} dict, padded to all 7 buckets

Notes
-----
Split out from the old DemographicAnalyzer (which also handled age and
gender via FairFace ViTs). Age and gender now live in
InsightFaceAnalyzer; this file owns ethnicity exclusively.

The model emits 5 classes but we widen to the legacy 7-bucket FairFace
schema so the rest of the app's distribution shape stays stable.
Unseen buckets stay at 0.0.
"""

from typing import Any

from PIL import Image
from transformers import pipeline


MODEL_ID = "cledoux42/Ethnicity_Test_v003"

# Legacy schema preserved from the old DemographicAnalyzer.
RACE_LABELS = [
    "White", "Black", "Latino_Hispanic", "East Asian",
    "Southeast Asian", "Indian", "Middle Eastern",
]


class EthnicityAnalyzer:
    def __init__(self):
        self.classifier = None
        try:
            self.classifier = pipeline("image-classification", model=MODEL_ID)
        except Exception as exc:
            print(f"[EthnicityAnalyzer] Failed to load {MODEL_ID}: {exc}")

    def analyze(self, img_rgb) -> dict[str, Any]:
        if self.classifier is None:
            return self._empty_result()

        try:
            pil = Image.fromarray(img_rgb)
            preds = self.classifier(pil, top_k=7)
        except Exception as exc:
            print(f"[EthnicityAnalyzer] Prediction failed: {exc}")
            return self._empty_result()

        if not preds:
            return self._empty_result()

        # Top prediction → canonical label.
        top = preds[0]
        top_label = self._normalize(top["label"])

        distribution = {label: 0.0 for label in RACE_LABELS}
        for pred in preds:
            label = self._normalize(pred["label"])
            if label in distribution:
                distribution[label] = round(float(pred["score"]), 3)

        return {
            "ethnicity": top_label,
            "ethnicity_confidence": round(float(top["score"]), 3),
            "ethnicity_distribution": distribution,
        }

    @staticmethod
    def _normalize(label: str) -> str:
        """Map model output (5-class) to canonical 7-bucket label."""
        normalized = label.strip().lower().replace("-", "_")
        aliases = {
            # Legacy FairFace 7-class
            "white": "White",
            "black": "Black",
            "latino_hispanic": "Latino_Hispanic",
            "latino hispanic": "Latino_Hispanic",
            "east asian": "East Asian",
            "southeast asian": "Southeast Asian",
            "indian": "Indian",
            "middle eastern": "Middle Eastern",
            # cledoux42 5-class → widen into 7-bucket schema
            "african": "Black",
            "asian": "East Asian",
            "caucasian": "White",
            "hispanic": "Latino_Hispanic",
        }
        return aliases.get(normalized, label)

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        return {
            "ethnicity": "unknown",
            "ethnicity_confidence": 0.0,
            "ethnicity_distribution": {label: 0.0 for label in RACE_LABELS},
        }
