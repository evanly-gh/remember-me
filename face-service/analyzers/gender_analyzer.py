"""
GenderAnalyzer — FairFace ViT for binary gender.

Model
-----
- HF repo  : dima806/fairface_gender_image_detection
- Arch     : Vision Transformer (ViT-B/16), 86M params
- Trained  : FairFace dataset (gender-balanced)
- Reported : 93.4% accuracy
- License  : Apache 2.0
- Source   : https://huggingface.co/dima806/fairface_gender_image_detection

Why this instead of InsightFace's bundled head
----------------------------------------------
InsightFace's `genderage.onnx` does argmax only — it doesn't expose a
softmax confidence. Borderline calls and confident calls look identical
in the UI ("Female 100%"). FairFace gives a real probability so the UI
can show graded confidence.

Inputs
------
img_rgb : np.ndarray (H, W, 3) uint8. Typically a face crop produced
          by `_crop_to_face` in app.py.

Outputs (dict)
--------------
gender              : "male" | "female"
gender_confidence   : float in [0, 1] (argmax softmax probability)
gender_distribution : { "male": p, "female": p }
gender_model_source : "fairface" | "unavailable"
"""

from typing import Any

from PIL import Image
from transformers import pipeline


MODEL_ID = "dima806/fairface_gender_image_detection"
LABELS = ["male", "female"]


class GenderAnalyzer:
    def __init__(self):
        self.classifier = None
        try:
            self.classifier = pipeline("image-classification", model=MODEL_ID)
        except Exception as exc:
            print(f"[GenderAnalyzer] Failed to load {MODEL_ID}: {exc}")

    def analyze(self, img_rgb) -> dict[str, Any]:
        if self.classifier is None:
            return self._empty_result()

        try:
            pil = Image.fromarray(img_rgb)
            preds = self.classifier(pil, top_k=2)
        except Exception as exc:
            print(f"[GenderAnalyzer] Prediction failed: {exc}")
            return self._empty_result()

        if not preds:
            return self._empty_result()

        # FairFace labels can come back capitalised; normalise to lowercase.
        distribution = {label: 0.0 for label in LABELS}
        for pred in preds:
            label = str(pred["label"]).strip().lower()
            if label in distribution:
                distribution[label] = round(float(pred["score"]), 3)

        # Top class wins; expose the actual softmax score as confidence.
        top = preds[0]
        top_label = str(top["label"]).strip().lower()
        if top_label not in LABELS:
            return self._empty_result()

        return {
            "gender": top_label,
            "gender_confidence": round(float(top["score"]), 3),
            "gender_distribution": distribution,
            "gender_model_source": "fairface",
        }

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        return {
            "gender": "unknown",
            "gender_confidence": 0.0,
            "gender_distribution": {label: 0.0 for label in LABELS},
            "gender_model_source": "unavailable",
        }
