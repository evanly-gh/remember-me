"""
Public pretrained demographic classifiers.

Models used (all public, with published accuracy):
- Age:       dima806/fairface_age_image_detection   (~59% top-1 on FairFace age buckets)
- Gender:    dima806/fairface_gender_image_detection (~93.4% on FairFace)
- Ethnicity: cledoux42/Ethnicity_Test_v003          (ViT, 79.6% accuracy, macro-F1 0.797)

The ethnicity model replaces the former NikhilJaddu/fairface-race-vit checkpoint,
which had no published performance metrics on the HF model card.
"""

from typing import Any

from PIL import Image
from transformers import pipeline


AGE_MODEL_ID = "dima806/fairface_age_image_detection"
GENDER_MODEL_ID = "dima806/fairface_gender_image_detection"
RACE_MODEL_ID = "cledoux42/Ethnicity_Test_v003"

AGE_LABELS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
GENDER_LABELS = ["Male", "Female"]
# cledoux42/Ethnicity_Test_v003 outputs 5 classes: african, asian, caucasian, hispanic, indian.
# We keep the legacy 7-bucket schema internally so the rest of the app still works;
# unseen buckets simply stay at 0.0 in the distribution.
RACE_LABELS = ["White", "Black", "Latino_Hispanic", "East Asian", "Southeast Asian", "Indian", "Middle Eastern"]


class DemographicAnalyzer:
    def __init__(self):
        self.age_classifier = self._load_classifier(AGE_MODEL_ID)
        self.gender_classifier = self._load_classifier(GENDER_MODEL_ID)
        self.race_classifier = self._load_classifier(RACE_MODEL_ID)

    @staticmethod
    def _load_classifier(model_id: str):
        try:
            return pipeline("image-classification", model=model_id)
        except Exception as exc:
            print(f"[DemographicAnalyzer] Failed to load '{model_id}': {exc}")
            return None

    def analyze(self, img_rgb) -> dict[str, Any]:
        pil = Image.fromarray(img_rgb)

        age_predictions = self._safe_predict(self.age_classifier, pil, top_k=3)
        gender_predictions = self._safe_predict(self.gender_classifier, pil, top_k=2)
        race_predictions = self._safe_predict(self.race_classifier, pil, top_k=7)

        if not age_predictions and not gender_predictions and not race_predictions:
            return {
                "age_range": "unknown",
                "age_estimate": 0.0,
                "age_confidence": 0.0,
                "gender": "unknown",
                "gender_confidence": 0.0,
                "ethnicity": "unknown",
                "ethnicity_confidence": 0.0,
                "age_distribution": {label: 0.0 for label in AGE_LABELS},
                "ethnicity_distribution": {label: 0.0 for label in RACE_LABELS},
            }

        age_prediction = age_predictions[0] if age_predictions else {"label": "unknown", "score": 0.0}
        gender_prediction = gender_predictions[0] if gender_predictions else {"label": "unknown", "score": 0.0}
        race_prediction = race_predictions[0] if race_predictions else {"label": "unknown", "score": 0.0}

        age_label = self._normalize_age_label(age_prediction["label"])
        gender_label = self._normalize_gender_label(gender_prediction["label"])
        race_label = self._normalize_race_label(race_prediction["label"])

        return {
            "age_range": age_label,
            "age_estimate": self._age_estimate_from_label(age_label),
            "age_confidence": round(float(age_prediction["score"]), 3),
            "gender": gender_label.lower(),
            "gender_confidence": round(float(gender_prediction["score"]), 3),
            "ethnicity": race_label,
            "ethnicity_confidence": round(float(race_prediction["score"]), 3),
            "age_distribution": self._distribution_map(age_predictions, self._normalize_age_label, AGE_LABELS),
            "ethnicity_distribution": self._distribution_map(race_predictions, self._normalize_race_label, RACE_LABELS),
        }

    @staticmethod
    def _normalize_age_label(label: str) -> str:
        normalized = label.strip().lower()
        if normalized == "more than 70":
            return "70+"
        return AGE_LABELS[AGE_LABELS.index(label)] if label in AGE_LABELS else label

    @staticmethod
    def _normalize_gender_label(label: str) -> str:
        normalized = label.strip().lower()
        if normalized in {"male", "female"}:
            return normalized.capitalize()
        return label

    @staticmethod
    def _normalize_race_label(label: str) -> str:
        normalized = label.strip().lower().replace("-", "_")
        race_aliases = {
            # Original FairFace 7-class labels
            "white": "White",
            "black": "Black",
            "latino_hispanic": "Latino_Hispanic",
            "latino hispanic": "Latino_Hispanic",
            "east asian": "East Asian",
            "southeast asian": "Southeast Asian",
            "indian": "Indian",
            "middle eastern": "Middle Eastern",
            # cledoux42/Ethnicity_Test_v003 5-class labels → map into our schema
            "african": "Black",
            "asian": "East Asian",
            "caucasian": "White",
            "hispanic": "Latino_Hispanic",
        }
        return race_aliases.get(normalized, label)

    @staticmethod
    def _age_estimate_from_label(label: str) -> float:
        mapping = {
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
        return mapping.get(label, 0.0)

    @classmethod
    def _distribution_map(cls, predictions, normalizer, all_labels):
        distribution = {label: 0.0 for label in all_labels}
        for prediction in predictions:
            normalized_label = normalizer(prediction["label"])
            if normalized_label in distribution:
                distribution[normalized_label] = round(float(prediction["score"]), 3)
        return distribution

    @staticmethod
    def _safe_predict(classifier, image, top_k: int):
        if classifier is None:
            return []
        try:
            return classifier(image, top_k=top_k)
        except Exception as exc:
            print(f"[DemographicAnalyzer] Prediction failed: {exc}")
            return []
