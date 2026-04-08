"""
Public pretrained demographic classifiers.

This replaces the broken private FairFace weight download with public HF models:
- Age: dima806/fairface_age_image_detection
- Gender: dima806/fairface_gender_image_detection
- Race: NikhilJaddu/fairface-race-vit

The output remains compatible with the rest of the app, but predictions come
from real pretrained weights instead of random fallback values.
"""

from typing import Any

from PIL import Image
from transformers import pipeline


AGE_MODEL_ID = "dima806/fairface_age_image_detection"
GENDER_MODEL_ID = "dima806/fairface_gender_image_detection"
RACE_MODEL_ID = "NikhilJaddu/fairface-race-vit"

AGE_LABELS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
GENDER_LABELS = ["Male", "Female"]
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
            raise RuntimeError(f"Could not load public pretrained model '{model_id}': {exc}") from exc

    def analyze(self, img_rgb) -> dict[str, Any]:
        pil = Image.fromarray(img_rgb)

        age_predictions = self.age_classifier(pil, top_k=3)
        gender_predictions = self.gender_classifier(pil, top_k=2)
        race_predictions = self.race_classifier(pil, top_k=7)

        age_prediction = age_predictions[0]
        gender_prediction = gender_predictions[0]
        race_prediction = race_predictions[0]

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
            "white": "White",
            "black": "Black",
            "latino_hispanic": "Latino_Hispanic",
            "latino hispanic": "Latino_Hispanic",
            "east asian": "East Asian",
            "southeast asian": "Southeast Asian",
            "indian": "Indian",
            "middle eastern": "Middle Eastern",
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
