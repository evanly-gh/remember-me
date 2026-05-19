"""
DemographicAnalyzer — age, gender, ethnicity via three ViT classifiers.

Models
------
- Age       : dima806/fairface_age_image_detection
              ViT-B/16, ~59% top-1 on FairFace 9 age buckets.
- Gender    : dima806/fairface_gender_image_detection
              ViT-B/16, ~93.4% on FairFace.
- Ethnicity : cledoux42/Ethnicity_Test_v003
              ViT, 79.6% accuracy, macro-F1 0.797. 5-class output that
              we widen into the legacy 7-bucket FairFace schema so the
              rest of the app's distribution shape doesn't change.

All three are Apache 2.0 and Hugging Face image-classification pipelines.

Inputs
------
img_rgb : np.ndarray (H, W, 3) uint8

Outputs (dict)
--------------
age_range, age_estimate (softmax-weighted continuous), age_confidence,
age_distribution, gender, gender_confidence, ethnicity,
ethnicity_confidence, ethnicity_distribution.

Notes
-----
The FairFace age model is a 9-bucket classifier (0-2, 3-9, …, 70+),
which means the argmax bucket midpoint is always one of nine fixed
numbers (24.5 for 20-29, etc.). To recover a smooth continuous estimate
we compute the expected value across the full softmax — see
``_weighted_age_estimate``.
"""

from typing import Any

from PIL import Image
from transformers import pipeline


AGE_MODEL_ID = "dima806/fairface_age_image_detection"
GENDER_MODEL_ID = "dima806/fairface_gender_image_detection"
RACE_MODEL_ID = "cledoux42/Ethnicity_Test_v003"

AGE_LABELS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
GENDER_LABELS = ["Male", "Female"]
# cledoux42 ships 5 classes (african, asian, caucasian, hispanic, indian),
# but we keep the legacy 7-bucket FairFace label space internally so the
# downstream distribution dict shape stays stable. Unseen buckets stay 0.
RACE_LABELS = ["White", "Black", "Latino_Hispanic", "East Asian", "Southeast Asian", "Indian", "Middle Eastern"]


class DemographicAnalyzer:
    def __init__(self):
        # Each classifier is a HF image-classification pipeline. They lazy
        # download weights from HF on first instantiation and cache them
        # under /root/.cache/huggingface inside the container.
        self.age_classifier = self._load_classifier(AGE_MODEL_ID)
        self.gender_classifier = self._load_classifier(GENDER_MODEL_ID)
        self.race_classifier = self._load_classifier(RACE_MODEL_ID)

    @staticmethod
    def _load_classifier(model_id: str):
        """Build one HF image-classification pipeline, logging on failure.

        A failed load returns None so the rest of the service continues
        to function and `analyze()` falls back to "unknown" demographics.
        """
        try:
            return pipeline("image-classification", model=model_id)
        except Exception as exc:
            print(f"[DemographicAnalyzer] Failed to load '{model_id}': {exc}")
            return None

    def analyze(self, img_rgb) -> dict[str, Any]:
        # Convert the numpy frame to a PIL Image once and reuse it for
        # all three classifier calls.
        pil = Image.fromarray(img_rgb)

        # top_k=len(labels) so we get the full softmax for each model.
        # We need the full age distribution to compute the weighted
        # expected-value age estimate.
        age_predictions = self._safe_predict(self.age_classifier, pil, top_k=len(AGE_LABELS))
        gender_predictions = self._safe_predict(self.gender_classifier, pil, top_k=2)
        race_predictions = self._safe_predict(self.race_classifier, pil, top_k=7)

        # If every classifier failed we degrade gracefully with a stub.
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

        # HF pipelines return predictions pre-sorted by score descending,
        # so prediction[0] is always the argmax class.
        age_prediction = age_predictions[0] if age_predictions else {"label": "unknown", "score": 0.0}
        gender_prediction = gender_predictions[0] if gender_predictions else {"label": "unknown", "score": 0.0}
        race_prediction = race_predictions[0] if race_predictions else {"label": "unknown", "score": 0.0}

        # Models occasionally return label aliases ("more than 70" instead
        # of "70+", "African" instead of "Black"). The normalisers map
        # everything back to our canonical schema.
        age_label = self._normalize_age_label(age_prediction["label"])
        gender_label = self._normalize_gender_label(gender_prediction["label"])
        race_label = self._normalize_race_label(race_prediction["label"])

        return {
            "age_range": age_label,
            "age_estimate": self._weighted_age_estimate(age_predictions),
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
        """Map model output to canonical AGE_LABELS entry."""
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
        """Coalesce cledoux42's 5 classes into our 7-bucket schema."""
        normalized = label.strip().lower().replace("-", "_")
        race_aliases = {
            # Legacy FairFace 7-class labels
            "white": "White",
            "black": "Black",
            "latino_hispanic": "Latino_Hispanic",
            "latino hispanic": "Latino_Hispanic",
            "east asian": "East Asian",
            "southeast asian": "Southeast Asian",
            "indian": "Indian",
            "middle eastern": "Middle Eastern",
            # cledoux42/Ethnicity_Test_v003 5-class labels
            "african": "Black",
            "asian": "East Asian",
            "caucasian": "White",
            "hispanic": "Latino_Hispanic",
        }
        return race_aliases.get(normalized, label)

    # Midpoint of each FairFace age bucket — used as the per-bucket
    # "value" when we marginalise over the predicted distribution.
    _AGE_MIDPOINTS = {
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

    @classmethod
    def _weighted_age_estimate(cls, predictions: list[dict]) -> float:
        """Softmax-weighted expected age across all FairFace buckets.

        FairFace is a 9-bucket classifier; the argmax always snaps to one
        of nine fixed midpoints (24.5 for 20-29, etc.). Treating its
        softmax as a probability distribution and taking the expected
        value gives a continuous number that moves with confidence
        (23.1 for someone very confidently 20-29, 28.4 if some mass leaks
        into 30-39). Still bounded by bucket midpoints — true per-year
        accuracy would need a regression model.
        """
        total_weight = 0.0
        weighted_sum = 0.0
        for pred in predictions:
            label = cls._normalize_age_label(pred["label"])
            midpoint = cls._AGE_MIDPOINTS.get(label)
            if midpoint is None:
                continue
            score = float(pred["score"])
            weighted_sum += midpoint * score
            total_weight += score
        if total_weight == 0:
            return 0.0
        return round(weighted_sum / total_weight, 1)

    @classmethod
    def _distribution_map(cls, predictions, normalizer, all_labels):
        """Flatten HF predictions into {canonical_label: score} dict.

        Unseen labels stay at 0.0 so the shape is always all_labels-sized.
        """
        distribution = {label: 0.0 for label in all_labels}
        for prediction in predictions:
            normalized_label = normalizer(prediction["label"])
            if normalized_label in distribution:
                distribution[normalized_label] = round(float(prediction["score"]), 3)
        return distribution

    @staticmethod
    def _safe_predict(classifier, image, top_k: int):
        """Wrap classifier(...) so a single model failure can't bring
        down the whole demographic block."""
        if classifier is None:
            return []
        try:
            return classifier(image, top_k=top_k)
        except Exception as exc:
            print(f"[DemographicAnalyzer] Prediction failed: {exc}")
            return []
