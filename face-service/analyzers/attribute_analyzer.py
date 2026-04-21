"""
Pretrained CLIP-based face attribute classification.

The old implementation tried to download unavailable CelebA weights and then
fell back to random predictions. This version uses a public zero-shot CLIP
model so the output is deterministic and grounded in pretrained weights.
"""

from typing import Any

from PIL import Image
from transformers import pipeline


CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

PAIRS = {
    "wearing_glasses": ("wearing eyeglasses", "not wearing eyeglasses"),
    "wearing_hat": ("wearing a hat", "not wearing a hat"),
    "has_beard": ("has a beard", "does not have a beard"),
    "mustache": ("has a mustache", "does not have a mustache"),
    "goatee": ("has a goatee", "does not have a goatee"),
    "sideburns": ("has sideburns", "does not have sideburns"),
    "has_bangs": ("has bangs", "does not have bangs"),
    "is_bald": ("is bald", "has hair"),
    "receding_hairline": ("has a receding hairline", "has a full hairline"),
    "wearing_earrings": ("wearing earrings", "not wearing earrings"),
    "wearing_necklace": ("wearing a necklace", "not wearing a necklace"),
    "wearing_necktie": ("wearing a necktie", "not wearing a necktie"),
    "heavy_makeup": ("wearing heavy makeup", "not wearing makeup"),
    "wearing_lipstick": ("wearing lipstick", "not wearing lipstick"),
    "big_nose": ("has a big nose", "has a small nose"),
    "pointy_nose": ("has a pointy nose", "has a rounded nose"),
    "big_lips": ("has big lips", "has thin lips"),
    "high_cheekbones": ("has high cheekbones", "has low cheekbones"),
    "oval_face_celeba": ("has an oval face", "has a non-oval face"),
    "double_chin": ("has a double chin", "does not have a double chin"),
    "chubby": ("has a chubby face", "has a slim face"),
    "rosy_cheeks": ("has rosy cheeks", "does not have rosy cheeks"),
    "bags_under_eyes": ("has bags under the eyes", "does not have bags under the eyes"),
    "narrow_eyes": ("has narrow eyes", "has wide eyes"),
    "arched_eyebrows": ("has arched eyebrows", "has straight eyebrows"),
    "bushy_eyebrows": ("has bushy eyebrows", "has thin eyebrows"),
    "pale_skin": ("has pale skin", "has medium skin"),
    "attractive": ("an attractive face", "an ordinary face"),
    "young": ("a young person", "an older person"),
    "smiling_celeba": ("smiling", "not smiling"),
    "mouth_open": ("mouth open", "mouth closed"),
}

HAIR_COLOR_LABELS = ["black hair", "blond hair", "brown hair", "gray hair"]
HAIR_TEXTURE_LABELS = ["straight hair", "wavy hair", "curly hair"]


class AttributeAnalyzer:
    def __init__(self):
        self.classifier = self._load_classifier()

    @staticmethod
    def _load_classifier():
        try:
            return pipeline("zero-shot-image-classification", model=CLIP_MODEL_ID)
        except Exception as exc:
            print(f"[AttributeAnalyzer] Failed to load CLIP classifier: {exc}")
            return None

    def analyze(self, img_rgb) -> dict[str, Any]:
        pil = Image.fromarray(img_rgb)

        candidate_labels = []
        for positive, negative in PAIRS.values():
            candidate_labels.extend([positive, negative])
        candidate_labels.extend(HAIR_COLOR_LABELS)
        candidate_labels.extend(HAIR_TEXTURE_LABELS)

        if self.classifier is None:
            return self._empty_result()

        try:
            predictions = self.classifier(
                pil,
                candidate_labels=candidate_labels,
                hypothesis_template="a photo of {}",
            )
        except Exception as exc:
            print(f"[AttributeAnalyzer] Prediction failed: {exc}")
            return self._empty_result()
        score_map = {item["label"]: float(item["score"]) for item in predictions}

        result: dict[str, Any] = {"_celeba_raw": {k: round(v, 3) for k, v in score_map.items()}}

        def pair_score(key: str) -> tuple[bool, float]:
            positive, negative = PAIRS[key]
            positive_score = score_map.get(positive, 0.0)
            negative_score = score_map.get(negative, 0.0)
            return positive_score >= negative_score, round(positive_score, 3)

        hair_color = max(HAIR_COLOR_LABELS, key=lambda label: score_map.get(label, 0.0))
        hair_texture = max(HAIR_TEXTURE_LABELS, key=lambda label: score_map.get(label, 0.0))

        result["hair_color_celeba"] = hair_color.split()[0]
        result["hair_color_scores"] = {
            label.split()[0]: round(score_map.get(label, 0.0), 3)
            for label in HAIR_COLOR_LABELS
        }
        result["hair_texture_celeba"] = hair_texture.split()[0]

        result["has_bangs"] = pair_score("has_bangs")[0]
        result["is_bald"] = pair_score("is_bald")[0]
        result["receding_hairline"] = pair_score("receding_hairline")[0]

        has_beard, beard_score = pair_score("has_beard")
        result["has_beard"] = has_beard
        result["facial_hair"] = {
            "5_o_clock_shadow": score_map.get("has a beard", 0.0) > 0.45,
            "goatee": pair_score("goatee")[0],
            "mustache": pair_score("mustache")[0],
            "sideburns": pair_score("sideburns")[0],
            "full_beard": has_beard and beard_score > 0.55,
        }

        result["wearing_glasses"] = pair_score("wearing_glasses")[0]
        result["wearing_earrings"] = pair_score("wearing_earrings")[0]
        result["wearing_hat"] = pair_score("wearing_hat")[0]
        result["wearing_necklace"] = pair_score("wearing_necklace")[0]
        result["wearing_necktie"] = pair_score("wearing_necktie")[0]

        result["heavy_makeup"] = pair_score("heavy_makeup")[0]
        result["wearing_lipstick"] = pair_score("wearing_lipstick")[0]

        result["big_nose"] = pair_score("big_nose")[0]
        result["pointy_nose"] = pair_score("pointy_nose")[0]
        result["big_lips"] = pair_score("big_lips")[0]
        result["high_cheekbones"] = pair_score("high_cheekbones")[0]
        result["oval_face_celeba"] = pair_score("oval_face_celeba")[0]
        result["double_chin"] = pair_score("double_chin")[0]
        result["chubby"] = pair_score("chubby")[0]
        result["rosy_cheeks"] = pair_score("rosy_cheeks")[0]
        result["bags_under_eyes"] = pair_score("bags_under_eyes")[0]
        result["narrow_eyes"] = pair_score("narrow_eyes")[0]
        result["arched_eyebrows"] = pair_score("arched_eyebrows")[0]
        result["bushy_eyebrows"] = pair_score("bushy_eyebrows")[0]
        result["pale_skin"] = pair_score("pale_skin")[0]
        result["attractive"] = pair_score("attractive")[0]
        result["young"] = pair_score("young")[0]
        result["smiling_celeba"] = pair_score("smiling_celeba")[0]
        result["mouth_open"] = pair_score("mouth_open")[0]

        return result

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        return {
            "_celeba_raw": {},
            "hair_color_celeba": "unknown",
            "hair_color_scores": {"black": 0.0, "blond": 0.0, "brown": 0.0, "gray": 0.0},
            "hair_texture_celeba": "unknown",
            "has_bangs": False,
            "is_bald": False,
            "receding_hairline": False,
            "has_beard": False,
            "facial_hair": {
                "5_o_clock_shadow": False,
                "goatee": False,
                "mustache": False,
                "sideburns": False,
                "full_beard": False,
            },
            "wearing_glasses": False,
            "wearing_earrings": False,
            "wearing_hat": False,
            "wearing_necklace": False,
            "wearing_necktie": False,
            "heavy_makeup": False,
            "wearing_lipstick": False,
            "big_nose": False,
            "pointy_nose": False,
            "big_lips": False,
            "high_cheekbones": False,
            "oval_face_celeba": False,
            "double_chin": False,
            "chubby": False,
            "rosy_cheeks": False,
            "bags_under_eyes": False,
            "narrow_eyes": False,
            "arched_eyebrows": False,
            "bushy_eyebrows": False,
            "pale_skin": False,
            "attractive": False,
            "young": False,
            "smiling_celeba": False,
            "mouth_open": False,
        }
