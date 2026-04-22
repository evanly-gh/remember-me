"""
CLIP zero-shot attribute classification.

Previous version put all ~70 candidate labels into a single zero-shot pipeline
call, which applied one softmax across every label at once. That meant each
binary pair ("wearing earrings" vs "not wearing earrings") received ~1/70 of
the probability mass and the comparison between positive and negative was
essentially noise — hence the hallucinated accessories.

This version encodes the image once with CLIPModel.get_image_features, then
runs a fresh 2-way softmax per binary pair. Group labels (hair color,
hair texture) get their own N-way softmax. All scores are now independent
of how many other labels we happen to be asking about.
"""

from typing import Any

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


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

# Some pairs default to False unless CLIP is confidently past this threshold.
# Stops borderline cases from being flipped to True on a 51/49 split.
ACCESSORY_THRESHOLD = 0.65
ACCESSORY_KEYS = {
    "wearing_earrings", "wearing_necklace", "wearing_necktie", "wearing_hat",
    "heavy_makeup", "wearing_lipstick",
}


def _prompt(text: str) -> str:
    return f"a photo of {text}"


class AttributeAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        try:
            self.model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(self.device).eval()
            self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        except Exception as exc:
            print(f"[AttributeAnalyzer] Failed to load CLIP: {exc}")

    @torch.no_grad()
    def analyze(self, img_rgb) -> dict[str, Any]:
        if self.model is None or self.processor is None:
            return self._empty_result()

        pil = Image.fromarray(img_rgb)

        # Encode image once.
        image_inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        image_features = self.model.get_image_features(**image_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Per-pair scoring: each pair gets its own independent 2-way softmax.
        pair_scores: dict[str, float] = {}
        for key, (positive, negative) in PAIRS.items():
            prompts = [_prompt(positive), _prompt(negative)]
            pair_scores[key] = self._softmax_positive(image_features, prompts)

        # Group scoring (N-way softmax within each group).
        color_scores = self._group_softmax(
            image_features, [_prompt(x) for x in HAIR_COLOR_LABELS]
        )
        texture_scores = self._group_softmax(
            image_features, [_prompt(x) for x in HAIR_TEXTURE_LABELS]
        )

        hair_color_name = HAIR_COLOR_LABELS[int(torch.argmax(torch.tensor(color_scores)))].split()[0]
        hair_texture_name = HAIR_TEXTURE_LABELS[int(torch.argmax(torch.tensor(texture_scores)))].split()[0]

        def flag(key: str) -> bool:
            score = pair_scores.get(key, 0.0)
            threshold = ACCESSORY_THRESHOLD if key in ACCESSORY_KEYS else 0.5
            return score >= threshold

        result: dict[str, Any] = {
            "_celeba_raw": {k: round(v, 3) for k, v in pair_scores.items()},
            "hair_color_celeba": hair_color_name,
            "hair_color_scores": {
                label.split()[0]: round(float(score), 3)
                for label, score in zip(HAIR_COLOR_LABELS, color_scores)
            },
            "hair_texture_celeba": hair_texture_name,
        }

        for key in PAIRS:
            result[key] = flag(key)

        beard_score = pair_scores.get("has_beard", 0.0)
        result["facial_hair"] = {
            "5_o_clock_shadow": 0.45 < beard_score < 0.7,
            "goatee": flag("goatee"),
            "mustache": flag("mustache"),
            "sideburns": flag("sideburns"),
            "full_beard": beard_score > 0.7,
        }

        return result

    @torch.no_grad()
    def _softmax_positive(self, image_features: torch.Tensor, prompts: list[str]) -> float:
        text_inputs = self.processor(
            text=prompts, return_tensors="pt", padding=True
        ).to(self.device)
        text_features = self.model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = (image_features @ text_features.T) * self.model.logit_scale.exp()
        probs = torch.softmax(logits, dim=-1)[0]
        return float(probs[0])

    @torch.no_grad()
    def _group_softmax(self, image_features: torch.Tensor, prompts: list[str]) -> list[float]:
        text_inputs = self.processor(
            text=prompts, return_tensors="pt", padding=True
        ).to(self.device)
        text_features = self.model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = (image_features @ text_features.T) * self.model.logit_scale.exp()
        probs = torch.softmax(logits, dim=-1)[0]
        return [float(p) for p in probs]

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        base: dict[str, Any] = {
            "_celeba_raw": {},
            "hair_color_celeba": "unknown",
            "hair_color_scores": {"black": 0.0, "blond": 0.0, "brown": 0.0, "gray": 0.0},
            "hair_texture_celeba": "unknown",
            "facial_hair": {
                "5_o_clock_shadow": False,
                "goatee": False,
                "mustache": False,
                "sideburns": False,
                "full_beard": False,
            },
        }
        for key in PAIRS:
            base[key] = False
        return base
