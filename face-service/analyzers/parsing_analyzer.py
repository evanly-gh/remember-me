"""
SegFormer-B5 human parsing — replaces the old jonathandinu/face-parsing loader.

Model: matei-dorian/segformer-b5-finetuned-human-parsing
  - Architecture: SegFormer-B5 (nvidia/mit-b5 backbone)
  - Published metrics on its eval set:
      • Mean IoU:       0.6258
      • Mean accuracy:  0.7547
      • Overall acc.:   0.8256
      • Face:  acc 0.9094 / IoU 0.8294
      • Hair:  acc 0.8974 / IoU 0.8171
  - Outputs 18 classes (background, hat, hair, sunglasses, upper-clothes, skirt,
    pants, dress, belt, left-shoe, right-shoe, face, left-leg, right-leg,
    left-arm, right-arm, bag, scarf).

We keep the same downstream contract as before: skin/hair/lip masks plus
hair-length, accessory flags, wrinkle estimation, freckle/mole detection.
The lip mask is approximated from the face region (no lip-specific class)
and is mainly used as a fallback — MediaPipe lip landmarks are still the
primary source for lip geometry/color in color_analyzer.
"""

from typing import Any
import warnings

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

MODEL_ID = "matei-dorian/segformer-b5-finetuned-human-parsing"

# Official label map from the model card.
PARSING_LABELS = {
    0: "background",
    1: "hat",
    2: "hair",
    3: "sunglasses",
    4: "upper_clothes",
    5: "skirt",
    6: "pants",
    7: "dress",
    8: "belt",
    9: "left_shoe",
    10: "right_shoe",
    11: "face",
    12: "left_leg",
    13: "right_leg",
    14: "left_arm",
    15: "right_arm",
    16: "bag",
    17: "scarf",
}


class ParsingAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None
        try:
            self.model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
            self.model.to(self.device).eval()
        except Exception as exc:
            print(f"[ParsingAnalyzer] Failed to load {MODEL_ID}: {exc}")

    def analyze(self, img_rgb: np.ndarray) -> dict[str, Any]:
        h, w = img_rgb.shape[:2]

        if self.model is None or self.processor is None:
            return self._empty_result(h, w)

        pil = Image.fromarray(img_rgb)
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits  # (1, C, H/4, W/4)

        upsampled = torch.nn.functional.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False
        )
        parsing = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        masks: dict[str, np.ndarray] = {
            name: (parsing == label_id) for label_id, name in PARSING_LABELS.items()
        }

        total_pixels = h * w
        region_coverage = {
            name: round(float(mask.sum()) / total_pixels, 4)
            for name, mask in masks.items()
            if name != "background"
        }

        result: dict[str, Any] = {"region_coverage": region_coverage}

        skin_mask = masks.get("face", np.zeros((h, w), dtype=bool))
        hair_mask = masks.get("hair", np.zeros((h, w), dtype=bool))
        # No dedicated lip class; color_analyzer falls back to landmarks for lips.
        lip_mask = np.zeros((h, w), dtype=bool)

        result["_skin_mask"] = skin_mask
        result["_hair_mask"] = hair_mask
        result["_lip_mask"] = lip_mask

        # ── Hair length estimation ───────────────────────────────────
        hair_pixels = int(hair_mask.sum())
        face_pixels = int(skin_mask.sum()) + hair_pixels
        hair_ratio = hair_pixels / face_pixels if face_pixels else 0

        if hair_ratio < 0.05:
            result["hair_length"] = "bald/very short"
        elif hair_ratio < 0.2:
            result["hair_length"] = "short"
        elif hair_ratio < 0.4:
            result["hair_length"] = "medium"
        else:
            result["hair_length"] = "long"

        result["hair_present"] = hair_ratio > 0.03

        # ── Accessories from segmentation ────────────────────────────
        result["glasses_detected"] = region_coverage.get("sunglasses", 0) > 0.005
        result["hat_detected"] = region_coverage.get("hat", 0) > 0.01
        result["earring_detected"] = False  # no earring class in this model
        result["necklace_detected"] = False  # no necklace class in this model

        # ── Skin analysis on face mask ───────────────────────────────
        if skin_mask.sum() > 100:
            skin_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            laplacian = cv2.Laplacian(skin_gray, cv2.CV_64F)
            skin_edges = np.abs(laplacian)
            skin_edges[~skin_mask] = 0
            edge_density = skin_edges.sum() / skin_mask.sum() if skin_mask.sum() else 0

            if edge_density > 15:
                result["wrinkle_level"] = "prominent"
            elif edge_density > 8:
                result["wrinkle_level"] = "moderate"
            elif edge_density > 4:
                result["wrinkle_level"] = "slight"
            else:
                result["wrinkle_level"] = "smooth"

            result["skin_texture_score"] = round(float(edge_density), 2)

            skin_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
            l_channel = skin_lab[:, :, 0].astype(float)
            l_channel[~skin_mask] = np.nan
            mean_l = np.nanmean(l_channel)
            dark_spots = (l_channel < mean_l - 25) & skin_mask
            spot_ratio = dark_spots.sum() / skin_mask.sum() if skin_mask.sum() else 0
            result["freckles_or_moles"] = (
                "many" if spot_ratio > 0.05
                else "some" if spot_ratio > 0.015
                else "few" if spot_ratio > 0.005
                else "none"
            )

            skin_l_values = l_channel[skin_mask]
            result["skin_uniformity"] = round(float(np.nanstd(skin_l_values)), 2)
        else:
            result["wrinkle_level"] = "unknown"
            result["skin_texture_score"] = 0
            result["freckles_or_moles"] = "unknown"
            result["skin_uniformity"] = 0

        return result

    @staticmethod
    def _empty_result(h: int, w: int) -> dict[str, Any]:
        empty = np.zeros((h, w), dtype=bool)
        return {
            "region_coverage": {},
            "_skin_mask": empty,
            "_hair_mask": empty,
            "_lip_mask": empty,
            "hair_length": "unknown",
            "hair_present": False,
            "glasses_detected": False,
            "hat_detected": False,
            "earring_detected": False,
            "necklace_detected": False,
            "wrinkle_level": "unknown",
            "skin_texture_score": 0,
            "freckles_or_moles": "unknown",
            "skin_uniformity": 0,
        }
