"""
ParsingAnalyzer — SegFormer-B5 human parsing for masks and skin stats.

Model
-----
- Architecture : SegFormer-B5 (nvidia/mit-b5 backbone)
- HF repo      : matei-dorian/segformer-b5-finetuned-human-parsing
- License      : Apache 2.0
- Eval metrics : mean IoU 0.626, overall acc 0.826
                 face acc 0.909 / IoU 0.829
                 hair acc 0.897 / IoU 0.817
- Classes (18) : background, hat, hair, sunglasses, upper_clothes, skirt,
                 pants, dress, belt, left_shoe, right_shoe, face,
                 left_leg, right_leg, left_arm, right_arm, bag, scarf

Inputs
------
img_rgb : np.ndarray (H, W, 3) uint8.
          Since `app.py` v3.0, this is typically a face-cropped image
          (produced by `_crop_to_face` using the InsightFace bbox)
          rather than the full photo. SegFormer behaves much more
          consistently when the face fills most of the input — the
          masks become tighter and skin-stat estimates less noisy.

Outputs (dict)
--------------
Internal masks (stripped from JSON):
    _skin_mask, _hair_mask
Public fields:
    region_coverage  — per-class fraction of pixels
    hair_length      — bald/very short | short | medium | long
    hair_present     — bool
    hat_detected     — bool, true when ≥1% of pixels are class "hat"
    wrinkle_level    — smooth | slight | moderate | prominent
    skin_texture_score, skin_uniformity, freckles_or_moles

Notes
-----
The wrinkle / texture / freckle fields are OpenCV statistics computed
over the SegFormer face mask, not direct model outputs. SegFormer
contributes the mask; OpenCV does the per-pixel math.
"""

from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

MODEL_ID = "matei-dorian/segformer-b5-finetuned-human-parsing"

# Class id → name as published by the model card. We index masks by
# these names downstream rather than raw integer ids.
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
        # CUDA when available, CPU otherwise. The HF Spaces free tier is
        # CPU-only, so SegFormer-B5 inference takes ~1-2 s per request.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None
        try:
            # Both processor and model weights come from the same repo;
            # processor handles resize/normalize/tensorize.
            self.processor = SegformerImageProcessor.from_pretrained(MODEL_ID)
            self.model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
            self.model.to(self.device).eval()
        except Exception as exc:
            print(f"[ParsingAnalyzer] Failed to load {MODEL_ID}: {exc}")

    def analyze(self, img_rgb: np.ndarray) -> dict[str, Any]:
        h, w = img_rgb.shape[:2]

        # If the model failed to load we return empty masks so the rest
        # of the pipeline (especially ColorAnalyzer) sees a consistent
        # shape and degrades cleanly to "unknown" fields.
        if self.model is None or self.processor is None:
            return self._empty_result(h, w)

        # SegFormer expects PIL; processor will resize internally.
        pil = Image.fromarray(img_rgb)
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)

        # Forward pass → logits at H/4 × W/4 resolution.
        with torch.no_grad():
            logits = self.model(**inputs).logits  # (1, C, H/4, W/4)

        # Upsample to original resolution, then argmax to get the
        # class id per pixel.
        upsampled = torch.nn.functional.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False
        )
        parsing = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        # Build a boolean mask per class. Cheap because we already have
        # the argmax map; each is one numpy equality check.
        masks: dict[str, np.ndarray] = {
            name: (parsing == label_id) for label_id, name in PARSING_LABELS.items()
        }

        # region_coverage = fraction of image occupied by each class.
        # Useful as a coarse "is this class even present" signal — e.g.
        # hat detection just checks if hat coverage exceeds a threshold.
        total_pixels = h * w
        region_coverage = {
            name: round(float(mask.sum()) / total_pixels, 4)
            for name, mask in masks.items()
            if name != "background"
        }

        result: dict[str, Any] = {"region_coverage": region_coverage}

        # Skin & hair masks are passed downstream to ColorAnalyzer.
        # Leading underscore → stripped from the final JSON payload.
        skin_mask = masks.get("face", np.zeros((h, w), dtype=bool))
        hair_mask = masks.get("hair", np.zeros((h, w), dtype=bool))
        result["_skin_mask"] = skin_mask
        result["_hair_mask"] = hair_mask

        # ── Hair length estimation ───────────────────────────────────
        # Ratio of hair pixels to (face + hair) pixels — bigger ratio
        # means longer hair extending past the face.
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

        # ── Hat detection ────────────────────────────────────────────
        # A real hat consistently covers >1% of pixels; below that we're
        # in noise / mis-segmentation territory.
        result["hat_detected"] = region_coverage.get("hat", 0) > 0.01

        # ── Skin texture / wrinkles / freckles ───────────────────────
        # IMPORTANT: SegFormer's "face" class covers the whole face
        # region INCLUDING eyes, eyebrows, lips and nostrils. Those
        # features are naturally darker than skin and have strong
        # edges, which used to inflate every metric:
        #
        # • the Laplacian-on-mask wrinkle score was really measuring
        #   eyebrow + eyelash edges, so almost every photo came back as
        #   "prominent" wrinkles;
        # • the LAB dark-spot freckle count was finding the eyebrows
        #   and pupils as "spots", so almost every photo came back as
        #   "many".
        #
        # Mitigation in this pass: erode the face mask substantially
        # (≈8 px) so we sample only the deep interior — cheeks,
        # forehead, chin — and bump the thresholds. Not perfect; a
        # proper fix wires MediaPipe landmarks in to mask out
        # eyes/brows/lips explicitly.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        interior_mask = cv2.erode(
            skin_mask.astype(np.uint8), kernel, iterations=4
        ).astype(bool)

        if interior_mask.sum() > 100:
            # Wrinkles → Laplacian edge density over the interior mask.
            skin_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            laplacian = cv2.Laplacian(skin_gray, cv2.CV_64F)
            skin_edges = np.abs(laplacian)
            skin_edges[~interior_mask] = 0
            edge_density = (
                skin_edges.sum() / interior_mask.sum() if interior_mask.sum() else 0
            )

            # Thresholds rebumped for the interior-only mask: it
            # naturally has much less edge energy than the un-eroded
            # version, so the bands shifted down. Tune on real photos.
            if edge_density > 22:
                result["wrinkle_level"] = "prominent"
            elif edge_density > 14:
                result["wrinkle_level"] = "moderate"
            elif edge_density > 8:
                result["wrinkle_level"] = "slight"
            else:
                result["wrinkle_level"] = "smooth"

            result["skin_texture_score"] = round(float(edge_density), 2)

            # Freckles/moles → count pixels well below mean L* lightness.
            # Working in LAB rather than RGB makes the threshold tone-
            # independent (a freckle is "darker than surrounding skin"
            # regardless of base skin tone). Restrict to interior_mask
            # so eyes/brows/lips don't get counted as dark spots.
            skin_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
            l_channel = skin_lab[:, :, 0].astype(float)
            l_channel[~interior_mask] = np.nan
            mean_l = np.nanmean(l_channel)
            # Stricter dark-spot threshold (was −25 L*). Real freckles
            # and moles are typically 30+ L* below cheek brightness;
            # smaller deltas were picking up shadows and pores.
            dark_spots = (l_channel < mean_l - 32) & interior_mask
            spot_ratio = (
                dark_spots.sum() / interior_mask.sum() if interior_mask.sum() else 0
            )
            # Bands also tightened — "many" is genuinely lots of spots.
            result["freckles_or_moles"] = (
                "many" if spot_ratio > 0.04
                else "some" if spot_ratio > 0.015
                else "few" if spot_ratio > 0.005
                else "none"
            )

            # Uniformity = std-dev of L* over the interior. Higher = more
            # variation (uneven skin tone, shadows, scarring).
            skin_l_values = l_channel[interior_mask]
            result["skin_uniformity"] = round(float(np.nanstd(skin_l_values)), 2)
        else:
            result["wrinkle_level"] = "unknown"
            result["skin_texture_score"] = 0
            result["freckles_or_moles"] = "unknown"
            result["skin_uniformity"] = 0

        return result

    @staticmethod
    def _empty_result(h: int, w: int) -> dict[str, Any]:
        """Stub returned when the SegFormer model fails to load.

        Shape must match the success path so downstream code can rely
        on key presence without conditional checks.
        """
        empty = np.zeros((h, w), dtype=bool)
        return {
            "region_coverage": {},
            "_skin_mask": empty,
            "_hair_mask": empty,
            "hair_length": "unknown",
            "hair_present": False,
            "hat_detected": False,
            "wrinkle_level": "unknown",
            "skin_texture_score": 0,
            "freckles_or_moles": "unknown",
            "skin_uniformity": 0,
        }
