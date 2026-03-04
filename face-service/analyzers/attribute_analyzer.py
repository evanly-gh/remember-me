"""
CelebA ResNet-18 — 40 binary facial attributes.

CelebA is a large-scale face attributes dataset with 200k images,
each annotated with 40 binary labels like "Wearing_Hat", "Goatee",
"Heavy_Makeup", etc.

We train or download a ResNet-18 fine-tuned on these 40 attributes,
then group the raw predictions into semantic categories.
"""

import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import models, transforms

CELEBA_ATTRIBUTES = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
    "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
    "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace",
    "Wearing_Necktie", "Young",
]

# Optional: HuggingFace repo hosting the fine-tuned weights
CELEBA_REPO = "nateraw/celeba-resnet18"
CELEBA_FILE = "pytorch_model.bin"

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class AttributeAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()

    def _load_model(self) -> nn.Module:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 40)

        try:
            weight_path = hf_hub_download(
                repo_id=CELEBA_REPO,
                filename=CELEBA_FILE,
            )
            state_dict = torch.load(weight_path, map_location=self.device, weights_only=True)
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"[AttributeAnalyzer] Could not load CelebA weights: {e}")
            print("[AttributeAnalyzer] Falling back to random weights")

        model.to(self.device).eval()
        return model

    def analyze(self, img_rgb: np.ndarray) -> dict[str, Any]:
        pil = Image.fromarray(img_rgb)
        tensor = TRANSFORM(pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)[0]

        probs = torch.sigmoid(logits).cpu().numpy()
        raw: dict[str, float] = {
            attr: round(float(probs[i]), 3)
            for i, attr in enumerate(CELEBA_ATTRIBUTES)
        }
        active = {k for k, v in raw.items() if v > 0.5}

        result: dict[str, Any] = {"_celeba_raw": raw}

        # ── Hair Color ───────────────────────────────────────────────
        hair_colors = {
            "black": raw.get("Black_Hair", 0),
            "blond": raw.get("Blond_Hair", 0),
            "brown": raw.get("Brown_Hair", 0),
            "gray": raw.get("Gray_Hair", 0),
        }
        if "Bald" in active:
            result["hair_color_celeba"] = "bald"
        else:
            result["hair_color_celeba"] = max(hair_colors, key=hair_colors.get)
        result["hair_color_scores"] = hair_colors

        # ── Hair Style ───────────────────────────────────────────────
        if "Straight_Hair" in active:
            result["hair_texture_celeba"] = "straight"
        elif "Wavy_Hair" in active:
            result["hair_texture_celeba"] = "wavy"
        else:
            result["hair_texture_celeba"] = "unknown"

        result["has_bangs"] = "Bangs" in active
        result["is_bald"] = "Bald" in active
        result["receding_hairline"] = "Receding_Hairline" in active

        # ── Facial Hair ──────────────────────────────────────────────
        has_beard = "No_Beard" not in active
        result["has_beard"] = has_beard
        result["facial_hair"] = {
            "5_o_clock_shadow": "5_o_Clock_Shadow" in active,
            "goatee": "Goatee" in active,
            "mustache": "Mustache" in active,
            "sideburns": "Sideburns" in active,
            "full_beard": has_beard and not any(
                k in active for k in ["Goatee", "Mustache", "5_o_Clock_Shadow"]
            ),
        }

        # ── Accessories ──────────────────────────────────────────────
        result["wearing_glasses"] = "Eyeglasses" in active
        result["wearing_earrings"] = "Wearing_Earrings" in active
        result["wearing_hat"] = "Wearing_Hat" in active
        result["wearing_necklace"] = "Wearing_Necklace" in active
        result["wearing_necktie"] = "Wearing_Necktie" in active

        # ── Makeup ───────────────────────────────────────────────────
        result["heavy_makeup"] = "Heavy_Makeup" in active
        result["wearing_lipstick"] = "Wearing_Lipstick" in active

        # ── Face Features ────────────────────────────────────────────
        result["big_nose"] = "Big_Nose" in active
        result["pointy_nose"] = "Pointy_Nose" in active
        result["big_lips"] = "Big_Lips" in active
        result["high_cheekbones"] = "High_Cheekbones" in active
        result["oval_face_celeba"] = "Oval_Face" in active
        result["double_chin"] = "Double_Chin" in active
        result["chubby"] = "Chubby" in active
        result["rosy_cheeks"] = "Rosy_Cheeks" in active
        result["bags_under_eyes"] = "Bags_Under_Eyes" in active
        result["narrow_eyes"] = "Narrow_Eyes" in active
        result["arched_eyebrows"] = "Arched_Eyebrows" in active
        result["bushy_eyebrows"] = "Bushy_Eyebrows" in active
        result["pale_skin"] = "Pale_Skin" in active
        result["attractive"] = "Attractive" in active
        result["young"] = "Young" in active
        result["smiling_celeba"] = "Smiling" in active
        result["mouth_open"] = "Mouth_Slightly_Open" in active

        return result
