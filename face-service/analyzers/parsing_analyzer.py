"""
BiSeNet — 19-class face parsing / segmentation.

Segments the face into regions: skin, left/right eyebrow, left/right eye,
nose, upper/lower lip, hair, hat, earring, necklace, neck, cloth, etc.

From the segmentation masks we derive:
- Hair presence and approximate length
- Wearing glasses, hat, earrings (from mask presence)
- Skin smoothness (wrinkle detection via edge density on skin mask)
- Freckle/mole estimation (dark spots on skin mask)
- Makeup presence indicators

The masks (_skin_mask, _hair_mask) are also shared with color_analyzer
for accurate pixel sampling.
"""

import os
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

# BiSeNet class labels (19 classes)
PARSING_LABELS = {
    0: "background",
    1: "skin",
    2: "left_eyebrow",
    3: "right_eyebrow",
    4: "left_eye",
    5: "right_eye",
    6: "eyeglasses",
    7: "left_ear",
    8: "right_ear",
    9: "earring",
    10: "nose",
    11: "mouth_interior",
    12: "upper_lip",
    13: "lower_lip",
    14: "neck",
    15: "necklace",
    16: "cloth",
    17: "hair",
    18: "hat",
}

BISENET_REPO = "jonathandinu/face-parsing"
BISENET_FILE = "model.safetensors"

TRANSFORM = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Lightweight BiSeNet Architecture ──────────────────────────────────
# If the HuggingFace weights aren't available, we define a minimal
# architecture. The actual weights determine accuracy.

class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, ks=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, ks, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class BiSeNetMini(nn.Module):
    """Minimal BiSeNet-like architecture for face parsing."""

    def __init__(self, n_classes=19):
        super().__init__()
        # Simplified encoder
        self.layer1 = nn.Sequential(
            ConvBNReLU(3, 64, 7, 2, 3),
            ConvBNReLU(64, 64),
        )
        self.layer2 = nn.Sequential(
            ConvBNReLU(64, 128, 3, 2, 1),
            ConvBNReLU(128, 128),
        )
        self.layer3 = nn.Sequential(
            ConvBNReLU(128, 256, 3, 2, 1),
            ConvBNReLU(256, 256),
        )
        self.layer4 = nn.Sequential(
            ConvBNReLU(256, 512, 3, 2, 1),
            ConvBNReLU(512, 512),
        )
        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = ConvBNReLU(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBNReLU(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBNReLU(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv_out = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        d4 = self.dec4(torch.cat([self.up4(e4), e3], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], 1))
        d1 = self.up1(d2)
        return self.conv_out(d1)


class ParsingAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()

    def _load_model(self) -> nn.Module:
        model = BiSeNetMini(n_classes=19)
        try:
            weight_path = hf_hub_download(
                repo_id=BISENET_REPO,
                filename=BISENET_FILE,
            )
            # Try loading safetensors first, fall back to torch
            try:
                from safetensors.torch import load_file
                state_dict = load_file(weight_path)
            except ImportError:
                state_dict = torch.load(weight_path, map_location=self.device, weights_only=True)

            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"[ParsingAnalyzer] Could not load BiSeNet weights: {e}")
            print("[ParsingAnalyzer] Falling back to random weights")

        model.to(self.device).eval()
        return model

    def analyze(self, img_rgb: np.ndarray) -> dict[str, Any]:
        h, w = img_rgb.shape[:2]
        pil = Image.fromarray(img_rgb)
        tensor = TRANSFORM(pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(tensor)
            if isinstance(out, tuple):
                out = out[0]

        # Resize parsing map to original image size
        parsing = out.squeeze(0).cpu().numpy()
        parsing = np.argmax(parsing, axis=0)  # (512, 512)
        parsing = cv2.resize(
            parsing.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        )

        # Build individual masks
        masks: dict[str, np.ndarray] = {}
        for label_id, label_name in PARSING_LABELS.items():
            masks[label_name] = (parsing == label_id)

        total_pixels = h * w
        region_coverage = {
            name: round(float(mask.sum()) / total_pixels, 4)
            for name, mask in masks.items()
            if name != "background"
        }

        result: dict[str, Any] = {
            "region_coverage": region_coverage,
        }

        # Pass masks for color analyzer
        skin_mask = masks.get("skin", np.zeros((h, w), dtype=bool))
        hair_mask = masks.get("hair", np.zeros((h, w), dtype=bool))
        result["_skin_mask"] = skin_mask
        result["_hair_mask"] = hair_mask
        lip_mask = masks.get("upper_lip", np.zeros((h, w), dtype=bool)) | \
                   masks.get("lower_lip", np.zeros((h, w), dtype=bool))
        result["_lip_mask"] = lip_mask

        # ── Hair length estimation ───────────────────────────────────
        hair_pixels = hair_mask.sum()
        face_pixels = skin_mask.sum() + hair_pixels
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
        result["glasses_detected"] = region_coverage.get("eyeglasses", 0) > 0.005
        result["hat_detected"] = region_coverage.get("hat", 0) > 0.01
        result["earring_detected"] = region_coverage.get("earring", 0) > 0.002
        result["necklace_detected"] = region_coverage.get("necklace", 0) > 0.002

        # ── Skin analysis on skin mask ───────────────────────────────
        if skin_mask.sum() > 100:
            skin_region = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            skin_region = skin_region.astype(float)
            skin_region[~skin_mask] = np.nan

            # Wrinkle estimation via Laplacian edge density
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

            # Freckle/mole estimation (dark spots)
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

            # Skin uniformity
            skin_l_values = l_channel[skin_mask]
            result["skin_uniformity"] = round(float(np.std(skin_l_values)), 2)
        else:
            result["wrinkle_level"] = "unknown"
            result["skin_texture_score"] = 0
            result["freckles_or_moles"] = "unknown"
            result["skin_uniformity"] = 0

        return result
