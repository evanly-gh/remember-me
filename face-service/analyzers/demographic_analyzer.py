"""
FairFace ResNet-34 — Age, gender, and race estimation.

The FairFace model was trained on ~100k balanced images across
7 race groups, 9 age buckets, and 2 genders. It's one of the
most widely used fair demographic classifiers.

Downloads the model from HuggingFace Hub on first use.
"""

import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import models, transforms

FAIRFACE_REPO = "joshualin24/FairFace"
FAIRFACE_FILE = "res34_fair_align_multi_7_20190809.pt"

AGE_LABELS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
GENDER_LABELS = ["Male", "Female"]
RACE_LABELS = ["White", "Black", "Latino_Hispanic", "East Asian", "Southeast Asian", "Indian", "Middle Eastern"]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class DemographicAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()

    def _load_model(self) -> nn.Module:
        """Load FairFace ResNet-34 with 3 classification heads."""
        model = models.resnet34(weights=None)

        # FairFace uses 3 separate FC heads
        model.fc = nn.Linear(model.fc.in_features, 18)  # 9 age + 2 gender + 7 race

        # Try to download weights from HuggingFace
        try:
            weight_path = hf_hub_download(
                repo_id=FAIRFACE_REPO,
                filename=FAIRFACE_FILE,
            )
            state_dict = torch.load(weight_path, map_location=self.device, weights_only=True)
            # Handle different state dict formats
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"[DemographicAnalyzer] Could not load FairFace weights: {e}")
            print("[DemographicAnalyzer] Using random weights — predictions will be inaccurate")

        model.to(self.device).eval()
        return model

    def analyze(self, img_rgb: np.ndarray) -> dict[str, Any]:
        pil = Image.fromarray(img_rgb)
        tensor = TRANSFORM(pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(tensor)

        logits = out[0].cpu().numpy()

        # Split logits into age (9), gender (2), race (7)
        age_logits = logits[:9]
        gender_logits = logits[9:11]
        race_logits = logits[11:18]

        age_probs = self._softmax(age_logits)
        gender_probs = self._softmax(gender_logits)
        race_probs = self._softmax(race_logits)

        age_idx = int(np.argmax(age_probs))
        gender_idx = int(np.argmax(gender_probs))
        race_idx = int(np.argmax(race_probs))

        # Calculate weighted age estimate
        age_midpoints = [1, 6, 14.5, 24.5, 34.5, 44.5, 54.5, 64.5, 75]
        age_estimate = float(np.dot(age_probs, age_midpoints))

        return {
            "age_range": AGE_LABELS[age_idx],
            "age_estimate": round(age_estimate, 1),
            "age_confidence": round(float(age_probs[age_idx]), 3),
            "gender": GENDER_LABELS[gender_idx].lower(),
            "gender_confidence": round(float(gender_probs[gender_idx]), 3),
            "ethnicity": RACE_LABELS[race_idx],
            "ethnicity_confidence": round(float(race_probs[race_idx]), 3),
            "age_distribution": {
                label: round(float(p), 3) for label, p in zip(AGE_LABELS, age_probs)
            },
            "ethnicity_distribution": {
                label: round(float(p), 3) for label, p in zip(RACE_LABELS, race_probs)
            },
        }

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()
