"""
HSEmotion — EfficientNet-B0 for emotion recognition.

Classifies the dominant emotion in a face image into one of 8 categories:
Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise.

Also provides valence (positive/negative) and arousal (calm/excited)
scores derived from the emotion distribution.
"""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

EMOTION_LABELS = [
    "anger", "contempt", "disgust", "fear",
    "happiness", "neutral", "sadness", "surprise",
]

# Valence weights for each emotion (-1 to +1)
VALENCE_MAP = {
    "anger": -0.6,
    "contempt": -0.3,
    "disgust": -0.7,
    "fear": -0.6,
    "happiness": 0.9,
    "neutral": 0.0,
    "sadness": -0.7,
    "surprise": 0.3,
}

# Arousal weights for each emotion (0 to 1)
AROUSAL_MAP = {
    "anger": 0.8,
    "contempt": 0.3,
    "disgust": 0.5,
    "fear": 0.9,
    "happiness": 0.7,
    "neutral": 0.1,
    "sadness": 0.3,
    "surprise": 0.9,
}

TRANSFORM = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class EmotionAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()

    def _load_model(self) -> nn.Module:
        if not HAS_TIMM:
            print("[EmotionAnalyzer] timm not installed — using fallback CNN")
            return self._fallback_model()

        try:
            # Try HSEmotion pre-trained model
            model = timm.create_model(
                "tf_efficientnet_b0_ns",
                pretrained=True,
                num_classes=8,
            )
        except Exception as e:
            print(f"[EmotionAnalyzer] Could not load EfficientNet: {e}")
            model = timm.create_model(
                "tf_efficientnet_b0_ns",
                pretrained=False,
                num_classes=8,
            )

        model.to(self.device).eval()
        return model

    def _fallback_model(self) -> nn.Module:
        """Simple fallback if timm is not available."""
        from torchvision import models
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, 8)
        model.to(self.device).eval()
        return model

    def analyze(self, img_rgb: np.ndarray) -> dict[str, Any]:
        pil = Image.fromarray(img_rgb)
        tensor = TRANSFORM(pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)[0]

        probs = torch.softmax(logits, dim=0).cpu().numpy()

        emotion_scores = {
            label: round(float(probs[i]), 3)
            for i, label in enumerate(EMOTION_LABELS)
        }

        primary_idx = int(np.argmax(probs))
        primary_emotion = EMOTION_LABELS[primary_idx]
        primary_confidence = float(probs[primary_idx])

        # Secondary emotion (second highest)
        sorted_idx = np.argsort(probs)[::-1]
        secondary_emotion = EMOTION_LABELS[int(sorted_idx[1])]

        # Calculate valence and arousal
        valence = sum(
            probs[i] * VALENCE_MAP[label]
            for i, label in enumerate(EMOTION_LABELS)
        )
        arousal = sum(
            probs[i] * AROUSAL_MAP[label]
            for i, label in enumerate(EMOTION_LABELS)
        )

        return {
            "primary_emotion": primary_emotion,
            "emotion_confidence": round(primary_confidence, 3),
            "secondary_emotion": secondary_emotion,
            "emotion_scores": emotion_scores,
            "valence": round(float(valence), 3),
            "arousal": round(float(arousal), 3),
            "mood": (
                "positive" if valence > 0.2
                else "negative" if valence < -0.2
                else "neutral"
            ),
        }
