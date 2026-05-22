"""
BeautyAnalyzer — learned facial-beauty regression on SCUT-FBP5500.

Model
-----
- Architecture : timm ResNet-50 with a single-output regression head
                 (output value in the SCUT-FBP5500 1.0–5.0 score range,
                 averaged from 60 human raters per image).
- Dataset      : SCUT-FBP5500 (5,500 faces, gender + race balanced).
                 https://github.com/HCIILAB/SCUT-FBP5500-Database-Release
- Trained by   : the user, via the training kit at `training/beauty/`.
- Expected MAE : ~0.25–0.30 on the standard test split; Pearson r ≥ 0.85.
- License      : training data is research-only; the trained weights
                 you produce are yours to license as you choose.

Weight loading
--------------
Two ways the analyzer finds weights, tried in order:
1. Local file at `models/beauty_regressor.pt` (drop in after training).
2. Hugging Face Hub repo, controlled by `BEAUTY_HF_REPO_ID` env var
   (e.g. `your-username/scut-fbp5500-resnet50`), loaded via
   huggingface_hub.hf_hub_download.

If neither resolves, the analyzer logs a warning and returns
`beauty_score: None`, which AestheticAnalyzer detects and falls back
to the pure rule-based chopped score.

Inputs
------
img_rgb : np.ndarray (H, W, 3) uint8

Inference resolution is ``BEAUTY_IMG_SIZE`` (default 224). If you trained
with ``--img-size 256`` (the Hyak ``train.slurm`` default), set
``BEAUTY_IMG_SIZE=256`` in the face-service environment so preprocessing
matches the checkpoint.

Outputs (dict)
--------------
beauty_score          : float in [1.0, 5.0] (SCUT-FBP5500 native range)
                        or None if no model is available
beauty_score_norm     : float in [0.0, 100.0] (linearly rescaled)
beauty_model_source   : "local" | "huggingface" | "unavailable"
"""

import os
from typing import Any

import numpy as np
from PIL import Image

try:
    import torch
    import timm
    from torchvision import transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


LOCAL_WEIGHTS_PATH = os.environ.get(
    "BEAUTY_WEIGHTS_PATH", "models/beauty_regressor.pt"
)
HF_REPO_ID = os.environ.get("BEAUTY_HF_REPO_ID")  # e.g. "user/scut-fbp5500-resnet50"
HF_FILENAME = os.environ.get("BEAUTY_HF_FILENAME", "beauty_regressor.pt")
BACKBONE = os.environ.get("BEAUTY_BACKBONE", "resnet50")
# Must match training `--img-size` (224 for older checkpoints, 256 for newer Hyak recipe).
BEAUTY_IMG_SIZE = int(os.environ.get("BEAUTY_IMG_SIZE", "256"))

# Standard ImageNet stats — SCUT-FBP5500 fine-tunes from ImageNet-pretrained
# backbones so we use the same normalisation at inference time.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class BeautyAnalyzer:
    def __init__(self):
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if HAS_TORCH else None
        )
        self.model = None
        self.source = "unavailable"
        self.transform = None

        if not HAS_TORCH:
            print(
                "[BeautyAnalyzer] torch / timm not installed — beauty_score "
                "will be None and AestheticAnalyzer will fall back to rules."
            )
            return

        weights_path = self._resolve_weights_path()
        if weights_path is None:
            local_status = (
                f"found at {LOCAL_WEIGHTS_PATH}"
                if os.path.exists(LOCAL_WEIGHTS_PATH)
                else f"not found at {LOCAL_WEIGHTS_PATH}"
            )
            hub_status = (
                f"BEAUTY_HF_REPO_ID={HF_REPO_ID} (download failed, see prior log line)"
                if HF_REPO_ID
                else "BEAUTY_HF_REPO_ID is unset"
            )
            print(
                "[BeautyAnalyzer] No usable weights — "
                f"local: {local_status}; hub: {hub_status}. "
                "Train one via `training/beauty/train.py` and drop the "
                ".pt into face-service/models/, or set BEAUTY_HF_REPO_ID "
                "to a public HF model repo containing the .pt."
            )
            return

        try:
            # Build backbone with a single regression output. Matches the
            # training script's architecture exactly — see
            # training/beauty/train.py.
            self.model = timm.create_model(BACKBONE, pretrained=False, num_classes=1)
            state = torch.load(weights_path, map_location=self.device)
            # Support both bare state_dicts and {"state_dict": ...} wrappers.
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state, strict=True)
            self.model.to(self.device).eval()

            # Inference resize must match training `--img-size` (see BEAUTY_IMG_SIZE).
            self.transform = transforms.Compose([
                transforms.Resize((BEAUTY_IMG_SIZE, BEAUTY_IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
            print(
                f"[BeautyAnalyzer] Loaded {BACKBONE} weights from "
                f"{weights_path} (source={self.source})"
            )
        except Exception as exc:
            print(f"[BeautyAnalyzer] Failed to load beauty model: {exc}")
            self.model = None

    def _resolve_weights_path(self) -> str | None:
        """Local file wins, HF Hub is the fallback.

        Also records `self.source` ("local" or "huggingface") so the
        analyze() result can report where the weights came from.
        """
        if os.path.exists(LOCAL_WEIGHTS_PATH):
            self.source = "local"
            return LOCAL_WEIGHTS_PATH
        if HF_REPO_ID:
            try:
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
                self.source = "huggingface"
                return path
            except Exception as exc:
                print(f"[BeautyAnalyzer] HF Hub download failed: {exc}")
        return None

    def _set_source(self, source: str) -> None:
        self.source = source

    def analyze(self, img_rgb: np.ndarray) -> dict[str, Any]:
        if self.model is None or self.transform is None:
            return self._empty_result()

        try:
            pil = Image.fromarray(img_rgb).convert("RGB")
            tensor = self.transform(pil).unsqueeze(0).to(self.device)
            # Wrap inference in no_grad to save memory; HAS_TORCH is
            # guaranteed True here because self.model wouldn't exist
            # otherwise.
            with torch.no_grad():
                raw = self.model(tensor).squeeze().item()  # scalar in ~[1, 5]
        except Exception as exc:
            print(f"[BeautyAnalyzer] Inference failed: {exc}")
            return self._empty_result()

        # Clamp to SCUT-FBP5500's nominal 1–5 range; out-of-range outputs
        # mean the regressor's extrapolating beyond its training labels.
        score = max(1.0, min(5.0, float(raw)))

        # Linear rescale to 0–100 for downstream display. 1.0 → 0, 5.0 → 100.
        norm = (score - 1.0) / 4.0 * 100.0

        return {
            "beauty_score": round(score, 3),
            "beauty_score_norm": round(norm, 1),
            "beauty_model_source": self.source or "local",
        }

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        return {
            "beauty_score": None,
            "beauty_score_norm": None,
            "beauty_model_source": "unavailable",
        }
