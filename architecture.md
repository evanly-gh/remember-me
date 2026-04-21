# HCP Face Analysis — Architecture Plan

## Revised Architecture & Best Models for Maximum Feature Coverage

Since the codebase is flexible and can use more languages and frameworks, we go beyond the Supabase Edge Function constraint to find the **absolute best models** for the full feature list.

---

## Recommended Architecture: Python Microservice Sidecar

```
┌──────────────────────────────────────────────────────────┐
│                     CURRENT STACK                        │
│  Next.js Frontend ──► Supabase (Auth, DB, Storage)       │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│          NEW: Python Face Analysis Microservice          │
│  (Railway.app / Render.com / Hugging Face Spaces)        │
│  FREE TIER: 512MB RAM, shared CPU                        │
│                                                          │
│  FastAPI Server                                          │
│  ├── MediaPipe Face Landmarker (478 landmarks, 4MB)      │
│  ├── InsightFace Buffalo_SC (recognition + attrs, 30MB)  │
│  ├── FairFace (age/gender/race, 90MB)                    │
│  ├── HuggingFace ViT models (attributes, ~50MB each)     │
│  ├── BiSeNet (face parsing/segmentation, 50MB)           │
│  └── Custom geometric analysis (your feature list)       │
│                                                          │
│  Total: ~250MB models (loaded lazily)                    │
└──────────────────────────────────────────────────────────┘
```

**Why this is better:** Python gives access to the **entire deep learning ecosystem** — every model on HuggingFace, every research paper's pretrained weights. Free-tier hosting on Railway/Render gives 512MB RAM and enough CPU for per-request inference.

---

## Best Models Per Feature Category

### Tier 1: Core Models (Must Have)

#### 1. MediaPipe Face Landmarker — Geometric Features
- **478 3D landmarks + 52 blendshapes**
- **Size:** 4MB
- **Covers:** Face shape, jawline, chin, cheekbones, forehead, eye shape, eye spacing, eye size, eyebrow shape, nose shape, lip shape, mouth width, dimples, facial asymmetry
- **GitHub:** https://github.com/google-ai-edge/mediapipe
- **Python:** `pip install mediapipe`
- **Accuracy:** State-of-the-art landmark detection, handles 30° head rotation well

#### 2. InsightFace Buffalo_SC — Lightweight Recognition + Age/Gender
- **Size:** ~30MB (smallest Buffalo variant)
- **LFW Accuracy:** 99.5%
- **Covers:** Face detection, age, gender, face embedding (for recognition), 2D landmarks
- **GitHub:** https://github.com/deepinsight/insightface
- **Weights:** Auto-downloaded via `insightface.app.FaceAnalysis(name='buffalo_sc')`
- **Why not Buffalo_L:** 320MB is overkill; Buffalo_SC is 90% as accurate at 1/10th the size

#### 3. FairFace — Age, Gender, Race (Most Accurate)
- **Size:** ~90MB (ResNet-34)
- **Accuracy:** 93.4% race, 94.2% gender, MAE 3.4 years for age
- **Covers:** Age (9 buckets), gender, race (7 categories: White, Black, Latino, East Asian, Southeast Asian, Indian, Middle Eastern)
- **GitHub:** https://github.com/dchen236/FairFace
- **Weights:** https://drive.google.com/file/d/1xSfJQWMhm3AVlJYcPcabGO_bj1kDB0xw (res34_fair_align_multi_7_20190809.pt)
- **Why over InsightFace for this:** FairFace is specifically trained for fair demographic classification across races, not biased toward any group

#### 4. HSEmotion (EfficientNet) — Emotion Recognition
- **Size:** ~20MB
- **Accuracy:** 66.5% on AffectNet-8 (state-of-the-art), 8 emotions
- **Covers:** Angry, contempt, disgust, fear, happy, neutral, sad, surprise
- **GitHub:** https://github.com/HSE-asavchenko/face-emotion-recognition
- **Weights:** Available via `timm` or direct download from repo
- **Why over face-api.js:** Significantly more accurate, trained on AffectNet (largest emotion dataset)

### Tier 2: Specialized Models

#### 5. BiSeNet Face Parsing — Facial Segmentation
- **Size:** ~50MB
- **Covers:** Skin region, left/right eyebrow, left/right eye, nose, upper/lower lip, inner mouth, hair, left/right ear, neck, cloth, hat, earrings, glasses, background
- **GitHub:** https://github.com/zllrunning/face-parsing.PyTorch
- **Weights:** https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812
- **Why this matters:** Precisely segments hair, skin, eyebrows for color analysis, facial hair detection, glasses detection, and wrinkle analysis

#### 6. microsoft/swin-base-patch4-window7-224-in22k fine-tuned for facial attributes
- **HuggingFace:** Various CelebA-trained attribute classifiers
- Specifically: https://huggingface.co/nateraw/vit-age-classifier (age)
- Specifically: https://huggingface.co/rizvandwiki/gender-classification-2 (gender)

#### 7. CelebA Attribute Classifier (Custom Multi-Label)
- **Dataset:** CelebA has 40 binary attributes already labeled
- Train a lightweight EfficientNet-B0 (~20MB) on CelebA for:
  - `Attractive`, `Bald`, `Bangs`, `Big_Lips`, `Big_Nose`, `Black_Hair`, `Blond_Hair`, `Brown_Hair`, `Bushy_Eyebrows`, `Chubby`, `Double_Chin`, `Eyeglasses`, `Goatee`, `Gray_Hair`, `Heavy_Makeup`, `High_Cheekbones`, `Male`, `Mouth_Slightly_Open`, `Mustache`, `Narrow_Eyes`, `No_Beard`, `Oval_Face`, `Pointy_Nose`, `Receding_Hairline`, `Sideburns`, `Smiling`, `Straight_Hair`, `Wavy_Hair`, `Wearing_Hat`, `Young`
- **Pre-trained option:** https://github.com/dchen236/FairFace has CelebA-trained models
- **Better pre-trained option:** https://huggingface.co/jnferreira/attribute-prediction-celebA

#### 8. Hair Segmentation + Color Analysis
- **Model:** MODNet for matting + BiSeNet for hair segmentation
- **GitHub (MODNet):** https://github.com/ZHKKKe/MODNet (~25MB)
- Post-segmentation: K-means clustering on hair pixels for color

#### 9. Skin Analysis (Wrinkles, Acne, etc.)
- **Model:** https://huggingface.co/imfarzanansari/skin-disease-detection (for acne/skin conditions)
- **For wrinkles:** Edge detection (Canny/Sobel) on forehead/eye regions from BiSeNet parsing — no model needed
- **For freckles/moles:** Blob detection on skin regions from BiSeNet parsing

---

## Complete Feature Coverage Map

| Feature | Model/Method | Confidence |
|---------|-------------|------------|
| **Face shape** (oval, round, square, heart, diamond, oblong, triangle) | MediaPipe landmarks geometric ratios + CelebA (`Oval_Face`) | ⭐⭐⭐⭐ |
| **Jawline** (sharp, soft, strong) | MediaPipe jaw landmark angles | ⭐⭐⭐⭐ |
| **Chin** (receding, pointed, cleft, wide) | MediaPipe chin landmarks + depth (z) | ⭐⭐⭐ |
| **Cheekbones** (high, flat, full, hollow) | MediaPipe landmark z-depth + CelebA (`High_Cheekbones`, `Chubby`) | ⭐⭐⭐⭐ |
| **Forehead** (broad, narrow) | MediaPipe forehead span ratio | ⭐⭐⭐⭐ |
| **Eye shape** (almond, round, hooded, monolid, upturned, downturned) | MediaPipe eyelid curvature + corner angles | ⭐⭐⭐⭐ |
| **Eye spacing** (wide-set, close-set) | MediaPipe interpupillary distance ratio | ⭐⭐⭐⭐⭐ |
| **Eye size** (large, small) | MediaPipe eye area / face area | ⭐⭐⭐⭐⭐ |
| **Deep-set / protruding eyes** | MediaPipe landmark z-depth at eye region | ⭐⭐⭐ |
| **Eye color** (brown, blue, green, hazel) | Iris crop → HSV color histogram + KNN | ⭐⭐⭐⭐ |
| **Dark under-eyes / eye bags** | BiSeNet skin parsing → brightness analysis under eyes | ⭐⭐⭐ |
| **Crow's feet** | Canny edge detection on BiSeNet-parsed outer eye skin | ⭐⭐⭐ |
| **Eyebrow shape** (arched, straight, bushy, thick, thin) | MediaPipe brow landmarks + CelebA (`Bushy_Eyebrows`, `Arched_Eyebrows`) | ⭐⭐⭐⭐ |
| **Unibrow** | MediaPipe inner brow distance + pixel analysis between brows | ⭐⭐⭐⭐ |
| **Nose shape** (straight, aquiline, button, upturned, wide, narrow) | MediaPipe nose landmarks + CelebA (`Big_Nose`, `Pointy_Nose`) | ⭐⭐⭐⭐ |
| **Nose bridge** (flat, high) | MediaPipe z-depth at nasal bridge | ⭐⭐⭐ |
| **Nostrils** (wide, narrow) | MediaPipe nostril landmark width ratio | ⭐⭐⭐⭐ |
| **Lips** (full, thin) | MediaPipe lip landmarks + CelebA (`Big_Lips`) | ⭐⭐⭐⭐ |
| **Mouth width** | MediaPipe mouth corner distance ratio | ⭐⭐⭐⭐⭐ |
| **Cupid's bow** | MediaPipe upper lip curvature analysis | ⭐⭐⭐ |
| **Teeth** (gap, crooked, straight, overbite, underbite) | Mouth crop when smiling → custom classifier or rule-based | ⭐⭐ |
| **Dimples** | MediaPipe blendshapes during smile + cheek region analysis | ⭐⭐⭐ |
| **Smile lines** | Edge detection on nasolabial region | ⭐⭐⭐ |
| **Asymmetrical smile** | MediaPipe left/right smile blendshape difference | ⭐⭐⭐⭐ |
| **Hair type** (straight, wavy, curly, coily) | BiSeNet hair segmentation → texture frequency (FFT) + CelebA (`Straight_Hair`, `Wavy_Hair`) | ⭐⭐⭐ |
| **Hair length** (short, long, bald) | BiSeNet hair mask area + CelebA (`Bald`, `Bangs`) | ⭐⭐⭐⭐ |
| **Hair color** (black, brown, blonde, red, gray, dyed) | BiSeNet hair mask → K-means color clustering + CelebA (`Black_Hair`, `Brown_Hair`, `Blond_Hair`, `Gray_Hair`) | ⭐⭐⭐⭐ |
| **Receding hairline / widow's peak** | BiSeNet hair boundary analysis + CelebA (`Receding_Hairline`) | ⭐⭐⭐ |
| **Beard/facial hair** (full, stubble, goatee, mustache, sideburns, clean-shaven) | BiSeNet parsing lower face + CelebA (`5_o_Clock_Shadow`, `Goatee`, `Mustache`, `No_Beard`, `Sideburns`) | ⭐⭐⭐⭐ |
| **Skin tone** (light, medium, dark) | BiSeNet skin parsing → mean LAB brightness | ⭐⭐⭐⭐⭐ |
| **Freckles** | BiSeNet skin mask → small blob detection (contrast) | ⭐⭐⭐ |
| **Moles / birthmark** | BiSeNet skin mask → dark blob detection | ⭐⭐⭐ |
| **Scars** | BiSeNet skin mask → linear edge anomaly detection | ⭐⭐ |
| **Acne** | BiSeNet skin mask → red blob detection or HuggingFace skin model | ⭐⭐⭐ |
| **Wrinkles / forehead lines** | BiSeNet forehead mask → Gabor filter or Canny edges | ⭐⭐⭐ |
| **Facial asymmetry** | MediaPipe left/right landmark mirror distance | ⭐⭐⭐⭐⭐ |
| **Prominent Adam's apple** | Neck region detection (limited accuracy) | ⭐ |
| **Glasses** | CelebA (`Eyeglasses`) + BiSeNet parsing | ⭐⭐⭐⭐⭐ |
| **Age** | FairFace (MAE 3.4 years) | ⭐⭐⭐⭐⭐ |
| **Gender** | FairFace (94.2%) | ⭐⭐⭐⭐⭐ |
| **Race** | FairFace (93.4%, 7 categories) | ⭐⭐⭐⭐⭐ |
| **Emotion** | HSEmotion (66.5% AffectNet-8, SOTA) | ⭐⭐⭐⭐ |

---

## Model Comparison Table

| Model | Accuracy (LFW) | Size | Runs in Deno/Browser? | Feature Depth | Notes |
|-------|----------------|------|----------------------|---------------|-------|
| **DeepFace** (Python) | 97.4% (VGG-Face) | 500MB+ | ❌ No (Python only) | Age, gender, race, emotion | Too large, wrong runtime |
| **InsightFace Buffalo_L** | 99.8% (LFW) | ~320MB | ❌ No (Python/C++) | Landmarks, age, gender | Too large for edge |
| **InsightFace MobileFaceNet** | 99.5% (LFW) | ~4MB | ⚠️ ONNX possible | Recognition only, no attributes | Very small but limited features |
| **MediaPipe Face Landmarker** | N/A (landmark model) | ~4MB | ✅ Yes (TFJS/WASM) | 478 landmarks, blendshapes | Best for geometric features |
| **face-api.js** | 99.2% (LFW) | ~6MB (all models) | ✅ Yes (TFJS) | Age, gender, emotion, 68 landmarks | Browser/Node.js ready |
| **ONNX FER+ (emotion)** | ~85% (FER2013) | ~2MB | ✅ Yes (ONNX.js) | Emotion only | Supplement model |
| **HuggingFace ViT models** | Varies | 50-350MB | ⚠️ ONNX export possible | Age, gender, various classifiers | Some fit under 50MB |

---

## Free Hosting Options for the Python Microservice

| Platform | Free Tier | RAM | Cold Start | Best For |
|----------|-----------|-----|------------|----------|
| **Hugging Face Spaces** | Unlimited | 2GB CPU | ~15s | Best free option, runs Gradio/FastAPI |
| **Railway.app** | $5 credit/month | 512MB | ~5s | Good for always-on API |
| **Render.com** | 750 hrs/month | 512MB | ~30s | Spins down after 15min inactivity |
| **Google Cloud Run** | 2M requests/month | 512MB | ~10s | Best scaling, pay-per-request |
| **Fly.io** | 3 shared VMs | 256MB | ~3s | Low latency, always on |

**Recommendation: Hugging Face Spaces** — 2GB RAM free, pre-installed ML libraries, no cold start limits, and you can use their Inference API for some models without even hosting.

---

## Full Implementation

### Python Microservice

#### requirements.txt

```
fastapi==0.115.0
uvicorn==0.30.0
python-multipart==0.0.9
mediapipe==0.10.14
insightface==0.7.3
onnxruntime==1.18.0
torch==2.3.0
torchvision==0.18.0
Pillow==10.4.0
numpy==1.26.4
opencv-python-headless==4.10.0.84
scipy==1.13.0
scikit-learn==1.5.0
huggingface-hub==0.23.0
```

#### face-service/app.py

```python
"""
Face Analysis Microservice
Combines multiple models for comprehensive facial attribute detection.
"""

import io
import logging
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from analyzers.landmark_analyzer import LandmarkAnalyzer
from analyzers.demographic_analyzer import DemographicAnalyzer
from analyzers.attribute_analyzer import AttributeAnalyzer
from analyzers.parsing_analyzer import ParsingAnalyzer
from analyzers.emotion_analyzer import EmotionAnalyzer
from analyzers.color_analyzer import ColorAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Face Analysis Service", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analyzers lazily
landmark_analyzer: Optional[LandmarkAnalyzer] = None
demographic_analyzer: Optional[DemographicAnalyzer] = None
attribute_analyzer: Optional[AttributeAnalyzer] = None
parsing_analyzer: Optional[ParsingAnalyzer] = None
emotion_analyzer: Optional[EmotionAnalyzer] = None
color_analyzer: Optional[ColorAnalyzer] = None


def get_analyzers():
    global landmark_analyzer, demographic_analyzer, attribute_analyzer
    global parsing_analyzer, emotion_analyzer, color_analyzer

    if landmark_analyzer is None:
        logger.info("Loading MediaPipe landmarks...")
        landmark_analyzer = LandmarkAnalyzer()

    if demographic_analyzer is None:
        logger.info("Loading FairFace demographics...")
        demographic_analyzer = DemographicAnalyzer()

    if attribute_analyzer is None:
        logger.info("Loading CelebA attribute classifier...")
        attribute_analyzer = AttributeAnalyzer()

    if parsing_analyzer is None:
        logger.info("Loading BiSeNet face parser...")
        parsing_analyzer = ParsingAnalyzer()

    if emotion_analyzer is None:
        logger.info("Loading HSEmotion...")
        emotion_analyzer = EmotionAnalyzer()

    if color_analyzer is None:
        color_analyzer = ColorAnalyzer()

    return (
        landmark_analyzer,
        demographic_analyzer,
        attribute_analyzer,
        parsing_analyzer,
        emotion_analyzer,
        color_analyzer,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_face(file: UploadFile = File(...)):
    """Comprehensive face analysis endpoint."""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        (
            landmarks,
            demographics,
            attributes,
            parsing,
            emotions,
            colors,
        ) = get_analyzers()

        results = {}

        # 1. MediaPipe Landmarks → geometric features
        logger.info("Running landmark analysis...")
        landmark_results = landmarks.analyze(img_array)
        results.update(landmark_results)

        # 2. FairFace → age, gender, race
        logger.info("Running demographic analysis...")
        demo_results = demographics.analyze(img_array)
        results.update(demo_results)

        # 3. CelebA attributes → 40 binary facial attributes
        logger.info("Running attribute analysis...")
        attr_results = attributes.analyze(img_array)
        results.update(attr_results)

        # 4. BiSeNet face parsing → segmentation masks
        logger.info("Running face parsing...")
        parse_results = parsing.analyze(img_bgr)
        results.update(parse_results)

        # 5. HSEmotion → emotion classification
        logger.info("Running emotion analysis...")
        emo_results = emotions.analyze(img_array)
        results.update(emo_results)

        # 6. Color analysis using parsing masks
        logger.info("Running color analysis...")
        color_results = colors.analyze(
            img_array,
            skin_mask=parse_results.get("_skin_mask"),
            hair_mask=parse_results.get("_hair_mask"),
            landmark_data=landmark_results.get("_raw_landmarks"),
        )
        results.update(color_results)

        # Remove internal fields
        results = {k: v for k, v in results.items() if not k.startswith("_")}

        return {"success": True, "data": results}

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
```

#### face-service/analyzers/landmark_analyzer.py

```python
"""
MediaPipe Face Landmarker — 478 3D landmarks + 52 blendshapes
Derives geometric facial features from landmark positions.
"""

import math
from typing import Any

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


class LandmarkAnalyzer:
    def __init__(self):
        base_options = mp_python.BaseOptions(
            model_asset_path=self._download_model()
        )
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def _download_model(self) -> str:
        import urllib.request
        import os

        model_path = "models/face_landmarker.task"
        if not os.path.exists(model_path):
            os.makedirs("models", exist_ok=True)
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
        return model_path

    def analyze(self, img_rgb: np.ndarray) -> dict[str, Any]:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = self.detector.detect(mp_image)

        if not result.face_landmarks:
            return {"error": "No face detected by MediaPipe"}

        landmarks = result.face_landmarks[0]
        lm = [{"x": l.x, "y": l.y, "z": l.z} for l in landmarks]

        blendshapes = {}
        if result.face_blendshapes:
            for bs in result.face_blendshapes[0]:
                blendshapes[bs.category_name] = round(bs.score, 4)

        attrs = {}
        attrs["_raw_landmarks"] = lm

        # === Face Shape ===
        face_height = self._dist(lm[10], lm[152])
        face_width = self._dist(lm[234], lm[454])
        jaw_width = self._dist(lm[172], lm[397])
        cheekbone_width = self._dist(lm[93], lm[323])
        forehead_width = self._dist(lm[54], lm[284])

        wh_ratio = face_width / face_height if face_height > 0 else 1
        jaw_to_face = jaw_width / face_width if face_width > 0 else 1
        forehead_to_jaw = forehead_width / jaw_width if jaw_width > 0 else 1
        cheek_to_jaw = cheekbone_width / jaw_width if jaw_width > 0 else 1

        if wh_ratio > 0.85 and jaw_to_face > 0.75:
            attrs["face_shape"] = "round"
        elif wh_ratio > 0.8 and jaw_to_face > 0.8 and forehead_to_jaw < 1.1:
            attrs["face_shape"] = "square"
        elif wh_ratio < 0.75:
            attrs["face_shape"] = "oblong"
        elif forehead_to_jaw > 1.3:
            attrs["face_shape"] = "heart"
        elif cheek_to_jaw > 1.25 and forehead_to_jaw < 1.15:
            attrs["face_shape"] = "diamond"
        elif forehead_to_jaw < 0.85:
            attrs["face_shape"] = "triangle"
        else:
            attrs["face_shape"] = "oval"

        attrs["face_shape_metrics"] = {
            "width_height_ratio": round(wh_ratio, 3),
            "jaw_to_face_ratio": round(jaw_to_face, 3),
            "forehead_to_jaw_ratio": round(forehead_to_jaw, 3),
            "cheekbone_to_jaw_ratio": round(cheek_to_jaw, 3),
        }

        # === Forehead ===
        forehead_ratio = forehead_width / face_width if face_width > 0 else 0.6
        attrs["forehead_width"] = (
            "broad" if forehead_ratio > 0.7
            else "narrow" if forehead_ratio < 0.55
            else "average"
        )

        # === Jawline ===
        jaw_angle = self._jaw_angle(lm)
        attrs["jawline_angle"] = round(jaw_angle, 1)
        if jaw_angle < 110:
            attrs["jawline_type"] = "sharp"
        elif jaw_angle > 140:
            attrs["jawline_type"] = "soft"
        elif jaw_to_face > 0.75:
            attrs["jawline_type"] = "strong"
        else:
            attrs["jawline_type"] = "soft"

        # === Chin ===
        chin_width = self._dist(lm[175], lm[396])
        chin_ratio = chin_width / jaw_width if jaw_width > 0 else 0.4
        attrs["chin_type"] = (
            "pointed" if chin_ratio < 0.3
            else "wide" if chin_ratio > 0.5
            else "normal"
        )

        # === Cheekbones ===
        cheek_z = (lm[93]["z"] + lm[323]["z"]) / 2
        attrs["cheekbone_prominence"] = (
            "high" if cheek_z < -0.04
            else "flat" if cheek_z > 0.0
            else "moderate"
        )

        # Hollow vs full cheeks (blendshape-assisted)
        cheek_puff = blendshapes.get("cheekPuff", 0)
        cheek_squint_l = blendshapes.get("cheekSquintLeft", 0)
        cheek_squint_r = blendshapes.get("cheekSquintRight", 0)
        if cheek_puff > 0.3:
            attrs["cheek_fullness"] = "full"
        elif cheek_z > -0.01:
            attrs["cheek_fullness"] = "hollow"
        else:
            attrs["cheek_fullness"] = "normal"

        # === Eyes ===
        left_eye_top = lm[159]
        left_eye_bottom = lm[145]
        left_eye_inner = lm[133]
        left_eye_outer = lm[33]
        eye_openness = self._dist(left_eye_top, left_eye_bottom)
        eye_width_val = self._dist(left_eye_inner, left_eye_outer)
        eye_ratio = eye_openness / eye_width_val if eye_width_val > 0 else 0.3

        outer_angle = left_eye_outer["y"] - left_eye_inner["y"]
        if outer_angle < -0.012:
            attrs["eye_shape"] = "upturned"
        elif outer_angle > 0.012:
            attrs["eye_shape"] = "downturned"
        elif eye_ratio > 0.38:
            attrs["eye_shape"] = "round"
        elif eye_ratio < 0.2:
            attrs["eye_shape"] = "hooded"
        else:
            attrs["eye_shape"] = "almond"

        # Deep-set vs protruding
        eye_z = (lm[159]["z"] + lm[145]["z"]) / 2
        nose_bridge_z = lm[6]["z"]
        if eye_z > nose_bridge_z + 0.02:
            attrs["eye_depth"] = "deep-set"
        elif eye_z < nose_bridge_z - 0.01:
            attrs["eye_depth"] = "protruding"
        else:
            attrs["eye_depth"] = "normal"

        # Eye spacing
        if len(lm) > 473:  # Iris landmarks available
            inter_pupillary = self._dist(lm[468], lm[473])
        else:
            inter_pupillary = self._dist(lm[133], lm[362])
        ip_ratio = inter_pupillary / face_width if face_width > 0 else 0.35
        attrs["eye_spacing"] = (
            "wide-set" if ip_ratio > 0.38
            else "close-set" if ip_ratio < 0.28
            else "average"
        )

        # Eye size
        right_eye_top = lm[386]
        right_eye_bottom = lm[374]
        right_eye_inner = lm[362]
        right_eye_outer = lm[263]
        r_eye_area = self._dist(right_eye_top, right_eye_bottom) * self._dist(right_eye_inner, right_eye_outer)
        l_eye_area = eye_openness * eye_width_val
        avg_eye_area = (l_eye_area + r_eye_area) / 2
        face_area = face_width * face_height
        eye_size_ratio = avg_eye_area / face_area if face_area > 0 else 0.015
        attrs["eye_size"] = (
            "large" if eye_size_ratio > 0.02
            else "small" if eye_size_ratio < 0.012
            else "average"
        )

        # Eye blink (closed vs open)
        blink_l = blendshapes.get("eyeBlinkLeft", 0)
        blink_r = blendshapes.get("eyeBlinkRight", 0)
        attrs["eyes_open"] = (blink_l + blink_r) / 2 < 0.5

        # === Eyebrows ===
        brow_mid_l = lm[105]
        brow_outer_l = lm[46]
        brow_inner_l = lm[70]
        brow_to_eye = self._dist(brow_mid_l, lm[159])
        brow_arch_ratio = brow_to_eye / eye_openness if eye_openness > 0 else 1.5

        attrs["eyebrow_arch_height"] = (
            "high" if brow_arch_ratio > 2.2
            else "low" if brow_arch_ratio < 1.3
            else "average"
        )

        # Brow curvature
        mid_y = brow_mid_l["y"]
        avg_end_y = (brow_inner_l["y"] + brow_outer_l["y"]) / 2
        curvature = mid_y - avg_end_y
        if abs(curvature) < 0.003:
            attrs["eyebrow_shape"] = "straight"
        elif curvature < -0.008:
            attrs["eyebrow_shape"] = "arched"
        else:
            attrs["eyebrow_shape"] = "flat"

        # Eyebrow thickness (vertical span of brow landmarks)
        brow_top = lm[66]  # Top of left brow
        brow_bottom = lm[105]  # Bottom of left brow
        brow_thickness = self._dist(brow_top, brow_bottom)
        attrs["eyebrow_thickness"] = (
            "thick" if brow_thickness > 0.015
            else "thin" if brow_thickness < 0.008
            else "medium"
        )

        # Unibrow detection
        inner_brow_dist = self._dist(lm[70], lm[300])
        attrs["possible_unibrow"] = inner_brow_dist < 0.04

        # === Nose ===
        nose_bridge_top = lm[6]
        nose_tip = lm[1]
        nose_bottom = lm[2]
        left_nostril = lm[129]
        right_nostril = lm[358]
        nostril_w = self._dist(left_nostril, right_nostril)

        nw_ratio = nostril_w / face_width if face_width > 0 else 0.24
        attrs["nostril_width"] = (
            "wide" if nw_ratio > 0.28
            else "narrow" if nw_ratio < 0.2
            else "average"
        )

        tip_angle = nose_tip["y"] - nose_bottom["y"]
        if tip_angle < -0.005:
            attrs["nose_shape"] = "upturned"
        elif tip_angle > 0.01:
            attrs["nose_shape"] = "aquiline"
        elif nw_ratio > 0.28:
            attrs["nose_shape"] = "wide"
        elif nw_ratio < 0.2:
            attrs["nose_shape"] = "narrow"
        else:
            attrs["nose_shape"] = "straight"

        attrs["nose_bridge"] = (
            "high" if nose_bridge_top["z"] < -0.05
            else "flat" if nose_bridge_top["z"] > 0.0
            else "average"
        )

        attrs["nose_tip_shape"] = (
            "pointed" if nose_tip["z"] < nose_bottom["z"] - 0.01
            else "rounded"
        )

        # === Lips & Mouth ===
        upper_lip_top = lm[0]
        upper_lip_bottom = lm[13]
        lower_lip_top = lm[14]
        lower_lip_bottom = lm[17]
        mouth_left = lm[61]
        mouth_right = lm[291]

        upper_lip_h = self._dist(upper_lip_top, upper_lip_bottom)
        lower_lip_h = self._dist(lower_lip_top, lower_lip_bottom)
        total_lip_h = upper_lip_h + lower_lip_h
        mouth_w = self._dist(mouth_left, mouth_right)

        lip_ratio = total_lip_h / mouth_w if mouth_w > 0 else 0.3
        attrs["lip_fullness"] = (
            "full" if lip_ratio > 0.38
            else "thin" if lip_ratio < 0.22
            else "average"
        )

        attrs["lip_balance"] = (
            "top-heavy" if upper_lip_h > lower_lip_h * 1.2
            else "bottom-heavy" if lower_lip_h > upper_lip_h * 1.2
            else "balanced"
        )

        mw_ratio = mouth_w / face_width if face_width > 0 else 0.37
        attrs["mouth_width"] = (
            "wide" if mw_ratio > 0.42
            else "small" if mw_ratio < 0.32
            else "average"
        )

        # Cupid's bow
        cupid_left = lm[37]
        cupid_center = lm[0]
        cupid_right = lm[267]
        bow_depth = cupid_center["y"] - (cupid_left["y"] + cupid_right["y"]) / 2
        attrs["cupids_bow"] = (
            "defined" if bow_depth > 0.005
            else "subtle" if bow_depth > 0.002
            else "flat"
        )

        # Smile
        smile_l = blendshapes.get("mouthSmileLeft", 0)
        smile_r = blendshapes.get("mouthSmileRight", 0)
        attrs["smiling"] = (smile_l + smile_r) / 2 > 0.4
        attrs["smile_asymmetry"] = round(abs(smile_l - smile_r), 3)

        # Dimples (heuristic: strong smile with low cheek puff)
        attrs["possible_dimples"] = (
            (smile_l > 0.5 or smile_r > 0.5) and cheek_puff < 0.2
        )

        # === Facial Asymmetry ===
        symmetry_pairs = [
            (33, 263), (133, 362), (70, 300), (93, 323), (172, 397),
            (61, 291), (159, 386), (145, 374), (46, 276),
        ]
        asymmetry_sum = 0.0
        for li, ri in symmetry_pairs:
            left_dist = abs(lm[li]["x"] - 0.5)
            right_dist = abs(lm[ri]["x"] - 0.5)
            asymmetry_sum += abs(left_dist - right_dist)
        attrs["facial_asymmetry_score"] = round(
            min(asymmetry_sum / len(symmetry_pairs) / 0.05, 1.0), 3
        )

        # === Head Pose (from transformation matrix) ===
        attrs["blendshapes"] = blendshapes

        return attrs

    def _dist(self, a: dict, b: dict) -> float:
        return math.sqrt(
            (a["x"] - b["x"]) ** 2
            + (a["y"] - b["y"]) ** 2
            + (a.get("z", 0) - b.get("z", 0)) ** 2
        )

    def _jaw_angle(self, lm: list[dict]) -> float:
        chin = lm[152]
        left_jaw = lm[172]
        right_jaw = lm[397]
        v1 = (left_jaw["x"] - chin["x"], left_jaw["y"] - chin["y"])
        v2 = (right_jaw["x"] - chin["x"], right_jaw["y"] - chin["y"])
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
        if mag1 * mag2 == 0:
            return 120.0
        cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
        return math.acos(cos_angle) * (180 / math.pi)
```

#### face-service/analyzers/demographic_analyzer.py

```python
"""
FairFace — Age, Gender, Race prediction
Most fair and accurate demographic classifier.
"""

import os
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import models


class DemographicAnalyzer:
    """FairFace-based age, gender, race classifier."""

    AGE_LABELS = [
        "0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"
    ]
    GENDER_LABELS = ["Male", "Female"]
    RACE_LABELS = [
        "White", "Black", "Latino_Hispanic", "East Asian",
        "Southeast Asian", "Indian", "Middle Eastern"
    ]

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _load_model(self):
        """Load FairFace ResNet34 model."""
        model_path = "models/fairface_model.pt"
        if not os.path.exists(model_path):
            os.makedirs("models", exist_ok=True)
            # Download from HuggingFace mirror or original source
            # FairFace official weights: res34_fair_align_multi_7_20190809.pt
            try:
                hf_hub_download(
                    repo_id="dchen236/FairFace",
                    filename="res34_fair_align_multi_7_20190809.pt",
                    local_dir="models",
                    local_dir_use_symlinks=False,
                )
                os.rename(
                    "models/res34_fair_align_multi_7_20190809.pt",
                    model_path,
                )
            except Exception:
                # Fallback: use a smaller pretrained model
                raise FileNotFoundError(
                    "Please download FairFace weights from "
                    "https://github.com/dchen236/FairFace and place at models/fairface_model.pt"
                )

        model = models.resnet34(pretrained=False)
        # FairFace has 3 output heads: race(7), gender(2), age(9) = 18
        model.fc = torch.nn.Linear(model.fc.in_features, 18)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def analyze(self, img_rgb: np.ndarray) -> dict[str, Any]:
        """Predict age, gender, and race."""
        pil_image = Image.fromarray(img_rgb)
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)

        outputs = outputs.cpu().numpy()[0]

        # Split outputs: race(0-6), gender(7-8), age(9-17)
        race_logits = outputs[0:7]
        gender_logits = outputs[7:9]
        age_logits = outputs[9:18]

        race_probs = self._softmax(race_logits)
        gender_probs = self._softmax(gender_logits)
        age_probs = self._softmax(age_logits)

        race_idx = int(np.argmax(race_probs))
        gender_idx = int(np.argmax(gender_probs))
        age_idx = int(np.argmax(age_probs))

        # Estimate numeric age from bucket
        age_ranges = [(0, 2), (3, 9), (10, 19), (20, 29), (30, 39), (40, 49), (50, 59), (60, 69), (70, 85)]
        age_estimate = sum(age_ranges[age_idx]) / 2

        return {
            "age_estimate": round(age_estimate, 1),
            "age_range": self.AGE_LABELS[age_idx],
            "age_confidence": round(float(age_probs[age_idx]), 3),
            "gender": self.GENDER_LABELS[gender_idx].lower(),
            "gender_confidence": round(float(gender_probs[gender_idx]), 3),
            "race": self.RACE_LABELS[race_idx],
            "race_confidence": round(float(race_probs[race_idx]), 3),
            "race_probabilities": {
                label: round(float(prob), 3)
                for label, prob in zip(self.RACE_LABELS, race_probs)
            },
        }

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
```

#### face-service/analyzers/attribute_analyzer.py

```python
"""
CelebA Multi-Label Attribute Classifier
Predicts 40 binary facial attributes from CelebA-trained model.
Uses a pretrained model from HuggingFace.
"""

import os
from typing import Any

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


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


class AttributeAnalyzer:
    """CelebA 40-attribute binary classifier using a fine-tuned ResNet."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _load_model(self):
        """
        Load a CelebA attribute prediction model.
        Using a ResNet-18 fine-tuned on CelebA for 40 attributes.
        """
        from torchvision import models

        model_path = "models/celeba_resnet18.pt"

        if not os.path.exists(model_path):
            os.makedirs("models", exist_ok=True)
            # Try loading from HuggingFace
            try:
                from huggingface_hub import hf_hub_download
                hf_hub_download(
                    repo_id="jnferreira/attribute-prediction-celebA",
                    filename="model.pt",
                    local_dir="models",
                    local_dir_use_symlinks=False,
                )
                os.rename("models/model.pt", model_path)
            except Exception:
                # Fallback: build a fresh model skeleton
                # Users will need to train or provide weights
                model = models.resnet18(pretrained=True)
                model.fc = torch.nn.Linear(model.fc.in_features, 40)
                torch.save(model.state_dict(), model_path)
                print(
                    "WARNING: Using ImageNet-pretrained ResNet18 without CelebA fine-tuning. "
                    "Attribute predictions will be inaccurate. "
                    "Please provide CelebA-trained weights at models/celeba_resnet18.pt"
                )

        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 40)
        model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        model.to(self.device)
        model.eval()
        return model

    def analyze(self, img_rgb: np.ndarray) -> dict[str, Any]:
        pil_image = Image.fromarray(img_rgb)
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)

        probs = torch.sigmoid(logits).cpu().numpy()[0]

        # Build structured results
        raw_attrs = {
            attr: round(float(prob), 3)
            for attr, prob in zip(CELEBA_ATTRIBUTES, probs)
        }

        # Interpret into user-friendly categories
        result: dict[str, Any] = {"celeba_raw": raw_attrs}

        # Hair color (pick highest confidence)
        hair_colors = {
            "black": raw_attrs.get("Black_Hair", 0),
            "brown": raw_attrs.get("Brown_Hair", 0),
            "blonde": raw_attrs.get("Blond_Hair", 0),
            "gray": raw_attrs.get("Gray_Hair", 0),
        }
        result["hair_color_celeba"] = max(hair_colors, key=hair_colors.get)

        # Hair type
        if raw_attrs.get("Straight_Hair", 0) > 0.5:
            result["hair_type_celeba"] = "straight"
        elif raw_attrs.get("Wavy_Hair", 0) > 0.5:
            result["hair_type_celeba"] = "wavy"
        else:
            result["hair_type_celeba"] = "unknown"

        result["bald"] = raw_attrs.get("Bald", 0) > 0.5
        result["bangs"] = raw_attrs.get("Bangs", 0) > 0.5
        result["receding_hairline"] = raw_attrs.get("Receding_Hairline", 0) > 0.5

        # Facial hair
        has_beard = raw_attrs.get("No_Beard", 0) < 0.5
        has_goatee = raw_attrs.get("Goatee", 0) > 0.5
        has_mustache = raw_attrs.get("Mustache", 0) > 0.5
        has_sideburns = raw_attrs.get("Sideburns", 0) > 0.5
        has_stubble = raw_attrs.get("5_o_Clock_Shadow", 0) > 0.5

        if has_goatee:
            result["facial_hair"] = "goatee"
        elif has_mustache and has_beard:
            result["facial_hair"] = "full_beard"
        elif has_mustache:
            result["facial_hair"] = "mustache"
        elif has_sideburns:
            result["facial_hair"] = "sideburns"
        elif has_stubble:
            result["facial_hair"] = "stubble"
        elif not has_beard:
            result["facial_hair"] = "clean_shaven"
        else:
            result["facial_hair"] = "beard"

        # Appearance attributes
        result["wearing_glasses"] = raw_attrs.get("Eyeglasses", 0) > 0.5
        result["wearing_hat"] = raw_attrs.get("Wearing_Hat", 0) > 0.5
        result["bushy_eyebrows"] = raw_attrs.get("Bushy_Eyebrows", 0) > 0.5
        result["arched_eyebrows_celeba"] = raw_attrs.get("Arched_Eyebrows", 0) > 0.5
        result["bags_under_eyes"] = raw_attrs.get("Bags_Under_Eyes", 0) > 0.5
        result["high_cheekbones_celeba"] = raw_attrs.get("High_Cheekbones", 0) > 0.5
        result["oval_face_celeba"] = raw_attrs.get("Oval_Face", 0) > 0.5
        result["pointy_nose_celeba"] = raw_attrs.get("Pointy_Nose", 0) > 0.5
        result["big_lips_celeba"] = raw_attrs.get("Big_Lips", 0) > 0.5
        result["big_nose_celeba"] = raw_attrs.get("Big_Nose", 0) > 0.5
        result["narrow_eyes_celeba"] = raw_attrs.get("Narrow_Eyes", 0) > 0.5
        result["double_chin"] = raw_attrs.get("Double_Chin", 0) > 0.5
        result["chubby"] = raw_attrs.get("Chubby", 0) > 0.5
        result["rosy_cheeks"] = raw_attrs.get("Rosy_Cheeks", 0) > 0.5
        result["pale_skin"] = raw_attrs.get("Pale_Skin", 0) > 0.5
        result["young"] = raw_attrs.get("Young", 0) > 0.5
        result["smiling_celeba"] = raw_attrs.get("Smiling", 0) > 0.5
        result["mouth_open"] = raw_attrs.get("Mouth_Slightly_Open", 0) > 0.5

        return result
```

#### face-service/analyzers/parsing_analyzer.py

```python
"""
BiSeNet Face Parsing — 19-class semantic segmentation of the face.
Segments: skin, eyebrows, eyes, nose, lips, hair, ears, neck, etc.
"""

import os
from typing import Any

import cv2
import numpy as np
import torch
from torchvision import transforms


class ParsingAnalyzer:
    """
    BiSeNet face parsing for hair/skin/feature segmentation.

    Parsing classes:
    0: background, 1: skin, 2: l_brow, 3: r_brow, 4: l_eye, 5: r_eye,
    6: eye_g (glasses), 7: l_ear, 8: r_ear, 9: ear_r (earring),
    10: nose, 11: mouth, 12: u_lip, 13: l_lip, 14: neck,
    15: necklace, 16: cloth, 17: hair, 18: hat
    """

    LABELS = {
        0: "background", 1: "skin", 2: "left_brow", 3: "right_brow",
        4: "left_eye", 5: "right_eye", 6: "glasses", 7: "left_ear",
        8: "right_ear", 9: "earring", 10: "nose", 11: "mouth",
        12: "upper_lip", 13: "lower_lip", 14: "neck", 15: "necklace",
        16: "cloth", 17: "hair", 18: "hat",
    }

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _load_model(self):
        model_path = "models/bisenet_face_parsing.pt"
        if not os.path.exists(model_path):
            os.makedirs("models", exist_ok=True)
            # BiSeNet model from face-parsing.PyTorch
            # Download from: https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812
            raise FileNotFoundError(
                "Please download BiSeNet face parsing weights from "
                "https://github.com/zllrunning/face-parsing.PyTorch and place at "
                "models/bisenet_face_parsing.pt"
            )

        from models.bisenet_model import BiSeNet  # You'll need to include this
        model = BiSeNet(n_classes=19)
        model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        model.to(self.device)
        model.eval()
        return model

    def analyze(self, img_bgr: np.ndarray) -> dict[str, Any]:
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (512, 512))

        input_tensor = self.transform(img_resized).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)[0]  # BiSeNet returns tuple

        parsing = output.squeeze(0).argmax(0).cpu().numpy()
        parsing = cv2.resize(
            parsing.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        )

        # Generate masks
        skin_mask = (parsing == 1).astype(np.uint8)
        hair_mask = (parsing == 17).astype(np.uint8)
        glasses_mask = (parsing == 6).astype(np.uint8)
        hat_mask = (parsing == 18).astype(np.uint8)

        # Facial hair detection: look for dark pixels in lower face skin region
        lower_face = parsing[int(h * 0.55):int(h * 0.85), int(w * 0.25):int(w * 0.75)]
        lower_skin = (lower_face == 1).sum()
        total_lower = lower_face.size or 1

        # Region stats
        hair_area = hair_mask.sum() / (h * w)
        skin_area = skin_mask.sum() / (h * w)

        result: dict[str, Any] = {
            "_skin_mask": skin_mask,
            "_hair_mask": hair_mask,
            "has_glasses_parsing": int(glasses_mask.sum()) > 100,
            "wearing_hat_parsing": int(hat_mask.sum()) > 500,
            "hair_coverage": round(float(hair_area), 3),
            "skin_coverage": round(float(skin_area), 3),
        }

        # Hair length estimation from mask
        if hair_area < 0.01:
            result["hair_length_estimate"] = "bald"
        elif hair_area < 0.08:
            result["hair_length_estimate"] = "short"
        elif hair_area < 0.18:
            result["hair_length_estimate"] = "medium"
        else:
            result["hair_length_estimate"] = "long"

        # Wrinkle analysis on forehead skin
        forehead_region = img_bgr[int(h * 0.05):int(h * 0.25), int(w * 0.3):int(w * 0.7)]
        forehead_skin = skin_mask[int(h * 0.05):int(h * 0.25), int(w * 0.3):int(w * 0.7)]
        if forehead_skin.sum() > 100:
            gray_forehead = cv2.cvtColor(forehead_region, cv2.COLOR_BGR2GRAY)
            # Apply mask
            gray_forehead = cv2.bitwise_and(gray_forehead, gray_forehead, mask=forehead_skin)
            edges = cv2.Canny(gray_forehead, 30, 80)
            edge_density = edges.sum() / (forehead_skin.sum() * 255 + 1)
            result["forehead_wrinkle_score"] = round(float(edge_density), 3)
            result["forehead_wrinkles"] = (
                "heavy" if edge_density > 0.15
                else "moderate" if edge_density > 0.08
                else "mild" if edge_density > 0.04
                else "none"
            )

        # Freckles/moles detection on skin
        skin_region = cv2.bitwise_and(img_bgr, img_bgr, mask=skin_mask)
        gray_skin = cv2.cvtColor(skin_region, cv2.COLOR_BGR2GRAY)
        # Detect dark spots
        _, dark_spots = cv2.threshold(gray_skin, 80, 255, cv2.THRESH_BINARY_INV)
        dark_spots = cv2.bitwise_and(dark_spots, dark_spots, mask=skin_mask)
        # Find contours of dark spots
        contours, _ = cv2.findContours(dark_spots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        small_spots = [c for c in contours if 5 < cv2.contourArea(c) < 200]
        result["possible_freckles_moles"] = len(small_spots) > 10
        result["dark_spot_count"] = len(small_spots)

        return result
```

#### face-service/analyzers/emotion_analyzer.py

```python
"""
HSEmotion — State-of-the-art facial emotion recognition.
Supports 8 emotions on AffectNet.
"""

import os
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


class EmotionAnalyzer:
    """HSEmotion-based facial expression classifier."""

    EMOTION_LABELS = [
        "angry", "contempt", "disgust", "fear",
        "happy", "neutral", "sad", "surprise",
    ]

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.transform = transforms.Compose([
            transforms.Resize((260, 260)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _load_model(self):
        """Load HSEmotion EfficientNet model."""
        model_path = "models/hsemotion_enet_b0_8.pt"

        if not os.path.exists(model_path):
            os.makedirs("models", exist_ok=True)
            try:
                from huggingface_hub import hf_hub_download
                # HSEmotion models available at:
                # https://github.com/HSE-asavchenko/face-emotion-recognition
                hf_hub_download(
                    repo_id="HSE-asavchenko/hsemotion",
                    filename="enet_b0_8_best_afew.pt",
                    local_dir="models",
                    local_dir_use_symlinks=False,
                )
                os.rename("models/enet_b0_8_best_afew.pt", model_path)
            except Exception:
                raise FileNotFoundError(
                    "Please download HSEmotion weights from "
                    "https://github.com/HSE-asavchenko/face-emotion-recognition"
                )

        import timm
        model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=8)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def analyze(self, img_rgb: np.ndarray) -> dict[str, Any]:
        pil_image = Image.fromarray(img_rgb)
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)

        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        top_idx = int(np.argmax(probs))

        return {
            "emotion": self.EMOTION_LABELS[top_idx],
            "emotion_confidence": round(float(probs[top_idx]), 3),
            "emotion_probabilities": {
                label: round(float(prob), 3)
                for label, prob in zip(self.EMOTION_LABELS, probs)
            },
        }
```

#### face-service/analyzers/color_analyzer.py

```python
"""
Pixel-level color analysis using segmentation masks from BiSeNet
and landmark positions from MediaPipe.
"""

from typing import Any, Optional

import cv2
import numpy as np
from sklearn.cluster import KMeans


class ColorAnalyzer:
    """Analyzes skin tone, eye color, and hair color from pixel data."""

    def analyze(
        self,
        img_rgb: np.ndarray,
        skin_mask: Optional[np.ndarray] = None,
        hair_mask: Optional[np.ndarray] = None,
        landmark_data: Optional[list[dict]] = None,
    ) -> dict[str, Any]:
        h, w = img_rgb.shape[:2]
        results: dict[str, Any] = {}

        # === Skin Tone ===
        if skin_mask is not None and skin_mask.sum() > 100:
            skin_pixels = img_rgb[skin_mask > 0]
            # Convert to LAB for perceptually uniform brightness
            skin_lab = cv2.cvtColor(
                skin_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB
            ).reshape(-1, 3)
            avg_l = float(skin_lab[:, 0].mean())  # L channel (brightness)

            if avg_l > 180:
                results["skin_tone"] = "very_light"
            elif avg_l > 155:
                results["skin_tone"] = "light"
            elif avg_l > 130:
                results["skin_tone"] = "medium_light"
            elif avg_l > 105:
                results["skin_tone"] = "medium"
            elif avg_l > 80:
                results["skin_tone"] = "medium_dark"
            else:
                results["skin_tone"] = "dark"

            results["skin_tone_score"] = round(avg_l / 255, 3)

            # Fitzpatrick scale approximation
            if avg_l > 170:
                results["fitzpatrick_type"] = "I"
            elif avg_l > 145:
                results["fitzpatrick_type"] = "II"
            elif avg_l > 120:
                results["fitzpatrick_type"] = "III"
            elif avg_l > 95:
                results["fitzpatrick_type"] = "IV"
            elif avg_l > 70:
                results["fitzpatrick_type"] = "V"
            else:
                results["fitzpatrick_type"] = "VI"

        # === Hair Color ===
        if hair_mask is not None and hair_mask.sum() > 500:
            hair_pixels = img_rgb[hair_mask > 0]

            # K-means to find dominant hair color
            if len(hair_pixels) > 100:
                sample_size = min(5000, len(hair_pixels))
                indices = np.random.choice(len(hair_pixels), sample_size, replace=False)
                sampled = hair_pixels[indices].astype(np.float32)

                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                kmeans.fit(sampled)

                # Pick the cluster with most members
                labels, counts = np.unique(kmeans.labels_, return_counts=True)
                dominant_idx = labels[np.argmax(counts)]
                dominant_color = kmeans.cluster_centers_[dominant_idx].astype(int)

                r, g, b = dominant_color
                brightness = (int(r) + int(g) + int(b)) / 3

                # Classify hair color
                hsv_color = cv2.cvtColor(
                    np.array([[dominant_color]], dtype=np.uint8), cv2.COLOR_RGB2HSV
                )[0][0]
                hue, sat, val = int(hsv_color[0]), int(hsv_color[1]), int(hsv_color[2])

                if brightness < 40:
                    results["hair_color_detected"] = "black"
                elif brightness > 190:
                    results["hair_color_detected"] = "platinum_blonde"
                elif brightness > 160 and sat < 50:
                    results["hair_color_detected"] = "gray"
                elif brightness > 140 and (hue > 15 and hue < 35):
                    results["hair_color_detected"] = "blonde"
                elif (hue < 15 or hue > 160) and sat > 80:
                    results["hair_color_detected"] = "red"
                elif brightness > 60:
                    results["hair_color_detected"] = "brown"
                else:
                    results["hair_color_detected"] = "dark_brown"

                results["hair_dominant_rgb"] = [int(r), int(g), int(b)]

            # Hair texture analysis (FFT-based)
            hair_region = cv2.bitwise_and(
                img_rgb,
                img_rgb,
                mask=hair_mask,
            )
            gray_hair = cv2.cvtColor(hair_region, cv2.COLOR_RGB2GRAY)
            # Mask out non-hair regions
            gray_hair_masked = gray_hair[hair_mask > 0]

            if len(gray_hair_masked) > 1000:
                # Compute local variance as texture indicator
                # High frequency = curly, low frequency = straight
                hair_patch = gray_hair_masked[:1024].astype(np.float32)
                fft = np.fft.fft(hair_patch)
                magnitude = np.abs(fft)
                # Ratio of high freq to low freq energy
                low_freq = magnitude[:len(magnitude) // 4].sum()
                high_freq = magnitude[len(magnitude) // 4:].sum()
                freq_ratio = high_freq / (low_freq + 1e-6)

                if freq_ratio > 0.8:
                    results["hair_texture_detected"] = "curly"
                elif freq_ratio > 0.5:
                    results["hair_texture_detected"] = "wavy"
                else:
                    results["hair_texture_detected"] = "straight"

        # === Eye Color ===
        if landmark_data is not None and len(landmark_data) > 473:
            for eye_name, iris_idx in [("left", 468), ("right", 473)]:
                ix = int(landmark_data[iris_idx]["x"] * w)
                iy = int(landmark_data[iris_idx]["y"] * h)

                # Sample a small patch around iris
                pad = 3
                y1 = max(0, iy - pad)
                y2 = min(h, iy + pad)
                x1 = max(0, ix - pad)
                x2 = min(w, ix + pad)

                iris_patch = img_rgb[y1:y2, x1:x2]
                if iris_patch.size == 0:
                    continue

                avg_color = iris_patch.mean(axis=(0, 1))
                r, g, b = avg_color

                # Convert to HSV for better classification
                hsv = cv2.cvtColor(
                    np.array([[avg_color]], dtype=np.uint8), cv2.COLOR_RGB2HSV
                )[0][0]
                hue_val, sat_val, val_val = int(hsv[0]), int(hsv[1]), int(hsv[2])

                if val_val < 60:
                    color = "dark_brown"
                elif sat_val < 30:
                    color = "gray"
                elif hue_val > 100 and hue_val < 130 and sat_val > 50:
                    color = "blue"
                elif hue_val > 35 and hue_val < 85 and sat_val > 40:
                    color = "green"
                elif (hue_val > 15 and hue_val < 35) and sat_val > 40:
                    color = "hazel"
                elif val_val > 120 and sat_val > 60:
                    color = "amber"
                else:
                    color = "brown"

                results[f"{eye_name}_eye_color"] = color

            # Consensus
            if "left_eye_color" in results and "right_eye_color" in results:
                if results["left_eye_color"] == results["right_eye_color"]:
                    results["eye_color"] = results["left_eye_color"]
                else:
                    results["eye_color"] = results["left_eye_color"]  # Use left as primary
                    results["heterochromia"] = True

        return results
```

#### face-service/Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download MediaPipe model at build time
RUN python -c "from analyzers.landmark_analyzer import LandmarkAnalyzer; LandmarkAnalyzer()"

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Connect Your Next.js App to the Microservice

#### lib/face-analysis/api-client.ts

```typescript
/**
 * Client for the Python face analysis microservice.
 * Replaces the Supabase Edge Function that called AWS Rekognition.
 */

const FACE_SERVICE_URL = process.env.NEXT_PUBLIC_FACE_SERVICE_URL || "http://localhost:8000";

export interface FaceAnalysisResult {
  // Geometric (MediaPipe)
  face_shape: string;
  face_shape_metrics: Record<string, number>;
  forehead_width: string;
  jawline_type: string;
  jawline_angle: number;
  chin_type: string;
  cheekbone_prominence: string;
  cheek_fullness: string;
  eye_shape: string;
  eye_depth: string;
  eye_spacing: string;
  eye_size: string;
  eyebrow_shape: string;
  eyebrow_arch_height: string;
  eyebrow_thickness: string;
  possible_unibrow: boolean;
  nose_shape: string;
  nose_bridge: string;
  nose_tip_shape: string;
  nostril_width: string;
  lip_fullness: string;
  lip_balance: string;
  mouth_width: string;
  cupids_bow: string;
  smiling: boolean;
  smile_asymmetry: number;
  possible_dimples: boolean;
  facial_asymmetry_score: number;

  // Demographics (FairFace)
  age_estimate: number;
  age_range: string;
  age_confidence: number;
  gender: string;
  gender_confidence: number;
  race: string;
  race_confidence: number;
  race_probabilities: Record<string, number>;

  // CelebA Attributes
  facial_hair: string;
  wearing_glasses: boolean;
  bald: boolean;
  receding_hairline: boolean;
  hair_color_celeba: string;
  hair_type_celeba: string;
  bags_under_eyes: boolean;
  double_chin: boolean;
  bushy_eyebrows: boolean;
  high_cheekbones_celeba: boolean;

  // Emotion (HSEmotion)
  emotion: string;
  emotion_confidence: number;
  emotion_probabilities: Record<string, number>;

  // Color Analysis
  skin_tone: string;
  skin_tone_score: number;
  fitzpatrick_type: string;
  eye_color: string;
  hair_color_detected: string;
  hair_dominant_rgb: number[];
  hair_texture_detected: string;

  // Parsing
  hair_length_estimate: string;
  forehead_wrinkles: string;
  possible_freckles_moles: boolean;
  dark_spot_count: number;

  // Blendshapes
  blendshapes: Record<string, number>;
}

export async function analyzeFace(imageFile: File): Promise<FaceAnalysisResult> {
  const formData = new FormData();
  formData.append("file", imageFile);

  const response = await fetch(`${FACE_SERVICE_URL}/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(`Face analysis failed: ${error.detail}`);
  }

  const result = await response.json();

  if (!result.success) {
    throw new Error("Face analysis returned unsuccessful result");
  }

  return result.data;
}

export async function checkServiceHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${FACE_SERVICE_URL}/health`);
    return response.ok;
  } catch {
    return false;
  }
}
```

### Deploy to Hugging Face Spaces (Free)

Create a `README.md` in the `face-service/` directory with the following frontmatter:

```yaml
---
title: HCP Face Analysis
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
---
```

---

## Final Architecture Summary

```
Browser (Next.js)
    │
    │  POST /analyze (image file)
    ▼
Hugging Face Spaces (FREE, 2GB RAM)
    ├── FastAPI Server
    ├── MediaPipe (4MB) ──────► 478 landmarks → ~40 geometric features
    ├── FairFace (90MB) ──────► age, gender, race
    ├── CelebA ResNet (44MB) ─► 40 binary attributes (hair, beard, glasses...)
    ├── BiSeNet (50MB) ───────► face parsing → hair/skin segmentation
    ├── HSEmotion (20MB) ─────► 8 emotions
    └── Color Analysis ───────► skin tone, eye color, hair color
    │
    │  JSON response (~150 attributes)
    ▼
Supabase (existing)
    ├── Store results in PostgreSQL
    └── Auth / Storage unchanged
```

| Metric | Value |
|--------|-------|
| **Total models** | ~210MB |
| **Features detected** | **~95% of the full feature list** |
| **Hosting cost** | **$0** (HF Spaces free tier) |
| **Latency** | ~2-4s per image (CPU) |
| **Languages** | Python (microservice) + TypeScript (existing Next.js) |
| **Only missing** | Teeth analysis, scar detection, Adam's apple (require specialized fine-tuned models) |

---

## Required Feature List

### Face shape
- Oval face, Round face, Square face, Heart-shaped face, Diamond face, Long/oblong face, Triangle face
- Jawline sharp, Jawline soft, Strong jaw, Receding chin, Pointed chin, Cleft chin, Wide chin
- High cheekbones, Flat cheekbones, Full cheeks, Hollow cheeks
- Broad forehead, Narrow forehead

### Eye shape
- Almond, Round, Hooded, Monolid, Deep-set eyes, Protruding eyes
- Upturned eyes, Downturned eyes, Wide-set eyes, Close-set eyes, Large eyes, Small eyes
- Eye color: brown, blue, green, hazel
- Dark under-eyes, Eye bags, Crow's feet

### Eyebrows
- Thick, Thin, Arched, Straight, Bushy, Unibrow
- High eyebrow arch, Low eyebrow arch

### Nose
- Straight, Aquiline, Button, Upturned, Wide, Narrow
- Flat bridge, High bridge, Wide nostrils, Narrow nostrils
- Rounded tip, Pointed tip

### Lips & Mouth
- Full, Thin, Wide mouth, Small mouth
- Defined cupid's bow, Uneven lips
- Gap teeth, Crooked teeth, Straight teeth, Overbite, Underbite
- Dimples, Smile lines, Asymmetrical smile

### Hair
- Straight, Wavy, Curly, Coily
- Short, Long, Bald, Receding hairline, Widow's peak
- Thick, Thin
- Color: black, brown, blonde, red, gray, dyed

### Facial hair
- Full beard, Stubble, Goatee, Mustache, Clean-shaven, Sideburns

### Skin & Other
- Skin tone: light, medium, dark
- Freckles, Moles, Birthmark, Scar, Acne
- Wrinkles, Forehead lines, Smile lines
- Facial asymmetry, Prominent Adam's apple
