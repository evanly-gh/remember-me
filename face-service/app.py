"""
HCP Face Analysis Microservice
==============================

FastAPI service that runs twelve specialised analyzers over a single
photo and merges their outputs into one facial-attribute dictionary,
including a face-recognition embedding for cross-photo grouping and a
numeric "chopped score" aesthetic rating.

Pipeline (in execution order)
-----------------------------
1.  InsightFaceAnalyzer        InsightFace buffalo_l (ONNX). SCRFD
                               detection + ArcFace 512-d embedding +
                               106 landmarks. Age & gender delegated
                               to FairFace ViTs (steps 3a / 3b).

2.  LandmarkAnalyzer           MediaPipe Face Landmarker. 478 3D
                               landmarks + 52 ARKit blendshapes →
                               geometric features, smiling, mouth_open.

3a. AgeAnalyzer                FairFace ViT, softmax-weighted across 9
                               age buckets. Replaces the InsightFace
                               age regression which routinely missed
                               by 30+ years on certain face types.

3b. GenderAnalyzer             FairFace ViT (~93.4% acc). Replaces the
                               InsightFace gender head so we get a real
                               softmax confidence instead of argmax 1.0.

3c. EthnicityAnalyzer          cledoux42/Ethnicity_Test_v003 ViT.
                               5-class ethnicity widened to a 7-bucket
                               schema for legacy compatibility.

4.  ParsingAnalyzer            SegFormer-B5 human parsing. Receives the
                               face-cropped image. Emits face/hair
                               masks + hair length + hat detection +
                               OpenCV-derived skin stats.

5.  EmotionAnalyzer            HSEmotion EfficientNet-B0. 8-class
                               emotion + valence/arousal/mood.

6.  ColorAnalyzer              Pure OpenCV LAB/HSV statistics. Uses
                               SegFormer masks + MediaPipe lip/iris
                               landmarks. No ML model.

7.  ObstructionAnalyzer        dima806 ViT-B/16. Glasses, sunglasses,
                               mask. ~99% precision on each.

8.  HairTypeAnalyzer           dima806 ViT-B/16. Curly/dreadlocks/kinky/
                               straight/wavy. ~93% accuracy.

9.  BeautyAnalyzer             Optional. ResNet-50 trained on
                               SCUT-FBP5500 (see training/beauty/).
                               Outputs a 1.0–5.0 beauty score plus a
                               0–100 normalised version. Falls back to
                               None when no weights are loaded — the
                               AestheticAnalyzer then uses rule-based
                               scoring only.

10. AestheticAnalyzer          Pure-Python aggregator. Reads the merged
                               dict from previous analyzers and produces
                               the final `chopped_score` (0–100, higher
                               = more chopped) and a per-factor
                               breakdown.

Endpoints
---------
GET  /                  service banner
GET  /health            liveness check
POST /analyze           multipart file upload
POST /analyze-base64    JSON {"image": "<base64>"}

All analyzers are lazily instantiated on first request to keep
cold-start latency manageable on the Hugging Face Spaces free tier.
"""

import os
# hf_transfer makes initial model downloads from the HF Hub much faster.
# The default HF_HUB_DOWNLOAD_TIMEOUT (10 s) is too short for the larger
# ViT checkpoints on a cold start.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"

import io
import logging
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from analyzers.landmark_analyzer import LandmarkAnalyzer
from analyzers.ethnicity_analyzer import EthnicityAnalyzer
from analyzers.parsing_analyzer import ParsingAnalyzer
from analyzers.emotion_analyzer import EmotionAnalyzer
from analyzers.color_analyzer import ColorAnalyzer
from analyzers.obstruction_analyzer import ObstructionAnalyzer
from analyzers.hair_type_analyzer import HairTypeAnalyzer
from analyzers.insightface_analyzer import InsightFaceAnalyzer
from analyzers.age_analyzer import AgeAnalyzer
from analyzers.gender_analyzer import GenderAnalyzer
from analyzers.beauty_analyzer import BeautyAnalyzer
from analyzers.aesthetic_analyzer import AestheticAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="HCP Face Analysis Service", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your domain in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy slots, one per analyzer. The first request pays the full
# model-load cost; subsequent requests are warm.
insightface_analyzer: Optional[InsightFaceAnalyzer] = None
landmark_analyzer: Optional[LandmarkAnalyzer] = None
age_analyzer: Optional[AgeAnalyzer] = None
gender_analyzer: Optional[GenderAnalyzer] = None
ethnicity_analyzer: Optional[EthnicityAnalyzer] = None
parsing_analyzer: Optional[ParsingAnalyzer] = None
emotion_analyzer: Optional[EmotionAnalyzer] = None
color_analyzer: Optional[ColorAnalyzer] = None
obstruction_analyzer: Optional[ObstructionAnalyzer] = None
hair_type_analyzer: Optional[HairTypeAnalyzer] = None
beauty_analyzer: Optional[BeautyAnalyzer] = None
aesthetic_analyzer: Optional[AestheticAnalyzer] = None


def _to_json_safe(value):
    """Recursively coerce numpy scalars/arrays into JSON-serialisable types.

    Several analyzers return numpy floats/booleans (e.g. from `np.std`
    or boolean mask logic). FastAPI's default JSON encoder doesn't
    handle those, so we normalise everything here before returning.
    """
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(v) for v in value]
    return value


def get_analyzers():
    """Lazy-load all analyzer models on first use.

    Each analyzer is instantiated once per process and reused across
    requests. First request pays the full model-load cost; subsequent
    requests are warm.
    """
    global insightface_analyzer, landmark_analyzer
    global age_analyzer, gender_analyzer, ethnicity_analyzer
    global parsing_analyzer, emotion_analyzer, color_analyzer
    global obstruction_analyzer, hair_type_analyzer
    global beauty_analyzer, aesthetic_analyzer

    if insightface_analyzer is None:
        logger.info("Loading InsightFace buffalo_l bundle...")
        insightface_analyzer = InsightFaceAnalyzer()

    if landmark_analyzer is None:
        logger.info("Loading MediaPipe Face Landmarker...")
        landmark_analyzer = LandmarkAnalyzer()

    if age_analyzer is None:
        logger.info("Loading FairFace age analyzer...")
        age_analyzer = AgeAnalyzer()

    if gender_analyzer is None:
        logger.info("Loading FairFace gender analyzer...")
        gender_analyzer = GenderAnalyzer()

    if ethnicity_analyzer is None:
        logger.info("Loading Ethnicity classifier...")
        ethnicity_analyzer = EthnicityAnalyzer()

    if parsing_analyzer is None:
        logger.info("Loading SegFormer face parser...")
        parsing_analyzer = ParsingAnalyzer()

    if emotion_analyzer is None:
        logger.info("Loading HSEmotion model...")
        emotion_analyzer = EmotionAnalyzer()

    if color_analyzer is None:
        color_analyzer = ColorAnalyzer()

    if obstruction_analyzer is None:
        logger.info("Loading face obstruction classifier...")
        obstruction_analyzer = ObstructionAnalyzer()

    if hair_type_analyzer is None:
        logger.info("Loading hair type classifier...")
        hair_type_analyzer = HairTypeAnalyzer()

    if beauty_analyzer is None:
        logger.info("Loading beauty regressor (or no-op if untrained)...")
        beauty_analyzer = BeautyAnalyzer()

    if aesthetic_analyzer is None:
        aesthetic_analyzer = AestheticAnalyzer()

    return (
        insightface_analyzer,
        landmark_analyzer,
        age_analyzer,
        gender_analyzer,
        ethnicity_analyzer,
        parsing_analyzer,
        emotion_analyzer,
        color_analyzer,
        obstruction_analyzer,
        hair_type_analyzer,
        beauty_analyzer,
        aesthetic_analyzer,
    )


def _crop_to_face(img_rgb: np.ndarray, bbox, padding: float = 0.4) -> np.ndarray:
    """Crop the image to a face-centred rectangle with extra context.

    SegFormer and the ViT classifiers tend to do better with the face
    occupying a large fraction of the input. We pad the InsightFace
    bbox by `padding` (fraction of bbox size) so context like ears,
    hair, and the top of the shoulders is preserved.

    Returns the full image unchanged if bbox is None, malformed, or
    the resulting crop would be degenerate.
    """
    if bbox is None or len(bbox) != 4:
        return img_rgb
    h, w = img_rgb.shape[:2]
    try:
        x1, y1, x2, y2 = bbox
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        pad_x = bw * padding
        pad_y = bh * padding
        cx1 = max(0, int(x1 - pad_x))
        cy1 = max(0, int(y1 - pad_y))
        cx2 = min(w, int(x2 + pad_x))
        cy2 = min(h, int(y2 + pad_y))
        if cx2 - cx1 < 32 or cy2 - cy1 < 32:
            return img_rgb
        return img_rgb[cy1:cy2, cx1:cx2]
    except Exception:
        return img_rgb


def _run_pipeline(img_array: np.ndarray) -> dict:
    """Run all ten analyzers against `img_array` and return the merged dict.

    Shared by /analyze and /analyze-base64. Kept as a function rather
    than inlined twice so the per-step ordering is the single source
    of truth.
    """
    (
        insight,
        landmarks,
        ages,
        genders,
        ethnicities,
        parsing,
        emotions,
        colors,
        obstructions,
        hair_types,
        beauty,
        aesthetics,
    ) = get_analyzers()

    results: dict = {}

    # Step 1: InsightFace — detection + ArcFace 512-d recognition
    # embedding + 106 landmarks. Age and gender both delegated to
    # FairFace ViTs in step 3 because the bundled genderage head was
    # too inaccurate (called 20-yr-olds "52" in real photos).
    logger.info("Running InsightFace analysis...")
    insight_results = insight.analyze(img_array)
    results.update(insight_results)

    # Compute a face crop once and pass it to every downstream analyzer
    # that benefits from it (parsing, ethnicity, obstruction, hair type,
    # beauty regressor). Falls back to the full image when InsightFace
    # didn't find a face.
    face_crop = _crop_to_face(img_array, insight_results.get("face_bbox"))

    # Step 2: MediaPipe landmarks (works on the full image; it has its
    # own internal detector).
    logger.info("Running landmark analysis...")
    landmark_results = landmarks.analyze(img_array)
    results.update(landmark_results)

    # Step 3a: FairFace age. Softmax-weighted estimate across 9
    # buckets — slides between bucket midpoints when the model is
    # uncertain instead of snapping. Much more reliable than
    # InsightFace's regression head on younger faces.
    logger.info("Running FairFace age analysis...")
    results.update(ages.analyze(face_crop))

    # Step 3b: FairFace gender. Provides a real softmax confidence
    # score so the UI can show graded uncertainty.
    logger.info("Running FairFace gender analysis...")
    results.update(genders.analyze(face_crop))

    # Step 3c: ethnicity classifier — likes a tighter face crop.
    logger.info("Running ethnicity analysis...")
    results.update(ethnicities.analyze(face_crop))

    # Step 4: SegFormer parsing on the face crop (cleaner masks).
    logger.info("Running face parsing...")
    parse_results = parsing.analyze(face_crop)
    results.update(parse_results)

    # Step 5: HSEmotion on the face crop.
    logger.info("Running emotion analysis...")
    results.update(emotions.analyze(face_crop))

    # Step 6: pixel-level colour analysis. Uses the face/hair masks
    # from step 4 (already in face-crop coordinate space) and the
    # MediaPipe lip/iris landmarks from step 2 (still in full-image
    # space, normalised). We pass `face_crop` so mask coordinates
    # line up; landmarks are in normalised coordinates so they map
    # correctly to either image.
    logger.info("Running color analysis...")
    color_results = colors.analyze(
        face_crop,
        skin_mask=parse_results.get("_skin_mask"),
        hair_mask=parse_results.get("_hair_mask"),
        landmarks=landmark_results.get("_raw_landmarks"),
    )
    results.update(color_results)

    # Step 7: obstruction classifier — also benefits from a face crop.
    logger.info("Running obstruction analysis...")
    results.update(obstructions.analyze(face_crop))

    # Step 8: hair-type classifier.
    logger.info("Running hair-type analysis...")
    results.update(hair_types.analyze(face_crop))

    # Step 9: learned beauty regressor (no-op if no weights present).
    logger.info("Running beauty regressor...")
    results.update(beauty.analyze(face_crop))

    # Step 10: aesthetic aggregator. Reads the merged dict; no image
    # input. Always runs last so it can see every other analyzer's
    # outputs.
    logger.info("Running aesthetic aggregator...")
    results.update(aesthetics.analyze(results))

    # Drop internal/scratch fields (leading underscore) before
    # returning. Keeps masks and raw landmark lists out of the JSON.
    results = {k: v for k, v in results.items() if not k.startswith("_")}

    return results


@app.get("/")
async def root():
    """Service banner — confirms the server is reachable and which version."""
    return {
        "name": "HCP Face Analysis Service",
        "version": "3.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze",
            "analyze-base64": "/analyze-base64",
        }
    }


@app.get("/health")
async def health():
    """Liveness probe. Used by the Express server and HF Spaces uptime checks."""
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_face(file: UploadFile = File(...)):
    """Multipart endpoint for direct uploads.

    Runs the full ten-step pipeline and returns the merged attribute
    dict. See `analyze_face_base64` for the JSON-body variant the
    Express server calls.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = np.array(image)
        results = _run_pipeline(img_array)
        return {"success": True, "data": _to_json_safe(results)}

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-base64")
async def analyze_face_base64(body: dict):
    """JSON-body endpoint accepting `{"image": "<base64>"}`.

    This is what the Node/Express server forwards client requests to
    so we don't have to push multipart payloads through the proxy.
    The pipeline body is identical to `/analyze`.
    """
    import base64

    try:
        image_b64 = body.get("image", "")
        if not image_b64:
            raise HTTPException(status_code=400, detail="No image data provided")

        # Strip a possible "data:image/...;base64," prefix.
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]

        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(image)
        results = _run_pipeline(img_array)
        return {"success": True, "data": _to_json_safe(results)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
