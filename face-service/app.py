"""
HCP Face Analysis Microservice
==============================

FastAPI service that runs seven specialized analyzers over a single photo
and merges their outputs into one ~100-field facial-attribute dictionary.

Pipeline (in execution order)
-----------------------------
1.  MediaPipe Face Landmarker   478 3D landmarks + 52 ARKit blendshapes.
                                Produces all geometric face/eye/nose/lip/
                                jaw features plus smiling and mouth-open.

2.  DemographicAnalyzer         Three ViT classifiers (FairFace age,
                                FairFace gender, Ethnicity_Test_v003).
                                Age is reported as a softmax-weighted
                                continuous estimate, not a bucket midpoint.

3.  ParsingAnalyzer             SegFormer-B5 human parsing. Emits face
                                and hair pixel masks plus hair length,
                                hat detection, and skin texture/wrinkle/
                                freckle/uniformity stats computed via
                                OpenCV over the face mask.

4.  EmotionAnalyzer             HSEmotion EfficientNet-B0 8-class output
                                plus derived valence, arousal, mood.

5.  ColorAnalyzer               Pixel-level LAB/HSV statistics. Reads
                                masks from step 3 and lip/iris landmarks
                                from step 1. No ML model.

6.  ObstructionAnalyzer         dima806 ViT-B/16. Glasses, sunglasses,
                                mask flags with ~99% precision/recall.

7.  HairTypeAnalyzer            dima806 ViT-B/16. Curly/dreadlocks/kinky/
                                straight/wavy at ~93% accuracy.

Endpoints
---------
GET  /                  service banner
GET  /health            liveness check
POST /analyze           multipart file upload
POST /analyze-base64    JSON {"image": "<base64>"}

Both POST endpoints run the same pipeline. All analyzers are lazily
instantiated on first request to keep cold-start latency manageable
on the Hugging Face Spaces free tier.
"""

import os
# hf_transfer gives much faster model downloads from the HF Hub on first
# inference. HF_HUB_DOWNLOAD_TIMEOUT defaults to 10s which is too short
# for the larger ViT checkpoints on a cold start.
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
from analyzers.demographic_analyzer import DemographicAnalyzer
from analyzers.parsing_analyzer import ParsingAnalyzer
from analyzers.emotion_analyzer import EmotionAnalyzer
from analyzers.color_analyzer import ColorAnalyzer
from analyzers.obstruction_analyzer import ObstructionAnalyzer
from analyzers.hair_type_analyzer import HairTypeAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="HCP Face Analysis Service", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Analyzers are initialized lazily on first request to reduce cold-start time
landmark_analyzer: Optional[LandmarkAnalyzer] = None
demographic_analyzer: Optional[DemographicAnalyzer] = None
parsing_analyzer: Optional[ParsingAnalyzer] = None
emotion_analyzer: Optional[EmotionAnalyzer] = None
color_analyzer: Optional[ColorAnalyzer] = None
obstruction_analyzer: Optional[ObstructionAnalyzer] = None
hair_type_analyzer: Optional[HairTypeAnalyzer] = None


def _to_json_safe(value):
    """Recursively coerce numpy scalars/arrays into JSON-serialisable types.

    Several analyzers return numpy floats/booleans (e.g. from `np.std`
    or boolean mask logic). FastAPI's default JSON encoder doesn't
    handle those, so we normalise everything here before returning.
    """
    # Numpy first — these checks would otherwise be caught by isinstance
    # for dict/list because numpy.generic types are duck-typed.
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.generic):
        return value.item()
    # Recurse into nested containers.
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
    global landmark_analyzer, demographic_analyzer
    global parsing_analyzer, emotion_analyzer, color_analyzer
    global obstruction_analyzer, hair_type_analyzer

    if landmark_analyzer is None:
        logger.info("Loading MediaPipe Face Landmarker...")
        landmark_analyzer = LandmarkAnalyzer()

    if demographic_analyzer is None:
        logger.info("Loading FairFace demographics model...")
        demographic_analyzer = DemographicAnalyzer()

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

    return (
        landmark_analyzer,
        demographic_analyzer,
        parsing_analyzer,
        emotion_analyzer,
        color_analyzer,
        obstruction_analyzer,
        hair_type_analyzer,
    )


@app.get("/")
async def root():
    """Service banner — confirms the server is reachable and which version."""
    return {
        "name": "HCP Face Analysis Service",
        "version": "2.0.0",
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

    Runs all seven analyzers and returns the merged attribute dict.
    See `analyze_face_base64` for the JSON-body variant the Express
    server calls.
    """
    try:
        # Decode the upload into an RGB numpy array. All analyzers
        # work in RGB; we don't actually need BGR but keeping it as a
        # local in case a future analyzer wants the OpenCV-native order.
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = np.array(image)

        (
            landmarks,
            demographics,
            parsing,
            emotions,
            colors,
            obstructions,
            hair_types,
        ) = get_analyzers()

        results = {}

        # Step 1: MediaPipe Landmarks → all geometric features + blendshapes.
        logger.info("Running landmark analysis...")
        landmark_results = landmarks.analyze(img_array)
        results.update(landmark_results)

        # Step 2: FairFace + Ethnicity ViT → demographics.
        logger.info("Running demographic analysis...")
        demo_results = demographics.analyze(img_array)
        results.update(demo_results)

        # Step 3: SegFormer-B5 human parsing → masks + hair length + skin stats.
        logger.info("Running face parsing...")
        parse_results = parsing.analyze(img_array)
        results.update(parse_results)

        # Step 4: HSEmotion → 8-class emotion + valence/arousal/mood.
        logger.info("Running emotion analysis...")
        emo_results = emotions.analyze(img_array)
        results.update(emo_results)

        # Step 5: Pixel color analysis. Uses the face/hair masks from step 3
        # and MediaPipe lip/iris landmarks from step 1.
        logger.info("Running color analysis...")
        color_results = colors.analyze(
            img_array,
            skin_mask=parse_results.get("_skin_mask"),
            hair_mask=parse_results.get("_hair_mask"),
            landmarks=landmark_results.get("_raw_landmarks"),
        )
        results.update(color_results)

        # Step 6: ObstructionViT → glasses / sunglasses / mask flags.
        logger.info("Running obstruction analysis...")
        results.update(obstructions.analyze(img_array))

        # Step 7: HairTypeViT → curly/dreadlocks/kinky/straight/wavy.
        logger.info("Running hair-type analysis...")
        results.update(hair_types.analyze(img_array))

        # Remove internal fields (prefixed with underscore)
        results = {k: v for k, v in results.items() if not k.startswith("_")}

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

        # Strip data URI prefix if present
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]

        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(image)

        (
            landmarks,
            demographics,
            parsing,
            emotions,
            colors,
            obstructions,
            hair_types,
        ) = get_analyzers()

        results = {}

        # Same seven-step pipeline as /analyze. Kept inline (rather
        # than factored out) so the per-step `logger.info` cadence and
        # ordering stay obvious when reading either endpoint top-down.
        landmark_results = landmarks.analyze(img_array)
        results.update(landmark_results)

        demo_results = demographics.analyze(img_array)
        results.update(demo_results)

        parse_results = parsing.analyze(img_array)
        results.update(parse_results)

        emo_results = emotions.analyze(img_array)
        results.update(emo_results)

        color_results = colors.analyze(
            img_array,
            skin_mask=parse_results.get("_skin_mask"),
            hair_mask=parse_results.get("_hair_mask"),
            landmarks=landmark_results.get("_raw_landmarks"),
        )
        results.update(color_results)

        results.update(obstructions.analyze(img_array))
        results.update(hair_types.analyze(img_array))

        # Drop internal/scratch fields (leading underscore) before
        # returning. Keeps masks and raw landmark lists out of the JSON.
        results = {k: v for k, v in results.items() if not k.startswith("_")}

        return {"success": True, "data": _to_json_safe(results)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
