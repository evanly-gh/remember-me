"""
Face Analysis Microservice
Combines multiple pretrained models for comprehensive facial attribute detection.

Models used:
1. MediaPipe Face Landmarker — 478 3D landmarks + 52 blendshapes → geometric features
2. FairFace — age, gender, race classification
3. CelebA Attribute Classifier — 40 binary facial attributes
4. BiSeNet Face Parsing — 19-class pixel-level segmentation
5. HSEmotion — 8-class emotion recognition
6. Color Analyzer — pixel-level skin tone, eye color, hair color (no AI)
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
attribute_analyzer: Optional[AttributeAnalyzer] = None
parsing_analyzer: Optional[ParsingAnalyzer] = None
emotion_analyzer: Optional[EmotionAnalyzer] = None
color_analyzer: Optional[ColorAnalyzer] = None


def get_analyzers():
    """Lazy-load all analyzer models on first use."""
    global landmark_analyzer, demographic_analyzer, attribute_analyzer
    global parsing_analyzer, emotion_analyzer, color_analyzer

    if landmark_analyzer is None:
        logger.info("Loading MediaPipe Face Landmarker...")
        landmark_analyzer = LandmarkAnalyzer()

    if demographic_analyzer is None:
        logger.info("Loading FairFace demographics model...")
        demographic_analyzer = DemographicAnalyzer()

    if attribute_analyzer is None:
        logger.info("Loading CelebA attribute classifier...")
        attribute_analyzer = AttributeAnalyzer()

    if parsing_analyzer is None:
        logger.info("Loading BiSeNet face parser...")
        parsing_analyzer = ParsingAnalyzer()

    if emotion_analyzer is None:
        logger.info("Loading HSEmotion model...")
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
    """Health check endpoint — use to keep the service warm."""
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_face(file: UploadFile = File(...)):
    """
    Comprehensive face analysis endpoint.

    Accepts an image file upload and returns ~100+ facial attributes
    by running 6 models/analyzers in sequence.
    """
    try:
        # Read and decode the uploaded image
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

        # Step 1: MediaPipe Landmarks → geometric features (~40 attributes)
        logger.info("Running landmark analysis...")
        landmark_results = landmarks.analyze(img_array)
        results.update(landmark_results)

        # Step 2: FairFace → age, gender, race
        logger.info("Running demographic analysis...")
        demo_results = demographics.analyze(img_array)
        results.update(demo_results)

        # Step 3: CelebA → 40 binary facial attributes
        logger.info("Running attribute analysis...")
        attr_results = attributes.analyze(img_array)
        results.update(attr_results)

        # Step 4: BiSeNet → pixel segmentation → hair length, wrinkles, spots
        logger.info("Running face parsing...")
        parse_results = parsing.analyze(img_bgr)
        results.update(parse_results)

        # Step 5: HSEmotion → emotion classification
        logger.info("Running emotion analysis...")
        emo_results = emotions.analyze(img_array)
        results.update(emo_results)

        # Step 6: Color analysis using masks from Step 4 + landmarks from Step 1
        logger.info("Running color analysis...")
        landmark_data = landmark_results.get("_raw_landmarks")
        print(type(landmark_data))
        color_results = colors.analyze(
            img_array,
            skin_mask=parse_results.get("_skin_mask"),
            hair_mask=parse_results.get("_hair_mask"),
            landmarks=landmark_data,
        )
        results.update(color_results)

        # Remove internal fields (prefixed with underscore)
        results = {k: v for k, v in results.items() if not k.startswith("_")}

        return {"success": True, "data": results}

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-base64")
async def analyze_face_base64(body: dict):
    """
    Alternative endpoint that accepts base64-encoded image data.
    This matches the format the Express server sends.
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

        landmark_results = landmarks.analyze(img_array)
        results.update(landmark_results)

        demo_results = demographics.analyze(img_array)
        results.update(demo_results)

        attr_results = attributes.analyze(img_array)
        results.update(attr_results)

        parse_results = parsing.analyze(img_bgr)
        results.update(parse_results)

        emo_results = emotions.analyze(img_array)
        results.update(emo_results)

        landmark_data = landmark_results.get("_raw_landmarks")
        print(type(landmark_data))
        color_results = colors.analyze(
            img_array,
            skin_mask=parse_results.get("_skin_mask"),
            hair_mask=parse_results.get("_hair_mask"),
            landmarks=landmark_data,
        )
        results.update(color_results)

        results = {k: v for k, v in results.items() if not k.startswith("_")}

        return {"success": True, "data": results}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
