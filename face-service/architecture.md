# HCP Face Analysis — Architecture

## Pipeline

A single photo runs through ten analyzers. Their outputs are merged
into one dictionary; later analyzers can overwrite keys from earlier
ones (only intentional in a couple of places — `_run_pipeline` in
[app.py](app.py) is the single source of truth).

```
Photo (RGB ndarray)
  │
  ├─► [1] InsightFaceAnalyzer  (insightface buffalo_l, ONNX)
  │       → face_bbox, face_confidence, face_embedding (512-d ArcFace),
  │         age_estimate (piecewise-calibrated). Gender comes from
  │         FairFace (step 3a) for a real softmax confidence.
  │
  ├─► Build face crop from face_bbox + padding. Downstream analyzers
  │   that benefit from a tighter input read the crop; MediaPipe gets
  │   the full image because it has its own detector.
  │
  ├─► [2] LandmarkAnalyzer  (MediaPipe Face Landmarker)
  │       478 landmarks + 52 blendshapes → all geometric features,
  │       smiling, mouth_open (via blendshapes.jawOpen), eyes_open,
  │       facial_asymmetry_score, smile_asymmetry, possible_dimples,
  │       possible_unibrow.
  │
  ├─► [3a] GenderAnalyzer  (dima806/fairface_gender ViT)
  │       → gender, gender_confidence, gender_distribution
  │       (cropped input). Replaces the InsightFace gender head so
  │       we get a real softmax confidence.
  │
  ├─► [3b] EthnicityAnalyzer  (cledoux42/Ethnicity_Test_v003 ViT)
  │       → ethnicity, ethnicity_confidence, ethnicity_distribution
  │       (cropped input).
  │
  ├─► [4] ParsingAnalyzer  (SegFormer-B5 human parsing)
  │       → _skin_mask, _hair_mask, hat_detected, hair_length,
  │         hair_present, wrinkle_level, skin_texture_score,
  │         skin_uniformity, freckles_or_moles
  │       (cropped input — cleaner masks).
  │
  ├─► [5] EmotionAnalyzer  (HSEmotion EfficientNet-B0)
  │       → primary/secondary emotion, emotion_scores, valence,
  │         arousal, mood (cropped input).
  │
  ├─► [6] ColorAnalyzer  (no ML — OpenCV LAB/HSV)
  │       Reads SegFormer masks + MediaPipe lip/iris landmarks.
  │       → skin_tone (Fitzpatrick + L*/a*/b* + hex), skin_undertone,
  │         eye_color, hair_color (name + hex), hair_texture
  │         (coarse, fallback), lip_color (shade + hex)
  │
  ├─► [7] ObstructionAnalyzer  (dima806/face_obstruction ViT-B/16)
  │       → wearing_glasses, wearing_sunglasses, wearing_mask,
  │         obstruction_scores (cropped input).
  │
  ├─► [8] HairTypeAnalyzer  (dima806/hair_type ViT-B/16)
  │       → hair_type (curly/dreadlocks/kinky/straight/wavy),
  │         hair_type_confidence (cropped input).
  │
  ├─► [9] BeautyAnalyzer  (ResNet-50 trained on SCUT-FBP5500)
  │       Optional. Loads local weights or HF Hub; if absent, output
  │       is None and AestheticAnalyzer falls back to rules.
  │       → beauty_score (1.0–5.0), beauty_score_norm (0–100),
  │         beauty_model_source.
  │
  └─► [10] AestheticAnalyzer  (no model)
          Reads the merged dict from steps 1–9 and produces the
          final chopped_score (0–100) plus chopped_breakdown
          showing each factor's signed contribution.
```

Internal/scratch keys use a leading underscore (`_skin_mask`,
`_hair_mask`, `_raw_landmarks`, `_insight_landmarks_2d`). `app.py`
strips them before returning JSON.

## Attribute → source map

| Section | Field(s) | Source |
|---|---|---|
| Demographics | face_bbox, face_confidence, face_embedding (512-d), age_estimate (piecewise-calibrated), age_range | InsightFace buffalo_l |
| Demographics | gender, gender_confidence, gender_distribution | FairFace ViT |
| Demographics | ethnicity, ethnicity_confidence, ethnicity_distribution | cledoux42 ViT |
| Emotion | primary/secondary emotion, emotion_scores, valence, arousal, mood | HSEmotion EffNet-B0 |
| Face Structure | face_shape (+ 4 ratios), jawline_type/angle, chin_type, cheekbone_prominence, cheek_fullness, forehead_width, facial_asymmetry_score | MediaPipe Face Landmarker |
| Hair | hair_length, hair_present | SegFormer-B5 |
| Hair | hair_type (+ confidence) | HairTypeViT (dima806) |
| Hair | hair_color, hair hex | ColorAnalyzer |
| Eyes | eye_shape, eye_depth, eye_spacing, eye_size, eyes_open | MediaPipe |
| Eyes | eye_color | ColorAnalyzer |
| Eyebrows | eyebrow_shape, eyebrow_arch_height, eyebrow_thickness, possible_unibrow | MediaPipe |
| Nose | nose_shape, nose_bridge, nose_tip_shape, nostril_width | MediaPipe |
| Lips & Mouth | lip_fullness, lip_balance, mouth_width, cupids_bow, smile_asymmetry, possible_dimples, smiling, mouth_open | MediaPipe (last two via blendshapes) |
| Lips & Mouth | lip_color (shade + hex) | ColorAnalyzer (mask from MediaPipe) |
| Skin | skin_tone (Fitzpatrick, L*/a*/b*, hex), skin_undertone | ColorAnalyzer |
| Skin | wrinkle_level, skin_texture_score, skin_uniformity | SegFormer mask + OpenCV stats (`freckles_or_moles` still computed server-side but no longer displayed — detector was too noisy) |
| Accessories | wearing_glasses, wearing_sunglasses, wearing_mask | ObstructionViT (dima806) |
| Accessories | wearing_hat | SegFormer (hat class coverage) |
| Aesthetics | beauty_score (1–5), beauty_score_norm (0–100) | BeautyAnalyzer (SCUT-FBP5500 ResNet-50) |
| Aesthetics | chopped_score (0–100), chopped_breakdown | AestheticAnalyzer (rule + learned blend) |

## Face matching

InsightFace's ArcFace head emits a 512-d L2-normalised recognition
embedding. We store it alongside each contact in
`people.face_embedding` (pgvector). On a new photo save, the client
queries Supabase for any contact with cosine similarity ≥ 0.55 to the
new embedding and prompts the user *"this looks like {name}, add to
that profile?"* before creating a new contact.

LFW accuracy is 99.83%; IJB-B at FAR=1e-4 is 96.21%. For grouping
photos in a personal collection (similar lighting, same camera) this
is excellent. Identical twins and close family members can match — the
0.55 threshold makes the prompt opt-in rather than auto-merge.

## Training the beauty regressor

Live source in [training/beauty/](../training/beauty/). The script
fine-tunes a timm ResNet-50 on SCUT-FBP5500. After training, drop the
resulting `beauty_regressor.pt` into `face-service/models/` (or push
to HF Hub and set `BEAUTY_HF_REPO_ID`). `BeautyAnalyzer` picks it up
automatically on the next process boot.

Until weights exist, `beauty_score` returns None and the AestheticAnalyzer
gracefully falls back to a pure rule-based chopped score.

## Deployment

The service builds as a Docker image targeting Hugging Face Spaces
free tier (2 GB RAM, shared CPU). MediaPipe `.task` and the
InsightFace buffalo_l bundle are pulled at build time; all other
Hugging Face models lazy-download on first inference and cache under
`/root/.cache/huggingface`.

The Node/Express server forwards `/analyze-face` requests to
`FACE_SERVICE_URL/analyze-base64`. The React Native client never
talks to this service directly.

## Adding a new analyzer

1. Drop a new module under `analyzers/` with a class exposing
   `__init__()` and `analyze(...) -> dict`.
2. Import + add a lazy-load block in `app.py`'s `get_analyzers()`.
3. Add a `results.update(...)` call inside `_run_pipeline` at the
   right pipeline position.
4. Surface the new keys in
   [client/src/screens/EditProfileScreen.js](../client/src/screens/EditProfileScreen.js)
   and add a legend row.

Order matters: later analyzers overwrite earlier keys on collision.
The aesthetic aggregator runs last so it can see everything.
