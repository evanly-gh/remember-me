# HCP Face Analysis — Architecture

## Pipeline

A single photo is fed through seven analyzers. Their outputs are merged
into one dictionary; later analyzers overwrite any colliding keys from
earlier ones.

```
Photo (RGB ndarray)
  │
  ├─► [1] MediaPipe Face Landmarker
  │       478 landmarks + 52 blendshapes
  │       → all geometric features (face/eye/nose/eyebrow/lip/jaw shape),
  │         smiling (mouthSmile blendshapes), eyes_open, possible_dimples,
  │         possible_unibrow, facial_asymmetry_score, blendshapes dict
  │
  ├─► [2] FairFace + Ethnicity ViT (DemographicAnalyzer)
  │       → age_range, age_estimate (softmax-weighted continuous), age_confidence,
  │         gender + confidence, ethnicity + confidence, full distributions
  │
  ├─► [3] SegFormer-B5 human parsing (ParsingAnalyzer)
  │       → per-class pixel masks (face, hair, hat, …)
  │       → hair_length, hair_present, hat_detected,
  │         wrinkle_level, skin_texture_score, skin_uniformity, freckles_or_moles
  │       (uses OpenCV stats over the SegFormer face mask for the skin rows)
  │
  ├─► [4] HSEmotion EfficientNet-B0 (EmotionAnalyzer)
  │       → primary/secondary emotion, emotion_scores (8 classes),
  │         valence, arousal, mood
  │
  ├─► [5] ColorAnalyzer (no ML — OpenCV LAB/HSV)
  │       inputs: SegFormer skin/hair masks + MediaPipe landmarks
  │       → skin_tone (Fitzpatrick + L*/a*/b* + hex), skin_undertone,
  │         eye_color, hair_color (name + hex), hair_texture (pixel-Laplacian, coarse),
  │         lip_color (shade + hex)  ← lip mask built from MediaPipe outer-minus-inner lip
  │
  ├─► [6] ObstructionViT — dima806/face_obstruction_image_detection
  │       → wearing_glasses, wearing_sunglasses, wearing_mask,
  │         obstruction_top, obstruction_scores
  │
  └─► [7] HairTypeViT — dima806/hair_type_image_detection
          → hair_type (curly/dreadlocks/kinky/straight/wavy),
            hair_type_confidence, hair_type_scores
```

All masks and other internal fields use a leading underscore in the key
(e.g. `_skin_mask`). `app.py` strips those before returning JSON so the
client never sees them.

## Attribute → source map

The EditProfileScreen renders only fields backed by one of these
analyzers. Anything previously fed by the FaRL zero-shot classifier
has been removed because its outputs were too noisy to trust.

| Section | Field(s) | Source |
|---|---|---|
| Demographics | gender, age (continuous), age_range, ethnicity, distributions | FairFace + Ethnicity ViT |
| Emotion | primary/secondary emotion, scores, valence, arousal, mood | HSEmotion |
| Face Structure | face_shape (+ 4 ratios), jawline_type/angle, chin_type, cheekbone_prominence, cheek_fullness, forehead_width, facial_asymmetry_score | MediaPipe |
| Hair | hair_length, hair_present | SegFormer |
| Hair | hair_type (+ confidence) | HairTypeViT |
| Hair | hair_color, hair hex | ColorAnalyzer |
| Eyes | eye_shape, eye_depth, eye_spacing, eye_size, eyes_open | MediaPipe |
| Eyes | eye_color | ColorAnalyzer |
| Eyebrows | eyebrow_shape, eyebrow_arch_height, eyebrow_thickness, possible_unibrow | MediaPipe |
| Nose | nose_shape, nose_bridge, nose_tip_shape, nostril_width | MediaPipe |
| Lips & Mouth | lip_fullness, lip_balance, mouth_width, cupids_bow, smile_asymmetry, possible_dimples, smiling, mouth_open | MediaPipe (last two via blendshapes) |
| Lips & Mouth | lip_color (shade + hex) | ColorAnalyzer (mask from MediaPipe) |
| Skin | skin_tone (Fitzpatrick, L*/a*/b*, hex), skin_undertone | ColorAnalyzer |
| Skin | wrinkle_level, skin_texture_score, skin_uniformity, freckles_or_moles | SegFormer mask + OpenCV stats |
| Accessories | wearing_glasses, wearing_sunglasses, wearing_mask | ObstructionViT |
| Accessories | wearing_hat | SegFormer (hat class coverage) |

## Deployment

The service is built as a Docker image targeting Hugging Face Spaces
free tier (2GB RAM, shared CPU). The MediaPipe `.task` is pulled at
build time; all Hugging Face models lazy-download on first inference
and cache under `/root/.cache/huggingface` inside the container.

The Node/Express server forwards `/analyze-face` requests to
`FACE_SERVICE_URL/analyze-base64`. The React Native client never talks
to this service directly.

## Adding a new analyzer

1. Drop a new module under `analyzers/` exposing a class with
   `__init__()` and `analyze(img_rgb) -> dict`.
2. Import it in `app.py`, add a global slot and a lazy-load block in
   `get_analyzers()`, and append a `results.update(...)` call to both
   `/analyze` and `/analyze-base64`.
3. Surface the new keys in `client/src/screens/EditProfileScreen.js`
   and add a legend row in the "Analysis Method Details" section.

Order matters: later analyzers overwrite earlier keys on collision.
The specialized ViT classifiers run last so they win over any coarser
signal.
