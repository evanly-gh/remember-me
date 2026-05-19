---
title: HCP Face Analysis Service
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# HCP Face Analysis Microservice

FastAPI service that runs seven specialized analyzers over a single photo
and returns a merged dictionary of ~100 facial attributes.

## Models

| # | Component | Model | Task | Size |
|---|-----------|-------|------|------|
| 1 | MediaPipe Face Landmarker | `face_landmarker.task` (Google) | 478 3D landmarks + 52 ARKit blendshapes — geometric features, smiling, mouth-open | ~4 MB |
| 2 | FairFace age | `dima806/fairface_age_image_detection` (ViT-B/16) | 9-bucket age → softmax-weighted continuous estimate | ~340 MB |
| 2 | FairFace gender | `dima806/fairface_gender_image_detection` (ViT-B/16) | Binary gender (~93.4% acc) | ~340 MB |
| 2 | Ethnicity | `cledoux42/Ethnicity_Test_v003` (ViT) | 5-class ethnicity (~79.6% acc) | ~340 MB |
| 3 | Human parsing | `matei-dorian/segformer-b5-finetuned-human-parsing` | 18-class pixel segmentation → masks + hair length + hat | ~340 MB |
| 4 | Emotion | HSEmotion `enet_b0_8_best_afew` (EfficientNet-B0) | 8-class emotion + valence/arousal | ~20 MB |
| 5 | Color analysis | (no model — OpenCV LAB/HSV) | Skin tone, hair color, eye color, lip color | 0 MB |
| 6 | Obstruction | `dima806/face_obstruction_image_detection` (ViT-B/16) | glasses / sunglasses / mask (~99% precision) | ~340 MB |
| 7 | Hair type | `dima806/hair_type_image_detection` (ViT-B/16) | curly/dreadlocks/kinky/straight/wavy (~93% acc) | ~340 MB |

All analyzers are lazy-loaded on first request. The MediaPipe weight
file is pre-downloaded at Docker build time; all Hugging Face models
are cached on first inference.

## API endpoints

- `GET /` — service info
- `GET /health` — liveness check
- `POST /analyze` — multipart file upload
- `POST /analyze-base64` — JSON `{ "image": "<base64>" }`

## Usage

```bash
curl -X POST https://YOUR-SPACE.hf.space/analyze-base64 \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64-encoded-image>"}'
```

See [architecture.md](./architecture.md) for the pipeline diagram and the
full per-attribute model attribution table.
