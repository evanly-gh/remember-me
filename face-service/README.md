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

A FastAPI-based facial analysis service that combines 6 specialized ML models
to extract 100+ facial attributes from a single photograph.

## Models Used

| Model | Task | Size |
|-------|------|------|
| MediaPipe Face Landmarker | 478 3D landmarks + blendshapes | ~4 MB |
| FairFace ResNet-34 | Age, gender, ethnicity | ~90 MB |
| CelebA ResNet-18 | 40 binary attributes | ~44 MB |
| BiSeNet | Face region segmentation | ~50 MB |
| HSEmotion EfficientNet-B0 | 8-class emotion | ~20 MB |
| Custom color analysis | Skin/eye/hair color | 0 MB |

## API Endpoints

- `GET /health` — Health check
- `POST /analyze` — Multipart file upload
- `POST /analyze-base64` — JSON body with base64 image

## Usage

```bash
curl -X POST https://YOUR-SPACE.hf.space/analyze-base64 \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64-encoded-image>"}'
```
