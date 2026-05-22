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

FastAPI service that runs ten specialised analyzers over a single
photo and returns a merged dictionary of facial attributes plus a
face-recognition embedding and an aesthetic "chopped score."

## Models

| # | Component | Model | Task | Size |
|---|-----------|-------|------|------|
| 1 | InsightFace | `buffalo_l` (SCRFD + ArcFace ResNet50, ONNX) | Detection + 512-d recognition embedding + 106 landmarks (99.83% LFW) + piecewise-calibrated age | ~280 MB |
| 2 | MediaPipe Face Landmarker | `face_landmarker.task` (Google) | 478 3D landmarks + 52 ARKit blendshapes — geometric features, smiling, mouth-open | ~4 MB |
| 3a | Gender | `dima806/fairface_gender_image_detection` (ViT) | Binary gender + softmax confidence (~93.4% acc) | ~340 MB |
| 3b | Ethnicity | `cledoux42/Ethnicity_Test_v003` (ViT) | 5-class ethnicity (~79.6% acc) | ~340 MB |
| 4 | Human parsing | `matei-dorian/segformer-b5-finetuned-human-parsing` | 18-class pixel segmentation → masks + hair length + hat | ~340 MB |
| 5 | Emotion | HSEmotion `enet_b0_8_best_afew` (EfficientNet-B0) | 8-class emotion + valence/arousal | ~20 MB |
| 6 | Color analysis | (no model — OpenCV LAB/HSV) | Skin tone, hair color, eye color, lip color | 0 MB |
| 7 | Obstruction | `dima806/face_obstruction_image_detection` (ViT-B/16) | glasses / sunglasses / mask (~99% precision) | ~340 MB |
| 8 | Hair type | `dima806/hair_type_image_detection` (ViT-B/16) | curly/dreadlocks/kinky/straight/wavy (~93% acc) | ~340 MB |
| 9 | Beauty regression | timm ResNet-50, fine-tuned on SCUT-FBP5500 | 1.0–5.0 beauty score (~Pearson r ≥ 0.85 expected) | ~100 MB |
| 10 | Aesthetic aggregator | (no model — Python rules) | Combines learned beauty + heuristic factors → chopped_score (0–100) | 0 MB |

InsightFace `buffalo_l` is pre-downloaded at Docker build time. The
MediaPipe weight file is pre-downloaded too. All Hugging Face models
lazy-load on first inference and cache locally for the lifetime of
the process. The SCUT-FBP5500-trained beauty regressor must be
produced via [`training/beauty/`](../training/beauty/) — until you
drop weights in `models/beauty_regressor.pt` (or set
`BEAUTY_HF_REPO_ID`), `beauty_score` returns null and the chopped
score uses heuristics only.

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

See [architecture.md](./architecture.md) for the pipeline diagram and
the full per-attribute → source map.
