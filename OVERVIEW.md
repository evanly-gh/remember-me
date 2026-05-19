# HCP — Project Overview

This document is a top-down tour of what this codebase is, how it's wired
together, and which models do what. It assumes you can read code but
doesn't assume you've used React Native, FastAPI, Supabase, or modern
vision-language models before — concepts are explained when they
appear. Read top-to-bottom for the full picture, or jump to the section
you need from the table of contents.

---

## Table of Contents

1. [Project at a glance](#1-project-at-a-glance)
2. [System architecture](#2-system-architecture)
3. [The three runtimes](#3-the-three-runtimes)
4. [Face analysis pipeline — what each model does](#4-face-analysis-pipeline)
5. [Data model](#5-data-model)
6. [Client UX walkthrough](#6-client-ux-walkthrough)
7. [Design decisions and trade-offs](#7-design-decisions-and-trade-offs)
8. [Development setup](#8-development-setup)
9. [Deployment](#9-deployment)
10. [Performance characteristics](#10-performance-characteristics)
11. [Known limitations and roadmap](#11-known-limitations-and-roadmap)
12. [Glossary](#12-glossary)

---

## 1. Project at a glance

**HCP** ("remember-me") is a mobile app for keeping track of people you
meet. You take their photo, jot down where and when you met, and the
app extracts ~100 facial attributes from the photo (age, gender,
emotion, glasses, hair colour, …). Later you can find someone by name,
by attribute, or by free-text search ("the woman with curly red hair I
met at the dog park").

The app has three pieces:

```
client/         React Native (Expo) — the iOS/Android app
server/         Node + Express     — thin proxy in front of the ML
face-service/   Python + FastAPI   — runs the ML models
```

Plus a managed dependency: **Supabase** hosts the user accounts, the
contacts database, and the photo storage. Nothing in this repo runs a
database directly.

---

## 2. System architecture

```
┌────────────────────────┐          ┌────────────────────────┐
│   React Native client  │          │      Supabase          │
│   (Expo, iOS/Android)  │ ───────► │ • Auth (email/pass)    │
│                        │          │ • people table         │
│   photos, search, UI   │ ◄─────── │ • photos bucket        │
└────────────┬───────────┘          └────────────────────────┘
             │ POST /analyze-face   ▲
             │   { image: base64 }  │ embeddings via HF API
             ▼                      │ (sentence-transformers
┌────────────────────────┐          │  all-MiniLM-L6-v2)
│   Express server       │          │
│   (Node, port 3000)    │          │
│                        │          │
│   forwards image to    │          │
│   face-service         │          │
└────────────┬───────────┘          │
             │ POST /analyze-base64
             ▼
┌────────────────────────┐
│   Python face-service  │
│   (FastAPI on HF Spaces)│
│                        │
│   7 ML analyzers       │
│   produce ~100 fields  │
└────────────────────────┘
```

**Why three tiers?** The client can't run the heavier models on a
phone (battery + size limits). Supabase Edge Functions are JS/TS only
and capped on RAM, so they can't host Python ML either. Splitting the
work into a Python sidecar lets us pull any HuggingFace model we want
without fighting the runtime constraints. The Node server in the
middle is mostly a forwarding shim today; it gives us a place to add
auth checks or rate-limiting without redeploying the ML service.

---

## 3. The three runtimes

### 3.1 React Native client

Lives in [client/](client/). Built with **Expo** (a managed React
Native distribution that handles native builds and updates).

**Key screens** (in [client/src/screens/](client/src/screens/)):

| Screen | Purpose |
|---|---|
| `AuthScreen` | Email/password sign-in via Supabase Auth |
| `RecordScreen` | Camera, location capture, save a new contact |
| `LookupScreen` | List + semantic search across saved contacts |
| `EditProfileScreen` | Edit notes; view the full facial-analysis panel |
| `SettingsScreen` | Theme toggle, log out, debug toggles |

**State management**: light. React's built-in `useState`/`useEffect`,
two context providers in [client/src/context/](client/src/context/)
(`AuthContext` for the logged-in user, `ThemeContext` for dark mode),
and `AsyncStorage` for sticky preferences. No Redux, no Zustand — the
data lives in Supabase and gets re-fetched on focus.

**Theming**: every screen reads colours from `useTheme()`. Switching
to dark mode is a single toggle and propagates everywhere.

**Selectable text**: `Text.defaultProps.selectable = true` is set
once in [client/App.js](client/App.js) so every label across the app
is long-press copyable without per-component effort.

### 3.2 Express server

A single file: [server/src/app.js](server/src/app.js). One real
endpoint:

```
POST /analyze-face   { image: <base64-data-uri> }
   → forwards to ${FACE_SERVICE_URL}/analyze-base64
   → returns whatever the face-service returns
```

It exists for three reasons:

1. **CORS** — the face-service can lock down its allow-list to just
   this server's origin.
2. **Future auth** — we can validate Supabase JWTs here before
   spending GPU cycles in the face-service.
3. **Decoupling** — the client URL stays stable even if the
   face-service moves between HF Spaces and a different host.

### 3.3 Python face-service

Lives in [face-service/](face-service/). FastAPI app in
[face-service/app.py](face-service/app.py) plus seven analyzer
modules in [face-service/analyzers/](face-service/analyzers/).

Each analyzer is a small class with two methods: `__init__()` loads
the model, `analyze(img_rgb) → dict` runs inference and returns a
flat dict of attribute fields. The orchestrator in `app.py` calls
them in order and merges their outputs into one big result dict.

**Lazy loading**: models aren't loaded at import time — the first
incoming request pays the load cost, subsequent requests are warm.
This is important on HuggingFace Spaces free tier where the container
spins down after ~30 min of inactivity.

---

## 4. Face analysis pipeline

When `/analyze-base64` runs, seven analyzers fire in this order. Each
later analyzer can overwrite keys produced by earlier ones — the
specialized ViT models intentionally run last so their outputs win
over coarser signals.

```
photo
  │
  ▼
┌────────────────────────────┐   geometric features,
│ 1. MediaPipe Landmarker    │   blendshapes → smiling, mouth open,
└────────────────────────────┘   facial asymmetry, …
  │
  ▼
┌────────────────────────────┐   age (continuous), gender,
│ 2. DemographicAnalyzer     │   ethnicity + full distributions
│    (3 ViT classifiers)     │
└────────────────────────────┘
  │
  ▼
┌────────────────────────────┐   face mask, hair mask, hat detected,
│ 3. ParsingAnalyzer         │   hair length, wrinkles, freckles,
│    (SegFormer-B5)          │   skin uniformity
└────────────────────────────┘
  │
  ▼
┌────────────────────────────┐   8-class emotion + valence + arousal
│ 4. EmotionAnalyzer         │
│    (HSEmotion / EffNet-B0) │
└────────────────────────────┘
  │
  ▼
┌────────────────────────────┐   skin tone (Fitzpatrick + hex),
│ 5. ColorAnalyzer           │   undertone, hair colour, eye colour,
│    (OpenCV — no ML model)  │   lip colour. Uses masks + landmarks.
└────────────────────────────┘
  │
  ▼
┌────────────────────────────┐   glasses / sunglasses / mask
│ 6. ObstructionAnalyzer     │
│    (dima806 ViT-B/16)      │
└────────────────────────────┘
  │
  ▼
┌────────────────────────────┐   hair texture (curly / dreadlocks /
│ 7. HairTypeAnalyzer        │   kinky / straight / wavy)
│    (dima806 ViT-B/16)      │
└────────────────────────────┘
  │
  ▼
merged result dict → strip underscore-prefixed internal fields → JSON
```

Now each model in detail.

---

### 4.1 MediaPipe Face Landmarker

[face-service/analyzers/landmark_analyzer.py](face-service/analyzers/landmark_analyzer.py)

**What it is.** A pre-trained model from Google that takes a face
photo and returns 478 dots placed on the face (the eyebrows, eye
corners, nose ridge, lip outline, jaw, etc.). Each dot has an `(x, y, z)`
coordinate, so we know not just *where* a feature is on the image but
roughly how far it sits in front of or behind other features. It also
returns 52 **blendshapes** — these are facial-muscle activation
scores between 0 and 1 (the same set used by Apple's ARKit for
emoji-style face tracking). Examples: `mouthSmileLeft`, `jawOpen`,
`eyeBlinkRight`.

**Specs:**
- Architecture: a small CNN (the "face landmark" model) chained
  behind a face detector (BlazeFace).
- Size: ~4 MB, float16-quantised
- Source: [Google MediaPipe Tasks](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)
- License: Apache 2.0
- Auto-downloaded from `storage.googleapis.com` on first run; cached
  in `models/face_landmarker.task`.

**What we extract from it.** MediaPipe by itself doesn't tell you
"this face is heart-shaped" or "this eyebrow is arched" — it just
gives you 478 dots. The analyzer turns those dots into categorical
attributes by computing distances, ratios, and angles between
specific landmark indices. For example:

- Face shape is decided by four ratios (width/height,
  jaw-to-face-width, forehead-to-jaw, cheekbone-to-jaw) run through a
  decision cascade.
- Eye spacing is the pupil-to-pupil distance divided by face width.
- "Smiling" is true when the average of `mouthSmileLeft` and
  `mouthSmileRight` blendshape scores exceeds 0.4.

**Accuracy.** Landmark models aren't usually scored on accuracy %.
The standard metric is **Normalized Mean Error (NME)** — the average
distance between predicted and true landmark locations, normalised by
the eye-to-eye distance. MediaPipe's NME on standard benchmarks is
~5%, which is roughly state-of-the-art for mobile-size models. We
care less about that and more about the fact that it survives 30°
head rotation gracefully, which is what matters for casual phone
photos.

**Pros:**
- Tiny (4 MB) and runs in milliseconds even on CPU.
- 3D coordinates let us measure things like cheekbone prominence
  (which is mostly a depth signal).
- The blendshape head is high-quality and gives us "smiling" and
  "mouth open" for free — both are way more reliable than asking a
  zero-shot classifier the same question.

**Cons:**
- Every derived attribute (face shape, eye shape, etc.) relies on
  hand-tuned thresholds. If lighting or head angle changes a ratio
  by 5%, the categorical bucket can flip.
- Only handles one face at a time — we set `num_faces=1`.

---

### 4.2 FairFace age & gender — `dima806/fairface_age_image_detection`, `dima806/fairface_gender_image_detection`

[face-service/analyzers/demographic_analyzer.py](face-service/analyzers/demographic_analyzer.py)

**What it is.** Two Vision Transformers (ViTs) fine-tuned on the
**FairFace** dataset — a face dataset specifically curated to be
balanced across race, gender, and age so the resulting models don't
inherit demographic bias from internet-scraped training data.

**ViT in one paragraph.** A Vision Transformer chops an image into
fixed 16×16-pixel patches, flattens each patch into a vector, and
feeds the resulting sequence into a transformer (the same
architecture as a language model). The transformer learns
relationships between patches via self-attention. For image
classification, a special `[CLS]` token's final state is fed into a
small linear layer that outputs one score per class.

**Specs:**
- Architecture: ViT-B/16 (12 transformer layers, ~86M params)
- Base: `google/vit-base-patch16-224-in21k` (pre-trained on
  ImageNet-21k), then fine-tuned by user `dima806` for FairFace.
- Size: ~340 MB each (yes, hefty)
- License: Apache 2.0
- Source: [dima806/fairface_age_image_detection](https://huggingface.co/dima806/fairface_age_image_detection),
  [dima806/fairface_gender_image_detection](https://huggingface.co/dima806/fairface_gender_image_detection)

**What they output.**
- **Age**: a probability distribution over 9 buckets — `0-2`, `3-9`,
  `10-19`, `20-29`, `30-39`, `40-49`, `50-59`, `60-69`, `70+`. The
  argmax is the predicted age range.
- **Gender**: binary softmax over Male / Female.

**How scoring works.** Each model outputs raw scores called
**logits**, one per class. We pass them through **softmax** (which
turns them into probabilities that sum to 1) and pick the top one as
the prediction.

**The age trick.** A 9-bucket classifier always snaps to one of nine
fixed midpoints (e.g. 24.5 for the 20-29 bucket). That makes every
person in their twenties show up as exactly 24.5, which is useless.
We work around this by computing the **expected value** across the
full softmax — sum over `(bucket_midpoint × probability)` for all
nine buckets. If the model is very confident the person is 20-29 the
estimate stays near 24.5; if some probability mass leaks into 30-39,
the estimate slides up to 27.something. The number isn't true
per-year accuracy, but it varies smoothly across people instead of
snapping.

**Accuracy:**
- Age: ~59% top-1 on FairFace test set (i.e. the model picks the
  exact correct bucket 59% of the time). Lower than you'd hope, but
  with the softmax-weighted estimate the practical error is more like
  ±5 years.
- Gender: ~93.4% on FairFace.

**Pros:**
- High accuracy, especially on gender.
- Balanced training data means it doesn't fall off a cliff on
  darker-skinned or non-Western faces the way some older models do.
- Both have published model cards with per-class precision/recall.

**Cons:**
- 340 MB per model is large; combined with the ethnicity model
  below, demographics alone is ~1 GB on disk.
- Age is bucketed not regression — true per-year would need a
  different architecture (see roadmap).

---

### 4.3 Ethnicity — `cledoux42/Ethnicity_Test_v003`

Same file as 4.2.

**What it is.** Another ViT-style classifier trained to predict one
of 5 ethnicity categories.

**Specs:**
- Architecture: ViT
- Classes: `african`, `asian`, `caucasian`, `hispanic`, `indian`
- Accuracy: 79.6% overall, macro-F1 0.797
- License: Apache 2.0
- Source: [cledoux42/Ethnicity_Test_v003](https://huggingface.co/cledoux42/Ethnicity_Test_v003)

**Why we have it.** The older `NikhilJaddu/fairface-race-vit`
checkpoint that this project used originally had no published
performance metrics. Replacing it with a documented model was a clean
win.

**How we use its output.** This model emits only 5 classes, but we
keep the legacy 7-bucket FairFace label space internally (`White`,
`Black`, `Latino_Hispanic`, `East Asian`, `Southeast Asian`,
`Indian`, `Middle Eastern`) so downstream code doesn't have to
branch. The mapping happens in
`DemographicAnalyzer._normalize_race_label` — `african → Black`,
`caucasian → White`, etc. Unseen buckets stay at 0.0 probability.

**Pros:** simple, fast, documented accuracy.
**Cons:** only 5 classes; `caucasian` lumps Middle Eastern faces
into "White"; macro-F1 0.797 means at least one class is meaningfully
worse than that.

---

### 4.4 SegFormer-B5 human parsing — `matei-dorian/segformer-b5-finetuned-human-parsing`

[face-service/analyzers/parsing_analyzer.py](face-service/analyzers/parsing_analyzer.py)

**What it is.** A **semantic segmentation** model. Where a classifier
asks "what is this image?", a segmenter asks "what class does each
*pixel* belong to?" The output is the same shape as the input but
each pixel is a label like "hair" or "face" or "background".

**SegFormer in one paragraph.** Most segmenters use CNNs. SegFormer
swaps the encoder for a hierarchical transformer (like ViT but with
multi-scale features, similar to how a CNN naturally downsamples).
The decoder is a tiny MLP — just a few linear layers that fuse the
encoder's multi-scale features into a per-pixel class map. The whole
thing is small and fast relative to other segmenters and held SOTA on
common benchmarks when it dropped (2021).

**Specs:**
- Architecture: SegFormer-B5 (the largest variant), `nvidia/mit-b5` backbone
- Size: ~340 MB
- Classes (18): background, hat, hair, sunglasses, upper_clothes,
  skirt, pants, dress, belt, left_shoe, right_shoe, face, left_leg,
  right_leg, left_arm, right_arm, bag, scarf
- License: Apache 2.0 (architecture); model card doesn't specify
  weight licence
- Source: [matei-dorian/segformer-b5-finetuned-human-parsing](https://huggingface.co/matei-dorian/segformer-b5-finetuned-human-parsing)

**Reported accuracy** (per model card):
- Mean IoU: **0.626**
- Mean accuracy: 0.755
- Overall accuracy: 0.826
- Face class: accuracy 0.909, IoU 0.829
- Hair class: accuracy 0.897, IoU 0.817

**IoU = Intersection over Union.** If the model predicts the
"hair" region and you compare it to the true hair region, IoU is
`overlap / union`. 1.0 is perfect, 0.5 is a typical "decent" score
for segmentation, 0.8+ is excellent.

**What we extract.** The raw masks (face & hair) get passed to
ColorAnalyzer for pixel-level colour work. We also compute:
- **Hair length**: ratio of hair pixels to (face + hair) pixels. <5%
  → "bald/very short", <20% → "short", <40% → "medium", else "long".
- **Hat detected**: true when ≥1% of pixels are class "hat".
- **Wrinkle level, skin texture, skin uniformity, freckles**: OpenCV
  statistics computed *over* the SegFormer face mask. The Laplacian
  filter responds to local intensity curvature — high std-dev of the
  Laplacian over a region means lots of fine detail, which correlates
  with wrinkles. Working in LAB colour space (not RGB) makes the
  freckle detection tone-independent: a freckle is "darker than
  surrounding skin" regardless of base skin colour.

**Pros:**
- Best-in-class accuracy on face and hair classes (the two we care
  about most).
- Provides masks that other parts of the pipeline depend on — without
  good masks ColorAnalyzer produces garbage.

**Cons:**
- 340 MB and the slowest single inference in the pipeline (~1–2 s on
  CPU, the HF Spaces free tier).
- No lip class — we work around this by deriving lip masks from
  MediaPipe landmarks in ColorAnalyzer.
- No earring/necklace/necktie classes — those attributes were
  removed from the UI rather than guessed at.

---

### 4.5 HSEmotion EfficientNet-B0 — `enet_b0_8_best_afew`

[face-service/analyzers/emotion_analyzer.py](face-service/analyzers/emotion_analyzer.py)

**What it is.** A CNN trained on the AffectNet / AFEW emotion
datasets to classify one of 8 basic emotions from a face image. Comes
from Savchenko et al.'s [HSEmotion library](https://github.com/HSE-asavchenko/face-emotion-recognition).
We install it as a pip package and call `predict_emotions()`.

**EfficientNet-B0 in one paragraph.** A CNN family that systematically
scales depth, width, and resolution together to get the best
accuracy-per-parameter. B0 is the smallest variant (~5M params, 20 MB
on disk). Released by Google in 2019 and still a great
"low-cost-baseline" choice for image classification.

**Specs:**
- Architecture: EfficientNet-B0 (5M params)
- Checkpoint: `enet_b0_8_best_afew`
- Size: ~20 MB
- Source: [HSE-asavchenko/face-emotion-recognition](https://github.com/HSE-asavchenko/face-emotion-recognition)
- License: Apache 2.0
- Classes (8): anger, contempt, disgust, fear, happiness, neutral,
  sadness, surprise

**Accuracy:** ~66.5% on AffectNet-8. That's near SOTA for 8-class
facial emotion — emotion classification is genuinely hard and even
human inter-rater agreement maxes out around 70-80%.

**How scoring works.** The CNN outputs 8 logits, we softmax them, the
argmax is the primary emotion. The second-highest is reported as
"secondary emotion" — useful when the model is genuinely torn
(e.g. neutral vs. sadness on a flat face).

**Valence and arousal.** Two extra scalars we compute from the same
distribution:
- **Valence** = `Σ (probability × per-emotion valence weight)` in
  range [-1, +1]. Happiness contributes +0.9, sadness contributes
  -0.7, neutral 0.0. So a confidently-happy face gets valence ~0.9.
- **Arousal** = similar weighted sum but with arousal weights in [0,
  1]. Fear and anger have high arousal (0.9, 0.8), neutral and
  sadness low (0.1, 0.3).
- The weights are hand-set in `VALENCE_MAP` and `AROUSAL_MAP` in
  [emotion_analyzer.py](face-service/analyzers/emotion_analyzer.py)
  — they aren't learned, they're our way of projecting an 8-class
  distribution down to two intuitive axes.

`mood` is then bucketed off `valence`: > 0.2 → positive, < -0.2 →
negative, else neutral.

**Pros:**
- Tiny (20 MB), one of the fastest models in the pipeline.
- Good public accuracy on a genuinely hard task.
- Valence/arousal projection gives us a useful scalar mood signal
  for free.

**Cons:**
- 66% is honest but means roughly 1 in 3 predictions is wrong.
- 8 emotions can't capture mixed feelings — "wistful" doesn't exist
  in the label space.
- Loading the checkpoint requires `torch.load(weights_only=False)`
  because the file is pickled as a full `timm` model object (not a
  clean state dict). We scope a context manager around just the
  HSEmotion init so the rest of the process stays on PyTorch 2.6's
  safer default.

---

### 4.6 ColorAnalyzer — no ML model

[face-service/analyzers/color_analyzer.py](face-service/analyzers/color_analyzer.py)

**What it is.** A pure OpenCV module. No neural network, no weights
to download. It takes the masks from SegFormer + landmarks from
MediaPipe and computes colours and tones by averaging the pixels
inside specific regions.

**Why it's a "model" in the loose sense.** Even though there's no AI,
every output here is a categorical bucket produced by a threshold
cascade, which is the same shape of output the ML models produce.
Conceptually it's the cheapest classifier in the pipeline.

**Outputs:**

| Field | How it's computed |
|---|---|
| `skin_tone.fitzpatrick` | Mean L* (perceptual lightness) over the face mask, looked up in Fitzpatrick bands |
| `skin_tone.hex_color` | Average RGB over face mask → hex string |
| `skin_undertone` | Mean LAB b\* (yellow-blue axis): >+12 warm, <−8 cool, else neutral |
| `eye_color` | HSV histogram bucket over the iris region (landmarks 468-473) |
| `hair_color.name` | LAB-trimmed median over hair mask → HSV buckets |
| `hair_color.hex` | Same median converted to hex |
| `hair_texture` | std(Laplacian) over the *eroded* hair mask. Coarse fallback only — HairTypeViT is the authoritative source |
| `lip_color.shade` | HSV of average pixel inside MediaPipe lip polygon (outer minus inner) |

**Two non-obvious decisions worth knowing:**

1. **LAB beats RGB for skin tone.** RGB values shift wildly with
   camera white balance. LAB's L* is a perceptual lightness that
   stays roughly stable, so the same skin under tungsten vs sunlight
   bins into the same Fitzpatrick type. We also do skin-undertone
   classification using LAB's b* axis, which separates yellow leaning
   (warm) from blue leaning (cool).

2. **Trimmed median beats k-means for hair colour.** Early versions
   used k=2 k-means and reported the larger cluster, but that
   cluster can flip between "shadow side of black hair" and
   "highlight side of black hair" depending on lighting, giving
   different colours for the same person across photos. The current
   approach drops the top/bottom 10% of L* (specular highlights and
   deep shadows) and takes the median of what's left — robust to
   outliers, deterministic, no flicker.

**Pros:**
- Zero ML cost. Runs in microseconds on CPU.
- Deterministic — the same image always produces the same colours
  byte-for-byte.
- No model maintenance, no licensing concerns, no HuggingFace
  downloads.

**Cons:**
- Quality is bounded by mask quality. If SegFormer mis-segments hair
  and includes skin pixels, the hair colour skews.
- Hand-tuned thresholds. The HSV bands that decide "blue eyes" vs
  "green eyes" were dialled in by eye and might need retuning for
  different lighting conditions.

---

### 4.7 ObstructionAnalyzer — `dima806/face_obstruction_image_detection`

[face-service/analyzers/obstruction_analyzer.py](face-service/analyzers/obstruction_analyzer.py)

**What it is.** A ViT-B/16 fine-tuned to classify what (if anything)
is obstructing a face. Six classes: `sunglasses`, `glasses`, `mask`,
`hand`, `other`, `none`. Trained by the same author as the FairFace
models we use in 4.2.

**Specs:**
- Architecture: ViT-B/16
- Size: ~340 MB
- License: Apache 2.0
- Source: [dima806/face_obstruction_image_detection](https://huggingface.co/dima806/face_obstruction_image_detection)

**Accuracy** (from the model card):

| Class | Precision | Recall |
|---|---|---|
| Sunglasses | 99.7% | 99.85% |
| Glasses | 99.0% | 99.7% |
| Mask | 99.7% | 99.85% |
| Hand | 75.0% | 70.9% |
| Other | 72.0% | 76.1% |
| None | 99.8% | 98.6% |

The first three are essentially solved. Hand/other are weaker — we
don't surface them as booleans in the UI for that reason.

**How we use it.** The pipeline turns the softmax into three
booleans:
- `wearing_glasses = max(P(glasses), P(sunglasses)) > 0.5`
- `wearing_sunglasses = P(sunglasses) > 0.5`
- `wearing_mask = P(mask) > 0.5`

The full distribution is also exposed as `obstruction_scores` if a
caller wants to inspect raw probabilities.

**Pros:**
- Surgical: optimised for exactly the binary flags we want, instead
  of asking a general-purpose model to "guess if there are glasses".
- Better than the zero-shot CLIP/FaRL approach this replaced — those
  models had ~70-80% accuracy on glasses; this is ~99%.

**Cons:**
- Big (340 MB). Each ViT in the pipeline carries this weight.
- The 6-class space is fixed — adding "hat" would mean a new model.

---

### 4.8 HairTypeAnalyzer — `dima806/hair_type_image_detection`

[face-service/analyzers/hair_type_analyzer.py](face-service/analyzers/hair_type_analyzer.py)

**What it is.** Another ViT-B/16 from the same author. Predicts hair
texture as one of 5 classes: `curly`, `dreadlocks`, `kinky`,
`straight`, `wavy`.

**Specs:**
- Architecture: ViT-B/16
- Size: ~340 MB
- License: Apache 2.0
- Source: [dima806/hair_type_image_detection](https://huggingface.co/dima806/hair_type_image_detection)

**Accuracy:** 93% overall.

| Class | F1 |
|---|---|
| Dreadlocks | 0.978 |
| Kinky | 0.949 |
| Straight | 0.927 |
| Curly | 0.902 |
| Wavy | 0.884 |

(F1 is the harmonic mean of precision and recall — 1.0 is perfect.)

**How we use it.** Same pattern as 4.7: full softmax distribution
exposed as `hair_type_scores`, argmax exposed as `hair_type`,
top-1 probability as `hair_type_confidence`. The UI displays
`hair_type` with the confidence underneath.

This model replaces the coarse Laplacian-std fallback in
ColorAnalyzer (4.6) — both fields exist in the result dict but the
UI only displays this one.

**Pros:**
- 93% accuracy on a niche task is excellent.
- Trained specifically for hair texture, so it handles dark hair
  textures correctly where colour-based heuristics fail.

**Cons:**
- 340 MB, like all ViTs in this pipeline.
- 5 classes means "wavy-curly" gets bucketed one way or the other.

---

### 4.9 (Bonus) Sentence embeddings for search — `sentence-transformers/all-MiniLM-L6-v2`

[client/src/lib/embeddings.js](client/src/lib/embeddings.js)

Not part of the face-service pipeline, but worth covering since it's
the other ML model in the project. Used client-side, not server-side.

**What it is.** A sentence-transformer that maps any short piece of
text to a 384-dimensional vector ("embedding"). Texts with similar
meaning end up close to each other in that 384-d space. We use it to
make contacts searchable by free text: "the woman with curly red hair
I met at the dog park" produces a vector close to a contact whose
saved attributes include "Female", "curly hair", "red hair",
"Location: dog park".

**Specs:**
- Architecture: MiniLM (a distilled BERT — 6 layers, 22M params)
- Output dimension: 384
- Size: 80 MB, but we don't host it — we call the
  [HF Inference API](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
  with `EXPO_PUBLIC_HF_API_KEY`.
- License: Apache 2.0

**How it fits in.** When you save a contact, the client builds a
short description string out of name, location, demographics, and
the reliable facial-analysis fields (see `buildSearchableText` in
[embeddings.js](client/src/lib/embeddings.js)). That string is sent
to HF, which returns a 384-d vector. The vector is stored alongside
the contact in Supabase. Search compares the search-query vector to
stored vectors using cosine similarity.

**Pros:** runs in the cloud (no local model), fast, free tier covers
hobby use.
**Cons:** requires an API key; offline use needs us to host the
model ourselves; the key being missing silently disables embeddings
(the warning fires once on startup).

---

## 5. Data model

### Supabase tables

The `people` table is the main store. Each row is a single
photo-event pair — if you meet the same person three times, you get
three rows that share a `name` and `user_id`.

Selected columns:

| Column | Type | Notes |
|---|---|---|
| `id` | uuid (PK) | auto |
| `user_id` | uuid (FK) | the logged-in user |
| `name` | text | the contact's name |
| `phone`, `title`, `event`, `location`, `notes` | text | optional metadata |
| `date` | text | "YYYY-MM-DD" |
| `photo_url` | text | public URL into the photos storage bucket |
| `facial_details` | jsonb | the full `/analyze` response |
| `embedding` | vector(384) | search vector (if `pgvector` is enabled) |
| `created_at` | timestamptz | auto |

### Storage

Photos live in a public Supabase storage bucket called `photos`,
keyed by `<user_id>/<timestamp>.jpg`. The public URL is what gets
saved into `people.photo_url`.

### `facial_details` JSON shape

This is the dictionary returned by the face-service, wrapped in
`{ success: true, data: {...} }`. The `data` object has the full set
of ~100 attributes — see the per-attribute → source table in
[face-service/architecture.md](face-service/architecture.md) for the
canonical list.

---

## 6. Client UX walkthrough

### Tab 1 — Contacts (Lookup)

The list of saved people. Tapping a row opens **EditProfileScreen**.
The search box at the top runs against `name`, `notes`, and the
sentence-embedding vector (if available).

### Tab 2 — Add (Record)

The capture flow:

1. Take a photo (CameraView) or pick one from the library.
2. Optionally capture current GPS location.
3. Tap a chip (Name / Relation / Event / Location / Date / Notes) to
   expand a quick-edit panel.
4. Save.

Behind the scenes on save:
- The photo uploads to Supabase storage.
- A new row goes into `people` with the metadata.
- The image is base64'd and sent to the Express server
  (`/analyze-face`), which forwards to the face-service.
- The analysis response is stored in `facial_details`.
- `buildSearchableText` runs and the resulting string is sent to HF
  for embedding; the 384-d vector is saved in `embedding`.

### EditProfileScreen

Three blocks:

- **Header & info fields** — the editable metadata.
- **Photo gallery** — every photo of this contact (deduped by
  `name`), with a "Delete this photo" button.
- **Facial Analysis** — only visible if the user enables it in
  Settings. Renders every reliable attribute, grouped into sections
  (Demographics, Emotion, Face Structure, Hair, Eyes, Eyebrows, Nose,
  Lips & Mouth, Skin, Accessories), with each row showing which
  model produced the value.

Text in EditProfileScreen is selectable (long-press to copy) because
of the global `Text.defaultProps.selectable = true` set in
[App.js](client/App.js).

### Tab 3 — Settings

Theme toggle, "show facial details" debug toggle, sign out.

---

## 7. Design decisions and trade-offs

**Why MediaPipe over face-api.js.** face-api.js is great for the
browser (TFJS, tiny models), but only gives 68 2D landmarks and no
blendshapes. We rely heavily on the 478 3D landmarks + 52 blendshapes
for things like cheekbone prominence (needs Z), eyes_open (uses
`eyeBlink` blendshapes), and smiling (uses `mouthSmile` blendshapes).

**Why FaRL was removed.** FaRL is a face-tuned CLIP variant we
originally used for zero-shot attribute prompts ("is this person
wearing glasses?"). It worked for some attributes but routinely
hallucinated others — predicting "wearing earrings" or "receding
hairline" when neither was present. CLIP-style zero-shot is genuinely
not great for fine-grained facial binary attributes, and the
hallucinations were leaking into search embeddings. We replaced the
two attributes worth keeping (glasses, hair texture) with specialised
ViTs from `dima806` and dropped the rest.

**Why specialised `dima806` ViTs over training our own.** They're
published, documented, Apache-licensed, and the precision/recall on
the classes we care about are at or above 95%. Training our own
classifier on CelebA would take a GPU and a week; this took ten
minutes of integration. The [CUSTOM_ATTRIBUTE_MODEL_PLAN.md](CUSTOM_ATTRIBUTE_MODEL_PLAN.md)
spells out the path to going further if the off-the-shelf models stop
being enough.

**Why ColorAnalyzer is pure OpenCV.** Pixel-level colour averaging
inside a known-good mask is the kind of task where an ML model would
add latency and non-determinism without making the result more
correct. The LAB-trimmed median approach is robust to lighting
changes and runs in microseconds.

**Why FairFace + Ethnicity as separate models.** Avoids re-loading
one large multi-task checkpoint. Each is independently swappable.

**Why lazy loading.** HuggingFace Spaces free tier spins the
container down after ~30 min of inactivity. Loading all seven models
at process start would make the first request after a cold start
take 60+ seconds. With lazy loading, only the first analyzer's
weights are loaded on the first request; subsequent requests warm up
the others.

---

## 8. Development setup

### Prereqs
- Node 18+
- Python 3.11
- Expo CLI (`npm i -g expo-cli`)
- Docker (for face-service)
- A Supabase project (free tier is enough)
- (Optional) Hugging Face account + API key for embeddings

### Environment variables

`client/.env`:
```
EXPO_PUBLIC_SUPABASE_URL=...
EXPO_PUBLIC_SUPABASE_ANON_KEY=...
EXPO_PUBLIC_HF_API_KEY=...        # optional, disables embeddings if absent
```

`server/.env`:
```
FACE_SERVICE_URL=http://localhost:7860   # or your HF Space URL
```

### Run each tier

**Face service (Docker):**
```bash
cd face-service
docker build -t hcp-face .
docker run -p 7860:7860 hcp-face
```

**Express server:**
```bash
cd server
npm install
node src/app.js
```

**React Native client:**
```bash
cd client
npm install
npx expo start
```
Scan the QR with the Expo Go app, or hit `i` / `a` in the terminal
to launch the iOS simulator / Android emulator.

---

## 9. Deployment

**Face service → Hugging Face Spaces.** The
[face-service/README.md](face-service/README.md) header is the HF
Spaces YAML — pushing this folder to a `docker` Space and HF builds
+ hosts the container. Free tier gives 2 GB RAM, shared CPU, and
spins down after inactivity. URL becomes
`https://USERNAME-spacename.hf.space`.

**Express server.** Any Node host — Railway, Render, Fly, your own
VM. Set `FACE_SERVICE_URL` to the HF Space URL.

**React Native client.** Built with `eas build` (Expo Application
Services) for App Store / Play Store; or just keep it in Expo Go for
development.

---

## 10. Performance characteristics

Numbers are rough — measured on HF Spaces free tier (shared CPU, no
GPU). All times are per-request on a warm container.

| Step | Model | Cold-load time | Warm inference |
|---|---|---|---|
| 1 | MediaPipe Landmarker | ~1 s | ~50 ms |
| 2 | DemographicAnalyzer (3 ViTs) | ~10 s | ~500 ms |
| 3 | SegFormer-B5 parsing | ~5 s | ~1.5 s |
| 4 | HSEmotion EfficientNet-B0 | ~2 s | ~100 ms |
| 5 | ColorAnalyzer | 0 | ~30 ms |
| 6 | ObstructionViT | ~4 s | ~400 ms |
| 7 | HairTypeViT | ~4 s | ~400 ms |
| **Total (warm)** | | | **~3 s** |
| **Total (cold start)** | | ~30 s | **~30 s + 3 s** |

The bottleneck is SegFormer; the next-biggest contributors are the
ViTs. On a GPU each would drop to ~50 ms inference.

**Cold start budget.** From the user's perspective, the first photo
after the HF Space wakes up takes ~30 seconds. We mitigate this with
a periodic GET `/health` ping from the client to keep the Space warm
when the user is active.

---

## 11. Known limitations and roadmap

**Per-year age accuracy** — currently bound by the 9-bucket FairFace
classifier. Real per-year (±2-3 year MAE) needs a regression model
like MiVOLO. Listed in roadmap.

**Attributes with no good off-the-shelf model** — bangs, bald,
receding hairline, facial hair (beard/mustache/goatee/sideburns),
heavy makeup, lipstick, earrings, necklace, necktie. These were
previously powered by zero-shot CLIP/FaRL and removed because the
results were too noisy. The path forward is to fine-tune a small
classification head on top of a face-tuned encoder for these
specific attributes — see [CUSTOM_ATTRIBUTE_MODEL_PLAN.md](CUSTOM_ATTRIBUTE_MODEL_PLAN.md).

**Single-face only** — MediaPipe is configured with `num_faces=1`.
Group photos return analysis of whichever face MediaPipe locks onto
first, with no indication that others were ignored. Multi-face
support would mean returning a list of result dicts instead of one.

**No face matching** — we don't compute or compare face-recognition
embeddings (ArcFace, FaceNet etc.), so we can't auto-suggest "this
looks like someone you've met before". A face-recognition model
(InsightFace Buffalo_SC, ~30 MB) would slot in as analyzer #8.

**No quality gate** — a blurry / sideways / occluded face still gets
analysed, and the model just returns its best guess. A simple face
quality classifier as a preflight would let us reject obviously bad
photos before spending the inference budget.

---

## 12. Glossary

**Blendshape.** A single facial-muscle activation score between 0
and 1. ARKit defines 52 standard blendshapes (mouthSmileLeft,
jawOpen, eyeBlinkRight, etc.). MediaPipe's face landmarker emits
ARKit-compatible blendshapes alongside its 478 landmarks.

**CLIP.** OpenAI's "Contrastive Language-Image Pretraining" model.
Maps images and text into a shared embedding space so you can ask
"how similar is this image to the phrase 'a photo of someone wearing
glasses'?" Used for zero-shot classification. Not currently in this
project — we evaluated it (FaRL variant) and dropped it because
zero-shot accuracy on fine facial attributes was poor.

**Cosine similarity.** Standard way of measuring how similar two
embedding vectors are. Computes the cosine of the angle between
them, ranges [-1, +1]. 1.0 means identical direction, 0.0 means
orthogonal (unrelated).

**Embedding.** A fixed-length vector that represents an input (text,
image, audio) in a way where "similar things" are close together.
This project uses 384-d text embeddings from `all-MiniLM-L6-v2` for
search.

**EfficientNet.** A CNN architecture family (Google, 2019) that
scales depth, width, and resolution jointly for best accuracy per
parameter. B0 is the smallest, B7 the largest. We use B0 for emotion.

**F1 score.** Harmonic mean of precision and recall. Used when both
false positives and false negatives matter equally. 1.0 is perfect,
0.5 is poor.

**FaRL.** A face-tuned variant of CLIP (Microsoft Research, CVPR
2022). Same architecture as CLIP ViT-B/16 but trained on LAION-Face
(50 M face-text pairs). We tried it, found zero-shot facial attribute
accuracy still wasn't good enough, removed it.

**Fitzpatrick scale.** A six-step skin-type classification (Type I —
Very Fair, through Type VI — Dark Brown/Black) originally developed
for dermatology to predict sunburn risk. Used here as a categorical
skin-tone field.

**FastAPI.** A modern Python web framework. Faster than Flask, with
type hints and automatic OpenAPI docs. Hosts the face-service.

**HF / Hugging Face.** A platform that hosts ML models and datasets,
plus an `Inference API` for calling models without hosting them
yourself. Most of our models live there.

**HSV / LAB.** Two alternative colour spaces to RGB.
- HSV (Hue, Saturation, Value): closer to how humans describe
  colour. Used for hue-bucket classification (eye colour, hair
  colour).
- LAB (L\*, a\*, b\*): L\* is perceptual lightness 0-100, a\* is
  green↔red, b\* is blue↔yellow. Closer to perceptual uniformity
  than RGB and stable across lighting. Used for skin tone and
  undertone.

**IoU (Intersection over Union).** Segmentation accuracy metric. If
you predict a region and there's a true region, IoU is `overlap /
union`. 1.0 perfect, 0.5 decent, 0.8+ excellent.

**Laplacian.** A second-derivative image filter that responds to
local intensity curvature. Strong responses on edges and fine
texture. We use std(Laplacian) inside the face mask as a wrinkle
proxy and inside the hair mask as a hair-texture proxy.

**Logits.** The raw, unnormalised scores a classifier outputs before
softmax. Useful when you need to do math on them (multiply, add,
average) before turning them into probabilities.

**MediaPipe.** Google's cross-platform framework for ML pipelines —
hand tracking, pose estimation, face mesh, etc. We use the Face
Landmarker task.

**Mean IoU (mIoU).** Average IoU across all classes in a
segmentation dataset.

**NME (Normalized Mean Error).** Landmark-prediction error normalised
by interocular distance. Smaller is better. ~5% is good for mobile
landmark models.

**SegFormer.** A semantic-segmentation architecture (NeurIPS 2021)
combining a hierarchical transformer encoder with a simple MLP
decoder. We use SegFormer-B5 for human parsing.

**Semantic segmentation.** Per-pixel classification — every pixel
gets a class label. The output is a same-size map. Contrast with
"image classification" (one label per whole image) and "object
detection" (boxes around objects).

**Softmax.** The function that turns raw logits into probabilities
that sum to 1. Defined as `exp(x_i) / Σ exp(x_j)`. Standard final
step for classifiers.

**Supabase.** An open-source Firebase alternative — hosted Postgres
+ Auth + Storage. The contacts, photos, and embeddings all live
there.

**ViT (Vision Transformer).** A transformer architecture applied to
images by chopping them into patches and treating each patch as a
token. ViT-B/16 = "base" size, 16×16-pixel patches. Most of our
specialised classifiers (FairFace, Ethnicity, Obstruction, HairType)
are ViT-B/16.

**Zero-shot classification.** Asking a model that was never
explicitly trained on a label to nonetheless classify it. Typically
done via text prompts in CLIP-style models. Easy to add new labels
but accuracy is lower than dedicated fine-tuned classifiers.
