# HCP Face Analysis — Models & Architecture

## How the Whole System Works (Start to Finish)

### The Old Way (What We Had)

```
1. User takes selfie in React Native app (Expo)
2. Photo converted to base64
3. Base64 sent to Express.js server (server/src/app.js)
4. Express spawns Python child process (rekognition/cli_wrapper.py)
5. Python sends image bytes to AWS Rekognition API ($$$)
6. AWS returns ~15 basic attributes (age range, gender, smile, glasses, beard, emotion)
7. Results stored as facial_details JSON in Supabase 'people' table
8. Frontend displays results on LookupScreen / EditProfileScreen
```

**Problems:**
- AWS Rekognition costs money per API call
- Only returns ~15 basic attributes
- We want ~150+ facial attributes
- Requires AWS credentials and internet connectivity to AWS

### The New Way (What We're Building)

```
1. User takes selfie in React Native app (Expo)          ← UNCHANGED
2. Photo converted to base64                              ← UNCHANGED
3. Base64 sent to Express.js server (server/src/app.js)   ← UPDATED endpoint
4. Express forwards image to Python FastAPI microservice   ← NEW
5. FastAPI runs 6 local models (no external API calls)     ← NEW
6. Returns ~150 facial attributes as JSON                  ← NEW
7. Results stored as facial_details JSON in Supabase       ← UNCHANGED (more data)
8. Frontend displays results (enhanced search/display)     ← UPDATED
```

---

## What is FastAPI?

FastAPI is a **Python web framework** — the same concept as Express.js but for Python.

```
Express.js  → JavaScript web framework → listens for HTTP requests, returns JSON
FastAPI     → Python web framework     → listens for HTTP requests, returns JSON
```

We need Python because every serious facial analysis model (MediaPipe, FairFace, BiSeNet,
HSEmotion) is written in Python with PyTorch/TensorFlow. These models don't exist in
JavaScript. So we need a Python server to run them.

FastAPI is NOT a separate concept from a "server" — it IS the server. It's just written
in Python instead of JavaScript.

---

## What Are the Hosting Resource Requirements?

### RAM (Random Access Memory)

RAM = temporary working memory while processing (like desk space).

When a model runs inference on an image:
1. Model weights loaded from disk into RAM (~210MB for all 6 models)
2. Input image loaded into RAM (~5-15MB)
3. Intermediate calculations in RAM (~50-100MB)
4. **Total RAM needed: ~300-400MB peak**

### Storage (Disk)

Storage = permanent file storage (like a filing cabinet).

- Model weight files: ~210MB
- Python packages (PyTorch, MediaPipe, OpenCV, etc.): ~250MB
- Application code: ~50KB
- **Total disk needed: ~500MB**

### Hosting Platform Comparison

| Platform | Free Tier | RAM | Storage | Cold Start | Best For |
|----------|-----------|-----|---------|------------|----------|
| **Hugging Face Spaces** | Unlimited | 2GB | 50GB | ~30-60s after idle | Best free option ✅ |
| Railway.app | $5 credit/mo | 512MB | 1GB | ~5s | Always-on API |
| Render.com | 750 hrs/mo | 512MB | 750MB | ~30s | Spins down after 15min |
| Google Cloud Run | 2M req/mo | 512MB-4GB | 10GB | ~10s | Best scaling |
| Fly.io | 3 shared VMs | 256MB | 3GB | ~3s | Low latency |

**Recommendation: Hugging Face Spaces** — 2GB RAM (5x what we need), 50GB storage (100x what
we need), free forever, pre-installed ML libraries.

**Cold start workaround:** The server sleeps after ~15min of no requests. First request
after sleep takes ~30-60s to wake up. Fix: add a health-check ping from Express every
10 minutes, or accept the delay.

---

## The 6 Models — What Each Does and Why

### Model 1: MediaPipe Face Landmarker (Geometric Features)
- **Size:** 4MB
- **What it does:** Detects the face, places 478 precise 3D points on it, measures 52 blendshapes (muscle movements)
- **What we derive from it:** Face shape, jawline, chin, cheekbones, forehead, eye shape, eye spacing, eye size, eyebrow shape, nose shape, lip shape, mouth width, dimples, facial asymmetry (~40 features)
- **How:** Pure math on the 3D point coordinates (distances, ratios, angles). No additional AI needed.
- **GitHub:** https://github.com/google-ai-edge/mediapipe
- **Weights:** Auto-downloaded from Google Storage (~4MB)

### Model 2: FairFace (Age, Gender, Race)
- **Size:** 90MB (ResNet-34)
- **Accuracy:** 93.4% race, 94.2% gender, MAE 3.4 years for age
- **What it does:** Classifies demographics from a face image
- **Output:** Age (9 buckets), gender (2), race (7 categories)
- **Why this over InsightFace:** FairFace is trained specifically for fair demographic classification across all races
- **GitHub:** https://github.com/dchen236/FairFace
- **Weights:** res34_fair_align_multi_7_20190809.pt from HuggingFace

### Model 3: CelebA Attribute Classifier (40 Binary Attributes)
- **Size:** 44MB (ResNet-18)
- **What it does:** Predicts 40 binary attributes trained on 200,000 celebrity faces
- **Output:** Bald, bangs, beard, goatee, mustache, sideburns, glasses, hat, bushy eyebrows, high cheekbones, oval face, pointy nose, big lips, receding hairline, straight/wavy hair, hair colors, smiling, young, etc.
- **Note:** We already have model/basic_cnn.py doing this — the new version uses a pre-trained ResNet-18 for better accuracy
- **GitHub:** https://huggingface.co/jnferreira/attribute-prediction-celebA

### Model 4: BiSeNet Face Parsing (Segmentation)
- **Size:** 50MB
- **What it does:** Labels every pixel as one of 19 categories (skin, hair, left eye, right eye, nose, upper lip, lower lip, glasses, hat, neck, etc.)
- **What we derive from it:**
  - Hair: length (from mask area), color (K-means clustering on hair pixels), texture (FFT frequency analysis)
  - Skin: tone (LAB lightness), freckles/moles (dark blob detection), wrinkles (edge detection on forehead)
  - Glasses/hat detection (backup confirmation)
- **GitHub:** https://github.com/zllrunning/face-parsing.PyTorch
- **Weights:** https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812

### Model 5: HSEmotion (Emotion Recognition)
- **Size:** 20MB (EfficientNet-B0)
- **Accuracy:** 66.5% on AffectNet-8 (state-of-the-art — humans only agree ~72%)
- **What it does:** Classifies facial expression into 8 emotions
- **Output:** angry, contempt, disgust, fear, happy, neutral, sad, surprise
- **GitHub:** https://github.com/HSE-asavchenko/face-emotion-recognition

### Model 6: Color Analysis (No AI — Pure Pixel Math)
- **Size:** 0MB (code only)
- **What it does:** Uses masks from Model 4 + landmarks from Model 1 to sample pixel colors
- **Output:**
  - Skin tone (LAB color space → Fitzpatrick scale I-VI)
  - Eye color (iris patch → HSV classification → brown/blue/green/hazel/amber)
  - Hair color (K-means on hair pixels → black/brown/blonde/red/gray)
  - Hair texture (FFT on grayscale hair → straight/wavy/curly)

---

## Complete Feature Coverage

| Feature | Model/Method | Confidence |
|---------|-------------|------------|
| **Face shape** (oval, round, square, heart, diamond, oblong, triangle) | MediaPipe geometric ratios + CelebA | ⭐⭐⭐⭐ |
| **Jawline** (sharp, soft, strong) | MediaPipe jaw landmark angles | ⭐⭐⭐⭐ |
| **Chin** (receding, pointed, cleft, wide) | MediaPipe chin landmarks + depth | ⭐⭐⭐ |
| **Cheekbones** (high, flat, full, hollow) | MediaPipe z-depth + CelebA | ⭐⭐⭐⭐ |
| **Forehead** (broad, narrow) | MediaPipe forehead span ratio | ⭐⭐⭐⭐ |
| **Eye shape** (almond, round, hooded, monolid, upturned, downturned) | MediaPipe eyelid curvature | ⭐⭐⭐⭐ |
| **Eye spacing** (wide-set, close-set) | MediaPipe interpupillary distance | ⭐⭐⭐⭐⭐ |
| **Eye size** (large, small) | MediaPipe eye/face area ratio | ⭐⭐⭐⭐⭐ |
| **Deep-set / protruding eyes** | MediaPipe z-depth at eye region | ⭐⭐⭐ |
| **Eye color** (brown, blue, green, hazel) | Iris crop → HSV classification | ⭐⭐⭐⭐ |
| **Dark under-eyes / eye bags** | BiSeNet + brightness analysis | ⭐⭐⭐ |
| **Crow's feet** | Canny edges on outer eye skin | ⭐⭐⭐ |
| **Eyebrow shape** (arched, straight, bushy, thick, thin) | MediaPipe brow landmarks + CelebA | ⭐⭐⭐⭐ |
| **Unibrow** | MediaPipe inner brow distance | ⭐⭐⭐⭐ |
| **Nose shape** (straight, aquiline, button, upturned, wide, narrow) | MediaPipe nose landmarks + CelebA | ⭐⭐⭐⭐ |
| **Nose bridge** (flat, high) | MediaPipe z-depth at bridge | ⭐⭐⭐ |
| **Nostrils** (wide, narrow) | MediaPipe nostril width ratio | ⭐⭐⭐⭐ |
| **Lips** (full, thin) | MediaPipe lip landmarks + CelebA | ⭐⭐⭐⭐ |
| **Mouth width** | MediaPipe mouth corners ratio | ⭐⭐⭐⭐⭐ |
| **Cupid's bow** | MediaPipe upper lip curvature | ⭐⭐⭐ |
| **Teeth** (gap, crooked, overbite) | Mouth crop → classifier (limited) | ⭐⭐ |
| **Dimples** | MediaPipe blendshapes + cheek analysis | ⭐⭐⭐ |
| **Smile lines** | Edge detection on nasolabial region | ⭐⭐⭐ |
| **Asymmetrical smile** | MediaPipe L/R blendshape diff | ⭐⭐⭐⭐ |
| **Hair type** (straight, wavy, curly, coily) | BiSeNet hair mask → FFT + CelebA | ⭐⭐⭐ |
| **Hair length** (short, long, bald) | BiSeNet hair mask area + CelebA | ⭐⭐⭐⭐ |
| **Hair color** (black, brown, blonde, red, gray) | BiSeNet mask → K-means + CelebA | ⭐⭐⭐⭐ |
| **Receding hairline / widow's peak** | BiSeNet boundary + CelebA | ⭐⭐⭐ |
| **Facial hair** (beard, stubble, goatee, mustache, sideburns) | BiSeNet + CelebA attributes | ⭐⭐⭐⭐ |
| **Skin tone** (light, medium, dark + Fitzpatrick I-VI) | BiSeNet skin mask → LAB brightness | ⭐⭐⭐⭐⭐ |
| **Freckles / moles** | BiSeNet skin mask → blob detection | ⭐⭐⭐ |
| **Scars** | BiSeNet skin → edge anomalies | ⭐⭐ |
| **Acne** | BiSeNet skin → red blob detection | ⭐⭐⭐ |
| **Wrinkles / forehead lines** | BiSeNet forehead mask → Canny edges | ⭐⭐⭐ |
| **Facial asymmetry** | MediaPipe L/R landmark mirror | ⭐⭐⭐⭐⭐ |
| **Glasses** | CelebA + BiSeNet parsing | ⭐⭐⭐⭐⭐ |
| **Age** | FairFace (MAE 3.4 years) | ⭐⭐⭐⭐⭐ |
| **Gender** | FairFace (94.2%) | ⭐⭐⭐⭐⭐ |
| **Race** | FairFace (93.4%, 7 categories) | ⭐⭐⭐⭐⭐ |
| **Emotion** | HSEmotion (66.5% AffectNet-8 SOTA) | ⭐⭐⭐⭐ |

---

## Processing Flow in Detail

### Step 0: Deployment (One-Time)

```
Developer machine
    │  git push to Hugging Face Spaces
    ▼
Hugging Face builds Docker container:
    1. Installs Python 3.11
    2. Installs PyTorch, MediaPipe, OpenCV, FastAPI, etc. (~250MB)
    3. Downloads 5 model weight files (~210MB):
       - face_landmarker.task        (4MB)   ← MediaPipe
       - fairface_model.pt           (90MB)  ← FairFace
       - celeba_resnet18.pt          (44MB)  ← CelebA
       - bisenet_face_parsing.pt     (50MB)  ← BiSeNet
       - hsemotion_enet_b0_8.pt      (20MB)  ← HSEmotion
    4. Starts FastAPI server on port 8000
    5. Server sits idle, waiting for HTTP requests

URL: https://your-username-hcp-face.hf.space
```

### Step 1: User Takes Photo (React Native — Unchanged)

```
RecordScreen.js
    │  User taps shutter button
    │  CameraView.takePictureAsync({ base64: true })
    │  → Returns photo.uri (for preview) and photo.base64 (for analysis)
    ▼
```

### Step 2: Image Sent to Express Server

```
RecordScreen.js → analyzeFace(photo.base64)
    │
    │  POST http://<express-server>:3000/analyze-face
    │  Body: { image: "<base64 string>" }
    ▼
server/src/app.js receives request
```

### Step 3: Express Forwards to Python Microservice

```
server/src/app.js
    │
    │  POST https://your-username-hcp-face.hf.space/analyze
    │  Body: multipart/form-data with image file
    ▼
FastAPI (face-service/app.py) receives the image
    │
    │  Decodes bytes → PIL Image → numpy array (H × W × 3 RGB values)
    ▼
```

### Step 4-9: Six Models Run Sequentially

```
Model 1: MediaPipe (4MB)
    Input:  RGB pixel array
    Output: 478 landmark coordinates + 52 blendshapes
    Then:   Geometric math → 40+ attributes (face shape, eye shape, etc.)
    Time:   ~200ms

Model 2: FairFace (90MB)
    Input:  Image resized to 224×224
    Output: age range, gender, race with confidence scores
    Time:   ~300ms

Model 3: CelebA (44MB)
    Input:  Image resized to 224×224
    Output: 40 binary attribute probabilities
    Time:   ~200ms

Model 4: BiSeNet (50MB)
    Input:  Image resized to 512×512
    Output: Pixel-level segmentation mask (19 classes)
    Then:   Hair length, wrinkle detection, spot detection
    Time:   ~400ms

Model 5: HSEmotion (20MB)
    Input:  Image resized to 224×224
    Output: 8 emotion probabilities
    Time:   ~150ms

Model 6: Color Analysis (0MB — pure code)
    Input:  Original image + skin mask from Model 4 + landmarks from Model 1
    Output: Skin tone, eye color, hair color, hair texture
    Time:   ~100ms
```

### Step 10: Results Combined and Returned

```
FastAPI combines all outputs → single JSON response (~5KB)
    │
    ▼
Express.js receives JSON, forwards to React Native
    │
    ▼
RecordScreen.js stores in state: setAnalysis(result)
    │
    ▼
On submit: saved to Supabase 'people' table as facial_details JSON
    │
    ▼
LookupScreen can search across all ~100+ attributes
EditProfileScreen can display all facial details
```

---

## Final Architecture Diagram

```
React Native (Expo) Mobile App
    │
    │  POST /analyze-face (base64 image)
    ▼
Express.js Server (server/src/app.js)
    │
    │  POST /analyze (image file)
    ▼
Python FastAPI Microservice (Hugging Face Spaces — FREE, 2GB RAM)
    ├── MediaPipe (4MB) ──────► 478 landmarks → ~40 geometric features
    ├── FairFace (90MB) ──────► age, gender, race
    ├── CelebA ResNet (44MB) ─► 40 binary attributes (hair, beard, glasses...)
    ├── BiSeNet (50MB) ───────► face parsing → hair/skin segmentation
    ├── HSEmotion (20MB) ─────► 8 emotions
    └── Color Analysis ───────► skin tone, eye color, hair color
    │
    │  JSON response (~100+ attributes)
    ▼
Supabase (Unchanged)
    ├── Auth (unchanged)
    ├── Storage for photos (unchanged)
    └── PostgreSQL 'people' table → facial_details JSON column (more data now)
```

| Metric | Old (Rekognition) | New (6 Models) |
|--------|-------------------|----------------|
| Features detected | ~15 | **~100+** |
| Cost per request | $0.001+ (adds up) | **$0** |
| Hosting cost | AWS charges | **$0** (HF Spaces free) |
| Total model size | N/A (cloud API) | ~210MB |
| Latency | ~1-2s | ~2-4s (CPU) |
| External dependency | AWS account + keys | None (self-hosted) |

---

## What Changes in the Codebase

| File / Area | What Changes |
|-------------|-------------|
| `face-service/` (NEW) | Entire Python microservice — FastAPI + 6 analyzers |
| `server/src/app.js` | `/analyze-face` now forwards to Python microservice instead of spawning local Python |
| `client/src/screens/RecordScreen.js` | `analyzeFace()` sends to Express which forwards to microservice; handles new response format |
| `client/src/screens/LookupScreen.js` | Search filter updated for new attribute names/structure |
| `client/src/screens/EditProfileScreen.js` | Facial details display updated for ~100+ attributes |
| `client/.env.example` | Add `EXPO_PUBLIC_FACE_SERVICE_URL` |
| `rekognition/` | Can be removed (replaced by face-service) |
