# HCP Architecture — Current Implementation (May 2026)

## System Overview

HCP (Human Connection Platform) is a face recognition and facial attribute analysis system with three main components:

```
┌─────────────────────────────────────────────────────────────────┐
│                    React Native Client (Expo)                    │
│  - Camera capture (RecordScreen.js)                              │
│  - Profile management (EditProfileScreen.js)                    │
│  - Search/lookup (LookupScreen.js)                              │
│  - Theme & auth context (ThemeContext, AuthContext)             │
└────────────┬────────────────────────────────────────────────────┘
             │ HTTP (base64 image + metadata)
             ▼
┌─────────────────────────────────────────────────────────────────┐
│              Node.js Express Server (server/src/app.js)          │
│  - Receives image from React client                              │
│  - Forwards to Python face-service                              │
│  - Routes: /hello, /analyze-face                                │
└────────────┬────────────────────────────────────────────────────┘
             │ HTTP (base64 image)
             ▼
┌─────────────────────────────────────────────────────────────────┐
│        Python FastAPI Service (face-service/app.py)             │
│  - 6 sequential analyzers (see "Analysis Pipeline" below)        │
│  - Returns ~120+ facial attributes as JSON                       │
│  - Runs on localhost:8000 (or HuggingFace Spaces in prod)        │
└────────────┬────────────────────────────────────────────────────┘
             │ JSON (facial_details)
             ▼
┌─────────────────────────────────────────────────────────────────┐
│            Supabase PostgreSQL (cloud database)                  │
│  - Table: people (user_id, name, phone, title, event,           │
│             location, date, notes, facial_details JSON)          │
│  - Storage: photos bucket (S3-compatible)                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow: Taking a Photo to Storing Results

### Step 1: User takes selfie (RecordScreen.js)
- Camera captures image → converted to base64 PNG
- User fills metadata: name, phone, title, event, location, date, notes
- "Submit" button clicked

### Step 2: Submit to Express server (RecordScreen.js → server/src/app.js)
```javascript
fetch('http://SERVER_URL/analyze-face', {
  method: 'POST',
  body: JSON.stringify({ image: base64Data })
})
```

### Step 3: Express forwards to Python service (server/src/app.js)
```javascript
const response = await fetch('${FACE_SERVICE_URL}/analyze-base64', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ image: base64Data })
})
```
- Express is a **proxy** — it just forwards the request
- FACE_SERVICE_URL = "http://localhost:8000" (dev) or HuggingFace Spaces URL (prod)

### Step 4: Python analyzes image (face-service/app.py)
See "Analysis Pipeline" section below — 6 models run in sequence, ~15-30 seconds total

### Step 5: Results stored in Supabase (RecordScreen.js)
```javascript
const { data, error } = await supabase
  .from('people')
  .insert({
    user_id: user.id,
    name, phone, title, event, location, date, notes,
    photo_url: 'https://supabase.co/storage/v1/...',
    facial_details: result // Raw JSON from Python service
  })
```

### Step 6: Display results (EditProfileScreen.js)
- User navigates to contact's profile
- Facial analysis data is loaded from `facial_details` JSON column
- Organized into 11 sections with method labels (MediaPipe, CLIP, FairFace, etc.)
- "Miscellaneous" section hidden by default (expandable)
- Color analysis shown at bottom of each relevant section

---

## Analysis Pipeline: The 6 Analyzers

All analyzers run **sequentially** on a single image in `face-service/app.py`:

### Analyzer 1: MediaPipe Face Landmarker (4 MB)
**File:** `face-service/analyzers/landmark_analyzer.py`
**Time:** ~2-3 seconds
**Source:** https://developers.google.com/mediapipe/solutions/vision/face_landmarker
**Outputs:**
- 478 3D facial landmarks (x, y, z coordinates)
- 52 blendshape scores (eye blink, mouth open, cheek puff, etc.)
- **Derived attributes:** face shape, face shape metrics, jawline type/angle, chin type, cheekbone prominence, cheek fullness, forehead width, eye shape/depth/spacing/size, eyebrow shape/arch/thickness, nose shape/bridge/tip, mouth width/lip fullness/cupid's bow, smile asymmetry, possible dimples, facial asymmetry score, eyes open

**Why it's first:** Other analyzers (color, emotion) need these landmarks as input for precise localization

### Analyzer 2: FairFace Demographics (170 MB)
**File:** `face-service/analyzers/demographic_analyzer.py`
**Time:** ~3-4 seconds
**Models Used:**
- Age: `dima806/fairface_age_image_detection` (ViT-B, 59% top-1 accuracy on 9 age buckets)
- Gender: `dima806/fairface_gender_image_detection` (ViT-B, 93.4% accuracy)
- Ethnicity: `cledoux42/Ethnicity_Test_v003` (ViT, 79.6% accuracy, 5 classes)
**Outputs:**
- gender, gender_confidence
- age_range (e.g., "20-29"), age_estimate (numeric), age_confidence
- age_distribution (scores for all 9 buckets: 0-2, 3-9, 10-19, ..., 70+)
- ethnicity, ethnicity_confidence
- ethnicity_distribution (scores for 7 classes: White, Black, Latino_Hispanic, East Asian, Southeast Asian, Indian, Middle Eastern)

### Analyzer 3: CLIP Attribute Classification (340 MB)
**File:** `face-service/analyzers/attribute_analyzer.py`
**Time:** ~4-5 seconds
**Model:** `openai/clip-vit-base-patch32` (ViT-B/32, zero-shot)
**Technique:** Binary pair classification — each attribute gets its own 2-way softmax:
- "wearing eyeglasses" vs "not wearing eyeglasses" → score 0.0-1.0
- Higher accessory threshold (0.65) to avoid false positives on jewelry
**Outputs:** ~30 binary attributes + hair color/texture classification:
- Boolean: wearing_glasses, wearing_hat, has_beard, mustache, goatee, sideburns, has_bangs, is_bald, receding_hairline, wearing_earrings, wearing_necklace, wearing_necktie, heavy_makeup, wearing_lipstick, big_nose, pointy_nose, big_lips, high_cheekbones, oval_face_celeba, double_chin, chubby, rosy_cheeks, bags_under_eyes, narrow_eyes, arched_eyebrows, bushy_eyebrows, pale_skin, attractive, young, smiling_celeba, mouth_open
- Categorical: hair_color_celeba (black, blond, brown, gray), hair_texture_celeba (straight, wavy, curly)
- Scores: hair_color_scores (dict with color probabilities)
- Grouped: facial_hair (5_o_clock_shadow, full_beard, goatee, mustache, sideburns)
- Raw: _celeba_raw (all ~30 attributes with raw scores 0.0-1.0)

### Analyzer 4: SegFormer Human Parsing (335 MB)
**File:** `face-service/analyzers/parsing_analyzer.py`
**Time:** ~2-3 seconds
**Model:** `matei-dorian/segformer-b5-finetuned-human-parsing` (SegFormer-B5, mIoU 0.626)
**Outputs:** 18-class semantic segmentation mask (face, hair, sunglasses, upper-clothes, skirt, pants, dress, belt, shoes, left-leg, right-leg, left-arm, right-arm, bag, scarf, hat, background)
**Derived attributes:**
- region_coverage (dict: face, hair, hat, sunglasses, etc. coverage %)
- _skin_mask, _hair_mask, _lip_mask (boolean numpy arrays, used by ColorAnalyzer)
- hair_length (bald/very short, short, medium, long — based on hair_pixels / face_pixels ratio)
- hair_present (boolean)
- glasses_detected, hat_detected, earring_detected (false, no class), necklace_detected (false, no class)
- Skin analysis on face_mask:
  - wrinkle_level (smooth, slight, moderate, prominent — based on Laplacian edge density)
  - skin_texture_score (float: edge density)
  - freckles_or_moles (few, some, many — based on dark spot ratio in L* channel)
  - skin_uniformity (float: std dev of L* values)

### Analyzer 5: HSEmotion Emotion Recognition (18 MB)
**File:** `face-service/analyzers/emotion_analyzer.py`
**Time:** ~1-2 seconds
**Model:** `enet_b0_8_best_afew` from HSEmotion package (EfficientNet-B0 fine-tuned on AFEW/AffectNet)
**Classes:** anger, contempt, disgust, fear, happiness, neutral, sadness, surprise
**Outputs:**
- primary_emotion (highest probability class)
- emotion_confidence (float 0.0-1.0)
- secondary_emotion (second highest)
- emotion_scores (dict: all 8 emotions with probabilities)
- valence (float -1 to +1: negative, neutral, positive mood estimation)
- arousal (float 0 to 1: calm to excited)
- mood (positive, neutral, or negative based on valence threshold)

### Analyzer 6: ColorAnalyzer — Pixel-Level Analysis (0 MB, no AI)
**File:** `face-service/analyzers/color_analyzer.py`
**Time:** ~1-2 seconds
**Technique:** OpenCV LAB/HSV color space analysis (no neural network)
**Inputs:** RGB image + landmarks + masks from previous analyzers
**Outputs:**
- **Skin tone** (from skin_mask):
  - fitzpatrick (Type I - Very Fair through Type VI - Dark Brown/Black based on LAB L*)
  - lab_lightness, lab_a, lab_b (LAB color space values)
  - hex_color, rgb (average skin color)
  - skin_undertone (warm/cool/neutral based on b* channel)
- **Eye color** (from iris landmarks):
  - Classification: brown, hazel, green, blue, gray, amber
  - Uses HSV hue ranges on iris region (468-477 landmarks)
- **Hair color** (from hair_mask):
  - name (black, red/auburn, blond, brown, gray/white, dark brown, unknown)
  - hex, rgb, hsv
  - Technique: LAB L*-trimmed median (removes highlights/shadows)
- **Hair texture** (from hair_mask):
  - Classification: straight, wavy, curly/coily, unknown
  - Technique: std dev of Laplacian on eroded mask (local intensity variation)
- **Lip color** (from lip_mask or lower face region):
  - shade (rosy/red, pink, dark, natural, unknown)
  - hex, rgb

---

## Data Schema: What Gets Stored

### Supabase Table: `people`

| Column | Type | Description |
|--------|------|-------------|
| id | uuid | Primary key (auto-generated) |
| user_id | uuid | Foreign key to auth.users |
| name | text | Contact name |
| phone | text | Contact phone (nullable) |
| title | text | Relation/title (nullable) |
| event | text | Occasion/event (nullable) |
| location | text | Location (nullable) |
| date | text | Date (YYYY-MM-DD, nullable) |
| notes | text | User notes (nullable) |
| photo_url | text | Supabase storage URL to photo |
| facial_details | jsonb | **Raw output from face-service** (~120+ attributes) |
| embedding | vector | Embedding vector for similarity search (nullable, for future) |
| created_at | timestamp | Auto-generated timestamp |
| updated_at | timestamp | Auto-generated timestamp |

### facial_details JSON Structure

```json
{
  "success": true,
  "data": {
    "gender": "Female",
    "gender_confidence": 0.954,
    "age_range": "20-29",
    "age_estimate": 24.5,
    "age_confidence": 0.621,
    "age_distribution": {"0-2": 0.0, "3-9": 0.0, ...},
    "ethnicity": "East Asian",
    "ethnicity_confidence": 0.823,
    "ethnicity_distribution": {"White": 0.05, "East Asian": 0.823, ...},
    
    "primary_emotion": "happiness",
    "emotion_confidence": 0.712,
    "secondary_emotion": "neutral",
    "emotion_scores": {"anger": 0.001, "happiness": 0.712, ...},
    "valence": 0.8,
    "arousal": 0.6,
    "mood": "positive",
    
    "face_shape": "oval",
    "face_shape_metrics": {"width_height_ratio": 0.82, ...},
    "jawline_type": "soft",
    "jawline_angle": 128.5,
    "chin_type": "normal",
    
    "hair_length": "medium",
    "hair_color_celeba": "brown",
    "hair_texture_celeba": "straight",
    "hair_color": {"name": "brown", "hex": "#6b4423", "rgb": [107, 68, 35]},
    "hair_texture": "straight",
    
    "eye_shape": "almond",
    "eye_color": "brown",
    "eye_spacing": "average",
    "eye_size": "average",
    
    "skin_tone": {
      "fitzpatrick": "Type III - Medium",
      "lab_lightness": 65.3,
      "lab_a": 12.5,
      "lab_b": 18.2,
      "hex_color": "#d4a574",
      "rgb": [212, 165, 116]
    },
    "skin_undertone": "warm",
    "wrinkle_level": "smooth",
    "freckles_or_moles": "few",
    
    "wearing_glasses": false,
    "wearing_hat": false,
    "heavy_makeup": true,
    "wearing_lipstick": true,
    
    ...120+ more attributes
  }
}
```

---

## Environment Configuration

### Express Server (server/.env)
```
FACE_SERVICE_URL=http://localhost:8000  # For local development
# OR in production:
# FACE_SERVICE_URL=https://username-hcp-service.hf.space
```

### Python Service (face-service/)
- Uses HuggingFace Transformers library
- Models auto-downloaded on first use to `~/.cache/huggingface/`
- MediaPipe model cached to `./models/face_landmarker.task`
- All models loaded on first request (cold start: 30-60 seconds)
- Subsequent requests: 15-30 seconds per image

### React Client (client/)
- AsyncStorage: stores "showFacialDetails" preference
- Supabase client: authenticated access to people table & photos storage
- Embedding library: generates searchable text from facial attributes

---

## Performance Metrics

### Analysis Time per Image
| Analyzer | Time | Model Size |
|----------|------|-----------|
| MediaPipe | 2-3s | 4 MB |
| FairFace | 3-4s | 170 MB |
| CLIP | 4-5s | 340 MB |
| SegFormer | 2-3s | 335 MB |
| HSEmotion | 1-2s | 18 MB |
| ColorAnalyzer | 1-2s | 0 MB |
| **Total** | **~15-20s** | **~870 MB** |

**Cold start** (first request after server sleep): +30-60s to load all models into RAM

### Storage Requirements
- Model weights: ~870 MB
- Python + PyTorch + OpenCV: ~200 MB
- Total: ~1.1 GB disk

### Memory Requirements
- Peak RAM during inference: ~300-400 MB
- HuggingFace Spaces: 2 GB available (5x what we need)

---

## Analysis Sections & Display (EditProfileScreen.js)

The facial_details JSON is organized into 11 display sections:

1. **Demographics** (7 attributes) — FairFace, Ethnicity_Test_v003
2. **Emotion** (7 attributes) — HSEmotion
3. **Face Structure** (9 attributes) — MediaPipe
4. **Hair** (11 attributes + color at bottom) — CLIP, ColorAnalyzer, SegFormer
5. **Eyes** (7 attributes + color at bottom) — MediaPipe, ColorAnalyzer
6. **Eyebrows** (6 attributes) — MediaPipe, CLIP
7. **Nose** (6 attributes) — MediaPipe, CLIP
8. **Lips & Mouth** (11 attributes + color at bottom) — MediaPipe, CLIP, ColorAnalyzer
9. **Skin** (17 attributes) — ColorAnalyzer, CLIP, SegFormer
10. **Accessories** (7 attributes) — CLIP, SegFormer
11. **Miscellaneous** (hidden/expandable) — raw landmarks, blendshapes, CelebA raw scores, young, attractive, facial hair details

Each attribute labeled with its source method: `(MediaPipe)`, `(CLIP)`, `(FairFace)`, etc.

Footer includes 7-line legend explaining each analyzer's capabilities and accuracy.

---

## Known Limitations

1. **Age detection is weak** — FairFace achieves only 59% top-1 accuracy on age buckets
2. **Cold start delay** — First request after server sleep takes 30-60 seconds
3. **No GPU support** — CPU-only inference (can be upgraded on Spaces or cloud)
4. **Ethnicity model has only 5 classes** — mapped into legacy 7-bucket schema internally
5. **Ear/necklace detection unreliable** — SegFormer has no earring/necklace class; CLIP-based detection can hallucinate
6. **Single-face only** — MediaPipe configured for `num_faces=1`; crashes gracefully on multi-face images

---

## Deployment (Production)

### Recommended: HuggingFace Spaces
1. Create free Space at https://huggingface.co/spaces
2. Upload `face-service/` folder with Dockerfile or requirements.txt
3. Spaces provides:
   - 2GB RAM (plenty for our 400MB peak usage)
   - 50GB storage (100x what we need)
   - Auto-scaling (spins down after 15 min idle)
   - Free HTTPS URL
4. Set `FACE_SERVICE_URL` in Express env to Spaces URL
5. Add health check endpoint to keep warm: `/health` pinged every 10 minutes

### Alternative: Railway, Render, Google Cloud Run
See `models.md` for comparison table.

---

## Future Enhancements

- [ ] GPU support (NVIDIA CUDA) for 3-5x speedup
- [ ] Batch processing (analyze multiple photos in one request)
- [ ] Similarity search using embeddings (find similar faces in database)
- [ ] Model caching/quantization (reduce cold start)
- [ ] A/B testing: which models are most useful to users?
- [ ] Fine-tuning FairFace on our user base for better age/ethnicity accuracy
