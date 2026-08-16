# RememberMe

**A mobile app for remembering the people you meet — capture a photo, and computer vision extracts ~100 facial attributes you can later search by free text.**

RememberMe (internally "HCP") is a cross-platform mobile contact manager built around a face-analysis pipeline. You take someone's photo and jot down where and when you met them; the app runs the image through a Python microservice that extracts roughly 100 facial attributes (age, gender, emotion, glasses, hair colour/texture, face shape, skin tone, and more). Later you can find that person by name, by attribute, or by natural-language search such as *"the woman with curly red hair I met at the dog park."*

---

## Overview

The system is split into three runtimes plus a managed backend:

```
client/         React Native (Expo)   — the iOS/Android app
server/         Node + Express        — thin proxy in front of the ML service
face-service/   Python + FastAPI      — runs the ML models (7 analyzers)
Supabase        (managed)             — auth, Postgres database, photo storage, vector search
```

**Why three tiers?** The heavier vision models can't run on a phone (battery/size), and Supabase Edge Functions are JS/TS-only with capped RAM, so they can't host Python ML. Splitting inference into a Python sidecar lets the project use any Hugging Face model without fighting runtime constraints. The Node server in the middle is a forwarding shim that gives a stable client-facing URL and a place to add auth/rate-limiting without redeploying the ML service.

```
┌────────────────────────┐          ┌────────────────────────┐
│   React Native client  │          │      Supabase          │
│   (Expo, iOS/Android)  │ ───────► │ • Auth (email/pass)    │
│   photos, search, UI   │ ◄─────── │ • people table         │
└────────────┬───────────┘          │ • photos bucket        │
             │ POST /analyze-face   │ • pgvector search      │
             ▼                      └────────────────────────┘
┌────────────────────────┐   text embeddings via HF Inference API
│   Express server       │   (sentence-transformers/all-MiniLM-L6-v2)
│   (Node, port 3000)    │
└────────────┬───────────┘
             │ POST /analyze-base64
             ▼
┌────────────────────────┐
│   Python face-service  │
│   (FastAPI, HF Spaces)  │
│   7 analyzers → ~100    │
│   facial attributes     │
└────────────────────────┘
```

---

## Features

- Email/password sign-in (Supabase Auth).
- Add a person with a photo (camera or library), name, phone, relation, event, location, date, and notes.
- Automatic facial analysis on save — ~100 attributes attached to each record.
- Optional GPS location capture on the Add screen.
- Hybrid search: exact text matching **plus** natural-language semantic search over facial attributes and metadata.
- Per-contact profile view with the full facial-analysis breakdown, grouped into sections, each row labelled with the model that produced it (view is opt-in via Settings).
- Photo gallery per contact (deduped by name) with per-photo delete.
- App-wide dark/light theme toggle; all text is long-press copyable.

---

## Architecture / How It Works

### Mobile app (`client/`)

Built with **Expo**-managed React Native. Screens live in [`client/src/screens/`](client/src/screens/):

| Screen | Purpose |
|---|---|
| `AuthScreen` | Email/password sign-in via Supabase Auth |
| `RecordScreen` | Camera / image picker, GPS capture, save a new contact |
| `LookupScreen` | Contact list + hybrid (exact + semantic) search |
| `EditProfileScreen` | Edit metadata, photo gallery, full facial-analysis panel |
| `SettingsScreen` | Theme toggle, "show facial details" toggle, sign out |

State management is intentionally light: React `useState`/`useEffect`, two context providers ([`AuthContext`](client/src/context/AuthContext.js), [`ThemeContext`](client/src/context/ThemeContext.js)), and `AsyncStorage` for sticky preferences. Data lives in Supabase and is re-fetched on screen focus — no Redux/Zustand. Navigation uses React Navigation (bottom tabs + native stack). Every `<Text>` is made selectable app-wide in [`client/App.js`](client/App.js).

**On save** ([`RecordScreen.js`](client/src/screens/RecordScreen.js)): the photo uploads to Supabase Storage, a `people` row is inserted, the base64 image is POSTed to the Express server (`/analyze-face`), the returned analysis is stored in `facial_details`, and `buildSearchableText` produces a description string that is embedded and stored for search.

### CV / face-analysis pipeline (`face-service/`)

A **FastAPI** app ([`face-service/app.py`](face-service/app.py)) that runs seven analyzers sequentially over one image and merges their outputs into a single ~100-field dictionary. Each analyzer is a small class ([`face-service/analyzers/`](face-service/analyzers/)) with `__init__()` (loads model) and `analyze(img_rgb) → dict`. Models are **lazy-loaded** on first request to keep cold-start latency manageable on the Hugging Face Spaces free tier. Later analyzers can overwrite keys from earlier ones (specialized ViTs run last so their outputs win).

| # | Analyzer | Model | What it produces |
|---|---|---|---|
| 1 | [`landmark_analyzer.py`](face-service/analyzers/landmark_analyzer.py) | MediaPipe Face Landmarker (~4 MB) | 478 3D landmarks + 52 ARKit blendshapes → face/eye/nose/lip/jaw geometry, smiling, mouth-open |
| 2 | [`demographic_analyzer.py`](face-service/analyzers/demographic_analyzer.py) | 3 ViT-B/16 classifiers (FairFace age, FairFace gender, `cledoux42/Ethnicity_Test_v003`) | Age (softmax-weighted continuous estimate), gender, ethnicity + full distributions |
| 3 | [`parsing_analyzer.py`](face-service/analyzers/parsing_analyzer.py) | SegFormer-B5 human parsing (~340 MB) | 18-class pixel masks → face/hair masks, hair length, hat, wrinkle/freckle/skin-uniformity stats (OpenCV) |
| 4 | [`emotion_analyzer.py`](face-service/analyzers/emotion_analyzer.py) | HSEmotion EfficientNet-B0 (~20 MB) | 8-class emotion + derived valence, arousal, mood |
| 5 | [`color_analyzer.py`](face-service/analyzers/color_analyzer.py) | None — pure OpenCV (LAB/HSV) | Skin tone (Fitzpatrick + hex), undertone, hair colour, eye colour, lip colour |
| 6 | [`obstruction_analyzer.py`](face-service/analyzers/obstruction_analyzer.py) | `dima806/face_obstruction_image_detection` (ViT-B/16) | glasses / sunglasses / mask flags |
| 7 | [`hair_type_analyzer.py`](face-service/analyzers/hair_type_analyzer.py) | `dima806/hair_type_image_detection` (ViT-B/16) | hair texture: curly / dreadlocks / kinky / straight / wavy |

MediaPipe provides the geometry and the masks that ColorAnalyzer depends on, so it and SegFormer run early; the color pass reuses their outputs rather than re-inferring. Full per-attribute → source attribution is in [`face-service/architecture.md`](face-service/architecture.md), and a top-to-bottom tour with model deep-dives is in [`OVERVIEW.md`](OVERVIEW.md).

### Node/Express server (`server/`)

A single file, [`server/src/app.js`](server/src/app.js). One real endpoint, `POST /analyze-face`, strips the data-URI prefix and forwards `{ image }` to `${FACE_SERVICE_URL}/analyze-base64`, returning the response unchanged. It exists to keep the client URL stable across face-service moves, to centralize CORS, and to host future JWT auth / rate-limiting before spending inference cycles.

### Semantic search

Search is **hybrid**, implemented in [`LookupScreen.js`](client/src/screens/LookupScreen.js):

1. **Exact match** — a Supabase `ilike` query across `name`, `phone`, `title`, `location`, `event`, `notes`, and `searchable_text`.
2. **Semantic match** — the query is embedded with `sentence-transformers/all-MiniLM-L6-v2` (384-d) via the Hugging Face Inference API ([`client/src/lib/embeddings.js`](client/src/lib/embeddings.js)), then compared against stored contact vectors through a Supabase RPC (`search_contacts`, cosine similarity, `match_threshold` 0.3) backed by `pgvector`.
3. Results are merged and deduped (exact matches ranked first, then most-recent-per-name).

When a contact is saved, [`buildSearchableText`](client/src/lib/embeddings.js) assembles a description from name/metadata and the **reliable** facial-analysis fields (glasses/sunglasses/mask from ObstructionViT, hat from SegFormer, smiling from MediaPipe blendshapes, hair colour/texture, eye colour, face structure, emotion, etc.), which is embedded and stored alongside the row. Embeddings degrade gracefully — if `EXPO_PUBLIC_HF_API_KEY` is absent, search silently falls back to exact matching.

---

## Tech Stack

- **Mobile:** React Native 0.81, Expo 54, React 19, React Navigation (tabs + native stack), `expo-camera`, `expo-image-picker`, `expo-location`, AsyncStorage.
- **Backend proxy:** Node.js, Express 4, `cors`, `dotenv`, nodemon.
- **ML service:** Python 3.11, FastAPI, Uvicorn, PyTorch + Torchvision, Transformers, timm, MediaPipe, HSEmotion, OpenCV (headless), Pillow, NumPy; `hf_transfer` for fast model downloads.
- **Data / infra:** Supabase (Postgres, Auth, Storage, `pgvector`), Hugging Face Inference API (text embeddings), Hugging Face Spaces (Docker deployment of the face-service).

---

## ML Model Details

All face-service models are pre-trained/off-the-shelf, chosen for documented accuracy and permissive licensing. Reported metrics (from model cards / benchmarks):

- **FairFace gender** (ViT-B/16): ~93.4% accuracy.
- **FairFace age** (ViT-B/16): ~59% top-1 over 9 buckets; the pipeline reports a **softmax-weighted continuous estimate** (Σ bucket-midpoint × probability) instead of a bucket midpoint, so estimates vary smoothly across people (~±5 yr practical error).
- **Ethnicity** (`cledoux42/Ethnicity_Test_v003`, ViT): ~79.6% overall, macro-F1 ~0.797; 5 classes mapped into a legacy 7-bucket schema internally.
- **SegFormer-B5 human parsing:** mean IoU ~0.626; face class acc 0.909 / IoU 0.829, hair class acc 0.897 / IoU 0.817.
- **HSEmotion EfficientNet-B0:** ~66.5% on AffectNet-8 (near SOTA for 8-class emotion).
- **Obstruction ViT** (`dima806`): ~99% precision/recall on glasses / sunglasses / mask.
- **Hair-type ViT** (`dima806`): ~93% overall accuracy.
- **Search embeddings:** `all-MiniLM-L6-v2` (distilled BERT, 22M params, 384-d output).

### Self-trained beauty regressor (SCUT-FBP5500)

Separate from the seven off-the-shelf analyzers, the author trained and published a custom facial-beauty model to Hugging Face: **[`evanlyhf/scut-fbp5500-beauty`](https://huggingface.co/evanlyhf/scut-fbp5500-beauty)**.

- **Architecture:** a single-output regression head on top of a **timm ResNet-50** backbone, fine-tuned end-to-end.
- **Dataset:** [SCUT-FBP5500](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release) — 5,500 frontal face images, each labelled with a facial-beauty score averaged from 60 human raters on a 1–5 scale (the standard benchmark for facial-beauty prediction).
- **Predicts:** a continuous beauty score, a `float` in `[1.0, 5.0]`, where higher = more conventionally attractive per the dataset's averaged human ratings.
- **Artifact:** the trained weights are published as `beauty_regressor.pt` (≈94 MB) in the model repo. The model card does not publish evaluation metrics (no MAE/correlation is reported on the card), and no training notebook is linked from the repo.

**Integration status: trained and published to Hugging Face, but not yet wired into the running app.** The current shipping pipeline is the seven analyzers above; the face-service exposes no beauty analyzer, and no client/server code calls the SCUT model or its HF endpoint. Consuming the beauty score (e.g. as an eighth analyzer in `face-service/`, or via the HF Inference API) is a planned next step, not a live feature.

**Design note on the earlier "attractiveness" attribute.** An earlier version of the pipeline used a zero-shot CLIP/FaRL analyzer for ~30 binary attributes (including a coarse binary "attractive" flag); it was removed because zero-shot accuracy on fine-grained facial attributes was poor and its hallucinations were polluting search embeddings. A separate plan to replace those ~30 CLIP flags with a self-trained ResNet-50 / ViT **multi-label** classifier (data collection via CelebA, BCE loss, frozen backbone + fine-tuned head, target >95% per-attribute) is documented in [`CUSTOM_ATTRIBUTE_MODEL_PLAN.md`](CUSTOM_ATTRIBUTE_MODEL_PLAN.md); that multi-label classifier is distinct from the SCUT-FBP5500 regressor above and remains a roadmap item.

---

## Setup

### Prerequisites

- Git, Node.js 18+ (20+ recommended), Python 3.11
- Expo Go on a phone, or an iOS/Android simulator
- A Supabase project (free tier is enough)
- Optional: Docker Desktop (to run the face-service in a container)
- Optional: a Hugging Face account + API key (for embeddings and/or Spaces deployment)

### Environment variables

`client/.env` (see [`client/.env.example`](client/.env.example)):

```dotenv
EXPO_PUBLIC_SUPABASE_URL=https://<your-project-ref>.supabase.co
EXPO_PUBLIC_SUPABASE_ANON_KEY=<your-supabase-anon-key>
EXPO_PUBLIC_FACE_ANALYSIS_URL=http://<your-node-server>:3000/analyze-face
EXPO_PUBLIC_HF_API_KEY=<your-hf-key>        # optional; disables embeddings if absent
```

The client talks to the Node server, not the Python service directly. On a physical phone, `localhost` points to the phone — use your computer's LAN IP.

`server/.env`:

```dotenv
FACE_SERVICE_URL=http://localhost:8000       # or your HF Space URL
```

### Install

```bash
cd client && npm install
cd ../server && npm install
cd ../face-service && python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1   |   macOS/Linux: source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
```

### Run locally (three terminals)

```bash
# 1) Python face-service
cd face-service && uvicorn app:app --host 0.0.0.0 --port 8000
#    health check: curl http://localhost:8000/health

# 2) Node server
cd server && FACE_SERVICE_URL=http://localhost:8000 npm start
#    health check: curl http://localhost:3000/hello

# 3) Expo client
cd client && npm start
```

### Run the face-service in Docker

```bash
cd face-service
docker build -t hcp-face .
docker run -p 7860:7860 hcp-face      # then set FACE_SERVICE_URL=http://localhost:7860
```

### Deploy the face-service to Hugging Face Spaces

The [`face-service/README.md`](face-service/README.md) header is the HF Spaces YAML config. Push `face-service/` to a **Docker** Space; HF builds and hosts it. Confirm `https://<your-space>.hf.space/health`, then point the Node server's `FACE_SERVICE_URL` at the Space. (Free tier spins down after inactivity; a periodic `/health` ping keeps it warm.)

### Supabase requirements

- A `people` table (columns below) and a public `photos` storage bucket keyed by `<user_id>/<timestamp>.jpg`.
- `pgvector` enabled with an `embedding vector(384)` column and a `search_contacts` RPC for cosine-similarity search.

`people` columns: `id` (uuid PK), `user_id` (uuid FK), `name`, `phone`, `title`, `event`, `location`, `date`, `notes`, `photo_url`, `facial_details` (jsonb), `searchable_text` (text), `embedding` (vector(384)), `created_at`.

---

## Usage

1. Sign in.
2. Go to the **Add** tab, enter the person's name and any details, and take or pick a photo (optionally capture GPS).
3. Save — the photo uploads, facial analysis runs, and a searchable embedding is stored.
4. Use the **Contacts** tab to search by name, metadata, or free-text descriptions of appearance, and tap a contact to view/edit their profile and facial-analysis breakdown.

### Useful endpoints

- Node server: `GET /hello`, `POST /analyze-face`
- Face-service: `GET /health`, `GET /`, `POST /analyze` (multipart), `POST /analyze-base64` (JSON `{ "image": "<base64>" }`)

---

## Project Structure

```
remember-me/
├── client/                       # Expo React Native app
│   ├── App.js                    # navigation, providers, global config
│   └── src/
│       ├── screens/              # Auth, Record, Lookup, EditProfile, Settings
│       ├── context/              # AuthContext, ThemeContext
│       └── lib/                  # supabase.js, embeddings.js
├── server/                       # Node + Express proxy
│   └── src/app.js
├── face-service/                 # Python FastAPI ML microservice
│   ├── app.py                    # pipeline orchestrator + endpoints
│   ├── analyzers/                # 7 analyzer modules
│   ├── Dockerfile                # HF Spaces / local container
│   ├── requirements.txt
│   ├── README.md                 # HF Spaces config + model table
│   └── architecture.md           # per-attribute model attribution
├── OVERVIEW.md                   # deep-dive tour of the whole system
├── architecture.md               # system architecture reference
├── CUSTOM_ATTRIBUTE_MODEL_PLAN.md# roadmap: self-trained attribute classifier
└── README.md
```

---

## Privacy Note

This app processes photographs of people and derives sensitive attributes (including demographic and appearance estimates) that are stored per-user in Supabase behind authentication. Facial analysis is **opt-in to display** in the UI. If deploying beyond personal use, review consent, data-retention, and applicable biometric/privacy regulations, and lock down the currently permissive CORS policy on the face-service. Attribute estimates are model predictions with the accuracies noted above — they are approximate and can be wrong.
