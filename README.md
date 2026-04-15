# HCP Project Setup Guide (From Zero)

This guide is for a brand-new machine where nothing is installed yet.

It covers:
1. Everything to install manually
2. Every account to create
3. All required environment variables
4. Exact terminal commands and where to run them
5. How to run all services together

## 1. Architecture Overview

This project has 3 local services plus hosted Supabase:

1. Expo React Native app (mobile client)
Path: `client/`

2. Node.js Express server (API gateway/proxy)
Path: `server/`

3. Python FastAPI face analysis service (ML models)
Path: `face-service/`

4. Supabase (hosted: auth, database, storage)

Request flow:
1. App sends image to Node at `/analyze-face`
2. Node forwards to Python at `/analyze-base64`
3. Python runs models and returns `{ success: true, data: {...} }`
4. App stores results in Supabase

## 2. Accounts You Must Create

Create these first before running code.

### 2.1 Supabase (Required)

1. Sign up at https://supabase.com
2. Create a new project
3. Save:
	- Project URL
	- Anon public key
4. In Supabase SQL Editor, create your `people` table and policies used by the app.
5. In Supabase Storage, create bucket `photos`.

Minimum table columns used by app screens:
1. `id` (uuid / primary key)
2. `user_id`
3. `name`
4. `photo_url`
5. `event`
6. `location`
7. `date`
8. `notes`
9. `title`
10. `facial_details` (json/jsonb)
11. `created_at`

### 2.2 Hugging Face (Recommended for deployed face-service)

1. Sign up at https://huggingface.co
2. Create a new Space
3. Space type: Docker
4. Point it to the contents of `face-service/`

You can skip Hugging Face for local development and run face-service on localhost.

### 2.3 Expo Account (Optional but recommended)

1. Sign up at https://expo.dev
2. Needed if you want cloud builds and easier device workflows.

## 3. Software To Install (Brand-New Machine)

Install in this order.

### 3.1 Git

Download and install:
https://git-scm.com/download/win

Verify:

```powershell
git --version
```

### 3.2 Node.js LTS (20+)

Download and install:
https://nodejs.org/en/download

Verify:

```powershell
node -v
npm -v
```

### 3.3 Python 3.11

Download and install:
https://www.python.org/downloads/

Important during install:
1. Check "Add python.exe to PATH"
2. Install pip

Verify:

```powershell
python --version
pip --version
```

### 3.4 Visual C++ Build Tools (recommended on Windows for some Python packages)

Install:
https://visualstudio.microsoft.com/visual-cpp-build-tools/

Select:
1. Desktop development with C++
2. Windows 10/11 SDK

### 3.5 Optional: Docker Desktop

Needed only if you want to run face-service via Docker locally.
https://www.docker.com/products/docker-desktop/

### 3.6 Expo Go on Phone

Install Expo Go:
https://expo.dev/client

## 4. Get the Project Code

From a terminal, run:

```powershell
cd C:\Users\<your-user>\OneDrive\Documents\VSCode
git clone <your-repo-url> HCP
cd HCP
```

If you already have the project, just go to root:

```powershell
cd C:\Users\evanl\OneDrive\Documents\VSCode\HCP
```

### START HERE IF YOU ALREADY HAVE (Node, Git, Supabase, HuggingFace, ExpoGo) setup
## 5. Install Dependencies (All Folders)

Run these exactly where shown.

### 5.1 Install client deps

```powershell
cd client
npm install
```

### 5.2 Install server deps

```powershell
cd ..\server
npm install
```

### 5.3 Create Python virtual env and install face-service deps

```powershell
cd ..\face-service
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If activation is blocked by execution policy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 6. Environment Variables (skip if already set)
## 6.1 Client env file (`client/.env`)

Create `client/.env` (copy from `client/.env.example`) and set:

```dotenv
EXPO_PUBLIC_SUPABASE_URL=https://<your-project-ref>.supabase.co
EXPO_PUBLIC_SUPABASE_ANON_KEY=<your-supabase-anon-key>
EXPO_PUBLIC_FACE_ANALYSIS_URL=http://localhost:3000/analyze-face
```

Notes:
1. Client talks to Node server, not directly to Python.
2. If testing on a physical phone, `localhost` is your phone itself.
	- Use your computer LAN IP instead, example:
	- `http://192.168.1.25:3000/analyze-face`


## 6.2 Server env variable (`FACE_SERVICE_URL`)

`server/src/app.js` reads:

1. `FACE_SERVICE_URL` from process env
2. Falls back to `http://localhost:8000`

For a fresh clone, you do not need to set anything. Local development uses the built-in default:

`http://localhost:8000`

Only set this if you want Node to call the deployed Hugging Face Space instead:

```powershell
$env:FACE_SERVICE_URL="https://evanlyhf-rememberme.hf.space"
```

## 7. Run the Project Locally (3 terminals)

Open 3 terminals.

### Terminal A: Python face-service

```powershell
cd C:\Users\evanl\OneDrive\Documents\VSCode\HCP\face-service
.\.venv\Scripts\Activate.ps1
uvicorn app:app --host 0.0.0.0 --port 8000
```

Expected health check:

```powershell
curl http://localhost:8000/health
```

Should return:

```json
{"status":"ok"}
```

### Terminal B: Node server

```powershell
cd C:\Users\evanl\OneDrive\Documents\VSCode\HCP\server
$env:FACE_SERVICE_URL="http://localhost:8000"
npm start
```

Expected test:

```powershell
curl http://localhost:3000/hello
```

### Terminal C: Expo client

```powershell
cd C:\Users\evanl\OneDrive\Documents\VSCode\HCP\client
npm start
```

Then:
1. Scan QR code with Expo Go
2. Login/sign up in app
3. Take a photo in Record screen
4. Confirm `facial_details` is saved in Supabase row

## 8. Deploy face-service to Hugging Face Spaces

Folder to deploy: `face-service/`

Important files already present:
1. `face-service/Dockerfile`
2. `face-service/README.md` (HF Space metadata)

Steps:
1. Create new Hugging Face Space (Docker SDK)
2. Push `face-service/` contents
3. Wait for build to complete
4. Test health endpoint:
	- `https://<your-space>.hf.space/health`
5. Point Node to space URL:
	- `FACE_SERVICE_URL=https://<your-space>.hf.space`

## 9. API Endpoints Summary

### Node server (`server/src/app.js`)

1. `GET /hello`
2. `POST /analyze-face`

Request body:

```json
{ "image": "<base64 image>" }
```

### Python face-service (`face-service/app.py`)

1. `GET /health`
2. `POST /analyze` (multipart file)
3. `POST /analyze-base64` (JSON with base64)

Response shape from Python:

```json
{
  "success": true,
  "data": {
	 "gender": "male",
	 "age_range": "20-29",
	 "primary_emotion": "neutral",
	 "face_shape": "oval"
  }
}
```

## 10. One-Time Verification Checklist

1. `face-service` starts with no import errors
2. `GET http://localhost:8000/health` works
3. `server` starts and prints face service URL
4. `GET http://localhost:3000/hello` works
5. Expo app loads on device/emulator
6. Photo capture works
7. Analyze request returns `success: true`
8. Supabase row inserted into `people`
9. `facial_details` contains analysis JSON

## 11. Common Issues and Fixes

### "Import cv2/fastapi/numpy could not be resolved"

Cause: Python environment not selected or packages not installed.

Fix:
1. Activate `.venv`
2. `pip install -r requirements.txt`
3. In VS Code, select interpreter from `face-service/.venv`.

### Expo app on phone cannot reach localhost

Use your computer LAN IP in `EXPO_PUBLIC_FACE_ANALYSIS_URL`.

### Node cannot reach Python service

Check `FACE_SERVICE_URL` and confirm `/health` works at that URL.

### First analysis request is slow

Expected. Models are lazy-loaded on first call.

## 12. What Is No Longer Required

AWS Rekognition credentials are no longer needed for the new pipeline.
The old Rekognition scripts can remain in repo for reference, but current runtime path is Node -> Python service -> local models.

# HF vs Local testing
You do not need to run the Python face-service locally if the HF Space is running.

If you want the full local setup instead of HF:

In face-service:
create/activate the Python venv
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
In server:
set FACE_SERVICE_URL=http://localhost:8000
npm start
In client:
set EXPO_PUBLIC_FACE_ANALYSIS_URL to your Node server URL
npm start


### 5.1 Install client deps

```powershell
cd client
npm install
```

### 5.2 Install server deps

```powershell
cd ..\server
npm install
```

### 5.3 Create Python virtual env and install face-service deps

```powershell
cd ..\face-service
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If activation is blocked by execution policy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 6. Environment Variables (skip if already set)
## 6.1 Client env file (`client/.env`)

Create `client/.env` (copy from `client/.env.example`) and set:

```dotenv
EXPO_PUBLIC_SUPABASE_URL=https://<your-project-ref>.supabase.co
EXPO_PUBLIC_SUPABASE_ANON_KEY=<your-supabase-anon-key>
EXPO_PUBLIC_FACE_ANALYSIS_URL=http://localhost:3000/analyze-face
```

Notes:
1. Client talks to Node server, not directly to Python.
2. If testing on a physical phone, `localhost` is your phone itself.
	- Use your computer LAN IP instead, example:
	- `http://192.168.1.25:3000/analyze-face`


## 6.2 Server env variable (`FACE_SERVICE_URL`)

`server/src/app.js` reads:

1. `FACE_SERVICE_URL` from process env
2. Falls back to `http://localhost:8000`

For local run in PowerShell (same terminal session):

```powershell
$env:FACE_SERVICE_URL="http://localhost:8000"
```

If using deployed Hugging Face Space:

```powershell
$env:FACE_SERVICE_URL="https://evanlyhf-rememberme.hf.space"
```

## 7. Run the Project Locally (3 terminals)

Open 3 terminals.

### Terminal A: Python face-service
cd C:\Users\evanl\OneDrive\Documents\VSCode\HCP\face-service
.\.venv\Scripts\Activate.ps1
uvicorn app:app --host 0.0.0.0 --port 8000


Expected health check:
curl http://localhost:8000/health

Should return:

```json
{"status":"ok"}
```

### Terminal B: Node server

```powershell
cd C:\Users\evanl\OneDrive\Documents\VSCode\HCP\server
npm start
```

If you want to use the Hugging Face Space for face analysis instead of your local Python service, set `FACE_SERVICE_URL` before running `npm start`:

```powershell
$env:FACE_SERVICE_URL="https://evanlyhf-rememberme.hf.space"
```

Expected test:

```powershell
curl http://localhost:3000/hello
```

### Terminal C: Expo client

cd C:\Users\evanl\OneDrive\Documents\VSCode\HCP\client
npm start


# Updating HF
git clone https://huggingface.co/spaces/evanlyhf/RememberMe
cd RememberMe
Copy-Item -Path "C:\Users\evanl\OneDrive\Documents\VSCode\HCP\face-service\*" -Destination . -Recurse -Force -Exclude ".venv","__pycache__"
# In app.py, ensure the main block runs on the correct port
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)  # HF Spaces uses port 7860
	
git add .
git commit -m "Push face-service to Hugging Face Space"
git push
