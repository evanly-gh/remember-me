# RememberMe

RememberMe is a mobile contact manager with photo capture, face analysis, Supabase-backed storage, and a Python face-analysis service.

## For Regular Users

### What You Can Do

- Sign up and log in with the app.
- Add a person with a photo, name, notes, location, date, and optional relationship details.
- Capture a photo from the Add screen and automatically attach facial analysis data to the saved record.
- Browse saved contacts in the Contacts tab.
- Search contacts by name or by facial traits such as glasses, beard, emotion, hair color, face shape, and age range.
- Open a contact and update the profile later from the edit screen.
- Switch app settings from the Settings tab.

### How To Use It

1. Open the app and sign in.
2. Go to the Add tab.
3. Enter the person's name and any extra details.
4. Tap the photo area to take a picture.
5. Save the record.
6. Use the Contacts tab to search, review, and edit saved profiles.

## For Developers

### Project Layout

- `client/` - Expo React Native app
- `server/` - Node.js Express API that forwards face-analysis requests
- `face-service/` - Python FastAPI face-analysis microservice
- Supabase - authentication, database, and storage

### Dependencies

Install these before running the project:

- Git
- Node.js 20 or newer
- Python 3.11
- Expo Go on a phone, or simulator support if you want to test the mobile app
- Optional: Docker Desktop if you want to run the face service in a container locally

You also need accounts and project setup for:

- Supabase
- Hugging Face, if you want to test or deploy the face service as a Space

The main package dependencies are listed in:

- `client/package.json`
- `server/package.json`
- `face-service/requirements.txt`

### Environment Variables

Create `client/.env` with:

```dotenv
EXPO_PUBLIC_SUPABASE_URL=https://<your-project-ref>.supabase.co
EXPO_PUBLIC_SUPABASE_ANON_KEY=<your-supabase-anon-key>
EXPO_PUBLIC_FACE_ANALYSIS_URL=http://<your-node-server>:3000/analyze-face
```

Notes:

- The client talks to the Node server, not directly to the Python service.
- If you test on a physical phone, `localhost` points to the phone, so use your computer's LAN IP instead.

Set this environment variable for the Node server:

```powershell
$env:FACE_SERVICE_URL="http://localhost:8000"
```

If you point the server at a Hugging Face Space, set `FACE_SERVICE_URL` to that Space URL instead.

The Python face service does not require a separate environment file for local development.

### Install Dependencies

From the repository root:

```powershell
cd client
npm install

cd ..\server
npm install

cd ..\face-service
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual environment activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Run Locally

Use three terminals for the full local stack.

Terminal 1 - Python face service:

```powershell
cd C:\Users\evanl\Documents\VSCode\HCP\face-service
.\.venv\Scripts\Activate.ps1
uvicorn app:app --host 0.0.0.0 --port 8000
```

Check it with:

```powershell
curl http://localhost:8000/health
```

Terminal 2 - Node server:

```powershell
cd C:\Users\evanl\Documents\VSCode\HCP\server
$env:FACE_SERVICE_URL="http://localhost:8000"
npm start
```

Check it with:

```powershell
curl http://localhost:3000/hello
```

Terminal 3 - Expo client:

```powershell
cd C:\Users\evanl\Documents\VSCode\HCP\client
npm start
```

### Test With a Hugging Face Space

Use this path if you want to avoid running the Python service locally.

1. Create a Hugging Face Space using the Docker option.
2. Deploy the contents of `face-service/`.
3. Wait for the Space to finish building.
4. Confirm the health endpoint responds at `https://<your-space>.hf.space/health`.
5. Point the Node server to the Space:

```powershell
$env:FACE_SERVICE_URL="https://<your-space>.hf.space"
```

6. Start the Node server with `npm start`.
7. Start the Expo client with `npm start`.

### Useful Endpoints

- `GET /hello` on the Node server
- `POST /analyze-face` on the Node server
- `GET /health` on the Python face service
- `POST /analyze` on the Python face service
- `POST /analyze-base64` on the Python face service

### Data Requirements

Supabase needs a `people` table and a `photos` storage bucket. The app expects fields such as:

- `id`
- `user_id`
- `name`
- `photo_url`
- `event`
- `location`
- `date`
- `notes`
- `title`
- `facial_details`
- `created_at`

### Runtime Flow

1. The app captures an image and saves contact data to Supabase.
2. The client sends the image to the Node server at `/analyze-face`.
3. The Node server forwards the request to the Python service at `/analyze-base64`.
4. The Python service returns facial analysis JSON.
5. The client stores that analysis in `facial_details` for the saved record.

### Notes

- The face-analysis models are lazy-loaded, so the first request can be slow.
- AWS Rekognition is no longer part of the current runtime path.