FROM python:3.11-slim

# System deps for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 wget \
    libegl1 libgles2 libgomp1 \
    g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download MediaPipe model at build time so first request is fast.
# All other models (FairFace, SegFormer, HSEmotion, ObstructionViT,
# HairTypeViT) are pulled from Hugging Face on first request and cached
# in /root/.cache/huggingface for the rest of the process lifetime.
RUN mkdir -p models && \
    wget -q -O models/face_landmarker.task \
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

COPY . .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
