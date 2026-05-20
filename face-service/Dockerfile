FROM python:3.11-slim

# System deps for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 wget unzip \
    libegl1 libgles2 libgomp1 \
    g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download MediaPipe model at build time so first request is fast.
# Hugging Face models (Ethnicity ViT, SegFormer, HSEmotion, ObstructionViT,
# HairTypeViT) and the InsightFace buffalo_l bundle are pulled lazily on
# first request and cached in /root/.cache for the lifetime of the
# container.
RUN mkdir -p models && \
    wget -q -O models/face_landmarker.task \
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

# Pre-download InsightFace buffalo_l bundle (detection + recognition +
# age + gender + landmarks) so the first /analyze call doesn't pay the
# ~280MB download. The bundle auto-extracts under ~/.insightface/models/
# on first use.
RUN mkdir -p /root/.insightface/models && \
    wget -q -O /root/.insightface/models/buffalo_l.zip \
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip" && \
    cd /root/.insightface/models && unzip -q buffalo_l.zip -d buffalo_l && rm buffalo_l.zip

# unzip wasn't in the system deps; add it via the apt block at the top.

COPY . .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
