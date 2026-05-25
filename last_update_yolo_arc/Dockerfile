# YOLOv8-face + ArcFace — Hugging Face Spaces / Render Docker.
FROM python:3.11-slim-bookworm

WORKDIR /app

ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV PYTHONUNBUFFERED=1
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics
ENV YOLO_MODEL=/app/yolov8n-face.pt

COPY yolo_arc_model/requirements.txt ./requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libglib2.0-0 \
    libgomp1 \
    libgl1 \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefer-binary -r requirements.txt \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY main.py .
COPY quality_model /app/quality_model

# Download face YOLO weights (GitHub akanametov URL returns 404).
RUN curl -fL --retry 5 --retry-delay 3 \
    "https://huggingface.co/deepghs/yolo-face/resolve/main/yolov8n-face/model.pt" \
    -o /app/yolov8n-face.pt

# Pre-warm YOLO + ArcFace at build time
RUN python - <<'PYEOF'
from ultralytics import YOLO
print("Loading yolov8n-face.pt ...")
YOLO("/app/yolov8n-face.pt")
print("YOLO face model ready.")

from deepface import DeepFace
import numpy as np
print("Downloading ArcFace weights ...")
dummy = np.zeros((112, 112, 3), dtype=np.uint8)
DeepFace.represent(
    img_path=dummy,
    model_name="ArcFace",
    enforce_detection=False,
    detector_backend="skip",
)
print("ArcFace model ready.")
PYEOF

ENV PORT=7860
EXPOSE 7860

CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT}"
