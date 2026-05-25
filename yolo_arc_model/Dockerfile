# YOLO+ArcFace Space — build when repo root IS the Space (files in root).
# Upload to HF root: Dockerfile, main.py, requirements.txt, README.md (no enhancer.py needed)
FROM python:3.11-slim-bookworm

WORKDIR /app

ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV PYTHONUNBUFFERED=1
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics
ENV YOLO_MODEL=/app/yolov8n-face.pt
ENV QUALITY_ENHANCE=1

COPY requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libglib2.0-0 \
    libgomp1 \
    libgl1 \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefer-binary -r /app/requirements.txt \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY main.py /app/main.py

RUN curl -fL --retry 5 --retry-delay 3 \
    "https://huggingface.co/deepghs/yolo-face/resolve/main/yolov8n-face/model.pt" \
    -o /app/yolov8n-face.pt

RUN python - <<'PYEOF'
from ultralytics import YOLO
YOLO("/app/yolov8n-face.pt")
from deepface import DeepFace
import numpy as np
DeepFace.represent(
    img_path=np.zeros((112, 112, 3), dtype=np.uint8),
    model_name="ArcFace",
    enforce_detection=False,
    detector_backend="skip",
)
print("YOLO + ArcFace ready.")
PYEOF

ENV PORT=7860
EXPOSE 7860

CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT}"
