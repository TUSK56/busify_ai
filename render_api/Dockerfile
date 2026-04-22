# InsightFace / ONNX — no dlib. Suitable for Hugging Face Spaces & Render.
# insightface builds native extensions (mesh_core_cython); build-essential is required for pip,
# then removed in the same RUN layer to keep the image smaller than leaving gcc installed.
FROM python:3.11-slim-bookworm

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgomp1 \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefer-binary -r requirements.txt \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY main.py .

# HF Spaces default; Render can override.
ENV PORT=7860
EXPOSE 7860

CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT}"
