# GFPGAN quality Space — build when repo root IS the Space (all files in root).
# Upload to HF: Dockerfile, main.py, enhancer.py, requirements.txt, GFPGANv1.4.pth, README.md
FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV GFPGAN_MODEL_PATH=/app/GFPGANv1.4.pth
ENV GFPGAN_UPSCALE=1
ENV GFPGAN_ENABLE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefer-binary -r /app/requirements.txt

COPY enhancer.py main.py /app/
COPY GFPGANv1.4.pth /app/GFPGANv1.4.pth

ENV PORT=7860
EXPOSE 7860

CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT}"
