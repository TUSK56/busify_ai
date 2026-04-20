"""
Busify face embedding API for Render (or any container host).
POST /embed — multipart file "file" (JPEG/PNG), returns 128-D embedding JSON.
"""
from __future__ import annotations

import io
from typing import Any

import face_recognition
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

app = FastAPI(title="Busify Face Embed", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/embed")
async def embed(file: UploadFile = File(...)) -> dict[str, Any]:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")

    try:
        img = face_recognition.load_image_file(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid image: {exc}") from exc

    locs = face_recognition.face_locations(img, model="hog")
    if not locs:
        raise HTTPException(status_code=422, detail="no_face_detected")

    encs = face_recognition.face_encodings(img, locs, num_jitters=1)
    if not encs:
        raise HTTPException(status_code=422, detail="encoding_failed")

    vec = np.asarray(encs[0], dtype=np.float64)
    return {
        "dimensions": int(vec.shape[0]),
        "embedding": vec.tolist(),
    }
