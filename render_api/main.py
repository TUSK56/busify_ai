"""
Busify FaceMatch-style API (InsightFace + ONNX, no dlib).
Compatible with Hugging Face Spaces / Render Docker.

- GET  /health
- POST /embed  — multipart field "file" (JPEG/PNG) → { dimensions, embedding }
- POST /match  — JSON { imageBase64, candidates: [{ studentId, embedding: [float] }] }
                 → { matchFound, matchedStudentId, confidence, distance, status, topCandidates }
Inspired by the same architecture as https://huggingface.co/blackmamba2408/FaceMatch (512-D, cosine).
"""
from __future__ import annotations

import base64
import os
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict

app = FastAPI(title="Busify FaceMatch API", version="2.0.0")

_face_app = None


def _get_face_app():
    global _face_app
    if _face_app is None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise HTTPException(
                status_code=500,
                detail="insightface not installed",
            ) from exc
        name = os.environ.get("INSIGHTFACE_MODEL_NAME", "buffalo_l")
        providers = ["CPUExecutionProvider"]
        _face_app = FaceAnalysis(name=name, providers=providers)
        _face_app.prepare(ctx_id=0, det_size=(640, 640))
    return _face_app


def _bgr_from_bytes(raw: bytes) -> np.ndarray | None:
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _embedding_from_bgr(img: np.ndarray) -> np.ndarray:
    faces = _get_face_app().get(img)
    if not faces:
        raise ValueError("no_face_detected")
    emb = np.asarray(faces[0].normed_embedding, dtype=np.float32)
    return emb


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "provider": "insightface-onnx"}


@app.post("/embed")
async def embed(file: UploadFile = File(...)) -> dict[str, Any]:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")
    img = _bgr_from_bytes(raw)
    if img is None:
        raise HTTPException(status_code=400, detail="invalid_image")
    try:
        vec = _embedding_from_bgr(img)
    except ValueError as exc:
        if str(exc) == "no_face_detected":
            return {
                "dimensions": 0,
                "embedding": [],
                "status": "no_face_detected",
            }
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "dimensions": int(vec.shape[0]),
        "embedding": vec.astype(float).tolist(),
    }


class CandidateIn(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    studentId: int
    embedding: list[float]


class MatchBody(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    imageBase64: str
    candidates: list[CandidateIn]


@app.post("/match")
async def match(body: MatchBody) -> dict[str, Any]:
    raw = body.imageBase64.strip()
    if "base64," in raw:
        raw = raw.split("base64,", 1)[1]
    try:
        img_bytes = base64.b64decode(raw, validate=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid_base64: {exc}") from exc

    img = _bgr_from_bytes(img_bytes)
    if img is None:
        raise HTTPException(status_code=400, detail="invalid_image")

    try:
        probe = _embedding_from_bgr(img).astype(np.float64)
    except ValueError as exc:
        if str(exc) == "no_face_detected":
            return {
                "matchFound": False,
                "matchedStudentId": None,
                "confidence": 0.0,
                "distance": None,
                "status": "no_face_detected",
                "topCandidates": [],
            }
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    min_dim = 512
    scored: list[tuple[int, float]] = []
    for c in body.candidates:
        if len(c.embedding) < min_dim:
            continue
        v = np.asarray(c.embedding[:min_dim], dtype=np.float64)
        n = np.linalg.norm(v)
        if n < 1e-6:
            continue
        v = v / n
        sim = float(np.dot(probe, v))
        scored.append((c.studentId, sim))

    if not scored:
        return {
            "matchFound": False,
            "matchedStudentId": None,
            "confidence": 0.0,
            "distance": None,
            "status": "unknown",
            "topCandidates": [],
        }

    scored.sort(key=lambda x: x[1], reverse=True)
    # Cosine similarity on L2-normalized 512-D embeddings (InsightFace). Default was 0.32
    # and was too permissive; wrong faces could still exceed it. Override with FACE_MIN_COSINE_SIM.
    min_sim = float(os.environ.get("FACE_MIN_COSINE_SIM", "0.45"))
    best_id, best_sim = scored[0]
    if best_sim < min_sim:
        top = [
            {
                "studentId": sid,
                "distance": round(1.0 - s, 6),
                "confidence": round(s, 6),
            }
            for sid, s in scored[:5]
        ]
        return {
            "matchFound": False,
            "matchedStudentId": None,
            "confidence": round(best_sim, 6),
            "distance": round(1.0 - best_sim, 6),
            "status": "unknown",
            "topCandidates": top,
        }

    top = [
        {
            "studentId": sid,
            "distance": round(1.0 - s, 6),
            "confidence": round(s, 6),
        }
        for sid, s in scored[:5]
    ]
    return {
        "matchFound": True,
        "matchedStudentId": best_id,
        "confidence": round(best_sim, 6),
        "distance": round(1.0 - best_sim, 6),
        "status": "matched",
        "topCandidates": top,
    }
