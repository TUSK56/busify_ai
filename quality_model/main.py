"""
Busify Quality API — GFPGAN face restoration (step 1 of pipeline).

- GET  /health
- POST /enhance  — JPEG in (multipart file), enhanced JPEG out
"""
from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, Response, UploadFile

from enhancer import _get_gfpgan, _weights_available, enhance_jpeg_bytes

app = FastAPI(title="Busify Quality API", version="1.0.0")


@app.on_event("startup")
def _warmup() -> None:
    if _weights_available():
        _get_gfpgan()


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "provider": "gfpgan",
        "weights": "ready" if _weights_available() else "missing",
    }


@app.post("/enhance")
async def enhance(file: UploadFile = File(...)) -> Response:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")
    try:
        out = enhance_jpeg_bytes(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(content=out, media_type="image/jpeg")
