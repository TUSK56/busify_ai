"""
Busify Face API — quality enhance → YOLOv8-face → ArcFace.

Pipeline: image → (optional GFPGAN / quality Space) → YOLO crop → ArcFace embed/match

- GET  /health
- POST /embed  — enrollment: augmented average embedding + quality gate
- POST /match  — scan: single crop, same contract as before
"""
from __future__ import annotations

import base64
import os
import urllib.request
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict

app = FastAPI(title="Busify Face API", version="5.0.0")


def _bgr_to_jpeg_bytes(img: np.ndarray, quality: int = 92) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("jpeg_encode_failed")
    return buf.tobytes()


def _enhance_via_quality_api(img_bgr: np.ndarray) -> np.ndarray | None:
    """Call the separate quality Space (QUALITY_API_URL)."""
    base = os.environ.get("QUALITY_API_URL", "").strip().rstrip("/")
    if not base:
        return None
    try:
        jpeg = _bgr_to_jpeg_bytes(img_bgr)
        boundary = "----busifyquality"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="face.jpg"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n"
        ).encode("utf-8") + jpeg + f"\r\n--{boundary}--\r\n".encode("utf-8")
        req = urllib.request.Request(
            f"{base}/enhance",
            data=body,
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        timeout = int(os.environ.get("QUALITY_API_TIMEOUT", "90"))
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out = resp.read()
        arr = np.frombuffer(out, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as exc:
        print(f"Quality API ({base}) failed: {exc}", flush=True)
        return None


def prepare_image_for_face_pipeline(img_bgr: np.ndarray) -> np.ndarray:
    """Step 1: POST to quality Space, else use original image."""
    if os.environ.get("QUALITY_ENHANCE", "1").strip().lower() in ("0", "false", "no"):
        return img_bgr
    enhanced = _enhance_via_quality_api(img_bgr)
    return enhanced if enhanced is not None else img_bgr

_yolo_model = None
_arcface_model_name = os.environ.get("ARCFACE_MODEL_NAME", "ArcFace")

_FACE_YOLO_DOWNLOAD_URL = (
    "https://huggingface.co/deepghs/yolo-face/resolve/main/yolov8n-face/model.pt"
)
_DEFAULT_YOLO_LOCAL = "/app/yolov8n-face.pt"


def _enhance_image(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    mean_brightness = np.mean(l_enhanced)
    if mean_brightness < 80:
        gamma = 0.6
    elif mean_brightness > 180:
        gamma = 1.4
    else:
        gamma = 1.0
    if gamma != 1.0:
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
            dtype=np.uint8,
        )
        l_enhanced = cv2.LUT(l_enhanced, table)
    enhanced_lab = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return cv2.bilateralFilter(enhanced, d=7, sigmaColor=50, sigmaSpace=50)


def _enhance_face_crop(face_bgr: np.ndarray) -> np.ndarray:
    if os.environ.get("FACE_CROP_ENHANCE", "1").strip().lower() in ("0", "false", "no"):
        return face_bgr
    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    lab = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _ensure_yolo_weights_path() -> str:
    configured = os.environ.get("YOLO_MODEL", "").strip()
    if configured.startswith("http://") or configured.startswith("https://"):
        configured = ""
    local_path = configured if configured else _DEFAULT_YOLO_LOCAL
    if os.path.isfile(local_path):
        return local_path
    parent = os.path.dirname(local_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    import urllib.request

    print(f"Downloading yolov8n-face weights to {local_path} ...", flush=True)
    urllib.request.urlretrieve(_FACE_YOLO_DOWNLOAD_URL, local_path)
    return local_path


def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise HTTPException(status_code=500, detail="ultralytics not installed") from exc
        _yolo_model = YOLO(_ensure_yolo_weights_path())
    return _yolo_model


def _bgr_from_bytes(raw: bytes) -> np.ndarray | None:
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _largest_face_box(
    img: np.ndarray, *, min_confidence: float | None = None
) -> tuple[tuple[int, int, int, int], float] | None:
    """Return padded box and YOLO confidence for the largest face."""
    min_conf = (
        float(min_confidence)
        if min_confidence is not None
        else float(os.environ.get("FACE_MIN_DET_SCORE", "0.50"))
    )
    results = _get_yolo()(img, imgsz=640, verbose=False)
    best: tuple[float, float, int, int, int, int] | None = None
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue
        for box in boxes:
            conf = float(box.conf[0])
            if conf < min_conf:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            area = float((x2 - x1) * (y2 - y1))
            if best is None or area > best[0]:
                best = (area, conf, x1, y1, x2, y2)
    if best is None:
        return None
    _, det_conf, x1, y1, x2, y2 = best
    h, w = img.shape[:2]
    pad_x = int((x2 - x1) * 0.10)
    pad_y = int((y2 - y1) * 0.10)
    box = (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(w, x2 + pad_x),
        min(h, y2 + pad_y),
    )
    return box, det_conf


def light_normalize_crop(face_bgr: np.ndarray) -> np.ndarray:
    """Light brightness correction after GFPGAN/YOLO — only extreme cases."""
    mean_brightness = float(np.mean(face_bgr))
    if mean_brightness > 200 or mean_brightness < 50:
        gamma = 0.7 if mean_brightness > 200 else 1.3
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
            dtype=np.uint8,
        )
        return cv2.LUT(face_bgr, table)
    return face_bgr


def _extract_face_crop(
    img: np.ndarray, *, min_yolo_confidence: float | None = None
) -> tuple[np.ndarray, float]:
    """GFPGAN (unchanged) → YOLO crop. Returns (crop, yolo_confidence)."""
    prepared = prepare_image_for_face_pipeline(img)
    detect_view = _enhance_image(prepared)
    found = _largest_face_box(detect_view, min_confidence=min_yolo_confidence)
    if found is None:
        found = _largest_face_box(prepared, min_confidence=min_yolo_confidence)
    if found is None:
        found = _largest_face_box(img, min_confidence=min_yolo_confidence)
    if found is None:
        raise ValueError("no_face_detected")
    box, det_conf = found
    x1, y1, x2, y2 = box
    return prepared[y1:y2, x1:x2].copy(), det_conf


def _enrollment_quality_gate(face_crop: np.ndarray, yolo_confidence: float) -> None:
    """Stricter gates for parent enrollment only (after GFPGAN + YOLO)."""
    h, w = face_crop.shape[:2]
    if min(h, w) < 80:
        raise HTTPException(
            status_code=400,
            detail="Face crop too small. Please scan closer to the camera.",
        )
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_var < 40.0:
        raise HTTPException(
            status_code=400,
            detail=(
                "Image is still unclear after enhancement. "
                "Please scan in better lighting."
            ),
        )
    if yolo_confidence < 0.70:
        raise HTTPException(
            status_code=400,
            detail=(
                "Face not clearly detected. Please face the camera directly "
                "in good lighting."
            ),
        )


def _arcface_embedding_from_crop(
    face_bgr: np.ndarray, *, apply_crop_clahe: bool = True
) -> np.ndarray:
    try:
        from deepface import DeepFace
    except ImportError as exc:
        raise HTTPException(status_code=500, detail="deepface not installed") from exc
    if face_bgr.size == 0:
        raise ValueError("no_face_detected")
    if apply_crop_clahe:
        face_bgr = _enhance_face_crop(face_bgr)
    rows = DeepFace.represent(
        img_path=face_bgr,
        model_name=_arcface_model_name,
        enforce_detection=False,
        detector_backend="skip",
        align=True,
    )
    if not rows:
        raise ValueError("no_face_detected")
    emb = np.asarray(rows[0]["embedding"], dtype=np.float32)
    norm = float(np.linalg.norm(emb))
    if norm < 1e-6:
        raise ValueError("no_face_detected")
    return emb / norm


def extract_single_embedding(face_bgr: np.ndarray) -> np.ndarray:
    """ArcFace on one crop (enrollment augment path; no extra CLAHE)."""
    normalized = light_normalize_crop(face_bgr)
    return _arcface_embedding_from_crop(normalized, apply_crop_clahe=False)


def averaged_enrollment_embedding(face_bgr: np.ndarray, n_augments: int = 5) -> np.ndarray:
    """Five scale variations on the GFPGAN+YOLO crop → mean → L2 normalize."""
    rng = np.random.default_rng(42)
    h, w = face_bgr.shape[:2]
    embeddings: list[np.ndarray] = []
    for _ in range(n_augments):
        scale = float(rng.uniform(0.92, 1.08))
        nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
        scaled = cv2.resize(face_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        crop = cv2.resize(scaled, (w, h), interpolation=cv2.INTER_LINEAR)
        embeddings.append(extract_single_embedding(crop))
    mean_vec = np.mean(np.stack(embeddings, axis=0), axis=0)
    norm = float(np.linalg.norm(mean_vec))
    if norm < 1e-6:
        raise ValueError("no_face_detected")
    return (mean_vec / norm).astype(np.float32)


def _embedding_enrollment(img: np.ndarray) -> np.ndarray:
    face, yolo_conf = _extract_face_crop(img, min_yolo_confidence=0.70)
    _enrollment_quality_gate(face, yolo_conf)
    face = light_normalize_crop(face)
    return averaged_enrollment_embedding(face)


def _embedding_match(img: np.ndarray) -> np.ndarray:
    """Scan/match: GFPGAN → YOLO → light normalize → single ArcFace embedding."""
    face, _ = _extract_face_crop(img)
    face = light_normalize_crop(face)
    return _arcface_embedding_from_crop(face, apply_crop_clahe=False)


def _accept_similarity(candidate_count: int) -> float:
    if candidate_count == 1:
        return float(os.environ.get("FACE_MIN_COSINE_SIM_SINGLE", "0.75"))
    return float(os.environ.get("FACE_MIN_COSINE_SIM", "0.65"))


def _margin_is_ok(
    scored: list[tuple[int, float]],
    min_sim: float,
    min_margin: float,
    strong_sim: float,
) -> bool:
    if len(scored) < 2:
        return True
    best_sim = scored[0][1]
    runner_up_sim = scored[1][1]
    if runner_up_sim < min_sim:
        return True
    required_margin = min_margin * (0.5 if best_sim >= strong_sim else 1.0)
    return (best_sim - runner_up_sim) >= required_margin


@app.get("/health")
def health() -> dict[str, str]:
    quality_url = os.environ.get("QUALITY_API_URL", "").strip()
    return {
        "status": "ok",
        "provider": "quality-yolov8-face-arcface-v5",
        "qualityApi": quality_url or "local-or-disabled",
    }


@app.post("/embed")
async def embed(file: UploadFile = File(...)) -> dict[str, Any]:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")
    img = _bgr_from_bytes(raw)
    if img is None:
        raise HTTPException(status_code=400, detail="invalid_image")
    try:
        vec = _embedding_enrollment(img)
    except HTTPException:
        raise
    except ValueError as exc:
        msg = str(exc)
        if msg == "no_face_detected":
            return {
                "dimensions": 0,
                "embedding": [],
                "status": "no_face_detected",
            }
        raise HTTPException(status_code=422, detail=msg) from exc
    return {
        "dimensions": int(vec.shape[0]),
        "embedding": vec.astype(float).tolist(),
        "status": "ok",
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
        img_bytes = base64.b64decode(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid_base64: {exc}") from exc

    img = _bgr_from_bytes(img_bytes)
    if img is None:
        raise HTTPException(status_code=400, detail="invalid_image")

    try:
        probe = _embedding_match(img).astype(np.float64)
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

    min_dim = int(probe.shape[0])
    scored: list[tuple[int, float]] = []
    for c in body.candidates:
        if len(c.embedding) < min_dim:
            continue
        v = np.asarray(c.embedding[:min_dim], dtype=np.float64)
        n = np.linalg.norm(v)
        if n < 1e-6:
            continue
        v = v / n
        scored.append((c.studentId, float(np.dot(probe, v))))

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
    min_sim = float(os.environ.get("FACE_MIN_COSINE_SIM", "0.65"))
    min_margin = float(os.environ.get("FACE_MIN_TOP1_MARGIN", "0.04"))
    strong_sim = float(os.environ.get("FACE_STRONG_COSINE_SIM", "0.72"))
    accept_sim = _accept_similarity(len(scored))
    best_id, best_sim = scored[0]
    margin_ok = _margin_is_ok(scored, min_sim, min_margin, strong_sim)

    def _top_payload() -> list[dict[str, Any]]:
        return [
            {
                "studentId": sid,
                "distance": round(1.0 - s, 6),
                "confidence": round(s, 6),
            }
            for sid, s in scored[:5]
        ]

    if best_sim < accept_sim:
        return {
            "matchFound": False,
            "matchedStudentId": None,
            "confidence": round(best_sim, 6),
            "distance": round(1.0 - best_sim, 6),
            "status": "not_in_gallery",
            "topCandidates": _top_payload(),
        }

    if not margin_ok:
        return {
            "matchFound": False,
            "matchedStudentId": None,
            "confidence": round(best_sim, 6),
            "distance": round(1.0 - best_sim, 6),
            "status": "ambiguous",
            "topCandidates": _top_payload(),
        }

    return {
        "matchFound": True,
        "matchedStudentId": best_id,
        "confidence": round(best_sim, 6),
        "distance": round(1.0 - best_sim, 6),
        "status": "matched",
        "topCandidates": _top_payload(),
    }
