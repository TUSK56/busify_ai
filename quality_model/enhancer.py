"""
GFPGAN face restoration — used by quality_model Space and optionally in-process from yolo_arc.
"""
from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

_gfpgan_restorer: Any = None


def _default_weights_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    default = os.path.join(here, "GFPGANv1.4.pth")
    if not os.path.isfile(default):
        default = "/app/GFPGANv1.4.pth"
    return os.environ.get("GFPGAN_MODEL_PATH", default)


def _weights_available() -> bool:
    return os.path.isfile(_default_weights_path())


def _get_gfpgan():
    global _gfpgan_restorer
    if _gfpgan_restorer is not None:
        return _gfpgan_restorer
    if not _weights_available():
        return None
    try:
        from gfpgan import GFPGANer
    except ImportError as exc:
        print(f"GFPGAN import failed: {exc}", flush=True)
        return None
    path = _default_weights_path()
    upscale = int(os.environ.get("GFPGAN_UPSCALE", "1"))
    print(f"Loading GFPGAN from {path} (upscale={upscale}) ...", flush=True)
    _gfpgan_restorer = GFPGANer(
        model_path=path,
        upscale=upscale,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )
    return _gfpgan_restorer


def bgr_from_bytes(raw: bytes) -> np.ndarray | None:
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def bgr_to_jpeg_bytes(img: np.ndarray, quality: int = 92) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("jpeg_encode_failed")
    return buf.tobytes()


def enhance_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """Run GFPGAN on full frame; returns input unchanged if model unavailable."""
    if os.environ.get("GFPGAN_ENABLE", "1").strip().lower() in ("0", "false", "no"):
        return img_bgr
    restorer = _get_gfpgan()
    if restorer is None:
        return img_bgr
    try:
        _, _, restored = restorer.enhance(
            img_bgr,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
        )
        if restored is None:
            return img_bgr
        return restored
    except Exception as exc:
        print(f"GFPGAN enhance failed: {exc}", flush=True)
        return img_bgr


def enhance_jpeg_bytes(raw: bytes) -> bytes:
    img = bgr_from_bytes(raw)
    if img is None:
        raise ValueError("invalid_image")
    return bgr_to_jpeg_bytes(enhance_bgr(img))


def enhance_via_quality_api(img_bgr: np.ndarray) -> np.ndarray | None:
    """POST image to external quality Space (QUALITY_API_URL). Returns None on failure."""
    base = os.environ.get("QUALITY_API_URL", "").strip().rstrip("/")
    if not base:
        return None
    try:
        import urllib.request

        jpeg = bgr_to_jpeg_bytes(img_bgr)
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
        return bgr_from_bytes(out)
    except Exception as exc:
        print(f"Quality API ({base}) failed: {exc}", flush=True)
        return None


def prepare_image_for_face_pipeline(img_bgr: np.ndarray) -> np.ndarray:
    """
    Pipeline step 1: remote quality Space -> else local GFPGAN -> else unchanged.
    Used by yolo_arc before YOLO + ArcFace.
    """
    if os.environ.get("QUALITY_ENHANCE", "1").strip().lower() in ("0", "false", "no"):
        return img_bgr
    remote = enhance_via_quality_api(img_bgr)
    if remote is not None:
        return remote
    return enhance_bgr(img_bgr)
