from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import face_recognition
import numpy as np


@dataclass(frozen=True)
class FaceCoreConfig:
    tolerance: float = 0.48
    second_best_gap: float = 0.06
    detection_model: str = "hog"
    num_jitters: int = 1
    top_k: int = 3


@dataclass(frozen=True)
class FaceCandidate:
    name: str
    distance: float
    confidence: float


@dataclass(frozen=True)
class FaceIdentifyResult:
    status: str
    best: FaceCandidate | None
    top_candidates: list[FaceCandidate]


def load_gallery(encodings_path: str) -> tuple[list[np.ndarray], list[str]]:
    path = Path(encodings_path)
    if not path.exists():
        return [], []
    with path.open("rb") as fh:
        data = pickle.load(fh)
    encodings = [np.asarray(e, dtype=np.float64) for e in data.get("encodings", [])]
    names = [str(n).strip() for n in data.get("names", [])]
    if len(encodings) != len(names):
        return [], []
    return encodings, names


def save_gallery(encodings_path: str, encodings: Iterable[np.ndarray], names: Iterable[str]) -> None:
    payload = {"encodings": [np.asarray(e, dtype=np.float64) for e in encodings], "names": [str(n) for n in names]}
    with Path(encodings_path).open("wb") as fh:
        pickle.dump(payload, fh)


def encode_rgb_face(rgb_image: np.ndarray, config: FaceCoreConfig) -> np.ndarray:
    locations = face_recognition.face_locations(rgb_image, model=config.detection_model)
    if len(locations) != 1:
        raise ValueError("Expected exactly one face in image.")
    encodings = face_recognition.face_encodings(rgb_image, locations, num_jitters=config.num_jitters)
    if len(encodings) != 1:
        raise ValueError("Failed to encode detected face.")
    return np.asarray(encodings[0], dtype=np.float64)


def identify_face(
    face_encoding: np.ndarray,
    known_encodings: list[np.ndarray],
    known_names: list[str],
    config: FaceCoreConfig | None = None,
) -> FaceIdentifyResult:
    cfg = config or FaceCoreConfig()
    if not known_encodings:
        return FaceIdentifyResult(status="unknown", best=None, top_candidates=[])

    distances = face_recognition.face_distance(known_encodings, face_encoding)
    if distances.size == 0:
        return FaceIdentifyResult(status="unknown", best=None, top_candidates=[])

    ranking = np.argsort(distances)
    top = []
    for idx in ranking[: max(1, cfg.top_k)]:
        d = float(distances[idx])
        top.append(
            FaceCandidate(
                name=known_names[int(idx)],
                distance=d,
                confidence=max(0.0, min(1.0, 1.0 - (d / max(cfg.tolerance, 1e-6)))),
            )
        )

    best = top[0] if top else None
    if best is None or best.distance >= cfg.tolerance:
        return FaceIdentifyResult(status="unknown", best=None, top_candidates=top)

    if len(top) >= 2 and (top[1].distance - top[0].distance) < cfg.second_best_gap:
        return FaceIdentifyResult(status="ambiguous", best=best, top_candidates=top)

    return FaceIdentifyResult(status="matched", best=best, top_candidates=top)
