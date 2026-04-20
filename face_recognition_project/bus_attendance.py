"""
BusVision 360 — face recognition from webcam using a pre-built encodings database.

Requires encodings.pickle (e.g. from generate_encodings.py). Press q to quit.
"""

import os
import pickle

import cv2
import face_recognition
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ENCODINGS_FILE = "encodings.pickle"
CAMERA_INDEX = 0
RESIZE_SCALE = 0.5
MATCH_TOLERANCE = 0.55
UNKNOWN_LABEL = "غير معروف"
WINDOW_TITLE = "BusVision 360 - Student Face Recognition"
FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_known_faces(path: str):
    """Load face encodings and names from pickle. Exit if missing or invalid."""
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        print("Run generate_encodings.py first (with a students/ folder), or use bus_attendance_final.py.")
        raise SystemExit(1)

    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        encodings = data.get("encodings", []) or []
        names = data.get("names", []) or []
        if len(encodings) != len(names) or not encodings:
            raise ValueError("Invalid encodings database (empty or length mismatch).")
        print(f"Loaded {len(names)} student face(s).")
        return encodings, names
    except Exception as exc:
        print(f"Error loading {path}: {exc}")
        raise SystemExit(1) from exc


def recognize_face(known_encodings, known_names, face_encoding):
    """
    Return the best-matching name if distance is below MATCH_TOLERANCE,
    otherwise UNKNOWN_LABEL.
    """
    try:
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_index = int(np.argmin(distances))
        best_distance = float(distances[best_index])
        if best_distance < MATCH_TOLERANCE:
            return known_names[best_index]
    except Exception:
        pass
    return UNKNOWN_LABEL


def scale_box_to_full_frame(top, right, bottom, left, scale_factor):
    """Map face box from resized frame coordinates back to full-resolution frame."""
    inv = 1.0 / scale_factor
    return (
        int(top * inv),
        int(right * inv),
        int(bottom * inv),
        int(left * inv),
    )


def main():
    known_encodings, known_names = load_known_faces(ENCODINGS_FILE)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Could not open camera. Try CAMERA_INDEX 1 or 2.")
        raise SystemExit(1)

    print("Camera started. Press q to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame.")
                break

            small_frame = cv2.resize(
                frame, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE
            )
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small)
            face_encodings = face_recognition.face_encodings(
                rgb_small, face_locations
            )

            for (top, right, bottom, left), encoding in zip(
                face_locations, face_encodings
            ):
                name = recognize_face(known_encodings, known_names, encoding)

                top, right, bottom, left = scale_box_to_full_frame(
                    top, right, bottom, left, RESIZE_SCALE
                )

                color = (0, 255, 0) if name != UNKNOWN_LABEL else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

                text_y = max(20, top - 10)
                cv2.putText(
                    frame, name, (left, text_y), FONT, 0.9, color, 2
                )

            cv2.imshow(WINDOW_TITLE, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera closed.")


if __name__ == "__main__":
    main()
