import csv
import logging
import os
import pickle
import queue
import threading
from datetime import datetime
from typing import Optional, Sequence, Union

import cv2
import face_recognition
import numpy as np
from face_core import FaceCoreConfig, identify_face

# =============================================================================
# Configuration
# =============================================================================
FONT = cv2.FONT_HERSHEY_DUPLEX
ENCODINGS_FILE = "encodings.pickle"
CSV_FILE = "attendance_bus.csv"
WINDOW_TITLE = "BusVision 360"

# Processing resolution vs speed (higher factor = larger internal frame)
RESIZE_FACTOR = 0.75
TOLERANCE = 0.48
SECOND_BEST_GAP = 0.06
UNKNOWN_THRESHOLD = 12
FRAME_SKIP = 2
UPSAMPLE_TIMES = 1

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

CAMERA_INDEX = 0

# If > 0, reload encodings.pickle from disk every N camera frames (picks up
# enroll_new_student() from another process without restarting this script). 0 = never.
RELOAD_GALLERY_EVERY_FRAMES = 0

INPUT_QUEUE: queue.Queue[str] = queue.Queue()
UNKNOWN_LABEL = "Unknown"

# Real-time loop: persistent unknown handling
NOTIFY_ON_PERSISTENT_UNKNOWN = True

# If True: stdin-based naming (legacy). When False, use photo / encoding enrollment helpers.
USE_LEGACY_TERMINAL_ENROLLMENT = False

# Enrollment: primary image + up to this many extra photos (3–5 total typical).
ENROLL_MAX_ADDITIONAL_IMAGES = 4

# dlib jitter passes for enrollment encodings (1 = fast; 2–5 can help hard photos, slower).
ENROLL_NUM_JITTERS = 1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bus_attendance")


# =============================================================================
# Helpers — storage, CSV, terminal queue
# =============================================================================


def load_encodings(path: str, *, verbose: bool = True) -> tuple[list, list]:
    """
    Load face encodings and names from a pickle file.
    Returns empty lists if the file is missing, corrupt, or inconsistent.
    """
    if not os.path.exists(path):
        if verbose:
            print("No existing database found. A new one will be created after first enrollment.")
        return [], []

    try:
        with open(path, "rb") as file_handle:
            data = pickle.load(file_handle)

        encodings = data.get("encodings", []) or []
        names = data.get("names", []) or []

        if len(encodings) != len(names):
            if verbose:
                print("Database mismatch (encodings != names). Starting with empty database.")
            return [], []

        valid_encodings = []
        valid_names = []
        for encoding, raw_name in zip(encodings, names):
            array = np.asarray(encoding, dtype=np.float64)
            if array.shape == (128,) and isinstance(raw_name, str) and raw_name.strip():
                valid_encodings.append(array)
                valid_names.append(raw_name.strip())

        if verbose:
            print(f"Loaded {len(valid_names)} students")
        return valid_encodings, valid_names
    except Exception as exc:
        if verbose:
            print(f"Loading error: {exc}")
        logger.error("load_encodings failed for %s: %s", path, exc)
        return [], []


def save_encodings(path: str, encodings: list, names: list) -> None:
    """Atomically persist encodings and names to disk."""
    if len(encodings) != len(names):
        logger.warning("save_encodings skipped: length mismatch")
        print("Save skipped: encodings and names length mismatch.")
        return

    temp_path = path + ".tmp"
    try:
        with open(temp_path, "wb") as file_handle:
            pickle.dump({"encodings": encodings, "names": names}, file_handle)
        os.replace(temp_path, path)
    except Exception as exc:
        logger.error("save_encodings failed: %s", exc)
        print(f"Save error: {exc}")
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass


def ensure_csv(path: str) -> None:
    """Create the attendance CSV with a header row if it does not exist."""
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, "w", newline="", encoding="utf-8-sig") as file_handle:
                csv.writer(file_handle).writerow(["Name", "Date", "Time", "Status"])
    except Exception as exc:
        logger.error("ensure_csv failed: %s", exc)
        print(f"CSV init error: {exc}")


def append_attendance(path: str, name: str, status: str) -> None:
    """Append one attendance row to the CSV file."""
    now = datetime.now()
    row = [name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), status]
    try:
        with open(path, "a", newline="", encoding="utf-8-sig") as file_handle:
            csv.writer(file_handle).writerow(row)
        print(f"Recorded {name} at {row[2]} ({status})")
    except Exception as exc:
        logger.error("append_attendance failed: %s", exc)
        print(f"Attendance write error: {exc}")


def terminal_input_loop() -> None:
    """Background thread: read lines from stdin and push them to INPUT_QUEUE."""
    while True:
        try:
            line = input().strip()
            INPUT_QUEUE.put(line)
        except EOFError:
            break
        except Exception:
            continue


# =============================================================================
# Enrollment — shared finalize, photo-based registration, live encoding hook
# =============================================================================


def _finalize_enrollment(
    combined_encoding: np.ndarray,
    student_name: str,
    *,
    encodings_path: str = ENCODINGS_FILE,
    csv_path: str = CSV_FILE,
    success_detail: str = "",
) -> bool:
    """
    Single load of the gallery: duplicate check, append one 128-D vector, save, CSV row.

    Call only after you have a valid averaged or live-captured encoding.
    """
    clean_name = (student_name or "").strip()
    if not clean_name:
        msg = "Enrollment failed: student name is empty."
        print(msg)
        logger.error(msg)
        return False

    vec = np.asarray(combined_encoding, dtype=np.float64).reshape(-1)
    if vec.shape != (128,):
        msg = f"Enrollment failed: encoding must have shape (128,), got {vec.shape}."
        print(msg)
        logger.error(msg)
        return False

    try:
        known_encodings, known_names = load_encodings(encodings_path, verbose=False)
        if clean_name in known_names:
            msg = f"Enrollment failed: student '{clean_name}' already exists in the database."
            print(msg)
            logger.warning(msg)
            return False

        known_encodings.append(vec.copy())
        known_names.append(clean_name)
        save_encodings(encodings_path, known_encodings, known_names)
        ensure_csv(csv_path)
        append_attendance(csv_path, clean_name, "Enrolled - Present")

        suffix = f" {success_detail}" if success_detail else ""
        ok_msg = f"Enrollment succeeded: '{clean_name}'.{suffix}"
        print(ok_msg.strip())
        logger.info(ok_msg.strip())
        return True
    except Exception as exc:
        msg = f"Enrollment failed for '{clean_name}': {exc}"
        print(msg)
        logger.exception(msg)
        return False


def _numpy_array_to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """
    Convert a numpy image to contiguous RGB uint8 for face_recognition.

    Supports:
    - H×W grayscale
    - H×W×3 BGR (OpenCV) or H×W×3 float/RGB-like (scaled to 0–255)
    - H×W×4 BGRA
    """
    if image is None:
        raise ValueError("Image array is None.")
    arr = np.asarray(image)
    if arr.size == 0:
        raise ValueError("Image array is empty.")
    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D array; got shape {arr.shape}.")

    if arr.ndim == 2:
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    channels = arr.shape[2]
    if channels == 1:
        g = arr[:, :, 0]
        if g.dtype != np.uint8:
            g = np.clip(g, 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    if channels == 4:
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    if channels != 3:
        raise ValueError(f"Unsupported channel count: {channels}")

    if np.issubdtype(arr.dtype, np.floating):
        mx = float(np.max(arr)) if arr.size else 0.0
        if mx <= 1.0 + 1e-6:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def _load_rgb_image(image_path_or_array: Union[str, np.ndarray]) -> np.ndarray:
    """
    Load an RGB uint8 image for face_recognition.

    - str: path loaded with face_recognition (PIL); must exist and be readable.
    - ndarray: treated as OpenCV-style BGR/gray/BGRA unless float (see _numpy_array_to_rgb_uint8).
    """
    if isinstance(image_path_or_array, str):
        path = os.path.abspath(os.path.normpath(image_path_or_array))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Image file not found: {path}")
        try:
            rgb = face_recognition.load_image_file(path)
        except Exception as exc:
            raise ValueError(
                f"Could not read or decode image file (corrupt or unsupported format): {path} ({exc})"
            ) from exc
        out = np.asarray(rgb, dtype=np.uint8)
        if out.ndim != 3 or out.shape[2] != 3:
            raise ValueError(
                f"Loaded image must be RGB with 3 channels; got shape {out.shape}."
            )
        return np.ascontiguousarray(out, dtype=np.uint8)

    return _numpy_array_to_rgb_uint8(np.asarray(image_path_or_array))


def _encoding_from_rgb_single_face(
    rgb_image: np.ndarray,
    *,
    num_jitters: int,
    detection_model: str = "hog",
) -> np.ndarray:
    """Detect faces and return exactly one 128-D encoding."""
    if rgb_image.size == 0:
        raise ValueError("Image has no pixels.")

    locations = face_recognition.face_locations(rgb_image, model=detection_model)
    n = len(locations)
    if n == 0:
        raise ValueError("No face detected (need exactly one clear face in the image).")
    if n > 1:
        raise ValueError(
            f"Multiple faces detected ({n}). Use one face per photo for enrollment."
        )

    encodings = face_recognition.face_encodings(
        rgb_image, locations, num_jitters=num_jitters
    )
    if len(encodings) != 1:
        raise ValueError("Could not compute a face encoding (internal mismatch).")
    return np.asarray(encodings[0], dtype=np.float64)


def enroll_new_student(
    image_path_or_array: Union[str, np.ndarray],
    student_name: str,
    *,
    encodings_path: str = ENCODINGS_FILE,
    csv_path: str = CSV_FILE,
    additional_images: Optional[Sequence[Union[str, np.ndarray]]] = None,
    num_jitters: int = ENROLL_NUM_JITTERS,
    detection_model: str = "hog",
) -> bool:
    """
    Register one student from one or more photos: extract encodings, average them,
    then one gallery load + save via ``_finalize_enrollment``.
    """
    clean_name = (student_name or "").strip()
    if not clean_name:
        msg = "Enrollment failed: student name is empty."
        print(msg)
        logger.error(msg)
        return False

    extras: list[Union[str, np.ndarray]] = []
    if additional_images is not None:
        extras = list(additional_images)
        if len(extras) > ENROLL_MAX_ADDITIONAL_IMAGES:
            msg = (
                f"Enrollment failed: too many additional images ({len(extras)}). "
                f"Maximum {ENROLL_MAX_ADDITIONAL_IMAGES} extra images allowed "
                f"(plus one primary, total up to {ENROLL_MAX_ADDITIONAL_IMAGES + 1})."
            )
            print(msg)
            logger.warning(msg)
            return False

    vectors: list[np.ndarray] = []
    labels: list[str] = []

    def _extract(label: str, item: Union[str, np.ndarray]) -> None:
        try:
            rgb = _load_rgb_image(item)
            vec = _encoding_from_rgb_single_face(
                rgb, num_jitters=num_jitters, detection_model=detection_model
            )
            vectors.append(vec)
            labels.append(label)
        except FileNotFoundError as exc:
            raise ValueError(str(exc)) from exc
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"{label}: {exc}") from exc

    try:
        _extract("primary", image_path_or_array)
        for i, extra in enumerate(extras):
            _extract(f"additional_{i + 1}", extra)
    except ValueError as exc:
        msg = f"Enrollment failed for '{clean_name}': {exc}"
        print(msg)
        logger.error(msg)
        return False

    combined = np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float64)
    detail = f"Images: {', '.join(labels)}."
    return _finalize_enrollment(
        combined,
        clean_name,
        encodings_path=encodings_path,
        csv_path=csv_path,
        success_detail=detail,
    )


def enroll_current_unknown_face(
    encoding: np.ndarray,
    student_name: str,
    *,
    encodings_path: str = ENCODINGS_FILE,
    csv_path: str = CSV_FILE,
) -> bool:
    """
    Enroll using a 128-D vector already computed (e.g. ``pending_unknown_encoding`` from the camera loop).

    Reuses ``_finalize_enrollment`` — same persistence and attendance row as photo enrollment.
    """
    return _finalize_enrollment(
        encoding,
        student_name,
        encodings_path=encodings_path,
        csv_path=csv_path,
        success_detail="Source: live face encoding.",
    )


# =============================================================================
# Recognition — matching and geometry
# =============================================================================


def best_match(
    known_encodings: list,
    known_names: list,
    face_encoding,
    tolerance: float,
) -> tuple[str, float]:
    """
    Return the best matching name and its distance, or UNKNOWN_LABEL if the
    match is weak or ambiguous (second-best too close).
    """
    result = identify_face(
        np.asarray(face_encoding, dtype=np.float64),
        [np.asarray(item, dtype=np.float64) for item in known_encodings],
        known_names,
        FaceCoreConfig(tolerance=tolerance, second_best_gap=SECOND_BEST_GAP),
    )
    if result.status != "matched" or result.best is None:
        best_distance = result.top_candidates[0].distance if result.top_candidates else 1.0
        return UNKNOWN_LABEL, best_distance
    return result.best.name, result.best.distance


def clamp_bbox(
    top: int,
    right: int,
    bottom: int,
    left: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Clamp a bounding box to image dimensions."""
    top = max(0, min(top, height - 1))
    bottom = max(0, min(bottom, height - 1))
    left = max(0, min(left, width - 1))
    right = max(0, min(right, width - 1))
    return top, right, bottom, left


def open_camera() -> cv2.VideoCapture | None:
    """Open the default camera; try DirectShow on Windows first, then default backend."""
    capture = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture = cv2.VideoCapture(CAMERA_INDEX)
    if not capture.isOpened():
        return None
    try:
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return capture


# =============================================================================
# Main — real-time camera loop
# =============================================================================


def main() -> None:
    known_encodings, known_names = load_encodings(ENCODINGS_FILE)
    ensure_csv(CSV_FILE)

    already_attended: set[str] = set()
    unknown_face_count = 0
    pending_unknown_encoding = None
    frame_count = 0

    if USE_LEGACY_TERMINAL_ENROLLMENT:
        legacy_prompt_shown = False
        threading.Thread(target=terminal_input_loop, daemon=True).start()

    cap = open_camera()
    if cap is None:
        print("Camera error - try index 1 or 2")
        return

    print("System ready - press q to quit")
    if not USE_LEGACY_TERMINAL_ENROLLMENT and NOTIFY_ON_PERSISTENT_UNKNOWN:
        print(
            "Note: unknown faces are not enrolled from the terminal. "
            "Use enroll_new_student() with a photo or enroll_current_unknown_face() with a live encoding."
        )

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID
    )
    display_lock = threading.Lock()
    last_face_locations = []
    last_face_names: list[str] = []
    last_inverse_scale = 1.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Camera frame read failed.")
                break

            frame_count += 1
            height, width = frame.shape[:2]

            if (
                RELOAD_GALLERY_EVERY_FRAMES > 0
                and frame_count % RELOAD_GALLERY_EVERY_FRAMES == 0
            ):
                e, n = load_encodings(ENCODINGS_FILE, verbose=False)
                known_encodings, known_names = e, n
                logger.info(
                    "Reloaded face gallery from disk (%d identities).",
                    len(known_names),
                )

            if frame_count % max(1, FRAME_SKIP) == 0:
                small_frame = cv2.resize(
                    frame,
                    (0, 0),
                    fx=RESIZE_FACTOR,
                    fy=RESIZE_FACTOR,
                    interpolation=cv2.INTER_LINEAR,
                )

                # Recognition: natural colors (no CLAHE on pixels used for encodings).
                rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                rgb_small = np.ascontiguousarray(rgb_small, dtype=np.uint8)
                # clahe (created before the loop) is available for optional display-only use, not for encoding.

                try:
                    face_locations = face_recognition.face_locations(
                        rgb_small,
                        number_of_times_to_upsample=UPSAMPLE_TIMES,
                        model="hog",
                    )
                    face_encodings = face_recognition.face_encodings(
                        rgb_small, face_locations
                    )
                except Exception as exc:
                    logger.error("Face processing error: %s", exc)
                    print(f"Face processing error: {exc}")
                    face_locations = []
                    face_encodings = []

                current_names: list[str] = []
                unknown_found = False
                last_unknown_encoding = None

                for _, face_encoding in zip(face_locations, face_encodings):
                    name, _ = best_match(
                        known_encodings, known_names, face_encoding, TOLERANCE
                    )

                    if name != UNKNOWN_LABEL:
                        if name not in already_attended:
                            already_attended.add(name)
                            append_attendance(CSV_FILE, name, "Present on bus")
                    else:
                        unknown_found = True
                        last_unknown_encoding = face_encoding

                    current_names.append(name)

                if unknown_found:
                    unknown_face_count += 1
                    pending_unknown_encoding = last_unknown_encoding
                else:
                    unknown_face_count = 0
                    pending_unknown_encoding = None

                # One-shot notice: only the frame where count first reaches UNKNOWN_THRESHOLD.
                if (
                    NOTIFY_ON_PERSISTENT_UNKNOWN
                    and not USE_LEGACY_TERMINAL_ENROLLMENT
                    and unknown_face_count == UNKNOWN_THRESHOLD
                    and pending_unknown_encoding is not None
                ):
                    print(
                        "\n[Notice] Unregistered face detected (persistent).\n"
                        "Enroll with a photo:\n"
                        "  from bus_attendance_final import enroll_new_student\n"
                        "  enroll_new_student(r'path\\to\\photo.jpg', 'Student Name')\n"
                        "Or use the live encoding from this session:\n"
                        "  from bus_attendance_final import enroll_current_unknown_face\n"
                        "  enroll_current_unknown_face(pending_encoding_array, 'Student Name')\n"
                        "Set RELOAD_GALLERY_EVERY_FRAMES > 0 here to pick up new enrollments without restart.\n"
                    )
                    logger.info(
                        "Persistent unknown: one-time notice (enroll_new_student / enroll_current_unknown_face)."
                    )

                if USE_LEGACY_TERMINAL_ENROLLMENT:
                    if (
                        unknown_face_count >= UNKNOWN_THRESHOLD
                        and pending_unknown_encoding is not None
                    ):
                        if not legacy_prompt_shown:
                            print(
                                "\n[Legacy] New face: enter the student name in this terminal, "
                                "or type 'ignore' to skip."
                            )
                            legacy_prompt_shown = True
                            logger.info("Legacy terminal enrollment prompt shown.")

                        try:
                            queued_name = INPUT_QUEUE.get_nowait()
                        except queue.Empty:
                            queued_name = None

                        if queued_name is not None:
                            clean_name = queued_name.strip()
                            if clean_name and clean_name.lower() != "ignore":
                                if clean_name in known_names:
                                    print(
                                        f"Name '{clean_name}' already exists. Skipping duplicate."
                                    )
                                    logger.warning(
                                        "Duplicate name from terminal: %s", clean_name
                                    )
                                else:
                                    known_encodings.append(
                                        np.asarray(
                                            pending_unknown_encoding, dtype=np.float64
                                        )
                                    )
                                    known_names.append(clean_name)
                                    save_encodings(
                                        ENCODINGS_FILE, known_encodings, known_names
                                    )
                                    print(f"Added {clean_name}")
                                    logger.info(
                                        "Legacy terminal enrollment added: %s", clean_name
                                    )

                                    if clean_name not in already_attended:
                                        already_attended.add(clean_name)
                                        append_attendance(
                                            CSV_FILE,
                                            clean_name,
                                            "Present on bus (new)",
                                        )

                            unknown_face_count = 0
                            pending_unknown_encoding = None
                            legacy_prompt_shown = False

                with display_lock:
                    last_face_locations = face_locations
                    last_face_names = current_names
                    last_inverse_scale = 1.0 / RESIZE_FACTOR

            with display_lock:
                draw_face_locations = list(last_face_locations)
                draw_face_names = list(last_face_names)
                draw_inverse_scale = last_inverse_scale
            for (top, right, bottom, left), name in zip(
                draw_face_locations, draw_face_names
            ):
                top = int(round(top * draw_inverse_scale))
                right = int(round(right * draw_inverse_scale))
                bottom = int(round(bottom * draw_inverse_scale))
                left = int(round(left * draw_inverse_scale))

                top, right, bottom, left = clamp_bbox(
                    top, right, bottom, left, width, height
                )
                color = (0, 255, 0) if name != UNKNOWN_LABEL else (0, 0, 255)

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                text_y = top - 10 if top - 10 > 20 else top + 20
                cv2.putText(
                    frame, name, (left, text_y), FONT, 0.8, color, 2, cv2.LINE_AA
                )

            cv2.putText(
                frame,
                "Attendance:",
                (10, 45),
                FONT,
                1,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                str(len(already_attended)),
                (190, 45),
                FONT,
                1,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_TITLE, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as exc:
        logger.exception("Runtime error: %s", exc)
        print(f"Runtime error: {exc}")
    finally:
        if known_encodings and len(known_encodings) == len(known_names):
            save_encodings(ENCODINGS_FILE, known_encodings, known_names)

        cap.release()
        cv2.destroyAllWindows()

        print("\nSession closed")
        print(f"Attendance saved to: {os.path.abspath(CSV_FILE)}")
        print(f"Total present: {len(already_attended)}")


if __name__ == "__main__":
    main()
