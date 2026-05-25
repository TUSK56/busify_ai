# Hugging Face deploy guide (Busify face pipeline)

Your build failed because the Dockerfile used `COPY quality_model/requirements.txt` but on Hugging Face **all files sit in the Space root** (no `quality_model/` subfolder).

---

## Space 1: `TUSK000/quality_model` (GFPGAN)

### Files to upload (Space root — exactly these)

| File | From your PC |
|------|----------------|
| `Dockerfile` | `last_update_yolo_arc/quality_model/Dockerfile` |
| `main.py` | `last_update_yolo_arc/quality_model/main.py` |
| `enhancer.py` | `last_update_yolo_arc/quality_model/enhancer.py` |
| `requirements.txt` | `last_update_yolo_arc/quality_model/requirements.txt` |
| `GFPGANv1.4.pth` | `last_update_yolo_arc/quality_model/GFPGANv1.4.pth` |
| `README.md` | optional |

**Do not** upload a `quality_model/` folder — put files directly in the Space root (as you already did).

### Environment variables (Settings → Variables)

| Key | Value | Required |
|-----|-------|----------|
| `GFPGAN_MODEL_PATH` | `/app/GFPGANv1.4.pth` | Optional (default) |
| `GFPGAN_UPSCALE` | `1` | Optional |
| `GFPGAN_ENABLE` | `1` | Optional |

No secrets needed for quality Space.

### Test

```text
GET https://tusk000-quality-model.hf.space/health
```

---

## Space 2: `TUSK000/yolo_arc` (YOLO + ArcFace)

### Files to upload (Space root)

| File | From your PC |
|------|----------------|
| `Dockerfile` | `last_update_yolo_arc/yolo_arc_model/Dockerfile` |
| `main.py` | `last_update_yolo_arc/yolo_arc_model/main.py` |
| `requirements.txt` | `last_update_yolo_arc/yolo_arc_model/requirements.txt` |
| `README.md` | optional |

**Use the files inside `yolo_arc_model/`**, not the parent `last_update_yolo_arc/` copies (they are for local monorepo builds only).

You do **not** upload `GFPGANv1.4.pth` or `enhancer.py` to yolo_arc.

### Environment variables (Settings → Variables) — yolo_arc only

| Key | Value | Required |
|-----|-------|----------|
| `QUALITY_API_URL` | `https://tusk000-quality-model.hf.space` | **Yes** (no trailing slash) |
| `QUALITY_ENHANCE` | `1` | Optional |
| `QUALITY_API_TIMEOUT` | `90` | Optional |
| `YOLO_MODEL` | `/app/yolov8n-face.pt` | Optional |

**Do not** add `GFPGAN_MODEL_PATH`, `GFPGAN_ENABLE`, or `GFPGAN_UPSCALE` on yolo_arc — those are for the **quality_model** Space only.

---

## Backend (MonsterASP / appsettings)

App and backend call **only the yolo_arc Space**:

| Key | Value |
|-----|-------|
| `FaceMatch__BaseUrl` | `https://tusk000-yolo-arc.hf.space` |
| `FaceMatch__EmbeddingVersion` | `yolov8-arcface-v6-quality` (after redeploy, then re-embed all) |

Flutter parent app uses the same yolo URL for `/embed` (no change if URL unchanged).

---

## Which folder to use?

| Deploy target | Use these files |
|---------------|-----------------|
| **HF quality Space** | `quality_model/*` (flat in Space root) |
| **HF yolo Space** | `yolo_arc_model/*` **plus** `quality_model/enhancer.py` |
| **Local docker from repo root** | Parent `last_update_yolo_arc/Dockerfile` + `main.py` (optional, not for HF upload) |

---

## Pipeline

```text
Flutter / Backend → yolo_arc (/embed, /match)
                         → quality_model (/enhance)  [via QUALITY_API_URL]
                         → YOLO + ArcFace
```

After both Spaces are **Running**, bump `EmbeddingVersion` and re-embed all students.
