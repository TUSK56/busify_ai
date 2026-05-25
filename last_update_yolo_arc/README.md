---
title: Yolo Arc
emoji: 🎓
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

## Two-step face pipeline (v5)

| Step | Folder | HF Space | Role |
|------|--------|----------|------|
| 1 | `quality_model/` | Busify Quality | GFPGAN — clearer photo |
| 2 | `yolo_arc_model/` or root `main.py` | Busify Yolo Arc | YOLO + ArcFace — embed / match |

Set on the **yolo_arc** Space: `QUALITY_API_URL=https://your-quality-space.hf.space`

The Flutter app and .NET backend still call only the yolo_arc URL (`/embed`, `/match`).

See `quality_model/README.md` and `yolo_arc_model/README.md`.

## Match tuning (env)

| Variable | Default | Meaning |
|----------|---------|---------|
| `YOLO_MODEL` | `/app/yolov8n-face.pt` | Local face weights (auto-downloaded from Hugging Face if missing) |
| `FACE_MIN_COSINE_SIM` | `0.55` | Min similarity when 2+ students on bus |
| `FACE_MIN_COSINE_SIM_SINGLE` | `0.68` | Stricter min when only 1 enrolled face (blocks impostors) |
| `FACE_MIN_TOP1_MARGIN` | `0.04` | Gap when runner-up is also plausible |
| `FACE_STRONG_COSINE_SIM` | `0.62` | Halves margin when top score is strong |
| `FACE_MIN_DET_SCORE` | `0.50` | YOLO face detection confidence |
| `FACE_CROP_ENHANCE` | `1` | Set `0` to disable mild CLAHE on face crop |

After redeploying this build, **re-save each student face photo** (parent signup / add child) so enrollments use the same face-YOLO pipeline as live scans.
