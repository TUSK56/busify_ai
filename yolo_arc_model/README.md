---
title: Busify Yolo Arc
emoji: 🎓
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Busify YOLO + ArcFace — Step 2

Upload to **this Space root** (only 4 files):

- `Dockerfile`
- `main.py`
- `requirements.txt`
- `README.md` (optional)

Do **not** set `GFPGAN_*` variables here — those belong on the **quality_model** Space only.

## Required variable (yolo_arc Space only)

| Key | Value |
|-----|-------|
| `QUALITY_API_URL` | `https://tusk000-quality-model.hf.space` |

Backend / Flutter use **this Space URL** for `/embed` and `/match`.

See `HF_DEPLOY.md` in the repo for full details.
