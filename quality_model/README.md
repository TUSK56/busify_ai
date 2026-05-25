---
title: Busify Quality
emoji: ✨
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# Busify Quality (GFPGAN) — Step 1

Upload these files to **this Space root** (not in a subfolder):

- `Dockerfile`
- `main.py`
- `enhancer.py`
- `requirements.txt`
- `GFPGANv1.4.pth`

## Variables (Settings)

| Key | Value |
|-----|-------|
| `GFPGAN_MODEL_PATH` | `/app/GFPGANv1.4.pth` |
| `GFPGAN_ENABLE` | `1` |

See `HF_DEPLOY.md` in the repo for full wiring to yolo_arc.
