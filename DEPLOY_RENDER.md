# Deploy face API on Render + push this repo to GitHub

## 1) Commit and push to `busify_ai`

From your machine (replace path if needed):

```powershell
cd D:\ahmed\face_recognition_project
git init
git add .
git commit -m "Add face embedding API and project files"
git branch -M main
git remote add origin https://github.com/TUSK56/busify_ai.git
git push -u origin main
```

If GitHub shows authentication errors, use a **Personal Access Token** as the password, or install GitHub CLI (`gh auth login`).

**Note:** large `*.dat` files are listed in `.gitignore` so the repo stays small. The Docker build installs `dlib`/`face_recognition` from pip.

## 2) Create a Render Web Service (Docker)

1. Open [Render Dashboard](https://dashboard.render.com) → **New** → **Web Service**.
2. Connect the `TUSK56/busify_ai` repository (or use **Public Git URL**).
3. **Root directory:** `render_api` (important).
4. **Environment:** **Docker** (Render will use `render_api/Dockerfile`).
5. **Instance type:** start with the smallest paid CPU if builds fail on free tier (dlib builds are heavy).
6. Deploy. Wait for the first build (can take 10–20+ minutes).
7. Copy the public URL, e.g. `https://busify-ai.onrender.com`.

### Smoke test

- `GET https://YOUR_SERVICE/health` → should return `{"status":"ok"}`.
- `POST https://YOUR_SERVICE/embed` with a face JPEG as multipart form field `file`.

## 3) Wire the .NET backend (optional next step)

Set an environment variable on your API host, for example:

- `FACE_SERVICE_URL=https://YOUR_SERVICE.onrender.com`

Then extend `FaceRecognitionService` to POST enrollment images to `/embed` and store the returned `embedding` JSON in `student_face_profile` (already created on parent signup with `[]` until you connect this).

## 4) Flutter app

Parent signup now sends `photoBase64` + `faceLabel`; the API saves the image under `/uploads/student-photos/` and creates a `student_face_profile` row. After the Python service returns real embeddings, matching quality improves.
