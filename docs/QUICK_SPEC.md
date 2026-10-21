# Facial Access Control — Quick Spec

> One-page condensed spec. See `SPEC.md` for full detail, `BUG_REPORT.md` for issues, `PLAN.md` for the roadmap.

## What it is
A Flask app that watches Hikvision RTSP cameras (or a webcam), detects faces with InsightFace, classifies them with IsolationForest + CatBoost, logs recognized accesses to SQLite, and shows them on a dashboard.

## Stack
- **Backend:** Flask + Flask-CORS, Python 3.
- **ML:** InsightFace (SCRFD + ArcFace 512-d) → IsolationForest (outlier) → CatBoost (classifier).
- **Storage:** SQLite (`instance/face_recognition.db`); tables `FaceRecords`, `Incidents`.
- **Cameras:** RTSP via OpenCV/FFmpeg (server-side MJPEG); `getUserMedia` (browser-side, POST to `/getdata/`).
- **Frontend:** Jinja2 templates + vanilla JS; Bootstrap Icons + Font Awesome.

## Top-level files
| File | Role |
|------|------|
| `app.py` | Flask routes, orchestration, startup |
| `config.py` | Cameras, model paths, thresholds |
| `database.py` | SQLite schema + CRUD |
| `face_analyzer.py` | InsightFace wrapper |
| `classifier.py` | Load/predict/retrain classifier |
| `rtsp_camera.py` | Per-camera reader + inference threads, MJPEG |
| `train_models.py` | Standalone training script |
| `templates/` | Dashboard pages |
| `static/js/main.js` | Camera cards + detection rendering |

## Key endpoints
- `GET /` — monitoring dashboard
- `GET /api/cameras/stream/<id>` — MJPEG stream
- `GET /api/cameras/<id>/detections` — latest AI detections
- `POST /getdata/` — webcam frame → detections (base64 JPEG in, JSON out)
- `GET /records`, `/incidents`, `/addpeople`, `/search`, `/calendar` — pages
- `POST /addpeople` (with `action=train`) — retrain classifier

## ML pipeline
Frame → SCRFD detection → ArcFace 512-d embedding → IsolationForest (inlier?) → CatBoost (label + confidence) → `recognized` if confidence ≥ 0.9 else `unknown`/`processing`.

## Config knobs
`HIK_USER`, `HIK_PASS`, `HIK_PORT`, `RTSP_TRANSPORT`, `RTSP_INFERENCE_INTERVAL` (0.2 s), `MJPEG_QUALITY` (75), `CONFIDENCE_THRESHOLD` (0.9), `MIN_IMAGES_REQUIRED` (10), `FLASK_SECRET`.

## Top 3 bugs to fix first
1. **Status hardcoded** — `save_face_record` always writes `'recognized'`; unknown faces never persisted; KPIs wrong. *(🔴 `database.py`)*
2. **Training crashes on CPU** — `retrain_model` forces `task_type="GPU"`. *(✅ Fixed in this pass)*
3. **DB floods** — RTSP inference logs the same face every 200 ms. *(✅ Fixed in this pass — 30 s per-camera cooldown)*

## Production-readiness gaps
- No dashboard auth; default secret in source; CORS fully open; no CSRF.
- Hikvision credentials defaulted to `admin/admin123` in source.
- Calendar view is `Math.random()`; records page hardcodes `CAM-03 / Area B – Hallway`.
- `debug=True` + module-level `camera_manager.start()` double-starts threads under the Werkzeug reloader.

## Deployment shape
Single Python process; per-camera daemon threads for RTSP; SQLite single-writer; suitable for a small office / single building with ≤ 8 cameras. Beyond that, move to Postgres + a queue.
