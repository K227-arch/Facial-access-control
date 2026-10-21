# Facial Access Control — System Specification

> Version 1.0 · Generated from codebase analysis · 2026-10-19

## 1. Overview

A Flask-based facial recognition access-control system. It ingests live video from Hikvision RTSP IP cameras (or a browser webcam), runs InsightFace to detect and embed faces, classifies them with an Isolation Forest + CatBoost pipeline, logs recognized accesses to SQLite, and surfaces them through a web dashboard with monitoring, records, incidents, search, and calendar views.

### 1.1 Primary users
- **Security operator** — watches the live monitoring dashboard, reviews records and incidents.
- **Enrollment operator** — adds new people and retrains the classifier.
- **Administrator** — configures cameras, credentials, and model paths (currently via env vars / `config.py`).

## 2. Goals & Non-Goals

**In scope**
- Multi-camera live monitoring (RTSP server-side + webcam client-side).
- Face detection, embedding, identity classification, and unknown-face rejection.
- Access event logging + KPI reporting.
- People enrollment with on-demand model retraining.
- Security incident logging.
- Search and calendar views over historical records.

**Out of scope (today)**
- Dashboard authentication / user login.
- Email or SMS alerting (config stub exists, not wired).
- Multi-tenant or multi-site.
- Mobile app.
- Hardware door-relay / lock integration.
- Real persistent tracking (the `deep_sort/` folder is unused).

## 3. Architecture

```
                  ┌─────────────────────────────┐
                  │   Browser (per-camera card) │
                  │   - RTSP:  <img> MJPEG      │
                  │   - Webcam: getUserMedia    │
                  └──────────────┬──────────────┘
                                 │  poll /getdata/ (webcam) or
                                 │  /api/cameras/<id>/detections (rtsp)
                                 ▼
┌──────────────────────────────────────────────────────────────┐
│                        Flask app (app.py)                     │
│  Routes: /, /records, /incidents, /addpeople, /search,        │
│          /calendar, /api/cameras/*, /api/people, /getdata/      │
└──────┬───────────────────────────────────────┬───────────────┘
       │                                         │
       ▼                                         ▼
┌───────────────┐                      ┌──────────────────────┐
│ rtsp_camera.py│                      │  face_analyzer.py    │
│ (per cam:     │  frame → analyze →   │  InsightFace SCRFD  │
│  reader +     │ ───────────────────► │  + ArcFace glintr100 │
│  inference    │                      │  512-dim embeddings  │
│  threads)     │                      └──────────┬──────────┘
└───────────────┘                                 │
                                                  ▼
                                       ┌─────────────────────┐
                                       │ classifier.py       │
                                       │ IsolationForest     │
                                       │  + CatBoost         │
                                       └──────────┬──────────┘
                                                  │
                                                  ▼
                                       ┌─────────────────────┐
                                       │ database.py         │
                                       │ SQLite: FaceRecords │
                                       │   Incidents         │
                                       └─────────────────────┘
```

### 3.1 Component map
| Component | File | Responsibility |
|----------|------|----------------|
| Web server / routes | `app.py` | HTTP endpoints, template rendering, orchestration |
| Config | `config.py` | Camera list, model paths, thresholds, mail stub |
| DB layer | `database.py` | SQLite schema, CRUD for records & incidents |
| Face analyzer | `face_analyzer.py` | InsightFace wrapper (detection + recognition) |
| Classifier | `classifier.py` | IsolationForest + CatBoost load/predict/retrain |
| RTSP manager | `rtsp_camera.py` | Per-camera reader + inference threads, MJPEG |
| Utils | `utils.py` | `rgb_to_hex`, logging setup |
| Training (legacy) | `train.py`, `train_models.py` | Standalone training scripts (duplicated) |
| Seed | `seed_db.py` | Inserts 3 sample rows for dev |
| Frontend | `templates/*.html`, `static/js/*.js`, `static/css/*.css` | Dashboard UI |

## 4. Data Model

### 4.1 `FaceRecords` (SQLite — `instance/face_recognition.db`)
| Column | Type | Notes |
|--------|------|-------|
| RecordID | INTEGER PK AUTOINCREMENT | |
| Name | TEXT NOT NULL | Classified label |
| Confidence | REAL | 0.0–1.0 |
| Timestamp | TEXT | `'YYYY-MM-DD HH:MM:SS'` |
| Status | TEXT | Always `'recognized'` today (see Bug #1) |
| BoundingBox | TEXT | `str([x1,y1,x2,y2])` |
| Embedding | BLOB | `numpy.ndarray.tobytes()` of 512-d float32 |

### 4.2 `Incidents`
| Column | Type | Notes |
|--------|------|-------|
| IncidentID | INTEGER PK AUTOINCREMENT | |
| Name | TEXT | |
| Status | TEXT NOT NULL | `unauthorized` / `suspicious` / `denied` |
| Description | TEXT | Free text |
| Timestamp | TEXT | `'YYYY-MM-DD HH:MM:SS'` |

### 4.3 Storage on disk
- `./facedata/<PersonName>/*.jpg` — enrollment images.
- `./Models/*.onnx` — InsightFace detection + recognition.
- `./Models/{isolation_forest,catboost_model,label_encoder}.pkl` — trained classifier artifacts.
- `./instance/face_recognition.db` — SQLite database.

## 5. API Reference

### 5.1 Pages (HTML)
| Method | Route | Handler | Notes |
|--------|-------|---------|-------|
| GET | `/` | `home` | Monitoring dashboard |
| GET | `/camera-test` | `camera_test` | Camera testing page |
| GET | `/records` | `records_page` | Records + KPIs + incidents; `?range=all\|today\|week\|month` |
| GET | `/incidents` | `view_incidents` | All incidents |
| POST | `/incidents/add` | `add_incident` | JSON body |
| GET | `/addpeople` | `add_people` (GET) | Enrollment form |
| POST | `/addpeople` | `add_people` (POST) | Upload photos; `action=save\|train` |
| GET | `/search` | `search_page` | Search form |
| GET | `/calendar` | `calendar_page` | Calendar view (simulated data) |

### 5.2 REST / JSON
| Method | Route | Purpose |
|--------|-------|---------|
| GET | `/api/people` | List enrolled people (id, name, avatar) |
| DELETE | `/api/people/<name>` | Delete a person's folder |
| GET | `/api/search` | Search records (`q`, `from`, `to`, `status`) — date filters ignored (Bug #7) |
| POST | `/getdata/` | Webcam mode: base64 image → detections |
| GET | `/api/cameras/config` | Camera list (credentials stripped) |
| GET | `/api/cameras/status` | Live connection status |
| GET | `/api/cameras/stream/<cam_id>` | MJPEG stream |
| GET | `/api/cameras/<cam_id>/snapshot` | Single JPEG frame |
| GET | `/api/cameras/<cam_id>/detections` | Latest AI detections for a camera |
| POST | `/api/cameras/<cam_id>/start` | Start a camera |
| POST | `/api/cameras/<cam_id>/stop` | Stop a camera |
| POST | `/api/cameras/<cam_id>/configure` | Hot-update RTSP URL |

## 6. Configuration

### 6.1 Environment variables
| Variable | Default | Used by |
|----------|---------|---------|
| `FLASK_SECRET` | `'a-very-secret-key'` | Flask session signing (Bug #14) |
| `HIK_USER` | `'admin'` | RTSP credentials |
| `HIK_PASS` | `'admin123'` | RTSP credentials (Bug #5) |
| `HIK_PORT` | `554` | RTSP port |
| `RTSP_TRANSPORT` | `'tcp'` | FFmpeg transport |
| `RTSP_RECONNECT_DELAY` | `5` | Seconds between reconnects |
| `RTSP_MAX_FAILURES` | `10` | Consecutive failures before reconnect |
| `MJPEG_QUALITY` | `75` | JPEG quality 1–100 |
| `RTSP_INFERENCE_INTERVAL` | `0.2` | Seconds between inferences per camera |
| `MAIL_USERNAME` | (unset) | Email — config typo (Bug #5) |
| `MAIL_PASSWORD` | (unset) | Email — config typo (Bug #5) |

### 6.2 Constants
- `CONFIDENCE_THRESHOLD = 0.9` — minimum CatBoost confidence to mark `recognized`.
- `MIN_IMAGES_REQUIRED = 10` — gate for the "Train Model" button.
- `STATUS_COLORS` — RGB for `recognized` (green), `unknown` (red), `processing` (yellow).

## 7. ML Pipeline

1. **Frame source** — RTSP reader thread or browser-uploaded base64 JPEG.
2. **Detection** — InsightFace SCRFD 10G (`scrfd_10g_bnkps.onnx`) at `det_size=(640,640)`.
3. **Embedding** — ArcFace `glintr100.onnx` → 512-d float32 vector per face.
4. **Outlier rejection** — `IsolationForest(contamination=0.3)` flags embeddings as `-1` (unknown) or `1` (inlier).
5. **Classification** — `CatBoostClassifier` predicts the label; `predict_proba().max()` is the confidence.
6. **Decision** — `recognized` if inlier AND confidence ≥ 0.9; else `processing` or `unknown`.
7. **Persistence** — only `recognized` faces are saved today (Bug #1).

### 7.1 Training (`classifier.retrain_model`)
- Walks `./facedata/<person>/*.jpg`, runs the analyzer, collects embeddings + labels.
- Trains IsolationForest, LabelEncoder, CatBoost in sequence.
- Persists to `./Models/*.pkl`.
- **Current implementation forces `task_type="GPU"`** (Bug #2) — crashes on CPU-only machines.

## 8. Threading & Concurrency

- **Flask main process** — serves HTTP via Werkzeug.
- **Per RTSP camera** — two daemon threads:
  - `_reader_thread` — owns the `cv2.VideoCapture`, stores latest frame + JPEG.
  - `_inference_thread` — pulls latest frame every `RTSP_INFERENCE_INTERVAL`, runs analyzer + classifier, updates `_detections` and re-encodes the annotated JPEG.
- **Retraining** — runs on the request thread guarded by `retraining_lock`.
- **Caveat** — `app.run(debug=True)` enables the Werkzeug reloader which double-starts the camera manager (Bug #4).

## 9. Frontend

### 9.1 Pages
| Page | Template | Notes |
|------|----------|-------|
| Monitoring | `index.html` | 4 camera cards, stat cards, recent detections, activity feed |
| People | `addpeople.html` | List + add form, drag-drop upload, train button |
| Records | `records.html` | KPIs, peak hours, recent incidents, records grid |
| Incidents | `incidents.html` | Incident list + pagination (search bar wired but no-op) |
| Search | `search.html` | Name + date range + status filters |
| Calendar | `calendar.html` | Month grid with **simulated** `Math.random()` data |

### 9.2 Camera system (`main.js`)
- `BaseCameraSystem` — abstract lifecycle (connect/disconnect/status).
- `RTSPCameraSystem` — points an `<img>` at the MJPEG endpoint, polls `/detections` every 200 ms, draws boxes on canvas.
- `WebcamCameraSystem` — `getUserMedia` → 10 fps capture → POST to `/getdata/` → draws boxes.

## 10. Security Posture (current state)

⚠ **Not production-ready.** See `BUG_REPORT.md` for the full list. Highlights:
- Default Flask secret key committed in source.
- Hikvision credentials defaulted to `admin/admin123` in source.
- `CORS(app)` is fully permissive.
- No CSRF protection on state-changing endpoints.
- No authentication on any dashboard route.
- No input size limits on uploaded images or `/getdata/` payloads.

## 11. Known Limitations

- No persistent face tracking (`deep_sort/` is present but unused). Same person across frames produces new `track_id`s.
- Database flooding: each recognized face is logged every inference tick (~5 fps) with no deduplication (Bug #3).
- Records page shows hardcoded `Location` and `Camera ID` for every record.
- Dashboard stat cards (active cameras, face detections) are partially hardcoded.
- Calendar view is fully simulated client-side; no backend integration.
- Three duplicated training implementations (`train.py`, `train_models.py`, `classifier.retrain_model`).
- Search date filters and `status` filter accepted but ignored server-side.

## 12. Glossary
- **InsightFace** — face analysis toolkit providing SCRFD detection + ArcFace recognition.
- **SCRFD** — Sample and Computation Redistribution for Efficient Face Detection.
- **ArcFace** — additive angular margin loss face recognition model; produces 512-d embeddings.
- **Isolation Forest** — unsupervised outlier detector; used to reject unknown faces.
- **CatBoost** — gradient-boosted trees classifier over the 512-d embeddings.
- **MJPEG** — Motion JPEG; a stream of JPEG frames served as `multipart/x-mixed-replace`.
- **RTSP** — Real-Time Streaming Protocol; used by Hikvision IP cameras.
