# Implementation Plan — Facial Access Control

> Phased roadmap. Each phase is independently shippable. Effort estimates assume one developer.

## Phase 0 — Stabilize (1–2 days)
**Goal:** the existing system runs end-to-end without crashes on a stock CPU laptop.

- [ ] **0.1** Auto-detect GPU vs CPU in `classifier.retrain_model` (Bug #2). *High value, low risk.*
- [ ] **0.2** Fix `os.getenv` misuse in `config.py` MAIL block (Bug #5). *Trivial.*
- [ ] **0.3** Guard RTSP / analyzer init against the Werkzeug reloader's double-start (Bug #4). Wrap startup in `if not os.environ.get('WERKZEUG_RUN_MAIN') and app.debug:` or pass `use_reloader=False`.
- [ ] **0.4** Add per-camera deduplication so a stationary person isn't logged 5×/second (Bug #3). 30-second cooldown per `label` per camera.
- [ ] **0.5** Persist the actual `Status` (`recognized`/`unknown`/`processing`) in `save_face_record` so KPIs and filters work (Bug #1).

**Exit criteria:** Run the app fresh, point a webcam at a face, see exactly one record per ~30 s, "Train Model" completes on CPU.

## Phase 1 — Correctness (3–5 days)
**Goal:** what the UI shows matches what actually happened.

- [ ] **1.1** Schema migration: add `CameraID`, `Location` columns to `FaceRecords`. Backfill as `'CAM-XX'` from a new `camera_id` field threaded through the inference path.
- [ ] **1.2** `records.html`: render real `CameraID`/`Location` per record instead of hardcoded `CAM-03 / Area B – Hallway`.
- [ ] **1.3** `index.html`: replace hardcoded stats (`127`, `8`, `0`, `1 of 2`) with values from a new `/api/kpis?range=today` endpoint.
- [ ] **1.4** `app.py::search_records`: implement `from`, `to`, and `status` filters against `FaceRecords`.
- [ ] **1.5** `database.py::get_peak_hours`: honor `time_range` (currently ignored).
- [ ] **1.6** `incidents.html`: wire the `searchInput` to filter the visible list (currently dead).

**Exit criteria:** Records page shows correct camera per record; dashboard stats update live; search returns date-filtered results.

## Phase 2 — Hardening (3–5 days)
**Goal:** safe to expose on a trusted LAN.

- [ ] **2.1** Replace default Flask secret with env-var-only + fail-closed startup if unset.
- [ ] **2.2** Strip default Hikvision credentials from `config.py`; require `HIK_USER`/`HIK_PASS` to be set.
- [ ] **2.3** Add CSRF protection (Flask-WTF) to `/incidents/add`, `/addpeople`, `/api/people/<name>` DELETE, `/api/cameras/<id>/configure`.
- [ ] **2.4** Tighten `CORS(app)` to an explicit origin allow-list (or remove if same-origin only).
- [ ] **2.5** Add rate limiting on `/getdata/` and `/api/cameras/<id>/configure` (Flask-Limiter).
- [ ] **2.6** Add upload size + content-type validation on `/addpeople` and `/getdata/`.
- [ ] **2.7** Sanitize the `name` field on `/addpeople` (length, character whitelist).

**Exit criteria:** App refuses to start with missing secrets; CSRF tokens enforced; CORS restricted; no unbounded uploads.

## Phase 3 — Training UX (2–3 days)
**Goal:** training works from the UI on any machine with progress feedback.

- [ ] **3.1** Consolidate training: delete `train.py`, fold `train_models.py` logic into `classifier.retrain_model`. One code path.
- [ ] **3.2** Lower `MIN_IMAGES_REQUIRED` default to 5; warn below 10 instead of hiding the button.
- [ ] **3.3** Run training in a background thread; expose `/api/train/status` for polling. Consider Flask-SocketIO (already in `requirements.txt`).
- [ ] **3.4** Persist training metadata (`./Models/training_meta.json`): last trained at, # embeddings, # people, accuracy.
- [ ] **3.5** Add a "Retrain" button on the dashboard's People page that doesn't require uploading new photos.

**Exit criteria:** Train from UI on CPU laptop, see progress, no page freeze.

## Phase 4 — Features (1–2 weeks)
**Goal:** close the gap between the UI affordances and the backend.

- [ ] **4.1** Dashboard auth: Flask-Login with a single admin user (env-configured).
- [ ] **4.2** Email/Slack alert on unknown-face incident (wire up the `Config` mail stub; or Slack webhook).
- [ ] **4.3** Calendar view: replace `Math.random()` simulation with `/api/calendar/<year>/<month>` backed by real records.
- [ ] **4.4** Records export to CSV/Excel.
- [ ] **4.5** Per-camera enable/disable + live RTSP URL edit in the UI (backend already supports `/configure`).
- [ ] **4.6** Optional: real face tracking with `deep_sort` so a person walking through frame keeps one `track_id`.
- [ ] **4.7** Optional: door-relay integration via Hikvision ISAPI or a GPIO callback.

## Phase 5 — Operations (ongoing)
- [ ] **5.1** Dockerize: `Dockerfile` + `docker-compose.yml` with named volumes for `Models/` and `instance/`.
- [ ] **5.2** `/healthz` endpoint (DB reachable, models loaded, ≥1 camera connected).
- [ ] **5.3** Structured JSON logging + log rotation.
- [ ] **5.4** SQLite backup/restore script (cron + `VACUUM INTO`).
- [ ] **5.5** CI: `ruff`/`black` lint + `pytest` for `database.py` and `classifier.py` (unit-testable without GPU).
- [ ] **5.6** Pre-commit hook + `.gitignore` audit: keep `Models/*.pkl`, `instance/*.db`, `myenv/`, `__pycache__/`, `catboost_info/`, `embeddings.npy`, `labels.npy` out of Git.
- [ ] **5.7** Remove orphaned Node files (`package.json`, `package-lock.json`) or document their purpose.

## Cross-cutting decisions to make
1. **Tracking:** implement real tracking (deep_sort) or keep per-frame inference + 30 s cooldown? Recommend cooldown first (Phase 0.4), tracking in Phase 4.6.
2. **Storage:** stay on SQLite or move to Postgres for multi-writer concurrency? SQLite is fine up to ~1 write/sec; recommend Postgres if camera count grows beyond 8.
3. **Auth scope:** single admin user or RBAC (operator vs. admin)? Recommend single admin first.
4. **Deployment:** Docker on a dedicated mini-PC, or Windows-native? Current code is Windows-path-friendly; Docker is cleaner for Linux cameras.

## Sequencing rationale
Phase 0 fixes the bugs that make the system actively misbehave today (crashes, DB flooding, wrong status). Phase 1 fixes the bugs that make the system lie to the user (hardcoded data, ignored filters). Phase 2 makes it safe to put on a network. Phases 3–5 add value and operability. Each phase's exit criteria are verifiable by a manual smoke test.
