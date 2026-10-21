# Bug Report — Facial Access Control

> Generated from codebase analysis · 2026-10-19
> Severity: 🔴 Critical · 🟠 High · 🟡 Medium · ⚪ Low

## Summary

| # | Sev | Title | File | Status |
|---|-----|-------|------|--------|
| 1 | 🔴 | Status hardcoded to `recognized` in DB | `database.py` | ✅ Fixed |
| 2 | 🔴 | `retrain_model` forces `task_type="GPU"` | `classifier.py` | ✅ Fixed |
| 3 | 🔴 | RTSP inference floods DB with duplicates | `rtsp_camera.py` | ✅ Fixed |
| 4 | 🔴 | `debug=True` double-starts camera threads | `app.py` | ✅ Fixed |
| 5 | 🟠 | `os.getenv` misuse for mail credentials | `config.py` | ✅ Fixed |
| 6 | 🟠 | `retrain_model` ignores config paths, writes `iso_forest` twice | `classifier.py` | Open |
| 7 | 🟠 | `/api/search` ignores `from`, `to`, `status` filters | `app.py` | ✅ Fixed |
| 8 | 🟠 | Dashboard stat cards hardcoded | `templates/index.html` | Open |
| 9 | 🟠 | `records.html` hardcodes Location & Camera ID | `templates/records.html` | Open |
| 10 | 🟡 | Calendar uses `Math.random()` simulated data | `templates/calendar.html` | Open |
| 11 | 🟡 | `incidents.html` search bar is a no-op | `templates/incidents.html` | ✅ Fixed |
| 12 | 🟡 | `get_peak_hours` ignores `time_range` | `database.py` | ✅ Fixed |
| 13 | 🟡 | Three duplicated training implementations | `train.py`, `train_models.py`, `classifier.py` | Open |
| 14 | 🟡 | Default Flask secret committed in source | `app.py`, `config.py` | ✅ Fixed |
| 15 | 🟡 | No CSRF on state-changing routes; CORS fully permissive | `app.py` | Open |
| 16 | 🟡 | Hikvision credentials defaulted in source | `config.py` | Open |
| 17 | ⚪ | `MIN_IMAGES_REQUIRED=10` blocks training; current datasets have ~4 | `app.py` | Open |
| 18 | ⚪ | `PCA_MODEL_PATH` configured but unused | `config.py` | Open |
| 19 | ⚪ | `tracked_faces`, `next_face_id`, `DB_CONFIG` dead code | `app.py`, `config.py` | Open |
| 20 | ⚪ | Committed junk: `embeddings.npy`, `labels.npy`, `catboost_info/`, `myenv/`, `__pycache__/` | repo root | Open |
| 21 | ⚪ | `package.json` / `package-lock.json` with no Node usage | repo root | Open |
| 22 | ⚪ | `save_face_record` will crash if `embedding` is None | `database.py` | ✅ Fixed |
| 23 | ⚪ | `record.Confidence * 100` may render as `95.0%` | `templates/records.html` | Open |

---

## ✅ Applied Fixes

### Fix #1 — Status persisted correctly (`database.py`, `rtsp_camera.py`, `app.py`)
**Problem.** `save_face_record` always wrote `'recognized'` as the status literal, and callers only called it when `status == 'recognized'`. Unknown and processing faces were never saved; KPI counts for `unknown` were always 0; the records-page filters for unknown/entering/exiting matched nothing.

**Fix.** Added `status` as an explicit parameter to `save_face_record` with `'recognized'` as default for backward compatibility. Updated both call sites (`get_data` in `app.py` and `_inference_loop` in `rtsp_camera.py`) to pass the actual status through. Both paths now save all detected faces (all statuses) subject to the dedup cooldown.

Also guards `embedding.tobytes()` against `None` (Bug #22) in the same change.

### Fix #2 — CatBoost GPU/CPU auto-detection (`classifier.py`)
`retrain_model` forced `task_type="GPU"`, crashing on CPU-only machines. Now tries GPU, falls back to CPU on any exception.

### Fix #3 — Per-camera deduplication (`rtsp_camera.py`)
The inference loop saved a DB row every ~200 ms per recognized face. Added a 30-second cooldown per `(camera, label)` pair. Extended in the Fix #1 pass to cover all statuses (not just `recognized`) and to be placed before the status check, so dedup applies uniformly.

### Fix #4 — Reloader double-start (`app.py`)
`app.run(debug=True)` without `use_reloader=False` spawned a child process that re-ran module-level startup (models, camera threads) a second time. Fixed with `use_reloader=False`. Debug mode is still controllable via `FLASK_DEBUG` env var. Both parent and child previously leaked RTSP sockets.

### Fix #5 — Mail credential env vars (`config.py`)
`os.getenv('admin@gmail.com')` and `os.getenv('admin123')` always returned `None` (those strings aren't env-var names). Fixed to `os.getenv('MAIL_USERNAME')` and `os.getenv('MAIL_PASSWORD')`.

### Fix #7 — `/api/search` date and status filters (`app.py`)
The `from`, `to`, and `status` query params were extracted but never applied. Now all four params (`q`, `from`, `to`, `status`) correctly filter the results list.

### Fix #11 — Incidents search bar wired up (`templates/incidents.html`)
`searchInput` had no `input` event listener. Rewrote the inline script to separate `allRows` (immutable master list) from `filteredRows` (filtered subset). `applySearch` recomputes `filteredRows` on every keystroke and resets pagination. Pagination's `nextBtn` guard was also missing a `totalPages` check (could over-page). Both corrected.

### Fix #12 — `get_peak_hours` respects `time_range` (`database.py`)
The function accepted `time_range` but the SQL never filtered. Now mirrors the same `WHERE` clause logic already used in `get_records`, so the Peak Hours panel on the records page reflects the selected time tab.

### Fix #14 — Ephemeral Flask secret (`app.py`)
Removed the hardcoded default `'a-very-secret-key'`. If `FLASK_SECRET` is unset the app generates a `secrets.token_hex(32)` ephemeral key at startup and prints a warning. Sessions still work but won't survive restarts, which is safe for dev. In production the operator must set `FLASK_SECRET`.

---

## Open Bugs (detailed)

### 🟠 #6 — `retrain_model` ignores config paths and writes IsolationForest twice
**Where:** `classifier.py::retrain_model` hardcodes `./Models/catboost_model.pkl` etc. and calls `pickle.dump(iso_forest_model, f)` twice.
**Impact:** Mostly harmless today because config paths happen to match the hardcoded ones, but it breaks the config abstraction and is wasteful.
**Fix:** Import and use `ISOLATION_FOREST_PATH`, `SVM_MODEL_PATH`, `LABEL_ENCODER_PATH` from `config`; remove the duplicate dump.

### 🟠 #7 — ~~`/api/search` ignores date and status filters~~ ✅ Fixed (see Applied Fixes)

### 🟡 #10 — Calendar view uses `Math.random()`
**Where:** `templates/index.html` shows `127`, `8`, `0`, `1 of 2` literally in the HTML.
**Impact:** Dashboard lies on first paint. Only some cards are corrected by `main.js` after detections arrive; "Active Cameras" stays hardcoded.
**Fix:** Add a `/api/kpis?range=today` endpoint that returns live counts, and render them server-side via Jinja (eliminating the first-paint lie).

### 🟠 #9 — `records.html` hardcodes Location & Camera ID
**Where:** `templates/records.html` prints `Area B – Hallway` and `CAM-03` for every record.
**Impact:** Records page is misleading for any camera other than CAM-03. Also: the `FaceRecords` schema doesn't even store camera ID.
**Fix:** Schema migration to add `CameraID`, `Location` columns; thread `cam_id` through `_save_record` → `save_face_record`; render from the record.

### 🟡 #10 — Calendar view uses `Math.random()`
**Where:** `templates/calendar.html::loadMonthEvents` generates fake daily counts.
**Impact:** The calendar is decorative. Users may assume it reflects real data.
**Fix:** Add `/api/calendar/<year>/<month>` returning per-day aggregates from `FaceRecords` + `Incidents`; fetch and render.

### 🟡 #11 — ~~`incidents.html` search bar is a no-op~~ ✅ Fixed (see Applied Fixes)

### 🟡 #12 — ~~`get_peak_hours` ignores `time_range`~~ ✅ Fixed (see Applied Fixes)

### 🟡 #13 — Three duplicated training implementations
**Where:** `train.py`, `train_models.py`, `classifier.py::retrain_model`.
**Impact:** Divergent defaults (GPU vs CPU, `contamination=0.3` vs `0.1`, augmentation vs none). Bug fixes applied to one don't propagate.
**Fix:** Delete `train.py`. Move `train_models.py`'s cleaner logic into `classifier.retrain_model`. Keep one entry point.

### 🟡 #14 — ~~Default Flask secret committed~~ ✅ Fixed (see Applied Fixes)

### 🟡 #15 — No CSRF; CORS fully permissive
**Where:** `app.py` `CORS(app)` and POST/DELETE routes accept any origin.
**Impact:** A malicious page on the same LAN can submit incidents, delete people, or reconfigure cameras.
**Fix:** Add Flask-WTF CSRF; restrict `CORS` to known origins or remove it (templates are same-origin).

### 🟡 #16 — Hikvision credentials defaulted in source
**Where:** `config.py` `HIK_PASS = os.getenv('HIK_PASS', 'admin123')`.
**Impact:** A deployed instance that forgets to set env vars will try `admin/admin123` against every camera.
**Fix:** Remove the default; log a warning if unset; optionally refuse to start cameras.

### ⚪ #17 — `MIN_IMAGES_REQUIRED=10` blocks training
**Where:** `app.py` constant; `addpeople.js` hides the Train button below the threshold.
**Impact:** Current datasets (4 images per person) can't be trained from the UI. Combined with Bug #2 the training path was doubly broken.
**Fix:** Lower default to 5; warn rather than hide. Also: data augmentation in `train.py` exists and should be ported into `retrain_model` so 5 images produce ~10 effective samples.

### ⚪ #18 — `PCA_MODEL_PATH` unused
**Where:** `config.py`. No import site.
**Fix:** Delete the line or wire up PCA dim-reduction before CatBoost.

### ⚪ #19 — Dead code
**Where:** `app.py` `tracked_faces`, `next_face_id` (globals, never read); `config.py` `DB_CONFIG` (MSSQL, never used; the app uses SQLite).
**Fix:** Delete.

### ⚪ #20 — Committed junk in repo
**Where:** `embeddings.npy`, `labels.npy`, `catboost_info/`, `myenv/`, `Lib/`, `Scripts/`, `__pycache__/`.
**Fix:** `git rm --cached` them; extend `.gitignore` (`myenv/`, `Lib/`, `Scripts/`, `catboost_info/`, `__pycache__/`, `*.npy`).

### ⚪ #21 — Orphaned Node files
**Where:** `package.json`, `package-lock.json`.
**Fix:** Delete unless a Node toolchain is actually used (none detected).

### ⚪ #22 — ~~`save_face_record` crashes on None embedding~~ ✅ Fixed (guarded in the same change as Fix #1)

### ⚪ #23 — ~~Confidence rendered as `95.0%`~~ ✅ Fixed (now `{{ (record.Confidence * 100) | round(0) | int }}%`)
