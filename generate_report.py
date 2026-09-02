"""
generate_report.py
Generates the RTSP + AI Integration technical report as a Word document.
Run: python3 generate_report.py
"""

from docx import Document
from docx.shared import Pt, Inches, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import datetime

doc = Document()

# ── Page margins ────────────────────────────────────────────────────────────
for section in doc.sections:
    section.top_margin    = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin   = Cm(3.0)
    section.right_margin  = Cm(2.5)

# ── Helper utilities ─────────────────────────────────────────────────────────
def set_font(run, name='Calibri', size=11, bold=False, italic=False, color=None):
    run.font.name  = name
    run.font.size  = Pt(size)
    run.font.bold  = bold
    run.font.italic = italic
    if color:
        run.font.color.rgb = RGBColor(*color)

def heading(text, level=1, color=(31, 73, 125)):
    p = doc.add_heading(text, level=level)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for run in p.runs:
        run.font.color.rgb = RGBColor(*color)
        run.font.name = 'Calibri'
    return p

def body(text, space_after=6):
    p = doc.add_paragraph(text)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for run in p.runs:
        set_font(run)
    p.paragraph_format.space_after = Pt(space_after)
    return p

def bullet(text, level=0):
    p = doc.add_paragraph(text, style='List Bullet')
    p.paragraph_format.left_indent = Inches(0.3 + level * 0.25)
    for run in p.runs:
        set_font(run)
    return p

def numbered(text):
    p = doc.add_paragraph(text, style='List Number')
    for run in p.runs:
        set_font(run)
    return p

def code_block(text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent  = Inches(0.4)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after  = Pt(4)
    run = p.add_run(text)
    run.font.name  = 'Courier New'
    run.font.size  = Pt(9)
    run.font.color.rgb = RGBColor(0, 70, 127)
    # light grey shading
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'),   'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'),  'F2F2F2')
    pPr.append(shd)
    return p

def divider():
    p = doc.add_paragraph('─' * 90)
    for run in p.runs:
        run.font.color.rgb = RGBColor(180, 180, 180)
        run.font.size = Pt(7)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(2)

def add_table(headers, rows, col_widths=None):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = 'Table Grid'
    t.alignment = WD_TABLE_ALIGNMENT.LEFT
    # Header row
    hrow = t.rows[0]
    for i, h in enumerate(headers):
        cell = hrow.cells[i]
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        cell.paragraphs[0].clear()
        run = cell.paragraphs[0].add_run(h)
        set_font(run, bold=True, size=10, color=(255, 255, 255))
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:val'),   'clear')
        shd.set(qn('w:color'), 'auto')
        shd.set(qn('w:fill'),  '1F497D')
        tcPr.append(shd)
    # Data rows
    for r_idx, row_data in enumerate(rows):
        row = t.rows[r_idx + 1]
        fill = 'FFFFFF' if r_idx % 2 == 0 else 'EBF1F5'
        for c_idx, cell_text in enumerate(row_data):
            cell = row.cells[c_idx]
            cell.paragraphs[0].clear()
            run = cell.paragraphs[0].add_run(cell_text)
            set_font(run, size=10)
            tc = cell._tc
            tcPr = tc.get_or_add_tcPr()
            shd = OxmlElement('w:shd')
            shd.set(qn('w:val'),   'clear')
            shd.set(qn('w:color'), 'auto')
            shd.set(qn('w:fill'),  fill)
            tcPr.append(shd)
    if col_widths:
        for i, w in enumerate(col_widths):
            for row in t.rows:
                row.cells[i].width = Inches(w)
    doc.add_paragraph()
    return t

# ════════════════════════════════════════════════════════════════════════════
# COVER PAGE
# ════════════════════════════════════════════════════════════════════════════
doc.add_paragraph()
doc.add_paragraph()

title_p = doc.add_paragraph()
title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title_p.add_run('RTSP Integration & AI-Powered Face Detection')
set_font(run, size=24, bold=True, color=(31, 73, 125))

sub_p = doc.add_paragraph()
sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = sub_p.add_run('Using Hikvision IP Cameras on a LAN Network')
set_font(run, size=16, italic=True, color=(68, 114, 196))

doc.add_paragraph()
divider()
doc.add_paragraph()

meta_lines = [
    ('System',   'Facial Access Control System'),
    ('Version',  '2.0'),
    ('Date',     datetime.date.today().strftime('%d %B %Y')),
    ('Author',   'System Engineering Team'),
    ('Status',   'Technical Reference Document'),
]
for label, value in meta_lines:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r1 = p.add_run(f'{label}:  ')
    set_font(r1, bold=True, size=12, color=(31, 73, 125))
    r2 = p.add_run(value)
    set_font(r2, size=12)

doc.add_page_break()

# ════════════════════════════════════════════════════════════════════════════
# TABLE OF CONTENTS (manual)
# ════════════════════════════════════════════════════════════════════════════
heading('Table of Contents', level=1)
toc_entries = [
    ('1.', 'Executive Summary'),
    ('2.', 'System Architecture Overview'),
    ('3.', 'RTSP Protocol — Technical Background'),
    ('4.', 'Hikvision Camera Configuration'),
    ('5.', 'Server-Side RTSP Integration (rtsp_camera.py)'),
    ('6.', 'AI Model Pipeline'),
    ('7.', 'Flask API — Camera Endpoints'),
    ('8.', 'Front-End Integration (main.js)'),
    ('9.', 'Network Setup — LAN / Hikvision Switch'),
    ('10.', 'Configuration Reference'),
    ('11.', 'Data Flow Diagram'),
    ('12.', 'Troubleshooting'),
    ('13.', 'Security Considerations'),
    ('14.', 'Conclusion'),
]
for num, title in toc_entries:
    p = doc.add_paragraph()
    r1 = p.add_run(f'  {num}  ')
    set_font(r1, bold=True, size=11, color=(31, 73, 125))
    r2 = p.add_run(title)
    set_font(r2, size=11)

doc.add_page_break()

# ════════════════════════════════════════════════════════════════════════════
# 1. EXECUTIVE SUMMARY
# ════════════════════════════════════════════════════════════════════════════
heading('1. Executive Summary')
body(
    'This document describes the technical integration of Hikvision IP cameras '
    'with an AI-powered facial recognition and access control system. The system '
    'uses the Real-Time Streaming Protocol (RTSP) to pull live video from up to four '
    'Hikvision cameras over a Local Area Network (LAN) managed by a Hikvision PoE '
    'switch. Each video stream is processed server-side by a Python/Flask application '
    'that runs InsightFace deep learning models for face detection and a CatBoost '
    'classifier for identity recognition. Detection results are streamed back to a '
    'web dashboard in real time via MJPEG and REST API polling.'
)
body(
    'The architecture is entirely server-side: the browser does not access the cameras '
    'directly. This means Hikvision cameras work identically to any other RTSP-capable '
    'IP camera — no proprietary SDK or plugin is required.'
)
divider()

# ════════════════════════════════════════════════════════════════════════════
# 2. SYSTEM ARCHITECTURE
# ════════════════════════════════════════════════════════════════════════════
heading('2. System Architecture Overview')
body(
    'The system is composed of three logical layers that communicate over the LAN:'
)
bullet('Physical Layer — Hikvision IP cameras connected to a Hikvision PoE switch.')
bullet('Server Layer — A Python/Flask application running on a host machine on the same LAN. '
       'It reads RTSP streams, runs AI inference, and exposes HTTP endpoints.')
bullet('Client Layer — A web browser (on any device on the LAN) that displays the live '
       'MJPEG video feed and receives detection results via REST API polling.')

doc.add_paragraph()
body('Component summary:')
add_table(
    ['Component', 'Technology', 'Role'],
    [
        ['Hikvision IP Camera',    'H.264/H.265 RTSP',          'Captures and streams live video'],
        ['Hikvision PoE Switch',   'IEEE 802.3af/at',            'Provides power and LAN connectivity'],
        ['RTSP Reader',            'OpenCV + FFmpeg (TCP)',       'Connects to each camera, decodes frames'],
        ['Face Detector',          'InsightFace (SCRFD ONNX)',   'Detects face bounding boxes per frame'],
        ['Face Embedder',          'InsightFace (ArcFace ONNX)', 'Extracts 512-dim face embeddings'],
        ['Anomaly Filter',         'IsolationForest (sklearn)',  'Rejects unknown/unseen embeddings'],
        ['Identity Classifier',    'CatBoost',                   'Maps embedding to known person name'],
        ['MJPEG Stream Server',    'Flask + multipart HTTP',     'Sends annotated JPEG frames to browser'],
        ['Web Dashboard',          'HTML/CSS/JavaScript',        'Displays live feeds and detection events'],
        ['Database',               'SQLite',                     'Stores recognition records and incidents'],
    ],
    col_widths=[2.0, 2.0, 3.0]
)
divider()

# ════════════════════════════════════════════════════════════════════════════
# 3. RTSP PROTOCOL
# ════════════════════════════════════════════════════════════════════════════
heading('3. RTSP Protocol — Technical Background')
body(
    'Real-Time Streaming Protocol (RTSP) is an application-layer protocol defined in '
    'RFC 2326 (and updated by RFC 7826). It is the de-facto standard for streaming '
    'video from IP surveillance cameras. RTSP itself is a control protocol — it '
    'establishes, pauses, and tears down media sessions. The actual video data is '
    'carried by the Real-time Transport Protocol (RTP), typically encapsulated over '
    'either UDP or TCP.'
)

heading('3.1 RTSP URL Format', level=2)
body('Hikvision cameras expose RTSP streams at a well-known URL pattern:')
code_block('rtsp://<username>:<password>@<camera_ip>:<port>/Streaming/Channels/<channel>')
body('Where:')
add_table(
    ['Parameter', 'Typical Value', 'Description'],
    [
        ['username',  'admin',          'Camera login username'],
        ['password',  'admin123',       'Camera login password'],
        ['camera_ip', '192.168.1.64',   'Camera IP on the LAN'],
        ['port',      '554',            'Standard RTSP port'],
        ['channel',   '101',            'Main stream (1080p / 2MP+)'],
        ['channel',   '102',            'Sub-stream (640×480, lower bitrate)'],
    ],
    col_widths=[1.5, 1.8, 3.7]
)

heading('3.2 TCP vs UDP Transport', level=2)
body(
    'RTSP/RTP can operate over UDP (lower latency, possible packet loss) or TCP '
    '(reliable, slightly higher latency). For a controlled LAN environment with a '
    'managed Hikvision switch, TCP is preferred and configured as the default in '
    'this system. TCP transport avoids frame corruption caused by dropped UDP packets '
    'and works correctly through managed switches that may block multicast UDP.'
)
code_block('RTSP_TRANSPORT = "tcp"  # set in config.py or via env var RTSP_TRANSPORT')
divider()

# ════════════════════════════════════════════════════════════════════════════
# 4. HIKVISION CAMERA CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
heading('4. Hikvision Camera Configuration')
body(
    'Before the system can connect to a Hikvision camera, the following must be '
    'configured on the camera itself via its web interface (http://<camera_ip>) '
    'or the Hikvision iVMS-4200 / SADP tool:'
)

heading('4.1 Network Settings', level=2)
bullet('Set a static IP address on the same subnet as the server (e.g. 192.168.1.x).')
bullet('Ensure the subnet mask is 255.255.255.0 and the gateway points to the switch/router.')
bullet('Confirm RTSP port is 554 (default) — check under Configuration → Network → Advanced → Other.')

heading('4.2 Stream Settings', level=2)
bullet('Main Stream (Channel 101): H.264 or H.265, 1080p @ 15–25 fps, bitrate 2–4 Mbps.')
bullet('Sub-Stream (Channel 102): H.264, 640×480 @ 10–15 fps, bitrate 512 Kbps.')
bullet('For AI inference, the sub-stream (102) reduces server CPU load significantly '
       'while still providing sufficient resolution for face detection.')

heading('4.3 User Credentials', level=2)
body(
    'Create a dedicated operator account on the camera with Live View permission only. '
    'Avoid using the admin account in production. Store credentials as environment '
    'variables rather than in source code:'
)
code_block(
    'export HIK_USER=operator\n'
    'export HIK_PASS=SecurePassword123\n'
    'export HIK_PORT=554'
)

heading('4.4 RTSP Authentication', level=2)
body(
    'Hikvision cameras support both Basic and Digest authentication. OpenCV/FFmpeg '
    'handles this automatically when credentials are embedded in the URL. Digest '
    'authentication (the default on most Hikvision firmware) is more secure and '
    'is recommended.'
)
divider()

# ════════════════════════════════════════════════════════════════════════════
# 5. SERVER-SIDE RTSP INTEGRATION
# ════════════════════════════════════════════════════════════════════════════
heading('5. Server-Side RTSP Integration (rtsp_camera.py)')
body(
    'The file rtsp_camera.py contains all server-side camera management logic. '
    'It is structured around two classes: RTSPCamera (one instance per camera) '
    'and RTSPCameraManager (the singleton that owns all instances).'
)

heading('5.1 RTSPCamera — Per-Camera Thread Architecture', level=2)
body('Each RTSPCamera instance runs two background daemon threads:')

body('Thread 1 — Reader Thread (_reader_loop):')
bullet('Opens an OpenCV VideoCapture object pointed at the RTSP URL using the FFmpeg backend.')
bullet('Sets the OpenCV buffer size to 1 frame to always retrieve the freshest image.')
bullet('Reads frames in a tight loop and stores the latest frame in a thread-safe buffer.')
bullet('Simultaneously encodes each frame as a JPEG and stores it for MJPEG streaming.')
bullet('On read failure, increments a failure counter. After RTSP_MAX_FAILURES consecutive '
       'failures (default 10), closes the capture and waits RTSP_RECONNECT_DELAY seconds '
       '(default 5) before reconnecting automatically.')

doc.add_paragraph()
body('Thread 2 — Inference Thread (_inference_loop):')
bullet('Sleeps for RTSP_INFERENCE_INTERVAL seconds (default 0.2 s = 5 fps) between iterations.')
bullet('Reads the latest frame from the thread-safe buffer.')
bullet('Passes the frame through the AI pipeline (see Section 6).')
bullet('Stores the resulting detection list in a thread-safe detections buffer.')
bullet('Re-encodes the frame as JPEG with bounding boxes and labels drawn on it.')

heading('5.2 OpenCV / FFmpeg RTSP Connection', level=2)
body('The RTSP connection is opened with the following FFmpeg options:')
code_block(
    "os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (\n"
    "    'rtsp_transport;tcp|'       # use TCP for reliable delivery\n"
    "    'buffer_size;4194304|'      # 4 MB socket buffer for high-bitrate streams\n"
    "    'stimeout;5000000'          # 5 second socket timeout\n"
    ")\n"
    "cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)"
)

heading('5.3 RTSPCameraManager — Singleton', level=2)
body(
    'RTSPCameraManager is the single access point for all camera operations. It is '
    'instantiated once at module level and imported by app.py:'
)
code_block('from rtsp_camera import camera_manager')
body('Key methods:')
add_table(
    ['Method', 'Description'],
    [
        ['camera_manager.start()',                  'Start all enabled RTSP cameras at app boot'],
        ['camera_manager.get_jpeg(cam_id)',         'Return latest annotated JPEG bytes for a camera'],
        ['camera_manager.get_detections(cam_id)',   'Return latest detection list for a camera'],
        ['camera_manager.get_status()',             'Return connection status of all cameras'],
        ['camera_manager.start_camera(cam_id)',     'Start a specific camera'],
        ['camera_manager.stop_camera(cam_id)',      'Stop a specific camera'],
        ['camera_manager.update_camera_url(id, url)', 'Hot-swap RTSP URL without app restart'],
        ['camera_manager.mjpeg_generator(cam_id)', 'Generator yielding multipart MJPEG frames'],
    ],
    col_widths=[3.0, 4.0]
)
divider()

# ════════════════════════════════════════════════════════════════════════════
# 6. AI MODEL PIPELINE
# ════════════════════════════════════════════════════════════════════════════
heading('6. AI Model Pipeline')
body(
    'Every frame processed by the inference thread passes through a three-stage '
    'AI pipeline. All models run on CPU (CoreML acceleration available on Apple '
    'Silicon via ONNXRuntime).'
)

heading('6.1 Stage 1 — Face Detection (InsightFace SCRFD)', level=2)
body(
    'The InsightFace library runs the SCRFD (Sample and Computation Redistribution '
    'Face Detector) model — a lightweight anchor-based detector. For each input frame '
    'it returns a list of face objects, each containing:'
)
bullet('bbox — bounding box coordinates [x1, y1, x2, y2] in pixels.')
bullet('det_score — detection confidence score (0–1).')
bullet('kps — 5-point facial keypoints (eyes, nose, mouth corners).')
bullet('embedding — 512-dimensional ArcFace embedding (from the recognition model).')

body(
    'The detection model runs at 640×640 input resolution (det_size). If more than one '
    'face appears in a frame, all are processed independently in a loop.'
)

heading('6.2 Stage 2 — Anomaly Filtering (IsolationForest)', level=2)
body(
    'Before classifying a face embedding, it is passed through an IsolationForest '
    'model trained on all known face embeddings. IsolationForest is an unsupervised '
    'anomaly detection algorithm. If the embedding falls outside the distribution of '
    'known faces (predict returns -1), the face is immediately labelled "unknown" '
    'and classification is skipped. This prevents the classifier from making '
    'overconfident wrong predictions on unseen individuals.'
)

heading('6.3 Stage 3 — Identity Classification (CatBoost)', level=2)
body(
    'If the IsolationForest accepts the embedding as in-distribution, it is passed '
    'to a CatBoost gradient boosting classifier. CatBoost predicts the class index '
    '(person ID), which is decoded by a LabelEncoder to the person\'s name. The '
    'predict_proba output gives a confidence score. If confidence is below the '
    'CONFIDENCE_THRESHOLD (default 0.9), the result is labelled "processing" '
    '(uncertain) rather than "recognized".'
)

doc.add_paragraph()
body('Status codes returned by the pipeline:')
add_table(
    ['Status', 'Colour', 'Meaning'],
    [
        ['recognized',  'Green  (#34c924)', 'Known person, confidence ≥ 0.9'],
        ['processing',  'Yellow (#ffc107)', 'Known person, confidence < 0.9'],
        ['unknown',     'Red    (#dc3545)', 'IsolationForest rejected — person not in database'],
    ],
    col_widths=[1.5, 2.0, 3.5]
)

heading('6.4 Model Files', level=2)
add_table(
    ['File', 'Model Type', 'Purpose'],
    [
        ['~/.insightface/models/buffalo_l/det_10g.onnx',    'SCRFD-10G',    'Face detection'],
        ['~/.insightface/models/buffalo_l/w600k_r50.onnx',  'ArcFace R50',  'Face embedding (512-dim)'],
        ['Models/isolation_forest.pkl',                      'IsolationForest', 'Anomaly filtering'],
        ['Models/catboost_model.pkl',                        'CatBoost',     'Identity classification'],
        ['Models/label_encoder.pkl',                         'LabelEncoder', 'Class index → person name'],
    ],
    col_widths=[3.2, 1.8, 2.0]
)
divider()

# ════════════════════════════════════════════════════════════════════════════
# 7. FLASK API ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════
heading('7. Flask API — Camera Endpoints')
body(
    'The Flask application (app.py) exposes the following REST endpoints for '
    'camera management and video streaming. All endpoints are served on port 5000.'
)
add_table(
    ['Method', 'Endpoint', 'Description'],
    [
        ['GET',  '/api/cameras/config',                'Returns the camera config list (credentials redacted)'],
        ['GET',  '/api/cameras/status',                'Returns live connection status for all cameras'],
        ['GET',  '/api/cameras/stream/<id>',           'MJPEG stream — continuous annotated JPEG feed'],
        ['GET',  '/api/cameras/<id>/snapshot',         'Single JPEG snapshot from a camera'],
        ['GET',  '/api/cameras/<id>/detections',       'Latest AI detection results for a camera (JSON)'],
        ['POST', '/api/cameras/<id>/start',            'Start (or restart) a specific camera'],
        ['POST', '/api/cameras/<id>/stop',             'Stop a specific camera stream'],
        ['POST', '/api/cameras/<id>/configure',        'Hot-update RTSP URL without restarting app'],
        ['POST', '/getdata/',                          'Single-frame inference (used by webcam mode)'],
    ],
    col_widths=[0.8, 2.8, 3.4]
)

heading('7.1 MJPEG Streaming Endpoint', level=2)
body(
    'The /api/cameras/stream/<id> endpoint returns a multipart/x-mixed-replace HTTP '
    'response — the standard format for MJPEG streaming. The browser receives a '
    'continuous sequence of JPEG frames separated by a MIME boundary. This approach '
    'requires no WebSocket, WebRTC, or browser plugin, and works in all modern browsers '
    'using a plain HTML <img> tag:'
)
code_block('<img src="/api/cameras/stream/0" style="width:100%;">')

heading('7.2 Detection Polling Endpoint', level=2)
body(
    'The /api/cameras/<id>/detections endpoint returns the most recent detection list '
    'produced by the inference thread. The front-end polls this every 200 ms to get '
    'bounding box coordinates, labels, confidence scores, and status for each detected '
    'face. Example response:'
)
code_block(
    '[\n'
    '  {\n'
    '    "bbox":       [142.0, 88.0, 310.0, 290.0],\n'
    '    "status":     "recognized",\n'
    '    "label":      "John Smith",\n'
    '    "confidence": 0.97,\n'
    '    "timestamp":  "2026-09-01 14:32:15",\n'
    '    "color":      "#34c924",\n'
    '    "track_id":   1,\n'
    '    "cam_id":     "CAM-01"\n'
    '  }\n'
    ']'
)
divider()

# ════════════════════════════════════════════════════════════════════════════
# 8. FRONT-END INTEGRATION
# ════════════════════════════════════════════════════════════════════════════
heading('8. Front-End Integration (main.js)')
body(
    'The web dashboard (templates/index.html) uses a unified JavaScript camera system '
    'that automatically selects the correct mode for each camera based on the server '
    'configuration fetched from /api/cameras/config.'
)

heading('8.1 RTSPCameraSystem', level=2)
body('Used when stream_type is "rtsp" in the config. Behaviour:')
numbered('On connect, sets the <img class="rtsp-feed"> src to /api/cameras/stream/<id>.')
numbered('The browser\'s built-in HTTP client handles the multipart MJPEG stream continuously.')
numbered('A setInterval timer polls /api/cameras/<id>/detections every 200 ms.')
numbered('Detection results are drawn as bounding boxes on a <canvas> overlay.')
numbered('A separate 5-second status check polls /api/cameras/status to update the LIVE/RECONNECTING badge.')

heading('8.2 WebcamCameraSystem', level=2)
body('Used when stream_type is "webcam". Behaviour:')
numbered('Opens the local USB/built-in webcam via navigator.mediaDevices.getUserMedia.')
numbered('Captures frames at 10 fps, encodes them as base64 JPEG.')
numbered('POSTs each frame to /getdata/ and receives detection results.')
numbered('Draws bounding boxes on the canvas overlay.')

heading('8.3 Detection Sidebar', level=2)
body(
    'Both camera modes feed into a shared detection history sidebar. Each new detection '
    'creates an entry showing the person\'s name, camera ID, timestamp, status badge '
    '(RECOGNIZED / UNKNOWN / PROCESSING), and confidence percentage. The sidebar keeps '
    'the most recent 80 entries. Global counters for total detections, unknown faces, '
    'and recognized faces are updated in the KPI cards at the top of the dashboard.'
)
divider()

# ════════════════════════════════════════════════════════════════════════════
# 9. NETWORK SETUP
# ════════════════════════════════════════════════════════════════════════════
heading('9. Network Setup — LAN / Hikvision Switch')
body(
    'The system requires the server machine and all cameras to be on the same IP subnet '
    'connected through the Hikvision managed/PoE switch.'
)

heading('9.1 Recommended Network Topology', level=2)
add_table(
    ['Device', 'Role', 'Example IP', 'Notes'],
    [
        ['Hikvision Switch',  'Layer-2 switch / PoE',    '192.168.1.1',  'Gateway / management IP'],
        ['Server / PC',       'Flask application host',  '192.168.1.10', 'Must be on same subnet'],
        ['Camera 1 (CAM-01)', 'Main entrance',           '192.168.1.64', 'Static IP recommended'],
        ['Camera 2 (CAM-02)', 'Side entrance',           '192.168.1.65', 'Static IP recommended'],
        ['Camera 3 (CAM-03)', 'Parking area',            '192.168.1.66', 'Static IP recommended'],
        ['Camera 4 (CAM-04)', 'Back exit',               '192.168.1.67', 'Static IP recommended'],
        ['Client browser',    'Dashboard viewer',        '192.168.1.x',  'Any device on LAN'],
    ],
    col_widths=[1.8, 1.8, 1.6, 2.0]
)

heading('9.2 Switch Configuration', level=2)
bullet('Enable RSTP (Rapid Spanning Tree Protocol) on the switch to prevent loops.')
bullet('Assign camera ports to a dedicated VLAN (e.g. VLAN 10) for security isolation.')
bullet('Enable PoE on camera ports (802.3af for standard cameras, 802.3at for PTZ).')
bullet('Set port bandwidth to at least 10 Mbps per camera for H.264 main stream at 4 Mbps.')
bullet('Disable unused ports.')

heading('9.3 Firewall / OS', level=2)
bullet('Ensure TCP port 554 is not blocked on the server\'s firewall.')
bullet('Ensure TCP port 5000 is accessible from client browser machines.')
bullet('On macOS: System Settings → Network → Firewall — allow Python/Flask.')

heading('9.4 Connectivity Test', level=2)
body('Before starting the application, verify camera reachability:')
code_block(
    '# Ping test\n'
    'ping 192.168.1.64\n\n'
    '# RTSP stream test with FFmpeg\n'
    'ffmpeg -rtsp_transport tcp \\\n'
    '       -i "rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101" \\\n'
    '       -frames:v 1 snapshot.jpg\n\n'
    '# Check if snapshot.jpg was created\n'
    'ls -lh snapshot.jpg'
)
divider()

# ════════════════════════════════════════════════════════════════════════════
# 10. CONFIGURATION REFERENCE
# ════════════════════════════════════════════════════════════════════════════
heading('10. Configuration Reference')
body(
    'All camera and stream settings are centralised in config.py. '
    'Sensitive values (credentials) should be set as environment variables.'
)

add_table(
    ['Parameter', 'Default', 'Env Variable', 'Description'],
    [
        ['HIK_USER',               'admin',   'HIK_USER',               'RTSP username for all cameras'],
        ['HIK_PASS',               'admin123','HIK_PASS',               'RTSP password for all cameras'],
        ['HIK_PORT',               '554',     'HIK_PORT',               'RTSP port'],
        ['RTSP_TRANSPORT',         'tcp',     'RTSP_TRANSPORT',         'RTP transport: tcp or udp'],
        ['RTSP_RECONNECT_DELAY',   '5',       'RTSP_RECONNECT_DELAY',   'Seconds before reconnect attempt'],
        ['RTSP_MAX_FAILURES',      '10',      'RTSP_MAX_FAILURES',      'Consecutive failures before reconnect'],
        ['MJPEG_QUALITY',          '75',      'MJPEG_QUALITY',          'JPEG quality for MJPEG stream (1–100)'],
        ['RTSP_INFERENCE_INTERVAL','0.2',     'RTSP_INFERENCE_INTERVAL','Seconds between AI inference calls'],
        ['CONFIDENCE_THRESHOLD',   '0.9',     '—',                      'Min confidence for "recognized" status'],
    ],
    col_widths=[2.0, 1.0, 2.0, 2.5]
)
divider()

# ════════════════════════════════════════════════════════════════════════════
# 11. DATA FLOW DIAGRAM
# ════════════════════════════════════════════════════════════════════════════
heading('11. Data Flow Diagram')
body('The complete data flow from camera to dashboard is described below:')
code_block(
    '┌─────────────────────────────────────────────────────────────────────────┐\n'
    '│                        HIKVISION LAN NETWORK                           │\n'
    '│                                                                         │\n'
    '│  ┌──────────┐  RTSP/TCP  ┌─────────────────────────────────────────┐   │\n'
    '│  │ CAM-01   ├───────────►│          Flask Server (Python)          │   │\n'
    '│  │ CAM-02   ├───────────►│                                         │   │\n'
    '│  │ CAM-03   ├───────────►│  RTSPCamera (per camera):               │   │\n'
    '│  │ CAM-04   ├───────────►│   Thread 1: cv2.VideoCapture (FFmpeg)   │   │\n'
    '│  └──────────┘            │             ↓ raw BGR frame             │   │\n'
    '│  (Hikvision PoE Switch)  │             ↓ JPEG encode               │   │\n'
    '│                          │             ↓ store in frame buffer      │   │\n'
    '│                          │                                         │   │\n'
    '│                          │   Thread 2: Inference loop (5 fps)      │   │\n'
    '│                          │     ↓ InsightFace SCRFD → detect faces  │   │\n'
    '│                          │     ↓ ArcFace R50 → 512-dim embedding   │   │\n'
    '│                          │     ↓ IsolationForest → filter unknown  │   │\n'
    '│                          │     ↓ CatBoost → classify identity      │   │\n'
    '│                          │     ↓ draw boxes → annotated JPEG       │   │\n'
    '│                          │     ↓ save to SQLite (if recognized)    │   │\n'
    '│                          │                                         │   │\n'
    '│                          │  Flask Routes:                          │   │\n'
    '│                          │   /api/cameras/stream/<id>  → MJPEG     │   │\n'
    '│                          │   /api/cameras/<id>/detections → JSON   │   │\n'
    '│                          └──────────────┬──────────────────────────┘   │\n'
    '│                                         │ HTTP (port 5000)             │\n'
    '│                          ┌──────────────▼──────────────────────────┐   │\n'
    '│                          │         Browser Dashboard               │   │\n'
    '│                          │  <img src="/stream/0"> → MJPEG display  │   │\n'
    '│                          │  fetch("/detections/0") → draw canvas   │   │\n'
    '│                          │  Detection sidebar + KPI counters        │   │\n'
    '│                          └─────────────────────────────────────────┘   │\n'
    '└─────────────────────────────────────────────────────────────────────────┘'
)
divider()

# ════════════════════════════════════════════════════════════════════════════
# 12. TROUBLESHOOTING
# ════════════════════════════════════════════════════════════════════════════
heading('12. Troubleshooting')
add_table(
    ['Symptom', 'Likely Cause', 'Resolution'],
    [
        ['"Network is unreachable" in logs',
         'Server and cameras on different subnets',
         'Set server IP to same subnet as cameras (e.g. 192.168.1.x)'],
        ['"VideoCapture failed to open"',
         'Wrong IP, port, or credentials',
         'Test with ffmpeg CLI; verify HIK_USER, HIK_PASS, IP in config.py'],
        ['Camera shows "RECONNECTING" badge',
         'Camera unreachable or rebooting',
         'System will auto-reconnect every 5 s; check camera power and cable'],
        ['Low detection accuracy',
         'Sub-stream resolution too low, or model not retrained',
         'Switch to main stream (channel 101); retrain model via Add People page'],
        ['High server CPU usage',
         'Main stream at 1080p on all 4 cameras',
         'Use sub-stream (channel 102); reduce RTSP_INFERENCE_INTERVAL'],
        ['Browser shows blank camera slots',
         'JS config fetch failed or camera disabled',
         'Open DevTools console; check /api/cameras/config response'],
        ['JPEG snapshot is grey "NO SIGNAL"',
         'Camera thread has not received a frame yet',
         'Wait a few seconds after app start; check camera reachability'],
        ['sklearn InconsistentVersionWarning',
         'Models trained on older sklearn version',
         'Retrain models: /addpeople → Train Model button'],
    ],
    col_widths=[2.2, 2.2, 2.6]
)
divider()

# ════════════════════════════════════════════════════════════════════════════
# 13. SECURITY CONSIDERATIONS
# ════════════════════════════════════════════════════════════════════════════
heading('13. Security Considerations')
bullet('Never store RTSP credentials in source code. Use environment variables: '
       'HIK_USER, HIK_PASS.')
bullet('The /api/cameras/configure endpoint allows changing RTSP URLs at runtime — '
       'restrict access to trusted network clients or add authentication middleware.')
bullet('The Flask development server (debug=True) should not be used in production. '
       'Use Gunicorn or uWSGI behind Nginx.')
bullet('Place cameras on a dedicated VLAN isolated from the general office network.')
bullet('Use HTTPS for the Flask dashboard in production (TLS termination at Nginx).')
bullet('Ensure the Hikvision switch firmware is up to date to patch known CVEs.')
bullet('Disable the Hikvision camera\'s default admin account after creating a '
       'dedicated operator account.')
divider()

# ════════════════════════════════════════════════════════════════════════════
# 14. CONCLUSION
# ════════════════════════════════════════════════════════════════════════════
heading('14. Conclusion')
body(
    'This document has described the complete integration path from Hikvision IP cameras '
    'over RTSP to the AI face recognition pipeline and web dashboard. The key design '
    'decision — performing all video processing server-side — makes the system camera-agnostic: '
    'any RTSP-capable camera can be substituted for or added to the Hikvision units by '
    'updating the CAMERAS list in config.py.'
)
body(
    'The two-thread architecture (reader + inference) per camera ensures the MJPEG stream '
    'remains smooth regardless of AI inference latency, since frame capture and AI processing '
    'are fully decoupled. Automatic reconnection means the system recovers from camera '
    'reboots, network blips, or switch port resets without any manual intervention.'
)
body(
    'Once the LAN is configured with the server and cameras on the same subnet, '
    'the system is fully operational: cameras connect automatically at app start, '
    'live annotated feeds appear in the browser dashboard, and all recognition events '
    'are logged to the SQLite database.'
)

doc.add_paragraph()
divider()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run(f'— End of Document — Generated {datetime.date.today().strftime("%d %B %Y")} —')
set_font(run, italic=True, size=9, color=(128, 128, 128))

# ── Save ────────────────────────────────────────────────────────────────────
output_path = 'RTSP_Hikvision_AI_Integration_Report.docx'
doc.save(output_path)
print(f'Report saved: {output_path}')
