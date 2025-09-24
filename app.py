import os
import threading
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, send_from_directory
from flask_cors import CORS
# import app as ap
from flask import Flask, request, jsonify
import numpy as np

# start deepsort

from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker

# end deepsort
import base64
from datetime import datetime
from face_analyzer import FaceAnalyzer
from classifier import FaceClassifier, retrain_model
from config import STATUS_COLORS
from utils import rgb_to_hex
from database import (
    init_db,
    save_face_record,
    get_records,
    get_kpi_counts,
    get_peak_hours,
    get_recent_incidents,
    save_incident,
    get_all_incidents
)
import cv2
from werkzeug.utils import secure_filename

# Configuration
DATASET_DIR = './facedata'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MIN_IMAGES_REQUIRED = 10

tracked_faces = {}
next_face_id = 0


app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET', 'a-very-secret-key')
CORS(app)

# Initialize database and models
init_db()
analyzer = FaceAnalyzer()
classifier = FaceClassifier()

# Thread-safe retraining
retraining_lock = threading.Lock()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def background_retrain():
    if retraining_lock.locked():
        return

    with retraining_lock:
        try:
            flash("Starting retraining...","info")
            retrain_model(analyzer)
            print("Retraining complete")
            flash("Starting retraining...","info")

        except Exception as e:
            print(f"Retraining failed: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/records')
def records_page():
    try:
        range_filter = request.args.get('range', 'all')
        records = get_records(time_range=range_filter)
        kpis = get_kpi_counts(records)
        peak_hours = get_peak_hours(range_filter)
        recent_incidents = get_recent_incidents(limit=5)

        return render_template('records.html',
                               records=records,
                               kpis=kpis,
                               peak_hours=peak_hours,
                               recent_incidents=recent_incidents,
                               active_range=range_filter,
                               timestamp=datetime.now().timestamp()
                               )
    except Exception as e:
        print(f"Error: {e}")
        return f"Server Error {e}", 500

@app.route('/api/people')
def get_people():
    people = []
    if not os.path.exists(DATASET_DIR):
        return jsonify([])

    for name in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, name)
        if not os.path.isdir(person_dir):
            continue

        images = [f for f in os.listdir(person_dir) if allowed_file(f)]
        if images:
            avatar_path = url_for('serve_facedata', filename=f'{name}/{images[0]}')
            people.append({
                'id': name,
                'name': name,
                'avatar': avatar_path
            })

    return jsonify(people)

@app.route('/static/facedata/<path:filename>')
def serve_facedata(filename):
    return send_from_directory(DATASET_DIR, filename)

@app.route('/api/people/<name>', methods=['DELETE'])
def delete_person(name):
    person_dir = os.path.join(DATASET_DIR, secure_filename(name))
    if os.path.exists(person_dir):
        try:
            import shutil
            shutil.rmtree(person_dir)
            return jsonify({'success': True})
        except Exception as e:
            print(f"Delete failed: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    return jsonify({'success': False, 'error': 'Person not found'}), 404



@app.route('/addpeople', methods=['GET', 'POST'])
def add_people():
    name = request.form.get('name', '').strip() or request.args.get('name', '').strip()
    existing_count = 0

    if name:
        person_dir = os.path.join(DATASET_DIR, secure_filename(name))
        if os.path.exists(person_dir):
            existing_count = len([f for f in os.listdir(person_dir) if os.path.isfile(os.path.join(person_dir, f))])

    if request.method == 'POST':
        files = request.files.getlist('photos')
        action = request.form.get('action')  # Get the button action

        if not name:
            flash('Enter a name', 'danger')
            return redirect(request.url)

        if not files or all(f.filename == '' for f in files):
            flash('Select at least one photo', 'danger')
            return redirect(request.url)

        person_dir = os.path.join(DATASET_DIR, secure_filename(name))
        os.makedirs(person_dir, exist_ok=True)

        saved = 0
        for f in files:
            if allowed_file(f.filename):
                filename = secure_filename(f.filename)
                f.save(os.path.join(person_dir, filename))
                saved += 1

        total_images = existing_count + saved
        flash(f'Saved {saved} photos. Total: {total_images}', 'success')

        # Trigger training if "Train Model" was clicked
        if action == 'train':
            # threading.Thread(target=background_retrain).start()
            retrain_model(analyzer)
            flash('Model training started in the background.', 'info')

        return redirect(url_for('add_people', name=name))

    return render_template(
        'addpeople.html',
        MIN_IMAGES_REQUIRED=MIN_IMAGES_REQUIRED,
        existing_count=existing_count,
        name=name
    )


@app.route('/incidents')
def view_incidents():
    try:
        all_incidents = get_all_incidents()
        return render_template('incidents.html', incidents=all_incidents)
    except Exception as e:
        print(f"Error: {e}")
        return "Server Error", 500

@app.route('/incidents/add', methods=['POST'])
def add_incident():
    data = request.get_json(force=True)
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    save_incident(
        name=data['name'],
        status=data['status'],
        description=data.get('description', ''),
        timestamp=ts
    )

    return jsonify({
        'success': True,
        'name': data['name'],
        'status': data['status'].capitalize(),
        'description': data.get('description', ''),
        'timestamp': ts
    })


# /////////////////////////////




# Global tracker state
tracker = None
metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.5)
track_info = {}
MIN_RECOGNITION_COUNT = 3
MIN_CONFIDENCE = 0.6


@app.route('/getdata/', methods=['POST'])
def get_data():
    global tracker, track_info

    # Initialize tracker if not exists
    if tracker is None:
        tracker = Tracker(metric)

    data = request.get_json() or {}
    image_b64 = data.get('image', '')

    try:
        # Decode base64 image
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
        image_data = base64.b64decode(image_b64)
        arr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify(error=f"Invalid image data: {str(e)}"), 400

    # Detect faces
    faces = analyzer.analyze(frame)

    # Create DeepSORT detections
    detections = []
    for face in faces:
        x1, y1, x2, y2 = face.bbox
        w, h = x2 - x1, y2 - y1
        detections.append(Detection(
            [x1, y1, w, h],
            1.0,  # Detection confidence
            face.embedding  # Appearance features
        ))

    # Update tracker
    tracker.predict()
    tracker.update(detections)

    results = []
    active_tracks = set()

    # Process each active track
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue

        track_id = track.track_id
        active_tracks.add(track_id)

        # Initialize track info if new
        if track_id not in track_info:
            track_info[track_id] = {
                'recognized_count': 0,
                'unrecognized_count': 0,
                'label': None,
                'confidence': 0.0,
                'last_feature': None,
                'status': 'unknown',
                'last_updated': datetime.now()
            }

        info = track_info[track_id]
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        current_feature = None

        # Find matching detection for this track
        for detection in detections:
            # Use bounding box distance to match track to detection
            track_bbox = track.to_tlbr()
            det_bbox = detection.to_tlbr()
            iou = bb_intersection_over_union(track_bbox, det_bbox)

            if iou > 0.5:  # Consider it a match if IOU > 50%
                current_feature = detection.feature
                info['last_feature'] = current_feature
                break

        # If no current feature, use last stored feature
        if current_feature is None and info['last_feature'] is not None:
            current_feature = info['last_feature']

        # Only attempt recognition if we have features and haven't reached recognition count
        if current_feature is not None and info['recognized_count'] < MIN_RECOGNITION_COUNT:
            status, label, conf = classifier.classify(current_feature)

            if status == 'recognized' and conf > MIN_CONFIDENCE:
                info['recognized_count'] += 1
                info['unrecognized_count'] = 0
                info['label'] = label
                info['confidence'] = conf
                info['status'] = 'recognized'
                save_face_record(label, conf, track.to_tlbr().tolist(), current_feature, timestamp)
            else:
                info['unrecognized_count'] += 1
                info['status'] = 'unknown'

                # Stop recognizing after 3 unrecognized attempts
                if info['unrecognized_count'] >= MIN_RECOGNITION_COUNT:
                    info['status'] = 'unknown_permanent'
        elif info['recognized_count'] >= MIN_RECOGNITION_COUNT:
            # Use stored recognition info
            status = 'recognized'
            label = info['label']
            conf = info['confidence']
            info['status'] = 'recognized_permanent'
        else:
            # Not enough recognition attempts
            status = info['status']
            label = info['label']
            conf = info['confidence']

        # Prepare result
        results.append({
            'bbox': track.to_tlbr().tolist(),
            'status': status,
            'label': label,
            'confidence': conf,
            'timestamp': timestamp,
            'color': rgb_to_hex(STATUS_COLORS.get(status, (255, 255, 0))),
            'track_id': track_id
        })

    # Cleanup tracks that are no longer active
    for track_id in list(track_info.keys()):
        if track_id not in active_tracks:
            # Only remove if not permanently recognized
            if track_info[track_id].get('status') not in ['recognized_permanent', 'recognized']:
                del track_info[track_id]

    return jsonify(results)


def bb_intersection_over_union(boxA, boxB):
    # Calculate Intersection over Union (IOU) of two bounding boxes
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# //////////////////////////////////////////////////////

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)