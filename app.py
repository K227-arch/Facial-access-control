import os
import threading
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, send_from_directory
from flask_cors import CORS
import numpy as np
import base64
from datetime import datetime
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

# Try to import face analyzer and classifier, handle gracefully if missing
try:
    from face_analyzer import FaceAnalyzer
    FACE_ANALYZER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Face analyzer not available: {e}")
    FACE_ANALYZER_AVAILABLE = False

try:
    from classifier import FaceClassifier, retrain_model
    CLASSIFIER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Classifier not available: {e}")
    CLASSIFIER_AVAILABLE = False

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

# Initialize models if available
analyzer = None
classifier = None

if FACE_ANALYZER_AVAILABLE:
    try:
        analyzer = FaceAnalyzer()
        print("Face analyzer initialized successfully")
    except Exception as e:
        print(f"Failed to initialize face analyzer: {e}")
        analyzer = None

if CLASSIFIER_AVAILABLE:
    try:
        classifier = FaceClassifier()
        print("Classifier initialized successfully")
    except Exception as e:
        print(f"Failed to initialize classifier: {e}")
        classifier = None

# Thread-safe retraining
retraining_lock = threading.Lock()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def background_retrain():
    if not analyzer:
        flash("Face analyzer not available for training", "error")
        return
        
    if retraining_lock.locked():
        return

    with retraining_lock:
        try:
            flash("Starting retraining...", "info")
            if CLASSIFIER_AVAILABLE:
                retrain_model(analyzer)
                print("Retraining complete")
                flash("Retraining complete", "success")
            else:
                flash("Classifier not available for training", "error")
        except Exception as e:
            print(f"Retraining failed: {e}")
            flash(f"Retraining failed: {e}", "error")

@app.route('/')
def home():
    return render_template('index.html', current_page='home')

@app.route('/camera-test')
def camera_test():
    return render_template('camera_test.html', current_page='camera-test')

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
                               timestamp=datetime.now().timestamp(),
                               current_page='records'
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
            if analyzer and CLASSIFIER_AVAILABLE:
                try:
                    retrain_model(analyzer)
                    flash('Model training completed successfully.', 'success')
                except Exception as e:
                    flash(f'Model training failed: {e}', 'error')
            else:
                flash('Training not available - missing dependencies.', 'warning')

        return redirect(url_for('add_people', name=name))

    return render_template(
        'addpeople.html',
        MIN_IMAGES_REQUIRED=MIN_IMAGES_REQUIRED,
        existing_count=existing_count,
        name=name,
        current_page='addpeople'
    )


@app.route('/incidents')
def view_incidents():
    try:
        all_incidents = get_all_incidents()
        return render_template('incidents.html', incidents=all_incidents, current_page='incidents')
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


@app.route('/search')
def search_page():
    """Search page for finding face records"""
    return render_template('search.html', current_page='search')


@app.route('/calendar')
def calendar_page():
    """Calendar page for viewing detection history"""
    return render_template('calendar.html', current_page='calendar')


@app.route('/api/search')
def search_records():
    """API endpoint for searching face records"""
    query = request.args.get('q', '').strip()
    date_from = request.args.get('from', '')
    date_to = request.args.get('to', '')
    
    # Basic search implementation
    records = get_records('all')
    
    if query:
        records = [r for r in records if query.lower() in r['Name'].lower()]
    
    return jsonify(records)


# /////////////////////////////




@app.route('/getdata/', methods=['POST'])
def get_data():
    """Simple face detection endpoint without tracking"""
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

    results = []
    
    if not analyzer:
        # Return a demo result if analyzer is not available
        return jsonify([{
            'bbox': [100, 100, 200, 200],
            'status': 'unknown',
            'label': 'Demo Mode',
            'confidence': 0.0,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'color': rgb_to_hex(STATUS_COLORS.get('unknown', (255, 255, 0))),
            'track_id': 1
        }])

    try:
        # Detect faces using the analyzer
        faces = analyzer.analyze(frame)
        
        for i, face in enumerate(faces):
            # Convert NumPy types to Python native types for JSON serialization
            x1, y1, x2, y2 = face.bbox
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
            
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Try to classify if classifier is available
            if classifier:
                try:
                    status, label, conf = classifier.classify(face.embedding)
                    if status == 'recognized':
                        save_face_record(label, float(conf), [x1, y1, x2, y2], face.embedding, timestamp)
                except Exception as e:
                    print(f"Classification error: {e}")
                    status, label, conf = 'unknown', 'Unknown', 0.0
            else:
                status, label, conf = 'unknown', 'No Classifier', 0.0
            
            results.append({
                'bbox': [x1, y1, x2, y2],
                'status': status,
                'label': label,
                'confidence': float(conf),
                'timestamp': timestamp,
                'color': rgb_to_hex(STATUS_COLORS.get(status, (255, 255, 0))),
                'track_id': i + 1
            })
            
    except Exception as e:
        print(f"Face detection error: {e}")
        return jsonify(error=f"Face detection failed: {str(e)}"), 500

    return jsonify(results)

# //////////////////////////////////////////////////////

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)