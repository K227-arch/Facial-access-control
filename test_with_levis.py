import pickle
import cv2
import requests
import time
from datetime import datetime
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo

# Configuration
CONFIDENCE_THRESHOLD = 0.6
SERVER_URL = "http://127.0.0.1:8000/getdata/"
COOLDOWN_SEC = 1  # Minimum seconds between reports for same face
CAMERA_ID = 1

# Status colors for visualization
STATUS_COLORS = {
    "recognized": (52, 201, 36),  # Green
    "unknown": (220, 53, 69),  # Red
    "processing": (255, 193, 7)  # Yellow
}


class FaceRecognitionSystem:
    def __init__(self):
        self.last_sent = {}  # Track last sent time per person
        self.load_models()
        self.initialize_face_analysis()

    def load_models(self):
        """Load all required ML models"""
        print("Loading models...")
        with open('./FaceRecognitionModels/isolation_forest.pkl', 'rb') as f:
            self.iso_forest = pickle.load(f)
        with open('./FaceRecognitionModels/svm_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        with open('./FaceRecognitionModels/label_encoder.pkl', 'rb') as f:
            self.le = pickle.load(f)

    def initialize_face_analysis(self):
        """Initialize face detection and recognition models"""
        print("Initializing face analysis...")
        det_path = "./FaceRecognitionModels/antelopev2/detection/scrfd_10g_bnkps.onnx"
        rec_path = "./FaceRecognitionModels/antelopev2/recognition/glintr100.onnx"

        self.det_model = model_zoo.get_model(det_path, task='detection')
        self.rec_model = model_zoo.get_model(rec_path, task='recognition')

        self.app = FaceAnalysis()
        self.app.det_model = self.det_model
        self.app.rec_model = self.rec_model
        self.app.models = {'detection': self.det_model, 'recognition': self.rec_model}
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

    def classify_face(self, embedding):
        """Classify a face embedding and trigger server communication"""
        is_outlier = self.iso_forest.predict([embedding])[0]

        if is_outlier == -1:
            self.send_detection("Unknown", 0.0, "unknown")
            return "unknown", "Unknown", 0.0

        pred = self.model.predict([embedding])[0]
        label = self.le.inverse_transform([pred])[0]
        confidence = float(self.model.predict_proba([embedding]).max())

        if confidence >= CONFIDENCE_THRESHOLD:
            self.send_detection(label, confidence, "recognized")
            return "recognized", label, confidence

        return "processing", "Processing...", confidence

    def send_detection(self, name, confidence, status):
        """Send detection data to server with cooldown checks"""
        current_time = time.time()

        # Skip if sent recently
        if name in self.last_sent and (current_time - self.last_sent[name]) < COOLDOWN_SEC:
            return False

        data = {
            "name": name,
            "time": datetime.now().strftime("%X"),
            "camera": CAMERA_ID,
            "confidence": confidence,
            "status": status
        }

        try:
            response = requests.post(SERVER_URL, json=data, timeout=3)
            if response.status_code == 200:
                self.last_sent[name] = current_time
                print(f"Sent data: {data}")
                return True
        except requests.exceptions.RequestException as e:
            print(f"Server error: {e}")

        return False

    def process_frame(self, frame):
        """Process a single frame and return results"""
        faces = self.app.get(frame)
        results = []

        for face in faces:
            bbox = face.bbox.astype(int)
            status, label, conf = self.classify_face(face.embedding)

            results.append({
                'bbox': bbox,
                'status': status,
                'label': label,
                'confidence': conf,
                'time': datetime.now().strftime('%X')
            })

        return results

    def draw_results(self, frame, results):
        """Visualize results on the frame"""
        for res in results:
            x1, y1, x2, y2 = res['bbox']
            status = res['status']
            color = STATUS_COLORS.get(status, (0, 255, 255))
            label = res['label']
            conf = res['confidence']

            text = f"{label.capitalize()} {conf * 100:.1f}%" if status == "recognized" or status == "processing" else label

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def run(self):
        """Main loop for real-time processing"""
        cap = cv2.VideoCapture(0)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                results = self.process_frame(frame)

                # Visualize results
                self.draw_results(frame, results)
                cv2.imshow('Face Recognition', frame)

                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    fr_system = FaceRecognitionSystem()
    fr_system.run()