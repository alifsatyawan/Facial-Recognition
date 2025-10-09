"""
Flask application with SocketIO for facial recognition web interface.
"""

import os
import sys
import base64
import json
import numpy as np
from pathlib import Path
from datetime import datetime

from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2

# Force eventlet to use gevent instead (Python 3.12 compatibility)
import os
os.environ['FLASK_SOCKETIO_ASYNC_MODE'] = 'threading'

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from modules.detector import FaceDetector
from modules.embedder import FaceEmbedder
from modules.classifier import SVMClassifier
from modules.dataset import DatasetManager
import config

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global instances
detector = None
embedder = None
classifier = None
dataset_manager = None

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Settings
current_settings = {
    "use_alignment": True,
    "threshold": 0.6,
    "performance_mode": False,
    "show_fps": True
}

# Recognition state
recognition_state = {
    "is_running": False,
    "fps": 0,
    "frame_count": 0,
    "last_time": datetime.now()
}


def init_components():
    """Initialize face detection and embedding components."""
    global detector, embedder, dataset_manager
    
    # Check for shape predictor
    shape_predictor_path = None
    if current_settings["use_alignment"]:
        if config.SHAPE_PREDICTOR_PATH.exists():
            shape_predictor_path = str(config.SHAPE_PREDICTOR_PATH)
    
    detector = FaceDetector(
        shape_predictor_path=shape_predictor_path,
        align_faces=current_settings["use_alignment"]
    )
    embedder = FaceEmbedder()
    dataset_manager = DatasetManager(DATA_DIR)


@app.route('/', methods=['GET'])
def index():
    """Root endpoint."""
    return jsonify({
        "message": "Facial Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "status": "/api/status",
            "persons": "/api/persons",
            "capture": "/api/capture",
            "train": "/api/train",
            "settings": "/api/settings"
        }
    })


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status."""
    return jsonify({
        "status": "ok",
        "settings": current_settings,
        "recognition_running": recognition_state["is_running"],
        "fps": recognition_state["fps"]
    })


@app.route('/api/persons', methods=['GET'])
def get_persons():
    """Get list of all persons in the dataset."""
    persons = dataset_manager.list_persons()
    return jsonify({
        "persons": [{"name": name, "count": count} for name, count in persons]
    })


@app.route('/api/person/<name>', methods=['DELETE'])
def delete_person(name):
    """Delete a person from the dataset."""
    person_dir = DATA_DIR / name
    if person_dir.exists():
        import shutil
        shutil.rmtree(person_dir)
        return jsonify({"success": True, "message": f"Deleted {name}"})
    return jsonify({"success": False, "message": "Person not found"}), 404


@app.route('/api/capture', methods=['POST'])
def capture_face():
    """Capture and save a face image."""
    data = request.json
    name = data.get('name')
    image_data = data.get('image')
    
    if not name or not image_data:
        return jsonify({"success": False, "message": "Missing name or image"}), 400
    
    # Decode base64 image
    image_bytes = base64.b64decode(image_data.split(',')[1])
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Detect faces
    if current_settings["use_alignment"] and hasattr(detector, 'detect_and_align_faces'):
        boxes, aligned_faces = detector.detect_and_align_faces(img)
        if aligned_faces:
            face_img = aligned_faces[0]
        elif boxes:
            x1, y1, x2, y2 = boxes[0]
            face_img = img[y1:y2, x1:x2]
        else:
            return jsonify({"success": False, "message": "No face detected"}), 400
    else:
        boxes = detector.detect_faces(img)
        if boxes:
            x1, y1, x2, y2 = boxes[0]
            face_img = img[y1:y2, x1:x2]
        else:
            return jsonify({"success": False, "message": "No face detected"}), 400
    
    # Save face
    person_dir = DATA_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = int(datetime.now().timestamp() * 1000)
    img_path = person_dir / f"{timestamp}.jpg"
    cv2.imwrite(str(img_path), face_img)
    
    return jsonify({"success": True, "message": f"Captured face for {name}"})


@app.route('/api/train', methods=['POST'])
def train_model():
    """Train the recognition model."""
    try:
        # Index dataset
        image_paths, labels, label_to_index = dataset_manager.index_dataset()
        
        if len(image_paths) == 0:
            return jsonify({"success": False, "message": "No training data found"}), 400
        
        # Send progress updates
        socketio.emit('training_progress', {
            'step': 'Indexing', 
            'progress': 0.2,
            'message': f'Found {len(image_paths)} images'
        })
        
        # Build embeddings
        embeddings = embedder.embed_paths(image_paths, batch_size=32)
        embeddings = np.asarray(embeddings)
        
        socketio.emit('training_progress', {
            'step': 'Building embeddings', 
            'progress': 0.6,
            'message': 'Generated embeddings'
        })
        
        # Save embeddings
        label_indices = np.array([label_to_index[l] for l in labels], dtype=np.int64)
        np.savez(ARTIFACTS_DIR / "embeddings.npz", x=embeddings, y=label_indices)
        
        with open(ARTIFACTS_DIR / "label_map.json", "w") as f:
            json.dump({str(v): k for k, v in label_to_index.items()}, f, indent=2)
        
        # Train classifier if multiple classes
        if len(label_to_index) >= 2:
            global classifier
            classifier = SVMClassifier()
            classifier.train(embeddings, label_indices, c=1.0)
            classifier.save(ARTIFACTS_DIR / "svm.pkl")
            
            socketio.emit('training_progress', {
                'step': 'Training classifier', 
                'progress': 0.9,
                'message': 'Trained SVM classifier'
            })
        
        socketio.emit('training_progress', {
            'step': 'Complete', 
            'progress': 1.0,
            'message': 'Training completed successfully'
        })
        
        return jsonify({"success": True, "message": "Model trained successfully"})
        
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update recognition settings."""
    data = request.json
    current_settings.update(data)
    
    # Reinitialize detector if alignment setting changed
    if 'use_alignment' in data:
        detector._align_faces = data['use_alignment']
    
    return jsonify({"success": True, "settings": current_settings})


@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print('Client connected')
    emit('connected', {'data': 'Connected to server'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print('Client disconnected')
    recognition_state["is_running"] = False


@socketio.on('start_recognition')
def handle_start_recognition():
    """Start recognition mode."""
    global classifier
    
    # Load classifier if needed
    if classifier is None:
        clf_path = ARTIFACTS_DIR / "svm.pkl"
        label_path = ARTIFACTS_DIR / "label_map.json"
        
        if clf_path.exists() and label_path.exists():
            classifier = SVMClassifier()
            classifier.load(clf_path)
            
            with open(label_path) as f:
                label_map = json.load(f)
        else:
            # Single person mode
            label_map = {}
    
    recognition_state["is_running"] = True
    emit('recognition_started', {'status': 'started'})


@socketio.on('stop_recognition')
def handle_stop_recognition():
    """Stop recognition mode."""
    recognition_state["is_running"] = False
    emit('recognition_stopped', {'status': 'stopped'})


@socketio.on('video_frame')
def handle_video_frame(data):
    """Process video frame for face recognition."""
    if not recognition_state["is_running"]:
        return
    
    # Decode frame
    image_data = data['image']
    image_bytes = base64.b64decode(image_data.split(',')[1])
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Update FPS
    recognition_state["frame_count"] += 1
    current_time = datetime.now()
    time_diff = (current_time - recognition_state["last_time"]).total_seconds()
    
    if time_diff > 1.0:
        recognition_state["fps"] = recognition_state["frame_count"] / time_diff
        recognition_state["frame_count"] = 0
        recognition_state["last_time"] = current_time
    
    # Detect faces
    if current_settings["use_alignment"] and hasattr(detector, 'detect_and_align_faces'):
        boxes, aligned_faces = detector.detect_and_align_faces(frame)
    else:
        boxes = detector.detect_faces(frame)
        aligned_faces = []
    
    # Recognize faces
    results = []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        face = aligned_faces[i] if i < len(aligned_faces) else frame[y1:y2, x1:x2]
        
        if face.size == 0:
            continue
        
        # Get embedding
        embedding = embedder.embed_images([face])[0]
        embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
        
        # Recognize
        if classifier is not None:
            prob, pred = classifier.predict_proba([embedding])
            # Get label from label_map
            label_map_path = ARTIFACTS_DIR / "label_map.json"
            if label_map_path.exists():
                with open(label_map_path) as f:
                    label_map = json.load(f)
                label = label_map.get(str(pred[0]), "unknown")
                confidence = float(prob[0, pred[0]]) if prob is not None else 0.0
            else:
                label = "unknown"
                confidence = 0.0
        else:
            label = "no_model"
            confidence = 0.0
        
        results.append({
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "label": label,
            "confidence": confidence,
            "threshold_met": confidence >= current_settings["threshold"]
        })
    
    # Send results back
    emit('detection_result', {
        "faces": results,
        "fps": recognition_state["fps"],
        "timestamp": current_time.isoformat()
    })


if __name__ == '__main__':
    # Initialize components
    init_components()
    
    # Ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run the app
    print("Starting Flask server on http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
