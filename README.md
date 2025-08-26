### Facial Recognition (POC)

Real-time facial recognition using OpenCV, FaceNet embeddings, and an SVM classifier. Dlib is supported if installed for detection/alignment, with an automatic fallback to MTCNN (from `facenet-pytorch`). Target performance: realtime on CPU (≥15 FPS depends on hardware).

### Features
- **Face detection & alignment**: Dlib (if available) or MTCNN fallback
- **Embeddings**: FaceNet `InceptionResnetV1` (VGGFace2)
- **Classifier**: Scikit-learn SVM with probability output
- **CLI**: dataset capture, embedding building, training, and realtime recognition

### Quickstart
1) Create environment and install deps
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2) Initialize folders
```bash
python cli.py init
```

3) Capture some samples for a person (press "c" to capture, "q" to quit)
```bash
python cli.py capture --name "Alice" --num 30
```

4) Build embeddings and train SVM
```bash
python cli.py build-embeddings
python cli.py train
```

5) Run realtime recognition
```bash
python cli.py realtime
```

### Data layout
- `data/<person_name>/*.jpg`
- `artifacts/embeddings.npz` (embeddings, labels)
- `artifacts/label_map.json` (index → label mapping)
- `artifacts/svm.pkl` (trained classifier)

### Dlib (optional)
This POC auto-falls back to MTCNN. If you want Dlib-based detection/alignment:
- Install dlib (macOS often requires brew/conda; wheels may not exist):
  - macOS: `brew install cmake boost` then build from source or use conda
  - Linux: `sudo apt-get install cmake` then `pip install dlib` (may still require build deps)
- Download landmarks model (if you use dlib alignment):
  - `shape_predictor_68_face_landmarks.dat` from the dlib model zoo
  - Configure its path via `--shape-predictor` in the CLI

### Notes
- The first run will download FaceNet weights automatically.
- Performance depends on CPU/GPU and camera resolution. Lowering frame size can boost FPS.


