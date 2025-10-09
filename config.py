"""Configuration for facial recognition system."""

import os
import pathlib

# Base paths
BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = BASE_DIR / "models"

# Dlib model paths
SHAPE_PREDICTOR_PATH = MODELS_DIR / "shape_predictor_68_face_landmarks.dat"
SHAPE_PREDICTOR_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

# Face alignment parameters
FACE_ALIGNMENT_SIZE = 160  # Output size for aligned faces (matches FaceNet input)
LEFT_EYE_CENTER = (0.35, 0.35)  # Normalized position for left eye in aligned image
RIGHT_EYE_CENTER = (0.65, 0.35)  # Normalized position for right eye in aligned image
FACE_PADDING = 0.2  # Padding around face for alignment (20%)

# Detection parameters
MIN_FACE_SIZE = 20  # Minimum face size in pixels
DETECTION_CONFIDENCE = 0.7  # Minimum confidence for face detection

# Embedding parameters
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_DIMENSION = 512

# Recognition parameters
DEFAULT_SIMILARITY_THRESHOLD = 0.6
DEFAULT_SVM_C = 1.0
