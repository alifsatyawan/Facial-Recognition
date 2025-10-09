from typing import List, Tuple, Optional

import cv2
import numpy as np
import os

try:
    import dlib  # Optional
    _DLIB_AVAILABLE = True
except Exception:
    _DLIB_AVAILABLE = False

from facenet_pytorch import MTCNN
from .alignment import FaceAligner


class FaceDetector:
    """Face detector with optional dlib and fallback to MTCNN."""

    def __init__(self, shape_predictor_path: Optional[str] = None, align_faces: bool = False):
        """
        Initialize face detector.

        Args:
            shape_predictor_path: Path to dlib shape predictor model (for alignment)
            align_faces: Whether to align faces using landmarks
        """
        self._use_dlib = _DLIB_AVAILABLE
        self._align_faces = align_faces and _DLIB_AVAILABLE and shape_predictor_path
        self._aligner = None
        self._mtcnn = None

        if not self._use_dlib:
            self._mtcnn = MTCNN(keep_all=True, device='cpu', post_process=False)
        else:
            # HOG-based detector is lightweight
            self._dlib_detector = dlib.get_frontal_face_detector()

            # Initialize aligner if requested and shape predictor exists
            if self._align_faces and os.path.exists(shape_predictor_path):
                try:
                    self._aligner = FaceAligner(shape_predictor_path)
                except Exception as e:
                    print(f"Warning: Failed to initialize face aligner: {e}")
                    self._align_faces = False

    def detect_faces(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Return list of boxes (x1, y1, x2, y2) in pixel coords."""
        if self._use_dlib:
            try:
                # Ensure input frame is contiguous before any operations
                frame_bgr = np.ascontiguousarray(frame_bgr, dtype=np.uint8)
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                # Ensure gray is also contiguous for dlib compatibility
                if gray.ndim != 2:
                    print(f"Error: Gray image has {gray.ndim} dimensions, expected 2")
                    raise RuntimeError("Invalid grayscale dimensions")
                gray = np.ascontiguousarray(gray, dtype=np.uint8)

                rects = self._dlib_detector(gray, 0)
                boxes = []
                for r in rects:
                    boxes.append((r.left(), r.top(), r.right(), r.bottom()))
                return boxes
            except RuntimeError as e:
                # Dlib failed, fall back to MTCNN
                if self._mtcnn is None:
                    print(f"Warning: Dlib failed ({e}), initializing MTCNN fallback...")
                    self._mtcnn = MTCNN(keep_all=True, device='cpu', post_process=False)
                    self._use_dlib = False  # Permanently switch to MTCNN
                    print("Switched to MTCNN detector for future detections")
                # Fall through to MTCNN detection below

        # MTCNN detection
        if self._mtcnn is None:
            self._mtcnn = MTCNN(keep_all=True, device='cpu', post_process=False)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        boxes, _ = self._mtcnn.detect(rgb)
        results: List[Tuple[int, int, int, int]] = []
        if boxes is None:
            return results
        h, w = frame_bgr.shape[:2]
        for b in boxes:
            x1, y1, x2, y2 = [int(max(0, min(v, w if i % 2 == 0 else h))) for i, v in enumerate(b)]
            results.append((x1, y1, x2, y2))
        return results
    
    def detect_and_align_faces(self, frame_bgr: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[np.ndarray]]:
        """
        Detect faces and optionally align them.
        
        Args:
            frame_bgr: Input frame in BGR format
            
        Returns:
            Tuple of (boxes, aligned_faces)
            - boxes: List of (x1, y1, x2, y2) tuples
            - aligned_faces: List of aligned face images (empty if alignment disabled)
        """
        if self._use_dlib and self._align_faces and self._aligner:
            # Use dlib for detection and alignment
            # Ensure input frame is contiguous before any operations
            frame_bgr = np.ascontiguousarray(frame_bgr, dtype=np.uint8)

            # Convert to RGB as dlib expects RGB format
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Workaround for numpy/dlib compatibility issue
            # Ensure contiguous array for dlib compatibility
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                print(f"Error: RGB image has shape {rgb.shape}, expected (H, W, 3)")
                boxes = self.detect_faces(frame_bgr)
                return boxes, []

            rgb_copy = np.ascontiguousarray(rgb, dtype=np.uint8)

            # Debug: check array properties
            if not rgb_copy.flags['C_CONTIGUOUS']:
                print("Warning: RGB array is not C-contiguous after conversion")
            if rgb_copy.dtype != np.uint8:
                print(f"Warning: RGB dtype is {rgb_copy.dtype}, expected uint8")

            try:
                dlib_rects = self._dlib_detector(rgb_copy, 0)
            except RuntimeError as e:
                # Fallback to regular detection without alignment
                print(f"Warning: Dlib detection failed ({e}), falling back to MTCNN")
                # Disable alignment permanently and switch to MTCNN
                self._align_faces = False
                self._use_dlib = False
                if self._mtcnn is None:
                    self._mtcnn = MTCNN(keep_all=True, device='cpu', post_process=False)
                print("Disabled dlib alignment, using MTCNN for all future detections")
                boxes = self.detect_faces(frame_bgr)
                return boxes, []
            
            boxes = []
            aligned_faces = []
            
            for rect in dlib_rects:
                # Get bounding box
                x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
                boxes.append((x1, y1, x2, y2))
                
                # Align face
                try:
                    landmarks = self._aligner.get_landmarks(rgb_copy, rect)
                    aligned = self._aligner.align_face(frame_bgr, landmarks)
                    aligned_faces.append(aligned)
                except Exception as e:
                    # Fallback to cropped face
                    face = frame_bgr[y1:y2, x1:x2]
                    if face.size > 0:  # Check if face is not empty
                        face = cv2.resize(face, (self._aligner.desired_size, self._aligner.desired_size))
                        aligned_faces.append(face)
            
            return boxes, aligned_faces
        else:
            # No alignment, just detection
            boxes = self.detect_faces(frame_bgr)
            return boxes, []


