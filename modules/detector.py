from typing import List, Tuple

import cv2
import numpy as np

try:
    import dlib  # Optional
    _DLIB_AVAILABLE = True
except Exception:
    _DLIB_AVAILABLE = False

from facenet_pytorch import MTCNN


class FaceDetector:
    """Face detector with optional dlib and fallback to MTCNN."""

    def __init__(self):
        self._use_dlib = _DLIB_AVAILABLE
        self._mtcnn = None
        if not self._use_dlib:
            self._mtcnn = MTCNN(keep_all=True, device='cpu', post_process=False)
        else:
            # HOG-based detector is lightweight
            self._dlib_detector = dlib.get_frontal_face_detector()

    def detect_faces(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Return list of boxes (x1, y1, x2, y2) in pixel coords."""
        if self._use_dlib:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            rects = self._dlib_detector(gray, 0)
            boxes = []
            for r in rects:
                boxes.append((r.left(), r.top(), r.right(), r.bottom()))
            return boxes
        else:
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


