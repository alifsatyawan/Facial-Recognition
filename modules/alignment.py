"""Face alignment utilities using dlib landmarks."""

from typing import Tuple, Optional, List
import numpy as np
import cv2
import math

try:
    import dlib
    _DLIB_AVAILABLE = True
except ImportError:
    _DLIB_AVAILABLE = False


class FaceAligner:
    """Aligns faces using dlib's 68-point landmarks."""
    
    def __init__(self, shape_predictor_path: str, desired_size: int = 160,
                 left_eye_center: Tuple[float, float] = (0.35, 0.35),
                 right_eye_center: Tuple[float, float] = (0.65, 0.35)):
        """
        Initialize face aligner.
        
        Args:
            shape_predictor_path: Path to dlib's shape predictor model
            desired_size: Output size for aligned face
            left_eye_center: Normalized position for left eye (0-1)
            right_eye_center: Normalized position for right eye (0-1)
        """
        if not _DLIB_AVAILABLE:
            raise ImportError("dlib is required for face alignment. Install with: pip install dlib")
        
        self.predictor = dlib.shape_predictor(shape_predictor_path)
        self.desired_size = desired_size
        self.left_eye_center = left_eye_center
        self.right_eye_center = right_eye_center
        
        # Convert normalized positions to pixel coordinates
        self.desired_left_eye = (int(left_eye_center[0] * desired_size),
                                 int(left_eye_center[1] * desired_size))
        self.desired_right_eye = (int(right_eye_center[0] * desired_size),
                                  int(right_eye_center[1] * desired_size))
    
    def get_landmarks(self, image: np.ndarray, face_rect) -> np.ndarray:
        """
        Get 68 facial landmarks.
        
        Args:
            image: Grayscale image
            face_rect: dlib rectangle for face
            
        Returns:
            Array of shape (68, 2) with landmark coordinates
        """
        shape = self.predictor(image, face_rect)
        landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
        return landmarks
    
    def get_eye_centers(self, landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate eye centers from landmarks.
        
        Args:
            landmarks: 68 facial landmarks
            
        Returns:
            Tuple of (left_eye_center, right_eye_center)
        """
        # Left eye landmarks: 36-41, Right eye landmarks: 42-47
        left_eye_indices = list(range(36, 42))
        right_eye_indices = list(range(42, 48))
        
        left_eye_center = landmarks[left_eye_indices].mean(axis=0)
        right_eye_center = landmarks[right_eye_indices].mean(axis=0)
        
        return left_eye_center, right_eye_center
    
    def align_face(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Align face using eye positions.
        
        Args:
            image: Input image (BGR)
            landmarks: 68 facial landmarks
            
        Returns:
            Aligned face image
        """
        # Get eye centers
        left_eye, right_eye = self.get_eye_centers(landmarks)
        
        # Calculate angle between eyes
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = math.degrees(math.atan2(dy, dx))
        
        # Calculate scale
        dist = np.linalg.norm(right_eye - left_eye)
        desired_dist = self.desired_right_eye[0] - self.desired_left_eye[0]
        scale = desired_dist / dist
        
        # Get rotation matrix centered at midpoint between eyes
        eyes_center = ((left_eye[0] + right_eye[0]) // 2,
                       (left_eye[1] + right_eye[1]) // 2)
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
        
        # Update translation to position eyes correctly
        tx = self.desired_size * 0.5 - eyes_center[0]
        ty = self.desired_left_eye[1] - eyes_center[1]
        M[0, 2] += tx
        M[1, 2] += ty
        
        # Apply transformation
        aligned = cv2.warpAffine(image, M, (self.desired_size, self.desired_size),
                                 flags=cv2.INTER_CUBIC)
        
        return aligned
    
    def align_faces_from_rects(self, image: np.ndarray, 
                              face_rects: List) -> List[np.ndarray]:
        """
        Align multiple faces from dlib rectangles.
        
        Args:
            image: Input image (BGR)
            face_rects: List of dlib rectangles
            
        Returns:
            List of aligned face images
        """
        # Convert to RGB for dlib
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Ensure compatibility
        rgb_copy = np.array(rgb, dtype=np.uint8, order='C')
        
        aligned_faces = []
        
        for rect in face_rects:
            try:
                landmarks = self.get_landmarks(rgb_copy, rect)
                aligned = self.align_face(image, landmarks)
                aligned_faces.append(aligned)
            except Exception as e:
                # If alignment fails, crop the original face
                x1, y1 = rect.left(), rect.top()
                x2, y2 = rect.right(), rect.bottom()
                face = image[y1:y2, x1:x2]
                # Resize to desired size
                face = cv2.resize(face, (self.desired_size, self.desired_size))
                aligned_faces.append(face)
        
        return aligned_faces


def download_shape_predictor(target_path: str) -> bool:
    """
    Download dlib's shape predictor model.
    
    Args:
        target_path: Where to save the model
        
    Returns:
        True if successful
    """
    import urllib.request
    import bz2
    import os
    import ssl
    import certifi
    
    # Fix SSL certificate issue
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    bz2_path = target_path + ".bz2"
    
    print(f"Downloading shape predictor from {url}...")
    try:
        # Create request with SSL context
        request = urllib.request.Request(url)
        with urllib.request.urlopen(request, context=ssl_context) as response:
            with open(bz2_path, 'wb') as f:
                f.write(response.read())
        
        print("Extracting...")
        with bz2.BZ2File(bz2_path, 'rb') as f_in:
            with open(target_path, 'wb') as f_out:
                f_out.write(f_in.read())
        
        os.remove(bz2_path)
        print(f"Shape predictor saved to {target_path}")
        return True
    except Exception as e:
        print(f"Failed to download shape predictor: {e}")
        return False
