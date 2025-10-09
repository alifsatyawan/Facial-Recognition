#!/usr/bin/env python3
"""Test script for face alignment functionality."""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.detector import FaceDetector
from modules.alignment import download_shape_predictor
import config


def test_alignment():
    """Test face alignment with a sample image or webcam."""
    print("Testing Face Alignment...")
    
    # Check if shape predictor exists
    if not config.SHAPE_PREDICTOR_PATH.exists():
        print(f"Shape predictor not found at {config.SHAPE_PREDICTOR_PATH}")
        print("Would you like to download it? (y/n): ", end="")
        if input().lower() == 'y':
            config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            success = download_shape_predictor(str(config.SHAPE_PREDICTOR_PATH))
            if not success:
                print("Failed to download shape predictor")
                return
        else:
            print("Alignment requires shape predictor. Exiting.")
            return
    
    # Initialize detectors
    print("\nInitializing detectors...")
    detector_no_align = FaceDetector(align_faces=False)
    detector_with_align = FaceDetector(
        shape_predictor_path=str(config.SHAPE_PREDICTOR_PATH),
        align_faces=True
    )
    
    # Test with webcam
    print("\nTesting with webcam (press 'q' to quit)...")
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Failed to open camera")
        return
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Create side-by-side display
            h, w = frame.shape[:2]
            display = np.zeros((h, w * 2, 3), dtype=np.uint8)
            
            # Left side: Original detection
            boxes = detector_no_align.detect_faces(frame)
            left_frame = frame.copy()
            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(left_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(left_frame, "Original", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            display[:, :w] = left_frame
            
            # Right side: Aligned faces
            right_frame = frame.copy()
            if hasattr(detector_with_align, 'detect_and_align_faces'):
                boxes, aligned_faces = detector_with_align.detect_and_align_faces(frame)
                
                # Draw boxes
                for (x1, y1, x2, y2) in boxes:
                    cv2.rectangle(right_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Show aligned faces in corner
                if aligned_faces:
                    for i, aligned in enumerate(aligned_faces[:2]):  # Show max 2
                        y_offset = 10 + i * 170
                        x_offset = w - 170
                        if y_offset + 160 <= h:
                            right_frame[y_offset:y_offset+160, x_offset:x_offset+160] = aligned
                            cv2.rectangle(right_frame, (x_offset-2, y_offset-2), 
                                        (x_offset+162, y_offset+162), (255, 255, 0), 2)
            
            cv2.putText(right_frame, "With Alignment", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            display[:, w:] = right_frame
            
            cv2.imshow("Face Alignment Test", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print("\nAlignment test completed!")


if __name__ == "__main__":
    test_alignment()
