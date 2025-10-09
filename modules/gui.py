"""GUI components for facial recognition system using OpenCV."""

import os
import cv2
import numpy as np
import time
import pathlib
from typing import Optional, List, Tuple, Dict, Callable
from dataclasses import dataclass
from enum import Enum

from .detector import FaceDetector
from .embedder import FaceEmbedder
from .classifier import SVMClassifier
from .dataset import DatasetManager
from .performance import FrameOptimizer, optimize_camera_settings


class GUIState(Enum):
    """States for the GUI application."""
    MAIN_MENU = "main_menu"
    CAPTURE = "capture"
    RECOGNITION = "recognition"
    TRAINING = "training"
    GALLERY = "gallery"
    IMPORT = "import"


@dataclass
class Button:
    """Simple button for OpenCV GUI."""
    x: int
    y: int
    width: int
    height: int
    text: str
    color: Tuple[int, int, int] = (100, 100, 100)
    text_color: Tuple[int, int, int] = (255, 255, 255)
    hover_color: Tuple[int, int, int] = (150, 150, 150)
    
    def contains(self, x: int, y: int) -> bool:
        """Check if point is inside button."""
        return (self.x <= x <= self.x + self.width and 
                self.y <= y <= self.y + self.height)
    
    def draw(self, img: np.ndarray, hover: bool = False) -> None:
        """Draw button on image."""
        color = self.hover_color if hover else self.color
        cv2.rectangle(img, (self.x, self.y), 
                     (self.x + self.width, self.y + self.height),
                     color, -1)
        cv2.rectangle(img, (self.x, self.y), 
                     (self.x + self.width, self.y + self.height),
                     (50, 50, 50), 2)
        
        # Center text
        text_size = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        text_x = self.x + (self.width - text_size[0]) // 2
        text_y = self.y + (self.height + text_size[1]) // 2
        cv2.putText(img, self.text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.text_color, 1)


class MainMenu:
    """Main menu interface for facial recognition system."""
    
    def __init__(self, width: int = 800, height: int = 700):
        self.width = width
        self.height = height
        self.window_name = "Facial Recognition System"
        self.mouse_x = 0
        self.mouse_y = 0
        self.selected_action = None

        # Create buttons
        button_width = 200
        button_height = 50
        button_spacing = 15
        start_x = (width - button_width) // 2
        start_y = 120
        
        self.buttons = {
            "capture": Button(start_x, start_y, button_width, button_height, "Capture Faces"),
            "train": Button(start_x, start_y + button_height + button_spacing,
                           button_width, button_height, "Train Model"),
            "recognize": Button(start_x, start_y + 2 * (button_height + button_spacing),
                               button_width, button_height, "Start Recognition"),
            "gallery": Button(start_x, start_y + 3 * (button_height + button_spacing),
                             button_width, button_height, "View Gallery"),
            "manage": Button(start_x, start_y + 4 * (button_height + button_spacing),
                            button_width, button_height, "Manage People",
                            color=(100, 50, 50), hover_color=(150, 70, 70)),
            "import": Button(start_x, start_y + 5 * (button_height + button_spacing),
                            button_width, button_height, "Import Images"),
            "exit": Button(start_x, start_y + 6 * (button_height + button_spacing),
                          button_width, button_height, "Exit",
                          color=(50, 50, 150), hover_color=(70, 70, 200))
        }
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        self.mouse_x = x
        self.mouse_y = y
        
        if event == cv2.EVENT_LBUTTONDOWN:
            for action, button in self.buttons.items():
                if button.contains(x, y):
                    self.selected_action = action
                    break
    
    def show(self) -> Optional[str]:
        """Display main menu and return selected action."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Force window to front
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
        
        self.selected_action = None
        
        while True:
            # Create blank image
            img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 240
            
            # Draw title
            title = "Facial Recognition System"
            title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
            title_x = (self.width - title_size[0]) // 2
            cv2.putText(img, title, (title_x, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (50, 50, 50), 2)
            
            # Draw buttons
            for button in self.buttons.values():
                hover = button.contains(self.mouse_x, self.mouse_y)
                button.draw(img, hover)
            
            # Draw instructions
            cv2.putText(img, "Click a button or press ESC to exit",
                       (10, self.height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            
            cv2.imshow(self.window_name, img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                self.selected_action = "exit"
                break
            
            if self.selected_action:
                break
        
        cv2.destroyWindow(self.window_name)
        cv2.waitKey(1)  # Process any pending GUI events
        return self.selected_action


class InteractiveCapture:
    """Interactive face capture interface."""
    
    def __init__(self, detector: FaceDetector, 
                 data_dir: pathlib.Path,
                 width: int = 800, height: int = 600):
        self.detector = detector
        self.data_dir = data_dir
        self.width = width
        self.height = height
        self.window_name = "Face Capture"
        
        # State
        self.person_name = ""
        self.capturing = False
        self.auto_capture = False
        self.target_samples = 30
        self.captured_count = 0
        self.last_capture_time = 0
        self.capture_delay = 0.5  # seconds between captures
        
        # Performance optimization
        self.frame_optimizer = FrameOptimizer(
            resize_factor=0.75,  # Process at 75% resolution
            skip_frames=1  # Process every other frame for detection
        )
        
        # UI elements
        self.buttons = {
            "toggle_auto": Button(10, 10, 120, 40, "Auto: OFF"),
            "clear": Button(140, 10, 80, 40, "Clear"),
            "done": Button(230, 10, 80, 40, "Done",
                          color=(50, 150, 50), hover_color=(70, 200, 70))
        }
        
        self.mouse_x = 0
        self.mouse_y = 0
        self.clicked = False
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        self.mouse_x = x
        self.mouse_y = y
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked = True
            
            # Check button clicks
            if self.buttons["toggle_auto"].contains(x, y):
                self.auto_capture = not self.auto_capture
                self.buttons["toggle_auto"].text = f"Auto: {'ON' if self.auto_capture else 'OFF'}"
            elif self.buttons["clear"].contains(x, y):
                self.captured_count = 0
                # Clear saved images
                if self.person_name:
                    person_dir = self.data_dir / self.person_name
                    if person_dir.exists():
                        for img_file in person_dir.glob("*.jpg"):
                            img_file.unlink()
            elif self.buttons["done"].contains(x, y):
                self.capturing = False
    
    def capture_face(self, frame: np.ndarray, face_region: Tuple[int, int, int, int],
                    aligned_face: Optional[np.ndarray] = None) -> bool:
        """Capture a face image."""
        if not self.person_name:
            return False
            
        person_dir = self.data_dir / self.person_name
        person_dir.mkdir(parents=True, exist_ok=True)
        
        # Use aligned face if available, otherwise crop from frame
        if aligned_face is not None:
            face_img = aligned_face
        else:
            x1, y1, x2, y2 = face_region
            face_img = frame[y1:y2, x1:x2]
        
        # Save image
        timestamp = int(time.time() * 1000)
        img_path = person_dir / f"{timestamp}.jpg"
        cv2.imwrite(str(img_path), face_img)
        
        self.captured_count += 1
        self.last_capture_time = time.time()
        return True
    
    def run(self, camera_index: int = 0, person_name: Optional[str] = None) -> bool:
        """Run interactive capture interface."""
        if person_name:
            self.person_name = person_name
        else:
            # Simple text input for name
            self.person_name = self._get_person_name()
            if not self.person_name:
                return False
        
        cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Failed to open camera")
            return False
        
        # Optimize camera settings for better performance
        optimize_camera_settings(cap, width=480, height=360, fps=30, buffer_size=1)
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        cv2.createTrackbar("Samples", self.window_name, self.target_samples, 100,
                          lambda x: setattr(self, 'target_samples', max(1, x)))
        
        # Force window to front on macOS
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
        
        self.capturing = True
        self.captured_count = 0
        
        try:
            while self.capturing:
                # Check if window still exists
                try:
                    if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                        print("Capture window closed by user")
                        break
                except:
                    break

                ret, frame = cap.read()
                if not ret:
                    print("Warning: Failed to read frame from camera")
                    cv2.waitKey(100)
                    continue
                
                # Keep original frame for display and capture
                original_frame = frame.copy()
                
                # Process detection on optimized frame only when needed
                if self.frame_optimizer.should_process_frame():
                    # Optimize frame for detection
                    opt_frame, scale_factor = self.frame_optimizer.optimize_frame(frame)
                    
                    # Detect faces on optimized frame
                    if hasattr(self.detector, 'detect_and_align_faces'):
                        boxes, aligned_faces = self.detector.detect_and_align_faces(opt_frame)
                    else:
                        boxes = self.detector.detect_faces(opt_frame)
                        aligned_faces = []
                    
                    # Scale boxes back to original size
                    boxes = self.frame_optimizer.scale_boxes(boxes, scale_factor)
                    
                    # Store for use in non-processing frames
                    self._last_boxes = boxes
                    self._last_aligned_faces = aligned_faces
                else:
                    # Use cached detection results
                    boxes = getattr(self, '_last_boxes', [])
                    aligned_faces = getattr(self, '_last_aligned_faces', [])
                
                # Auto capture logic
                if self.auto_capture and len(boxes) > 0 and self.captured_count < self.target_samples:
                    if time.time() - self.last_capture_time > self.capture_delay:
                        # Re-run alignment on full resolution if needed
                        if aligned_faces and hasattr(self.detector, 'detect_and_align_faces'):
                            _, hq_aligned = self.detector.detect_and_align_faces(original_frame)
                            aligned = hq_aligned[0] if hq_aligned else None
                        else:
                            aligned = None
                        self.capture_face(original_frame, boxes[0], aligned)
                
                # Manual capture on click (only if not clicking on a button)
                if self.clicked and len(boxes) > 0 and self.captured_count < self.target_samples:
                    # Check if click is on a button first
                    button_clicked = False
                    for button in self.buttons.values():
                        if button.contains(self.mouse_x, self.mouse_y):
                            button_clicked = True
                            break
                    
                    # If not on a button, check if click is on a face
                    if not button_clicked:
                        for i, (x1, y1, x2, y2) in enumerate(boxes):
                            if x1 <= self.mouse_x <= x2 and y1 <= self.mouse_y <= y2:
                                # Re-run alignment on full resolution if needed
                                if aligned_faces and i < len(aligned_faces) and hasattr(self.detector, 'detect_and_align_faces'):
                                    _, hq_aligned = self.detector.detect_and_align_faces(original_frame)
                                    aligned = hq_aligned[i] if i < len(hq_aligned) else None
                                else:
                                    aligned = None
                                self.capture_face(original_frame, boxes[i], aligned)
                                break
                
                # Reset clicked flag after processing
                self.clicked = False
                
                # Draw UI on original frame
                display = original_frame.copy()
                
                # Draw faces
                for x1, y1, x2, y2 in boxes:
                    color = (0, 255, 0) if self.captured_count < self.target_samples else (0, 165, 255)
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                
                # Draw buttons
                for button in self.buttons.values():
                    hover = button.contains(self.mouse_x, self.mouse_y)
                    button.draw(display, hover)
                
                # Draw status
                status_y = 80
                cv2.putText(display, f"Person: {self.person_name}", (10, status_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
                cv2.putText(display, f"Captured: {self.captured_count}/{self.target_samples}",
                           (10, status_y + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
                
                # Progress bar
                bar_y = status_y + 50
                bar_width = 300
                bar_height = 20
                progress = min(1.0, self.captured_count / max(1, self.target_samples))
                cv2.rectangle(display, (10, bar_y), (10 + bar_width, bar_y + bar_height),
                             (200, 200, 200), -1)
                cv2.rectangle(display, (10, bar_y), 
                             (10 + int(bar_width * progress), bar_y + bar_height),
                             (0, 255, 0), -1)
                cv2.rectangle(display, (10, bar_y), (10 + bar_width, bar_y + bar_height),
                             (100, 100, 100), 2)
                
                # Instructions
                instructions = "Click face to capture | Space: toggle auto | Q/ESC: quit | Click Done when finished"
                cv2.putText(display, instructions, (10, 460),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                
                cv2.imshow(self.window_name, display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Q or ESC
                    break
                elif key == ord(' '):  # Space
                    self.auto_capture = not self.auto_capture
                    self.buttons["toggle_auto"].text = f"Auto: {'ON' if self.auto_capture else 'OFF'}"
                
        except Exception as e:
            print(f"Error during capture: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cap.release()
            cv2.destroyWindow(self.window_name)
            cv2.waitKey(1)  # Ensure window events are processed

        return self.captured_count > 0
    
    def _get_person_name(self) -> str:
        """Simple text input for person name."""
        window_name = "Enter Person Name"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 400, 200)
        
        # Force window to front
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        
        name = ""
        
        while True:
            img = np.ones((200, 400, 3), dtype=np.uint8) * 240
            
            cv2.putText(img, "Enter person name:", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
            cv2.putText(img, name + "_", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(img, "Press ENTER to confirm, ESC to cancel", (50, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            
            cv2.imshow(window_name, img)
            
            key = cv2.waitKey(0)

            if key == 27:  # ESC
                name = ""
                break
            elif key == 13 or key == 10:  # Enter (both CR and LF)
                if name:
                    break
            elif key == 8 or key == 127:  # Backspace or Delete key
                if len(name) > 0:
                    name = name[:-1]
            elif 32 <= (key & 0xFF) <= 126:  # Printable characters
                name += chr(key & 0xFF)
        
        cv2.destroyWindow(window_name)
        return name


class InteractiveRecognition:
    """Interactive real-time recognition interface."""
    
    def __init__(self, detector: FaceDetector, embedder: FaceEmbedder,
                 width: int = 800, height: int = 600):
        self.detector = detector
        self.embedder = embedder
        self.width = width
        self.height = height
        self.window_name = "Face Recognition"
        
        # Recognition components
        self.classifier = None
        self.label_map = {}
        self.single_person_mode = False
        self.reference_embedding = None
        self.reference_name = ""
        
        # Performance tracking
        self.fps = 0
        self.frame_times = []
        self.max_frame_times = 30
        
        # Performance optimization
        self.frame_optimizer = FrameOptimizer(
            resize_factor=0.6,  # Process at 60% resolution (better balance)
            skip_frames=2  # Process every 3rd frame for detection
        )
        
        # UI state
        self.threshold = 0.6
        self.show_fps = True
        self.mouse_x = 0
        self.mouse_y = 0
        self.should_exit = False
        self.use_alignment = self.detector._align_faces  # Get initial alignment state
        
        # Buttons
        align_text = "Align: ON" if self.use_alignment else "Align: OFF"
        self.buttons = {
            "toggle_mode": Button(10, 50, 100, 35, "Mode: SVM"),
            "toggle_fps": Button(115, 50, 80, 35, "FPS: ON"),
            "toggle_align": Button(200, 50, 90, 35, align_text),
            "exit": Button(295, 50, 60, 35, "Exit",
                          color=(150, 50, 50), hover_color=(200, 70, 70))
        }
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        self.mouse_x = x
        self.mouse_y = y
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.buttons["toggle_mode"].contains(x, y):
                self.single_person_mode = not self.single_person_mode
                mode_text = "Single" if self.single_person_mode else "SVM"
                self.buttons["toggle_mode"].text = f"Mode: {mode_text}"
            elif self.buttons["toggle_fps"].contains(x, y):
                self.show_fps = not self.show_fps
                self.buttons["toggle_fps"].text = f"FPS: {'ON' if self.show_fps else 'OFF'}"
            elif self.buttons["toggle_align"].contains(x, y):
                self.use_alignment = not self.use_alignment
                self.detector._align_faces = self.use_alignment
                self.buttons["toggle_align"].text = f"Align: {'ON' if self.use_alignment else 'OFF'}"
            elif self.buttons["exit"].contains(x, y):
                self.should_exit = True
    
    def threshold_callback(self, value):
        """Handle threshold trackbar changes."""
        self.threshold = value / 100.0
    
    def load_classifier(self, artifacts_dir: pathlib.Path) -> bool:
        """Load SVM classifier and label map."""
        try:
            import json
            
            clf_path = artifacts_dir / "svm.pkl"
            label_path = artifacts_dir / "label_map.json"
            
            if clf_path.exists() and label_path.exists():
                self.classifier = SVMClassifier()
                self.classifier.load(clf_path)
                
                with open(label_path) as f:
                    self.label_map = json.load(f)
                
                return True
        except Exception as e:
            print(f"Failed to load classifier: {e}")
        
        return False
    
    def setup_single_person(self, person_name: str, data_dir: pathlib.Path) -> bool:
        """Setup single person recognition mode."""
        person_dir = data_dir / person_name
        if not person_dir.exists():
            return False
        
        # Load reference images
        ref_paths = list(person_dir.glob("*.jpg"))[:50]
        if not ref_paths:
            return False
        
        ref_images = []
        for p in ref_paths:
            img = cv2.imread(str(p))
            if img is not None:
                # Try to extract face
                if hasattr(self.detector, 'detect_and_align_faces'):
                    boxes, aligned = self.detector.detect_and_align_faces(img)
                    if aligned:
                        ref_images.append(aligned[0])
                    elif boxes:
                        x1, y1, x2, y2 = boxes[0]
                        ref_images.append(img[y1:y2, x1:x2])
                    else:
                        ref_images.append(img)
                else:
                    boxes = self.detector.detect_faces(img)
                    if boxes:
                        x1, y1, x2, y2 = boxes[0]
                        ref_images.append(img[y1:y2, x1:x2])
                    else:
                        ref_images.append(img)
        
        if not ref_images:
            return False
        
        # Compute reference embedding
        embeddings = self.embedder.embed_images(ref_images)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        self.reference_embedding = embeddings.mean(axis=0)
        self.reference_embedding = self.reference_embedding / (np.linalg.norm(self.reference_embedding) + 1e-10)
        self.reference_name = person_name
        
        return True
    
    def update_fps(self):
        """Update FPS calculation."""
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # Keep only recent frame times
        if len(self.frame_times) > self.max_frame_times:
            self.frame_times.pop(0)
        
        # Calculate FPS
        if len(self.frame_times) > 1:
            time_diff = self.frame_times[-1] - self.frame_times[0]
            if time_diff > 0:
                self.fps = (len(self.frame_times) - 1) / time_diff
    
    def run(self, camera_index: int = 0) -> None:
        """Run interactive recognition interface."""
        # Reset state
        self.should_exit = False
        self.frame_times = []
        self.embedding_cache = {}  # Cache embeddings by face position
        self.last_detection_boxes = []  # Cache detection results
        self.last_detection_faces = []  # Cache aligned faces
        self.last_mouse_pos = (0, 0)  # Track mouse for optimized drawing
        
        cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Failed to open camera")
            return
        
        # Optimize camera settings for better performance
        optimize_camera_settings(cap, width=480, height=360, fps=30, buffer_size=1)
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        cv2.createTrackbar("Threshold", self.window_name, 
                          int(self.threshold * 100), 100, self.threshold_callback)
        
        # Force window to front
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
        
        # Small delay to ensure window is ready
        cv2.waitKey(100)
        
        running = True
        
        try:
            while running:
                # Check if window still exists (user might have closed it)
                try:
                    if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                        print("Window closed by user")
                        break
                except:
                    # Window was closed
                    break

                ret, frame = cap.read()
                if not ret:
                    # Camera read failed, but don't immediately exit - try a few times
                    print("Warning: Failed to read frame from camera")
                    cv2.waitKey(100)  # Wait a bit
                    continue

                # Update FPS
                self.update_fps()

                # Only detect faces on certain frames to improve performance
                if self.frame_optimizer.should_process_frame():
                    # Optimize frame for detection
                    opt_frame, scale_factor = self.frame_optimizer.optimize_frame(frame)

                    # Detect faces on optimized frame
                    if self.use_alignment and hasattr(self.detector, 'detect_and_align_faces'):
                        boxes, aligned_faces = self.detector.detect_and_align_faces(opt_frame)
                    else:
                        boxes = self.detector.detect_faces(opt_frame)
                        aligned_faces = []

                    # Scale boxes back to original size
                    boxes = self.frame_optimizer.scale_boxes(boxes, scale_factor)

                    # Cache results for next frames
                    self.last_detection_boxes = boxes
                    self.last_detection_faces = aligned_faces
                else:
                    # Use cached detection results
                    boxes = self.last_detection_boxes
                    aligned_faces = self.last_detection_faces

                # Perform recognition
                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    # Get face region
                    if i < len(aligned_faces) and aligned_faces[i] is not None:
                        face = aligned_faces[i]
                    else:
                        face = frame[y1:y2, x1:x2]

                    if face.size == 0:
                        continue

                    # Create cache key based on face position (with tolerance of 20 pixels)
                    cache_key = f"{x1//20}_{y1//20}_{x2//20}_{y2//20}"

                    # Check if we have a cached embedding for this position
                    if cache_key in self.embedding_cache:
                        embedding = self.embedding_cache[cache_key]
                    else:
                        # Get new embedding only if needed
                        embedding = self.embedder.embed_images([face])[0]
                        embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
                        # Cache it
                        self.embedding_cache[cache_key] = embedding

                        # Limit cache size
                        if len(self.embedding_cache) > 20:
                            # Remove oldest entries
                            keys_to_remove = list(self.embedding_cache.keys())[:10]
                            for k in keys_to_remove:
                                del self.embedding_cache[k]
                    
                    # Recognize
                    if self.single_person_mode and self.reference_embedding is not None:
                        # Cosine similarity
                        similarity = float(np.dot(embedding, self.reference_embedding))
                        is_match = similarity >= self.threshold
                        label = self.reference_name if is_match else "unknown"
                        confidence = similarity
                        color = (0, 255, 0) if is_match else (0, 0, 255)
                    elif self.classifier is not None:
                        # SVM classification
                        prob, pred = self.classifier.predict_proba([embedding])
                        label_idx = str(pred[0])
                        label = self.label_map.get(label_idx, "unknown")
                        confidence = float(prob[0, pred[0]]) if prob is not None else 0.0
                        is_match = confidence >= self.threshold
                        color = (0, 255, 0) if is_match else (0, 0, 255)
                    else:
                        label = "no model"
                        confidence = 0.0
                        color = (255, 255, 0)
                    
                    # Draw results
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    text = f"{label}: {confidence:.2f}"
                    cv2.putText(frame, text, (x1, max(0, y1 - 10)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw UI elements - optimized to only check hover when mouse moves
                mouse_moved = (self.mouse_x, self.mouse_y) != self.last_mouse_pos
                self.last_mouse_pos = (self.mouse_x, self.mouse_y)

                for button in self.buttons.values():
                    hover = button.contains(self.mouse_x, self.mouse_y) if mouse_moved else False
                    button.draw(frame, hover)
                
                # Draw FPS
                if self.show_fps:
                    fps_text = f"FPS: {self.fps:.1f}"
                    frame_height, frame_width = frame.shape[:2]
                    cv2.putText(frame, fps_text, (frame_width - 150, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw threshold
                thresh_text = f"Threshold: {self.threshold:.2f}"
                cv2.putText(frame, thresh_text, (10, 480),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
                
                # Draw instructions
                instructions = "Q/ESC: quit | M: toggle mode | F: toggle FPS | Click Exit to return to menu"
                cv2.putText(frame, instructions, (10, 510),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
                
                cv2.imshow(self.window_name, frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Q or ESC
                    running = False
                elif key == ord('m'):  # Toggle mode
                    self.single_person_mode = not self.single_person_mode
                    mode_text = "Single" if self.single_person_mode else "SVM"
                    self.buttons["toggle_mode"].text = f"Mode: {mode_text}"
                elif key == ord('f'):  # Toggle FPS
                    self.show_fps = not self.show_fps
                    self.buttons["toggle_fps"].text = f"FPS: {'ON' if self.show_fps else 'OFF'}"
                
                # Check if exit button was clicked
                if self.should_exit:
                    running = False
                
        finally:
            cap.release()
            cv2.destroyWindow(self.window_name)


class BatchImporter:
    """Batch import interface for loading face images from folders."""
    
    def __init__(self, detector: FaceDetector, data_dir: pathlib.Path,
                 width: int = 800, height: int = 600):
        self.detector = detector
        self.data_dir = data_dir
        self.width = width
        self.height = height
        self.window_name = "Batch Import"
        
        # Import state
        self.selected_folder = None
        self.person_name = ""
        self.import_progress = 0.0
        self.total_files = 0
        self.processed_files = 0
        self.successful_imports = 0
        self.failed_imports = 0
        self.is_importing = False
        self.import_complete = False
        
        self.status_messages = []
        self.max_messages = 8
        
    def add_status(self, message: str):
        """Add a status message."""
        self.status_messages.append(message)
        if len(self.status_messages) > self.max_messages:
            self.status_messages.pop(0)
    
    def import_from_folder(self, folder_path: str, person_name: str) -> bool:
        """Import images from a folder."""
        import glob
        
        # Find all image files
        patterns = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        for pattern in patterns:
            image_files.extend(glob.glob(os.path.join(folder_path, pattern)))
        
        if not image_files:
            self.add_status(f"No image files found in {folder_path}")
            return False
        
        self.total_files = len(image_files)
        self.processed_files = 0
        self.successful_imports = 0
        self.failed_imports = 0
        self.is_importing = True
        
        # Create person directory
        person_dir = self.data_dir / person_name
        person_dir.mkdir(parents=True, exist_ok=True)
        
        self.add_status(f"Found {self.total_files} images to process")
        
        # Process each image
        for img_path in image_files:
            self.processed_files += 1
            self.import_progress = self.processed_files / self.total_files
            
            try:
                # Load image
                img = cv2.imread(img_path)
                if img is None:
                    self.failed_imports += 1
                    self.add_status(f"Failed to load: {os.path.basename(img_path)}")
                    continue
                
                # Detect faces
                if hasattr(self.detector, 'detect_and_align_faces'):
                    boxes, aligned_faces = self.detector.detect_and_align_faces(img)
                else:
                    boxes = self.detector.detect_faces(img)
                    aligned_faces = []
                
                if not boxes:
                    self.failed_imports += 1
                    self.add_status(f"No face found: {os.path.basename(img_path)}")
                    continue
                
                # Save the first detected face
                if aligned_faces:
                    face_img = aligned_faces[0]
                else:
                    x1, y1, x2, y2 = boxes[0]
                    face_img = img[y1:y2, x1:x2]
                
                # Save with timestamp
                timestamp = int(time.time() * 1000000)  # microseconds for uniqueness
                out_path = person_dir / f"{timestamp}.jpg"
                cv2.imwrite(str(out_path), face_img)
                
                self.successful_imports += 1
                
            except Exception as e:
                self.failed_imports += 1
                self.add_status(f"Error processing {os.path.basename(img_path)}: {str(e)}")
        
        self.is_importing = False
        self.import_complete = True
        self.add_status(f"Import complete: {self.successful_imports} succeeded, {self.failed_imports} failed")
        
        return self.successful_imports > 0
    
    def show_dialog(self) -> bool:
        """Show import dialog and handle the import process."""
        cv2.namedWindow(self.window_name)
        
        # Get folder path and person name
        folder_input = self._get_text_input("Enter folder path:", "Folder path containing face images")
        if not folder_input or not os.path.exists(folder_input):
            cv2.destroyWindow(self.window_name)
            return False
        
        person_input = self._get_text_input("Enter person name:", "Name for this person")
        if not person_input:
            cv2.destroyWindow(self.window_name)
            return False
        
        self.selected_folder = folder_input
        self.person_name = person_input
        
        # Start import in background thread
        import threading
        thread = threading.Thread(
            target=self.import_from_folder,
            args=(self.selected_folder, self.person_name)
        )
        thread.start()
        
        # Show progress
        while True:
            img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 240
            
            # Title
            cv2.putText(img, "Importing Images", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 50, 50), 2)
            
            # Folder info
            cv2.putText(img, f"Folder: {os.path.basename(self.selected_folder)}", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
            cv2.putText(img, f"Person: {self.person_name}", (50, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
            
            if self.is_importing or self.import_complete:
                # Progress bar
                bar_x = 50
                bar_y = 170
                bar_width = self.width - 100
                bar_height = 30
                
                cv2.rectangle(img, (bar_x, bar_y), 
                             (bar_x + bar_width, bar_y + bar_height),
                             (200, 200, 200), -1)
                
                if self.import_progress > 0:
                    progress_width = int(bar_width * self.import_progress)
                    cv2.rectangle(img, (bar_x, bar_y),
                                 (bar_x + progress_width, bar_y + bar_height),
                                 (0, 200, 0), -1)
                
                cv2.rectangle(img, (bar_x, bar_y),
                             (bar_x + bar_width, bar_y + bar_height),
                             (100, 100, 100), 2)
                
                # Progress text
                progress_text = f"{self.processed_files}/{self.total_files} files"
                cv2.putText(img, progress_text, 
                           (bar_x + bar_width // 2 - 50, bar_y + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)
                
                # Stats
                stats_y = 230
                cv2.putText(img, f"Successful: {self.successful_imports}", (50, stats_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 1)
                cv2.putText(img, f"Failed: {self.failed_imports}", (250, stats_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 1)
            
            # Status messages
            msg_y = 280
            for msg in self.status_messages:
                cv2.putText(img, msg, (50, msg_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                msg_y += 25
            
            # Instructions
            if self.import_complete:
                cv2.putText(img, "Import complete! Press any key to continue...", 
                           (50, self.height - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 1)
            else:
                cv2.putText(img, "Importing... Please wait", 
                           (50, self.height - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            
            cv2.imshow(self.window_name, img)
            
            key = cv2.waitKey(100) & 0xFF
            if self.import_complete and key != 255:
                break
            elif key == 27:  # ESC
                break
        
        thread.join()
        cv2.destroyWindow(self.window_name)
        
        return self.successful_imports > 0
    
    def _get_text_input(self, prompt: str, description: str = "") -> str:
        """Get text input from user."""
        window_name = "Text Input"
        cv2.namedWindow(window_name)
        
        text = ""
        
        while True:
            img = np.ones((250, 600, 3), dtype=np.uint8) * 240
            
            cv2.putText(img, prompt, (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
            
            if description:
                cv2.putText(img, description, (50, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            
            # Text input box
            cv2.rectangle(img, (50, 120), (550, 160), (255, 255, 255), -1)
            cv2.rectangle(img, (50, 120), (550, 160), (150, 150, 150), 2)
            
            cv2.putText(img, text + "_", (60, 145),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            
            cv2.putText(img, "Press ENTER to confirm, ESC to cancel", (50, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            
            cv2.imshow(window_name, img)
            
            key = cv2.waitKey(0)

            if key == 27:  # ESC
                text = ""
                break
            elif key == 13 or key == 10:  # Enter (both CR and LF)
                if text:
                    break
            elif key == 8 or key == 127:  # Backspace or Delete key
                if len(text) > 0:
                    text = text[:-1]
            elif 32 <= (key & 0xFF) <= 126:  # Printable characters
                text += chr(key & 0xFF)
        
        cv2.destroyWindow(window_name)
        return text


class TrainingProgress:
    """Visual progress indicator for training operations."""
    
    def __init__(self, window_name: str = "Training Progress",
                 width: int = 600, height: int = 400):
        self.window_name = window_name
        self.width = width
        self.height = height
        self.current_step = ""
        self.progress = 0.0
        self.status_messages = []
        self.max_messages = 10
        self.is_complete = False
        self.error = None
        
    def update(self, step: str, progress: float, message: Optional[str] = None):
        """Update training progress."""
        self.current_step = step
        self.progress = max(0.0, min(1.0, progress))
        
        if message:
            self.status_messages.append(message)
            if len(self.status_messages) > self.max_messages:
                self.status_messages.pop(0)
        
        self._draw()
    
    def complete(self, success: bool = True, error: Optional[str] = None):
        """Mark training as complete."""
        self.is_complete = True
        self.progress = 1.0 if success else self.progress
        self.error = error
        
        if success:
            self.status_messages.append("Training completed successfully!")
        else:
            self.status_messages.append(f"Training failed: {error}")
        
        self._draw()
    
    def _draw(self):
        """Draw the progress window."""
        img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 240
        
        # Title
        title = "Training Neural Network"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        title_x = (self.width - title_size[0]) // 2
        cv2.putText(img, title, (title_x, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
        
        # Current step
        cv2.putText(img, f"Step: {self.current_step}", (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
        
        # Progress bar
        bar_x = 50
        bar_y = 130
        bar_width = self.width - 100
        bar_height = 30
        
        # Background
        cv2.rectangle(img, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height),
                     (200, 200, 200), -1)
        
        # Progress
        if self.progress > 0:
            progress_width = int(bar_width * self.progress)
            color = (0, 200, 0) if not self.error else (0, 0, 200)
            cv2.rectangle(img, (bar_x, bar_y),
                         (bar_x + progress_width, bar_y + bar_height),
                         color, -1)
        
        # Border
        cv2.rectangle(img, (bar_x, bar_y),
                     (bar_x + bar_width, bar_y + bar_height),
                     (100, 100, 100), 2)
        
        # Percentage
        percent_text = f"{int(self.progress * 100)}%"
        percent_size = cv2.getTextSize(percent_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        percent_x = bar_x + (bar_width - percent_size[0]) // 2
        percent_y = bar_y + (bar_height + percent_size[1]) // 2
        cv2.putText(img, percent_text, (percent_x, percent_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)
        
        # Status messages
        message_y = 200
        for msg in self.status_messages[-5:]:  # Show last 5 messages
            cv2.putText(img, msg, (50, message_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            message_y += 25
        
        # Instructions
        if not self.is_complete:
            cv2.putText(img, "Processing... Please wait", 
                       (50, self.height - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        cv2.imshow(self.window_name, img)
    
    def close(self):
        """Close the progress window."""
        cv2.destroyWindow(self.window_name)


class PersonManager:
    """Interface for managing registered people and deleting them."""

    def __init__(self, data_dir: pathlib.Path, width: int = 800, height: int = 600):
        self.data_dir = data_dir
        self.width = width
        self.height = height
        self.window_name = "Manage People"
        self.people = []
        self.selected_person = None
        self.mouse_x = 0
        self.mouse_y = 0
        self.scroll_offset = 0
        self.should_exit = False

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        self.mouse_x = x
        self.mouse_y = y

        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if back button was clicked
            if 10 <= x <= 110 and 10 <= y <= 50:
                self.should_exit = True
                return

            # Check if a delete button was clicked
            start_y = 100 - self.scroll_offset
            for i, (person, _) in enumerate(self.people):
                item_y = start_y + i * 70
                if item_y < 100 or item_y > self.height - 100:
                    continue

                # Delete button bounds - adjusted for new position
                if 600 <= x <= 700 and item_y + 10 <= y <= item_y + 50:
                    self.selected_person = person
                    if self.confirm_delete(person):
                        self.delete_person(person)
                        self.refresh_people_list()
                    break

        elif event == cv2.EVENT_MOUSEWHEEL:
            # Handle scrolling
            if flags > 0:
                self.scroll_offset = max(0, self.scroll_offset - 30)
            else:
                max_scroll = max(0, len(self.people) * 70 - (self.height - 200))
                self.scroll_offset = min(max_scroll, self.scroll_offset + 30)

    def refresh_people_list(self):
        """Refresh the list of registered people."""
        self.people = []
        if self.data_dir.exists():
            for person_dir in self.data_dir.iterdir():
                if person_dir.is_dir() and not person_dir.name.startswith('.'):
                    # Count images
                    image_count = len(list(person_dir.glob("*.jpg")))
                    if image_count > 0:
                        self.people.append((person_dir.name, image_count))
        self.people.sort()

    def delete_person(self, person_name: str):
        """Delete a person and all their data."""
        import shutil

        person_dir = self.data_dir / person_name
        if person_dir.exists():
            shutil.rmtree(person_dir)
            print(f"Deleted {person_name} and all associated data")

            # Also delete from artifacts if they exist
            artifacts_dir = self.data_dir.parent / "artifacts"
            if artifacts_dir.exists():
                # Mark artifacts as needing retraining
                svm_path = artifacts_dir / "svm.pkl"
                if svm_path.exists():
                    svm_path.unlink()
                embeddings_path = artifacts_dir / "embeddings.npz"
                if embeddings_path.exists():
                    embeddings_path.unlink()
                label_map_path = artifacts_dir / "label_map.json"
                if label_map_path.exists():
                    label_map_path.unlink()
                print("Cleared trained model - retraining required")

    def confirm_delete(self, person_name: str) -> bool:
        """Show confirmation dialog for deletion."""
        window_name = "Confirm Delete"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 400, 200)

        # Force window to front
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

        # Shared state
        class DialogState:
            def __init__(self):
                self.confirmed = None  # None = still open, True = delete, False = cancel
                self.mouse_x = 0
                self.mouse_y = 0

        state = DialogState()

        def mouse_callback(event, x, y, flags, param):
            state.mouse_x = x
            state.mouse_y = y

            if event == cv2.EVENT_LBUTTONDOWN:
                # Check if Delete button was clicked
                if 80 <= x <= 180 and 130 <= y <= 170:
                    state.confirmed = True
                # Check if Cancel button was clicked
                elif 220 <= x <= 320 and 130 <= y <= 170:
                    state.confirmed = False

        cv2.setMouseCallback(window_name, mouse_callback)

        while state.confirmed is None:
            img = np.ones((200, 400, 3), dtype=np.uint8) * 240

            # Title
            cv2.putText(img, "Confirm Deletion", (100, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)

            # Message
            cv2.putText(img, f"Delete {person_name}?", (50, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cv2.putText(img, "This cannot be undone!", (50, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)

            # Delete button with hover effect
            delete_hover = 80 <= state.mouse_x <= 180 and 130 <= state.mouse_y <= 170
            delete_color = (70, 70, 250) if delete_hover else (50, 50, 200)
            cv2.rectangle(img, (80, 130), (180, 170), delete_color, -1)
            cv2.putText(img, "Delete", (105, 155),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Cancel button with hover effect
            cancel_hover = 220 <= state.mouse_x <= 320 and 130 <= state.mouse_y <= 170
            cancel_color = (120, 120, 120) if cancel_hover else (100, 100, 100)
            cv2.rectangle(img, (220, 130), (320, 170), cancel_color, -1)
            cv2.putText(img, "Cancel", (245, 155),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Instructions
            cv2.putText(img, "Click button or press Y/N", (100, 185),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

            cv2.imshow(window_name, img)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('n') or key == ord('N'):  # ESC or N
                state.confirmed = False
            elif key == ord('y') or key == ord('Y'):  # Y for yes
                state.confirmed = True

        cv2.destroyWindow(window_name)
        cv2.waitKey(1)  # Ensure window is closed
        return state.confirmed == True

    def run(self):
        """Run the person management interface."""
        self.refresh_people_list()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        # Force window to front
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)

        self.should_exit = False
        self.scroll_offset = 0

        while not self.should_exit:
            img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 240

            # Draw header
            cv2.rectangle(img, (0, 0), (self.width, 80), (220, 220, 220), -1)
            cv2.putText(img, "Manage People", (300, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 50, 50), 2)

            # Back button
            back_button = Button(10, 10, 100, 40, "< Back",
                                color=(100, 100, 100), hover_color=(150, 150, 150))
            back_button.draw(img, back_button.contains(self.mouse_x, self.mouse_y))

            # Draw people list
            if not self.people:
                cv2.putText(img, "No people registered yet", (250, 300),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
            else:
                # Create a clipping region for the list
                list_area = img[100:self.height-50, 50:self.width-50]

                start_y = -self.scroll_offset
                for i, (person, count) in enumerate(self.people):
                    item_y = start_y + i * 70

                    # Skip if outside visible area
                    if item_y + 60 < 0 or item_y > list_area.shape[0]:
                        continue

                    # Draw item background
                    if 0 <= item_y < list_area.shape[0] - 60:
                        cv2.rectangle(list_area, (0, item_y),
                                     (list_area.shape[1], item_y + 60),
                                     (250, 250, 250), -1)
                        cv2.rectangle(list_area, (0, item_y),
                                     (list_area.shape[1], item_y + 60),
                                     (200, 200, 200), 1)

                        # Person name
                        cv2.putText(list_area, person, (20, item_y + 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)

                        # Photo count
                        count_text = f"{count} photos"
                        cv2.putText(list_area, count_text, (20, item_y + 45),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

                        # Delete button - moved left to be fully visible
                        delete_button = Button(550, item_y + 10, 100, 40, "Delete",
                                             color=(50, 50, 200), hover_color=(70, 70, 250))
                        delete_hover = (600 <= self.mouse_x <= 700 and
                                      100 + item_y + 10 <= self.mouse_y <= 100 + item_y + 50)
                        delete_button.draw(list_area, delete_hover)

            # Draw footer
            cv2.putText(img, f"Total: {len(self.people)} people registered",
                       (50, self.height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

            # Instructions
            if self.people:
                cv2.putText(img, "Click Delete to remove a person | Scroll to see more",
                           (300, self.height - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

            cv2.imshow(self.window_name, img)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

        cv2.destroyWindow(self.window_name)
        cv2.waitKey(1)  # Process any pending GUI events
