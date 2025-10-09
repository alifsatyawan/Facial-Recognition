"""Performance optimization utilities for facial recognition."""

import cv2
import numpy as np
from typing import Tuple, Optional


class FrameOptimizer:
    """Optimizes video frames for better performance."""
    
    def __init__(self, 
                 resize_factor: float = 1.0,
                 target_fps: int = 30,
                 skip_frames: int = 0):
        """
        Initialize frame optimizer.
        
        Args:
            resize_factor: Factor to resize frames (0.5 = half size)
            target_fps: Target FPS for processing
            skip_frames: Number of frames to skip between processing
        """
        self.resize_factor = resize_factor
        self.target_fps = target_fps
        self.skip_frames = skip_frames
        self.frame_count = 0
        
    def should_process_frame(self) -> bool:
        """Check if current frame should be processed."""
        if self.skip_frames <= 0:
            return True
        
        should_process = self.frame_count % (self.skip_frames + 1) == 0
        self.frame_count += 1
        return should_process
    
    def optimize_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Optimize frame for processing.
        
        Returns:
            Optimized frame and scale factor used
        """
        if self.resize_factor == 1.0:
            return frame, 1.0
        
        height, width = frame.shape[:2]
        new_width = int(width * self.resize_factor)
        new_height = int(height * self.resize_factor)
        
        # Use faster interpolation for downsizing
        if self.resize_factor < 1.0:
            optimized = cv2.resize(frame, (new_width, new_height), 
                                 interpolation=cv2.INTER_AREA)
        else:
            optimized = cv2.resize(frame, (new_width, new_height), 
                                 interpolation=cv2.INTER_LINEAR)
        
        return optimized, self.resize_factor
    
    def scale_boxes(self, boxes: list, scale_factor: float) -> list:
        """Scale bounding boxes back to original frame size."""
        if scale_factor == 1.0:
            return boxes
        
        inv_scale = 1.0 / scale_factor
        scaled_boxes = []
        
        for x1, y1, x2, y2 in boxes:
            scaled_boxes.append((
                int(x1 * inv_scale),
                int(y1 * inv_scale),
                int(x2 * inv_scale),
                int(y2 * inv_scale)
            ))
        
        return scaled_boxes


def optimize_camera_settings(cap: cv2.VideoCapture, 
                           width: Optional[int] = None,
                           height: Optional[int] = None,
                           fps: Optional[int] = None,
                           buffer_size: int = 1) -> None:
    """
    Optimize camera capture settings for better performance.
    
    Args:
        cap: VideoCapture object
        width: Desired frame width
        height: Desired frame height  
        fps: Desired FPS
        buffer_size: Camera buffer size (1 = no buffering)
    """
    # Set buffer size to reduce lag
    cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
    
    # Set resolution if specified
    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Set FPS if specified
    if fps is not None:
        cap.set(cv2.CAP_PROP_FPS, fps)
    
    # Disable auto-exposure for consistent performance (if supported)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual mode
    
    # Set faster codec if available
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
