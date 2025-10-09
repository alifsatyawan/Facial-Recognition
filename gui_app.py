#!/usr/bin/env python3
"""
GUI Application for Facial Recognition System

This provides a graphical interface for the facial recognition system,
including face capture, training, and real-time recognition.
"""

import os
import sys
import json
import pathlib
import argparse
from typing import Optional

import cv2
import numpy as np
from rich import print

# Add project root to path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from modules.gui import (
    MainMenu, InteractiveCapture, InteractiveRecognition,
    TrainingProgress, GUIState, BatchImporter, PersonManager
)
from modules.detector import FaceDetector
from modules.embedder import FaceEmbedder
from modules.classifier import SVMClassifier
from modules.dataset import DatasetManager
import config


class FacialRecognitionGUI:
    """Main GUI application for facial recognition."""
    
    def __init__(self, camera_index: int = 0, use_alignment: bool = True, 
                 performance_mode: bool = False):
        self.camera_index = camera_index
        self.use_alignment = use_alignment
        self.performance_mode = performance_mode
        
        # Setup paths
        self.base_dir = pathlib.Path(__file__).resolve().parent
        self.data_dir = self.base_dir / "data"
        self.artifacts_dir = self.base_dir / "artifacts"
        self.models_dir = self.base_dir / "models"
        
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._init_components()
        
    def _init_components(self):
        """Initialize face detection, embedding, and GUI components."""
        # Check for shape predictor if alignment is requested
        shape_predictor_path = None
        
        # In performance mode, disable alignment by default
        if self.performance_mode:
            self.use_alignment = False
            print("[cyan]Performance mode enabled - alignment disabled for better FPS[/cyan]")
        
        if self.use_alignment:
            shape_predictor_path = config.SHAPE_PREDICTOR_PATH
            if not shape_predictor_path.exists():
                print(f"[yellow]Shape predictor not found at {shape_predictor_path}[/yellow]")
                print("[yellow]Run 'python cli.py download-model' to download it[/yellow]")
                print("[yellow]Continuing without face alignment...[/yellow]")
                self.use_alignment = False
                shape_predictor_path = None
            else:
                shape_predictor_path = str(shape_predictor_path)
        
        # Initialize detector
        self.detector = FaceDetector(
            shape_predictor_path=shape_predictor_path,
            align_faces=self.use_alignment
        )
        
        # Initialize embedder
        self.embedder = FaceEmbedder()
        
        # Initialize GUI components
        self.main_menu = MainMenu()
        self.capture_ui = InteractiveCapture(self.detector, self.data_dir)
        self.recognition_ui = InteractiveRecognition(self.detector, self.embedder)
        
    def run(self):
        """Run the main GUI application loop."""
        print("[cyan]Starting Facial Recognition GUI...[/cyan]")
        if self.use_alignment:
            print("[green]Face alignment is enabled[/green]")
        else:
            print("[yellow]Face alignment is disabled[/yellow]")
        
        while True:
            # Show main menu
            action = self.main_menu.show()
            
            if action == "exit" or action is None:
                break
            elif action == "capture":
                self._handle_capture()
            elif action == "train":
                self._handle_training()
            elif action == "recognize":
                self._handle_recognition()
            elif action == "gallery":
                self._handle_gallery()
            elif action == "manage":
                self._handle_manage()
            elif action == "import":
                self._handle_import()
        
        print("[cyan]Goodbye![/cyan]")
        cv2.destroyAllWindows()  # Ensure all windows are closed
    
    def _handle_capture(self):
        """Handle face capture mode."""
        # Ensure all windows are closed before starting
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        try:
            success = self.capture_ui.run(self.camera_index)
            if success:
                print(f"[green]Successfully captured faces for {self.capture_ui.person_name}[/green]")
            else:
                print("[yellow]Capture cancelled[/yellow]")
        except Exception as e:
            print(f"[red]Error during capture: {e}[/red]")
            import traceback
            traceback.print_exc()
        finally:
            # Ensure proper cleanup
            cv2.destroyAllWindows()
            cv2.waitKey(10)  # Small delay to ensure windows are closed
    
    def _handle_training(self):
        """Handle model training."""
        # Create progress window
        progress = TrainingProgress()
        cv2.namedWindow(progress.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(progress.window_name, progress.width, progress.height)
        
        try:
            # Check if we have data
            dataset = DatasetManager(self.data_dir)
            persons = dataset.list_persons()
            
            if len(persons) == 0:
                progress.complete(False, "No training data found. Capture faces first.")
                cv2.waitKey(2000)  # Show message for 2 seconds
                return
            
            if len(persons) == 1:
                progress.update("Checking data", 0.1, 
                              f"Only one person found: {persons[0][0]}")
                cv2.waitKey(1)
                progress.update("Checking data", 0.2, 
                              "Consider using single-person mode in recognition")
                cv2.waitKey(1)
            
            # Build embeddings
            progress.update("Building embeddings", 0.3, "Indexing dataset...")
            cv2.waitKey(1)
            
            try:
                # Index dataset
                image_paths, labels, label_to_index = dataset.index_dataset()
                progress.update("Building embeddings", 0.4, 
                              f"Found {len(image_paths)} images")
                cv2.waitKey(1)
                
                # Create embeddings
                detector = None
                if self.use_alignment:
                    detector = self.detector
                
                embeddings = self.embedder.embed_paths(
                    image_paths, 
                    batch_size=32,
                    detector=detector
                )
                progress.update("Building embeddings", 0.7, 
                              "Embeddings generated")
                cv2.waitKey(1)
                
                # Save embeddings
                embeddings = np.asarray(embeddings)
                label_indices = np.array([label_to_index[l] for l in labels], 
                                       dtype=np.int64)
                
                np.savez(self.artifacts_dir / "embeddings.npz", 
                        x=embeddings, y=label_indices)
                
                with open(self.artifacts_dir / "label_map.json", "w") as f:
                    json.dump({str(v): k for k, v in label_to_index.items()}, 
                             f, indent=2)
                
                progress.update("Training classifier", 0.8, 
                              "Training SVM...")
                cv2.waitKey(1)
                
                # Train classifier if we have multiple classes
                if len(label_to_index) >= 2:
                    clf = SVMClassifier()
                    clf.train(embeddings, label_indices, c=1.0)
                    clf.save(self.artifacts_dir / "svm.pkl")
                    progress.update("Training classifier", 0.95, 
                                  "Classifier trained")
                else:
                    progress.update("Training classifier", 0.95, 
                                  "Skipped SVM (only one person)")
                
                cv2.waitKey(1)
                progress.complete(True)
                cv2.waitKey(2000)  # Show success for 2 seconds
                
            except Exception as e:
                progress.complete(False, str(e))
                cv2.waitKey(3000)  # Show error for 3 seconds
            
        finally:
            progress.close()
    
    def _handle_recognition(self):
        """Handle real-time recognition mode."""
        print("[cyan]Starting recognition mode...[/cyan]")
        
        # Ensure all windows are closed before starting
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
        # Try to load classifier
        clf_loaded = self.recognition_ui.load_classifier(self.artifacts_dir)
        
        if not clf_loaded:
            print("[yellow]No trained model found.[/yellow]")
            
            # Check for single person mode
            dataset = DatasetManager(self.data_dir)
            persons = dataset.list_persons()
            
            if len(persons) == 1:
                person_name = persons[0][0]
                print(f"[cyan]Found one person: {person_name}[/cyan]")
                print("[cyan]Setting up single-person recognition mode...[/cyan]")
                
                if self.recognition_ui.setup_single_person(person_name, self.data_dir):
                    self.recognition_ui.single_person_mode = True
                    self.recognition_ui.buttons["toggle_mode"].text = "Mode: Single"
                else:
                    print("[red]Failed to setup single-person mode[/red]")
                    return
            else:
                print("[yellow]Please train a model first.[/yellow]")
                return
        
        # Run recognition
        self.recognition_ui.run(self.camera_index)
    
    def _handle_gallery(self):
        """Handle gallery view (show captured faces)."""
        window_name = "Face Gallery"
        cv2.namedWindow(window_name)
        
        dataset = DatasetManager(self.data_dir)
        persons = dataset.list_persons()
        
        if not persons:
            # Show empty gallery message
            img = np.ones((400, 600, 3), dtype=np.uint8) * 240
            cv2.putText(img, "No faces captured yet", (150, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            cv2.putText(img, "Press any key to return", (150, 250),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            cv2.imshow(window_name, img)
            cv2.waitKey(0)
            cv2.destroyWindow(window_name)
            return
        
        # Create gallery view
        person_idx = 0
        
        while True:
            person_name, count = persons[person_idx]
            person_dir = self.data_dir / person_name
            
            # Load sample images
            image_files = list(person_dir.glob("*.jpg"))[:12]  # Show up to 12
            
            # Create gallery grid
            grid_cols = 4
            grid_rows = 3
            thumb_size = 150
            padding = 10
            
            gallery_width = grid_cols * thumb_size + (grid_cols + 1) * padding
            gallery_height = grid_rows * thumb_size + (grid_rows + 1) * padding + 100
            
            gallery = np.ones((gallery_height, gallery_width, 3), dtype=np.uint8) * 240
            
            # Title
            title = f"{person_name} ({count} images)"
            title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            title_x = (gallery_width - title_size[0]) // 2
            cv2.putText(gallery, title, (title_x, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
            
            # Navigation info
            nav_text = f"Person {person_idx + 1}/{len(persons)} | Left/Right: Navigate | Q: Quit"
            cv2.putText(gallery, nav_text, (padding, gallery_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            
            # Draw thumbnails
            for i, img_file in enumerate(image_files):
                if i >= grid_rows * grid_cols:
                    break
                
                row = i // grid_cols
                col = i % grid_cols
                
                x = col * thumb_size + (col + 1) * padding
                y = row * thumb_size + (row + 1) * padding + 60
                
                # Load and resize image
                img = cv2.imread(str(img_file))
                if img is not None:
                    # Make square crop
                    h, w = img.shape[:2]
                    size = min(h, w)
                    y_start = (h - size) // 2
                    x_start = (w - size) // 2
                    img = img[y_start:y_start+size, x_start:x_start+size]
                    img = cv2.resize(img, (thumb_size, thumb_size))
                    
                    # Place in gallery
                    gallery[y:y+thumb_size, x:x+thumb_size] = img
                    
                    # Draw border
                    cv2.rectangle(gallery, (x-1, y-1), 
                                 (x+thumb_size+1, y+thumb_size+1),
                                 (150, 150, 150), 1)
            
            cv2.imshow(window_name, gallery)
            
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == 81:  # Left arrow
                person_idx = (person_idx - 1) % len(persons)
            elif key == 83:  # Right arrow
                person_idx = (person_idx + 1) % len(persons)
        
        cv2.destroyWindow(window_name)
    
    def _handle_manage(self):
        """Handle person management (delete functionality)."""
        # Ensure all windows are closed before starting
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        manager = PersonManager(self.data_dir)
        manager.run()

        # After managing people, check if we need to retrain
        artifacts_exist = (
            (self.artifacts_dir / "svm.pkl").exists() or
            (self.artifacts_dir / "embeddings.npz").exists()
        )

        if not artifacts_exist:
            dataset = DatasetManager(self.data_dir)
            persons = dataset.list_persons()
            if persons:
                print("[yellow]Model artifacts were cleared. Please retrain the model.[/yellow]")

    def _handle_import(self):
        """Handle batch import of images."""
        importer = BatchImporter(self.detector, self.data_dir)
        success = importer.show_dialog()

        if success:
            print(f"[green]Successfully imported {importer.successful_imports} images for {importer.person_name}[/green]")
            if importer.failed_imports > 0:
                print(f"[yellow]Failed to import {importer.failed_imports} images[/yellow]")
        else:
            print("[yellow]Import cancelled or failed[/yellow]")


def main():
    """Main entry point for GUI application."""
    parser = argparse.ArgumentParser(description="Facial Recognition GUI")
    parser.add_argument("--camera", type=int, default=0, 
                       help="Camera index (default: 0)")
    parser.add_argument("--no-align", action="store_true",
                       help="Disable face alignment")
    parser.add_argument("--performance", action="store_true",
                       help="Enable performance mode (prioritize FPS over accuracy)")
    
    args = parser.parse_args()
    
    # Check for camera access on macOS
    if sys.platform == "darwin":
        # Test camera access
        cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            print("[red]Camera access denied![/red]")
            print("[yellow]On macOS:[/yellow]")
            print("[yellow]1. Go to System Settings → Privacy & Security → Camera[/yellow]")
            print("[yellow]2. Enable access for your Terminal application[/yellow]")
            print("[yellow]3. Fully quit Terminal and try again[/yellow]")
            cap.release()
            sys.exit(1)
        cap.release()
    
    # Run GUI
    app = FacialRecognitionGUI(
        camera_index=args.camera,
        use_alignment=not args.no_align,
        performance_mode=args.performance
    )
    
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        print(f"[red]Error: {e}[/red]")
        raise


if __name__ == "__main__":
    main()
