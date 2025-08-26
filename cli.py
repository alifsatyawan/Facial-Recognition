import os
import sys
import json
import time
import math
import glob
import pickle
import signal
import pathlib
from typing import List, Tuple, Optional

import click
from rich import print
from rich.console import Console
from rich.table import Table

import numpy as np

import cv2

from modules.detector import FaceDetector
from modules.embedder import FaceEmbedder
from modules.classifier import SVMClassifier
from modules.dataset import DatasetManager


console = Console()


BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


@click.group()
def cli():
    """Facial Recognition POC CLI"""


@cli.command()
def init():
    """Create data and artifacts directories."""
    ensure_dirs()
    print("[green]Initialized directories:[/green]", DATA_DIR, ARTIFACTS_DIR)


@cli.command()
@click.option("--name", required=True, help="Person's name")
@click.option("--num", default=30, show_default=True, help="Number of samples")
@click.option("--camera", default=0, show_default=True, help="Camera index")
@click.option("--auto", is_flag=True, default=False, help="Auto-capture when face detected")
def capture(name: str, num: int, camera: int, auto: bool):
    """Capture face images for a person from webcam.

    Controls:
      - Manual: press 'c' to capture, 'q' to quit
      - Auto:   use --auto to capture frames automatically when a face is detected
    """
    ensure_dirs()
    person_dir = DATA_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)

    detector = FaceDetector()

    # Try AVFoundation explicitly on macOS, then fallback
    cap = cv2.VideoCapture(camera, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print("[red]Failed to open camera[/red]")
        print("[yellow]On macOS: System Settings → Privacy & Security → Camera → enable your Terminal app. Then fully quit Terminal and retry.[/yellow]")
        sys.exit(1)

    if auto:
        print(f"[cyan]Auto-capturing for {name}... Press 'q' to quit[/cyan]")
    else:
        print(f"[cyan]Capturing for {name}... Press 'c' to capture, 'q' to quit[/cyan]")
    captured = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[red]Failed to read frame[/red]")
                break
            display = frame.copy()
            boxes = detector.detect_faces(frame)
            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, f"{captured}/{num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Capture", display)
            if auto:
                if len(boxes) > 0:
                    x1, y1, x2, y2 = boxes[0]
                    face = frame[y1:y2, x1:x2]
                    out_path = person_dir / f"{int(time.time()*1000)}.jpg"
                    cv2.imwrite(str(out_path), face)
                    captured += 1
                    time.sleep(0.05)
                    if captured >= num:
                        break
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            else:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('c') and len(boxes) > 0:
                    x1, y1, x2, y2 = boxes[0]
                    face = frame[y1:y2, x1:x2]
                    out_path = person_dir / f"{int(time.time()*1000)}.jpg"
                    cv2.imwrite(str(out_path), face)
                    captured += 1
                    if captured >= num:
                        break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"[green]Captured {captured} images for {name}[/green]")


@cli.command("realtime-one")
@click.option("--name", required=True, help="Person name in data/ to match against")
@click.option("--camera", default=0, show_default=True, help="Camera index")
@click.option("--threshold", default=0.6, show_default=True, help="Cosine similarity threshold (0-1)")
@click.option("--max-ref", default=60, show_default=True, help="Max reference images to build centroid")
def realtime_one(name: str, camera: int, threshold: float, max_ref: int):
    """Realtime single-person recognition via cosine similarity to that person's centroid.

    Requires images in data/<name>/*.jpg. No SVM needed.
    """
    ensure_dirs()
    person_dir = DATA_DIR / name
    if not person_dir.exists():
        print(f"[red]No folder: {person_dir}. Capture first with: cli.py capture --name '{name}' --auto[/red]")
        return
    # Gather reference images
    ref_paths = sorted([str(p) for p in person_dir.glob("*.jpg")])[:max_ref]
    if len(ref_paths) == 0:
        print(f"[red]No images in {person_dir}. Capture first.[/red]")
        return

    detector = FaceDetector()
    embedder = FaceEmbedder()

    # Build centroid from cropped faces; if detection misses, fallback to full image
    ref_images = []
    for p in ref_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        boxes = detector.detect_faces(img)
        if boxes:
            x1, y1, x2, y2 = boxes[0]
            img = img[y1:y2, x1:x2]
        ref_images.append(img)
    if len(ref_images) == 0:
        print("[red]Could not prepare reference images.[/red]")
        return
    refs = embedder.embed_images(ref_images)
    # L2 normalize and compute centroid
    def l2norm(a):
        eps = 1e-10
        n = np.linalg.norm(a, axis=1, keepdims=True) + eps
        return a / n
    refs = l2norm(refs)
    centroid = refs.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-10)

    # Open camera with AVFoundation first
    cap = cv2.VideoCapture(camera, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print("[red]Failed to open camera[/red]")
        print("[yellow]On macOS: System Settings → Privacy & Security → Camera → enable your Terminal app. Then fully quit Terminal and retry.[/yellow]")
        return

    print("[cyan]Press 'q' to quit[/cyan]")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            boxes = detector.detect_faces(frame)
            for (x1, y1, x2, y2) in boxes:
                face = frame[y1:y2, x1:x2]
                emb = embedder.embed_images([face])
                emb = l2norm(emb)
                sim = float(np.dot(emb[0], centroid))
                is_match = sim >= threshold
                color = (0, 255, 0) if is_match else (0, 0, 255)
                text = f"{name}:{sim:.2f}" if is_match else f"unknown:{sim:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.imshow("Realtime", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


@cli.command("build-embeddings")
@click.option("--batch-size", default=32, show_default=True)
def build_embeddings(batch_size: int):
    """Build embeddings for dataset and save to artifacts."""
    ensure_dirs()
    dataset = DatasetManager(DATA_DIR)
    image_paths, labels, label_to_index = dataset.index_dataset()
    if len(image_paths) == 0:
        print("[yellow]No images found in data/. Use capture first.[/yellow]")
        return
    embedder = FaceEmbedder()
    embeddings = embedder.embed_paths(image_paths, batch_size=batch_size)
    embeddings = np.asarray(embeddings)
    label_indices = np.array([label_to_index[l] for l in labels], dtype=np.int64)

    np.savez(ARTIFACTS_DIR / "embeddings.npz", x=embeddings, y=label_indices)
    with open(ARTIFACTS_DIR / "label_map.json", "w") as f:
        json.dump({str(v): k for k, v in label_to_index.items()}, f, indent=2)
    print("[green]Saved embeddings and label map in artifacts/[/green]")


@cli.command()
@click.option("--c", "--C", default=1.0, show_default=True, help="SVM C parameter")
def train(c: float):
    """Train SVM classifier on embeddings."""
    ensure_dirs()
    emb_path = ARTIFACTS_DIR / "embeddings.npz"
    if not emb_path.exists():
        print("[yellow]No embeddings found. Run build-embeddings first.[/yellow]")
        return
    data = np.load(emb_path)
    x = data["x"]
    y = data["y"]
    clf = SVMClassifier()
    clf.train(x, y, c)
    clf.save(ARTIFACTS_DIR / "svm.pkl")
    print("[green]Saved classifier to artifacts/svm.pkl[/green]")


@cli.command()
@click.option("--camera", default=0, show_default=True, help="Camera index")
@click.option("--threshold", default=0.6, show_default=True, help="Probability threshold")
def realtime(camera: int, threshold: float):
    """Run realtime recognition from webcam.

    If artifacts are missing, runs detection-only mode (draws boxes, no labels).
    """
    ensure_dirs()
    # Try to load artifacts
    label_map_path = ARTIFACTS_DIR / "label_map.json"
    clf_path = ARTIFACTS_DIR / "svm.pkl"
    have_artifacts = label_map_path.exists() and clf_path.exists()
    if have_artifacts:
        with open(label_map_path) as f:
            index_to_label = json.load(f)
        clf = SVMClassifier()
        clf.load(clf_path)
    else:
        index_to_label = {}
        clf = None
        print("[yellow]Artifacts missing: running detection-only mode.[/yellow]")

    detector = FaceDetector()
    embedder = FaceEmbedder() if have_artifacts else None

    cap = cv2.VideoCapture(camera, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print("[red]Failed to open camera[/red]")
        print("[yellow]On macOS: System Settings → Privacy & Security → Camera → enable your Terminal app. Then fully quit Terminal and retry.[/yellow]")
        sys.exit(1)

    print("[cyan]Press 'q' to quit[/cyan]")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            boxes = detector.detect_faces(frame)
            for (x1, y1, x2, y2) in boxes:
                if have_artifacts and embedder is not None and clf is not None:
                    face = frame[y1:y2, x1:x2]
                    emb = embedder.embed_images([face])[0]
                    prob, pred = clf.predict_proba([emb])
                    label = index_to_label.get(str(pred[0]), "unknown")
                    p = float(prob[0, pred[0]]) if prob is not None else 0.0
                    color = (0, 255, 0) if p >= threshold else (0, 0, 255)
                    text = f"{label}:{p:.2f}"
                else:
                    color = (255, 255, 0)
                    text = "face"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.imshow("Realtime", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


@cli.command()
def info():
    """Show dataset and artifact info."""
    ensure_dirs()
    dataset = DatasetManager(DATA_DIR)
    persons = dataset.list_persons()
    table = Table(title="Dataset")
    table.add_column("Person")
    table.add_column("Images", justify="right")
    for p, cnt in persons:
        table.add_row(p, str(cnt))
    console.print(table)

    emb = ARTIFACTS_DIR / "embeddings.npz"
    svm = ARTIFACTS_DIR / "svm.pkl"
    print(f"Embeddings: {'present' if emb.exists() else 'missing'}")
    print(f"Classifier: {'present' if svm.exists() else 'missing'}")


if __name__ == "__main__":
    cli()


