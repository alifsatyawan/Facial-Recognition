from typing import List, Optional

import os
import ssl
import certifi
import numpy as np
import cv2
import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1


class FaceEmbedder:
    """FaceNet embedder using facenet-pytorch (VGGFace2 pretrain)."""

    def __init__(self, device: str = 'cpu', image_size: int = 160):
        # Ensure HTTPS downloads use valid CA bundle on macOS/Python
        ca_path = certifi.where()
        os.environ.setdefault('SSL_CERT_FILE', ca_path)
        os.environ.setdefault('REQUESTS_CA_BUNDLE', ca_path)
        try:
            ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=ca_path)  # type: ignore
        except Exception:
            pass
        self.device = torch.device(device)
        self.image_size = image_size
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _preprocess(self, images_bgr: List[np.ndarray]) -> torch.Tensor:
        tensors = []
        for img_bgr in images_bgr:
            if img_bgr is None or img_bgr.size == 0:
                # fallback blank image to keep alignment
                img_bgr = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            tens = self.transform(img_rgb)
            tensors.append(tens)
        batch = torch.stack(tensors).to(self.device)
        return batch

    @torch.inference_mode()
    def embed_images(self, images_bgr: List[np.ndarray]) -> np.ndarray:
        batch = self._preprocess(images_bgr)
        if batch.shape[0] == 0:
            return np.zeros((0, 512), dtype=np.float32)
        emb = self.model(batch).cpu().numpy().astype(np.float32)
        return emb

    def embed_paths(self, paths: List[str], batch_size: int = 32, 
                   detector: Optional['FaceDetector'] = None) -> List[np.ndarray]:
        """
        Embed faces from image paths.
        
        Args:
            paths: List of image paths
            batch_size: Batch size for processing
            detector: Optional FaceDetector for alignment preprocessing
            
        Returns:
            List of embeddings
        """
        outputs: List[np.ndarray] = []
        total = len(paths)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            imgs = []
            for p in paths[start:end]:
                img = cv2.imread(p)
                if img is None:
                    imgs.append(None)
                    continue
                    
                # If detector with alignment is provided, use it
                if detector and hasattr(detector, '_align_faces') and detector._align_faces:
                    boxes, aligned_faces = detector.detect_and_align_faces(img)
                    if aligned_faces:
                        imgs.append(aligned_faces[0])  # Use first aligned face
                    else:
                        # Fallback to full image if no faces detected
                        imgs.append(img)
                else:
                    imgs.append(img)
                    
            emb = self.embed_images(imgs)
            outputs.append(emb)
        if len(outputs) == 0:
            return []
        return list(np.vstack(outputs))


