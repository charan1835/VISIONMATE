"""Face detection using OpenCV's bundled Haar cascade."""

import os
from typing import List, Tuple

import cv2
import numpy as np


class FaceDetector:
    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5):
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.classifier = cv2.CascadeClassifier(cascade_path)
        if self.classifier.empty():
            raise RuntimeError(f"Could not load face-detection cascade: {cascade_path}")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Return face boxes as (x1, y1, x2, y2) pixel coordinates."""
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Frame must be a non-empty NumPy image.")
        if frame.ndim == 2:
            gray = frame
        elif frame.ndim == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unsupported frame shape: {frame.shape}")

        faces = self.classifier.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(30, 30),
        )
        return [
            (int(x), int(y), int(x + width), int(y + height))
            for x, y, width, height in faces
        ]
