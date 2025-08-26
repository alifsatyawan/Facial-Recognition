from typing import List, Tuple
import pickle

import numpy as np
from sklearn.svm import SVC


class SVMClassifier:
    def __init__(self):
        self.model: SVC | None = None

    def train(self, x: np.ndarray, y: np.ndarray, c: float = 1.0) -> None:
        self.model = SVC(C=c, kernel='rbf', probability=True)
        self.model.fit(x, y)

    def predict_proba(self, x: List[np.ndarray]) -> Tuple[np.ndarray | None, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        X = np.vstack(x).astype(np.float32)
        probs = None
        try:
            probs = self.model.predict_proba(X)
        except Exception:
            probs = None
        preds = self.model.predict(X)
        return probs, preds

    def save(self, path) -> None:
        if self.model is None:
            raise RuntimeError("Model not trained")
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path) -> None:
        with open(path, 'rb') as f:
            self.model = pickle.load(f)


