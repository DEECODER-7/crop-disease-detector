"""
Inference logic used by the API. Kept separate from the FastAPI app itself
(main.py) so this logic can be unit-tested or reused (e.g. by a batch scoring
script) without needing to spin up a web server.
"""

import json
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from src.models.architecture import build_model
from src.models.config import (
    BEST_MODEL_PATH,
    CLASS_NAMES_PATH,
    IMAGE_SIZE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
)
from src.models.dataset import build_transforms


class DiseasePredictor:
    """
    Loads the model + class names ONCE (in __init__), then serves repeated
    predictions cheaply. This matters for production: reloading a model file
    from disk on every request would make the API unusably slow. FastAPI will
    create one instance of this at startup and reuse it for every request.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(CLASS_NAMES_PATH) as f:
            self.class_names: List[str] = json.load(f)

        self.model = build_model(num_classes=len(self.class_names))
        checkpoint = torch.load(BEST_MODEL_PATH, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Same transform used for validation (no augmentation) — inference-time
        # preprocessing MUST match training/validation preprocessing exactly,
        # otherwise the model sees inputs it was never calibrated for.
        self.transform = build_transforms(train=False)

    def predict(self, image: Image.Image, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Returns the top_k (class_name, confidence) pairs, sorted by confidence
        descending. Returning top-k instead of just the top-1 prediction gives
        the eventual UI something more useful to show than a single guess —
        e.g. "87% Tomato Early Blight, 8% Tomato Late Blight" is more honest
        and more useful than just "Tomato Early Blight."
        """
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = F.softmax(outputs, dim=1)[0]

        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(self.class_names)))

        return [
            (self.class_names[idx.item()], round(prob.item(), 4))
            for prob, idx in zip(top_probs, top_indices)
        ]
