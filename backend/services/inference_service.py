"""
Service d'inférence pour le modèle CheXpert
Charge le modèle Deep Learning une seule fois et expose une méthode de prédiction
"""
import os
import time
import threading
import logging
from typing import Dict, List

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

from backend.constants import PATHOLOGY_COLUMNS

logger = logging.getLogger(__name__)


class CheXpertDenseNet(nn.Module):
    """Architecture DenseNet121 adaptée à CheXpert."""

    def __init__(self, num_classes: int = 14):
        super().__init__()
        self.model = models.densenet121(weights=None)
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):  # type: ignore[override]
        return self.model(x)


class InferenceService:
    """Gère le chargement du modèle et les prédictions d'images."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = self._select_device()
        self.transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.model: nn.Module | None = None
        self._lock = threading.Lock()

    @staticmethod
    def _select_device() -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_model(self) -> None:
        """Charge le modèle une seule fois (thread-safe)."""
        if self.model is not None:
            return

        with self._lock:
            if self.model is not None:
                return

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Fichier de poids introuvable: {self.model_path}")

            logger.info("Chargement du modèle '%s' sur %s", self.model_path, self.device)

            model = CheXpertDenseNet(num_classes=len(PATHOLOGY_COLUMNS))
            state = torch.load(self.model_path, map_location=self.device)

            # Certains checkpoints peuvent être sauvegardés avec une clé 'state_dict'
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']

            if not isinstance(state, dict):
                raise ValueError("Checkpoint au format inattendu (doit être un dict).")

            # Compatibilité : accepte clés 'model.*', 'features.*', 'module.model.*'
            sample_key = next(iter(state.keys()))
            if sample_key.startswith("features."):
                state = {f"model.{k}": v for k, v in state.items()}
            elif sample_key.startswith("module.model."):
                state = {k.replace("module.", ""): v for k, v in state.items()}
            elif sample_key.startswith("module.features."):
                state = {k.replace("module.", "model.", 1): v for k, v in state.items()}

            model.load_state_dict(state, strict=True)  # type: ignore[arg-type]
            model.to(self.device)
            model.eval()

            self.model = model
            logger.info("Modèle chargé et prêt pour l'inférence")

    def predict(self, file_obj) -> Dict:
        """Retourne les prédictions triées par probabilité décroissante."""
        self._load_model()
        assert self.model is not None  # pour mypy

        try:
            file_obj.seek(0)
            image = Image.open(file_obj).convert('RGB')
        except Exception as exc:  # pragma: no cover - dépend des fichiers reçus
            raise ValueError(f"Impossible de lire l'image fournie: {exc}")

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.sigmoid(outputs).squeeze(0).cpu().numpy().tolist()
        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        predictions: List[Dict] = [
            {
                "label": label,
                "probability": float(prob),
                "percentage": round(float(prob) * 100, 2)
            }
            for label, prob in zip(PATHOLOGY_COLUMNS, probabilities)
        ]

        predictions.sort(key=lambda x: x["probability"], reverse=True)

        return {
            "predictions": predictions,
            "top3": predictions[:3],
            "inference_time_ms": duration_ms,
            "device": self.device.type,
            "model_path": self.model_path
        }


# Instance prête à l'emploi
MODEL_PATH = os.getenv("MODEL_PATH", "chexpert_densenet_epoch_15.pth")
inference_service = InferenceService(MODEL_PATH)
